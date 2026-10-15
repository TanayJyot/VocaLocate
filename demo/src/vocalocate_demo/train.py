"""Training: contrastive QVIM baseline, then fine-tuning with mined negatives.

Stage 1 (baseline) trains both encoders with a symmetric InfoNCE loss on
imitation/reference pairs, like the QVIM baseline. Stage 4 (fine-tuning)
continues from the baseline with training triples
    same imitation + correct SFX + selected negative SFX
using contrastive + triplet loss: the negatives are added as extra columns in
the InfoNCE logits, and a triplet margin term pushes each imitation closer to
its target than to its negative. The checkpoint kept is the epoch with the
best validation MRR, so the test split is never used to choose anything.
"""

from __future__ import annotations

import copy
import logging
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .data import Dataset, fit_frames
from .evaluate import evaluate_split
from .model import DualEncoder

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    frames: int = 256  # 2.56 s at the pipeline's 10 ms hop
    dim: int = 128
    width: int = 16
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    tau: float = 0.07
    ft_epochs: int = 12
    ft_lr: float = 3e-4
    triplet_margin: float = 0.2
    triplet_weight: float = 1.0
    freq_mask: int = 16  # SpecAugment-style frequency masking while training
    seed: int = 490
    device: str = "auto"

    def resolved_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def _seed(seed: int) -> np.random.Generator:
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _batch(ds: Dataset, ids: list[str], cfg: TrainConfig, rng: np.random.Generator | None, device) -> torch.Tensor:
    x = np.stack([fit_frames(ds.mel(i), cfg.frames, rng) for i in ids])
    if rng is not None and cfg.freq_mask:
        for row in x:
            w = int(rng.integers(0, cfg.freq_mask + 1))
            f0 = int(rng.integers(0, row.shape[0] - w + 1))
            row[f0 : f0 + w] = row.mean()
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def embed(model: DualEncoder, ds: Dataset, ids: list[str], tower: str, cfg: TrainConfig, bs: int = 128) -> np.ndarray:
    """Embed clips with the imitation or the SFX tower. Returns (N, dim) unit vectors."""
    model.eval()
    device = next(model.parameters()).device
    fn = model.embed_imitation if tower == "imitation" else model.embed_sfx
    out = [fn(_batch(ds, ids[i : i + bs], cfg, None, device)).cpu().numpy() for i in range(0, len(ids), bs)]
    return np.concatenate(out) if out else np.zeros((0, cfg.dim), dtype=np.float32)


def _unique_reference_batches(pairs, batch_size: int, rng: np.random.Generator) -> list[list[int]]:
    """Shuffle pairs into batches where no reference appears twice, so in-batch negatives are real negatives."""
    refs = pairs.reference_clip_id.to_numpy()
    batches: list[tuple[list[int], set]] = []
    for i in rng.permutation(len(pairs)):
        for rows, seen in batches:
            if len(rows) < batch_size and refs[i] not in seen:
                rows.append(int(i))
                seen.add(refs[i])
                break
        else:
            batches.append(([int(i)], {refs[i]}))
    return [rows for rows, _ in batches if len(rows) > 1]


def _val_mrr(model: DualEncoder, ds: Dataset, cfg: TrainConfig) -> float:
    return evaluate_split(model, ds, "val", cfg)["mrr"]


def _fit(
    model: DualEncoder,
    ds: Dataset,
    cfg: TrainConfig,
    epochs: int,
    lr: float,
    negatives: dict[str, list[str]] | None,
    label: str,
) -> tuple[DualEncoder, list[dict]]:
    device = next(model.parameters()).device
    rng = _seed(cfg.seed)
    pairs = ds.split_pairs("train")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps_per_epoch = len(_unique_reference_batches(pairs, cfg.batch_size, np.random.default_rng(0)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(epochs * steps_per_epoch, 1) + 10)

    best = (_val_mrr(model, ds, cfg) if negatives is not None else -1.0, copy.deepcopy(model.state_dict()), 0)
    history = [{"epoch": 0, "val_mrr": best[0]}] if negatives is not None else []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for idx in _unique_reference_batches(pairs, cfg.batch_size, rng):
            rows = pairs.iloc[idx]
            q = model.embed_imitation(_batch(ds, list(rows.imitation_clip_id), cfg, rng, device))
            p = model.embed_sfx(_batch(ds, list(rows.reference_clip_id), cfg, rng, device))
            logits = q @ p.T / model.tau
            target = torch.arange(len(idx), device=device)
            if negatives is None:
                loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
            else:
                # One mined negative per imitation, sampled from its list each step.
                neg_ids, has_neg = [], []
                for r in rows.itertuples():
                    cands = negatives.get(r.imitation_clip_id) or []
                    neg_ids.append(cands[int(rng.integers(len(cands)))] if cands else r.reference_clip_id)
                    has_neg.append(bool(cands))
                mask = torch.tensor(has_neg, device=device)
                n = model.embed_sfx(_batch(ds, neg_ids, cfg, rng, device))
                neg_logits = (q * n).sum(-1, keepdim=True) / model.tau
                neg_logits = neg_logits.masked_fill(~mask[:, None], float("-inf"))
                loss_nce = F.cross_entropy(torch.cat([logits, neg_logits], dim=1), target)
                loss_nce = 0.5 * (loss_nce + F.cross_entropy(logits.T, target))
                s_pos, s_neg = (q * p).sum(-1), (q * n).sum(-1)
                trip = F.relu(cfg.triplet_margin - s_pos + s_neg)[mask]
                loss = loss_nce + cfg.triplet_weight * (trip.mean() if trip.numel() else 0.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val = _val_mrr(model, ds, cfg)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_mrr": val, "sec": time.time() - t0})
        log.info("%s epoch %d loss %.3f val MRR %.3f (%.1fs)", label, epoch, np.mean(losses), val, time.time() - t0)
        if val > best[0]:
            best = (val, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    log.info("%s: kept epoch %d (val MRR %.3f)", label, best[2], best[0])
    return model, history


def train_baseline(ds: Dataset, cfg: TrainConfig) -> tuple[DualEncoder, list[dict]]:
    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    _seed(cfg.seed)
    n_mels = ds.mel(ds.pairs.reference_clip_id.iloc[0]).shape[0]
    model = DualEncoder(n_mels=n_mels, dim=cfg.dim, width=cfg.width, tau=cfg.tau).to(cfg.resolved_device())
    return _fit(model, ds, cfg, cfg.epochs, cfg.lr, None, "baseline")


def finetune(
    baseline: DualEncoder, ds: Dataset, negatives: dict[str, list[str]], cfg: TrainConfig, label: str
) -> tuple[DualEncoder, list[dict]]:
    model = copy.deepcopy(baseline)
    return _fit(model, ds, cfg, cfg.ft_epochs, cfg.ft_lr, negatives, label)


def save(model: DualEncoder, path, cfg: TrainConfig, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "hparams": model.hparams, "train": asdict(cfg), **(extra or {})}, path)


def load(path, device: str = "auto") -> tuple[DualEncoder, TrainConfig, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{**ckpt["train"], "device": device})
    model = DualEncoder(**ckpt["hparams"])
    model.load_state_dict(ckpt["state"])
    return model.to(cfg.resolved_device()).eval(), cfg, ckpt
