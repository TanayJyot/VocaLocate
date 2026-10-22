"""The QVIM 2025 winning recipe on our data, with the negative source as the only variable.

Bhattacharjee et al. (arXiv 2608.19174, Submission #2) train one shared MobileNetV3 with a
contrastive loss on (imitation, reference) pairs plus a triplet loss with semi-hard negatives,
from the first epoch, with augmentation on both branches. Their negatives are extra imitations
that never form pairs (VocalSketch's excluded takes), sampled by AudioSet class.

Here every run uses that recipe and changes one thing, so the comparison is fair:

* ``greif``    two towers, augmentation, contrastive only (the Greif et al. / QVIM baseline recipe)
* ``shared``   one shared encoder, augmentation, contrastive only (the reference for Δ)
* ``excluded_same``  + triplet, negatives = unpaired imitations of the *same* sound, random ones
  when it has none: the paper's own rule ("sampling same-class negatives when available and falling
  back to random sampling otherwise"); on VimSketch these are VocalSketch's rejected takes ("C")
* ``excluded_same_noaug``  the same without augmentation
* ``excluded`` + triplet, negatives = unpaired imitations of *related, different* sounds (our
  variant of C, which avoids treating a take of the anchor's own sound as a negative)
* ``excluded_noaug``  the same without augmentation
* ``retrieval``   + triplet, negatives = the current model's top wrong SFX, re-mined every few epochs (A)
* ``metadata``    + triplet, negatives = metadata-related SFX (B)
* ``random``      + triplet, negatives = random SFX (control)

Choices the paper leaves open, and ours:

* Anchor. For imitation negatives the anchor is the reference: it should be closer to its own
  imitation than to a rejected take of the same sound (``excluded_same``) or to an imitation of a
  related sound (``excluded``). For SFX negatives (A, B, random) the anchor
  is the imitation: it should be closer to its target than to the negative SFX.
* "Same class" in the Vocal Imitation Set means the same sound (one reference per class), so an
  excluded imitation of the same class would be a false negative. Related means a different
  reference that shares the category or at least `min_shared_tags` AudioSet ontology tags
  (the same tiers as version B); random other references fill in when none are related.
* Semi-hard: s_pos - margin < s_neg < s_pos (cosine). Each anchor scores `n_cands` sampled
  candidates per step and uses the hardest semi-hard one.
* Dynamic weight: the triplet term is the sum over anchors with an active semi-hard triplet,
  divided by the batch size, so it shrinks when few triplets are active.
* Temperature fixed at 0.07, AdamW with linear warm-up for one epoch and cosine decay from
  2e-4 to 5e-5, margin 0.6, as in the paper. Weight decay is not given there; we use 1e-4.
* Model selection on our validation split. When a benchmark is given it is also scored after every
  epoch, and each run additionally reports the epoch that scored best on it: the winners' protocol
  (they selected on QVIM DEV), kept apart from the clean, validation-selected number.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import queue
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vocalocate_pipeline.config import PipelineConfig

from . import train as T
from .augment import augment_batch
from .data import Dataset, SFXMeta, fit_frames, load_dataset
from .evaluate import evaluate_split
from .mining import METADATA_TIERS, relation
from .model import DualEncoder

log = logging.getLogger(__name__)

NEGATIVES = ("none", "excluded", "retrieval", "metadata", "random")


@dataclass
class RecipeConfig:
    name: str = "shared"
    shared: bool = True
    augment: bool = True
    negatives: str = "none"
    margin: float = 0.6
    triplet_weight: float = 1.0
    n_cands: int = 2  # candidates scored per anchor per step
    pool_size: int = 20  # candidate list per anchor
    min_shared_tags: int = 2
    excluded_match: str = "related"  # excluded negatives: "related" (different sounds) or "same" (paper)
    remine_every: int = 5  # retrieval: epochs between re-mining with the current model
    epochs: int = 30
    lr_max: float = 2e-4
    lr_min: float = 5e-5
    warmup_epochs: int = 1
    weight_decay: float = 1e-4
    tau: float = 0.07
    learn_tau: bool = False
    amp: bool = True  # mixed precision on CUDA


RECIPES = {
    "greif": RecipeConfig("greif", shared=False),
    "shared": RecipeConfig("shared"),
    "excluded_same": RecipeConfig("excluded_same", negatives="excluded", excluded_match="same"),
    "excluded_same_noaug": RecipeConfig("excluded_same_noaug", negatives="excluded", excluded_match="same",
                                        augment=False),
    "excluded": RecipeConfig("excluded", negatives="excluded"),
    "excluded_noaug": RecipeConfig("excluded_noaug", negatives="excluded", augment=False),
    "retrieval": RecipeConfig("retrieval", negatives="retrieval"),
    "metadata": RecipeConfig("metadata", negatives="metadata"),
    "random": RecipeConfig("random", negatives="random"),
}
REFERENCE_MODEL = "shared"
RECIPE_NAMES = {
    "greif": "Greif et al. recipe (two towers + augmentation, contrastive only)",
    "shared": "Winner's recipe without triplet (shared encoder + augmentation)",
    "excluded_same": "C: winner's method (rejected takes of the same sound as semi-hard negatives)",
    "excluded_same_noaug": "C without augmentation",
    "excluded": "C, related variant (excluded imitations of related, different sounds)",
    "excluded_noaug": "C, related variant, without augmentation",
    "retrieval": "A: retrieval-driven semi-hard negatives (re-mined)",
    "metadata": "B: metadata-guided semi-hard negatives",
    "random": "Control: random SFX negatives",
}


def recipe_dir(cfg: PipelineConfig, source: str, seed: int) -> Path:
    return cfg.lake_root / "models" / source / f"recipe-{seed}"


# --------------------------------------------------------------------------- candidate pools


def _pick(rng: np.random.Generator, tiers: list[list[str]], fill: list[str], k: int) -> list[str]:
    out: list[str] = []
    for tier in tiers:
        if len(out) >= k:
            break
        out += [tier[i] for i in rng.permutation(len(tier))[: k - len(out)]]
    if len(out) < k and fill:
        extra = [fill[i] for i in rng.permutation(len(fill))[: 4 * k]]
        out += [c for c in extra if c not in out][: k - len(out)]
    return out


def excluded_pools(ds: Dataset, rc: RecipeConfig, rng: np.random.Generator) -> dict[str, list[str]]:
    """Reference clip_id -> train unpaired imitations to draw negatives from.

    excluded_match="same": the reference's own unpaired imitations (rejected takes of the same
    sound), or random ones when it has none. "related": imitations of related, *different* sounds.
    """
    un = ds.split_unpaired("train")
    if un.empty:
        return {}
    meta = ds.meta()
    by_target: dict[str, list[str]] = {}
    for r in un.itertuples():
        by_target.setdefault(r.target_clip_id, []).append(r.clip_id)
    targets = list(by_target)
    pools = {}
    if rc.excluded_match == "same":
        everything = list(un.clip_id)
        for ref in ds.split_pairs("train").reference_clip_id.unique():
            own = by_target.get(ref, [])
            pick = own or everything  # random unpaired imitations when this sound has none of its own
            pools[ref] = [pick[i] for i in rng.permutation(len(pick))[: rc.pool_size]]
        return pools
    for ref in ds.split_pairs("train").reference_clip_id.unique():
        m = meta.get(ref, SFXMeta(None, frozenset(), None))
        tiers: dict[str, list[str]] = {t: [] for t in METADATA_TIERS}
        others: list[str] = []
        for t in targets:
            if t == ref:
                continue  # imitations of the same sound are positives, never negatives
            rel = relation(m, meta.get(t, SFXMeta(None, frozenset(), None)), rc.min_shared_tags)
            if rel == "same_recording":
                continue
            (tiers[rel] if rel in tiers else others).extend(by_target[t])
        pools[ref] = _pick(rng, [tiers[t] for t in METADATA_TIERS], others, rc.pool_size)
    return pools


def metadata_pools(ds: Dataset, rc: RecipeConfig, rng: np.random.Generator) -> dict[str, list[str]]:
    """Reference clip_id -> related train SFX by metadata tier (version B's pools, no band filter)."""
    meta = ds.meta()
    pool_ids = list(ds.split_candidates("train").clip_id)
    pools = {}
    for ref in ds.split_pairs("train").reference_clip_id.unique():
        m = meta.get(ref, SFXMeta(None, frozenset(), None))
        tiers: dict[str, list[str]] = {t: [] for t in METADATA_TIERS}
        for c in pool_ids:
            if c == ref:
                continue
            rel = relation(m, meta[c], rc.min_shared_tags)
            if rel in tiers:
                tiers[rel].append(c)
        pools[ref] = _pick(rng, [tiers[t] for t in METADATA_TIERS], [], rc.pool_size)
    return pools


def random_pool(ds: Dataset) -> list[str]:
    return list(ds.split_candidates("train").clip_id)


def retrieval_pools(model: DualEncoder, ds: Dataset, tcfg: T.TrainConfig, rc: RecipeConfig,
                    dup_quantile: float = 0.99) -> dict[str, list[str]]:
    """Imitation clip_id -> the current model's top wrong train SFX, minus likely false negatives."""
    pairs, pool = ds.split_pairs("train"), ds.split_candidates("train")
    meta = ds.meta()
    q = T.embed(model, ds, list(pairs.imitation_clip_id), "imitation", tcfg)
    c = T.embed(model, ds, list(pool.clip_id), "sfx", tcfg)
    ids = list(pool.clip_id)
    index = {cid: i for i, cid in enumerate(ids)}
    sims = q @ c.T
    cc = c @ c.T
    off = cc[~np.eye(len(ids), dtype=bool)]
    dup = float(np.quantile(off, dup_quantile)) if off.size else 1.0
    order = np.argsort(-sims, axis=1)
    pools = {}
    for i, r in enumerate(pairs.itertuples()):
        p = index.get(r.reference_clip_id)
        rec = meta[r.reference_clip_id].recording_id if p is not None else None
        out = []
        for j in order[i]:
            if j == p or (p is not None and cc[p, j] > dup):
                continue
            if rec is not None and meta[ids[j]].recording_id == rec:
                continue
            out.append(ids[j])
            if len(out) == rc.pool_size:
                break
        pools[r.imitation_clip_id] = out
    return pools


# --------------------------------------------------------------------------- training


def _lr_lambda(rc: RecipeConfig, steps_per_epoch: int, total: int):
    warm = max(rc.warmup_epochs * steps_per_epoch, 1)
    floor = rc.lr_min / rc.lr_max

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(total - warm, 1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    return f


def _semi_hard(s_pos: torch.Tensor, s_neg: torch.Tensor, valid: torch.Tensor, margin: float):
    """s_pos (B,), s_neg (B, K), valid (B, K) -> per-anchor loss and the active mask (B,)."""
    with torch.no_grad():
        semi = valid & (s_neg < s_pos[:, None]) & (s_neg > s_pos[:, None] - margin)
    masked = s_neg.masked_fill(~semi, float("-inf"))
    hardest = masked.max(dim=1).values
    active = semi.any(dim=1)
    loss = F.relu(margin - s_pos + torch.where(active, hardest, torch.zeros_like(hardest)))
    return loss * active, active


def _prefetch(fn, items, depth: int = 4):
    """Yield fn(item) for each item, computed in a background thread up to `depth` items ahead."""
    q: queue.Queue = queue.Queue(maxsize=depth)
    done = object()

    def work():
        try:
            for it in items:
                q.put(fn(it))
        except BaseException as e:  # surfaced in the consumer
            q.put(e)
        q.put(done)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    while (out := q.get()) is not done:
        if isinstance(out, BaseException):
            th.join()
            raise out
        yield out
    th.join()


def train_recipe(ds: Dataset, tcfg: T.TrainConfig, rc: RecipeConfig,
                 bench: Dataset | None = None) -> tuple[DualEncoder, list[dict], dict | None]:
    """Train one recipe; keep the epoch with the best validation MRR.

    With a benchmark dataset, its test split is also scored after every epoch, and the epoch that
    scored best on it is returned too, as {"epoch", "mrr", "state"} (None without a benchmark).
    """
    if rc.negatives not in NEGATIVES:
        raise ValueError(f"unknown negatives {rc.negatives!r}; expected one of {NEGATIVES}")
    device = tcfg.resolved_device()
    rng = T._seed(tcfg.seed)
    g = torch.Generator().manual_seed(tcfg.seed)
    tcfg = replace(tcfg, freq_mask=0)  # masking is part of the augmentation pool here
    n_mels = ds.mel(ds.pairs.reference_clip_id.iloc[0]).shape[0]
    model = DualEncoder(n_mels=n_mels, dim=tcfg.dim, width=tcfg.width, tau=rc.tau, encoder=tcfg.encoder,
                        pretrained=tcfg.pretrained, frames=tcfg.frames, shared=rc.shared).to(device)
    model.log_tau.requires_grad_(rc.learn_tau)

    pairs = ds.split_pairs("train")
    epoch_batches = [T._unique_reference_batches(pairs, tcfg.batch_size, rng) for _ in range(rc.epochs)]
    steps = [len(b) for b in epoch_batches]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=rc.lr_max, weight_decay=rc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(rc, steps[0], sum(steps)))
    use_amp = rc.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    pools: dict[str, list[str]] = {}
    any_pool: list[str] = []
    if rc.negatives == "excluded":
        pools = excluded_pools(ds, rc, rng)
        if not any(pools.values()):
            raise ValueError("no unpaired imitations in the train split; re-ingest the source with them")
    elif rc.negatives == "metadata":
        pools = metadata_pools(ds, rc, rng)
    elif rc.negatives == "random":
        any_pool = random_pool(ds)
    meta = ds.meta()

    pin = device.type == "cuda"
    if pin:
        torch.backends.cudnn.benchmark = True  # fixed input shapes

    def stack(ids):  # CPU: crop/pad and stack features
        x = torch.from_numpy(np.stack([fit_frames(ds.mel(i), tcfg.frames, rng) for i in ids]))
        return x.pin_memory() if pin else x

    def prepare(idx):
        """Everything a step needs that is not GPU work: candidate sampling and feature assembly."""
        rows = pairs.iloc[idx]
        neg_ids, valid = [], []
        if rc.negatives != "none":
            for r in rows.itertuples():
                if rc.negatives == "random":
                    rec = meta[r.reference_clip_id].recording_id
                    cands = []
                    while len(cands) < rc.n_cands:
                        c = any_pool[int(rng.integers(len(any_pool)))]
                        if c != r.reference_clip_id and (rec is None or meta[c].recording_id != rec):
                            cands.append(c)
                else:
                    key = r.imitation_clip_id if rc.negatives == "retrieval" else r.reference_clip_id
                    pool = pools.get(key) or []
                    cands = [pool[int(rng.integers(len(pool)))] for _ in range(rc.n_cands)] if pool else []
                valid.append([True] * len(cands) + [False] * (rc.n_cands - len(cands)))
                neg_ids += cands + [r.reference_clip_id] * (rc.n_cands - len(cands))  # padding, masked out
        return (len(idx), stack(list(rows.imitation_clip_id)), stack(list(rows.reference_clip_id)),
                stack(neg_ids) if neg_ids else None, torch.tensor(valid, dtype=torch.bool))

    def batch(x):
        x = x.to(device, non_blocking=True)
        return augment_batch(x, g) if rc.augment else x

    best = (-1.0, None, 0)
    bench_best = None
    history = []
    for epoch, batches in enumerate(epoch_batches, start=1):
        if rc.negatives == "retrieval" and (epoch - 1) % rc.remine_every == 0:
            t_m = time.time()
            pools = retrieval_pools(model, ds, tcfg, rc)
            log.info("%s: re-mined %d pools (%.1fs)", rc.name, len(pools), time.time() - t_m)
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "nce": 0.0, "triplet": 0.0, "active": 0.0}
        # One producer thread builds batches ahead of the GPU. It is the only user of rng during
        # the epoch, so the batches (and results) are the same as building them inline.
        for step, (b, xq, xp, xn, valid) in enumerate(_prefetch(prepare, batches), start=1):
            if step == 1 or step % 50 == 0:
                log.info("%s epoch %d step %d/%d (%.0fs)", rc.name, epoch, step, len(batches), time.time() - t0)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                q = model.embed_imitation(batch(xq))
                p = model.embed_sfx(batch(xp))
                logits = q @ p.T / model.tau
                target = torch.arange(b, device=device)
                nce = 0.5 * (F.cross_entropy(logits.float(), target) + F.cross_entropy(logits.T.float(), target))
                loss, trip, active = nce, torch.zeros((), device=device), torch.zeros(b, device=device)
                if xn is not None:
                    xb = batch(xn)
                    if rc.negatives == "excluded":
                        n = model.embed_imitation(xb).view(b, rc.n_cands, -1)
                        anchor, pos = p, q  # reference closer to its imitation than to a related sound's
                    else:
                        n = model.embed_sfx(xb).view(b, rc.n_cands, -1)
                        anchor, pos = q, p  # imitation closer to its target than to the negative SFX
                    s_pos = (anchor * pos).sum(-1).float()
                    s_neg = torch.einsum("bd,bkd->bk", anchor, n).float()
                    per, act = _semi_hard(s_pos, s_neg, valid.to(device), rc.margin)
                    trip = per.sum() / b  # dynamic weight: active fraction of the batch
                    active = act.float()
                    loss = nce + rc.triplet_weight * trip
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums["loss"] += float(loss.detach())
            sums["nce"] += float(nce.detach())
            sums["triplet"] += float(trip.detach())
            sums["active"] += float(active.mean())
        n_b = max(len(batches), 1)
        val = evaluate_split(model, ds, "val", tcfg)["mrr"]
        rec = {"epoch": epoch, **{k: v / n_b for k, v in sums.items()}, "val_mrr": val, "sec": time.time() - t0,
               "lr": sched.get_last_lr()[0]}
        if bench is not None:
            rec["bench_mrr"] = evaluate_split(model, bench, "test", tcfg)["mrr"]
            if bench_best is None or rec["bench_mrr"] > bench_best["mrr"]:
                bench_best = {"epoch": epoch, "mrr": rec["bench_mrr"], "state": copy.deepcopy(model.state_dict())}
        history.append(rec)
        log.info("%s epoch %d loss %.3f (nce %.3f, triplet %.3f, active %.2f) val MRR %.3f%s (%.1fs)", rc.name,
                 epoch, rec["loss"], rec["nce"], rec["triplet"], rec["active"], val,
                 f", bench MRR {rec['bench_mrr']:.3f}" if "bench_mrr" in rec else "", rec["sec"])
        if val > best[0]:
            best = (val, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    log.info("%s: kept epoch %d (val MRR %.3f)", rc.name, best[2], best[0])
    return model, history, bench_best


# --------------------------------------------------------------------------- suite


def run_recipe_suite(cfg: PipelineConfig, source: str, tcfg: T.TrainConfig, names: list[str],
                     bench: str | None = "qvim_dev", overrides: dict | None = None,
                     write_results: bool = True, preload: bool = False,
                     extra_benches: tuple[str, ...] = ()) -> dict:
    """Train each named recipe for one seed, score val/test (and the benchmark), write results.json.

    Finished conditions are skipped on a re-run (their <name>.json exists), so an interrupted
    suite resumes at the first unfinished condition. With write_results=False only the per-run
    <name>.json files are written, so parallel jobs for one seed never race on results.json; a
    later call with every name (all skipped) collects them. preload=True reads every feature
    file up front, in parallel (for a network volume). `bench` is scored after every epoch (and
    picks the benchmark-selected epoch); `extra_benches` are only scored at the end, on both the
    validation-selected and the benchmark-selected checkpoints, so they stay unseen by selection.
    """
    out = recipe_dir(cfg, source, tcfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(cfg, [source])
    if bench:
        bench_ds = load_dataset(cfg, [bench])
    extra = {b: load_dataset(cfg, [b]) for b in extra_benches if b and b != bench}
    if preload:
        t0 = time.time()
        n = ds.preload() + (bench_ds.preload() if bench else 0) + sum(e.preload() for e in extra.values())
        log.info("preloaded %d feature files in %.0fs", n, time.time() - t0)
    results = {
        "source": source,
        "kind": "recipe",
        "reference_model": REFERENCE_MODEL,
        "selection_benchmark": bench or None,
        "device": str(tcfg.resolved_device()),
        "train_config": asdict(tcfg),
        "data": {
            "train_pairs": int((ds.pairs.split == "train").sum()),
            "unpaired_train": int(len(ds.split_unpaired("train"))),
            "candidate_sfx_train": int((ds.candidates.split == "train").sum()),
        },
        "recipes": {},
        "models": {},
    }
    for name in names:
        rc = replace(RECIPES[name], **(overrides or {}))
        done = out / f"{name}.json"
        if done.exists():
            log.info("%s: already done, skipping", name)
            entry = json.loads(done.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            model, hist, bench_best = train_recipe(ds, tcfg, rc, bench_ds if bench else None)
            T.save(model, out / f"{name}.pt", tcfg, {"model_version": f"{source}-recipe-{name}-{tcfg.seed}",
                                                       "recipe": asdict(rc), "history": hist})
            best = max(hist, key=lambda h: h["val_mrr"])
            entry = {
                "model_version": f"{source}-recipe-{name}-{tcfg.seed}",
                "kept_epoch": best["epoch"],
                "val": evaluate_split(model, ds, "val", tcfg),
                "test": evaluate_split(model, ds, "test", tcfg),
                "train_sec": round(time.time() - t0, 1),
                "device": str(tcfg.resolved_device()),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "batch_size": tcfg.batch_size,
                "history": hist,
            }
            if bench:
                entry["benchmark"] = {bench: evaluate_split(model, bench_ds, "test", tcfg)}
            for b, e in extra.items():
                entry.setdefault("benchmark", {})[b] = evaluate_split(model, e, "test", tcfg)
            if bench_best is not None:
                # The winners' protocol: the epoch that scored best on the benchmark itself.
                model.load_state_dict(bench_best["state"])
                T.save(model, out / f"{name}.benchsel.pt", tcfg,
                       {"model_version": f"{source}-recipe-{name}-{tcfg.seed}-selected-on-{bench}",
                        "recipe": asdict(rc), "history": hist})
                entry["benchmark_selected_on_benchmark"] = {"kept_epoch": bench_best["epoch"],
                                                            bench: evaluate_split(model, bench_ds, "test", tcfg)}
                for b, e in extra.items():
                    entry["benchmark_selected_on_benchmark"][b] = evaluate_split(model, e, "test", tcfg)
            done.write_text(json.dumps({**entry, "recipe": asdict(rc)}, indent=2), encoding="utf-8")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["recipes"][name] = entry.pop("recipe", asdict(rc))
        results["models"][name] = entry
        log.info("%s: test MRR %.3f (%d queries)%s", name, entry["test"]["mrr"], entry["test"]["n_queries"],
                 f", {bench} MRR {entry['benchmark'][bench]['mrr']:.3f}" if bench and "benchmark" in entry else "")
    if write_results:
        (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results
