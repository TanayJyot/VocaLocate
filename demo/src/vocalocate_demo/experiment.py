"""The end-to-end ML flow from the "High-Level Technical Flow" table, on data from the A2 pipeline.

    1. baseline           contrastive dual encoder on the train split
    2-3. mining           retrieval-driven and metadata-guided semi-hard negatives (+ random control),
                          written to the lake's gold `hard_negatives` table
    5. tuning (--sweep)   per strategy, try a small grid of band / near-duplicate settings and keep
                          the one with the best validation MRR
    4. fine-tuning        triplet + contrastive loss with each set of negatives
    5. selection          every model keeps its best epoch by validation MRR
    8. evaluation         frozen models scored once on the held-out test split
    6-7. text branch      with a CLAP scorer: text hints vs SFX audio, fused with each voice model
                          (α tuned on validation); voice-only, text-only and fused test scores

Each model is saved under <lake>/models/<source>/<name>.pt, and each mining run
writes a row to `pipeline_runs` (flow "mining") for lineage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.config import PipelineConfig
from vocalocate_pipeline.flows import RunSummary, _new_run_id, _now, _record_run
from vocalocate_pipeline.lake import Lake
from vocalocate_pipeline.schemas import to_table

from . import train as T
from .data import Dataset, load_dataset
from .evaluate import evaluate_split
from .mining import MiningConfig, mine

log = logging.getLogger(__name__)

STRATEGIES = ("retrieval", "metadata", "random")
MODEL_NAMES = {
    "baseline": "QVIM baseline (no mining)",
    "random": "Control: random real-SFX negatives",
    "retrieval": "VocaLocate A: retrieval-driven semi-hard mining",
    "metadata": "VocaLocate B: pre-constructed metadata-guided mining",
}


def model_dir(cfg: PipelineConfig, source: str) -> Path:
    return cfg.lake_root / "models" / source


def model_version(name: str, source: str, tcfg: T.TrainConfig, mcfg: MiningConfig | None) -> str:
    blob = json.dumps({"train": asdict(tcfg) | {"device": None}, "mining": asdict(mcfg) if mcfg else None}, sort_keys=True)
    return f"{source}-{name}-{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


# Stage 5: mining settings tried per strategy when sweeping. Chosen by validation MRR only.
SWEEP_BANDS = (0.15, 0.25, 0.4)
SWEEP_DUP_QUANTILES = (0.99, 0.998)


def sweep_grid(base: MiningConfig) -> list[MiningConfig]:
    return [replace(base, band=b, dup_quantile=d) for b in SWEEP_BANDS for d in SWEEP_DUP_QUANTILES]


def _store(cfg: PipelineConfig, source: str, strategy: str, mined, n_pairs: int, started, run_id: str) -> None:
    """Write the chosen negatives to the gold hard_negatives table and a pipeline_runs row for lineage."""
    lake = Lake(cfg.lake_root)
    lake.write("hard_negatives", f"{source}__{strategy}", to_table("hard_negatives", mined.rows))
    summary = RunSummary(run_id=run_id, flow="mining", source=f"{source}:{strategy}")
    summary.n_input, summary.n_processed, summary.n_ok = n_pairs, len(mined.rows), len(mined.negatives)
    _record_run(lake, summary, cfg, started, "succeeded", json.dumps(mined.stats["selection"]))


def run_experiment(
    cfg: PipelineConfig,
    source: str,
    tcfg: T.TrainConfig | None = None,
    mcfg: MiningConfig | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    sweep: bool = False,
    clap=None,
) -> dict:
    tcfg, mcfg = tcfg or T.TrainConfig(), mcfg or MiningConfig()
    ds = load_dataset(cfg, [source])
    out_dir = model_dir(cfg, source)
    results: dict = {
        "source": source,
        "device": str(tcfg.resolved_device()),
        "train_config": asdict(tcfg),
        "mining_config": asdict(mcfg),
        "data": {
            split: {
                "pairs": int((ds.pairs.split == split).sum()),
                "references": int(ds.split_pairs(split).reference_clip_id.nunique()),
                "candidate_sfx": int((ds.candidates.split == split).sum()),
            }
            for split in ("train", "val", "test")
        },
        "models": {},
        "mining": {},
        "sweep": [],
    }

    t0 = time.time()
    base_version = model_version("baseline", source, tcfg, None)
    baseline, hist = T.train_baseline(ds, tcfg)
    T.save(baseline, out_dir / "baseline.pt", tcfg, {"model_version": base_version, "history": hist})
    results["models"]["baseline"] = _score(baseline, ds, tcfg, base_version, hist, time.time() - t0)

    # Stage 2: score the training pool once with the baseline; every miner reads these embeddings.
    pairs, pool, meta = ds.split_pairs("train"), ds.split_candidates("train"), ds.meta()
    q = T.embed(baseline, ds, list(pairs.imitation_clip_id), "imitation", tcfg)
    c = T.embed(baseline, ds, list(pool.clip_id), "sfx", tcfg)

    for strategy in strategies:
        t0 = time.time()
        grid = sweep_grid(mcfg) if sweep and strategy != "random" else [mcfg]
        best = None
        for m in grid:
            started, run_id = _now(), _new_run_id("mining")
            mined = mine(
                strategy, q, list(pairs.imitation_clip_id), list(pairs.reference_clip_id), c, list(pool.clip_id),
                meta, m, base_version, run_id,
            )
            log.info("mined %s (band %.2f, dup %.3f): %s", strategy, m.band, m.dup_quantile, mined.stats)
            model, hist = T.finetune(baseline, ds, mined.negatives, tcfg, strategy)
            val = max(h["val_mrr"] for h in hist)
            results["sweep"].append({"strategy": strategy, "band": m.band, "dup_quantile": m.dup_quantile, "val_mrr": val})
            if best is None or val > best[0]:
                best = (val, m, mined, model, hist, started, run_id)
        _, m, mined, model, hist, started, run_id = best
        _store(cfg, source, strategy, mined, len(pairs), started, run_id)
        results["mining"][strategy] = mined.stats
        version = model_version(strategy, source, tcfg, m)
        T.save(model, out_dir / f"{strategy}.pt", tcfg, {"model_version": version, "history": hist, "mining": asdict(m)})
        results["models"][strategy] = _score(model, ds, tcfg, version, hist, time.time() - t0)

    if clap is not None:
        models = {name: T.load(out_dir / f"{name}.pt", tcfg.device)[0] for name in results["models"]}
        results["text"] = run_text_branch(cfg, source, ds, models, tcfg, clap)
        (out_dir / "fusion.json").write_text(
            json.dumps({"clap": clap.version, "alpha": {k: v["alpha"] for k, v in results["text"]["models"].items()}},
                       indent=2),
            encoding="utf-8",
        )

    warehouse.load(cfg)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


def run_text_branch(cfg: PipelineConfig, source: str, ds: Dataset, models: dict, tcfg: T.TrainConfig, clap) -> dict:
    """Stages 6-7: CLAP text scores for imitations with hints, fused with each voice model's scores.

    α is tuned on validation per voice model; the test split is scored once.
    """
    from .fusion import evaluate_fusion
    from .text import cached_audio_embeddings

    meta = ds.meta()
    splits = {}
    for split in ("val", "test"):
        cands = ds.split_candidates(split)
        index = {cid: i for i, cid in enumerate(cands.clip_id)}
        pairs = ds.split_pairs(split)
        pairs = pairs[pairs.reference_clip_id.isin(index) & pairs.imitation_clip_id.isin(ds.hints)]
        audio = cached_audio_embeddings(cfg, f"{source}__clap", list(cands.clip_id), list(cands.clean_path), clap)
        hints = [ds.hints[i] for i in pairs.imitation_clip_id]
        splits[split] = {
            "cands": cands,
            "pairs": pairs,
            "text": clap.embed_text(hints) @ audio.T,
            "target": np.array([index[r] for r in pairs.reference_clip_id], dtype=int),
            "meta": [meta[cid] for cid in cands.clip_id],
        }
    out = {"clap": clap.version, "n_queries": {k: int(len(v["pairs"])) for k, v in splits.items()}, "models": {}}
    if not all(len(v["pairs"]) for v in splits.values()):
        log.warning("text branch: no val or test imitations with hints; skipping")
        return out
    for name, model in models.items():
        voice = {}
        for split, d in splits.items():
            q = T.embed(model, ds, list(d["pairs"].imitation_clip_id), "imitation", tcfg)
            c = T.embed(model, ds, list(d["cands"].clip_id), "sfx", tcfg)
            voice[split] = q @ c.T
        v, t = splits["val"], splits["test"]
        out["models"][name] = evaluate_fusion(
            voice["val"], v["text"], v["target"], v["meta"], voice["test"], t["text"], t["target"], t["meta"]
        )
        log.info("text branch %s: alpha %.1f, test %s", name, out["models"][name]["alpha"],
                 {k: round(m["mrr"], 3) for k, m in out["models"][name]["test"].items()})
    return out


# Published QVIM DEV-set MRRs (A2 Part Two), for the benchmark table.
QVIM_DEV_PUBLISHED = {
    "random ranking": 0.044,
    "QVIM challenge MobileNetV3 baseline (trained on VimSketch)": 0.273,
    "challenge winner": 0.293,
}


def run_benchmark(cfg: PipelineConfig, model_source: str, bench_source: str = "qvim_dev",
                  names: tuple[str, ...] = ("baseline", "random", "retrieval", "metadata"), device: str = "auto") -> dict:
    """Score models trained on `model_source` on the held-out test split of another source (external benchmark).

    Every imitation in the benchmark ranks all of its reference sounds, as in the QVIM challenge.
    Nothing is trained or tuned on the benchmark.
    """
    ds = load_dataset(cfg, [bench_source])
    out = {"model_source": model_source, "benchmark": bench_source, "models": {}}
    for name in names:
        path = model_dir(cfg, model_source) / f"{name}.pt"
        if not path.exists():
            continue
        model, tcfg, ckpt = T.load(path, device)
        out["models"][name] = {"model_version": ckpt["model_version"], **evaluate_split(model, ds, "test", tcfg)}
    if bench_source == "qvim_dev":
        out["published_mrr"] = QVIM_DEV_PUBLISHED
    (model_dir(cfg, model_source) / f"benchmark_{bench_source}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _score(model, ds: Dataset, tcfg: T.TrainConfig, version: str, hist: list[dict], sec: float) -> dict:
    best = max(hist, key=lambda h: h["val_mrr"]) if hist else {"epoch": 0, "val_mrr": None}
    return {
        "model_version": version,
        "kept_epoch": best["epoch"],
        "val": evaluate_split(model, ds, "val", tcfg),
        "test": evaluate_split(model, ds, "test", tcfg),
        "train_sec": round(sec, 1),
        "history": hist,
    }
