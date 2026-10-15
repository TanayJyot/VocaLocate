"""The end-to-end ML flow from the "High-Level Technical Flow" table, on data from the A2 pipeline.

    1. baseline           contrastive dual encoder on the train split
    2-3. mining           retrieval-driven and metadata-guided semi-hard negatives (+ random control),
                          written to the lake's gold `hard_negatives` table
    5. tuning (--sweep)   per strategy, try a small grid of band / near-duplicate settings and keep
                          the one with the best validation MRR
    4. fine-tuning        triplet + contrastive loss with each set of negatives
    5. selection          every model keeps its best epoch by validation MRR
    8. evaluation         frozen models scored once on the held-out test split

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

    warehouse.load(cfg)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


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
