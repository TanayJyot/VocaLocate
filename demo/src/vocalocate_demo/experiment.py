"""The end-to-end ML flow from the "High-Level Technical Flow" table, on data from the A2 pipeline.

    1. baseline           contrastive dual encoder on the train split
    2-3. mining           retrieval-driven and metadata-guided semi-hard negatives (+ random control),
                          written to the lake's gold `hard_negatives` table
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
from dataclasses import asdict
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


def run_mining(
    cfg: PipelineConfig, source: str, ds: Dataset, baseline, tcfg: T.TrainConfig, mcfg: MiningConfig, version: str
) -> dict:
    """Mine negatives for every strategy from the baseline and store them in the lake."""
    lake = Lake(cfg.lake_root)
    pairs = ds.split_pairs("train")
    pool = ds.split_candidates("train")
    meta = ds.meta()
    q = T.embed(baseline, ds, list(pairs.imitation_clip_id), "imitation", tcfg)
    c = T.embed(baseline, ds, list(pool.clip_id), "sfx", tcfg)
    out = {}
    for strategy in STRATEGIES:
        started = _now()
        summary = RunSummary(run_id=_new_run_id("mining"), flow="mining", source=f"{source}:{strategy}")
        mined = mine(
            strategy, q, list(pairs.imitation_clip_id), list(pairs.reference_clip_id), c, list(pool.clip_id),
            meta, mcfg, version, summary.run_id,
        )
        lake.write("hard_negatives", f"{source}__{strategy}", to_table("hard_negatives", mined.rows))
        summary.n_input, summary.n_processed, summary.n_ok = len(pairs), len(mined.rows), len(mined.negatives)
        _record_run(lake, summary, cfg, started, "succeeded", json.dumps(mined.stats["selection"]))
        log.info("mined %s: %s", strategy, mined.stats)
        out[strategy] = mined
    warehouse.load(cfg)
    return out


def run_experiment(
    cfg: PipelineConfig,
    source: str,
    tcfg: T.TrainConfig | None = None,
    mcfg: MiningConfig | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
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
    }

    t0 = time.time()
    base_version = model_version("baseline", source, tcfg, None)
    baseline, hist = T.train_baseline(ds, tcfg)
    T.save(baseline, out_dir / "baseline.pt", tcfg, {"model_version": base_version, "history": hist})
    results["models"]["baseline"] = _score(baseline, ds, tcfg, base_version, hist, time.time() - t0)

    mined = run_mining(cfg, source, ds, baseline, tcfg, mcfg, base_version)
    for strategy in strategies:
        results["mining"][strategy] = mined[strategy].stats
        t0 = time.time()
        version = model_version(strategy, source, tcfg, mcfg)
        model, hist = T.finetune(baseline, ds, mined[strategy].negatives, tcfg, strategy)
        T.save(model, out_dir / f"{strategy}.pt", tcfg, {"model_version": version, "history": hist})
        results["models"][strategy] = _score(model, ds, tcfg, version, hist, time.time() - t0)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
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
