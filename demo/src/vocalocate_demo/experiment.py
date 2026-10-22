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


def run_recipe_text_branch(cfg: PipelineConfig, source: str, run_dirs: list[Path], clap,
                           names: list[str] | None = None, device: str = "auto") -> dict:
    """The text-branch test (run_text_branch) on saved recipe models, one seed folder at a time.

    Each folder is a recipe-<seed>/ directory holding <name>.pt checkpoints (from `recipe`, locally or
    fetched from Modal). Writes text.json into each folder and returns {folder: result}.
    """
    from .recipe import RECIPES

    ds = load_dataset(cfg, [source])
    out = {}
    for d in map(Path, run_dirs):
        models, tcfg = {}, None
        for name in names or list(RECIPES):
            if (d / f"{name}.pt").exists():
                models[name], tcfg, _ = T.load(d / f"{name}.pt", device)
        if not models:
            raise FileNotFoundError(f"no recipe checkpoints (*.pt) in {d}")
        res = run_text_branch(cfg, source, ds, models, tcfg, clap)
        (d / "text.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        out[str(d)] = res
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


def run_text_benchmark(cfg: PipelineConfig, clap, source: str = "captdure") -> dict:
    """Score the frozen CLAP text branch on a captioned source's test split (text-to-audio retrieval).

    Each caption ranks every clip of the source; its target is the clip it was written for.
    Nothing is trained or tuned on it. Metrics are those of the voice benchmark: R@K, MRR,
    MRR within the target's class (fine), graded nDCG@10, plus R@1 at class level.
    """
    import duckdb

    from .data import SFXMeta
    from .evaluate import score_metrics
    from .text import cached_audio_embeddings

    con = duckdb.connect(str(cfg.warehouse_path), read_only=True)
    try:
        cands = con.execute(
            "SELECT clip_id, clean_path, category, tags, recording_id FROM v_sfx_candidates "
            "WHERE source = ? AND split = 'test' ORDER BY clip_id", [source]).df()
        caps = con.execute(
            "SELECT caption, target_clip_id, category FROM v_captions "
            "WHERE source = ? AND split = 'test' AND language = 'en' ORDER BY caption_id", [source]).df()
    finally:
        con.close()
    if cands.empty or caps.empty:
        raise ValueError(f"No captions for {source!r} in {cfg.warehouse_path}; run the dataset flow first")

    index = {cid: i for i, cid in enumerate(cands.clip_id)}
    caps = caps[caps.target_clip_id.isin(index)]
    audio = cached_audio_embeddings(cfg, f"{source}__clap", list(cands.clip_id), list(cands.clean_path), clap)
    text = clap.embed_text(list(caps.caption))
    sims = text @ audio.T
    target = np.array([index[c] for c in caps.target_clip_id])
    meta = [SFXMeta(r.category, frozenset(r.tags if r.tags is not None else ()), r.recording_id)
            for r in cands.itertuples()]
    metrics = score_metrics(sims, target, meta)
    cats = cands.category.to_numpy()
    metrics["class_recall@1"] = float(np.mean(cats[sims.argmax(axis=1)] == cats[target]))
    n = len(cands)
    per_class = {}
    for cat in sorted(set(caps.category)):
        rows = (caps.category == cat).to_numpy()
        per_class[cat] = {"n_captions": int(rows.sum()), **{k: v for k, v in score_metrics(
            sims[rows], target[rows], meta).items() if k in ("mrr", "recall@1", "recall@10")}}
    out = {
        "benchmark": source,
        "model_version": clap.version,
        "n_clips": n,
        "n_classes": int(cands.category.nunique()),
        "metrics": metrics,
        # Expected scores of a random ranking, for scale.
        "random": {"mrr": float(np.sum(1.0 / np.arange(1, n + 1)) / n), "recall@1": 1.0 / n, "recall@10": min(10, n) / n},
        "per_class": per_class,
    }
    out_dir = cfg.lake_root / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"text_{source}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
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


def run_onnx_benchmark(cfg: PipelineConfig, onnx_path: Path, bench_source: str = "qvim_dev", sr: int = 16_000,
                       seconds: float = 10.0, label: str = "onnx") -> dict:
    """Score an exported waveform-in, embedding-out ONNX model (e.g. thatsoundslikeme's model_v1.onnx).

    Queries and candidates are the same as `run_benchmark` (each test imitation ranks every
    test reference of the benchmark). Audio is the pipeline's clean 32 kHz WAV, resampled to
    `sr`, mono, first `seconds` (zero-padded), the preprocessing of the thatsoundslikeme scripts.
    """
    from math import gcd

    import onnxruntime as ort
    import soundfile as sf
    from scipy.signal import resample_poly

    from .evaluate import score_metrics

    ds = load_dataset(cfg, [bench_source])
    pairs, cands = ds.split_pairs("test"), ds.split_candidates("test")
    index = {cid: i for i, cid in enumerate(cands.clip_id)}
    pairs = pairs[pairs.reference_clip_id.isin(index)]
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    inp, outp = sess.get_inputs()[0], sess.get_outputs()[0]
    n = int(sr * seconds)

    def load(path: str) -> np.ndarray:
        x, s = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if s != sr:
            g = gcd(s, sr)
            x = resample_poly(x, sr // g, s // g).astype(np.float32)
        return np.pad(x[:n], (0, max(n - len(x), 0))).astype(np.float32)

    def embed(paths: list[str], bs: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(paths), bs):
            batch = np.stack([load(p) for p in paths[i : i + bs]])
            if len(inp.shape) == 3:
                batch = batch[None]
            e = np.asarray(sess.run([outp.name], {inp.name: batch})[0], dtype=np.float32)
            out.append(e.reshape(-1, e.shape[-1]))
        e = np.concatenate(out)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)

    q = embed(list(pairs.imitation_path))
    c = embed(list(cands.clean_path))
    target = np.array([index[r] for r in pairs.reference_clip_id])
    meta = ds.meta()
    metrics = score_metrics(q @ c.T, target, [meta[cid] for cid in cands.clip_id])
    out = {"model": str(onnx_path), "label": label, "benchmark": bench_source, "sample_rate": sr, "seconds": seconds,
           "providers": sess.get_providers(), "metrics": metrics,
           "published_mrr": QVIM_DEV_PUBLISHED if bench_source == "qvim_dev" else None}
    out_dir = cfg.lake_root / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_{bench_source}_{sr // 1000}k.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
