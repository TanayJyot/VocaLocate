"""The one-command demo: data -> pipeline -> baseline -> two mining approaches -> evaluation -> search.

    workdir/
      data/sfx_meta/          synthetic dataset in the sfx_meta layout (A2 "data we can generate")
      lake/                   the A2 data lake + warehouse, plus models/ and the mined negatives
      library/<project>/      a simulated messy user library: held-out SFX renamed SFX_0001.wav, ...
      queries/                held-out imitations to try, with queries.csv naming each one's target
      results.json, report.md
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np

from vocalocate_pipeline import warehouse
from vocalocate_pipeline.config import PipelineConfig
from vocalocate_pipeline.flows import run_dataset_flow

from . import train as T
from .experiment import MODEL_NAMES, model_dir, run_experiment
from .mining import MiningConfig
from .report import write_report
from .search import embed_query, index_library, search
from .synth import generate

log = logging.getLogger(__name__)

SOURCE = "sfx_meta"
PROJECTS = ("trailer_2025", "podcast_ep4", "game_jam", "unsorted")


def build_messy_library(cfg: PipelineConfig, workdir: Path, n_queries: int, seed: int = 490) -> list[dict]:
    """Copy held-out SFX into project folders under meaningless names, and pick queries to try."""
    rng = np.random.default_rng(seed)
    lib, qdir = workdir / "library", workdir / "queries"
    shutil.rmtree(lib, ignore_errors=True)
    shutil.rmtree(qdir, ignore_errors=True)
    cands = warehouse.query(
        cfg,
        f"""SELECT c.clip_id, r.uri, c.category FROM v_sfx_candidates c
            JOIN raw_files r ON r.source = c.source AND r.file_sha256 = c.clip_id
            WHERE c.source = '{SOURCE}' AND c.split = 'test' ORDER BY c.clip_id""",
    )
    new_name = {}
    for i, k in enumerate(rng.permutation(len(cands))):
        clip_id, uri, _ = cands[k]
        dest = lib / PROJECTS[int(rng.integers(len(PROJECTS)))] / f"SFX_{i:04d}.wav"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(uri, dest)
        new_name[clip_id] = str(dest.resolve())

    pairs = warehouse.query(
        cfg,
        f"""SELECT p.imitation_clip_id, ri.uri, p.reference_clip_id, p.label FROM v_training_pairs p
            JOIN raw_files ri ON ri.source = p.source AND ri.file_sha256 = p.imitation_clip_id
            WHERE p.source = '{SOURCE}' AND p.split = 'test' ORDER BY p.imitation_clip_id""",
    )
    # One query per reference, spread over categories.
    by_ref = {}
    for row in [pairs[i] for i in rng.permutation(len(pairs))]:
        by_ref.setdefault(row[2], row)
    picked, seen_cat = [], set()
    for row in sorted(by_ref.values(), key=lambda r: (r[3] in seen_cat, r[0])):
        if len(picked) == n_queries:
            break
        if row[2] in new_name:
            picked.append(row)
            seen_cat.add(row[3])
    qdir.mkdir(parents=True, exist_ok=True)
    queries = []
    for _, uri, ref, label in picked:
        dest = qdir / f"{Path(uri).stem}.wav"
        shutil.copy(uri, dest)
        queries.append({"query": str(dest.resolve()), "category": label, "target": new_name[ref]})
    with open(qdir / "queries.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query", "category", "target"])
        w.writeheader()
        w.writerows(queries)
    return queries


def demo_searches(cfg: PipelineConfig, workdir: Path, queries: list[dict], k: int = 5) -> dict:
    out = {"library": {}, "queries": []}
    indexes, models = {}, {}
    for name in MODEL_NAMES:
        path = model_dir(cfg, SOURCE) / f"{name}.pt"
        if not path.exists():
            continue
        model, tcfg, ckpt = T.load(path)
        t0 = time.time()
        # Scan the folder once (first model); later models reuse the same library index.
        idx = index_library(cfg, [workdir / "library"], "demo", model, tcfg, ckpt["model_version"], scan=not indexes)
        out["library"][name] = {"files": len(idx.clip_ids), "embedded": idx.n_embedded, "sec": time.time() - t0}
        indexes[name], models[name] = idx, (model, tcfg)
    for q in queries:
        entry = {**q, "results": {}}
        for name, idx in indexes.items():
            model, tcfg = models[name]
            vec = embed_query(Path(q["query"]), model, tcfg, cfg)
            hits = search(idx, vec, k=len(idx.clip_ids))
            target_rank = next(h["rank"] for h in hits if h["path"] == q["target"])
            entry["results"][name] = {"target_rank": target_rank, "top": hits[:k]}
        out["queries"].append(entry)
    return out


def run_demo(
    workdir: Path,
    cfg: PipelineConfig,
    tcfg: T.TrainConfig,
    mcfg: MiningConfig,
    small: bool = False,
    n_queries: int = 6,
    sweep: bool = False,
) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data_dir = workdir / "data" / SOURCE
    if not (data_dir / "metadata.csv").exists():
        log.info("generating synthetic dataset in %s", data_dir)
        kwargs = dict(recordings_per_category=4, distractors_per_category=1, imitations_per_reference=4) if small else {}
        generate(data_dir, **kwargs)
    log.info("running the A2 dataset flow")
    flow = run_dataset_flow(SOURCE, data_dir, cfg)
    results = run_experiment(cfg, SOURCE, tcfg, mcfg, sweep=sweep)
    results["pipeline_run"] = flow.as_dict()
    results["quality"] = [list(r) for r in warehouse.query(cfg, "SELECT * FROM v_clip_quality")]

    queries = build_messy_library(cfg, workdir, n_queries)
    results["search_demo"] = demo_searches(cfg, workdir, queries)
    results["total_sec"] = round(time.time() - t0, 1)
    (workdir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    write_report(results, workdir / "report.md")
    return results
