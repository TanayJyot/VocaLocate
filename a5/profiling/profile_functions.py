"""A5 Part One: profile five important VocaLocate functions with cProfile.

Each function runs on synthetic inputs that are sized like our real data (QVIM / VimSketch
scale), so the profile reflects where time goes in practice without needing the lake or a GPU.

    1. pipeline  flow_functions.audio.clean_array    - cleans every ingested clip (A2 pipeline)
    2. pipeline  flow_functions.features.log_mel     - featurises every clean clip and every query
    3. demo      evaluate.score_metrics              - runs after every training epoch and in fusion tuning
    4. demo      mining.mine                         - picks hard negatives for fine-tuning
    5. demo      train._unique_reference_batches     - shuffles pairs into batches, once per epoch

Run from the repo root:

    PYTHONPATH="pipeline;demo/src" python a5/profiling/profile_functions.py --label before

(use ":" instead of ";" in PYTHONPATH on Linux/macOS). For every function this writes
`results/<label>/<name>.prof` (load it with `python -m pstats` or snakeviz) and
`results/<label>/<name>.txt` (the top entries by cumulative time), plus a `summary.json`
with the profiled wall time, the best of three unprofiled runs and the call count.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vocalocate_demo.data import SFXMeta
from vocalocate_demo.evaluate import score_metrics
from vocalocate_demo.mining import MiningConfig, mine
from vocalocate_demo.train import _unique_reference_batches
from vocalocate_pipeline.config import AudioConfig, FeatureConfig
from vocalocate_pipeline.flow_functions.audio import clean_array
from vocalocate_pipeline.flow_functions.features import log_mel

HERE = Path(__file__).resolve().parent
SEED = 490


def _unit(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _meta(rng: np.random.Generator, n: int, n_cats: int = 40, n_tags: int = 60) -> list[SFXMeta]:
    """Candidate metadata: categories, a few tags each, and recordings with 1-3 takes."""
    out, rec = [], 0
    while len(out) < n:
        cat = f"cat{rng.integers(n_cats)}"
        for _ in range(int(rng.integers(1, 4))):
            tags = frozenset(f"t{t}" for t in rng.choice(n_tags, size=4, replace=False))
            out.append(SFXMeta(cat, tags, f"rec{rec}"))
        rec += 1
    return out[:n]


# --------------------------------------------------------------------------- the five workloads


def work_clean_array():
    """200 ten-second stereo 44.1 kHz clips: decode output -> 32 kHz mono, trimmed, normalised."""
    rng = np.random.default_rng(SEED)
    cfg = AudioConfig()
    clips = [(0.3 * rng.standard_normal((441_000, 2))).astype(np.float32) for _ in range(200)]

    def run():
        return [clean_array(c, 44_100, cfg) for c in clips]

    return run


def work_log_mel():
    """400 cleaned five-second 32 kHz clips -> (128, frames) log-mels."""
    rng = np.random.default_rng(SEED)
    cfg = FeatureConfig()
    clips = [(0.1 * rng.standard_normal(160_000)).astype(np.float32) for _ in range(400)]

    def run():
        return [log_mel(x, 32_000, cfg) for x in clips]

    return run


def work_score_metrics():
    """1,500 queries ranked against 3,000 candidates (a full-size VimSketch test split)."""
    rng = np.random.default_rng(SEED)
    n_q, n_c = 1_500, 3_000
    sims = _unit(rng, n_q, 128) @ _unit(rng, n_c, 128).T
    target = rng.integers(0, n_c, size=n_q)
    meta = _meta(rng, n_c)

    def run():
        return score_metrics(sims, target, meta)

    return run


def work_mine():
    """Retrieval-driven and metadata-guided mining for 1,500 imitations over a 1,500-SFX pool."""
    rng = np.random.default_rng(SEED)
    n = 1_500
    pool_emb = _unit(rng, n, 128)
    pool_ids = [f"sfx{i}" for i in range(n)]
    meta = dict(zip(pool_ids, _meta(rng, n)))
    pos = rng.integers(0, n, size=n)
    # Imitations sit near their positive, so the semi-hard band is populated.
    q_emb = pool_emb[pos] + 0.08 * rng.standard_normal((n, 128)).astype(np.float32)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    q_ids = [f"imi{i}" for i in range(n)]
    pos_ids = [pool_ids[p] for p in pos]
    cfg = MiningConfig()

    def run():
        return {
            s: mine(s, q_emb, q_ids, pos_ids, pool_emb, pool_ids, meta, cfg, "v", "r")
            for s in ("retrieval", "metadata")
        }

    return run


def work_unique_reference_batches():
    """12,000 training pairs over 3,000 references (about 4 imitations each), batch size 32."""
    rng = np.random.default_rng(SEED)
    refs = rng.integers(0, 3_000, size=12_000)
    pairs = pd.DataFrame({"reference_clip_id": [f"sfx{r}" for r in refs]})

    def run():
        return _unique_reference_batches(pairs, 32, np.random.default_rng(SEED))

    return run


WORKLOADS = {
    "clean_array": work_clean_array,
    "log_mel": work_log_mel,
    "score_metrics": work_score_metrics,
    "mine": work_mine,
    "unique_reference_batches": work_unique_reference_batches,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="before", help="results subfolder, e.g. before / after")
    ap.add_argument("--only", nargs="*", choices=list(WORKLOADS), help="profile only these")
    ap.add_argument("--top", type=int, default=25, help="rows kept in each text report")
    args = ap.parse_args()

    out_dir = HERE / "results" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, build in WORKLOADS.items():
        if args.only and name not in args.only:
            continue
        run = build()  # input generation is not profiled
        prof = cProfile.Profile()
        t0 = time.perf_counter()
        prof.enable()
        run()
        prof.disable()
        wall = time.perf_counter() - t0
        # cProfile charges every Python call, which exaggerates code with many small calls, so the
        # summary also keeps the best of three plain (unprofiled) runs.
        plain = []
        for _ in range(3):
            t1 = time.perf_counter()
            run()
            plain.append(time.perf_counter() - t1)
        prof.dump_stats(out_dir / f"{name}.prof")
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf).strip_dirs().sort_stats("cumulative")
        buf.write(f"# {name}: {wall:.3f} s wall time under cProfile ({args.label})\n")
        stats.print_stats(args.top)
        (out_dir / f"{name}.txt").write_text(buf.getvalue(), encoding="utf-8")
        summary[name] = {
            "profiled_sec": round(wall, 4),
            "plain_best_of_3_sec": round(min(plain), 4),
            "function_calls": stats.total_calls,
        }
        print(f"{name:28s} profiled {wall:7.3f} s   plain {min(plain):7.3f} s   {stats.total_calls:>10,d} calls")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
