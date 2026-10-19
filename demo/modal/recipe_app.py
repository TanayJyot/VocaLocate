"""Run the recipe comparison on Modal: every (seed, recipe) training in its own GPU container, in parallel.

The runs are independent, so 7 recipes x 3 seeds take about as long as one training. Each container
runs the normal CLI (`vocalocate-demo recipe --models <one> --no-results-file`) against a copy of the
lake on a Modal volume, so results are identical in form to a local run; a final CPU step collects
each seed's results.json.

One-time setup, from the machine that holds the lake (only the warehouse and the features are needed):

    modal volume create vocalocate-lake
    modal volume put vocalocate-lake <lake>/warehouse/vocalocate.duckdb /warehouse/vocalocate.duckdb
    modal volume put vocalocate-lake <lake>/gold/features/vis /gold/features/vis
    modal volume put vocalocate-lake <lake>/gold/features/qvim_dev /gold/features/qvim_dev

Feature paths in the warehouse are re-rooted onto the volume when loaded (data.rebase_lake_path).

Run (from the repo root):

    modal run demo/modal/recipe_app.py --seeds 9999 --models excluded --epochs 1   # probe: time one epoch
    modal run demo/modal/recipe_app.py                                            # 3 seeds x 7 recipes
    modal volume get vocalocate-lake /models/vis ./runs_modal/                    # fetch results
    vocalocate-demo summarize runs_modal/vis/recipe-490 runs_modal/vis/recipe-491 runs_modal/vis/recipe-492

A run that was interrupted resumes: finished recipes (<name>.json on the volume) are skipped.
The GPU type comes from VOCALOCATE_MODAL_GPU (default A100-80GB; batch 64 needs ~30 GB) and the
volume from VOCALOCATE_MODAL_VOLUME (default vocalocate-lake).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2]
GPU = os.environ.get("VOCALOCATE_MODAL_GPU", "A100-80GB")
VOLUME = os.environ.get("VOCALOCATE_MODAL_VOLUME", "vocalocate-lake")
RECIPES = "greif,shared,excluded,excluded_noaug,retrieval,metadata,random"  # cli.RECIPE_ORDER
LAKE = "/lake"

_ignore = ["**/__pycache__/**", "**/*.pyc", "**/tests/**", "**/.pytest_cache/**", "**/*.egg-info/**"]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch==2.5.1", "numpy<2.3", "scipy", "soundfile", "pandas", "pyarrow", "duckdb")
    .add_local_dir(REPO / "pipeline", "/repo/pipeline", copy=True, ignore=_ignore)
    .add_local_dir(REPO / "demo", "/repo/demo", copy=True, ignore=_ignore + ["modal/**"])
    .run_commands("pip install --no-deps -e /repo/pipeline -e /repo/demo")
)
lake = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App("vocalocate-recipe", image=image)


def _cli(args: list[str]) -> None:
    from vocalocate_demo.cli import main

    code = main(["--lake", LAKE, "-v", *args])
    if code not in (0, None):
        raise RuntimeError(f"vocalocate-demo {' '.join(args)} exited with {code}")


@app.function(gpu=GPU, volumes={LAKE: lake}, timeout=8 * 3600, cpu=8.0, memory=48 * 1024)
def train_one(source: str, name: str, seed: int, batch_size: int, epochs: int, bench: str) -> dict:
    import json
    import time

    lake.reload()
    out = Path(LAKE) / "models" / source / f"recipe-{seed}" / f"{name}.json"
    if out.exists():
        return {"seed": seed, "name": name, "status": "already done"}
    t0 = time.time()
    args = ["recipe", "--source", source, "--models", name, "--seed", str(seed), "--encoder", "mobilenetv3",
            "--device", "cuda", "--frames", "1000", "--batch-size", str(batch_size), "--bench", bench,
            "--no-results-file"]
    if epochs:
        args += ["--recipe-epochs", str(epochs)]
    _cli(args)
    lake.commit()
    res = json.loads(out.read_text(encoding="utf-8"))
    hist = res["history"]
    return {"seed": seed, "name": name, "status": "done", "wall_sec": round(time.time() - t0),
            "epoch_sec": round(sum(h["sec"] for h in hist) / len(hist), 1), "gpu": res.get("gpu"),
            "kept_epoch": res["kept_epoch"], "test_mrr": res["test"]["mrr"],
            "bench_mrr": res.get("benchmark", {}).get(bench, {}).get("mrr")}


@app.function(volumes={LAKE: lake}, timeout=3600, cpu=4.0, memory=16 * 1024)
def collect(source: str, seed: int, names: str, bench: str) -> None:
    """Write results.json for one seed from the finished per-recipe files (nothing retrains)."""
    lake.reload()
    _cli(["recipe", "--source", source, "--models", names, "--seed", str(seed), "--encoder", "mobilenetv3",
          "--device", "cpu", "--frames", "1000", "--bench", bench])
    lake.commit()


@app.local_entrypoint()
def main(seeds: str = "490,491,492", models: str = RECIPES, batch_size: int = 64, epochs: int = 0,
         source: str = "vis", bench: str = "qvim_dev"):
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    names = [n.strip() for n in models.split(",") if n.strip()]
    jobs = [(source, n, s, batch_size, epochs, bench) for s in seed_list for n in names]
    print(f"{len(jobs)} trainings on {GPU}, batch {batch_size}{f', {epochs} epochs' if epochs else ''}")
    failed = []
    for job, res in zip(jobs, train_one.starmap(jobs, return_exceptions=True, order_outputs=True)):
        if isinstance(res, Exception):
            failed.append(job)
            print(f"FAILED seed {job[2]} {job[1]}: {res!r}")
        else:
            print(res)
    for s in seed_list:
        if not any(j[2] == s for j in failed):
            collect.remote(source, s, ",".join(names), bench)
            print(f"seed {s}: results.json written")
    if failed:
        raise SystemExit(f"{len(failed)} trainings failed; re-run the same command to retry only those")
