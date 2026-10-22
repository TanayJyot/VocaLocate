"""Run the recipe comparison on Modal: every (seed, recipe) training in its own GPU container, in parallel.

The runs are independent, so 7 recipes x 3 seeds take about as long as one training. Each container
runs the normal CLI (`vocalocate-demo recipe --models <one> --no-results-file`) against a copy of the
lake on a Modal volume, so results are identical in form to a local run; a final CPU step collects
each seed's results.json.

Data comes from, and models go back to, the AWS lake (infra/): before training, `pull_lake` copies the
warehouse and the features from s3://<lake bucket> into the volume (only new or changed files), and
afterwards `publish` copies models/<source>/ to s3://<models bucket>/<source>/. Both assume an AWS
role with the container's Modal OIDC token, so no AWS keys are stored in Modal. One-time setup, per
environment, after `terraform apply` in infra/envs/<env>:

    terraform -chdir=infra/envs/prod output -raw modal_secret_command | sh   # creates vocalocate-aws-prod

Without AWS (`--no-sync`), upload the lake to the volume by hand instead, as before (the app still
references the secret, so create one with any values, e.g. `modal secret create vocalocate-aws-prod X=1`):

    modal volume create vocalocate-lake
    modal volume put vocalocate-lake <lake>/warehouse/vocalocate.duckdb /warehouse/vocalocate.duckdb
    modal volume put vocalocate-lake <lake>/gold/features/vis /gold/features/vis
    modal volume put vocalocate-lake <lake>/gold/features/qvim_dev /gold/features/qvim_dev

Feature paths in the warehouse are re-rooted onto the volume when loaded (data.rebase_lake_path);
the AWS pipeline already writes them under /lake, the volume's mount point.

Run (from the repo root):

    modal run demo/modal/recipe_app.py --seeds 9999 --models excluded --epochs 1   # probe: time one epoch
    modal run demo/modal/recipe_app.py                                            # 3 seeds x 7 recipes
    aws s3 sync s3://<models bucket>/vis ./runs_modal/vis                         # fetch results
    vocalocate-demo summarize runs_modal/vis/recipe-490 runs_modal/vis/recipe-491 runs_modal/vis/recipe-492

A run that was interrupted resumes: finished recipes (<name>.json on the volume) are skipped.
The GPU type comes from VOCALOCATE_MODAL_GPU (default A100-80GB; batch 64 needs ~30 GB), the
volume from VOCALOCATE_MODAL_VOLUME (default vocalocate-lake) and the AWS secret from
VOCALOCATE_MODAL_AWS_SECRET (default vocalocate-aws-prod; use vocalocate-aws-dev with `modal run -e dev`).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# Locally this file is demo/modal/recipe_app.py in the checkout. Inside a container Modal mounts it
# alone (as /root/recipe_app.py) and the image already holds the sources under /repo.
_parents = Path(__file__).resolve().parents
REPO = _parents[2] if modal.is_local() and len(_parents) > 2 else Path("/repo")
GPU = os.environ.get("VOCALOCATE_MODAL_GPU", "A100-80GB")
VOLUME = os.environ.get("VOCALOCATE_MODAL_VOLUME", "vocalocate-lake")
AWS_SECRET = os.environ.get("VOCALOCATE_MODAL_AWS_SECRET", "vocalocate-aws-prod")
RECIPES = "greif,shared,excluded,excluded_noaug,retrieval,metadata,random"  # cli.RECIPE_ORDER
LAKE = "/lake"

_ignore = ["**/__pycache__/**", "**/*.pyc", "**/tests/**", "**/.pytest_cache/**", "**/*.egg-info/**"]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch==2.5.1", "numpy<2.3", "scipy", "soundfile", "pandas", "pyarrow", "duckdb", "boto3")
    # Only the packages' sources: a working checkout can hold gigabytes of untracked run output.
    .add_local_file(REPO / "pipeline" / "pyproject.toml", "/repo/pipeline/pyproject.toml", copy=True)
    .add_local_dir(REPO / "pipeline" / "src", "/repo/pipeline/src", copy=True, ignore=_ignore)
    .add_local_file(REPO / "demo" / "pyproject.toml", "/repo/demo/pyproject.toml", copy=True)
    .add_local_dir(REPO / "demo" / "src", "/repo/demo/src", copy=True, ignore=_ignore)
    .run_commands("pip install --no-deps -e /repo/pipeline -e /repo/demo")
)
lake = modal.Volume.from_name(VOLUME, create_if_missing=True)
# The app name is part of the OIDC subject that the AWS role trusts (infra/modules/access).
app = modal.App("vocalocate-recipe", image=image)
aws = modal.Secret.from_name(AWS_SECRET)


def _cli(args: list[str]) -> None:
    from vocalocate_demo.cli import main

    code = main(["--lake", LAKE, "-v", *args])
    if code not in (0, None):
        raise RuntimeError(f"vocalocate-demo {' '.join(args)} exited with {code}")


def _s3():
    """S3 access as the training role, from this container's Modal identity token."""
    from vocalocate_pipeline.sync import s3_from_web_identity

    return s3_from_web_identity(os.environ["VOCALOCATE_AWS_ROLE_ARN"], os.environ["MODAL_IDENTITY_TOKEN"],
                                os.environ["AWS_REGION"], session_name="modal-vocalocate-recipe")


@app.function(volumes={LAKE: lake}, secrets=[aws], timeout=3 * 3600, cpu=4.0, memory=8 * 1024)
def pull_lake(source: str, bench: str) -> str:
    """Copy the warehouse and the features training needs from the S3 lake into the volume."""
    from vocalocate_pipeline.sync import sync

    lake.reload()
    prefixes = ["warehouse", f"gold/features/{source}", f"gold/features/{bench}"]
    res = sync(os.environ["VOCALOCATE_LAKE_URI"], LAKE, prefixes, s3=_s3())
    lake.commit()
    return f"pulled {res.copied} files ({res.bytes / 1e9:.2f} GB), {res.skipped} already up to date"


@app.function(volumes={LAKE: lake}, secrets=[aws], timeout=3600, cpu=2.0)
def publish(source: str) -> str:
    """Copy this source's checkpoints and results from the volume to the S3 models bucket."""
    from vocalocate_pipeline.sync import sync

    lake.reload()
    res = sync(f"{LAKE}/models/{source}", f"{os.environ['VOCALOCATE_MODELS_URI']}/{source}", s3=_s3())
    return f"published {res.copied} files ({res.bytes / 1e9:.2f} GB) to {os.environ['VOCALOCATE_MODELS_URI']}/{source}"


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
            "--no-results-file", "--preload"]
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
         source: str = "vis", bench: str = "qvim_dev", sync: bool = True):
    if sync:
        print(pull_lake.remote(source, bench))
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
    if sync:
        print(publish.remote(source))
    if failed:
        raise SystemExit(f"{len(failed)} trainings failed; re-run the same command to retry only those")
