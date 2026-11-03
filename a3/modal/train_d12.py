"""
A3 Part 2: d12 picochat ablations on Modal (one H100 per run).

Every run uses the same data, tokenizer, seed, batch and token budget; only the
architecture changes, selected with nanochat's --experiment flag so that each
ablation runs standalone against the baseline.

Setup (once):
    modal volume create vocalocate-a3
    modal volume put vocalocate-a3 <local tokenizer dir> /a3/tokenizer
    modal run train_d12.py::prep            # downloads the data shards into the volume
Smoke test (a few steps of every config, checks CUDA/FA3/CORE paths):
    modal run train_d12.py --smoke
Full runs (detached, results land in the volume under /a3/results):
    modal run --detach train_d12.py
"""
import json
import os
import re
import subprocess
import time

import modal

NANOCHAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nanochat")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    # kernels/hf-hub pinned to nanochat's uv.lock: kernels>=0.18 needs an explicit version= in get_kernel,
    # which nanochat doesn't pass, so FA3 would silently fall back to SDPA
    .pip_install("numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests", "jinja2", "datasets",
                 "kernels==0.11.7", "huggingface-hub==0.34.4", "rustbpe", "wandb")
    .add_local_dir(NANOCHAT, "/root/nanochat", ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache"])
)
vol = modal.Volume.from_name("vocalocate-a3")
app = modal.App("vocalocate-a3-d12", image=image)
BASE = "/vol/a3"  # NANOCHAT_BASE_DIR: data, tokenizer, checkpoints, eval bundle, wandb, logs

# d12 at nanochat defaults (width 768, 2048 context, SSSL windows, head dim 128). The token budget and
# batch are pinned to the baseline's compute-optimal values (12 tokens per scaling param = 1.32B tokens,
# 2^19 tokens per step) so ablations that add parameters (Diff V2) don't get a longer schedule.
STEPS = 2520
COMMON = ["--depth=12", f"--num-iterations={STEPS}", "--total-batch-size=524288", "--device-batch-size=32",
          "--core-metric-every=999999", "--sample-every=-1", "--save-every=-1"]
CONFIGS = {
    "d12-baseline": ["--experiment=baseline"],
    "d12-looped-l2": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1"],
    # compute-matched looping: 7 unique blocks (1 prelude + 5 core x2 + 1 coda) = 12 effective layers, so the
    # same FLOPs/token as the baseline with 42% fewer transformer weights. aspect 109 keeps width at 768
    # (7*109=763, rounded up to a multiple of head_dim 128). Mirrors base-d10 vs looped-d6-l2 in the d6 study.
    "d12-looped-u7": ["--experiment=looped", "--depth=7", "--aspect-ratio=109", "--n-loops=2", "--n-prelude=1",
                      "--n-coda=1"],
    "d12-xiprelu": ["--experiment=xiprelu"],
    "d12-diff-v2": ["--experiment=diff-v2"],
    "d12-depth-attention": ["--experiment=depth-attention"],
    "d12-mgvr": ["--experiment=mgvr"],
}
SMOKE = ["--num-iterations=30", "--eval-every=15", "--eval-tokens=1048576", "--core-metric-max-per-task=20"]


@app.function(volumes={"/vol": vol}, timeout=3600, cpu=8)
def prep(num_shards: int = 32):
    env = dict(os.environ, NANOCHAT_BASE_DIR=BASE)
    subprocess.run(["python", "-m", "nanochat.dataset", "-n", str(num_shards), "-w", "16"],
                   cwd="/root/nanochat", env=env, check=True)
    vol.commit()
    print(sorted(os.listdir(f"{BASE}/base_data_climbmix"))[-3:])


def _metrics(log):
    def last(pattern, cast=float):
        m = re.findall(pattern, log)
        return cast(m[-1].replace(",", "")) if m else None
    steps = [(int(a), float(b), float(c.replace(",", "")), float(d))
             for a, b, c, d in re.findall(r"step (\d+)/\d+ .*?dt: ([\d.]+)ms \| tok/sec: ([\d,]+) \| bf16_mfu: ([\d.]+)", log)]
    steady = [s for s in steps if s[0] >= 20] or steps
    return {
        "val_bpb": last(r"Validation bpb: ([\d.]+)"),
        "min_val_bpb": last(r"Minimum validation bpb: ([\d.]+)"),
        "core": last(r"CORE metric: (-?[\d.]+)"),
        "train_minutes": last(r"Total training time: ([\d.]+)m"),
        "peak_mem_mib": last(r"Peak memory usage: ([\d.]+)MiB"),
        "flops_per_token": last(r"Estimated FLOPs per token: ([\d.e+]+)"),
        "tok_per_sec": sum(s[2] for s in steady) / len(steady) if steady else None,
        "mfu_pct": sum(s[3] for s in steady) / len(steady) if steady else None,
        "fa3": "Using Flash Attention 3" in log,
    }


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=4 * 3600)
def train(tag: str, smoke: bool = False):
    t0 = time.time()
    run_tag = f"smoke-{tag}" if smoke else tag
    env = dict(os.environ, NANOCHAT_BASE_DIR=BASE, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", WANDB_MODE="offline", WANDB_DIR=f"{BASE}/wandb")
    os.makedirs(f"{BASE}/wandb", exist_ok=True)
    os.makedirs(f"{BASE}/logs", exist_ok=True)
    cmd = ["python", "-m", "scripts.base_train", *COMMON, *CONFIGS[tag], f"--run={run_tag}", f"--model-tag={run_tag}"]
    if smoke:
        cmd += SMOKE  # later flags win in argparse
    log_path = f"{BASE}/logs/{run_tag}.log"
    lines = []
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd="/root/nanochat", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)
        p.wait()
    log = "".join(lines)
    out = {"tag": run_tag, "returncode": p.returncode, "container_s": time.time() - t0, "cmd": cmd, **_metrics(log)}
    if p.returncode != 0:
        out["error_tail"] = log[-3000:]
    os.makedirs(f"{BASE}/results", exist_ok=True)
    with open(f"{BASE}/results/{run_tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(volumes={"/vol": vol}, timeout=5 * 3600)
def launch_all(smoke: bool = False, only: str = ""):
    """Spawns every run from inside Modal, so `modal run --detach` keeps them all alive (a detached app only
    keeps the last function the local client triggered) even if the laptop disconnects."""
    tags = [t for t in CONFIGS if not only or t in only.split(",")]
    calls = {t: train.spawn(t, smoke) for t in tags}
    print("spawned:", {t: c.object_id for t, c in calls.items()}, flush=True)
    results = {}
    for t, c in calls.items():
        try:
            results[t] = c.get()
        except Exception as e:
            results[t] = {"error": repr(e)}
    return results


@app.local_entrypoint()
def main(smoke: bool = False, only: str = ""):
    results = launch_all.remote(smoke, only)
    for t, r in results.items():
        print(t, json.dumps({k: r.get(k) for k in ("returncode", "val_bpb", "core", "train_minutes", "tok_per_sec",
                                                   "mfu_pct", "peak_mem_mib", "fa3", "container_s", "error")}), flush=True)
    name = "smoke_results.json" if smoke else "d12_results.json"
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "w") as f:
        json.dump(results, f, indent=2)
