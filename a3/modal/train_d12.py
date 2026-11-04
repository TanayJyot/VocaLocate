"""
A3 Part 2: d12 picochat ablations on Modal (one H100 per run).

Every run uses the same data, tokenizer, batch and token budget, and seed 42 unless SEEDS says otherwise; only the
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
    # milder compute-matched sharing: 9 unique blocks (3 prelude + 3 core x2 + 3 coda) = 12 effective layers,
    # 25% fewer weights. aspect 85 keeps width at 768 (9*85=765, rounded up to 768).
    "d12-looped-u9": ["--experiment=looped", "--depth=9", "--aspect-ratio=85", "--n-loops=2", "--n-prelude=3",
                      "--n-coda=3"],
    # compute-matched control for looped-l2: plain baseline trained for l2's FLOPs
    # (2520 steps * 12.74e8 / 7.60e8 FLOPs/token = 4224 steps). Later --num-iterations overrides COMMON's.
    "d12-baseline-long": ["--experiment=baseline", "--num-iterations=4224"],
    # looped-l2 plus a learned vector per core pass (2 x 768 extra params), so shared blocks know which pass they're in
    "d12-looped-l2-embed": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1", "--loop-embed"],
    # looped-l2 trained with a loop count drawn uniformly from {1,2,3} each step: same expected compute as l2,
    # evaluated at 2 loops by default and swept over loop counts afterwards (loop_sweep)
    "d12-looped-rand13": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1", "--n-loops-min=1",
                          "--n-loops-max=3"],
    "d12-xiprelu": ["--experiment=xiprelu"],
    "d12-diff-v2": ["--experiment=diff-v2"],
    "d12-depth-attention": ["--experiment=depth-attention"],
    "d12-mgvr": ["--experiment=mgvr"],
    # same-compute candidates from the 2026 modded-nanogpt records: learnable XSA (#82) and MUDD on the last two
    # layers (#81). Both start as the baseline (zero init); MUDD adds ~0.1M weights and ~0.1% FLOPs/token.
    "d12-xsa": ["--experiment=xsa"],
    "d12-mudd": ["--experiment=mudd"],
    # baseline at other init seeds (NANOCHAT_SEED; data order is unchanged) to measure seed-to-seed noise
    "d12-baseline-s1": ["--experiment=baseline"],
    "d12-baseline-s2": ["--experiment=baseline"],
}
SEEDS = {"d12-baseline-s1": 1, "d12-baseline-s2": 2}  # everything else uses nanochat's default seed 42
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
               OMP_NUM_THREADS="1", WANDB_MODE="offline", WANDB_DIR=f"{BASE}/wandb",
               NANOCHAT_SEED=str(SEEDS.get(tag, 42)))
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


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=2 * 3600)
def eval_loops(tag: str, n_loops: int):
    """val bpb + CORE of a trained looped checkpoint run with n_loops core passes (CORE as in training, bpb on 21M tokens)."""
    env = dict(os.environ, NANOCHAT_BASE_DIR=BASE, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    cmd = ["python", "-m", "scripts.base_eval", "--eval=core,bpb", f"--model-tag={tag}", f"--n-loops={n_loops}",
           "--max-per-task=500", "--split-tokens=20971520"]
    p = subprocess.run(cmd, cwd="/root/nanochat", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout[-4000:])
    def last(pattern):
        m = re.findall(pattern, p.stdout)
        return float(m[-1]) if m else None
    out = {"tag": tag, "n_loops": n_loops, "returncode": p.returncode, "val_bpb": last(r"val bpb: ([\d.]+)"),
           "core": last(r"CORE metric: (-?[\d.]+)")}
    if p.returncode != 0:
        out["error_tail"] = p.stdout[-3000:]
    os.makedirs(f"{BASE}/results", exist_ok=True)
    with open(f"{BASE}/results/sweep-{tag}-n{n_loops}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


def _spawn_once(launch_id, key, fn, *args):
    """Spawn fn(*args), or reattach to the call this launch already spawned. Modal can preempt and restart a
    launcher with the same input; without this, the restart would spawn (and pay for) every run a second time."""
    path = f"{BASE}/launches/{launch_id}/{key}.fc"
    vol.reload()
    if os.path.exists(path):
        return modal.FunctionCall.from_id(open(path).read().strip())
    call = fn.spawn(*args)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(call.object_id)
    vol.commit()
    return call


@app.function(volumes={"/vol": vol}, timeout=3 * 3600)
def loop_sweep(tags: str = "d12-looped-rand13,d12-looped-l2", loops: str = "1,2,3,4,6", launch_id: str = ""):
    """One container per (checkpoint, loop count), all in parallel."""
    jobs = [(t, int(n)) for t in tags.split(",") for n in loops.split(",")]
    launch_id = launch_id or f"sweep-{time.time_ns()}"
    calls = [_spawn_once(launch_id, f"{t}-n{n}", eval_loops, t, n) for t, n in jobs]
    print("spawned:", {f"{t}-n{n}": c.object_id for (t, n), c in zip(jobs, calls)}, flush=True)
    results = []
    for c in calls:
        try:
            results.append(c.get())
        except Exception as e:
            results.append({"error": repr(e)})
    return results


@app.function(volumes={"/vol": vol}, timeout=5 * 3600)
def launch_all(smoke: bool = False, only: str = "", launch_id: str = ""):
    """Spawns every run from inside Modal, so `modal run --detach` keeps them all alive (a detached app only
    keeps the last function the local client triggered) even if the laptop disconnects."""
    tags = [t for t in CONFIGS if not only or t in only.split(",")]
    launch_id = launch_id or f"train-{time.time_ns()}"
    calls = {t: _spawn_once(launch_id, t, train, t, smoke) for t in tags}
    print("spawned:", {t: c.object_id for t, c in calls.items()}, flush=True)
    results = {}
    for t, c in calls.items():
        try:
            results[t] = c.get()
        except Exception as e:
            results[t] = {"error": repr(e)}
    return results


@app.local_entrypoint()
def main(smoke: bool = False, only: str = "", out: str = ""):
    results = launch_all.remote(smoke, only, f"train-{time.time_ns()}") # id fixed by the client, so a restarted launcher reattaches
    for t, r in results.items():
        print(t, json.dumps({k: r.get(k) for k in ("returncode", "val_bpb", "core", "train_minutes", "tok_per_sec",
                                                   "mfu_pct", "peak_mem_mib", "fa3", "container_s", "error")}), flush=True)
    name = out or ("smoke_results.json" if smoke else "d12_results.json")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "w") as f:
        json.dump(results, f, indent=2)
