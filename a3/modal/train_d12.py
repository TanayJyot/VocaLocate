"""
A3 Parts 2, 3 and 5: d12 picochat ablations on Modal (one GPU per run) and the final d20 combined model (8 H100s). Tags ending in -a100, -rtx, -b200 or -l40s rerun an
ablation on that hardware for Part 3, and save checkpoints so a preempted run resumes instead of restarting.

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
import glob
import subprocess
import threading
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
    # Selective Self-Attention in the form OpenNeuro ran it (their A3): one scalar per head and layer, no extra matmuls
    "d12-ssa": ["--experiment=ssa"],
    # baseline at other init seeds (NANOCHAT_SEED; data order is unchanged) to measure seed-to-seed noise
    "d12-baseline-s1": ["--experiment=baseline"],
    "d12-baseline-s2": ["--experiment=baseline"],
    # looped-l2 and looped-l2-embed at the same extra seeds, to compare all three models over seeds 42 / 1 / 2
    "d12-looped-l2-s1": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1"],
    "d12-looped-l2-s2": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1"],
    "d12-looped-l2-embed-s1": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1", "--loop-embed"],
    "d12-looped-l2-embed-s2": ["--experiment=looped", "--n-loops=2", "--n-prelude=1", "--n-coda=1", "--loop-embed"],
}
# Part 3: the two final-model ablations on other hardware, same flags, seed and token budget as their H100 runs
HARDWARE = ("a100", "rtx", "b200", "l40s")
for _hw in HARDWARE:
    CONFIGS[f"d12-diff-v2-{_hw}"] = CONFIGS["d12-diff-v2"]
    CONFIGS[f"d12-looped-l2-embed-{_hw}"] = CONFIGS["d12-looped-l2-embed"]
# Without FA3, Diff V2 runs out of memory on the RTX PRO 6000 (96 GB) at device batch 32; the total batch is unchanged
CONFIGS["d12-diff-v2-rtx"] = CONFIGS["d12-diff-v2"] + ["--device-batch-size=16"]
# Looped-l2-embed needs about 46 GB at device batch 32, more than the L40S has; same total batch
CONFIGS["d12-looped-l2-embed-l40s"] = CONFIGS["d12-looped-l2-embed"] + ["--device-batch-size=16"]
# Part 5: Diff V2 + looped-l2-embed at d20, with the baseline d20's 9,960 steps x 524,288 = 5.22B tokens. Device batch 8
# fits 80 GB (d12 combined needs 55 GB at batch 32); the total batch is what sets the training.
CONFIGS["d20-combined"] = ["--experiment=combined", "--diff-v2", "--n-loops=2", "--n-prelude=1", "--n-coda=1", "--loop-embed",
                           "--depth=20", "--num-iterations=9960", "--device-batch-size=8", "--max-seq-len=2048", "--eval-every=500"]
SEEDS = {"d12-baseline-s1": 1, "d12-baseline-s2": 2, "d12-looped-l2-s1": 1, "d12-looped-l2-s2": 2,
         "d12-looped-l2-embed-s1": 1, "d12-looped-l2-embed-s2": 2}  # everything else uses nanochat's default seed 42
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


def _checkpoint_steps(ckpt_dir):
    return sorted(int(m) for f in glob.glob(f"{ckpt_dir}/model_*.pt") for m in re.findall(r"model_(\d+)\.pt", f))


def _last_checkpoint(ckpt_dir, nproc=1):
    """Newest step with a complete checkpoint. A step is skipped if a file is missing or smaller than the previous
    checkpoint's, since a container can be preempted while it is writing."""
    def sizes(n):
        paths = [f"{ckpt_dir}/model_{n:06d}.pt"] + [f"{ckpt_dir}/optim_{n:06d}_rank{r}.pt" for r in range(nproc)]
        return [os.path.getsize(x) if os.path.exists(x) else -1 for x in paths]
    good = [n for n in _checkpoint_steps(ckpt_dir) if os.path.exists(f"{ckpt_dir}/meta_{n:06d}.json") and min(sizes(n)) > 0]
    for i in range(len(good) - 1, -1, -1):
        if i == 0 or all(x >= y for x, y in zip(sizes(good[i]), sizes(good[i - 1]))):
            return good[i]
    return None


def _prune_checkpoints(ckpt_dir, keep=2):
    for n in _checkpoint_steps(ckpt_dir)[:-keep]:
        for f in glob.glob(f"{ckpt_dir}/*_{n:06d}*"):
            os.remove(f)


def _train(tag: str, smoke: bool = False, extra=(), nproc=1, sdpa=False):
    t0 = time.time()
    run_tag = f"smoke-{tag}" if smoke else tag
    resumable = tag.rsplit("-", 1)[-1] in HARDWARE or tag.startswith("d20-")
    env = dict(os.environ, NANOCHAT_BASE_DIR=BASE, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", WANDB_MODE="offline", WANDB_DIR=f"{BASE}/wandb",
               NANOCHAT_SEED=str(SEEDS.get(tag, 42)))
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if sdpa or "B200" in gpu or "RTX PRO" in gpu:
        # Blackwell: kernels-community/flash-attn3 has no build for it, but nanochat still loads it and crashes
        # ("no kernel image"); blocking the hub download makes it fall back to SDPA
        env["HF_HUB_OFFLINE"] = "1"
    os.makedirs(f"{BASE}/wandb", exist_ok=True)
    os.makedirs(f"{BASE}/logs", exist_ok=True)
    os.makedirs(f"{BASE}/runstate", exist_ok=True)
    vol.reload()
    launcher = ["python", "-m", "scripts.base_train"]
    if nproc > 1:
        launcher = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "-m", "scripts.base_train", "--"]
    cmd = [*launcher, *COMMON, *CONFIGS[tag], f"--run={run_tag}", f"--model-tag={run_tag}"]
    if smoke:
        cmd += SMOKE  # later flags win in argparse
    cmd += list(extra)
    ckpt_dir = f"{BASE}/base_checkpoints/{run_tag}"
    state_path = f"{BASE}/runstate/{run_tag}.json"
    prior_s, attempts, resume_step = 0.0, 1, None
    if resumable:
        if not smoke:
            cmd.append("--save-every=500")
        resume_step = _last_checkpoint(ckpt_dir, nproc)
        if resume_step is not None:
            cmd.append(f"--resume-from-step={resume_step}")
        if os.path.exists(state_path):
            state = json.load(open(state_path))
            prior_s, attempts = state["container_s"], state["attempts"] + 1
    done = threading.Event()

    def heartbeat():
        # Keeps checkpoints and the billed-time total on the volume, so a preempted run can pick up from them
        while not done.wait(300):
            try:
                _prune_checkpoints(ckpt_dir)
                with open(state_path, "w") as f:
                    json.dump({"container_s": prior_s + time.time() - t0, "attempts": attempts}, f)
                vol.commit()
            except Exception as e:
                print("heartbeat failed:", repr(e), flush=True)

    if resumable:
        threading.Thread(target=heartbeat, daemon=True).start()
    log_path = f"{BASE}/logs/{run_tag}.log"
    lines = []
    with open(log_path, "a" if resume_step is not None else "w") as f:
        f.write(" ".join(cmd) + "\n")
        p = subprocess.Popen(cmd, cwd="/root/nanochat", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="")
            f.write(line)
            lines.append(line)
        p.wait()
    done.set()
    log = "".join(lines)
    out = {"tag": run_tag, "returncode": p.returncode, "container_s": prior_s + time.time() - t0, "cmd": cmd, "gpu": gpu,
           "attempts": attempts, "resumed_from": resume_step, **_metrics(log)}
    if p.returncode != 0:
        out["error_tail"] = log[-3000:]
    os.makedirs(f"{BASE}/results", exist_ok=True)
    with open(f"{BASE}/results/{run_tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    vol.commit()
    return out


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=4 * 3600)
def train(tag: str, smoke: bool = False):
    return _train(tag, smoke)


# Modal fixes the GPU at decoration time, so one function per hardware. A100-80GB because looped-l2-embed peaks at 45 GB
# at device batch 32, and keeping that batch keeps these runs identical to the H100 ones.
@app.function(gpu="A100-80GB", volumes={"/vol": vol}, timeout=10 * 3600)
def train_a100(tag: str, smoke: bool = False):
    return _train(tag, smoke)


@app.function(gpu="RTX-PRO-6000", volumes={"/vol": vol}, timeout=10 * 3600)
def train_rtx(tag: str, smoke: bool = False):
    return _train(tag, smoke)


# Not B200+: it hands out B300s, and torch 2.9.1 cannot compile for them (sm_103a), even when assembled for sm_100a
@app.function(gpu="B200", volumes={"/vol": vol}, timeout=10 * 3600)
def train_b200(tag: str, smoke: bool = False):
    return _train(tag, smoke)


@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=10 * 3600)
def train_l40s(tag: str, smoke: bool = False):
    return _train(tag, smoke)


@app.function(gpu="H100!:8", volumes={"/vol": vol}, timeout=14 * 3600)
def train_d20(tag: str, smoke: bool = False):
    return _train(tag, smoke, nproc=8)


@app.function(volumes={"/vol": vol}, timeout=2 * 3600, cpu=8)
def prepare_d20(shards: int = 125):
    """Downloads the training shards d20 needs: 5.22B tokens is about 117 shards. Existing shards are kept."""
    env = dict(os.environ, NANOCHAT_BASE_DIR=BASE, PYTHONIOENCODING="utf-8")
    subprocess.run(["python", "-m", "nanochat.dataset", "-n", str(shards)], cwd="/root/nanochat", env=env, check=True)
    vol.commit()
    return len(glob.glob(f"{BASE}/base_data_climbmix/*.parquet"))


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=1800)
def sdpa_check():
    """H100 without FA3, the attention path Blackwell is limited to: does that explain the B200's low MFU?"""
    out = {}
    for tag in ("d12-diff-v2", "d12-looped-l2-embed"):
        r = _train(tag, True, sdpa=True)
        out[tag] = {k: r.get(k) for k in ("returncode", "fa3", "tok_per_sec", "mfu_pct", "peak_mem_mib", "gpu")}
    print("SDPA_CHECK", json.dumps(out), flush=True)
    return out


@app.function(gpu="H100!:4", volumes={"/vol": vol}, timeout=3600)
def d20_check():
    """End-to-end check of d20 on 4 GPUs: memory and speed, one checkpoint save, then a resume as if preempted."""
    tag, run_tag = "d20-combined", "smoke-d20-combined"
    ckpt_dir = f"{BASE}/base_checkpoints/{run_tag}"
    vol.reload()
    for f in glob.glob(f"{ckpt_dir}/*") + glob.glob(f"{BASE}/runstate/{run_tag}.json"):
        os.remove(f)
    args = ["--num-iterations=20", "--save-every=10", "--eval-every=-1", "--core-metric-every=-1"]
    first = _train(tag, True, args, nproc=4)
    for f in glob.glob(f"{ckpt_dir}/*_000020*"):
        os.remove(f)
    picked = _last_checkpoint(ckpt_dir, 4)
    second = _train(tag, True, args, nproc=4)
    files = {os.path.basename(f): os.path.getsize(f) for f in glob.glob(f"{ckpt_dir}/*_000010*")}
    result = {"first_rc": first["returncode"], "picked": picked, "second_rc": second["returncode"], "resumed_from": second["resumed_from"],
              "tok_per_sec": first["tok_per_sec"], "mfu_pct": first["mfu_pct"], "peak_mem_mib": first["peak_mem_mib"], "fa3": first["fa3"],
              "flops_per_token": first["flops_per_token"], "ckpt_gb": sum(files.values()) / 1e9, "first_s": first["container_s"],
              "error_tail": (first.get("error_tail", "") + second.get("error_tail", ""))[-1200:],
              "final_exists": os.path.exists(f"{ckpt_dir}/model_000020.pt")}
    print("D20_CHECK", json.dumps(result), flush=True)
    return result


def _trainer(tag):
    if tag.startswith("d20-"):
        return train_d20
    return {"a100": train_a100, "rtx": train_rtx, "b200": train_b200, "l40s": train_l40s}.get(tag.rsplit("-", 1)[-1], train)


@app.function(gpu="RTX-PRO-6000", volumes={"/vol": vol}, timeout=1800)
def resume_check():
    """End-to-end check of auto-resume: train to step 20, drop that checkpoint as if preempted, then resume from 10."""
    tag, run_tag = "d12-diff-v2-rtx", "smoke-d12-diff-v2-rtx"
    ckpt_dir = f"{BASE}/base_checkpoints/{run_tag}"
    vol.reload()
    for f in glob.glob(f"{ckpt_dir}/*") + glob.glob(f"{BASE}/runstate/{run_tag}.json"):
        os.remove(f)
    args = ["--num-iterations=20", "--save-every=10", "--eval-every=-1", "--core-metric-every=-1"]
    first = _train(tag, True, args)
    for f in glob.glob(f"{ckpt_dir}/*_000020*"):
        os.remove(f)
    picked = _last_checkpoint(ckpt_dir)
    second = _train(tag, True, args)
    result = {"first_rc": first["returncode"], "picked": picked, "second_rc": second["returncode"],
              "resumed_from": second["resumed_from"], "error_tail": second.get("error_tail", "")[-800:],
              "final_exists": os.path.exists(f"{ckpt_dir}/model_000020.pt")}
    print("RESUME_CHECK", json.dumps(result), flush=True)
    return result


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
    calls = {t: _spawn_once(launch_id, t, _trainer(t), t, smoke) for t in tags}
    print("spawned:", {t: c.hydrate().object_id for t, c in calls.items()}, flush=True)
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
