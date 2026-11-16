"""Run nanochat GSM8K RL (stock scripts/chat_rl.py, default hyperparameters) on Modal.

Based on Carmen's a4/part-2-sft/modal_chat_sft.py (same image and volume layout), with:
  * per-rollout jsonl logging through rl_rollout_logger.py (no edits to chat_rl.py);
  * periodic volume commits so logs/checkpoints survive a crash;
  * --smoke-steps N to run only N optimisation steps for timing.

Example (full run; uses the repo's a3/nanochat, or set NANOCHAT_REPO to another checkout):
    modal run --detach modal_chat_rl.py \
        --volume vocalocate-a3 --base-path a3 \
        --sft-tag d20-baseline-long-sft --model-tag d20-baseline-long --model-step 932 \
        --output-tag d20-baseline-long-rl --run-name d20-baseline-long-rl \
        --wandb-project a4-rl
Layout produced on the volume:
    <base>/rl_runs/<output-tag>/chatrl_checkpoints/<model-tag>/model_<step>.pt
    <base>/rl_runs/<output-tag>/rollouts/rank{0..7}.jsonl
    <base>/logs/<output-tag>.log, <base>/results/<output-tag>.json, <base>/wandb/
"""

import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import threading
import time

import modal

SCRIPT_PATH = Path(__file__).resolve()
HERE = SCRIPT_PATH.parent
if HERE == Path("/root"):
    REPO = Path("/root/nanochat")
else:
    REPO = Path(os.environ.get("NANOCHAT_REPO", HERE.parents[1] / "a3" / "nanochat")).expanduser().resolve()
    if not (REPO / "scripts" / "chat_rl.py").is_file():
        raise FileNotFoundError(f"Set NANOCHAT_REPO to a nanochat checkout: {REPO}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests",
        "jinja2", "datasets", "kernels==0.11.7", "huggingface-hub==0.34.4",
        "rustbpe", "wandb",
    )
)
image = image.add_local_dir(
    REPO, "/root/nanochat", ignore=[".venv", ".git", "**/__pycache__", "**/.pytest_cache"]
).add_local_file(HERE / "rl_rollout_logger.py", "/root/rl_rollout_logger.py").add_local_file(
    HERE / "eval_logger.py", "/root/eval_logger.py").add_local_file(HERE / "rewards.py", "/root/rewards.py")
app = modal.App("nanochat-a4-rl", image=image)


VOLUME_NAME = os.environ.get("RL_VOLUME", "vocalocate-a3")


@app.function(timeout=12 * 3600, cpu=8, gpu="H100!:8", volumes={"/vol": modal.Volume.from_name(VOLUME_NAME)})
def train(volume: str, base_path: str, sft_tag: str, model_tag: str, model_step: int,
          output_tag: str, run_name: str, gpus: int, smoke_steps: int, rl_args: tuple,
          wandb_mode: str, wandb_project: str, reward: str = "strict"):
    vol = modal.Volume.from_name(volume)
    base_dir = Path("/vol") / base_path
    sft_dir = base_dir / "sft_runs" / sft_tag
    ckpt = sft_dir / "chatsft_checkpoints" / model_tag / f"model_{model_step:06d}.pt"
    meta = sft_dir / "chatsft_checkpoints" / model_tag / f"meta_{model_step:06d}.json"
    for p in (ckpt, meta, base_dir / "tokenizer" / "tokenizer.pkl"):
        if not p.is_file():
            raise FileNotFoundError(p)

    run_dir = base_dir / "rl_runs" / output_tag
    out_dir = run_dir / "chatrl_checkpoints" / model_tag
    # Auto-resume: Modal retries a preempted call with the same inputs, so if this output tag already has
    # RL checkpoints, continue after the latest one instead of starting over (see rl_rollout_logger.py).
    saved = sorted(int(p.stem.split("_")[1]) for p in out_dir.glob("model_*.pt")) if out_dir.exists() else []
    resume = saved[-1] + 1 if saved else 0
    total_steps = smoke_steps or 7473 // 16  # GSM8K train size // examples_per_step (defaults)
    if resume >= total_steps:
        raise FileExistsError(f"Run already finished: {out_dir} has step {saved[-1]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    links = {"chatsft_checkpoints": sft_dir / "chatsft_checkpoints", "tokenizer": base_dir / "tokenizer"}
    if (sft_dir / "task_data").is_dir():
        links["task_data"] = sft_dir / "task_data"  # reuse cached GSM8K parquet shards
    for name, target in links.items():
        link = run_dir / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(exist_ok=True)
    # A resumed run must keep its reward: refuse to continue a tag under a different RL_REWARD.
    reward_file = run_dir / "reward.txt"
    if resume and reward_file.exists() and reward_file.read_text().strip() != reward:
        raise ValueError(f"{output_tag} was trained with reward {reward_file.read_text().strip()!r}, "
                         f"not {reward!r}; use a new --output-tag")
    reward_file.write_text(reward + "\n")
    log_dir = base_dir / "logs"
    log_path = log_dir / f"{output_tag}.log"
    t_offset, kept_log = 0.0, []
    if resume:
        # Drop rollouts and log lines from steps >= resume (they were redone after the preemption).
        for f in rollout_dir.glob("rank*.jsonl"):
            rows = [l for l in f.read_text().splitlines() if l.strip()]
            keep = [l for l in rows if json.loads(l)["step"] < resume]
            f.write_text("".join(l + "\n" for l in keep))
            print(f"[launcher] {f.name}: kept {len(keep)}/{len(rows)} records (step < {resume})", flush=True)
        if log_path.exists():
            lines = log_path.read_text().splitlines(keepends=True)
            cut = next((i for i, l in enumerate(lines) if f"Step {resume}/" in l), len(lines))
            kept_log = lines[:cut]
            stamps = [float(m.group(1)) for l in kept_log if (m := re.match(r"\[\s*([\d.]+)s\]", l))]
            t_offset = stamps[-1] if stamps else 0.0
        vol.commit()

    result_dir, wandb_dir = base_dir / "results", base_dir / "wandb"
    for d in (log_dir, result_dir, wandb_dir):
        d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(run_dir), RL_ROLLOUT_DIR=str(rollout_dir),
               WANDB_MODE=wandb_mode, WANDB_DIR=str(wandb_dir), WANDB_PROJECT=wandb_project,
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    if smoke_steps:
        env["RL_SMOKE_STEPS"] = str(smoke_steps)
    env["RL_REWARD"] = reward  # training reward spec (rewards.py); evaluation always stays strict
    if resume:
        env["RL_RESUME_STEP"] = str(resume)
        run_name = f"{run_name}-resume{resume}"

    command = ["torchrun", "--standalone", f"--nproc_per_node={gpus}", "/root/rl_rollout_logger.py", "--",
               f"--run={run_name}", f"--model-tag={model_tag}", f"--model-step={model_step}", *rl_args]
    print("Launching:", shlex.join(command), flush=True)

    stop = threading.Event()
    def committer():
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:  # never kill training over a commit hiccup
                print(f"[launcher] volume commit failed: {e}", flush=True)
    threading.Thread(target=committer, daemon=True).start()

    started = time.time() - t_offset  # resumed logs continue the original clock (preemption gap excluded)
    with log_path.open("w") as log:
        log.writelines(kept_log)
        if resume:
            log.write(f"[{t_offset:9.1f}s] [launcher] preempted; resuming at step {resume} from checkpoint {resume - 1}\n")
        proc = subprocess.Popen(command, cwd="/root/nanochat", env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            stamped = f"[{time.time() - started:9.1f}s] {line}"
            print(stamped, end="", flush=True)
            log.write(stamped)
        rc = proc.wait()
    stop.set()
    result = {"command": command, "input_checkpoint": str(ckpt), "output_dir": str(out_dir),
              "resumed_at_step": resume or None, "reward": reward,
              "rollout_dir": str(rollout_dir), "gpus": gpus, "smoke_steps": smoke_steps,
              "wandb_mode": wandb_mode, "wandb_project": wandb_project, "returncode": rc,
              "elapsed_seconds": time.time() - started, "log_path": str(log_path)}
    (result_dir / f"{output_tag}.json").write_text(json.dumps(result, indent=2))
    vol.commit()
    if rc:
        raise RuntimeError(f"RL failed with exit code {rc}; see {log_path}")
    return result


@app.function(timeout=4 * 3600, cpu=8, gpu="H100!:8", volumes={"/vol": modal.Volume.from_name(VOLUME_NAME)})
def final_eval(volume: str, base_path: str, output_tag: str, model_tag: str, checkpoints: tuple):
    """Stock chat_eval (all 5 tasks, full test sets, greedy, 512 new tokens: Karpathy's report-card
    protocol) on each (source, step) in `checkpoints`, logging every GSM8K completion."""
    vol = modal.Volume.from_name(volume)
    run_dir = Path("/vol") / base_path / "rl_runs" / output_tag
    eval_dir = run_dir / "final_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(run_dir), EVAL_LOG_DIR=str(eval_dir),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    summary = {}
    for source, step in checkpoints:
        name = f"{source}_step{step:06d}"
        env["EVAL_LOG_NAME"] = f"gsm8k_{name}"
        cmd = ["torchrun", "--standalone", "--nproc_per_node=8", "/root/eval_logger.py", "--",
               "-i", source, "-g", model_tag, "-s", str(step)]
        print("Launching:", shlex.join(cmd), flush=True)
        log_path = eval_dir / f"chat_eval_{name}.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen(cmd, cwd="/root/nanochat", env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                if "\r" in line:  # drop the in-place progress counter
                    line = line.rsplit("\r", 1)[-1]
                print(line, end="", flush=True)
                log.write(line)
            rc = proc.wait()
        text = log_path.read_text()
        import re
        summary[name] = {"returncode": rc,
                         **{m.group(1): float(m.group(2)) / 100 for m in re.finditer(r"^(\S+) accuracy: ([\d.]+)%", text, re.M)},
                         **({"ChatCORE": float(m.group(1))} if (m := re.search(r"ChatCORE metric: ([\d.-]+)", text)) else {})}
        vol.commit()
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    vol.commit()
    return summary


@app.function(timeout=600, volumes={"/vol": modal.Volume.from_name(VOLUME_NAME)})
def kickoff(*args, fn: str = "train"):
    """Spawn the GPU job from inside Modal and return at once. A detached app whose client drops
    mid-call can stop that call (it did once, on a laptop DNS failure); a call spawned by a
    container is not tied to the laptop at all."""
    call = {"train": train, "final_eval": final_eval}[fn].spawn(*args)
    return call.object_id


@app.local_entrypoint()
def main(volume: str, base_path: str, sft_tag: str, model_tag: str, model_step: int,
         output_tag: str, run_name: str, gpus: int = 8, smoke_steps: int = 0, rl_args: str = "",
         wandb_mode: str = "offline", wandb_project: str = "a4-rl", final_eval: str = "",
         reward: str = "strict"):
    if final_eval:
        # e.g. --final-eval sft:932,rl:466  (same protocol for every checkpoint)
        ckpts = tuple((s.split(":")[0], int(s.split(":")[1])) for s in final_eval.split(","))
        if any(src not in ("sft", "rl") for src, _ in ckpts):
            raise ValueError("--final-eval entries must be sft:<step> or rl:<step>")
        call_id = kickoff.remote(volume, base_path, output_tag, model_tag, ckpts, fn="final_eval")
        print(f"Spawned final-eval call {call_id}. Results: /{base_path}/rl_runs/{output_tag}/final_eval/")
        return
    if volume != VOLUME_NAME:
        raise ValueError(f"Set RL_VOLUME={volume} in the environment (the volume is bound at import time)")
    if gpus != 8:
        raise ValueError("This launcher is pinned to 8xH100 (Karpathy's setup)")
    for label, v in (("sft_tag", sft_tag), ("model_tag", model_tag), ("output_tag", output_tag)):
        if not v or "/" in v or "\\" in v or v in (".", ".."):
            raise ValueError(f"{label} must be a single directory name")
    if not run_name or run_name == "dummy":
        raise ValueError("Pass a W&B run name other than 'dummy'")
    extra = tuple(shlex.split(rl_args))
    if any(a.split("=", 1)[0] in ("--run", "--model-tag", "--model-step") for a in extra):
        raise ValueError("Pass run/checkpoint options through the named launcher arguments")
    import sys
    sys.path.insert(0, str(HERE))
    import rewards  # validate the reward spec locally before paying for GPUs
    rewards.parse_spec(reward)
    call_id = kickoff.remote(volume, base_path, sft_tag, model_tag, model_step, output_tag, run_name, gpus,
                             smoke_steps, extra, wandb_mode, wandb_project, reward)
    print(f"Spawned training call {call_id}. Follow /{base_path}/logs/{output_tag}.log on the volume.")
