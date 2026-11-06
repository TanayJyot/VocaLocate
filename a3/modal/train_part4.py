"""Train the A3 context extension on one H100, with local evaluation afterward."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import modal

ROOT = Path(__file__).resolve().parents[1]
NANOCHAT = "/root/nanochat"
BASE = Path("/vol/part4")
VOLUME_NAME = "vocalocate-a3-part4"
TOKENIZER_HASHES = {
    "tokenizer.pkl": "46a1a55e1154ec68b217dde7deffbfeb1f754e911982d9d7f53a7523b356ab89",
    "token_bytes.pt": "42317d1460274d9f175fe3532d0cc6a04d06396ed06d3338ce3f733c9f986665",
}
SHARD_NAMES = {f"shard_{i:05d}.parquet" for i in (*range(32), 6542)}
SMOKE_TAG = "d12-context-smoke"
TRAIN_TAG = "d12-context-part4"
SWITCH_STEP = 1260
FINAL_STEP = 2520
TOTAL_BATCH_SIZE = 524288

# Match the successful Part 2 CUDA/FA3 stack.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("numpy", "tiktoken", "filelock", "psutil", "pyarrow", "requests",
                 "jinja2", "datasets", "kernels==0.11.7", "huggingface-hub==0.34.4",
                 "rustbpe", "wandb")
    .add_local_dir(ROOT / "nanochat", NANOCHAT,
                   ignore=[".git", ".venv", "**/__pycache__", "**/.pytest_cache"])
    .add_local_file(ROOT / "tokenizer" / "tokenizer.pkl", "/root/part4-tokenizer/tokenizer.pkl")
    .add_local_file(ROOT / "tokenizer" / "token_bytes.pt", "/root/part4-tokenizer/token_bytes.pt")
)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("vocalocate-a3-part4", image=image)


def verify_tokenizer(directory):
    for name, expected in TOKENIZER_HASHES.items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")


def verify_inputs():
    verify_tokenizer(BASE / "tokenizer")
    actual = {path.name for path in (BASE / "base_data_climbmix").glob("*.parquet")}
    if actual != SHARD_NAMES:
        raise ValueError(f"Unexpected ClimbMix manifest: missing {sorted(SHARD_NAMES - actual)}, "
                         f"extra {sorted(actual - SHARD_NAMES)}")


@app.function(volumes={"/vol": vol}, timeout=45 * 60, cpu=8, max_containers=1)
def prep():
    """Verify Tanay's tokenizer and cache exactly 32 training shards plus validation."""
    source = Path("/root/part4-tokenizer")
    verify_tokenizer(source)
    target = BASE / "tokenizer"
    target.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_HASHES:
        shutil.copyfile(source / name, target / name)
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(BASE), PYTHONUNBUFFERED="1")
    subprocess.run([sys.executable, "-m", "nanochat.dataset", "-n", "32", "-w", "8"],
                   cwd=NANOCHAT, env=env, check=True)
    verify_inputs()
    vol.commit()
    print("Verified Tanay's tokenizer and cached 32 ClimbMix training shards plus validation.")


def checkpoint_meta(tag, step, sequence_len):
    directory = BASE / "base_checkpoints" / tag
    try:
        meta = json.loads((directory / f"meta_{step:06d}.json").read_text())
    except (OSError, ValueError):
        return None
    if meta["step"] != step or meta["model_config"]["sequence_len"] != sequence_len:
        return None
    if not all(zipfile.is_zipfile(directory / name)
               for name in (f"model_{step:06d}.pt", f"optim_{step:06d}_rank0.pt")):
        return None
    if not math.isfinite(meta["val_bpb"]):
        raise ValueError(f"Non-finite validation loss at step {step}")
    return meta


def run_stage(tag, sequence_len, end_step, total_steps, resume_step, batch_size, smoke):
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(BASE), NANOCHAT_SEED="42", PYTHONUNBUFFERED="1",
               PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1")
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        "--experiment=baseline", "--depth=12", "--window-pattern=SSSL", "--device-batch-size=32",
        "--core-metric-every=-1", "--sample-every=-1", f"--model-tag={tag}",
        f"--max-seq-len={sequence_len}", f"--num-iterations={total_steps}",
        f"--total-batch-size={batch_size}", f"--stop-at-step={end_step}",
        f"--resume-from-step={resume_step}", "--save-every=500",
    ]
    cmd += (["--eval-every=1", "--eval-tokens=131072", "--warmup-steps=1"] if smoke else
            ["--eval-every=250", "--eval-tokens=8388608"])
    log_dir = BASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{tag}-{sequence_len}.log").open("a", encoding="utf-8", buffering=1) as log:
        log.write(" ".join(cmd) + "\n")
        with subprocess.Popen(cmd, cwd=NANOCHAT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8") as process:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                if "without FA3" in line or "SDPA fallback" in line:
                    process.terminate()
                    raise RuntimeError("Expected Flash Attention 3 on the H100")
                if "loss: nan" in line.lower() or "loss: inf" in line.lower():
                    process.terminate()
                    raise RuntimeError(f"Non-finite training loss: {line.strip()}")
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
    meta = checkpoint_meta(tag, end_step, sequence_len)
    if meta is None:
        raise RuntimeError(f"Missing complete checkpoint at {tag} step {end_step}")
    vol.commit()
    print(f"Saved {tag} step {end_step}, context {sequence_len}, val bpb {meta['val_bpb']:.6f}")


def train_context(smoke):
    verify_inputs()
    if not smoke and checkpoint_meta(SMOKE_TAG, 6, 2048) is None:
        raise RuntimeError("Run the H100 smoke test successfully before starting formal training")

    tag = SMOKE_TAG if smoke else TRAIN_TAG
    switch_step, total_steps, batch_size = (2, 6, 65536) if smoke else (
        SWITCH_STEP, FINAL_STEP, TOTAL_BATCH_SIZE)
    if checkpoint_meta(tag, switch_step, 512) is None:
        resume = max((s for s in range(500, switch_step, 500) if checkpoint_meta(tag, s, 512)), default=-1)
        run_stage(tag, 512, switch_step, total_steps, resume, batch_size, smoke)

    if checkpoint_meta(tag, total_steps, 2048) is None:
        resume = max((s for s in range(500, total_steps, 500)
                      if s > switch_step and checkpoint_meta(tag, s, 2048)), default=switch_step)
        run_stage(tag, 2048, total_steps, total_steps, resume, batch_size, smoke)
    return {"checkpoint_tag": tag, "steps": [switch_step, total_steps]}


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=12 * 60, max_containers=1)
def smoke():
    return train_context(smoke=True)


@app.function(gpu="H100!", volumes={"/vol": vol}, timeout=90 * 60, max_containers=1)
def train():
    return train_context(smoke=False)
