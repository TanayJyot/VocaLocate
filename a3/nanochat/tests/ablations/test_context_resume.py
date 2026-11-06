"""Exercise a context switch without shortening the optimizer schedule."""
import json
import os
from pathlib import Path
import subprocess
import sys

from smoke_helpers import LOCAL_TRAINER, local_data


def test_stop_and_resume_with_longer_context(local_data):
    env = dict(os.environ, NANOCHAT_BASE_DIR=str(local_data), NANOCHAT_DTYPE="float32",
               PYTHONUTF8="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        env.pop(name, None)
    command = [
        sys.executable, "-c", LOCAL_TRAINER, "--device-type=cpu",
        "--depth=2", "--aspect-ratio=16", "--head-dim=8", "--window-pattern=L",
        "--device-batch-size=2", "--total-batch-size=64", "--num-iterations=6",
        "--warmup-steps=0", "--warmdown-ratio=1", "--eval-every=1", "--eval-tokens=64",
        "--core-metric-every=-1", "--sample-every=-1",
    ]
    cwd = Path(__file__).resolve().parents[2]
    checkpoint_dir = local_data / "base_checkpoints" / "d2"

    short = subprocess.run(command + ["--max-seq-len=16", "--stop-at-step=2"],
                           cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert short.returncode == 0, short.stdout + short.stderr
    assert "lrm: 0.84" in short.stdout  # step 1 on the six-step schedule
    assert {p.name for p in checkpoint_dir.glob("meta_*.json")} == {"meta_000002.json"}
    assert (checkpoint_dir / "optim_000002_rank0.pt").is_file()

    long = subprocess.run(command + ["--max-seq-len=32", "--resume-from-step=2"],
                          cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert long.returncode == 0, long.stdout + long.stderr
    assert "step 00002/00006" in long.stdout
    for step, length in ((2, 16), (6, 32)):
        meta = json.loads((checkpoint_dir / f"meta_{step:06d}.json").read_text())
        assert meta["model_config"]["sequence_len"] == length
        assert meta["total_batch_size"] == 64
