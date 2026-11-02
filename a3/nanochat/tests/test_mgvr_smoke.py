"""Exercise the real compiled trainer, checkpoint save and optimizer resume."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_xiprelu_smoke import LOCAL_TRAINER, local_data


@pytest.mark.parametrize("combined", [False, True])
def test_compiled_training_and_resume(local_data, combined):
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(local_data), NANOCHAT_DTYPE="float32",
                       PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        environment.pop(name, None)
    command = [sys.executable, "-c", LOCAL_TRAINER, "--mgvr", "--device-type=cpu",
               "--depth=2", "--aspect-ratio=16", "--head-dim=8", "--max-seq-len=16",
               "--window-pattern=L", "--device-batch-size=2", "--total-batch-size=32",
               "--eval-every=1", "--eval-tokens=32", "--core-metric-every=-1", "--sample-every=-1",
               "--warmup-steps=0", "--warmdown-ratio=0", "--weight-decay=0", "--run=dummy"]
    tag = "d2-mgvr"
    if combined:
        command += ["--diff-v2", "--depth-attention", "--mlp-activation=xiprelu", "--n-loops=2"]
        tag = "d2-l2-xiprelu-diff-v2-depth-attention-mgvr"
    directory = local_data / "base_checkpoints" / tag
    previous = None
    for step, options in ((3, ["--num-iterations=3"]), (4, ["--num-iterations=4", "--resume-from-step=3"])):
        result = subprocess.run(command + options, cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        meta = json.loads((directory / f"meta_{step:06d}.json").read_text())
        weights = torch.load(directory / f"model_{step:06d}.pt", weights_only=True)
        assert meta["model_config"]["mgvr"] is True
        assert meta["mgvr_metrics"]["additional_parameters"] == (4 + 4**2) * 32 + 4**2
        assert meta["mgvr_metrics"]["training_tokens_per_second"] > 0
        assert meta["mgvr_metrics"]["gpu_cost"] is None
        assert torch.isfinite(torch.tensor(meta["val_bpb"]))
        assert meta["mgvr_stats"]["added_to_current_norm"] > 0
        assert meta["mgvr_stats"]["offdiagonal_mix_norm"] > 0
        assert (directory / f"optim_{step:06d}_rank0.pt").is_file()
        mgvr = {name: value for name, value in weights.items() if ".mgvr." in name}
        assert len(mgvr) == 2 and all(torch.isfinite(p).all() and p.abs().sum() > 0 for p in mgvr.values())
        if previous is not None:
            assert all(not torch.equal(p, previous[name]) for name, p in mgvr.items())
        previous = mgvr
    assert {path.name for path in directory.parent.iterdir()} == {tag}
    # A user-supplied directory must not silently resume with MGVR switched off.
    mismatch = [arg for arg in command if arg != "--mgvr"]
    result = subprocess.run(mismatch + [f"--model-tag={tag}", "--num-iterations=5", "--resume-from-step=4"],
                            cwd=Path(__file__).resolve().parents[1], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "checkpoint's --mgvr setting" in result.stderr
