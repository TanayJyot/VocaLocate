"""Exercise the real trainer offline, reusing the shared smoke data."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from smoke_helpers import LOCAL_TRAINER, local_data


@pytest.mark.parametrize("activation", ["relu2"])
def test_compiled_diff_v2_training_and_resume(local_data, activation):
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(local_data),
                       NANOCHAT_DTYPE="float32", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUTF8="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        environment.pop(name, None)
    command = [
        sys.executable, "-c", LOCAL_TRAINER, "--diff-v2",
        "--device-type=cpu", "--depth=2", "--aspect-ratio=16", "--head-dim=8",
        "--max-seq-len=16", "--window-pattern=L", "--device-batch-size=2",
        "--total-batch-size=32", "--eval-every=1", "--eval-tokens=32",
        "--core-metric-every=-1", "--sample-every=-1", "--warmup-steps=0",
        "--warmdown-ratio=0", "--weight-decay=0", "--run=dummy",
    ]
    tag = "d2-diff-v2"
    directory = local_data / "base_checkpoints" / tag
    previous = None
    for step, options in ((3, ["--num-iterations=3"]),
                          (4, ["--num-iterations=4", "--resume-from-step=3"])):
        result = subprocess.run(command + options, cwd=Path(__file__).resolve().parents[2],
                                env=environment, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        meta = json.loads((directory / f"meta_{step:06d}.json").read_text())
        weights = torch.load(directory / f"model_{step:06d}.pt", weights_only=True)
        assert meta["model_config"]["diff_v2"] is True
        assert meta["model_config"]["n_head"] == meta["model_config"]["n_kv_head"] == 4
        assert torch.isfinite(torch.tensor(meta["val_bpb"])) and meta["val_bpb"] > 0
        assert all(torch.isfinite(value).all() for value in weights.values())
        assert (directory / f"optim_{step:06d}_rank0.pt").is_file()
        for layer in range(2):
            key = f"transformer.h.{layer}.attn.lambda_proj.weight"
            assert weights[key].shape == (4, 32) and weights[key].abs().sum() > 0
            assert weights[f"transformer.h.{layer}.attn.c_q.weight"].shape == (64, 32)
            if previous is not None:
                assert not torch.equal(weights[key], previous[key])
        assert "activation_stats" not in meta
        assert "SDPA fallback" in result.stdout
        previous = weights
        print(f"{tag} step {step}: {sum(v.numel() for v in weights.values()):,} parameters, "
              f"validation BPB {meta['val_bpb']:.6f}; SDPA, CPU FP32, compiled model/optimizer")
    assert {path.name for path in directory.parent.iterdir()} == {tag}
