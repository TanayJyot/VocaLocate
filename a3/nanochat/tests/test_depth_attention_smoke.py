"""Run the real compiled trainer and checkpoint resume on tiny offline data."""
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
    command = [sys.executable, "-c", LOCAL_TRAINER, "--depth-attention", "--device-type=cpu",
               "--depth=2", "--aspect-ratio=16", "--head-dim=8", "--max-seq-len=16",
               "--window-pattern=L", "--device-batch-size=2", "--total-batch-size=32",
               "--eval-every=1", "--eval-tokens=32", "--core-metric-every=-1", "--sample-every=-1",
               "--warmup-steps=0", "--warmdown-ratio=0", "--weight-decay=0", "--run=dummy"]
    tag = "d2-depth-attention"
    if combined:
        command += ["--diff-v2", "--mlp-activation=xiprelu", "--n-loops=2", "--depth-attention-stride=2"]
        tag = "d2-l2-xiprelu-diff-v2-depth-attention-s2"
    directory = local_data / "base_checkpoints" / tag
    previous = None
    for step, options in ((3, ["--num-iterations=3"]),
                          (4, ["--num-iterations=4", "--resume-from-step=3"])):
        result = subprocess.run(command + options, cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
        meta = json.loads((directory / f"meta_{step:06d}.json").read_text())
        weights = torch.load(directory / f"model_{step:06d}.pt", weights_only=True)
        assert meta["model_config"]["depth_attention"] is True
        assert meta["model_config"]["depth_attention_stride"] == (2 if combined else None)
        metrics = meta["depth_attention_metrics"]
        assert metrics["wall_clock_seconds"] > 0 and metrics["training_tokens_per_second"] > 0
        assert metrics["gpu_cost"] is None and metrics["core_metric"] is None
        assert metrics["peak_cuda_memory_bytes_per_rank"] is None
        assert torch.isfinite(torch.tensor(meta["val_bpb"]))
        assert all(torch.isfinite(value).all() for value in weights.values())
        assert (directory / f"optim_{step:06d}_rank0.pt").is_file()
        key = "transformer.h.0.attn.c_k.weight"
        if previous is not None:
            assert not torch.equal(weights[key], previous[key])
        previous = weights
        print(f"{tag} step {step}: {sum(v.numel() for v in weights.values()):,} parameters; "
              f"BPB={meta['val_bpb']:.6f}; wall={metrics['wall_clock_seconds']:.2f}s; "
              f"throughput={metrics['training_tokens_per_second']:.1f} tokens/s")
    assert {path.name for path in directory.parent.iterdir()} == {tag}
