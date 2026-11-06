"""Explicit experiment selection and its real trainer integration."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nanochat.experiments import EXPERIMENTS, select_experiment
from smoke_helpers import LOCAL_TRAINER, local_data


def options(**overrides):
    values = dict(experiment=None, n_loops=1, n_prelude=0, n_coda=0, loop_embed=False, diff_v2=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_experiments_are_the_kept_models():
    assert EXPERIMENTS == ("baseline", "looped", "diff-v2", "combined")


def test_selection_enables_only_requested_architecture():
    args = options(experiment="diff-v2")
    select_experiment(args)
    assert args.diff_v2 and args.n_loops == 1
    args = options(experiment="looped", n_loops=2)
    select_experiment(args)
    assert not args.diff_v2 and args.n_loops == 2


def test_selection_rejects_other_enabled_ablations():
    with pytest.raises(ValueError, match="conflicts with enabled ablations"):
        select_experiment(options(experiment="baseline", diff_v2=True))
    with pytest.raises(ValueError, match="conflicts with enabled ablations"):
        select_experiment(options(experiment="diff-v2", n_loops=2))


@pytest.mark.parametrize("flag", [dict(n_prelude=1), dict(n_coda=1), dict(loop_embed=True)])
def test_baseline_rejects_stray_architecture_options(flag):
    with pytest.raises(ValueError):
        select_experiment(options(experiment="baseline", **flag))


def test_combined_needs_both_changes():
    args = options(experiment="combined", diff_v2=True, n_loops=2, n_prelude=1, n_coda=1, loop_embed=True)
    select_experiment(args)
    assert args.diff_v2 and args.n_loops == 2
    with pytest.raises(ValueError, match="at least two"):
        select_experiment(options(experiment="combined", diff_v2=True))
    with pytest.raises(ValueError, match="n-loops >= 2"):
        select_experiment(options(experiment="looped"))


def test_trainer_rejects_conflicts_before_compute_or_data_loading():
    result = subprocess.run([sys.executable, "-m", "scripts.base_train", "--experiment=baseline", "--diff-v2"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "--experiment=baseline conflicts with enabled ablations: diff-v2" in result.stderr


def test_selected_diff_v2_runs_real_compiled_trainer(local_data):
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(local_data), NANOCHAT_DTYPE="float32",
                       WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        environment.pop(name, None)
    result = subprocess.run([
        sys.executable, "-c", LOCAL_TRAINER, "--experiment=diff-v2", "--device-type=cpu",
        "--depth=2", "--aspect-ratio=16", "--head-dim=8", "--max-seq-len=16",
        "--window-pattern=L", "--device-batch-size=2", "--total-batch-size=32",
        "--num-iterations=1", "--warmup-steps=0", "--warmdown-ratio=0",
        "--eval-every=-1", "--core-metric-every=-1", "--sample-every=-1",
    ], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((local_data / "base_checkpoints/d2-diff-v2/meta_000001.json").read_text())
    assert metadata["user_config"]["experiment"] == "diff-v2"
    assert metadata["model_config"]["diff_v2"] is True
