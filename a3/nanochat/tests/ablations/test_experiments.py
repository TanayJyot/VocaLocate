"""Explicit experiment selection and its real trainer integration."""
from itertools import product
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nanochat.experiments import EXPERIMENTS, select_experiment
from test_xiprelu_smoke import LOCAL_TRAINER, local_data


def options(**overrides):
    values = dict(experiment=None, n_loops=1, n_prelude=0, n_coda=0,
                  mlp_activation="relu2", diff_v2=False, depth_attention=False,
                  depth_attention_stride=None, mgvr=False)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("selected", EXPERIMENTS[:-1])
def test_selection_enables_only_requested_architecture(selected):
    args = options(experiment=selected, n_loops=3 if selected == "looped" else 1)
    select_experiment(args)
    assert args.mlp_activation == ("xiprelu" if selected == "xiprelu" else "relu2")
    assert args.n_loops == (3 if selected == "looped" else 1)
    for flag in ("diff-v2", "depth-attention", "mgvr"):
        assert getattr(args, flag.replace("-", "_")) == (flag == selected)


@pytest.mark.parametrize("selected,extra", [
    (selected, extra) for selected, extra in product(EXPERIMENTS[:-1], EXPERIMENTS[1:-1]) if selected != extra
])
def test_selection_rejects_other_enabled_ablations(selected, extra):
    flags = dict(n_loops=2 if selected == "looped" else 1)
    if extra == "looped":
        flags["n_loops"] = 2
    elif extra == "xiprelu":
        flags["mlp_activation"] = "xiprelu"
    else:
        flags[extra.replace("-", "_")] = True
    with pytest.raises(ValueError, match="conflicts with enabled ablations"):
        select_experiment(options(experiment=selected, **flags))


@pytest.mark.parametrize("flag", [dict(n_prelude=1), dict(n_coda=1), dict(depth_attention_stride=2),
                                  dict(loop_embed=True), dict(n_loops_max=3)])
def test_baseline_rejects_stray_architecture_options(flag):
    with pytest.raises(ValueError):
        select_experiment(options(experiment="baseline", **flag))


def test_intentional_combinations_and_legacy_calls():
    args = options(experiment="combined", mgvr=True, depth_attention=True)
    select_experiment(args)
    assert args.mgvr and args.depth_attention
    with pytest.raises(ValueError, match="at least two"):
        select_experiment(options(experiment="combined", mgvr=True))
    with pytest.raises(ValueError, match="n-loops >= 2"):
        select_experiment(options(experiment="looped"))
    legacy = options(mgvr=True, diff_v2=True, n_loops=2)
    before = vars(legacy).copy()
    select_experiment(legacy)
    assert vars(legacy) == before


def test_trainer_rejects_conflicts_before_compute_or_data_loading():
    result = subprocess.run([sys.executable, "-m", "scripts.base_train", "--experiment=baseline", "--mgvr"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 2
    assert "--experiment=baseline conflicts with enabled ablations: mgvr" in result.stderr


def test_selected_mgvr_runs_real_compiled_trainer(local_data):
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(local_data), NANOCHAT_DTYPE="float32",
                       WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        environment.pop(name, None)
    result = subprocess.run([
        sys.executable, "-c", LOCAL_TRAINER, "--experiment=mgvr", "--device-type=cpu",
        "--depth=2", "--aspect-ratio=16", "--head-dim=8", "--max-seq-len=16",
        "--window-pattern=L", "--device-batch-size=2", "--total-batch-size=32",
        "--num-iterations=1", "--warmup-steps=0", "--warmdown-ratio=0",
        "--eval-every=-1", "--core-metric-every=-1", "--sample-every=-1",
    ], env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = json.loads((local_data / "base_checkpoints/d2-mgvr/meta_000001.json").read_text())
    assert metadata["user_config"]["experiment"] == "mgvr"
    assert metadata["model_config"]["mgvr"] is True
    assert not metadata["model_config"].get("diff_v2", False)
    assert not metadata["model_config"].get("depth_attention", False)
