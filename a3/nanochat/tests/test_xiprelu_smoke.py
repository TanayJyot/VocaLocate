"""Offline CPU smoke test: python -m pytest -q tests/test_xiprelu_smoke.py.

Runs the real trainer with tiny generated data and AOTAutograd compilation.
CUDA, Flash Attention, FP8 and GPU kernel generation need GPU checks separately.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import tiktoken
import torch

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


# Compile the full model and optimizer graphs using the portable CPU backend.
# This changes only the test subprocess; production training keeps its backend.
LOCAL_TRAINER = """
import runpy
import torch

original_compile = torch.compile
def local_compile(model=None, **kwargs):
    kwargs.update(backend="aot_eager", fullgraph=True)
    return original_compile(model, **kwargs)

torch.compile = local_compile
runpy.run_module("scripts.base_train", run_name="__main__")
"""


def report(capsys, *lines):
    # Keep the useful report visible even with pytest's default output capture.
    with capsys.disabled():
        print("\n".join(lines), flush=True)


def test_xiprelu_formula_and_gradients(capsys):
    from nanochat.gpt import XIPReLU

    activation = XIPReLU()
    # Deliberately use different branch coefficients to catch swapped branches.
    # Normal training starts both coefficients at 0.8.
    with torch.no_grad():
        activation.raw_alpha_n.copy_(torch.tensor(0.3).expm1().log())
    inputs = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    actual = activation(inputs)
    expected = torch.tensor([0.2, -0.2, 0., 1.3, 4.2])
    expected_gradient = torch.tensor([-0.7, -0.1, 0.5, 2.1, 3.7])
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(inputs.grad, expected_gradient)
    rows = ["\nxIPReLU formula check: y = alpha_branch * x^2 + 0.5*x",
            "  Demo coefficients: positive=0.8, negative=0.3 (training starts both at 0.8)",
            "       x     ReLU^2   expected xIPReLU   actual xIPReLU   dy/dx"]
    baseline = inputs.detach().relu().square()
    for x, relu2, want, got, grad in zip(inputs.detach(), baseline, expected, actual.detach(), inputs.grad):
        rows.append(f"  {x.item():6.1f}   {relu2.item():8.3f}   {want.item():16.3f}   {got.item():14.3f}   {grad.item():6.3f}")
    report(capsys, *rows,
           "  PASS: both branches match known values; input gradients match the derivative.")


@pytest.fixture
def local_data(tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    encoding = tiktoken.Encoding(
        name="smoke-test-bytes", pat_str=SPLIT_PATTERN,
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={token: 256 + i for i, token in enumerate(SPECIAL_TOKENS)},
    )
    RustBPETokenizer(encoding, "<|bos|>").save(tokenizer_dir)
    torch.save(torch.tensor([1] * 256 + [0] * len(SPECIAL_TOKENS), dtype=torch.int32),
               tokenizer_dir / "token_bytes.pt")
    data_dir = tmp_path / "base_data_climbmix"
    data_dir.mkdir()
    table = pa.table({"text": ["A tiny offline example for activation testing."] * 1024})
    # The real loader uses the last shard for validation and the rest for training.
    for index in range(2):
        pq.write_table(table, data_dir / f"shard_{index:05d}.parquet", row_group_size=128)
    return tmp_path


@pytest.mark.parametrize("activation,n_loops", [("relu2", 1), ("xiprelu", 1), ("xiprelu", 2)])
def test_local_compiled_training_and_resume(local_data, activation, n_loops, capsys):
    label = "ReLU^2 baseline" if activation == "relu2" else "xIPReLU ablation"
    if n_loops > 1:
        label += f" with {n_loops} loops"
    report(capsys, f"\n{label}: CPU, 2 layers, width 32, batch 2 x 16 tokens",
           "  Compiling full model and optimizer graphs with aot_eager; training 3 steps...")
    environment = dict(os.environ, NANOCHAT_BASE_DIR=str(local_data),
                       NANOCHAT_DTYPE="float32", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONUTF8="1", WANDB_MODE="disabled", OMP_NUM_THREADS="1")
    # Each child runs on one CPU with no inherited distributed or compile-disable flags.
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHDYNAMO_DISABLE", "TORCH_COMPILE_DISABLE"):
        environment.pop(name, None)
    command = [
        sys.executable, "-c", LOCAL_TRAINER,
        "--device-type=cpu", "--depth=2", "--aspect-ratio=16", "--head-dim=8",
        "--max-seq-len=16", "--window-pattern=L", "--device-batch-size=2",
        "--total-batch-size=32", "--eval-every=1", "--eval-tokens=32",
        "--core-metric-every=-1", "--sample-every=-1", "--warmup-steps=0",
        "--warmdown-ratio=0", "--weight-decay=0", "--run=dummy",
    ]
    # Exercise the unchanged baseline default as well as the explicit ablation flag.
    if activation == "xiprelu":
        command.append("--mlp-activation=xiprelu")
    if n_loops > 1:
        command.append(f"--n-loops={n_loops}")
    tag = "d2" + (f"-l{n_loops}" if n_loops > 1 else "")
    if activation == "xiprelu":
        tag += "-xiprelu"
    checkpoint_dir = local_data / "base_checkpoints" / tag
    initial_weights = None
    history = {}
    for step, options in ((3, ["--num-iterations=3"]),
                          (4, ["--num-iterations=4", "--resume-from-step=3"])):
        result = subprocess.run(command + options, cwd=Path(__file__).resolve().parents[1],
                                env=environment, capture_output=True, encoding="utf-8", timeout=90)
        assert result.returncode == 0, f"{activation} step {step}\n{result.stdout}\n{result.stderr}"
        metadata = json.loads((checkpoint_dir / f"meta_{step:06d}.json").read_text())
        weights = torch.load(checkpoint_dir / f"model_{step:06d}.pt", map_location="cpu", weights_only=True)
        assert metadata["step"] == step
        assert metadata["model_config"]["n_loops"] == n_loops
        assert metadata["val_bpb"] > 0 and torch.isfinite(torch.tensor(metadata["val_bpb"]))
        assert (checkpoint_dir / f"optim_{step:06d}_rank0.pt").is_file()
        assert all(torch.isfinite(value).all() for value in weights.values())
        activation_keys = [name for name in weights if "raw_alpha" in name]
        if activation == "relu2":
            assert "mlp_activation" not in metadata["model_config"]
            assert "activation_stats" not in metadata
            assert activation_keys == []
        else:
            assert metadata["model_config"]["mlp_activation"] == "xiprelu"
            coefficients = metadata["activation_stats"]
            assert len(coefficients) == 4
            assert len(activation_keys) == 4
            for layer in range(2):
                for branch in ("p", "n"):
                    weight_key = f"transformer.h.{layer}.mlp.act.raw_alpha_{branch}"
                    coefficient_key = f"xiprelu/layer_{layer}/alpha_{branch}"
                    effective = torch.nn.functional.softplus(weights[weight_key]).item()
                    assert coefficients[coefficient_key] == pytest.approx(effective)
            # Both branches in both layers must learn, beyond their 0.8 initialization.
            assert all(abs(value - 0.8) > 1e-7 for value in coefficients.values())
        if initial_weights is not None:
            assert not torch.equal(weights["lm_head.weight"], initial_weights["lm_head.weight"])
            if activation == "xiprelu":
                for name in weights:
                    if "raw_alpha" in name:
                        assert not torch.equal(weights[name], initial_weights[name]), name
        history[step] = metadata
        if step == 3:
            validation = re.findall(r"Step (\d+) \| Validation bpb: ([0-9.eE+-]+)", result.stdout)
            assert validation and int(validation[0][0]) == 0, result.stdout
            first_bpb = float(validation[0][1])
            report(capsys, "  PASS: compiled forward/backward and optimizer updates; all weights finite.",
                   f"  PASS: checkpoint saved at step 3; validation BPB = {metadata['val_bpb']:.6f}.",
                   "  Resuming saved model and optimizer for step 4...")
        else:
            report(capsys, "  PASS: checkpoint and optimizer loaded; resumed training changes weights.")
        initial_weights = weights

    report(capsys, f"  Parameters: {sum(value.numel() for value in weights.values()):,}; learned activation scalars: {len(activation_keys)}.",
           f"  Validation BPB: start {first_bpb:.6f} -> step 3 {history[3]['val_bpb']:.6f} -> step 4 {history[4]['val_bpb']:.6f}")
    if activation == "xiprelu":
        rows = ["  Learned coefficients (positive/negative branches, two per layer):",
                "    layer  branch       initial       step 3       resumed step 4   total change"]
        for layer in range(2):
            for branch, branch_label in (("p", "positive"), ("n", "negative")):
                key = f"xiprelu/layer_{layer}/alpha_{branch}"
                before_resume = history[3]["activation_stats"][key]
                after_resume = history[4]["activation_stats"][key]
                rows.append(f"    {layer:5d}  {branch_label:8s}   0.80000000   {before_resume:.8f}   {after_resume:.8f}       {after_resume - 0.8:+.2e}")
        report(capsys, *rows, "  PASS: all four coefficients learned and continued updating after resume.")
    report(capsys, f"  PASS: {label} local execution checks passed.",
           "  These tiny synthetic-data losses check execution; they do not establish which ablation is better.")
