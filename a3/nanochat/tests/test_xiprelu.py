"""CPU checks for the activation, optimizer integration and checkpoint lifecycle."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import types

import pytest
import torch
import torch.nn.functional as F

from nanochat import checkpoint_manager
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, MLP, XIPReLU


def tiny_model(activation="xiprelu", **loop_config):
    config = GPTConfig(sequence_len=16, vocab_size=256, n_layer=2, n_head=4,
                       n_kv_head=2, n_embd=32, window_pattern="L",
                       mlp_activation=activation, **loop_config)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_piecewise_values_and_input_derivatives():
    activation = XIPReLU()
    with torch.no_grad():
        activation.raw_alpha_n.copy_(torch.tensor(0.3).expm1().log())
    x = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    y = activation(x)
    torch.testing.assert_close(y, torch.tensor([0.2, -0.2, 0., 1.3, 4.2]))
    y.sum().backward()
    # At zero both branches meet with derivative 0.5; negative slopes are allowed.
    torch.testing.assert_close(x.grad, torch.tensor([-0.7, -0.1, 0.5, 2.1, 3.7]))


def test_autograd_matches_numerical_gradients_for_inputs_and_coefficients():
    activation = XIPReLU().double()
    x = torch.tensor([-1.7, -0.1, 0., 0.3, 2.], dtype=torch.float64, requires_grad=True)
    def evaluate(x, raw_p, raw_n):
        return torch.func.functional_call(activation,
            {"raw_alpha_p": raw_p, "raw_alpha_n": raw_n}, (x,))
    assert torch.autograd.gradcheck(evaluate,
        (x, activation.raw_alpha_p, activation.raw_alpha_n))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_compute_dtype_shape_and_finite_gradients(dtype):
    activation = XIPReLU()
    x = torch.linspace(-10, 10, 48, dtype=dtype).reshape(2, 3, 8).requires_grad_()
    output = activation(x)
    assert output.shape == x.shape and output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().mean().backward()
    assert torch.isfinite(x.grad).all()
    for p in activation.parameters():
        assert p.dtype == torch.float32
        assert p.grad is not None and torch.isfinite(p.grad)


def test_activation_compiles_with_autograd():
    activation = XIPReLU()
    # Exercise graph capture and AOT autograd locally, without CUDA or native compilation.
    compiled = torch.compile(activation, backend="aot_eager", fullgraph=True)
    x = torch.tensor([-2., -0.1, 0., 1.], requires_grad=True)
    torch.testing.assert_close(compiled(x), activation(x))
    compiled(x).sum().backward()
    assert activation.raw_alpha_p.grad.abs() > 0
    assert activation.raw_alpha_n.grad.abs() > 0


@pytest.mark.parametrize("n_loops", [1, 3])
def test_meta_initialization_parameter_budget_and_optimizer_groups(n_loops):
    model = tiny_model(n_loops=n_loops)
    assert len(model.activation_stats()) == 4
    assert all(value == pytest.approx(0.8) for value in model.activation_stats().values())
    counts = []
    for activation in ("relu2", "xiprelu"):
        with torch.device("meta"):
            twelve_layers = GPT(GPTConfig(n_layer=12, mlp_activation=activation))
        counts.append(twelve_layers.num_scaling_params())
    assert counts[1]["total"] - counts[0]["total"] == 24
    assert counts[1]["scalars"] - counts[0]["scalars"] == 24
    assert counts[1]["transformer_matrices"] == counts[0]["transformer_matrices"]

    optimizer = model.setup_optimizer(scalar_lr=0.5)
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    for block in model.transformer.h:
        for p in block.mlp.act.parameters():
            group, = [g for g in optimizer.param_groups if any(p is q for q in g["params"])]
            assert group["kind"] == "adamw" and group["weight_decay"] == 0
            assert group["lr"] == pytest.approx(0.005)
    assert all(p.ndim == 2 for g in optimizer.param_groups if g["kind"] == "muon" for p in g["params"])


@pytest.mark.parametrize("n_loops", [1, 2])
def test_training_checkpoint_load_and_optimizer_resume(tmp_path, monkeypatch, n_loops):
    torch.manual_seed(42)
    model = tiny_model(n_loops=n_loops)
    optimizer = model.setup_optimizer()
    tokens = torch.arange(16).reshape(2, 8)
    targets = tokens.roll(-1, dims=1)
    initial = model.activation_stats()

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, targets=targets)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        # Run the actual Muon/AdamW update math without compiling CPU kernels.
        with torch.compiler.set_stance("force_eager"):
            optimizer.step()

    # Zero-initialized output projections block activation gradients on step 1.
    # Subsequent updates must learn both branch coefficients in every layer.
    for _ in range(3):
        update(model, optimizer)
    assert all(model.activation_stats()[key] != value for key, value in initial.items())

    metadata = dict(model_config=asdict(model.config), activation_stats=model.activation_stats())
    checkpoint_manager.save_checkpoint(tmp_path, 3, model.state_dict(), optimizer.state_dict(), metadata)
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer",
                        lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, saved = checkpoint_manager.build_model(tmp_path, 3, torch.device("cpu"), "eval")
    assert saved == json.loads(json.dumps(metadata))
    assert restored.config.mlp_activation == "xiprelu"
    assert restored.activation_stats() == model.activation_stats()
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    # CPU inference deliberately upcasts BF16 checkpoint embeddings. Training
    # resume uses load_checkpoint directly, preserving parameter/state dtypes.
    assert restored.config.n_loops == n_loops
    restored = tiny_model(n_loops=n_loops)
    weights, state, _ = checkpoint_manager.load_checkpoint(tmp_path, 3, torch.device("cpu"), load_optimizer=True)
    restored.load_state_dict(weights, strict=True, assign=True)
    restored_optimizer = restored.setup_optimizer()
    restored_optimizer.load_state_dict(state)
    update(model, optimizer)
    update(restored, restored_optimizer)
    for p, restored_p in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, restored_p, rtol=0, atol=0)


def test_default_mlp_is_relu_squared_and_rejects_unknown_activations():
    mlp = MLP(GPTConfig(n_embd=32))
    x = torch.randn(2, 3, 32)
    assert mlp.act is None
    torch.testing.assert_close(mlp(x), mlp.c_proj(F.relu(mlp.c_fc(x)).square()), rtol=0, atol=0)
    assert not any("alpha" in key for key in mlp.state_dict())
    with pytest.raises(ValueError, match="Unsupported MLP activation"):
        MLP(GPTConfig(mlp_activation="typo"))


def test_existing_baseline_checkpoint_loads_without_activation_config(tmp_path, monkeypatch):
    model = tiny_model("relu2").eval()
    config = asdict(model.config)
    config.pop("mlp_activation")
    checkpoint_manager.save_checkpoint(tmp_path, 0, model.state_dict(), None,
                                       dict(model_config=config))
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer",
                        lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, _ = checkpoint_manager.build_model(tmp_path, 0, torch.device("cpu"), "eval")
    assert restored.config.mlp_activation == "relu2"
    assert restored.activation_stats() == {}
    tokens = torch.arange(16).reshape(2, 8)
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("active_projections", [False, True])
def test_default_baseline_matches_original_model(monkeypatch, active_projections):
    """Preserve baseline initialization, gradients, counts and optimizer updates."""
    root = Path(__file__).resolve().parents[1]
    original = subprocess.run(["git", "show", "92d63d4:./nanochat/gpt.py"],
                              cwd=root, capture_output=True, encoding="utf-8")
    if original.returncode != 0:
        # Vendored copies may retain only the commit that first added nanochat.
        added = subprocess.check_output(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", "nanochat/gpt.py"],
            cwd=root, encoding="utf-8").split()
        if not added:
            pytest.skip("Original GPT baseline is unavailable in this clone")
        original = subprocess.run(["git", "show", f"{added[-1]}:./nanochat/gpt.py"],
                                  cwd=root, capture_output=True, encoding="utf-8")
        original.check_returncode()
    source = original.stdout
    reference = types.ModuleType("_xiprelu_baseline_reference")
    monkeypatch.setitem(sys.modules, reference.__name__, reference)
    exec(compile(source, "original/gpt.py", "exec"), reference.__dict__)
    config = dict(sequence_len=16, vocab_size=256, n_layer=2, n_head=4,
                  n_kv_head=2, n_embd=32, window_pattern="L")
    models = []
    for cls, config_cls in ((GPT, GPTConfig), (reference.GPT, reference.GPTConfig)):
        with torch.device("meta"):
            model = cls(config_cls(**config))
        model.to_empty(device="cpu")
        torch.manual_seed(42)
        model.init_weights()
        models.append(model)
    current, original = models
    assert current.activation_stats() == {}
    assert current.state_dict().keys() == original.state_dict().keys()
    for key, tensor in current.state_dict().items():
        torch.testing.assert_close(tensor, original.state_dict()[key], rtol=0, atol=0)
    assert current.num_scaling_params() == original.num_scaling_params()
    assert current.estimate_flops() == original.estimate_flops()
    assert current.setup_optimizer().state_dict() == original.setup_optimizer().state_dict()
    if active_projections:
        with torch.no_grad():
            for block, original_block in zip(current.transformer.h, original.transformer.h):
                for name in ("attn", "mlp"):
                    weight = getattr(block, name).c_proj.weight
                    weight.normal_(std=0.05)
                    getattr(original_block, name).c_proj.weight.copy_(weight)
    tokens = torch.arange(16).reshape(2, 8)
    torch.testing.assert_close(current(tokens), original(tokens), rtol=0, atol=0)
    for model in models:
        model(tokens, targets=tokens.roll(-1, dims=1)).backward()
    for param, original_param in zip(current.parameters(), original.parameters()):
        torch.testing.assert_close(param.grad, original_param.grad, rtol=0, atol=0)
    with torch.compiler.set_stance("force_eager"):
        for model in models:
            model.setup_optimizer().step()
    for param, original_param in zip(current.parameters(), original.parameters()):
        torch.testing.assert_close(param, original_param, rtol=0, atol=0)


@pytest.mark.parametrize("activation", ["relu2", "xiprelu"])
@pytest.mark.parametrize("n_loops", [1, 2])
def test_cached_decoding_matches_full_forward(activation, n_loops):
    torch.manual_seed(42)
    model = tiny_model(activation, n_loops=n_loops).eval()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
    config = model.config
    cache = KVCache(batch_size=2, num_heads=config.n_kv_head,
                    seq_len=config.sequence_len, head_dim=config.n_embd // config.n_head,
                    num_layers=config.effective_depth, device="cpu", dtype=COMPUTE_DTYPE)
    tokens = torch.arange(16).reshape(2, 8)
    with torch.no_grad():
        expected = model(tokens)
        prefill = model(tokens[:, :3], kv_cache=cache)
        decoded = [model(tokens[:, i:i+1], kv_cache=cache) for i in range(3, 8)]
    actual = torch.cat([prefill, *decoded], dim=1)
    assert cache.get_pos() == tokens.size(1)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
