"""MGVR arithmetic, isolation, training, cache and checkpoint acceptance tests."""
from dataclasses import asdict
from itertools import product
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nanochat import checkpoint_manager
import nanochat.flash_attention as fa
import nanochat.gpt as gpt
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, MGVR


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    torch._dynamo.reset() # Each parameterized model needs fresh fullgraph specialization slots.
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)
    yield
    torch._dynamo.reset()


def make_model(module=gpt, active=False, device="cpu", **overrides):
    options = dict(sequence_len=16, vocab_size=256, n_layer=3, n_head=4,
                   n_kv_head=2, n_embd=32, window_pattern="SSSL")
    options.update(overrides)
    torch.manual_seed(42)
    with torch.device("meta"):
        model = module.GPT(module.GPTConfig(**options))
    model.to_empty(device=device)
    model.init_weights()
    if active:
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
    return model


def mgvr_parameters(model):
    return [p for name, p in model.named_parameters() if ".mgvr." in name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads", [1, 2, 6])
def test_initial_identity_formula_cross_head_mixing_and_gradients(dtype, heads):
    module = MGVR(32, heads)
    torch.nn.init.zeros_(module.w_z)
    torch.nn.init.zeros_(module.lam)
    x = torch.randn(2, 3, 32, dtype=dtype, requires_grad=True)
    v, v0 = [torch.randn(2, 3, heads, 8, dtype=dtype, requires_grad=True) for _ in range(2)]
    assert torch.equal(module(x, v, v0), v)
    with torch.no_grad():
        module.lam.copy_(0.5 * torch.eye(heads))
    torch.testing.assert_close(module(x, v, v0), (v.float() + 0.5 * v0.float()).to(dtype), rtol=0, atol=0)
    with torch.no_grad():
        module.w_z.normal_(std=0.3)
        module.lam.normal_(std=0.2)
    actual = module(x, v, v0)
    z = x.float() @ module.w_z.T / 32**0.5
    expected = torch.empty_like(v)
    for b, t, g in product(range(2), range(3), range(heads)):
        added = sum((module.lam[g, h] + z[b, t, heads + g * heads + h].tanh()) * v0[b, t, h].float()
                    for h in range(heads))
        expected[b, t, g] = 2 * z[b, t, g].sigmoid() * v[b, t, g].float() + added
    torch.testing.assert_close(actual, expected, rtol=0.02 if dtype == torch.bfloat16 else 1e-5, atol=1e-6)
    actual.float().square().sum().backward()
    for p in (x, v, v0, module.w_z, module.lam):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert module.w_z.dtype == module.lam.dtype == torch.float32


def test_offdiagonal_static_and_dynamic_mixing():
    module = MGVR(2, 2)
    torch.nn.init.zeros_(module.w_z)
    torch.nn.init.zeros_(module.lam)
    x, v = torch.tensor([[[1., 0.]]]), torch.zeros(1, 1, 2, 1)
    v0 = torch.tensor([[[[3.], [7.]]]])
    with torch.no_grad():
        module.lam[0, 1] = 1
    assert torch.equal(module(x, v, v0), torch.tensor([[[[7.], [0.]]]]))
    with torch.no_grad():
        module.lam.zero_()
        module.w_z[3, 0] = torch.atanh(torch.tensor(0.5)) * 2**0.5
    torch.testing.assert_close(module(x, v, v0), torch.tensor([[[[3.5], [0.]]]]))


def test_first_layer_anchor_keeps_projection_and_ve_autograd():
    model, anchors = make_model(mgvr=True), []
    handle = model.transformer.h[1].attn.mgvr.register_forward_pre_hook(lambda m, inputs: anchors.append(inputs[2]))
    model(torch.arange(8).view(1, 8))
    handle.remove()
    gradients = torch.autograd.grad(anchors[0].float().square().sum(),
                                   (model.transformer.h[0].attn.c_v.weight, model.value_embeds["0"].weight))
    assert all(torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in gradients)


@pytest.fixture
def previous_gpt(monkeypatch):
    result = subprocess.run(["git", "show", "d854e74:./nanochat/gpt.py"],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    if result.returncode:
        pytest.skip("Pre-MGVR Git reference unavailable")
    module = ModuleType("_pre_mgvr_gpt")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(result.stdout, "pre_mgvr/gpt.py", "exec"), module.__dict__)
    module.COMPUTE_DTYPE = torch.float32
    return module


@pytest.mark.parametrize("activation,diff,depth,loops", list(product(("relu2", "xiprelu"), (False, True), (False, True), (1, 2))))
def test_disabled_preserves_every_prior_ablation(previous_gpt, monkeypatch, activation, diff, depth, loops):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled MGVR constructed or executed")
    monkeypatch.setattr(MGVR, "__init__", forbidden)
    monkeypatch.setattr(MGVR, "forward", forbidden)
    options = dict(mlp_activation=activation, diff_v2=diff, depth_attention=depth, n_loops=loops)
    current, previous = [make_model(module, active=True, **options) for module in (gpt, previous_gpt)]
    assert current.state_dict().keys() == previous.state_dict().keys()
    assert not mgvr_parameters(current)
    for name, value in current.state_dict().items():
        assert torch.equal(value, previous.state_dict()[name])
    current.load_state_dict(previous.state_dict(), strict=True)
    assert current.num_scaling_params() == previous.num_scaling_params()
    assert current.estimate_flops() == previous.estimate_flops()
    assert current.estimate_decode_flops(8) == previous.estimate_decode_flops(8)
    assert current.estimate_prefill_flops(8) == previous.estimate_prefill_flops(8)
    assert current.setup_optimizer().state_dict() == previous.setup_optimizer().state_dict()
    tokens = torch.arange(8).view(1, 8)
    assert torch.equal(current(tokens), previous(tokens))
    for model in (current, previous):
        model(tokens, targets=tokens.roll(-1, 1)).backward()
    for a, b in zip(current.parameters(), previous.parameters()):
        assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("combined", [False, True])
def test_initial_model_values_logits_loss_and_shared_weights(monkeypatch, dtype, combined):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    options = dict(mlp_activation="xiprelu", diff_v2=True, depth_attention=True, n_loops=2) if combined else {}
    baseline, model = [make_model(active=True, mgvr=enabled, **options) for enabled in (False, True)]
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[name]), name
    values = []
    original = fa.flash_attn.flash_attn_func
    def capture(q, k, v, **kwargs):
        values.append(v.detach().clone())
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(fa.flash_attn, "flash_attn_func", capture)
    tokens = torch.arange(8).view(1, 8)
    assert torch.equal(baseline(tokens), model(tokens))
    depth = model.eff_depth
    assert all(torch.equal(a, b) for a, b in zip(values[:depth], values[depth:]))
    assert torch.equal(baseline(tokens, targets=tokens), model(tokens, targets=tokens))
    # n_layer=3 includes layer-0 VE; the anchor must include its gated contribution.
    anchor = []
    handle = model.transformer.h[1].attn.mgvr.register_forward_pre_hook(lambda m, inputs: anchor.append(inputs[2]))
    model(tokens)
    handle.remove()
    assert torch.equal(anchor[0], values[depth])
    assert all(v is anchor[0] for v in anchor)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_optimizer_membership_learning_causality_and_diagnostics(monkeypatch, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(mgvr=True)
    optimizer = model.setup_optimizer()
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == len(list(model.parameters()))
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    ids = {id(p) for p in mgvr_parameters(model)}
    for group in optimizer.param_groups:
        if any(id(p) in ids for p in group["params"]):
            assert group["kind"] == "adamw" and group["weight_decay"] == 0
            assert all(id(p) in ids for p in group["params"])
    tokens = torch.arange(8).view(1, 8)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, targets=tokens.roll(-1, 1))
        loss.backward()
        assert torch.isfinite(loss)
        for p in mgvr_parameters(model):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert (p.grad.count_nonzero() == 0) if step == 0 else (p.grad.abs().sum() > 0)
        with torch.compiler.set_stance("force_eager"):
            optimizer.step()
    assert all(p.abs().sum() > 0 for p in mgvr_parameters(model))
    assert all(b.attn.c_v.weight.grad.abs().sum() > 0 for b in model.transformer.h)
    changed = tokens.clone()
    changed[:, 4:] += 100
    assert torch.equal(model(tokens)[:, :4], model(changed)[:, :4])
    stats = model.mgvr_stats(tokens)
    assert 0 < stats["gate_mean"] < 2
    assert stats["added_to_current_norm"] > 0 and stats["offdiagonal_mix_norm"] > 0
    assert all(not b.attn.mgvr._forward_hooks for b in model.transformer.h if b.attn.mgvr is not None)


@pytest.mark.parametrize("combined,window,dtype", list(product((False, True), ("L", "SSSL"), (torch.float32, torch.bfloat16))))
def test_active_mgvr_prefill_decode_and_fullgraph(monkeypatch, combined, window, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    options = dict(mlp_activation="xiprelu", diff_v2=True, depth_attention=True,
                   n_loops=2, n_prelude=1, n_coda=1) if combined else {}
    model = make_model(active=True, mgvr=True, window_pattern=window, **options)
    if window == "SSSL":
        model.window_sizes = [(2, 0)] * (model.eff_depth - 1) + [(16, 0)] # exercise actual truncation
    with torch.no_grad():
        for p in mgvr_parameters(model):
            p.normal_(std=0.1)
    cache = KVCache(1, 2, 16, 8, model.eff_depth, "cpu", dtype)
    tokens = torch.arange(8).view(1, 8)
    with torch.no_grad():
        expected = model(tokens)
        actual = torch.cat([model(tokens[:, :3], kv_cache=cache)] +
                           [model(tokens[:, i:i+1], kv_cache=cache) for i in range(3, 8)], 1)
    torch.testing.assert_close(actual, expected, rtol=0.03 if dtype == torch.bfloat16 else 1e-4, atol=0.003 if dtype == torch.bfloat16 else 1e-6)
    assert cache.get_pos() == 8 and cache.n_heads == 2
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(tokens), expected, rtol=0, atol=0)
    compiled(tokens, targets=tokens).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("enabled", [False, True])
def test_strict_checkpoint_roundtrip_and_flops(tmp_path, monkeypatch, enabled):
    model = make_model(active=True, mgvr=enabled, n_loops=2)
    config = asdict(model.config)
    if not enabled:
        config.pop("mgvr")
    checkpoint_manager.save_checkpoint(tmp_path, 0, model.state_dict(), None, dict(model_config=config))
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, _ = checkpoint_manager.build_model(tmp_path, 0, torch.device("cpu"), "eval")
    assert restored.config.mgvr == enabled
    tokens = torch.arange(8).view(1, 8)
    assert torch.equal(restored(tokens), model(tokens))
    baseline = make_model(n_loops=2)
    assert model.kv_bytes_per_token() == baseline.kv_bytes_per_token()
    assert model.num_scaling_params()["transformer_matrices"] == baseline.num_scaling_params()["transformer_matrices"]
    delta = model.num_matmul_params() - baseline.num_matmul_params()
    extra = model.estimate_mgvr_flops()
    assert model.estimate_flops() - baseline.estimate_flops() == 6 * delta + 3 * extra
    assert model.estimate_decode_flops(8) - baseline.estimate_decode_flops(8) == 2 * delta + extra
    assert model.estimate_prefill_flops(8) - baseline.estimate_prefill_flops(8) == 8 * (2 * delta + extra)


def test_depth12_parameter_budget_and_meta_initialization():
    with torch.device("meta"):
        baseline, model = [GPT(GPTConfig(mgvr=enabled)) for enabled in (False, True)]
    assert model.num_scaling_params()["total"] - baseline.num_scaling_params()["total"] == 355_212
    assert model.transformer.h[0].attn.mgvr is None
    assert model.transformer.h[1].attn.mgvr.w_z.shape == (42, 768)
    assert model.transformer.h[1].attn.mgvr.lam.shape == (6, 6)


@pytest.mark.parametrize("script,folder", [("chat_sft.py", "chatsft_checkpoints"), ("chat_rl.py", "chatrl_checkpoints")])
@pytest.mark.parametrize("enabled", [False, True])
def test_finetuning_save_blocks_preserve_flag_and_directory(tmp_path, monkeypatch, script, folder, enabled):
    source = Path(__file__).resolve().parents[2] / "scripts" / script
    block = next(node for node in ast.walk(ast.parse(source.read_text())) if isinstance(node, ast.If) and any(
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "save_checkpoint" for stmt in node.body))
    model = make_model(mgvr=enabled)
    namespace = dict(model=model, orig_model=model, depth=3, step=1, num_steps=2,
                     args=SimpleNamespace(model_tag=None, max_seq_len=16), base_dir=str(tmp_path),
                     get_base_dir=lambda: str(tmp_path), os=os, val_bpb=2., user_config={}, ddp_rank=0,
                     optimizer=SimpleNamespace(state_dict=lambda: {}),
                     tokenizer=SimpleNamespace(get_vocab_size=lambda: 256), save_checkpoint=checkpoint_manager.save_checkpoint)
    exec(compile(ast.Module(body=block.body, type_ignores=[]), str(source), "exec"), namespace)
    directory = tmp_path / folder / ("d3-mgvr" if enabled else "d3")
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, metadata = checkpoint_manager.build_model(directory, 1, torch.device("cpu"), "eval")
    assert restored.config.mgvr == enabled
    assert ("mgvr" in metadata["model_config"]) == enabled
    restored.load_state_dict(model.state_dict(), strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
@pytest.mark.parametrize("backend", ["sdpa", "fa3"])
def test_cuda_bf16_smoke(monkeypatch, backend):
    if not torch.cuda.is_bf16_supported() or (backend == "fa3" and not fa.HAS_FA3):
        pytest.skip("BF16/FA3 unavailable")
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.bfloat16)
    monkeypatch.setattr(fa, "USE_FA3", backend == "fa3")
    model = make_model(mgvr=True, active=True, device="cuda")
    tokens = torch.arange(8, device="cuda").view(1, 8)
    model(tokens, targets=tokens).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
