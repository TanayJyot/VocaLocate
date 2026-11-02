"""Offline Diff V2 correctness and regression checks; CUDA checks skip without hardware."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nanochat import checkpoint_manager
import nanochat.flash_attention as fa
import nanochat.gpt as gpt
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig, Linear


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)


def make_model(diff_v2=True, activation="relu2", active=False, device="cpu", **kwargs):
    config = GPTConfig(sequence_len=16, vocab_size=256, n_layer=2, n_head=4,
                       n_kv_head=2, n_embd=32, window_pattern="L",
                       mlp_activation=activation, diff_v2=diff_v2, **kwargs)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    if active:
        # Exercise real attention, rather than hiding errors behind zero output projections.
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
    return model


@pytest.fixture
def previous_gpt(monkeypatch):
    # Pin the local baseline + xIPReLU implementation immediately before this change.
    result = subprocess.run(
        ["git", "show", "2aece7fd74449ead75399995bb8352beb53ffb95:./nanochat/gpt.py"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    if result.returncode:
        pytest.skip("Pre-Diff-V2 Git reference unavailable in this clone")
    module = ModuleType("_pre_diff_v2_gpt")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(result.stdout, "pre_diff_v2/gpt.py", "exec"), module.__dict__)
    module.COMPUTE_DTYPE = torch.float32
    return module


@pytest.mark.parametrize("activation", ["relu2", "xiprelu"])
def test_disabled_matches_previous_initialization_forward_gradients_and_update(previous_gpt, activation):
    config = dict(sequence_len=16, vocab_size=256, n_layer=2, n_head=4,
                  n_kv_head=2, n_embd=32, window_pattern="L", mlp_activation=activation)
    models = []
    for module in (gpt, previous_gpt):
        with torch.device("meta"):
            model = module.GPT(module.GPTConfig(**config))
        model.to_empty(device="cpu")
        torch.manual_seed(42)
        model.init_weights()
        models.append(model)
    current, previous = models
    assert current.state_dict().keys() == previous.state_dict().keys()
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, previous.state_dict()[name], rtol=0, atol=0)
    assert current.num_scaling_params() == previous.num_scaling_params()
    assert current.estimate_flops() == previous.estimate_flops()
    assert current.estimate_decode_flops(8) == previous.estimate_decode_flops(8)
    assert current.estimate_prefill_flops(8) == previous.estimate_prefill_flops(8)
    assert current.setup_optimizer().state_dict() == previous.setup_optimizer().state_dict()
    # Identical nonzero projections ensure this checks attention and activation arithmetic.
    with torch.no_grad():
        for a, b in zip(current.transformer.h, previous.transformer.h):
            for name in ("attn", "mlp"):
                weight = getattr(a, name).c_proj.weight
                weight.normal_(std=0.05)
                getattr(b, name).c_proj.weight.copy_(weight)
    tokens = torch.arange(8).view(1, 8)
    torch.testing.assert_close(current(tokens), previous(tokens), rtol=0, atol=0)
    for model in models:
        model(tokens, targets=tokens.roll(-1, 1)).backward()
    for a, b in zip(current.parameters(), previous.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    with torch.compiler.set_stance("force_eager"):
        for model in models:
            model.setup_optimizer().step()
    for a, b in zip(current.parameters(), previous.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("enabled", [False, True])
def test_picochat_projection_shapes_and_meta_initialization(enabled):
    with torch.device("meta"):
        model = GPT(GPTConfig(diff_v2=enabled))
    attn = model.transformer.h[0].attn
    assert attn.n_head == 6 and attn.head_dim == 128
    assert attn.c_q.weight.shape == (1536 if enabled else 768, 768)
    assert attn.c_k.weight.shape == attn.c_v.weight.shape == (768, 768)
    assert attn.c_proj.weight.shape == (768, 768)
    if enabled:
        assert isinstance(attn.lambda_proj, Linear)
        assert attn.lambda_proj.weight.shape == (6, 768)
    else:
        assert attn.lambda_proj is None
        assert not any("lambda_proj" in key for key in model.state_dict())
    small = make_model(diff_v2=enabled)
    assert all(torch.isfinite(p).all() for p in small.parameters())
    if enabled:
        for block in small.transformer.h:
            lam = block.attn.lambda_proj(torch.randn(1, 4, 32)).sigmoid()
            torch.testing.assert_close(lam, torch.full_like(lam, 0.5), rtol=0, atol=0)


@pytest.mark.parametrize("heads,kv_heads", [(4, 1), (4, 2), (4, 4), (6, 6)])
def test_adjacent_shared_kv_pairing_and_learned_subtraction(monkeypatch, heads, kv_heads):
    torch.manual_seed(42)
    model = make_model()
    width, q_heads = heads * 8, heads * 2
    attn = gpt.CausalSelfAttention(GPTConfig(n_embd=width, n_head=heads, n_kv_head=kv_heads,
                                           diff_v2=True), layer_idx=0)
    with torch.no_grad():
        attn.c_proj.weight.copy_(torch.eye(width))
        attn.lambda_proj.weight.normal_(std=0.3)  # test nonconstant, token-dependent lambda
    captured = {}
    real_attention = fa.flash_attn_func

    def capture(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v)
        return real_attention(q, k, v, **kwargs)

    monkeypatch.setattr(gpt.flash_attn, "flash_attn_func", capture)
    x = torch.randn(1, 4, width)
    actual = attn(x, None, (model.cos[:, :4], model.sin[:, :4]), (-1, 0), None, 0)
    q, k, v = (captured[name].transpose(1, 2) for name in ("q", "k", "v"))
    assert q.shape == (1, q_heads, 4, 8)
    assert k.shape == v.shape == (1, kv_heads, 4, 8)
    # Independent per-head attention proves the backend's contiguous GQA mapping.
    outputs = [F.scaled_dot_product_attention(
        q[:, h:h+1], k[:, h // (q_heads // kv_heads):h // (q_heads // kv_heads) + 1],
        v[:, h // (q_heads // kv_heads):h // (q_heads // kv_heads) + 1], is_causal=True)
        for h in range(q_heads)]
    lam = attn.lambda_proj(x).sigmoid()
    expected = torch.cat([
        outputs[2*h] - lam[:, None, :, h, None] * outputs[2*h+1]
        for h in range(heads)], dim=1).transpose(1, 2).reshape(1, 4, width)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_projection_and_subtraction_dtype_finite_outputs_and_gradients(monkeypatch, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(active=True)
    attn = model.transformer.h[0].attn
    x = torch.randn(1, 4, 32, dtype=dtype, requires_grad=True)
    output = attn(x, None, (model.cos[:, :4], model.sin[:, :4]), (-1, 0), None, 0)
    assert output.shape == x.shape and output.dtype == dtype
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for layer in (attn.c_q, attn.lambda_proj):
        assert layer.weight.dtype == torch.float32
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("activation", ["relu2", "xiprelu"])
def test_full_model_causality_and_two_real_optimizer_updates(activation):
    torch.manual_seed(42)
    model = make_model(activation=activation)
    optimizer = model.setup_optimizer()
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped})
    assert {id(p) for p in grouped} == {id(p) for p in model.parameters()}
    tokens = torch.arange(8).view(1, 8)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, targets=tokens.roll(-1, 1))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for block in model.transformer.h:
            for layer in (block.attn.c_q, block.attn.lambda_proj):
                if step == 0:
                    assert layer.weight.grad.count_nonzero() == 0
                else:
                    assert layer.weight.grad.abs().sum() > 0
        with torch.compiler.set_stance("force_eager"):
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    changed = tokens.clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4], rtol=0, atol=0)


@pytest.mark.parametrize("window", [-1, 2])
@pytest.mark.parametrize("activation", ["relu2", "xiprelu"])
@pytest.mark.parametrize("n_loops", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prefill_and_cached_decoding_match_full_sequence(monkeypatch, window, activation, n_loops, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(activation=activation, active=True, n_loops=n_loops).eval()
    # A tiny explicit window exercises truncation with only eight tokens.
    model.window_sizes = [(window, 0)] * model.eff_depth
    cache = KVCache(1, 2, 16, 8, model.eff_depth, "cpu", dtype)
    tokens = torch.arange(8).view(1, 8)
    with torch.no_grad():
        expected = model(tokens)
        prefill = model(tokens[:, :3], kv_cache=cache)
        decoded = [model(tokens[:, i:i+1], kv_cache=cache) for i in range(3, 8)]
    assert cache.get_pos() == 8 and cache.n_heads == 2
    rtol, atol = (0.03, 0.003) if dtype == torch.bfloat16 else (1e-4, 1e-6)
    torch.testing.assert_close(torch.cat([prefill, *decoded], 1), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("activation", ["relu2", "xiprelu"])
def test_strict_checkpoint_roundtrip_and_old_config_default(tmp_path, monkeypatch, enabled, activation):
    model = make_model(diff_v2=enabled, activation=activation, active=True).eval()
    config = asdict(model.config)
    if not enabled:
        config.pop("diff_v2")  # existing checkpoints don't contain the new flag
    checkpoint_manager.save_checkpoint(tmp_path, 0, model.state_dict(), None, dict(model_config=config))
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer",
                        lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, _ = checkpoint_manager.build_model(tmp_path, 0, torch.device("cpu"), "eval")
    assert restored.config.diff_v2 == enabled
    tokens = torch.arange(8).view(1, 8)
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="size mismatch"):
        make_model(diff_v2=not enabled, activation=activation).load_state_dict(model.state_dict(), strict=True)


@pytest.mark.parametrize("n_loops", [1, 2])
def test_flops_count_extra_projections_once_and_double_attention_only(n_loops):
    base = make_model(diff_v2=False, n_loops=n_loops)
    diff = make_model(n_loops=n_loops)
    # Extra Q weights plus the small lambda projection, per unique block.
    extra = 32 * 32 + 32 * 4
    assert diff.num_scaling_params()["total"] - base.num_scaling_params()["total"] == 2 * extra
    delta_matmul = extra * base.eff_depth
    assert diff.num_matmul_params() - base.num_matmul_params() == delta_matmul
    assert diff.estimate_flops() - base.estimate_flops() == 6 * delta_matmul + 12 * 32 * 16 * base.eff_depth
    assert diff.estimate_decode_flops(8) - base.estimate_decode_flops(8) == 2 * delta_matmul + 4 * 32 * 8 * base.eff_depth
    assert diff.estimate_prefill_flops(8) - base.estimate_prefill_flops(8) == 2 * delta_matmul * 8 + 4 * 32 * 36 * base.eff_depth
    assert diff.kv_bytes_per_token() == base.kv_bytes_per_token()
    assert diff.kv_read_bytes(8) == base.kv_read_bytes(8)


@pytest.mark.parametrize("activation", ["relu2", "xiprelu"])
def test_fullgraph_compile_forward_and_backward(activation):
    model = make_model(activation=activation, active=True)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    tokens = torch.arange(8).view(1, 8)
    torch.testing.assert_close(compiled(tokens), model(tokens), rtol=0, atol=0)
    compiled(tokens, targets=tokens.roll(-1, 1)).backward()
    for block in model.transformer.h:
        assert block.attn.lambda_proj.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                    reason="CUDA BF16 hardware required")
@pytest.mark.parametrize("backend", ["sdpa", "fa3"])
@pytest.mark.parametrize("window", [-1, 2])
def test_cuda_bf16_training_cache_and_backend_agreement(monkeypatch, backend, window):
    if backend == "fa3" and not fa.HAS_FA3:
        pytest.skip("FA3 kernels unavailable")
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.bfloat16)
    monkeypatch.setattr(fa, "USE_FA3", backend == "fa3")
    model = make_model(active=True, device="cuda")
    model.window_sizes = [(window, 0)] * model.eff_depth
    tokens = torch.arange(8, device="cuda").view(1, 8)
    loss = model(tokens, targets=tokens.roll(-1, 1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    cache = KVCache(1, 2, 16, 8, model.eff_depth, "cuda", torch.bfloat16)
    with torch.no_grad():
        expected = model(tokens)
        actual = torch.cat([model(tokens[:, :3], kv_cache=cache)] +
                           [model(tokens[:, i:i+1], kv_cache=cache) for i in range(3, 8)], 1)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
        if backend == "fa3":
            monkeypatch.setattr(fa, "USE_FA3", False)
            torch.testing.assert_close(expected, model(tokens), rtol=0.03, atol=0.003)
