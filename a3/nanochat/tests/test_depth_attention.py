"""Offline Depth-Attention acceptance tests; GPU-only checks skip explicitly."""
import ast
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nanochat import checkpoint_manager
import nanochat.flash_attention as fa
import nanochat.engine as engine_module
import nanochat.gpt as gpt
from nanochat.depth_attention import depth_source_layers, mix_depth_values
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)


def make_model(active=True, device="cpu", **overrides):
    options = dict(sequence_len=16, vocab_size=256, n_layer=3, n_head=4,
                   n_kv_head=2, n_embd=32, window_pattern="L", depth_attention=True)
    options.update(overrides)
    with torch.device("meta"):
        model = GPT(GPTConfig(**options))
    model.to_empty(device=device)
    model.init_weights()
    if active:
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
            # Chunk-boundary tests must exercise NanoChat's learned smear.
            model.smear_lambda.fill_(0.3)
    return model


def cache_for(model, batch=1, device="cpu"):
    c = model.config
    return KVCache(batch, c.n_kv_head, 16, c.n_embd // c.n_head,
                   model.eff_depth, device, gpt.COMPUTE_DTYPE)


def test_manual_weighted_average_and_first_layer_identity():
    q = torch.tensor([[[[2., 0.]]]])
    old_k = torch.tensor([[[[1., 0.]]]])
    k = torch.tensor([[[[0., 1.]]]])
    old_v, v = torch.tensor([[[[3., 7.]]]]), torch.tensor([[[[11., 13.]]]])
    alpha = torch.exp(torch.tensor(2. / 2.**0.5))
    expected = (alpha * old_v + v) / (alpha + 1)
    torch.testing.assert_close(mix_depth_values(q, k, v, [(old_k, old_v)]), expected)
    assert mix_depth_values(q, k, v, []) is v


def test_same_position_only_and_recursive_mixed_values():
    q, k = torch.zeros(1, 3, 1, 2), torch.zeros(1, 3, 1, 2)
    v0, v1, v2 = [torch.full_like(k, value) for value in (2., 6., 10.)]
    mixed1 = mix_depth_values(q, k, v1, [(k, v0)])
    mixed2 = mix_depth_values(q, k, v2, [(k, v0), (k, mixed1)])
    torch.testing.assert_close(mixed2, torch.full_like(k, 16. / 3))
    changed = mixed1.clone()
    changed[:, 2] = 1000
    actual = mix_depth_values(q, k, v2, [(k, v0), (k, changed)])
    torch.testing.assert_close(actual[:, :2], mixed2[:, :2], rtol=0, atol=0)


def test_exact_source_selection_and_stride_validation():
    expected = {0: (0,), 4: (0, 4), 6: (0, 6), 9: (0, 6, 9), 11: (0, 6, 11)}
    for layer, sources in expected.items():
        assert depth_source_layers(layer, 6) == sources
    assert GPTConfig(n_layer=12).effective_depth_attention_stride == 6
    assert GPTConfig(n_layer=1).effective_depth_attention_stride == 1
    for bad_stride in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            make_model(depth_attention_stride=bad_stride)


@pytest.mark.parametrize("batch,time,heads,kv_heads,dim", [(1, 1, 6, 6, 128), (2, 5, 8, 2, 4), (3, 2, 4, 1, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shapes_precision_gqa_and_gradients(batch, time, heads, kv_heads, dim, dtype):
    torch.manual_seed(3)
    q = torch.randn(batch, time, heads, dim, dtype=dtype, requires_grad=True)
    sources = [(torch.randn(batch, time, kv_heads, dim, dtype=dtype, requires_grad=True),
                torch.randn(batch, time, kv_heads, dim, dtype=dtype, requires_grad=True)) for _ in range(3)]
    k, v = sources[-1]
    actual = mix_depth_values(q, k, v, sources[:-1])
    # Independent scalar reference: explicitly select the KV-sharing query heads.
    expected = torch.empty_like(v)
    group = heads // kv_heads
    for b in range(batch):
        for t in range(time):
            for h in range(kv_heads):
                query = q[b, t, h*group:(h+1)*group].float().mean(0)
                scores = torch.stack([query.dot(key[b, t, h].float()) / dim**0.5 for key, _ in sources])
                expected[b, t, h] = sum(weight * value[b, t, h].float()
                                        for weight, (_, value) in zip(scores.softmax(0), sources))
    assert actual.shape == v.shape and actual.dtype == dtype
    torch.testing.assert_close(actual, expected)
    actual.float().square().mean().backward()
    for tensor in [q, *(tensor for pair in sources for tensor in pair)]:
        assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_training_retains_only_stride_sources_after_value_embedding_and_qk_norm(monkeypatch):
    model = make_model(n_layer=12, depth_attention_stride=6)
    original_mix = gpt.mix_depth_values
    seen, produced = [], []

    def capture(q, k, v, earlier):
        # K/Q RMS is the baseline's 1.2 scale, proving normalization came first.
        torch.testing.assert_close(k.float().square().mean(-1), torch.full_like(k[..., 0], 1.44), rtol=1e-3, atol=1e-3)
        seen.append(tuple(next(i for i, pair in enumerate(produced) if pair[0] is old_k) for old_k, _ in earlier))
        mixed = original_mix(q, k, v, earlier)
        produced.append((k, mixed))
        return mixed

    monkeypatch.setattr(gpt, "mix_depth_values", capture)
    model(torch.arange(8).view(1, 8))
    assert seen == [depth_source_layers(i, 6)[:-1] for i in range(12)]
    # Selected references are exactly the mixed values, never raw earlier V.
    assert all(pair[1].grad_fn is not None for pair in produced)


@pytest.fixture
def previous_gpt(monkeypatch):
    # Immutable repository state before Depth-Attention, including all local ablations.
    result = subprocess.run(["git", "show", "0fb10042b1ce1c1f07bf9b744e7466d0733049b6:./nanochat/gpt.py"],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    if result.returncode:
        pytest.skip("Pre-Depth-Attention Git reference unavailable")
    module = ModuleType("_pre_depth_attention_gpt")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    exec(compile(result.stdout, "pre_depth_attention/gpt.py", "exec"), module.__dict__)
    module.COMPUTE_DTYPE = torch.float32
    return module


@pytest.mark.parametrize("options", [{}, {"mlp_activation": "xiprelu"}, {"diff_v2": True},
                                     {"n_loops": 2, "n_prelude": 1, "n_coda": 1},
                                     {"mlp_activation": "xiprelu", "diff_v2": True, "n_loops": 2}])
def test_disabled_bitwise_matches_previous_model_and_optimizer(previous_gpt, options, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled path invoked depth mixing")
    monkeypatch.setattr(gpt, "mix_depth_values", forbidden)
    config = dict(sequence_len=16, vocab_size=256, n_layer=3, n_head=4,
                  n_kv_head=2, n_embd=32, window_pattern="SSSL", **options)
    models = []
    for module in (gpt, previous_gpt):
        torch.manual_seed(42)
        with torch.device("meta"):
            model = module.GPT(module.GPTConfig(**config))
        model.to_empty(device="cpu")
        model.init_weights()
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
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
    tokens = torch.arange(16).view(2, 8)
    torch.testing.assert_close(current(tokens), previous(tokens), rtol=0, atol=0)
    with torch.no_grad():
        current_cache, previous_cache = cache_for(current, batch=2), cache_for(previous, batch=2)
        for start, end in ((0, 3), (3, 4), (4, 6), (6, 8)):
            torch.testing.assert_close(current(tokens[:, start:end], kv_cache=current_cache),
                                       previous(tokens[:, start:end], kv_cache=previous_cache), rtol=0, atol=0)
        torch.testing.assert_close(current_cache.k_cache, previous_cache.k_cache, rtol=0, atol=0)
        torch.testing.assert_close(current_cache.v_cache, previous_cache.v_cache, rtol=0, atol=0)
    # Existing checkpoints must strictly load into the disabled architecture.
    current.load_state_dict(previous.state_dict(), strict=True)
    for model in models:
        model(tokens, targets=tokens.roll(-1, 1)).backward()
    for a, b in zip(current.parameters(), previous.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    with torch.compiler.set_stance("force_eager"):
        for model in models:
            model.setup_optimizer().step()
    for a, b in zip(current.parameters(), previous.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("options", [{}, {"diff_v2": True, "mlp_activation": "xiprelu", "n_loops": 2}])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_optimizer_steps_gradient_flow_and_causality(monkeypatch, options, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(active=False, **options)
    optimizer = model.setup_optimizer()
    tokens = torch.arange(16).view(2, 8)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(tokens, targets=tokens.roll(-1, 1))
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if step > 0:
            for block in model.transformer.h:
                for projection in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                    assert projection.weight.grad.abs().sum() > 0
        with torch.compiler.set_stance("force_eager"):
            optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    changed = tokens.clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4], rtol=0, atol=0)


@pytest.mark.parametrize("pattern,window", [("L", None), ("SSSL", None), ("SSSL", 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("options", [{}, {"diff_v2": True, "mlp_activation": "xiprelu", "n_loops": 2}])
def test_cached_prefill_token_and_chunk_continuations(monkeypatch, pattern, window, dtype, options):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(window_pattern=pattern, **options).eval()
    if window is not None:
        model.window_sizes = [(window, 0)] * model.eff_depth
    tokens = torch.arange(16).view(2, 8)
    tolerance = dict(rtol=0.04, atol=0.004) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-6)
    with torch.no_grad():
        expected = model(tokens)
        for chunks in ([3, 1, 1, 1, 1, 1], [3, 2, 3]):
            cache = cache_for(model, batch=2)
            shape, pointers = cache.k_cache.shape, (cache.k_cache.data_ptr(), cache.v_cache.data_ptr())
            start, outputs = 0, []
            for length in chunks:
                outputs.append(model(tokens[:, start:start+length], kv_cache=cache))
                start += length
            torch.testing.assert_close(torch.cat(outputs, 1), expected, **tolerance)
            assert cache.get_pos() == 8 and cache.k_cache.shape == shape
            assert pointers == (cache.k_cache.data_ptr(), cache.v_cache.data_ptr())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_per_batch_offsets_match_individual_full_sequences(monkeypatch, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(n_layer=4, depth_attention_stride=2, diff_v2=True, mlp_activation="xiprelu").eval()
    model.window_sizes = [(2, 0), (-1, 0), (2, 0), (-1, 0)]
    cache = cache_for(model, batch=2)
    prefixes = [torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5, 6]])]
    continuation = torch.tensor([[7, 8], [9, 10]])
    with torch.no_grad():
        previous = []
        for row, prefix in enumerate(prefixes):
            individual = cache_for(model)
            model(prefix, kv_cache=individual)
            end = prefix.size(1)
            cache.k_cache[:, row, :end].copy_(individual.k_cache[:, 0, :end])
            cache.v_cache[:, row, :end].copy_(individual.v_cache[:, 0, :end])
            cache.cache_seqlens[row] = end
            previous.append(individual.prev_embedding)
        cache.prev_embedding = torch.cat(previous)
        actual = model(continuation, kv_cache=cache)
        expected = torch.cat([model(torch.cat([prefix, continuation[row:row+1]], 1))[:, -2:]
                              for row, prefix in enumerate(prefixes)])
    torch.testing.assert_close(actual, expected, rtol=0.04 if dtype == torch.bfloat16 else 1e-4,
                               atol=0.004 if dtype == torch.bfloat16 else 1e-6)
    assert cache.cache_seqlens.tolist() == [4, 6]


@pytest.mark.parametrize("loops", [1, 2])
def test_engine_greedy_generation_and_prefill_replication_match_naive(monkeypatch, loops):
    monkeypatch.setattr(engine_module, "COMPUTE_DTYPE", torch.float32)
    torch.manual_seed(42)
    model = make_model(diff_v2=True, mlp_activation="xiprelu", n_loops=loops).eval()
    # Unreachable control tokens keep this focused on model/cache generation.
    tokenizer = SimpleNamespace(encode_special=lambda token: 1000, get_bos_token_id=lambda: 1001)
    prompt = [1, 2, 3]
    expected = prompt + list(model.generate(prompt, max_tokens=5, temperature=0))
    engine = engine_module.Engine(model, tokenizer)
    actual, masks = engine.generate_batch(prompt, num_samples=3, max_tokens=5, temperature=0)
    assert actual == [expected] * 3
    assert masks == [[0, 0, 0, 1, 1, 1, 1, 1]] * 3


@pytest.mark.parametrize("enabled", [False, True])
def test_strict_checkpoint_roundtrip_and_old_metadata(tmp_path, monkeypatch, enabled):
    model = make_model(depth_attention=enabled, mlp_activation="xiprelu", diff_v2=True).eval()
    config = asdict(model.config)
    if not enabled:
        config.pop("depth_attention")
        config.pop("depth_attention_stride")
    checkpoint_manager.save_checkpoint(tmp_path, 0, model.state_dict(), None, dict(model_config=config))
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, _ = checkpoint_manager.build_model(tmp_path, 0, torch.device("cpu"), "eval")
    assert restored.config.depth_attention == enabled
    tokens = torch.arange(8).view(1, 8)
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)


@pytest.mark.parametrize("script,folder", [("chat_sft.py", "chatsft_checkpoints"), ("chat_rl.py", "chatrl_checkpoints")])
@pytest.mark.parametrize("enabled", [False, True])
def test_finetuning_checkpoint_preserves_opt_in_without_training(tmp_path, monkeypatch, script, folder, enabled):
    # Execute the real save block offline; importing these scripts would launch training.
    source = Path(__file__).resolve().parents[1] / "scripts" / script
    tree = ast.parse(source.read_text())
    block = next(node for node in ast.walk(tree) if isinstance(node, ast.If) and any(
        isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "save_checkpoint"
        for stmt in node.body))
    model = make_model(depth_attention=enabled, depth_attention_stride=2 if enabled else None)
    namespace = dict(model=model, orig_model=model, depth=3, step=1, num_steps=2,
                     args=SimpleNamespace(model_tag=None, max_seq_len=16), base_dir=str(tmp_path),
                     get_base_dir=lambda: str(tmp_path), os=os, val_bpb=2., user_config={}, ddp_rank=0,
                     optimizer=SimpleNamespace(state_dict=lambda: {}),
                     tokenizer=SimpleNamespace(get_vocab_size=lambda: 256),
                     save_checkpoint=checkpoint_manager.save_checkpoint)
    exec(compile(ast.Module(body=block.body, type_ignores=[]), str(source), "exec"), namespace)
    tag = "d3-depth-attention-s2" if enabled else "d3"
    directory = tmp_path / folder / tag
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: SimpleNamespace(get_vocab_size=lambda: 256))
    restored, _, metadata = checkpoint_manager.build_model(directory, 1, torch.device("cpu"), "eval")
    assert restored.config.depth_attention == enabled
    if not enabled:
        assert "depth_attention" not in metadata["model_config"]
        assert "depth_attention_stride" not in metadata["model_config"]
    tokens = torch.arange(8).view(1, 8)
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)


def test_no_added_parameters_or_cache_slots_and_flops_accounting():
    for loops in (1, 2):
        base = make_model(depth_attention=False, n_loops=loops)
        depth = make_model(n_loops=loops)
        assert base.state_dict().keys() == depth.state_dict().keys()
        assert base.num_scaling_params() == depth.num_scaling_params()
        assert base.kv_bytes_per_token() == depth.kv_bytes_per_token()
        candidates = sum(len(range(0, i, 1)) + 1 for i in range(1, depth.eff_depth))
        extra = 4 * 2 * 8 * candidates
        assert depth.estimate_flops() - base.estimate_flops() == 3 * extra
        assert depth.estimate_decode_flops(8) - base.estimate_decode_flops(8) == extra
        assert depth.estimate_prefill_flops(8) - base.estimate_prefill_flops(8) == 8 * extra
    with torch.device("meta"):
        base = GPT(GPTConfig())
        depth = GPT(GPTConfig(depth_attention=True))
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in depth.parameters())
    assert base.num_scaling_params() == depth.num_scaling_params()


@pytest.mark.parametrize("options", [{}, {"diff_v2": True, "mlp_activation": "xiprelu", "n_loops": 2}])
def test_fullgraph_compile_forward_and_backward(options):
    model = make_model(**options)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    tokens = torch.arange(8).view(1, 8)
    torch.testing.assert_close(compiled(tokens), model(tokens), rtol=0, atol=0)
    compiled(tokens, targets=tokens.roll(-1, 1)).backward()
    assert model.transformer.h[0].attn.c_k.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="CUDA BF16 hardware required")
@pytest.mark.parametrize("backend", ["sdpa", "fa3"])
def test_cuda_bf16_training_and_cached_generation(monkeypatch, backend):
    if backend == "fa3" and not fa.HAS_FA3:
        pytest.skip("FA3 kernels unavailable")
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.bfloat16)
    monkeypatch.setattr(fa, "USE_FA3", backend == "fa3")
    model = make_model(device="cuda", n_embd=64, n_head=4, n_kv_head=2)
    tokens = torch.arange(8, device="cuda").view(1, 8)
    loss = model(tokens, targets=tokens.roll(-1, 1))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer = model.setup_optimizer()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with torch.no_grad():
        cache = cache_for(model, device="cuda")
        actual = torch.cat([model(tokens[:, :3], kv_cache=cache), model(tokens[:, 3:5], kv_cache=cache)] +
                           [model(tokens[:, i:i+1], kv_cache=cache) for i in range(5, 8)], 1)
        expected = model(tokens)
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.004)
        if backend == "fa3":
            monkeypatch.setattr(fa, "USE_FA3", False)
            torch.testing.assert_close(expected, model(tokens), rtol=0.04, atol=0.004)
