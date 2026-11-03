"""
Tests for the looped (weight-shared recurrent depth) GPT. Run with:

python -m pytest tests/ablations/test_looped.py -v
"""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache
from nanochat.common import COMPUTE_DTYPE

DEVICE = torch.device("cpu")


def make(n_layer=4, n_loops=1, n_prelude=0, n_coda=0, window_pattern="L", seed=0, **extra):
    cfg = GPTConfig(sequence_len=64, vocab_size=300, n_layer=n_layer, n_head=2, n_kv_head=2, n_embd=64,
                    window_pattern=window_pattern, n_loops=n_loops, n_prelude=n_prelude, n_coda=n_coda, **extra)
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=DEVICE)
    model.init_weights()
    # c_proj is zero-initialised (blocks start as identity), randomise so the tests are not vacuous
    with torch.no_grad():
        for p in model.transformer.h.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return model


def test_layer_plan():
    cfg = GPTConfig(n_layer=4, n_prelude=1, n_coda=1, n_loops=3)
    assert cfg.layer_plan() == [0, 1, 2, 1, 2, 1, 2, 3]
    assert cfg.effective_depth == 8 and cfg.n_core == 2
    assert GPTConfig(n_layer=5).layer_plan() == [0, 1, 2, 3, 4]  # default = ordinary GPT
    with pytest.raises(AssertionError):
        GPTConfig(n_layer=2, n_prelude=1, n_coda=1).layer_plan()  # no core left


def test_default_matches_original_gpt():
    """n_loops=1 must reproduce the pre-change model exactly (same params, same logits, same flops)."""
    # upstream karpathy/nanochat commit just before looping was added (the un-looped reference model)
    root = Path(__file__).resolve().parents[2]
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, encoding="utf-8")
    src = git("show", "92d63d4:./nanochat/gpt.py")
    if src.returncode != 0:
        # nanochat vendored into another repo (no upstream history): use the commit that first added gpt.py
        added = git("log", "--diff-filter=A", "--format=%H", "--", "nanochat/gpt.py").stdout.split()
        src = git("show", f"{added[-1]}:./nanochat/gpt.py") if added else src
    if src.returncode != 0:
        pytest.skip("original gpt.py (upstream 92d63d4, or the commit that added it) not available in this clone")
    spec = importlib.util.spec_from_loader("gpt_orig", loader=None)
    orig = importlib.util.module_from_spec(spec)
    exec(compile(src.stdout, "gpt_orig.py", "exec"), orig.__dict__)
    cfg = dict(sequence_len=64, vocab_size=300, n_layer=4, n_head=2, n_kv_head=2, n_embd=64, window_pattern="L")
    models = []
    for mod in (orig, sys.modules["nanochat.gpt"]):
        torch.manual_seed(0)
        with torch.device("meta"):
            m = mod.GPT(mod.GPTConfig(**cfg))
        m.to_empty(device=DEVICE)
        m.init_weights()
        torch.manual_seed(1)
        with torch.no_grad():
            for p in m.transformer.h.parameters():
                p.add_(torch.randn_like(p) * 0.02)
        models.append(m)
    a, b = models
    idx = torch.randint(0, 300, (2, 32))
    assert torch.allclose(a(idx), b(idx), atol=1e-5)
    assert a.estimate_flops() == b.estimate_flops()
    assert a.num_scaling_params() == b.num_scaling_params()


def test_weights_are_shared_and_compute_scales():
    base = make(n_layer=4)
    looped = make(n_layer=4, n_loops=3)
    # same unique parameters, but 3x the executed depth for the core
    assert looped.num_scaling_params()["transformer_matrices"] == base.num_scaling_params()["transformer_matrices"]
    assert looped.eff_depth == 12 and looped.resid_lambdas.numel() == 12
    assert looped.estimate_flops() > 2.5 * base.estimate_flops()
    # exactly the same compute as an unshared 12-layer model on matmul params of the blocks
    big = make(n_layer=12)
    per_block = base.num_matmul_params() - (base.lm_head.weight.numel() + base.smear_gate.weight.numel())
    expected = 12 * per_block // 4 + base.lm_head.weight.numel() + base.smear_gate.weight.numel()
    assert looped.num_matmul_params() == expected
    assert big.num_matmul_params() == expected


def test_loops_change_output_and_grads_reach_shared_blocks():
    m1 = make(n_layer=3, n_loops=1)
    m2 = make(n_layer=3, n_loops=2)
    idx = torch.randint(0, 300, (2, 16))
    # m2 has 6 steps so its per-step scalars differ; only check the outputs differ and are finite
    out = m2(idx)
    assert torch.isfinite(out).all() and not torch.allclose(m1(idx), out)
    targets = torch.randint(0, 300, (2, 16))
    loss = m2(idx, targets)
    loss.backward()
    for b, block in enumerate(m2.transformer.h):
        assert block.attn.c_q.weight.grad is not None and block.attn.c_q.weight.grad.abs().sum() > 0, b


@pytest.mark.parametrize("kw", [dict(n_layer=3, n_loops=3), dict(n_layer=5, n_loops=2, n_prelude=1, n_coda=1),
                                dict(n_layer=5, n_loops=2, n_prelude=1, n_coda=1, n_loops_min=1, n_loops_max=3, loop_embed=True)])
def test_kv_cache_decode_matches_full_forward(kw):
    """Each executed step needs its own KV slot: incremental decode must equal the full forward pass."""
    m = make(**kw).eval()
    cfg = m.config
    T = 12
    idx = torch.randint(0, 300, (1, T))
    with torch.no_grad():
        full = m(idx)
        cache = KVCache(batch_size=1, num_heads=cfg.n_kv_head, seq_len=32, head_dim=cfg.n_embd // cfg.n_head,
                        num_layers=cfg.effective_depth, device=DEVICE, dtype=COMPUTE_DTYPE) # the cache holds k/v in the compute dtype (bf16 on Ampere+ CUDA boxes)
        pre = m(idx[:, :8], kv_cache=cache)
        steps = [pre[:, -1]]
        for t in range(8, T):
            steps.append(m(idx[:, t:t + 1], kv_cache=cache)[:, -1])
    assert torch.allclose(full[:, 7], steps[0], atol=1e-3)
    assert torch.allclose(full[:, -1], steps[-1], atol=1e-3)
    assert cache.get_pos() == T


# --- per-loop embeddings and random loop counts ---

def test_step_slots_for_variable_loop_counts():
    cfg = GPTConfig(n_layer=4, n_prelude=1, n_coda=1, n_loops=2, n_loops_min=1, n_loops_max=3)
    assert cfg.loop_range == (1, 3) and cfg.n_passes == 3 and cfg.n_step_slots == 8
    assert cfg.layer_plan(1) == [0, 1, 2, 3] and cfg.step_slots(1) == [0, 1, 2, 7]
    assert cfg.step_slots(2) == [0, 1, 2, 3, 4, 7]
    assert cfg.step_slots(3) == list(range(8))
    # passes beyond n_passes reuse the last pass's slots; the coda keeps its own
    assert cfg.step_slots(5) == [0, 1, 2, 3, 4, 5, 6, 5, 6, 5, 6, 7]
    # fixed loop counts are unchanged: one slot per executed step
    fixed = GPTConfig(n_layer=4, n_prelude=1, n_coda=1, n_loops=3)
    assert fixed.step_slots() == list(range(fixed.effective_depth)) and fixed.n_step_slots == fixed.effective_depth
    with pytest.raises(AssertionError):
        GPTConfig(n_loops=2, n_loops_max=1).loop_range


def test_zero_init_loop_embed_matches_plain_looped_model():
    plain = make(n_layer=4, n_loops=3, n_prelude=1, n_coda=1)
    embed = make(n_layer=4, n_loops=3, n_prelude=1, n_coda=1, loop_embed=True)
    assert embed.loop_embeds.shape == (3, 64) and embed.loop_embeds.abs().sum() == 0
    embed.load_state_dict({**plain.state_dict(), "loop_embeds": embed.loop_embeds.detach().clone()})
    idx = torch.randint(0, 300, (2, 16))
    torch.testing.assert_close(embed(idx), plain(idx), rtol=0, atol=0)
    counts = embed.num_scaling_params()
    assert counts["scalars"] - plain.num_scaling_params()["scalars"] == 3 * 64
    assert embed.estimate_flops() == plain.estimate_flops()
    # each pass gets its own vector, and it is trained by AdamW
    loss = embed(idx, torch.randint(0, 300, (2, 16)))
    loss.backward()
    assert (embed.loop_embeds.grad.abs().sum(dim=1) > 0).all()
    groups = embed.setup_optimizer().param_groups
    assert any(g["kind"] == "adamw" and any(p is embed.loop_embeds for p in g["params"]) for g in groups)
    with torch.no_grad():
        embed.loop_embeds.normal_()
    assert not torch.allclose(embed(idx), plain(idx))


def test_random_loop_model_runs_any_loop_count():
    m = make(n_layer=4, n_loops=2, n_prelude=1, n_coda=1, n_loops_min=1, n_loops_max=3)
    fixed3 = make(n_layer=4, n_loops=3, n_prelude=1, n_coda=1)
    # sized for 3 passes, so at 3 loops it is exactly the fixed 3-loop model
    fixed3.load_state_dict(m.state_dict())
    idx = torch.randint(0, 300, (2, 16))
    torch.testing.assert_close(m(idx, n_loops=3), fixed3(idx), rtol=0, atol=0)
    outs = [m(idx, n_loops=n) for n in (1, 2, 3, 5)]
    assert all(torch.isfinite(o).all() for o in outs)
    assert not torch.allclose(outs[0], outs[1]) and not torch.allclose(outs[2], outs[3])
    torch.testing.assert_close(m(idx), outs[1], rtol=0, atol=0) # default = n_loops
    # training cost is the average over the sampled loop counts
    expected = sum(m.estimate_flops(n) for n in (1, 2, 3)) / 3
    assert m.estimate_flops() == pytest.approx(expected)
    assert m.estimate_flops(3) == fixed3.estimate_flops()
    # the KV cache only covers the default loop count
    cache = KVCache(batch_size=2, num_heads=2, seq_len=32, head_dim=32, num_layers=m.config.effective_depth,
                    device=DEVICE, dtype=COMPUTE_DTYPE)
    with pytest.raises(AssertionError):
        m(idx[:, :4], kv_cache=cache, n_loops=3)


def test_loop_extensions_reject_depth_attention_and_mgvr():
    for extra in (dict(depth_attention=True), dict(mgvr=True)):
        for loops in (dict(loop_embed=True), dict(n_loops_min=1, n_loops_max=3)):
            with pytest.raises(AssertionError):
                make(n_layer=4, n_loops=2, n_prelude=1, n_coda=1, **extra, **loops)
