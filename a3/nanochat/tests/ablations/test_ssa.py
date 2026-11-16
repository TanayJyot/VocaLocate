"""SSA: the temperature formula, baseline equivalence at alpha=0, gradients, KV-cache agreement and experiment selection."""
from types import SimpleNamespace

import pytest
import torch

import nanochat.flash_attention as fa
import nanochat.gpt as gpt
from nanochat.engine import KVCache
from nanochat.experiments import select_experiment


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    torch._dynamo.reset()
    monkeypatch.setattr(fa, "USE_FA3", False)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)
    yield
    torch._dynamo.reset()


def make_model(active=False, **overrides):
    options = dict(sequence_len=16, vocab_size=256, n_layer=6, n_head=4, n_kv_head=4, n_embd=32, window_pattern="SSSL")
    options.update(overrides)
    torch.manual_seed(42)
    with torch.device("meta"):
        model = gpt.GPT(gpt.GPTConfig(**options))
    model.to_empty(device="cpu")
    model.init_weights()
    if active:
        with torch.no_grad():
            for block in model.transformer.h:
                torch.manual_seed(block.attn.layer_idx) # per layer, so drawing beta can't shift the other weights
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
                if block.attn.ssa_beta is not None:
                    block.attn.ssa_beta.normal_(mean=-1.0, std=0.5) # alpha around 0.7: a large effect
    return model


def test_init_and_shared_weights():
    base, model = make_model(), make_model(ssa=True)
    assert all(torch.equal(b.attn.ssa_beta, torch.full((4,), 1.5)) for b in model.transformer.h)
    shared = {k: v for k, v in model.state_dict().items() if k in base.state_dict()}
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in shared.items()), "shared weights must match at the same seed"


def test_zero_alpha_matches_baseline():
    idx = torch.randint(0, 256, (2, 16))
    base, model = make_model(active=True), make_model(active=True, ssa=True)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.ssa_beta.fill_(20.0) # exp(-exp(20)) = 0
    torch.testing.assert_close(model(idx), base(idx), rtol=0, atol=0)


def test_query_scaled_by_position_temperature_after_qk_norm():
    model = make_model(ssa=True)
    attn = model.transformer.h[1].attn
    betas = torch.tensor([1.5, 0.5, 0.0, -1.0])
    with torch.no_grad():
        attn.ssa_beta.copy_(betas)
    captured = {}
    orig = fa.flash_attn.flash_attn_func
    def spy(q, k, v, **kw):
        captured["q"] = q
        return orig(q, k, v, **kw)
    fa.flash_attn.flash_attn_func = spy
    try:
        x = torch.randn(2, 16, 32)
        cos_sin = model.cos[:, :16], model.sin[:, :16]
        with torch.no_grad():
            attn(x, None, cos_sin, (16, 0), None, 1)
    finally:
        fa.flash_attn.flash_attn_func = orig
    # QK norm makes each query's RMS 1 and nanochat scales it by 1.2, so the RMS left is exactly 1.2 * T(pos, head)
    rms = captured["q"].pow(2).mean(-1).sqrt()
    alpha = torch.exp(-torch.exp(betas))
    expected = 1.2 * (1 + alpha[None, None, :] * torch.log(torch.arange(1.0, 17.0))[None, :, None])
    torch.testing.assert_close(rms, expected.expand_as(rms), atol=1e-4, rtol=1e-4)


def test_beta_gets_gradients_and_an_optimizer_group():
    model = make_model(active=True, ssa=True)
    idx = torch.randint(0, 256, (2, 16))
    model(idx, targets=torch.randint(0, 256, (2, 16))).backward()
    for block in model.transformer.h:
        grad = block.attn.ssa_beta.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.setup_optimizer() # every parameter lands in exactly one optimizer group


def test_kv_cache_decode_matches_full_forward():
    model = make_model(active=True, ssa=True, window_pattern="L").eval()
    idx = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        full = model(idx)
        cache = KVCache(batch_size=1, num_heads=4, seq_len=16, head_dim=8, num_layers=6, device="cpu",
                        dtype=torch.float32)
        steps = [model(idx[:, :8], kv_cache=cache)]
        steps += [model(idx[:, t:t + 1], kv_cache=cache) for t in range(8, 12)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-4, rtol=1e-4)


def test_flops_unchanged_and_unsupported_mixes_rejected():
    assert make_model(ssa=True).estimate_flops() == make_model().estimate_flops()
    for other in ("diff_v2", "depth_attention"):
        with pytest.raises(AssertionError):
            make_model(ssa=True, **{other: True})


def test_selection():
    args = SimpleNamespace(experiment="ssa", n_loops=1, n_prelude=0, n_coda=0, mlp_activation="relu2", diff_v2=False,
                           depth_attention=False, depth_attention_stride=None, mgvr=False, xsa=False, ssa=False, mudd=False)
    select_experiment(args)
    assert args.ssa and not args.xsa
    args.experiment = "baseline"
    with pytest.raises(ValueError):
        select_experiment(args)
