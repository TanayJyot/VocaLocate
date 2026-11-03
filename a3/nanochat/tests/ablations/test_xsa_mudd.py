"""XSA and MUDD-lite: zero-init identity, the XSA formula, gradients, KV-cache agreement and experiment selection."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

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
        torch.manual_seed(7)
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
                if block.attn.xsa_alpha is not None:
                    block.attn.xsa_alpha.normal_(std=1.0)
            if model.mudd_w1 is not None:
                model.mudd_w2.normal_(std=0.5)
                model.mudd_b.normal_(std=0.5)
    return model


@pytest.mark.parametrize("flag", ["xsa", "mudd"])
def test_zero_init_matches_baseline(flag):
    idx = torch.randint(0, 256, (2, 16))
    base, model = make_model(), make_model(**{flag: True})
    with torch.no_grad():
        for block in (*base.transformer.h, *model.transformer.h): # make attention/MLP outputs nonzero, identically
            torch.manual_seed(block.attn.layer_idx)
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
    shared = {k: v for k, v in model.state_dict().items() if k in base.state_dict()}
    assert all(torch.equal(v, base.state_dict()[k]) for k, v in shared.items()), "shared weights must match at the same seed"
    torch.testing.assert_close(model(idx), base(idx), rtol=0, atol=0)


def test_xsa_removes_self_value_component():
    model = make_model(xsa=True)
    attn = model.transformer.h[1].attn
    with torch.no_grad():
        attn.xsa_alpha.fill_(20.0) # tanh -> 1: the full component along v is removed
    captured = {}
    orig = fa.flash_attn.flash_attn_func
    def spy(q, k, v, **kw):
        y = orig(q, k, v, **kw)
        captured["v"], captured["y"] = v, y
        return y
    fa.flash_attn.flash_attn_func = spy
    try:
        x = torch.randn(2, 16, 32)
        cos_sin = model.cos[:, :16], model.sin[:, :16]
        with torch.no_grad():
            attn.c_proj.weight.copy_(torch.eye(32))
            out = attn(x, None, cos_sin, (16, 0), None, 1).view(2, 16, 4, 8)
    finally:
        fa.flash_attn.flash_attn_func = orig
    vn = F.normalize(captured["v"], dim=-1)
    torch.testing.assert_close((out * vn).sum(-1), torch.zeros(2, 16, 4), atol=1e-5, rtol=0)
    y = captured["y"]
    torch.testing.assert_close(out, y - (y * vn).sum(-1, keepdim=True) * vn, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("flag", ["xsa", "mudd"])
def test_new_parameters_get_gradients_at_init(flag):
    model = make_model(**{flag: True})
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
    idx = torch.randint(0, 256, (2, 16))
    model(idx, targets=torch.randint(0, 256, (2, 16))).backward()
    params = [b.attn.xsa_alpha for b in model.transformer.h] if flag == "xsa" else [model.mudd_w2, model.mudd_b]
    for p in params:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    model.setup_optimizer() # every parameter lands in exactly one optimizer group


@pytest.mark.parametrize("flag", ["xsa", "mudd"])
def test_kv_cache_decode_matches_full_forward(flag):
    model = make_model(active=True, **{flag: True, "window_pattern": "L"}).eval()
    idx = torch.randint(0, 256, (1, 12))
    with torch.no_grad():
        full = model(idx)
        cache = KVCache(batch_size=1, num_heads=4, seq_len=16, head_dim=8, num_layers=6, device="cpu",
                        dtype=torch.float32)
        steps = [model(idx[:, :8], kv_cache=cache)]
        steps += [model(idx[:, t:t + 1], kv_cache=cache) for t in range(8, 12)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-4, rtol=1e-4)


def test_mudd_flops_and_param_accounting():
    base, model = make_model(), make_model(mudd=True)
    extra = model.mudd_w1.numel() + model.mudd_w2.numel()
    assert model.num_matmul_params() == base.num_matmul_params() + extra
    counts = model.num_scaling_params()
    assert counts["mudd"] == extra + model.mudd_b.numel()
    assert counts["transformer_matrices"] == base.num_scaling_params()["transformer_matrices"]


@pytest.mark.parametrize("selected", ["xsa", "mudd"])
def test_selection(selected):
    args = SimpleNamespace(experiment=selected, n_loops=1, n_prelude=0, n_coda=0, mlp_activation="relu2",
                           diff_v2=False, depth_attention=False, depth_attention_stride=None, mgvr=False,
                           xsa=False, mudd=False)
    select_experiment(args)
    assert getattr(args, selected) and not any(getattr(args, f) for f in ("xsa", "mudd") if f != selected)
    args.experiment, args.mgvr = "baseline", False
    with pytest.raises(ValueError):
        select_experiment(args) # the flag set above conflicts with --experiment=baseline
