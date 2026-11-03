"""All ablations explicitly off versus the imported, pre-ablation NanoChat."""
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest
import torch

import nanochat.flash_attention as fa
import nanochat.gpt as gpt
import nanochat.optim as optim_module
from nanochat.engine import KVCache

ORIGINAL_COMMIT = "0ed62a1"


@pytest.fixture
def original_modules(monkeypatch):
    modules = {}
    for name in ("optim", "gpt"):
        source = subprocess.check_output(["git", "show", f"{ORIGINAL_COMMIT}:./nanochat/{name}.py"],
                                         cwd=Path(__file__).resolve().parents[2], text=True)
        module = ModuleType(f"_original_nanochat_{name}")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source, f"original_nanochat/{name}.py", "exec"), module.__dict__)
        modules[name] = module
    modules["gpt"].MuonAdamW = modules["optim"].MuonAdamW
    return modules


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("kv_heads", [2, 4])
@pytest.mark.parametrize("pattern", ["L", "SSSL"])
def test_all_flags_disabled_matches_original(original_modules, monkeypatch, dtype, kv_heads, pattern):
    monkeypatch.setattr(fa, "USE_FA3", False)
    original = original_modules["gpt"]
    for module in (gpt, optim_module, original, original_modules["optim"]):
        monkeypatch.setattr(module, "COMPUTE_DTYPE", dtype)
    shared = dict(sequence_len=16, vocab_size=256, n_layer=4, n_head=4,
                  n_kv_head=kv_heads, n_embd=32, window_pattern=pattern)
    disabled = dict(n_loops=1, n_prelude=0, n_coda=0, mlp_activation="relu2",
                    diff_v2=False, depth_attention=False, depth_attention_stride=None, mgvr=False)
    models = []
    for module, extra in ((original, {}), (gpt, disabled)):
        torch.manual_seed(71)
        with torch.device("meta"):
            model = module.GPT(module.GPTConfig(**shared, **extra))
        model.to_empty(device="cpu")
        model.init_weights()
        # Exercise real attention/MLP behavior, rather than zero residual projections.
        torch.manual_seed(72)
        with torch.no_grad():
            for block in model.transformer.h:
                block.attn.c_proj.weight.normal_(std=0.05)
                block.mlp.c_proj.weight.normal_(std=0.05)
            model.smear_lambda.fill_(0.3)
        if pattern == "SSSL":
            model.window_sizes = [(2, 0)] * 3 + [(16, 0)]
        models.append(model)
    old, current = models
    assert old.state_dict().keys() == current.state_dict().keys()
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, current.state_dict()[name], rtol=0, atol=0)
    assert old.num_scaling_params() == current.num_scaling_params()
    assert old.estimate_flops() == current.estimate_flops()
    optimizers = [model.setup_optimizer() for model in models]
    assert optimizers[0].state_dict() == optimizers[1].state_dict()
    tokens = torch.arange(16).view(2, 8)
    for _ in range(2):
        torch.testing.assert_close(old(tokens), current(tokens), rtol=0, atol=0)
        losses = []
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss = model(tokens, targets=tokens.roll(-1, 1))
            losses.append(loss)
            loss.backward()
        torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
        for before, after in zip(old.parameters(), current.parameters()):
            torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0)
        with torch.compiler.set_stance("force_eager"):
            for optimizer in optimizers:
                optimizer.step()
        for before, after in zip(old.parameters(), current.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    caches = [KVCache(2, kv_heads, 16, 8, 4, "cpu", dtype) for _ in models]
    with torch.no_grad():
        for start, end in ((0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)):
            outputs = [model(tokens[:, start:end], kv_cache=cache) for model, cache in zip(models, caches)]
            torch.testing.assert_close(*outputs, rtol=0, atol=0)
    torch.testing.assert_close(caches[0].k_cache, caches[1].k_cache, rtol=0, atol=0)
    torch.testing.assert_close(caches[0].v_cache, caches[1].v_cache, rtol=0, atol=0)
