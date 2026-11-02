"""Independent, offline audit of nanochat Depth-Attention.

From the repository root:
    python -m dev.check_depth_attention_validity
    python -m dev.check_depth_attention_validity --inductor

The default invocation checks equations, disabled ATen graphs and cache calls.
The optional invocation checks real CPU Inductor forward/backward compilation.
It needs a working C++ toolchain and shared-library search path.
"""
import argparse
import copy
import itertools
import json
import subprocess
import sys
import time
from types import ModuleType

import torch
import torch.nn.functional as F
from torch.fx import map_arg
from torch.profiler import ProfilerActivity, profile

import nanochat.gpt as gpt
from nanochat.engine import KVCache

gpt.COMPUTE_DTYPE = torch.float32
torch.set_num_threads(1)


def model_for(**options):
    config = dict(sequence_len=64, vocab_size=256, n_layer=4, n_head=4,
                  n_kv_head=2, n_embd=32, window_pattern="L")
    config.update(options)
    torch.manual_seed(91)
    with torch.device("meta"):
        model = gpt.GPT(gpt.GPTConfig(**config))
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.1)
            block.mlp.c_proj.weight.normal_(std=0.1)
        model.smear_lambda.fill_(0.3)
    return model


def sdpa_depth_reference(q, k, v, earlier):
    # Treat each (batch, token, KV head) as a separate SDPA batch. Source
    # depth is SDPA's sequence dimension; there is one query in each batch.
    b, t, h, d = k.shape
    query = q.float().unflatten(2, (h, -1)).mean(3).reshape(-1, 1, d)
    sources = [*earlier, (k, v)]
    keys = torch.stack([pair[0].float() for pair in sources], 3).reshape(-1, len(sources), d)
    values = torch.stack([pair[1].float() for pair in sources], 3).reshape_as(keys)
    return F.scaled_dot_product_attention(query, keys, values).reshape(b, t, h, d).to(v.dtype)


def equations():
    original = gpt.mix_depth_values
    results = []
    options = [dict(n_kv_head=4), dict(n_kv_head=1),
               dict(diff_v2=True, mlp_activation="xiprelu", n_loops=2, n_prelude=1, n_coda=1)]
    for dtype, stride, variant in itertools.product(
            (torch.float32, torch.bfloat16), (1, 2, 99), options):
        gpt.COMPUTE_DTYPE = dtype
        actual_model = model_for(depth_attention=True, depth_attention_stride=stride, **variant)
        reference_model = copy.deepcopy(actual_model)
        tokens = torch.arange(16).reshape(2, 8)
        targets = tokens.roll(-1, 1)
        logits = actual_model(tokens)
        loss = actual_model(tokens, targets)
        loss.backward()
        try:
            gpt.mix_depth_values = sdpa_depth_reference
            expected = reference_model(tokens)
            reference_model(tokens, targets).backward()
        finally:
            gpt.mix_depth_values = original
        tolerance = dict(rtol=1e-4, atol=1e-6) if dtype == torch.float32 else dict(rtol=0.05, atol=0.0003)
        torch.testing.assert_close(logits, expected, **tolerance)
        max_grad_error = 0.0
        for (name, actual), (_, reference) in zip(actual_model.named_parameters(), reference_model.named_parameters()):
            torch.testing.assert_close(actual.grad, reference.grad, **tolerance, msg=lambda msg: name + ": " + msg)
            max_grad_error = max(max_grad_error, (actual.grad - reference.grad).abs().max().item())
        results.append(dict(dtype=str(dtype), stride=stride, variant=variant,
                            max_logits_error=(logits - expected).abs().max().item(),
                            max_gradient_error=max_grad_error))
    gpt.COMPUTE_DTYPE = torch.float32
    print(json.dumps(dict(sdpa_equation_checks=results)), flush=True)


def graph_signature(model, tokens):
    graph = torch._dynamo.export(model, aten_graph=True)(tokens).graph_module.graph
    index = {node: i for i, node in enumerate(graph.nodes)}
    return [(node.op, str(node.target), map_arg(node.args, lambda arg: index[arg]),
             map_arg(node.kwargs, lambda arg: index[arg])) for node in graph.nodes]


def disabled_graphs():
    # Hold the unrelated, pre-existing Diff V2 initialization edit constant.
    source = subprocess.check_output(["git", "show", "0fb1004:./nanochat/gpt.py"], text=True)
    previous = ModuleType("_depth_audit_previous_gpt")
    sys.modules[previous.__name__] = previous
    exec(compile(source, "previous_gpt.py", "exec"), previous.__dict__)
    previous.COMPUTE_DTYPE = torch.float32
    variants = [{}, dict(mlp_activation="xiprelu"), dict(diff_v2=True),
                dict(n_loops=2, n_prelude=1, n_coda=1),
                dict(n_loops=2, mlp_activation="xiprelu", diff_v2=True)]
    results = []
    tokens = torch.arange(16).reshape(2, 8)
    for options in variants:
        current = model_for(**options)
        old_options = vars(current.config).copy()
        old_options.pop("depth_attention")
        old_options.pop("depth_attention_stride")
        with torch.device("meta"):
            old = previous.GPT(previous.GPTConfig(**old_options))
        old.to_empty(device="cpu")
        old.init_weights()
        old.load_state_dict(current.state_dict(), strict=True)
        before, after = graph_signature(old, tokens), graph_signature(current, tokens)
        assert before == after, options
        results.append(dict(variant=options, aten_graph_nodes=len(after), identical=True))
    print(json.dumps(dict(disabled_aten_graph_checks=results)), flush=True)


def cache_profile():
    results = []
    for enabled in (False, True):
        model = model_for(depth_attention=enabled)
        cache = KVCache(2, 2, 64, 8, model.eff_depth, "cpu", torch.float32)
        with torch.no_grad():
            model(torch.tensor([[1, 2], [3, 4]]), kv_cache=cache)
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                model(torch.tensor([[5], [6]]), kv_cache=cache)
        counts = {event.key: event.count for event in prof.key_averages()}
        results.append(dict(enabled=enabled, effective_depth=model.eff_depth,
                            equal_calls=counts.get("aten::equal", 0),
                            local_scalar_calls=counts.get("aten::_local_scalar_dense", 0)))
    print(json.dumps(dict(cached_decode_cpu_profile=results)), flush=True)


def inductor():
    for options in [dict(depth_attention=False), dict(depth_attention=True),
                    dict(depth_attention=True, diff_v2=True, mlp_activation="xiprelu", n_loops=2)]:
        torch._dynamo.reset()
        eager = model_for(n_layer=2, **options)
        compiled_base = copy.deepcopy(eager)
        compiled = torch.compile(compiled_base, fullgraph=True, dynamic=False)
        tokens = torch.arange(16).reshape(2, 8)
        targets = tokens.roll(-1, 1)
        start = time.perf_counter()
        eager_loss = eager(tokens, targets)
        eager_loss.backward()
        loss = compiled(tokens, targets)
        loss.backward()
        torch.testing.assert_close(loss, eager_loss, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(compiled_base.parameters(), eager.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-3, atol=1e-6)
        for _ in range(3):
            compiled_base.zero_grad(set_to_none=True)
            compiled(tokens, targets).backward()
        print(json.dumps(dict(inductor_check=options, loss=loss.item(),
                              compile_and_run_seconds=time.perf_counter()-start)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inductor", action="store_true")
    args = parser.parse_args()
    if args.inductor:
        inductor()
    else:
        equations()
        disabled_graphs()
        cache_profile()
