"""Independent combined-attention arithmetic and executed-depth state checks."""
import copy
from itertools import product

import pytest
import torch
import torch.nn.functional as F

import nanochat.flash_attention as fa
import nanochat.gpt as gpt
from nanochat.engine import KVCache


@pytest.fixture(autouse=True)
def cpu_sdpa(monkeypatch):
    monkeypatch.setattr(fa, "USE_FA3", False)


def make_model(**options):
    config = dict(sequence_len=16, vocab_size=64, n_layer=3, n_head=4,
                  n_kv_head=2, n_embd=32, window_pattern="L", mlp_activation="xiprelu")
    config.update(options)
    torch.manual_seed(81)
    with torch.device("meta"):
        model = gpt.GPT(gpt.GPTConfig(**config))
    model.to_empty(device="cpu")
    model.init_weights()
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.normal_(std=0.05)
            block.mlp.c_proj.weight.normal_(std=0.05)
            if block.attn.mgvr is not None:
                block.attn.mgvr.w_z.normal_(std=0.1)
                block.attn.mgvr.lam.normal_(std=0.1)
        model.smear_lambda.fill_(0.3)
    return model


def reference_forward(model, tokens):
    """Bypass GPT/Block/Attention forward, MGVR.forward and mix_depth_values.

    Express depth mixing as a separate SDPA over source depths, and sequence
    attention as one explicit SDPA per query head to independently check GQA.
    """
    c = model.config
    batch, time = tokens.shape
    heads, groups, dim = c.n_head, c.n_kv_head, c.n_embd // c.n_head
    linear = lambda x, module: F.linear(x, module.weight.to(x.dtype))
    normalize = lambda x: F.rms_norm(x, (x.size(-1),))
    x = normalize(F.embedding(tokens, model.transformer.wte.weight).to(gpt.COMPUTE_DTYPE))
    gate = model.smear_lambda.to(x.dtype) * linear(x[:, 1:, :24], model.smear_gate).sigmoid()
    x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
    initial = x
    core_end = c.n_layer - c.n_coda
    plan = list(range(c.n_prelude)) + list(range(c.n_prelude, core_end)) * c.n_loops + list(range(core_end, c.n_layer))
    keys, values, anchor, backout = {}, {}, None, None
    positions = torch.arange(time)
    causal = positions[:, None] >= positions[None, :]
    for step, block_index in enumerate(plan):
        block = model.transformer.h[block_index]
        attn = block.attn
        x = model.resid_lambdas[step] * x + model.x0_lambdas[step] * initial
        attn_input = normalize(x)
        query_heads = heads * (2 if c.diff_v2 else 1)
        q = linear(attn_input, attn.c_q).view(batch, time, query_heads, dim)
        k = linear(attn_input, attn.c_k).view(batch, time, groups, dim)
        v = linear(attn_input, attn.c_v).view(batch, time, groups, dim)
        if str(block_index) in model.value_embeds:
            ve = F.embedding(tokens, model.value_embeds[str(block_index)].weight).to(x.dtype)
            ve = ve.view_as(v)
            ve_gate = 3 * linear(attn_input[..., :12], attn.ve_gate).sigmoid()
            v = v + ve_gate[..., None] * ve
        if step == 0:
            anchor = v
        if c.mgvr and block_index > 0:
            module = attn.mgvr
            z = F.linear(attn_input.float(), module.w_z) / c.n_embd**0.5
            gate = 2 * z[..., :groups].sigmoid()
            mix = module.lam + z[..., groups:].reshape(batch, time, groups, groups).tanh()
            addition = torch.matmul(mix, anchor.float())
            v = (gate[..., None] * v.float() + addition).to(v.dtype)
        cos, sin = model.cos[:, :time], model.sin[:, :time]
        q = normalize(gpt.apply_rotary_emb(q, cos, sin)) * 1.2
        k = normalize(gpt.apply_rotary_emb(k, cos, sin)) * 1.2
        if c.depth_attention and step > 0:
            sources = list(range(0, step, c.effective_depth_attention_stride))
            key_sources = torch.stack([keys[index].float() for index in sources] + [k.float()], 3)
            value_sources = torch.stack([values[index].float() for index in sources] + [v.float()], 3)
            sharing = query_heads // groups
            query = torch.stack([q[:, :, h*sharing:(h+1)*sharing].float().mean(2) for h in range(groups)], 2)
            mixed = F.scaled_dot_product_attention(
                query.reshape(-1, 1, dim), key_sources.reshape(-1, len(sources)+1, dim),
                value_sources.reshape(-1, len(sources)+1, dim))
            v = mixed.reshape_as(v).to(v.dtype)
        keys[step], values[step] = k, v
        window = model.window_sizes[step][0]
        mask = causal if window < 0 else causal & (positions[:, None] - positions[None, :] <= window)
        outputs = []
        for head in range(query_heads):
            kv_head = head // (query_heads // groups)
            outputs.append(F.scaled_dot_product_attention(
                q[:, :, head], k[:, :, kv_head], v[:, :, kv_head], attn_mask=mask))
        y = torch.stack(outputs, 2)
        if c.diff_v2:
            lam = linear(attn_input, attn.lambda_proj).sigmoid().to(y.dtype)
            y = torch.stack([y[:, :, 2*h] - lam[:, :, h, None] * y[:, :, 2*h+1] for h in range(heads)], 2)
        x = x + linear(y.reshape(batch, time, c.n_embd), attn.c_proj)
        hidden = linear(normalize(x), block.mlp.c_fc)
        if c.mlp_activation == "xiprelu":
            positive = F.softplus(block.mlp.act.raw_alpha_p).to(hidden.dtype)
            negative = F.softplus(block.mlp.act.raw_alpha_n).to(hidden.dtype)
            hidden = torch.where(hidden > 0, positive, negative) * hidden.square() + 0.5 * hidden
        else:
            hidden = hidden.relu().square()
        x = x + linear(hidden, block.mlp.c_proj)
        if step == len(plan) // 2:
            backout = x
    x = normalize(x - model.backout_lambda.to(x.dtype) * backout)
    logits = linear(x, model.lm_head)[..., :c.vocab_size].float()
    return 15 * (logits / 15).tanh()


@pytest.mark.parametrize("mgvr,diff,depth", list(product((False, True), repeat=3)))
@pytest.mark.parametrize("layout", [dict(n_loops=3), dict(n_loops=2, n_prelude=1, n_coda=1)])
@pytest.mark.parametrize("dtype,kv_heads", [(torch.float32, 4), (torch.bfloat16, 2)])
def test_combined_logits_and_gradients_match_independent_reference(monkeypatch, mgvr, diff, depth, layout, dtype, kv_heads):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled ablation executed")
    if not depth:
        monkeypatch.setattr(gpt, "mix_depth_values", forbidden)
    if not mgvr:
        monkeypatch.setattr(gpt.MGVR, "forward", forbidden)
    model = make_model(mgvr=mgvr, diff_v2=diff, depth_attention=depth,
                       depth_attention_stride=2 if depth else None, n_kv_head=kv_heads, **layout)
    model.window_sizes = [(2, 0)] * (model.eff_depth - 1) + [(16, 0)]
    reference = copy.deepcopy(model)
    tokens = torch.arange(12).view(2, 6)
    actual, expected = model(tokens), reference_forward(reference, tokens)
    tolerance = dict(rtol=1e-4, atol=1e-6) if dtype == torch.float32 else dict(rtol=0.06, atol=0.0005)
    torch.testing.assert_close(actual, expected, **tolerance)
    for logits in (actual, expected):
        F.cross_entropy(logits.reshape(-1, 64), tokens.roll(-1, 1).reshape(-1)).backward()
    for (name, actual_p), (_, reference_p) in zip(model.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(actual_p.grad, reference_p.grad, **tolerance, msg=lambda message: name + ": " + message)


@pytest.mark.parametrize("mgvr,depth", list(product((False, True), repeat=2)))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_repeated_block_zero_prefill_decode_and_cache_reset(monkeypatch, mgvr, depth, dtype):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", dtype)
    model = make_model(mgvr=mgvr, diff_v2=True, depth_attention=depth,
                       depth_attention_stride=1 if depth else None, n_loops=3)
    model.window_sizes = [(2, 0)] * (model.eff_depth - 1) + [(16, 0)]
    cache = KVCache(2, 2, 16, 8, model.eff_depth, "cpu", dtype)
    tolerance = dict(rtol=0.04, atol=0.004) if dtype == torch.bfloat16 else dict(rtol=1e-4, atol=1e-6)
    with torch.no_grad():
        for offset in (0, 12):
            cache.reset()
            tokens = torch.arange(offset, offset+12).view(2, 6)
            expected = model(tokens)
            outputs = [model(tokens[:, :3], kv_cache=cache)]
            outputs.extend(model(tokens[:, i:i+1], kv_cache=cache) for i in range(3, 6))
            torch.testing.assert_close(torch.cat(outputs, 1), expected, **tolerance)
            assert cache.get_pos() == 6


@pytest.mark.parametrize("stride", [1, 2, 99])
@pytest.mark.parametrize("prelude", [0, 1])
def test_looped_sources_and_anchor_are_execution_scoped(monkeypatch, stride, prelude):
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)
    model = make_model(mgvr=True, depth_attention=True, diff_v2=True,
                       depth_attention_stride=stride, n_loops=3, n_prelude=prelude)
    produced, anchors = [], []
    original_mix = gpt.mix_depth_values

    def capture(q, k, v, earlier):
        step = len(produced)
        assert len(earlier) == len(range(0, step, stride))
        for pair, index in zip(earlier, range(0, step, stride)):
            assert pair[0] is produced[index][0] and pair[1] is produced[index][1]
        result = original_mix(q, k, v, earlier)
        produced.append((k, result))
        return result

    monkeypatch.setattr(gpt, "mix_depth_values", capture)
    handles = [block.attn.mgvr.register_forward_pre_hook(lambda module, inputs: anchors.append(inputs[2]))
               for block in model.transformer.h if block.attn.mgvr is not None]
    try:
        previous_anchor = None
        for offset in (0, 12):
            produced.clear()
            anchors.clear()
            model(torch.arange(offset, offset+12).view(2, 6))
            assert len(produced) == model.eff_depth
            assert len(anchors) == sum(index > 0 for index in model.layer_plan)
            assert all(anchor is produced[0][1] for anchor in anchors)
            assert anchors[0] is not previous_anchor
            previous_anchor = anchors[0]
            assert anchors[0].grad_fn is not None
            assert len({id(k) for k, _ in produced}) == model.eff_depth
    finally:
        for handle in handles:
            handle.remove()
