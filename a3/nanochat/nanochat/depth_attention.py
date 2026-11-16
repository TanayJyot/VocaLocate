"""Parameter-free, same-token cross-layer value mixing (opt-in GPT ablation)."""
import math

import torch


def depth_source_layers(layer_idx, stride):
    """Zero-indexed executed depths, including the current layer exactly once."""
    return (*range(0, layer_idx, stride), layer_idx)


def mix_depth_values(q, k, v, earlier_kv):
    """Mix (B,T,Hkv,D) values; only the source-depth axis gets a softmax.

    Q can have more heads than K/V. Average contiguous KV-sharing query
    groups for scoring, including Diff V2's paired queries when combined.
    Earlier values are already depth-mixed and remain attached to autograd.
    """
    if not earlier_kv:
        return v
    batch, time, kv_heads, dim = k.shape
    query = q.float().reshape(batch, time, kv_heads, -1, dim).mean(dim=3)
    sources = (*earlier_kv, (k, v))
    scores = torch.stack([(query * key.float()).sum(-1) for key, _ in sources], dim=-1)
    weights = (scores / math.sqrt(dim)).softmax(dim=-1)
    # Accumulate in FP32 as well as scoring/softmax; no global dtype changes.
    mixed = sum(weights[..., i, None] * value.float() for i, (_, value) in enumerate(sources))
    return mixed.to(v.dtype)
