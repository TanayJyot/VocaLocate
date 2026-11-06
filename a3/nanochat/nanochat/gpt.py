"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP (optional xIPReLU ablation)
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
- optional looped (weight-shared recurrent depth) execution: prelude -> core x n_loops -> coda
"""

from functools import partial
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn
from nanochat.depth_attention import depth_source_layers, mix_depth_values

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    mlp_activation: str = "relu2"
    # Looped / recurrent-depth execution (all defaults = ordinary GPT, fully backward compatible).
    # The n_layer unique blocks are split into [prelude | core | coda]:
    #   prelude = first n_prelude blocks, run once
    #   core    = the middle n_layer - n_prelude - n_coda blocks, run n_loops times with SHARED weights
    #   coda    = last n_coda blocks, run once
    # Effective depth (layers actually executed) = n_prelude + n_core * n_loops + n_coda.
    n_loops: int = 1
    n_prelude: int = 0
    n_coda: int = 0
    # loop_embed: a learned vector per core pass, added to the residual stream at the start of that pass so the
    # shared blocks can tell which iteration they are in (zero init = starts as the plain looped model).
    loop_embed: bool = False
    # Random loop count: training draws the loop count uniformly from [n_loops_min, n_loops_max] each step
    # (0 = same as n_loops); n_loops stays the default used for eval and inference. Per-step scalars are sized for
    # n_passes = max(n_loops, n_loops_max) core passes; running more passes than that reuses the last pass's.
    n_loops_min: int = 0
    n_loops_max: int = 0
    # Differential Attention V2: two adjacent query heads per effective output head.
    diff_v2: bool = False
    depth_attention: bool = False
    depth_attention_stride: int | None = None
    mgvr: bool = False
    # Exclusive Self-Attention (arXiv 2603.09078), learnable form of modded-nanogpt record #82: each head removes
    # a fraction tanh(alpha) of its output's component along the token's own value vector (alpha zero-init).
    xsa: bool = False
    # ssa: Selective Self-Attention as OpenNeuro ran it (after Zhang et al., arXiv 2411.12892): queries are scaled by a
    # per-head position-dependent temperature T(pos) = 1 + exp(-exp(beta)) * log(pos + 1), beta init 1.5 (alpha ~ 0.011).
    ssa: bool = False
    # MUDD-lite (modded-nanogpt record #81): token-dependent dense connections on the last two layers only.
    # Layer n-2 mixes {x0, mid-layer residual, itself} into the residual and into layer n-1's values;
    # layer n-1 mixes {x0, mid-layer residual, layer n-2's output} into the final residual. Zero-init = identity.
    mudd: bool = False

    @property
    def effective_depth_attention_stride(self):
        stride = self.depth_attention_stride
        if stride is None:
            return max(1, self.n_layer // 2)
        if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
            raise ValueError("depth_attention_stride must be a positive integer")
        return stride

    @property
    def n_core(self):
        return self.n_layer - self.n_prelude - self.n_coda

    @property
    def effective_depth(self):
        return self.n_prelude + self.n_core * self.n_loops + self.n_coda

    @property
    def loop_range(self):
        """(min, max) loop count used in training; (n_loops, n_loops) unless random loop counts are on."""
        lo, hi = self.n_loops_min or self.n_loops, self.n_loops_max or self.n_loops
        assert 1 <= lo <= self.n_loops <= hi, f"need 1 <= n_loops_min <= n_loops <= n_loops_max, got {lo}, {self.n_loops}, {hi}"
        return lo, hi

    @property
    def n_passes(self):
        return self.loop_range[1]

    @property
    def n_step_slots(self):
        """Number of per-step scalar slots (== effective_depth unless n_loops_max > n_loops)."""
        return self.n_prelude + self.n_core * self.n_passes + self.n_coda

    def layer_plan(self, n_loops=None):
        """Block index used at each executed step, e.g. n_layer=4, prelude=1, coda=1, loops=3 -> [0,1,2,1,2,1,2,3]."""
        n_loops = self.n_loops if n_loops is None else n_loops
        assert n_loops >= 1 and self.n_prelude >= 0 and self.n_coda >= 0
        assert self.n_core >= 1, f"need at least one core block: n_layer={self.n_layer}, prelude={self.n_prelude}, coda={self.n_coda}"
        core = list(range(self.n_prelude, self.n_prelude + self.n_core))
        return list(range(self.n_prelude)) + core * n_loops + list(range(self.n_layer - self.n_coda, self.n_layer))

    def step_slots(self, n_loops=None):
        """Per-step scalar slot used at each executed step. Equals range(effective_depth) for a fixed loop count;
        core passes beyond n_passes reuse the last pass's slots, and the coda always keeps its own."""
        n_loops = self.n_loops if n_loops is None else n_loops
        last = self.n_passes - 1
        core = [self.n_prelude + min(j, last) * self.n_core + k for j in range(n_loops) for k in range(self.n_core)]
        coda_start = self.n_prelude + self.n_core * self.n_passes
        return list(range(self.n_prelude)) + core + list(range(coda_start, coda_start + self.n_coda))


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class MGVR(nn.Module):
    """Full cross-head Value residual; zero initialization preserves NanoChat Values.

    Unlike the reference's primary Lambda=0.5*I initialization, both parameters
    start at zero. Real initialization happens in GPT.init_weights after to_empty.
    """
    def __init__(self, n_embd, n_kv_head):
        super().__init__()
        self.n_kv_head = n_kv_head
        self.scale = n_embd ** -0.5
        self.w_z = nn.Parameter(torch.empty(n_kv_head + n_kv_head**2, n_embd, dtype=torch.float32))
        self.lam = nn.Parameter(torch.empty(n_kv_head, n_kv_head, dtype=torch.float32))

    def components(self, x, v0):
        G = self.n_kv_head
        z = F.linear(x.float(), self.w_z) * self.scale
        gate = 2 * torch.sigmoid(z[..., :G])
        mix = self.lam + torch.tanh(z[..., G:].unflatten(-1, (G, G)))
        added = torch.einsum("btgh,bthd->btgd", mix, v0.float())
        return gate, mix, added

    def forward(self, x, v, v0):
        gate, _, added = self.components(x, v0)
        return (gate.unsqueeze(-1) * v.float() + added).to(v.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.diff_v2 = config.diff_v2
        self.depth_attention = config.depth_attention
        if self.depth_attention:
            self.depth_stride = config.effective_depth_attention_stride
            self.depth_count = config.effective_depth
        self.n_q_head = self.n_head * (2 if self.diff_v2 else 1)
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_q_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.lambda_proj = Linear(self.n_embd, self.n_head, bias=False) if self.diff_v2 else None
        self.mgvr = MGVR(self.n_embd, self.n_kv_head) if config.mgvr and layer_idx > 0 else None
        self.xsa_alpha = nn.Parameter(torch.zeros(self.n_head)) if config.xsa else None # fake init, real init in init_weights()
        self.ssa_beta = nn.Parameter(torch.zeros(self.n_head)) if config.ssa else None # fake init, real init in init_weights()

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cache_idx, depth_kv=None, first_layer_v=None, v_extra=None):
        # cache_idx = executed step (virtual layer) index. With looping, one block runs several times and
        # each execution needs its own KV cache slot (its keys/values differ per iteration).
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_q_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # MUDD-lite: values mixed from earlier hidden states by the previous layer
        if v_extra is not None:
            v = v + v_extra

        if first_layer_v is not None:
            if cache_idx == 0:
                first_layer_v.append(v) # temporary, post-VE anchor; keep its autograd graph
            elif self.mgvr is not None:
                v = self.mgvr(x, v, first_layer_v[0])

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        if self.ssa_beta is not None:
            # After QK norm, which would otherwise erase the scale. Per-row positions, so decoding matches training.
            offset = 0 if kv_cache is None else kv_cache.cache_seqlens.long()[:, None]
            positions = offset + torch.arange(T, device=x.device)[None, :]
            alpha = torch.exp(-torch.exp(self.ssa_beta)).to(q.dtype).view(1, 1, self.n_head, 1)
            q = q * (1 + alpha * torch.log1p(positions.to(q.dtype))[:, :, None, None])

        if self.depth_attention:
            if kv_cache is None:
                if depth_kv is None:
                    raise ValueError("Depth-Attention training requires per-forward source references")
                v = mix_depth_values(q, k, v, depth_kv)
                if cache_idx % self.depth_stride == 0 and cache_idx + 1 < self.depth_count:
                    depth_kv.append((k, v))
            else:
                # Earlier layers have already written these *new* token positions.
                # Gather per row: never include the cached token history as depth sources.
                positions = kv_cache.cache_seqlens.long()[:, None] + torch.arange(T, device=x.device)
                rows = torch.arange(B, device=x.device)[:, None]
                sources = []
                for source in depth_source_layers(cache_idx, self.depth_stride)[:-1]:
                    source_k, source_v = kv_cache.get_layer_cache(source)
                    sources.append((source_k[rows, positions], source_v[rows, positions]))
                v = mix_depth_values(q, k, v, sources)

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(cache_idx)
            if self.depth_attention and not torch.equal(kv_cache.cache_seqlens, kv_cache.cache_seqlens[:1].expand(B)):
                # The existing SDPA wrapper assumes equal offsets. Row calls also work
                # with FA3, preserve its API, and write into the original cache views.
                y = torch.cat([
                    flash_attn.flash_attn_with_kvcache(
                        q[row:row+1], k_cache[row:row+1], v_cache[row:row+1],
                        k=k[row:row+1], v=v[row:row+1],
                        cache_seqlens=kv_cache.cache_seqlens[row:row+1],
                        causal=True, window_size=window_size,
                    ) for row in range(B)
                ], dim=0)
            else:
                y = flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
            # Advance position after last layer processes
            if cache_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        if self.xsa_alpha is not None:
            # XSA: drop a per-head fraction of the output along the token's own (post-mixing) value direction
            vn = F.normalize(v, dim=-1, eps=1e-4).repeat_interleave(self.n_head // self.n_kv_head, dim=2)
            alpha = torch.tanh(self.xsa_alpha).to(y.dtype).view(1, 1, self.n_head, 1)
            y = y - alpha * (y * vn).sum(-1, keepdim=True) * vn

        if self.diff_v2:
            # GQA groups are contiguous: adjacent partners always share the same K/V.
            # Apply after either backend/path so training, prefill and decoding agree.
            lam = torch.sigmoid(self.lambda_proj(x)).to(y).unsqueeze(-1)
            y = y[:, :, 0::2] - lam * y[:, :, 1::2]

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class XIPReLU(nn.Module):
    """Piecewise quadratic with two learned positive coefficients per MLP.

    xIPReLU(x) = softplus(raw_alpha_{p,n}) * x^2 + 0.5*x.
    See Huang & Schlag, https://arxiv.org/html/2411.13010v3#A6.
    """
    def __init__(self):
        super().__init__()
        self.raw_alpha_p = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.raw_alpha_n = nn.Parameter(torch.empty((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        # Inverse softplus: the effective coefficients both start at 0.8.
        # GPT.init_weights repeats this after meta -> to_empty materialization.
        raw_init = math.log(math.expm1(0.8))
        nn.init.constant_(self.raw_alpha_p, raw_init)
        nn.init.constant_(self.raw_alpha_n, raw_init)

    def forward(self, x):
        # Keep master scalars in fp32; preserve the activation's compute dtype.
        alpha_p = F.softplus(self.raw_alpha_p).to(dtype=x.dtype)
        alpha_n = F.softplus(self.raw_alpha_n).to(dtype=x.dtype)
        alpha = torch.where(x > 0, alpha_p, alpha_n)
        return alpha * x.square() + 0.5 * x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.mlp_activation not in {"relu2", "xiprelu"}:
            raise ValueError(f"Unsupported MLP activation: {config.mlp_activation!r}")
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act = XIPReLU() if config.mlp_activation == "xiprelu" else None

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() if self.act is None else self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache, cache_idx, depth_kv=None, first_layer_v=None, v_extra=None):
        if first_layer_v is not None:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cache_idx, depth_kv, first_layer_v)
        elif self.attn.depth_attention:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cache_idx, depth_kv)
        elif v_extra is not None:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cache_idx, v_extra=v_extra)
        else:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache, cache_idx)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        variable_loops = config.loop_range[0] < config.loop_range[1]
        assert not ((variable_loops or config.loop_embed) and (config.depth_attention or config.mgvr)), \
            "loop_embed / random loop counts are not supported together with depth_attention or mgvr"
        assert not (config.xsa and config.diff_v2), "xsa is not supported together with diff_v2"
        assert not (config.ssa and (config.diff_v2 or config.depth_attention)), "ssa is not supported together with diff_v2 or depth_attention"
        assert not (config.mudd and (config.n_loops > 1 or config.depth_attention or config.mgvr or config.n_layer < 4)), \
            "mudd needs a plain (non-looped) model of depth >= 4 without depth_attention or mgvr"
        self.slot_windows = self._compute_window_sizes(config) # one per per-step scalar slot
        self._plans = {n: self._execution_plan(n) for n in range(config.loop_range[0], config.loop_range[1] + 1)}
        self.layer_plan = config.layer_plan() # block index executed at each step (length = effective depth)
        self.eff_depth = len(self.layer_plan)
        self.window_sizes = self._plans[config.n_loops]["windows"] # one per executed step
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_step_slots))   # (per executed step slot, not per unique block) fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_step_slots))     # (per executed step slot) fake init, real init in init_weights()
        # Per-loop embeddings: one vector per core pass (None unless loop_embed)
        self.loop_embeds = nn.Parameter(torch.zeros(config.n_passes, config.n_embd)) if config.loop_embed else None
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # MUDD-lite: per MUDD layer (n-2, n-1), a small MLP on the hidden state gives per-token weights
        # (2 channels: values, residual) x (3 sources). Layer n-1 uses only the residual channel.
        if config.mudd:
            assert config.n_kv_head * (config.n_embd // config.n_head) == config.n_embd, "mudd values need kv_dim == n_embd"
            self.mudd_w1 = nn.Parameter(torch.empty(2, 64, config.n_embd))
            self.mudd_w2 = nn.Parameter(torch.empty(2, 64, 2, 3))
            self.mudd_b = nn.Parameter(torch.empty(2, 2, 3))
        else:
            self.mudd_w1 = self.mudd_w2 = self.mudd_b = None
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            if block.attn.mgvr is not None:
                nn.init.zeros_(block.attn.mgvr.w_z)
                nn.init.zeros_(block.attn.mgvr.lam)
            if block.attn.lambda_proj is not None:
                # Match the default bias-free nn.Linear in Microsoft's Diff V2 reference.
                torch.nn.init.kaiming_uniform_(block.attn.lambda_proj.weight, a=math.sqrt(5))
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.mlp.act is not None:
                block.mlp.act.reset_parameters()
            if block.attn.xsa_alpha is not None:
                torch.nn.init.zeros_(block.attn.xsa_alpha) # tanh(0) = 0: starts as plain attention
            if block.attn.ssa_beta is not None:
                torch.nn.init.constant_(block.attn.ssa_beta, 1.5) # alpha ~ 0.011: T(2047) ~ 1.09, close to plain attention

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_step_slots # schedules are over executed step slots (== effective depth for fixed loops)
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        # (with looping this doubles as "input injection" into every iteration, as in recurrent-depth models)
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
        if self.loop_embeds is not None:
            torch.nn.init.zeros_(self.loop_embeds)

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # MUDD last, so every other weight draws the same random numbers as the baseline at the same seed.
        # As in the speedrun: kaiming-uniform hidden layer, zero output weights and biases => starts as identity
        if self.mudd_w1 is not None:
            for j in range(2):
                torch.nn.init.kaiming_uniform_(self.mudd_w1[j], a=math.sqrt(5))
            torch.nn.init.zeros_(self.mudd_w2)
            torch.nn.init.zeros_(self.mudd_b)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across step slots (== executed layers for a fixed loop count)
        window_sizes = []
        for layer_idx in range(config.n_step_slots):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _execution_plan(self, n_loops):
        """What runs at each executed step for a given loop count: block index, scalar slot, window size, and
        (at the first step of each core pass) which loop embedding to add."""
        config = self.config
        blocks, slots = config.layer_plan(n_loops), config.step_slots(n_loops)
        windows = [self.slot_windows[s] for s in slots]
        windows[-1] = (config.sequence_len, 0) # final executed layer always gets full context
        pass_starts = {config.n_prelude + j * config.n_core: min(j, config.n_passes - 1) for j in range(n_loops)}
        return dict(blocks=blocks, slots=slots, windows=windows, pass_starts=pass_starts)

    def execution_plan(self, n_loops=None):
        n_loops = self.config.n_loops if n_loops is None else n_loops
        plan = self._plans[n_loops] if n_loops in self._plans else self._execution_plan(n_loops)
        if n_loops == self.config.n_loops: # the default loop count follows self.window_sizes, which callers may override
            plan = {**plan, "windows": self.window_sizes}
        return plan

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self, n_loops=None):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        With random loop counts (n_loops=None and n_loops_min < n_loops_max) this is the expected training cost,
        averaged over the uniformly sampled loop counts.
        """
        lo, hi = self.config.loop_range
        if n_loops is None and lo < hi:
            return sum(self.estimate_flops(n) for n in range(lo, hi + 1)) / (hi - lo + 1)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        h *= 2 if self.config.diff_v2 else 1 # attention runs before differential pairs are reduced
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.execution_plan(n_loops)["windows"]:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params(n_loops) + attn_flops
        if self.config.depth_attention:
            num_flops_per_token += 3 * self.estimate_depth_attention_flops()
        if self.config.mgvr:
            num_flops_per_token += 3 * self.estimate_mgvr_flops()
        return num_flops_per_token

    def estimate_mgvr_flops(self):
        """Approximate forward FLOPs/token beyond W_z: cross-head mixing + gating.

        W_z is counted in num_matmul_params; omits sigmoid/tanh and memory traffic.
        """
        G, D = self.config.n_kv_head, self.config.n_embd // self.config.n_head
        uses = sum(self.transformer.h[b].attn.mgvr is not None for b in self.layer_plan)
        return uses * (2 * G * G * D + 2 * G * D)

    def estimate_depth_attention_flops(self):
        """Approximate forward FLOPs/token: depth dots + weighted value sums.

        Omits small softmax/GQA reductions and memory traffic; training uses 3x.
        Source selection follows executed depth for weight-shared looped models.
        """
        if not self.config.depth_attention:
            return 0
        stride = self.config.effective_depth_attention_stride
        candidates = sum(len(depth_source_layers(i, stride)) for i in range(1, self.eff_depth))
        return 4 * self.config.n_kv_head * (self.config.n_embd // self.config.n_head) * candidates

    def num_matmul_params(self, n_loops=None):
        """
        The number of (execution-weighted) parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally from
        Linear weights and MGVR's explicit W_z, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        # With looping a shared block executes several times per token, so its params count once per execution.
        uses = {}
        for b in self.execution_plan(n_loops)["blocks"]:
            uses[b] = uses.get(b, 0) + 1
        matmul_params = 0
        for b, block in enumerate(self.transformer.h):
            matmul_params += uses[b] * sum(m.weight.numel() for m in block.modules() if isinstance(m, Linear))
            if block.attn.mgvr is not None:
                matmul_params += uses[b] * block.attn.mgvr.w_z.numel()
        # Linears outside the block stack (lm_head, smear_gate) run once
        block_linears = {id(m) for block in self.transformer.h for m in block.modules()}
        matmul_params += sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear) and id(m) not in block_linears)
        if self.mudd_w1 is not None: # MUDD's weight MLPs run once each (the source-weighted sums are elementwise)
            matmul_params += self.mudd_w1.numel() + self.mudd_w2.numel()
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        h *= 2 if self.config.diff_v2 else 1
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        if self.config.depth_attention:
            decode_flops += self.estimate_depth_attention_flops()
        if self.config.mgvr:
            decode_flops += self.estimate_mgvr_flops()
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        h *= 2 if self.config.diff_v2 else 1
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        if self.config.depth_attention:
            prefill_flops += self.estimate_depth_attention_flops() * num_tokens
        if self.config.mgvr:
            prefill_flops += self.estimate_mgvr_flops() * num_tokens
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.eff_depth * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters() if p.ndim == 2)
        mgvr = sum(p.numel() for b in self.transformer.h if b.attn.mgvr is not None for p in b.attn.mgvr.parameters())
        transformer_matrices -= mgvr # preserve the trainer's original scaling-law budget/schedule
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        scalars += sum(p.numel() for p in self.transformer.h.parameters() if p.ndim < 2)
        scalars += self.loop_embeds.numel() if self.loop_embeds is not None else 0
        # MUDD sits outside transformer_matrices so it doesn't change the trainer's scaling-law budget/schedule
        mudd = sum(p.numel() for p in (self.mudd_w1, self.mudd_w2, self.mudd_b) if p is not None)
        total =wte + value_embeds + lm_head + transformer_matrices + scalars + mgvr + mudd
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            **({'mgvr': mgvr} if self.config.mgvr else {}),
            **({'mudd': mudd} if self.config.mudd else {}),
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        mgvr_params = [p for b in self.transformer.h if b.attn.mgvr is not None for p in b.attn.mgvr.parameters()]
        mgvr_ids = {id(p) for p in mgvr_params}
        matrix_params = [p for p in self.transformer.h.parameters() if p.ndim == 2 and id(p) not in mgvr_ids]
        activation_params = [p for p in self.transformer.h.parameters() if p.ndim < 2]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        loop_embed_params = [self.loop_embeds] if self.loop_embeds is not None else []
        mudd_params = [p for p in (self.mudd_w1, self.mudd_w2, self.mudd_b) if p is not None]
        assert len(list(self.parameters())) == len(matrix_params) + len(activation_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(mgvr_params) + len(loop_embed_params) + len(mudd_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if activation_params:
            # Use the residual-scalar LR scale, without weight decay on the
            # unconstrained raw coefficients. Muon only accepts matrices.
            param_groups.append(dict(kind='adamw', params=activation_params, lr=scalar_lr * 0.01,
                                     betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if loop_embed_params:
            # Added to the residual stream like an embedding: a tenth of the token-embedding LR, no weight decay
            param_groups.append(dict(kind='adamw', params=loop_embed_params, lr=embedding_lr * dmodel_lr_scale * 0.1,
                                     betas=(0.8, 0.995), eps=1e-10, weight_decay=0.0))
        if mgvr_params:
            param_groups.append(dict(kind='adamw', params=mgvr_params, lr=unembedding_lr * dmodel_lr_scale,
                                     betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0))
        if mudd_params:
            # Small gating MLP like MGVR's W_z: AdamW at the unembedding LR, betas as in the speedrun, no weight decay
            param_groups.append(dict(kind='adamw', params=mudd_params, lr=unembedding_lr * dmodel_lr_scale,
                                     betas=(0.9, 0.99), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    @torch.no_grad()
    def activation_stats(self):
        """Effective coefficients for validation logging and checkpoint metadata."""
        stats = {}
        for i, block in enumerate(self.transformer.h):
            if block.mlp.act is not None:
                stats[f"xiprelu/layer_{i}/alpha_p"] = F.softplus(block.mlp.act.raw_alpha_p).item()
                stats[f"xiprelu/layer_{i}/alpha_n"] = F.softplus(block.mlp.act.raw_alpha_n).item()
        return stats

    @torch.no_grad()
    def mgvr_stats(self, idx):
        """Validation-only eager probe; aggregate on device and transfer once."""
        rows, handles = [], []
        def collect(module, inputs, output):
            x, v, v0 = inputs
            gate, mix, added = module.components(x, v0)
            offdiag = mix - torch.diag_embed(mix.diagonal(dim1=-2, dim2=-1))
            rows.append(torch.stack((gate.mean(), added.norm() / v.float().norm().clamp_min(1e-12),
                                     mix.norm(dim=(-2, -1)).mean(), offdiag.norm(dim=(-2, -1)).mean())))
        try:
            for block in self.transformer.h:
                if block.attn.mgvr is not None:
                    handles.append(block.attn.mgvr.register_forward_hook(collect))
            if handles:
                self(idx)
        finally:
            for handle in handles:
                handle.remove()
        names = ("gate_mean", "added_to_current_norm", "mix_norm", "offdiagonal_mix_norm")
        return dict(zip(names, torch.stack(rows).mean(0).cpu().tolist())) if rows else {}

    def _mudd_weights(self, x, j):
        """Per-token MUDD weights (B, T, 2 channels [values, residual], 3 sources) for MUDD layer j."""
        h = F.gelu(F.linear(norm(x), self.mudd_w1[j].to(x.dtype)))
        m = torch.einsum("btd,dcl->btcl", h, self.mudd_w2[j].to(x.dtype)) + self.mudd_b[j].to(x.dtype)
        return m * (0.2 / math.sqrt(3)) # speedrun's scale: biases live in the pre-scaled domain

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', n_loops=None):
        """n_loops overrides config.n_loops for this call (random-loop training, loop-count sweeps at eval)."""
        B, T = idx.size()
        plan = self.execution_plan(n_loops)
        assert kv_cache is None or len(plan["blocks"]) == self.eff_depth, "the KV cache has one slot per step of the default loop count"

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
        if self.config.depth_attention and kv_cache is not None:
            positions = kv_cache.cache_seqlens.long()[:, None] + torch.arange(T, device=idx.device)
            cos_sin = self.cos[0, positions], self.sin[0, positions]

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                first = x[:, :1]
                if self.config.depth_attention and x_pre_smear is not None:
                    # Continue the existing smear across a multi-token chunk boundary.
                    first_gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(first[..., :24]))
                    first = first + first_gate * x_pre_smear
                x = torch.cat([first, x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        backout_layer = len(plan["blocks"]) // 2  # cache at halfway point of the executed steps
        x_backout = None
        depth_kv = [] if self.config.depth_attention and kv_cache is None else None
        first_layer_v = [] if self.config.mgvr else None
        mudd_layer = len(plan["blocks"]) - 2 if self.mudd_w1 is not None else None # first of the two MUDD layers
        v_extra, x_mudd = None, None
        # i = executed step (KV cache slot), b = which (possibly shared) block runs, s = per-step scalar slot
        for i, (b, s, window) in enumerate(zip(plan["blocks"], plan["slots"], plan["windows"])):
            x = self.resid_lambdas[s] * x + self.x0_lambdas[s] * x0
            if self.loop_embeds is not None and i in plan["pass_starts"]:
                x = x + self.loop_embeds[plan["pass_starts"][i]].to(x.dtype)
            ve = self.value_embeds[str(b)](idx).to(x.dtype) if str(b) in self.value_embeds else None
            if self.config.mgvr:
                x = self.transformer.h[b](x, ve, cos_sin, window, kv_cache, i, depth_kv, first_layer_v)
            elif self.config.depth_attention:
                x = self.transformer.h[b](x, ve, cos_sin, window, kv_cache, i, depth_kv)
            elif v_extra is not None:
                x = self.transformer.h[b](x, ve, cos_sin, window, kv_cache, i, v_extra=v_extra)
                v_extra = None
            else:
                x = self.transformer.h[b](x, ve, cos_sin, window, kv_cache, i)
            if i == backout_layer:
                x_backout = x
            if i == mudd_layer:
                # values for the last layer and a residual update, from {x0, mid-layer residual, this layer's output}
                m = self._mudd_weights(x, 0)
                mv, mr = m[:, :, 0], m[:, :, 1]
                v_extra = mv[..., 0:1] * x0 + mv[..., 1:2] * x_backout + mv[..., 2:3] * x
                v_extra = v_extra.view(B, T, self.config.n_kv_head, -1)
                x_mudd = x
                x = (1 + mr[..., 2:3]) * x + mr[..., 0:1] * x0 + mr[..., 1:2] * x_backout
        if mudd_layer is not None:
            # after the last layer: residual update from {x0, mid-layer residual, layer n-2's output}
            mr = self._mudd_weights(x, 1)[:, :, 1]
            x = x + mr[..., 0:1] * x0 + mr[..., 1:2] * x_backout + mr[..., 2:3] * x_mudd
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
