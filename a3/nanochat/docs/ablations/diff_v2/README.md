# Differential Attention V2

Diff V2 is an optional attention ablation, independent of the existing MLP
activation ablation. `GPTConfig.diff_v2` defaults to `False`. The implementation
follows Microsoft's [reference code](https://github.com/microsoft/unilm/blob/master/Diff-Transformer/Diff-Transformer-V2/multihead_flashdiffv2.py)
and [explanation](https://huggingface.co/blog/microsoft/diff-attn-v2), adapted to
this checkout's existing QK normalization, RoPE, value embeddings and residuals.
The lambda projection uses the reference's default bias-free `nn.Linear` weight
distribution, explicitly applied after meta-device materialization. NanoChat's
custom `Linear` remains in use for dtype compatibility.

## How the code connects to the idea

1. **Ordinary attention retrieves information.** A token is represented by a
   vector `x`. A query describes what it is looking for, keys describe what past
   tokens contain, and values carry the information to retrieve. One attention
   head returns a weighted mixture of those values.
2. **Give each head a second query.** In `CausalSelfAttention.__init__`, `n_head`
   continues to mean the number of final output heads. `n_q_head` becomes twice
   that number only when Diff V2 is enabled. `c_q` learns the extra queries;
   `c_k` and `c_v` keep their existing sizes. This gives each token two learned
   ways to search the same information.
3. **Retrieve twice from shared keys and values.** The existing attention call
   accepts more query heads than K/V heads through grouped-query attention
   (GQA). Queries are grouped contiguously, so heads 0 and 1 share K/V, as do
   2 and 3, and so on. The existing positional encoding, QK normalization,
   1.2 scaling, causal mask, value embeddings and sliding windows still apply.
4. **Learn how much to subtract.** `lambda_proj(x)` produces one number per
   token and final head. `sigmoid` puts it between zero and one. For example,
   if the retrieved vectors are `[4, 2]` and `[2, 2]` and lambda is 0.5, the
   combined vector is `[4, 2] - 0.5 * [2, 2] = [3, 1]`.
5. **Combine adjacent partners.** Immediately after attention:

   ```python
   lam = torch.sigmoid(self.lambda_proj(x)).to(y).unsqueeze(-1)
   y = y[:, :, 0::2] - lam * y[:, :, 1::2]
   ```

   Common contributions can partially cancel. Both queries and the coefficient
   learn through the language-model loss; the second query is not assigned a
   noise label. Cancellation is a capability, not a guarantee of better BPB.
6. **Return to the existing model width.** For six heads of dimension 128, the
   internal 12 attention outputs become six differential outputs, then flatten
   back to 768 features. The existing `c_proj` and residual connection can
   consume them directly. `config.n_head` remains six; the K/V cache also keeps
   six heads. The subtraction is after both attention branches, so training,
   prompt prefill and token-by-token generation use the same rule.
7. **Initialize and optimize safely.** `GPT.init_weights()` explicitly
   initializes lambda weights after meta-device materialization using
   `kaiming_uniform_(weight, a=sqrt(5))`, equivalent to uniform bounds
   `+/-1/sqrt(n_embd)`. Initial coefficients vary across tokens and heads.
   All projections use NanoChat's `Linear`,
   keeping FP32 master weights while following activation dtype; `.to(y)` makes
   the coefficient match the attention output's dtype and device. The existing
   optimizer discovers the new matrices automatically, exactly once. Because
   `c_proj` starts at zero, query/lambda gradients initially vanish; they become
   nonzero after the output projection's first optimizer update.

The xIPReLU formula, coefficients, initialization and optimizer treatment are
unchanged. It operates in the MLP after attention and can be combined with Diff
V2, but combining them is a separate experiment from xIPReLU alone.

The lambda initialization matches the standalone Microsoft reference; the host
model's other initialization choices remain NanoChat's. Q/K/V (including the
added query rows) keep uniform std `1/sqrt(n_embd)` rather than the reference's
`1/sqrt(3*n_embd)`, and `c_proj` remains zero rather than randomly initialized.
Zero `c_proj` already makes the initial attention residual zero for any lambda;
it does not require zero lambda weights. QK normalization, 1.2 scaling, RoPE,
value embeddings, windows and residuals also remain NanoChat adaptations.
The reference module alone does not establish the full training system's
initialization policy.

This initialization revision changes no dimensions, parameter counts, compute
dtypes or checkpoint shapes. Existing checkpoints load their saved weights;
fresh runs consume extra random draws and may change other initialized values
and training metrics at the same seed.

## Toggle and checkpoint behavior

Append the flags below to the same training command and settings you already
use. Default tags shown are for depth 12, one loop, with no custom `--model-tag`.

| Experiment | Additional flags | Default checkpoint tag |
| --- | --- | --- |
| Baseline | none | `d12` |
| xIPReLU only | `--mlp-activation=xiprelu` | `d12-xiprelu` |
| Diff V2 only | `--diff-v2` | `d12-diff-v2` |
| Both | `--diff-v2 --mlp-activation=xiprelu` | `d12-xiprelu-diff-v2` |

Omit `--diff-v2` to turn it off; direct Python callers can set `diff_v2=False`.
Toggling chooses the architecture for a run, not a conversion of trained weights.
Old checkpoints missing the field default to disabled. Strict loading rejects
baseline/Diff V2 shape mismatches. Enabled checkpoint metadata saves the flag;
disabled model metadata omits it, preserving compatibility with the old loader.
SFT and RL inherit the architecture from the loaded checkpoint and preserve its
flag and distinct default tags when saving. A custom tag is an explicit path
override; use a distinct tag for each experiment.

For a matched training comparison, keep the existing tokenizer, data, seed,
batch size, context length, schedule and **explicit `--num-iterations`** the same.
The existing parameter/data-ratio or FLOPs-based stopping rules would otherwise
change the number of updates because Diff V2 adds parameters and compute.
No real baseline checkpoints, datasets or historical results were modified.
There is no Modal Part 2 runner, LeRoPE or NormFormer implementation in this
checkout; no external runner was edited or launched.

## Changed files

| File | Purpose |
| --- | --- |
| `nanochat/gpt.py` | Default-off flag, expanded queries, lambda projection, adjacent subtraction, initialization and attention FLOPs |
| `scripts/base_train.py` | CLI flag, config persistence and distinct default tag |
| `scripts/chat_sft.py` | Preserve the flag in its manually assembled checkpoint config and default tag |
| `scripts/chat_rl.py` | Preserve the flag and default tag when saving |
| `tests/ablations/test_diff_v2.py` | Arithmetic, baseline/xIPReLU regression, dtype, training, cache, checkpoint, FLOPs and compile checks |
| `tests/ablations/test_diff_v2_smoke.py` | Real trainer compilation, checkpoint saving and resume, reusing existing offline smoke data |
| `docs/ablations/diff_v2/README.md` | Explanation, usage and verification report |

The attention backend, optimizer implementation and existing tests are unchanged.
The production Python patch adds about 30 lines. No custom kernels or new runtime
dependencies are needed. Pytest was installed in the local environment to run tests.

## Verification of reference-aligned lambda initialization

On November 2, 2026, the revised initialization passed the following checks on
CPU SDPA with PyTorch 2.9.1:

- Diff V2 and Depth-Attention unit/integration checks: **87 passed, 6 skipped**
  (`tests/ablations/test_diff_v2.py` and `tests/ablations/test_depth_attention.py`). Skips require CUDA.
- All four real compiled trainer/checkpoint-resume smoke cases passed
  (`tests/ablations/test_diff_v2_smoke.py` and `tests/ablations/test_depth_attention_smoke.py`).
  Compilation uses the CPU `aot_eager` backend, not GPU Inductor.
- Shape/dtype coverage now includes one and six output heads of dimension 128,
  as well as tiny GQA heads, each in FP32, BF16 and FP16. Master projection
  weights remain FP32; outputs and finite gradients retain the expected dtypes.
- Initialization checks verify reference bounds, random weights of both signs,
  and coefficient variation across tokens and heads. Disabled Diff V2 remains
  bitwise equivalent to the pre-Diff-V2 baseline.
- Depth-Attention's historical comparison loads identical saved weights for
  Diff V2 cases, including the earlier zero-lambda weights. This tests checkpoint
  compatibility and disabled depth behavior independently of the intentional
  initialization change. Non-Diff-V2 cases still compare initial weights exactly.

No Modal/CUDA execution or production-quality benchmark was performed. The
historical smoke BPBs below belong to the earlier initialization and should not
be reported as measurements of this revision.

## Historical verification on November 2, 2026 (zero-lambda initialization)

The results in this section were recorded before the reference-aligned lambda
initialization revision. They describe the earlier implementation, not a fresh
verification or benchmark of the revised initialization.

Environment: macOS, Python 3.10.22, PyTorch 2.9.1, CPU SDPA. No CUDA GPU.

| Focused checks | Cases | Result |
| --- | ---: | --- |
| Disabled baseline and xIPReLU against pre-change commit: initialization, shapes, parameter counts, FLOPs, logits, gradients, optimizer groups and one update | 2 | Exact equality |
| PicoChat projection shapes and meta-to-CPU initialization, lambda starts at 0.5 | 2 | Passed |
| Adjacent shared-K/V pairing and learned subtraction, including 12 query heads / 6 KV heads | 4 | Passed |
| FP32, BF16 and FP16 attention: output dtype, FP32 master weights, finite/nonzero gradients | 3 | Passed on CPU |
| Full GPT causality and two actual Muon/AdamW updates, with/without xIPReLU | 2 | Passed |
| Full forward vs prefill plus decoding: full/short windows, one/two loops, both activations, FP32/BF16 | 16 | Passed on CPU |
| Strict checkpoint reload, old config default, reject incompatible architecture | 4 | Passed |
| FLOPs and parameter accounting, one/two loops; unchanged K/V storage | 2 | Passed |
| Full-graph model compilation and backward with `aot_eager`, both activations | 2 | Passed |
| Real compiled trainer: three updates, checkpoint and optimizer save, one resumed update, both activations | 2 | Passed |
| CUDA BF16 training/cache and FA3/SDPA comparison, full/short windows | 4 | Skipped: hardware unavailable |

Focused total: **39 passed, 4 skipped**. Full suite: **110 passed, 18 skipped,
1 failed**. The unrelated failure is `tests/test_execution.py::test_memory_limit`:
`nanochat/execution.py` explicitly bypasses resource limits on macOS, while the
test expects a memory error. That module is identical to the pre-change commit.
The existing baseline, xIPReLU and looped xIPReLU compiled trainer/resume tests
all passed.

Synthetic smoke results (two layers, width 32, 16-token sequences, 32 tokens per
update; these BPBs demonstrate execution, not model quality):

| Model | Parameters | BPB at step 3 | BPB after resume to step 4 |
| --- | ---: | ---: | ---: |
| Diff V2 + ReLU² | 57,678 | 8.013004 | 7.999906 |
| Diff V2 + xIPReLU | 57,682 | 8.012996 | 7.999894 |

For depth 12, width 768 and six effective heads, Diff V2 adds **7,133,184**
parameters regardless of tokenizer size. Structural matmul counting includes
the extra Q and lambda projections once; only the separate attention term is
doubled in training, prefill and decoding FLOPs. These are theoretical estimates,
following the existing estimator's conventions, not measured hardware work.

No full pretraining, CORE evaluation, Modal job, CUDA BF16, FA3, GPU Inductor
compilation, GPU step-time/throughput, runtime/cost or peak-memory measurement
was performed. Production-scale validation BPB and MFU are therefore unmeasured.
Existing trainer logs already report backend, parameters, validation BPB, CORE
when requested, step time, tokens/sec, estimated MFU and peak CUDA memory.
GPU billing/runtime must be recorded from the eventual runner. Local compile
checks use `aot_eager`; GPU kernel compilation still requires a short GPU smoke.

Reproduce the focused checks or all tests using the local environment:

```sh
.venv/bin/python -m pytest tests/ablations/test_diff_v2.py tests/ablations/test_diff_v2_smoke.py -v
.venv/bin/python -m pytest tests -q
```
