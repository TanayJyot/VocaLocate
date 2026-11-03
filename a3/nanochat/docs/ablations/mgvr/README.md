# Optional MGVR ablation

Enable with `GPTConfig(mgvr=True)` or append `--mgvr` to the existing
`python -m scripts.base_train` command. Omit it to keep the existing model.
The default checkpoint tag becomes `d12-mgvr`; combined flags retain their
existing suffixes and append `-mgvr`. SFT/RL retain the flag when loading and
saving a pretrained model and append the same suffix to their output tags.

The [official MGVR implementation](https://github.com/empero-org/mgvr)
defines per-token, per-KV-head gates and a full dynamic cross-head mixing
matrix. This implementation uses that formula after NanoChat's existing gated
Value Embeddings:

```text
z = W_z * normalized_attention_input / sqrt(d_model)
gate = 2 * sigmoid(z_gate)
mix = Lambda + tanh(z_mix)
modified_value[g] = gate[g] * current_value[g] + sum_h mix[g,h] * first_value[h]
```

`W_z=0` and `Lambda=0` give `gate=1`, `mix=0`, and **exactly the original Value**.
This deliberately differs from the reference's primary `Lambda=0.5*I`
initialization. It consumes no random numbers, so all original weights also
match a baseline initialized with the same seed. Both new matrices remain
trainable; the baseline's zero attention output projections initially block
their gradients, and the subsequent optimizer steps activate them.

Layer 0's post-VE Value is a local, graph-connected tensor, captured once per
forward call. Blocks 1 onward receive the same anchor. No model state or second
KV cache holds it. The existing cache receives the transformed Values using
unchanged dimensions and positions. With Depth-Attention enabled, MGVR runs
before depth mixing; both mechanisms remain active. In looped configurations,
MGVR parameters are shared with their owning unique blocks. Block 0 remains
unmodified on every execution; the anchor remains its first execution's Value.

For d12, width 768, six KV heads: each of 11 blocks adds `42*768 + 6*6`
parameters, for **355,212 total**. Gate/mixing math and master parameters use
FP32; the result returns to the original Value dtype. The new AdamW group has
zero weight decay, betas `(0.9, 0.999)`, epsilon `1e-8`, and the existing
unembedding learning-rate scale. Every original optimizer group is preserved.
New parameters are reported separately from `transformer_matrices` so the
trainer's automatic token/batch/learning-rate scaling keeps its original
inputs. Training/decode/prefill FLOPs include the gate projection and head
mixing; nonlinearities and memory traffic remain approximate.

The real trainer records validation BPB, CORE when enabled, step time,
throughput, approximate MFU, invocation wall time, peak CUDA memory, and GPU
cost when `--gpu-hourly-cost` is supplied. The `mgvr_metrics` checkpoint field
includes the additional parameter count and approximate forward FLOPs.
At validation intervals, a one-row probe of the pending training batch records
mean gate, added/current Value norm ratio, mixing norm, and off-diagonal mixing
norm. Hooks exist only during this eager probe; reductions stay on device
until one transfer. There is no per-training-step diagnostic synchronization.
Separate-evaluation BPB/CORE use the existing `scripts.base_eval` loader with
`--model-tag=d12-mgvr`; actual experiment scores require a trained checkpoint.

No Modal/Part 2 wrapper exists in this checkout. The existing `base_train`
runner is the training integration point. Keep the established tokenizer,
dataset, sequence length, seed, batch and iteration arguments when adding
`--mgvr` to a controlled experiment. No full A3 training was launched.

## Audit of existing switches

| Feature | Off/default | Opt-in | Isolation |
| --- | --- | --- | --- |
| xIPReLU | `mlp_activation="relu2"` | `--mlp-activation=xiprelu` | Only allocates activation scalars and executes its formula when selected |
| Diff V2 | `diff_v2=False` | `--diff-v2` | Extra queries/lambda projection and pair subtraction are conditional |
| Depth-Attention | `depth_attention=False` | `--depth-attention` | Source references, mixing and special cache handling are conditional; stride requires the flag |
| Looped depth | `n_loops=1` | `--n-loops=N` | Reuses the configured core via the layer plan; prelude/coda have their own arguments |
| MGVR | `mgvr=False` | `--mgvr` | No new parameters, anchor list or MGVR forward calls when off |
| Attention windows | baseline `SSSL` | `--window-pattern=L` or another pattern | Independent of architecture flags |
| FP8 | disabled | `--fp8` | Existing conversion is conditional; MGVR's raw FP32 matrices are outside that conversion |

LeRoPE and NormFormer are not implemented in this checkout. Value Embeddings,
QK normalization, smear/backout and residual scalars are existing baseline
features, preserved as they were. No isolation defect was found in the four
implemented architecture ablations, and none was rewritten.

## Verification

On 2026-11-02, the 12 relevant test files below produced **179 passed, 22
skipped**, across two non-overlapping invocations. After the shared metrics
names were generalized, MGVR and Depth-Attention trainer smoke tests were
rechecked: **4 passed**. `git diff --check` and parsing all modified production
files also passed.

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/ablations/test_mgvr.py tests/ablations/test_mgvr_smoke.py \
  tests/ablations/test_xiprelu.py tests/ablations/test_xiprelu_smoke.py \
  tests/ablations/test_diff_v2.py tests/ablations/test_diff_v2_smoke.py \
  tests/ablations/test_depth_attention.py tests/ablations/test_depth_attention_smoke.py \
  tests/ablations/test_looped.py tests/test_attention_fallback.py \
  tests/test_engine.py tests/test_optim.py
```

`tests/ablations/test_mgvr.py` covers scalar-reference arithmetic; static and dynamic
off-diagonal retrieval; exact initial FP32/BF16 identity including VE; same-seed
shared weights, logits and loss; projection/VE anchor autograd; real optimizer
membership, gradients and updates; causality; diagnostics; GQA; full/restricted
windows; prompt prefill/token decoding; fullgraph AOTAutograd compilation;
strict checkpoint loading; SFT/RL save blocks; parameter and FLOPs accounting.

All 16 combinations of xIPReLU, Diff V2, Depth-Attention and looping with MGVR
off are compared against immutable pre-MGVR commit `d854e74`: exact parameters,
state keys, logits, gradients, optimizer state/groups and FLOPs. Old checkpoint
metadata without `mgvr` defaults to off.

`tests/ablations/test_mgvr_smoke.py` trains the real compiled CPU trainer for three tiny
offline steps, saves model/optimizer/metadata, and resumes for another step.
It covers MGVR alone and all prior architecture flags combined, confirms both
new matrices learn and continue updating, checks diagnostics/output directories,
and rejects resuming with the wrong MGVR setting. It reuses existing smoke data.

CUDA/FA3 smoke cases are included but skip on this CPU-only host. GPU BF16,
CUDA Inductor, FP8 execution, distributed training, full-scale throughput/cost,
and real validation/separate-evaluation/CORE scores remain untested. Cache
handling inherits the baseline's batch-position/chunk assumptions; MGVR adds
no alternative cache management. SFT/RL save blocks are tested without launching
dataset-dependent finetuning.
