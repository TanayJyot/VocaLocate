# Optional Depth-Attention ablation

Depth-Attention lets a layer retrieve values from selected earlier depths at
the **same token position**, before the existing causal attention runs.
It adds no learned parameters. It is disabled by default.

References: [paper](https://aclanthology.org/2026.emnlp-main.2085/) and
[official implementation](https://github.com/LUMIA-Group/Depth-Attention/blob/main/src/llamafactory/model/modeling/modeling_llama_depth_attention.py).
This patch implements the guide's stride-only variant, without recent-window sources.

## Step 1: keep the experiment optional

`GPTConfig` in `nanochat/gpt.py` adds:

```python
depth_attention: bool = False
depth_attention_stride: int | None = None
```

When enabled, `None` means `max(1, n_layer // 2)`; explicit strides must be
positive integers. `n_layer` counts unique blocks. When combined with looping,
source indices count **executed steps** and can refer to different executions
of the same shared block. The default stride still uses the unique block count;
an explicit stride can instead be chosen for the effective depth.

With the flag off, the original block/attention calls execute. No source list is
allocated, no K/V references are retained, and the mixing helper is not called.
No random initialization, projection shapes, parameters, optimizer groups,
residual scaling, x0 mixing, value embeddings, or attention kernels change.

The local previous ablations are xIPReLU, Differential Attention V2, and looping.
LeRoPE and NormFormer are mentioned in the supplied guide but are absent from
this checkout; compatibility with absent implementations cannot be established.

## Step 2: select earlier depths

`depth_source_layers` in `nanochat/depth_attention.py` returns:

```text
sources(l) = {0, stride, 2*stride, ... strictly below l} plus {l}
```

For d12 with stride 6:

| Current layer (zero indexed) | Candidates |
| --- | --- |
| 0 | 0 |
| 4 | 0, 4 |
| 6 | 0, 6 |
| 9 | 0, 6, 9 |
| 11 | 0, 6, 11 |

Thus layer 10 in one-indexed prose reads layers 1, 7, and 10.

## Step 3: score and mix values

Suppose the current token is *python* at layer 10. Its current query scores
keys for *python* from layers 1, 7, and 10. The layer-axis softmax could assign
illustrative weights 0.15, 0.65, and 0.20. The new value would be:

```text
mixed_v10 = 0.15 * mixed_v1 + 0.65 * mixed_v7 + 0.20 * v10
```

The earlier values are themselves already mixed. This enables recursive reuse
without introducing a separate learned memory. The weights vary by token,
batch row, and KV head.

`mix_depth_values(q, k, v, earlier_kv)` implements:

```text
score[j] = dot(q_current[t], k_source_j[t]) / sqrt(head_dim)
alpha = softmax(scores, axis=source_depth)
mixed_value[t] = sum_j alpha[j] * value_source_j[t]
```

All tensors use `(B,T,H,D)` layout; the result has exactly V's shape and dtype.
The helper never scores one token against another token. Dot products, softmax,
and the weighted sum use FP32 locally, then the result is cast to V's dtype.
The first layer returns the original V directly.

For GQA, contiguous query heads sharing a KV head are averaged before depth
scoring. When combined with Diff V2, that average also includes its paired query
heads. This combination is an engineering extension: mechanical correctness
tests do not establish its modeling quality.

## Step 4: use the existing attention path

`CausalSelfAttention.forward` preserves this sequence:

1. Compute Q, K, and V with the existing projections.
2. Add the existing gated value embedding to V.
3. Apply the existing rotary embeddings, QK normalization, and 1.2 scaling.
4. If enabled, replace V with the depth-mixed V.
5. Call the same FA3/SDPA wrapper, with the same causal/window arguments.
6. Apply the existing Diff V2 subtraction if enabled, and the output projection.

Depth attention chooses **which depths' values** ordinary attention can read.
Ordinary attention then chooses **which token positions** to attend to.
Future-token information is never introduced by the depth operation.

## Step 5: carry sources during training and generation

During training, `GPT.forward` creates a temporary list and passes it through
`Block.forward`. Selected attention executions append their normalized/scaled K
and final mixed V. Only stride-selected depths needed by later layers enter the
list. These tensors are not detached or stored on modules; gradients can flow
through the depth selection into earlier K/V projections.

During generation, earlier executions have already written the new positions'
K and mixed V into their normal KV-cache slots. Later executions retrieve only
those positions, using each row's offset. They never treat the entire token
history as depth candidates. Cache storage/layout and position advancement are
unchanged. No additional persistent inference tensors are introduced.

Two pre-existing limitations need handling in the enabled path:

- The SDPA cache wrapper assumes equal batch offsets. For unequal offsets,
  depth attention calls the same wrapper once per row using original cache views.
  Rotary lookup also uses each row's offset. The wrapper itself is unchanged.
- Multi-token continuation must carry NanoChat's learned smear across the chunk
  boundary. The enabled path applies the same smear formula to the first new
  token using the existing cached previous embedding. The disabled path keeps
  the previous behavior.

## Step 6: flags, checkpoints, and measurements

Append these to your existing Part 2 training command, keeping its other settings:

```bash
--depth-attention
# Optional: --depth-attention-stride=6
```

Remove both flags to restore baseline execution. With default tags, an enabled
d12 run saves under `d12-depth-attention`; explicit stride 6 uses
`d12-depth-attention-s6`. Combined ablations retain their existing tag suffixes.
An explicit `--model-tag` still overrides naming; choose a separate directory
for each experiment. Use the same flags/tag to resume. A depth flag/stride
mismatch during resume raises an error because this parameter-free change
cannot be detected from tensor shapes.

`base_train.py` stores the enabled configuration in checkpoint metadata.
The existing checkpoint loader reconstructs it automatically; old metadata
defaults to disabled. Disabled checkpoints omit the new config keys so the
previous loader can still read them. SFT and RL also preserve enabled flags and
use distinct default suffixes when saving; disabled metadata stays unchanged.

The existing validation BPB/CORE evaluations and training settings stay in place.
Enabled runs additionally record `depth_attention_metrics` in checkpoint metadata
and print/log a final measurement summary:

- wall-clock seconds for this invocation, including setup, compilation, and evaluation;
- training-step time and aggregate training tokens/second, including warmup;
- peak CUDA allocation per rank, or null on CPU;
- latest validation BPB and CORE, or null when not evaluated;
- GPU cost only if `--gpu-hourly-cost=<rate>` is supplied on CUDA, calculated
  as wall hours × world size × rate (one GPU per rank; currency follows the rate).

The small FLOPs addition counts depth dot products and weighted sums, with a
3× forward/backward estimate for training. It excludes softmax/GQA reductions
and memory traffic. MFU is therefore still approximate; it cannot predict the
wall-clock cost of moving earlier tensors.

## Validation

Run the offline acceptance checks:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_depth_attention.py tests/test_depth_attention_smoke.py
```

The tests cover manual arithmetic, same-position mixing, recursive values, source
selection, GQA, FP32/BF16 CPU execution, gradients, real MuonAdamW updates,
causality, AOTAutograd full-graph compilation, cached prefill, token/chunk decoding,
unequal batch offsets, greedy Engine generation with replicated prefill caches,
strict checkpoint loading, and SFT/RL save metadata without fine-tuning.

Disabled-path regression compares against immutable pre-change Git revision
`0fb10042b1ce1c1f07bf9b744e7466d0733049b6`. With baseline, xIPReLU, Diff V2,
looping, and combined settings it checks bitwise identical initialized tensors,
active-attention logits, gradients, cached outputs/cache contents, and an optimizer
update. It also checks identical state keys, optimizer grouping, and FLOPs.
Existing checkpoint compatibility is tested using pre-change generated state
dictionaries and legacy metadata; no user checkpoint is overwritten.

Default d12 architecture comparison (meta device, no full model allocation):

| Setting | Total parameters | KV-cache bytes/token (FP32) | Estimated training FLOPs/token |
| --- | ---: | ---: | ---: |
| Existing model | 286,261,730 | 73,728 | 759,696,048 |
| Depth-Attention | 286,261,730 | 73,728 | 759,944,880 |

Each mode therefore uses 36,864 cache bytes/token at BF16. Training activation
memory can increase because selected earlier K/V must remain available to autograd.

Real trainer smoke results: CPU FP32, synthetic offline data, width 32,
2 unique layers, batch 2 × 16 tokens; compiled model and optimizer use
`aot_eager, fullgraph=True`. Three training steps plus checkpoint resume to step 4:

| Experiment | Parameters | Step 3 BPB | Resumed step 4 BPB |
| --- | ---: | ---: | ---: |
| Depth attention | 55,374 | 8.012901 | 7.999881 |
| + xIPReLU + Diff V2 + 2 loops, stride 2 | 57,686 | 8.012957 | 7.999794 |

Step-3 measurements, including compilation/warmup: 3.56 seconds and 34.5
training tokens/second for depth attention; 5.29 seconds and 23.3 tokens/second
for the combined case. These are tiny execution checks, not quality comparisons
or representative GPU performance measurements. CORE, GPU memory, and GPU cost
were not measured on this CPU-only machine.

CUDA BF16 and FA3 tests are included and explicitly skip without compatible
hardware. CUDA/FA3 kernels, GPU Inductor compilation, FP8, distributed training,
real-dataset CORE, and full SFT/RL training remain untested here. No full d12/d20
training run was launched.

Actual local results across the acceptance and existing regression suites:

| Suite | Passed | Skipped |
| --- | ---: | ---: |
| `test_depth_attention.py` | 44 | 2 |
| `test_depth_attention_smoke.py` | 2 | 0 |
| Existing looping, xIPReLU, Diff V2, attention fallback, trainer smoke, Engine suites | 80 | 14 |
| Total | 126 | 16 |

Python compilation checks and `git diff --check` also pass. Unrelated user changes,
including the pre-existing `uv.lock` modification, were left untouched.
