**Ablation conflict review — November 2, 2026**

The initial audit reviewed the working tree based on `fc73745`, including the
move of ablation documentation and tests. The follow-up at `77bc235` fixes the
MGVR optimizer conflict, adds an explicit experiment selector, and extends
baseline and cross-ablation regression coverage.

Two serious ablation integration defects were reproduced. MGVR's conflict with
the distributed optimizer is now fixed. Resume can still silently accept a
different loop layout. An inherited FP8 FLOPs accounting defect also remains
and can change the training horizon substantially. The ordinary single-process
tensor paths passed the checks below, including every on/off combination of
the implemented ablations.

**Fixed: MGVR's matrices violated the distributed AdamW sharding constraint**

Locations: [MGVR parameter allocation](../nanochat/gpt.py#L120),
[optimizer registration](../nanochat/gpt.py#L672), and
[AdamW reduction](../nanochat/optim.py#L282).

MGVR allocates `w_z` with shape `(G + G**2, n_embd)`, where `G` is the KV-head
count. It registers this matrix in AdamW. Before the fix, the distributed
optimizer required the first dimension of parameters with at least 1,024
elements to divide evenly by the process count, without a fallback.

For d12 with six KV heads, `w_z.shape == (42, 768)`. Calling the actual
`MuonAdamW._reduce_adamw` method with this parameter and world sizes four and
eight reproduces these assertions before any collective needs to execute:

```text
AdamW reduce_scatter requires shape[0] (42) divisible by world_size (4)
AdamW reduce_scatter requires shape[0] (42) divisible by world_size (8)
```

Those distributed MGVR configurations previously failed at their first
optimizer step. The fix uses averaged all-reduce gradients and full replicated
updates for parameters that cannot be evenly sharded. Evenly shardable large
parameters retain their original reduce-scatter/all-gather path. The model
formula, parameter shapes and optimizer group ordering are unchanged.

The tradeoff is replicated AdamW moment storage for affected matrices.
`tests/test_optim_distributed.py` verifies two, four and eight simulated ranks
with two, six and eight KV heads. It runs three real optimizer updates and
state roundtrips, checks both replicated and sharded moment shapes, and compares
every rank's weights with PyTorch AdamW using averaged gradients. Only transport
is simulated; actual NCCL collectives were not run on this CPU host.

**P1: resume can silently execute a different loop layout**

Locations: [layer-plan construction](../nanochat/gpt.py#L76) and
[base-training resume checks](../scripts/base_train.py#L211).

Resume validates MGVR and depth-attention settings explicitly, then relies on
strict tensor loading. It does not compare `n_loops`, `n_prelude`, `n_coda` or
the resulting layer plan against the saved configuration. Strict loading only
detects differences that alter parameter names or shapes.

Concrete reproduction, with four unique blocks and two loops:

| Configuration | Executed block indices |
| --- | --- |
| Prelude 1, coda 0 | `0,1,2,3,1,2,3` |
| Prelude 0, coda 1 | `0,1,2,0,1,2,3` |

Both have seven executed steps and identical state shapes. The actual trainer
resume block accepts this change, including when xIPReLU, Diff V2,
Depth-Attention and MGVR are all enabled. The default depth stride also remains
equal, so its existing validation does not catch it. Loaded weights then produce
different logits: maximum absolute differences in the tiny active-model probes
were approximately `0.00239` for looping alone and `0.00205` with all flags.

This silently changes weight reuse, per-step scalar assignments, and the model
being optimized. Changing `window_pattern` is another parameter-free behavioral
change that the same resume checks do not validate.

Suggested repair: validate the full saved architecture configuration, allowing
missing legacy keys to resolve to their defaults. Intentional architecture
conversion should be a separate operation from optimization resume.

**P1 when using a FLOPs budget: FP8 conversion removes most matmuls from accounting**

Locations: [FP8 conversion](../nanochat/fp8.py#L261),
[matmul counting](../nanochat/gpt.py#L547),
[trainer's post-conversion estimate](../scripts/base_train.py#L319), and
[FLOPs-based horizon](../scripts/base_train.py#L406).

`num_matmul_params()` recognizes the custom `gpt.Linear`. FP8 conversion replaces
eligible instances with `Float8Linear`, which inherits `nn.Linear` directly and
fails that type check. Their matmuls disappear from the estimate although their
weights and execution remain present. This also affects the output projection
outside the block stack.

Using the trainer's actual FP8 filter on a meta-device d12 model with Diff V2
and two loops produced:

| MGVR | Counted matmul parameters before/after FP8 | Training FLOPs/token before/after FP8 |
| --- | --- | --- |
| Off | `209,302,392 / 111,480` | `1,652,176,080 / 397,030,608` |
| On | `210,012,024 / 821,112` | `1,657,143,504 / 401,998,032` |

With `--target-flops` and no explicit iteration override, these estimates imply
about 4.16x and 4.12x as many steps, respectively, relative to the same arithmetic
count with the converted matmuls included. With explicit iterations the main
effect is incorrect FLOPs/MFU reporting. Actual FP8 kernels were not executed.

This defect already exists in the imported baseline's `num_matmul_params()`;
it was not introduced by the new ablations. It remains a significant conflict
when FP8 and FLOPs-budgeted ablation training are combined.

Suggested repair: count both regular and converted linear modules, retaining
the loop execution multiplier and MGVR's explicit projection count.

**P2: default names can collide between loop architectures and evaluation results**

Locations: [base tags](../scripts/base_train.py#L198),
[SFT tags](../scripts/chat_sft.py#L393),
[RL tags](../scripts/chat_rl.py#L311), and
[CORE result filenames](../scripts/base_eval.py#L154).

Executing the actual naming statements confirms:

- Base training names both loop layouts above `d4-l2`: prelude/coda are absent
  from the tag.
- With no explicit `--model-tag`, SFT and RL name both ordinary and looped d4
  models `d4`: even the loop count is absent. Explicit tags avoid this fallback.
- Separate CORE evaluation names its CSV only by checkpoint step. Different
  ablations evaluated at the same step write the same file using mode `w`.

These paths can overwrite previous artifacts or make the wrong experiment easy
to select. The saved configuration itself still retains the architecture flags.
Use distinct explicit tags until naming is corrected; include the selected tag
in evaluation filenames as well.

**Flag-by-flag assessment**

| Flag | Assessment |
| --- | --- |
| `--mlp-activation=relu2/xiprelu` | Conditional MLP formula and scalar allocation; optimizer includes scalars exactly once; checkpoint reconstruction and off-path regression tests pass. No core conflict found. |
| `--diff-v2` | Doubles internal query heads, reduces adjacent pairs back to model width, preserves KV dimensions; combination, cache, gradient and checkpoint tests pass. Its added parameters/compute can change automatic training budgets. |
| `--depth-attention` | Mixes values after VE/MGVR and before ordinary attention; MGVR's branch forwards depth references correctly. Causality, gradients, cache behavior and roundtrips pass. |
| `--depth-attention-stride` | CLI rejects nonpositive strides or a stride without the enabling flag; resume validates the effective stride. Default uses half the unique block count, while source indices count executed steps. Set explicitly when intending a stride based on effective depth. |
| `--mgvr` | Neutral initialization, active mixing, anchor autograd and combination checks pass. Distributed shape failure is fixed using replicated AdamW updates when needed. |
| `--n-loops`, `--n-prelude`, `--n-coda` | Execution plan, effective-depth scalar/cache sizing, shared gradients and decode checks pass. Same-shape resume changes and naming collisions remain. |
| `--window-pattern` | Restricted and full-window attention pass in the CPU matrix, including all ablations together. Changing the pattern during resume is not guarded. |
| `--fp8` | Conditional conversion leaves MGVR's explicit FP32 matrices outside FP8. FLOPs accounting defect above remains; GPU kernel compatibility was not established here. |
| `--fp8-recipe` | CLI advertises `rowwise`, but the local implementation accepts only `tensorwise` and raises for `rowwise`. This is an inherited runtime-option mismatch. |
| `--gpu-hourly-cost` | Reporting only; validated to require depth attention or MGVR and a nonnegative value. No model computation conflict found. |
| `--experiment` | Newly added central selection in `scripts.base_train`: named single-ablation selections reject other enabled ablations before compute/data setup. `combined` requires at least two explicit ablations. Omission preserves legacy CLI behavior. |

The combined value path is VE -> MGVR -> depth mixing -> causal sequence
attention -> Diff V2 subtraction -> output projection, followed by the selected
MLP activation. Looping applies that path according to executed depth. Neither
the MGVR branch nor Diff V2 bypasses depth mixing when enabled together.

MGVR deliberately excludes unique block zero from its learned correction,
including later loop visits to that same shared block; its first-execution
anchor stays attached to autograd. This is documented behavior. Likewise,
Depth-Attention's special handling of unequal cache offsets and smear across
multi-token continuation chunks is enabled only with its flag. Baseline chunk
and unequal-offset limitations remain when it is off; the ordinary Engine
prefill followed by one-token decoding passed in the tested combinations.

For controlled comparisons, pin the intended iteration/token budget and batch
size explicitly. Diff V2 changes the quantities used by automatic horizon and
hyperparameter calculations; MGVR deliberately reports its extra parameters
separately from the scaling-law budget. These are experiment-design differences
even when every forward path is functioning correctly.

**Focused follow-up on the three suspected interactions**

| Combination | Finding and evidence |
| --- | --- |
| MGVR + Depth-Attention | No value-passing defect found. The MGVR branches of both GPT and Block forward the depth-source list. MGVR transforms post-VE current V; depth attention mixes that result and retains the mixed result. The anchor is first-execution, post-VE V; at depth zero there are no earlier sources, so depth mixing preserves that anchor. Disabled-depth checks forbid calling the depth helper even when MGVR is active. |
| Diff V2 + Depth-Attention | No head-layout defect found. Only query heads double. The depth mixer averages contiguous queries sharing each KV head, including differential partners; K/V keep their original shapes. Sequence attention produces doubled query-head outputs, then differential subtraction restores the normal model width. Explicit per-head attention and an independent depth-SDPA reference agree with outputs and gradients. |
| Looping + MGVR / Depth-Attention | No stale-value or duplicate-parameter defect found in normal execution. Each execution computes fresh K/V; depth references distinguish executions and remain available across loops inside one forward. Cache slots use executed depth. Every forward creates a new MGVR anchor/source list; cache reset is checked across prompts. Shared parameters are constructed/initialized once and receive accumulated gradients from their repeated uses. The separate same-shape resume gap above still exists. |

`tests/ablations/test_ablation_interactions.py` implements the combined reference
without calling production GPT, Block, Attention, MGVR or depth-mixing forwards.
Depth attention is expressed as a separate SDPA over source depths; temporal
attention uses one SDPA per query head. Thirty-two parameterized comparisons
cover all MGVR/Diff/depth on/off combinations, full-core looping and a
prelude/core/coda layout, FP32 MHA and BF16 GQA, xIPReLU, active MGVR weights and
nonzero output projections. Model logits and all parameter gradients agree
within explicit dtype tolerances. Eight more tests check repeated block-zero
prefill/decode and cache reset; six inspect depth-source identities, retained
autograd and anchor scope for strides 1, 2 and 99. Existing historical tests
also compare disabled paths against the pre-ablation implementations.

There are deliberate modeling choices, rather than mechanical conflicts:
MGVR precedes depth mixing, depth scoring averages Diff V2's paired queries,
MGVR leaves shared block zero uncorrected on every visit, and the default depth
stride uses unique block count. Tests establish implementation consistency;
they do not establish that combining these modeling choices improves accuracy.

**Central experiment selection and original-baseline equivalence**

No centralized selector or separate NanoChat Part 2 launcher existed in this
checkout. The new [selector](../nanochat/experiments.py) is integrated into
`scripts.base_train` before compute/tokenizer/dataset setup. For example:

```sh
python -m scripts.base_train --experiment=baseline ...
python -m scripts.base_train --experiment=mgvr ...
python -m scripts.base_train --experiment=looped --n-loops=2 ...
python -m scripts.base_train --experiment=combined --mgvr --depth-attention ...
```

Named selections enable the selected ablation and reject any other active
ablation, including stray loop partition/depth-stride options. The selection
is recorded in the trainer's existing user metadata. Thirty-seven selector
tests include all conflicting single-ablation choices, a real parser rejection,
and a real compiled offline MGVR training step. External Part 2 launchers must
pass the selector when invoking this entry point; legacy calls without it
retain their prior behavior.

An existing `test_default_matches_original_gpt` already compared default logits,
counts and FLOPs against the imported baseline. The new
`tests/ablations/test_baseline_equivalence.py` explicitly disables every
architecture ablation and loads both original GPT and original optimizer code
from the pre-ablation import commit `0ed62a1`. Eight cases cover FP32/BF16,
MHA/GQA and full/restricted windows, with active projections and smear.
Weights, logits, loss, parameter gradients, two optimizer updates and cached
prefill/token-decode outputs all match exactly; parameter counts, FLOPs and
initial optimizer groups match too. These tests passed without skipping the
historical comparison.

**Verification and limits**

The follow-up combined suite completed with **279 passed, 22 skipped**:

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/ablations/ tests/test_optim_distributed.py tests/test_attention_fallback.py \
  tests/test_engine.py tests/test_optim.py
```

The initial, pre-fix combined suite had 179 passes and 22 skips. Its additional
temporary audit exercised **192 configurations**, all passing:
two compute dtypes (FP32/BF16), both activation choices, Diff V2 on/off,
Depth-Attention on/off, MGVR on/off, three loop layouts (ordinary, two full
loops, and two core loops with prelude/coda), and full/restricted windows.
Tiny models used GQA, nonzero output projections, nonzero smear, and active
MGVR matrices. Each case checked two actual optimizer steps, finite loss/weights/
gradients, exact optimizer membership, future-token causality, full versus
prefill/token cached logits, cache advancement, and strict checkpoint roundtrip
outputs. CPU BF16 cached comparisons used explicit numerical tolerances.

Targeted checks separately reproduced the distributed shape assertions, FP8
accounting loss, accepted changes in the actual resume block, and collisions
in the actual tag-building statements. Existing trainer smoke tests compile
with `aot_eager` and exercise save/resume. GPU FA3/FP8 execution, distributed
collectives, CUDA Inductor, large-scale memory/performance and model quality
were not established by these CPU checks. LeRoPE and NormFormer have no
implementation or flags in this checkout.
