# Ablation handover

Run everything below from the `nanochat/` project root. **Start with the baseline
and each ablation STANDALONE.** Improvements below are goals, not measured wins.

| Ablation | What changes / intended improvement | Standalone selection |
| --- | --- | --- |
| [Looped depth](looped/README.md) | Repeats core blocks with shared weights: more computation per parameter, aiming for stronger iterative reasoning. `--depth` counts unique blocks. | `--experiment=looped --n-loops=2` |
| [xIPReLU](xiprelu/README.md) | Replaces ReLU² with a quadratic activation whose positive/negative coefficients learn: more flexible activation and gradient flow. | `--experiment=xiprelu` |
| [Diff V2](diff_v2/README.md) | Subtracts a learned fraction of a second attention retrieval: aims to cancel shared attention noise. | `--experiment=diff-v2` |
| [Depth-Attention](depth_attention/README.md) | Mixes same-token values across selected earlier depths before causal attention: reuses earlier representations without new parameters. | `--experiment=depth-attention` |
| [MGVR](mgvr/README.md) | Gates current values and adds a learned cross-head mixture of first-layer values: aims to preserve and selectively reuse early information. | `--experiment=mgvr` |

## Local setup and training

Install CPU/test dependencies; use a separate cache for this small demo. Data
download and tokenizer training are needed once per cache, **not for tests**.

```sh
uv sync --extra cpu --group dev
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat-handover"
.venv/bin/python -m nanochat.dataset -n 2
.venv/bin/python -m scripts.tok_train --max-chars=1000000 --vocab-size=4096
```

Define this shared command once in the same shell. It trains a tiny CPU model
for 20 updates, with compilation and evaluation disabled for a quick demo.

```sh
train() {
  OMP_NUM_THREADS=1 NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
    .venv/bin/python -m scripts.base_train \
    --device-type=cpu --depth=4 --aspect-ratio=16 --head-dim=16 \
    --max-seq-len=64 --window-pattern=L --device-batch-size=1 \
    --total-batch-size=64 --num-iterations=20 --warmup-steps=0 \
    --warmdown-ratio=0 --eval-every=-1 --core-metric-every=-1 \
    --sample-every=-1 "$@"
}
```

**STANDALONE runs — one ablation at a time; these are the primary comparisons:**

```sh
train --experiment=baseline        --model-tag=handover-baseline
train --experiment=looped --n-loops=2 --model-tag=handover-looped
train --experiment=xiprelu         --model-tag=handover-xiprelu
train --experiment=diff-v2         --model-tag=handover-diff-v2
train --experiment=depth-attention --model-tag=handover-depth-attention
train --experiment=mgvr            --model-tag=handover-mgvr
```

Each named selection enables only its ablation and rejects conflicting flags
before compute/data setup. Example: `--experiment=mgvr --diff-v2` is rejected.
Without `--experiment`, legacy flags still work but have no standalone guard.
The selector changes architecture only; keep data, seed, batch and training
budget fixed for controlled comparisons. These tiny runs do not measure quality.

**Combined runs are separate experiments**, requiring `--experiment=combined`
and at least two explicit ablations. Raw flags are `--n-loops=2`,
`--mlp-activation=xiprelu`, `--diff-v2`, `--depth-attention`, and `--mgvr`:

```sh
train --experiment=combined --mgvr --depth-attention --model-tag=handover-combined
```

Checkpoints: `$NANOCHAT_BASE_DIR/base_checkpoints/<model-tag>/`. Use a new tag
for each run and preserve the original architecture flags when resuming. For
larger GPU runs, use `scripts.base_train` with these same selections and your
GPU/budget arguments; omit `TORCHDYNAMO_DISABLE=1`. Known resume-layout,
FP8-budget and naming limitations are in the
[conflict review](../../dev/ablation_conflict_review.md).

## Rerun implementation and smoke tests

**All ablation implementation, isolation, interaction and smoke tests, plus
optimizer, attention and decoding regressions** (the prior 279-pass suite):

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -rs \
  tests/ablations/ tests/test_optim_distributed.py \
  tests/test_attention_fallback.py tests/test_engine.py tests/test_optim.py
```

Smoke tests only (looping is exercised by the xIPReLU smoke cases too):

```sh
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -rs tests/ablations/test_*_smoke.py
```

Tests generate temporary offline data/tokenizers. Smoke tests run the real CPU
trainer using full-graph `aot_eager` compilation, save checkpoints and resume.
Keep full Git history: baseline/regression tests load historical code, including
`0ed62a1`. `-rs` shows skip reasons; hardware-dependent skips are unverified paths.

## What a pass establishes

Passing implementation tests supports the tested formulas, gradients, optimizer
updates, causality and cached decoding. Passing smoke tests supports short
training/save/resume integration. Selector, disabled-path and historical
equivalence tests specifically check isolation; combined references check
interactions. **No unintended feature leakage was found in the tested cases.**
This is evidence, **not proof of complete correctness** for every configuration.
CPU checks do not establish CUDA/FA3/FP8 or real multi-GPU correctness (distributed
transport is simulated), full-scale stability, or improved BPB/CORE scores.
