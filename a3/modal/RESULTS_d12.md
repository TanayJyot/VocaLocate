# A3 Part 2: d12 picochat ablations

Seven d12 runs, each on one Modal H100 (80 GB, Flash Attention 3), trained 2026-11-02 to 11-03.
Script: [train_d12.py](train_d12.py). Per-run logs and result JSONs are in `out/logs/` and `out/results/`.

## Setup (identical for every run)

| | |
|---|---|
| Model | nanochat d12, width 768, head dim 128, 2048-token context, `SSSL` sliding windows |
| Data | ClimbMix shards 0–31 (the same order for every run), tokenizer: 32,768 vocab |
| Budget | 2,520 steps × 524,288 tokens = **1.32B tokens** (12 tokens per scaling param of the baseline, nanochat's default) |
| Batch | 524,288 tokens/step (nanochat's d12 reference), device batch 32 × 2048 |
| Seed | 42 (nanochat default), one run per config |
| Eval | val bpb on 42M held-out tokens every 250 steps; CORE (22 tasks, ≤500 examples each) at the end |
| Code | VocaLocate PR #27 (`feat/a3-other-ablation-ideas`, `04ab375`); each run turns on one change with `--experiment=` |

The token budget and batch are **pinned to the baseline's values**. Otherwise nanochat's auto-sizing would give
Diff V2, which adds parameters, a 6% longer schedule, and the comparison would no longer be controlled.

## Results

| Run | What changes | Transformer weights | FLOPs/token | Val bpb ↓ | Δ bpb | CORE ↑ | Δ CORE |
|---|---|---|---|---|---|---|---|
| baseline | — | 84.9M | 7.60e8 | 0.8480 | — | 0.1608 | — |
| **looped-l2** | 10 middle blocks run twice (22 effective layers) | 84.9M | 12.74e8 (+68%) | **0.8380** | **−0.0100** | **0.1752** | **+0.0144** |
| **Diff V2** | paired queries, learned subtraction | 92.1M (+8%) | 9.02e8 (+19%) | 0.8415 | −0.0065 | 0.1727 | +0.0119 |
| Depth-Attention | same-token value mixing across earlier depths | 84.9M | 7.60e8 | 0.8467 | −0.0013 | 0.1510 | −0.0098 |
| MGVR | gated values + first-layer value mixture | 84.9M (+0.4M other) | 7.62e8 | 0.8467 | −0.0013 | 0.1637 | +0.0029 |
| xIPReLU | learned quadratic MLP activation | 84.9M | 7.60e8 | 0.8522 | +0.0042 | 0.1553 | −0.0055 |
| looped-u7 | 7 unique blocks, 5 run twice (12 effective layers) | 49.5M (−42%) | 7.60e8 | 0.8696 | +0.0216 | 0.1528 | −0.0080 |

Validation bpb during training (lower is better):

| Step | baseline | looped-l2 | Diff V2 | Depth-Attn | MGVR | xIPReLU | looped-u7 |
|---|---|---|---|---|---|---|---|
| 500 | 1.0062 | 0.9997 | 1.0015 | 1.0075 | 1.0050 | 1.0085 | 1.0480 |
| 1000 | 0.9491 | 0.9396 | 0.9418 | 0.9480 | 0.9465 | 0.9528 | 1.0025 |
| 1500 | 0.9049 | 0.8956 | 0.8986 | 0.9039 | 0.9036 | 0.9098 | 0.9544 |
| 2000 | 0.8715 | 0.8616 | 0.8648 | 0.8701 | 0.8700 | 0.8758 | 0.9061 |
| 2520 | 0.8480 | 0.8380 | 0.8415 | 0.8467 | 0.8467 | 0.8522 | 0.8696 |

The ranking is the same at every checkpoint from step 1000 on: the gaps are steady, not end-of-run noise.

## Reading the results

- **Looping at the same parameter count (looped-l2) gives the biggest gain**, −0.010 bpb and +0.014 CORE. It pays
  for that with 68% more compute per token, and so 66% more training time.
- **Diff V2 is the strongest of Carmen's four.** Its gain (−0.0065 bpb, +0.012 CORE) is clear at every
  checkpoint, but it also uses 19% more FLOPs and 8% more weights. So part of the gain may be extra capacity rather
  than the noise-cancelling mechanism. A baseline with matched FLOPs would settle that.
- **Depth-Attention and MGVR are ties on bpb** (−0.0013, smaller than what a different seed would plausibly
  change). Their CORE moves in opposite directions by about 0.01, which is within CORE's noise at this size.
- **xIPReLU is slightly worse** on both metrics.
- **Looping at the same compute (looped-u7) loses**, by +0.022 bpb, with 42% fewer weights. At d6 the equivalent
  pair (looped-d6-l2 vs base-d10) tied. The d6 runs saw only 1.8 tokens per parameter against 12 here, so the d6
  "free depth" result did not survive the move to compute-optimal training. When data is plentiful, unique
  weights matter more than repeated depth.

## Cost (Modal H100 at $3.95/hr, billed per second)

Container time includes startup, `torch.compile`, all validation evals and the final CORE eval.

| Run | Training | Container total | Tokens/s | MFU | Peak memory | Cost |
|---|---|---|---|---|---|---|
| baseline | 40.5 min | 52.3 min | 542k | 41.6% | 28.1 GB | $3.45 |
| looped-l2 | 67.3 min | 87.0 min | 326k | 41.9% | 45.1 GB | $5.73 |
| looped-u7 | 41.1 min | 51.9 min | 534k | 41.0% | 27.1 GB | $3.42 |
| xIPReLU | 41.7 min | 53.5 min | 527k | 40.5% | 28.1 GB | $3.52 |
| Diff V2 | 51.3 min | 64.1 min | 427k | 39.0% | 32.8 GB | $4.22 |
| Depth-Attention | 43.1 min | 55.9 min | 508k | 39.1% | 29.8 GB | $3.68 |
| MGVR | 48.4 min | 61.2 min | 453k | 34.9% | 31.7 GB | $4.03 |
| **Total** | | **7.10 H100-hours** | | | | **$28.05** |

Add $0.87 for the 30-step smoke test of every config, run before the full runs. The seven runs ran in parallel,
so the wall-clock time was about 1.5 hours.

MGVR runs 12% slower than its FLOPs imply (35% MFU vs ~41%). Carmen's PR replicates the AdamW update for
MGVR's unevenly shaped matrices instead of sharding it. That is a cost of the implementation, not of the idea.

## Caveats

- **One seed per run.** Gaps under about 0.003 bpb, and CORE gaps under about 0.01, should be treated as ties.
- **looped-u7 weight decay:** nanochat scales weight decay with the parameter count, so the run with 42% fewer
  weights got slightly higher weight decay than the baseline. This is nanochat's normal scaling rule, and the d6
  base-d10 comparison worked the same way.
- **PR #27 tests on Windows/CUDA:** 283 passed, 14 skipped (they need FA3, which Windows lacks) and 4 failed. All
  four failures are `test_cached_decoding_matches_full_forward` in `tests/ablations/test_xiprelu.py`, and they
  fail for plain ReLU² too. That test assumes float32 (atol 1e-6), but on CUDA the KV cache is bf16 (error about
  1.2e-4). It is a test-precision issue, not a model bug, and training is unaffected.

## W&B

The runs were logged with `WANDB_MODE=offline` and saved in `out/wandb/wandb/`. To upload them to your W&B
account (run `wandb login` first):

```bash
cd out/wandb/wandb
wandb sync offline-run-20261103_040506-2msa7208 offline-run-20261103_040529-e4cyz84k \
  offline-run-20261103_042935-cmrws97h offline-run-20261103_040531-cxowofgs \
  offline-run-20261103_040506-u4asolds offline-run-20261103_040507-kn36s4nh \
  offline-run-20261103_040507-y6nvh3l6
```

These are, in order: baseline, looped-l2, looped-u7, xIPReLU, Diff V2, Depth-Attention and MGVR. Only these
seven full runs are committed; the smoke-test runs and a stopped launch were left out.
