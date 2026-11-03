# A3 Part 2: d12 picochat ablations

Eleven d12 configurations (thirteen runs), each on one Modal H100 (80 GB, Flash Attention 3), trained 2026-11-02 to 11-03.
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
| Code | VocaLocate PR #27 (`feat/a3-other-ablation-ideas`, `04ab375`); each run turns on one change with `--experiment=`. `--loop-embed` and `--n-loops-min/max` were added on this branch (PR #28) |

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
| looped-u9 † | 9 unique blocks, middle 3 run twice (12 effective layers) | 63.7M (−25%) | 7.60e8 | 0.8560 | +0.0080 | 0.1491 | −0.0117 |
| baseline-long † | baseline trained 4,224 steps (2.21B tokens), looped-l2's total compute | 84.9M | 7.60e8 (×1.68 tokens) | 0.8280 | −0.0200 | 0.1667 | +0.0059 |
| looped-l2-embed | looped-l2 + a learned vector per core pass | 84.9M (+1.5k other) | 12.74e8 (+68%) | 0.8378 | −0.0102 | 0.1806 | +0.0198 |
| looped-rand13 | looped-l2 trained with 1–3 loops per step (uniform), eval at 2 | 84.9M | 12.69e8 (+67%, mean) | 0.8464 | −0.0016 | 0.1723 | +0.0115 |

† Mean of two identical runs (same config and seed; a launcher preemption ran each twice, see Caveats).
looped-u9: 0.8559 / 0.8561 bpb, CORE 0.1456 / 0.1526. baseline-long: 0.8280 / 0.8279 bpb, CORE 0.1690 / 0.1643.

Validation bpb during training (lower is better):

| Step | baseline | looped-l2 | Diff V2 | Depth-Attn | MGVR | xIPReLU | looped-u7 |
|---|---|---|---|---|---|---|---|
| 500 | 1.0062 | 0.9997 | 1.0015 | 1.0075 | 1.0050 | 1.0085 | 1.0480 |
| 1000 | 0.9491 | 0.9396 | 0.9418 | 0.9480 | 0.9465 | 0.9528 | 1.0025 |
| 1500 | 0.9049 | 0.8956 | 0.8986 | 0.9039 | 0.9036 | 0.9098 | 0.9544 |
| 2000 | 0.8715 | 0.8616 | 0.8648 | 0.8701 | 0.8700 | 0.8758 | 0.9061 |
| 2520 | 0.8480 | 0.8380 | 0.8415 | 0.8467 | 0.8467 | 0.8522 | 0.8696 |

The ranking is the same at every checkpoint from step 1000 on: the gaps are steady, not end-of-run noise.

## Compute-matched comparison

baseline-long gives the baseline the same total training FLOPs as looped-l2 (2,520 × 12.74e8 = 4,224 × 7.60e8
per token). With the baseline (×1.00) and baseline-long (×1.68), the gain from just training longer can be
interpolated (linearly in log compute) for any run that spends extra compute:

| Run | Compute | Val bpb | Baseline at same compute | Δ bpb | CORE | Baseline CORE at same compute | Δ CORE |
|---|---|---|---|---|---|---|---|
| looped-l2 | ×1.68 | 0.8380 | 0.8280 (measured) | **+0.0100** | 0.1752 | 0.1667 (measured) | +0.0085 |
| Diff V2 | ×1.19 | 0.8415 | ≈0.8413 (interpolated) | ≈+0.0002 | 0.1727 | ≈0.1628 (interpolated) | ≈+0.010 |

(Δ here is run minus compute-matched baseline: positive bpb is worse, positive CORE is better.)

- **Spending looped-l2's extra compute on more tokens beats spending it on loops**, by 0.010 bpb, which is as
  large as looped-l2's original gain over the baseline. Its CORE edge (+0.009) is close to the run-to-run
  spread measured below, so it is not established.
- **Diff V2's bpb gain is fully explained by its extra compute.** Its CORE gain (≈+0.010) survives the
  correction but is at the edge of noise. The interpolation is an estimate; an exact control would be a
  baseline run at ~3,000 steps.
- **The same-compute looped runs lose less as sharing shrinks.** With 25% fewer weights (u9) the loss is
  +0.008 bpb, with 42% fewer (u7) +0.022. Neither ties the baseline, so at this token budget weight sharing
  costs quality rather than saving it for free.

## Looped extensions: per-loop embedding and random loop count

Both start from looped-l2 (1 prelude + 10 core blocks + 1 coda, same tokens, same seed) and change one thing:

- **looped-l2-embed (`--loop-embed`)**: a learned 768-dim vector per core pass is added to the residual stream at
  the start of that pass, so the shared blocks can tell pass 1 from pass 2. It is zero-initialised, so the run
  starts exactly as looped-l2. Each executed step already had its own `resid_lambdas`/`x0_lambdas` scalars;
  this adds a direction, not just a scale.
- **looped-rand13 (`--n-loops-min=1 --n-loops-max=3`)**: every optimizer step draws its loop count uniformly from
  {1, 2, 3}, so the expected compute equals looped-l2's. Validation during training and the final CORE use 2
  loops. Per-step scalars are sized for 3 passes; more passes than that reuse the last pass's scalars.

**The per-loop embedding does nothing for bpb.** l2-embed tracked looped-l2 to within 0.0003 at every
checkpoint (0.8378 vs 0.8380 at the end). Its CORE is 0.005 higher, which is inside the 0.005–0.007
run-to-run spread measured below. Two passes with per-step scalars already know which pass they are in.

**Random loop count costs 0.008 bpb at the trained depth, and buys a model that works at any depth.** Both looped
checkpoints were re-evaluated at other loop counts (`loop_sweep` in train_d12.py; bpb on 21M val tokens, so it reads
about 0.001 higher than the 42M-token training eval; CORE as in training):

| Loops at eval | Effective layers | FLOPs/token | looped-l2 bpb | looped-l2 CORE | looped-rand13 bpb | looped-rand13 CORE |
|---|---|---|---|---|---|---|
| 1 | 12 | 7.60e8 (baseline's) | 1.1902 | 0.0785 | 0.8565 | 0.1709 |
| **2 (trained)** | 22 | 12.74e8 | **0.8391** | **0.1752** | 0.8474 | 0.1723 |
| 3 | 32 | 17.9e8 | 0.9511 | 0.1282 | 0.8479 | 0.1723 |
| 4 | 42 | 23.0e8 | 0.9322 | 0.1261 | 0.8505 | 0.1721 |
| 6 | 62 | 33.3e8 | 0.9836 | 0.1140 | 0.8619 | 0.1677 |

- **A fixed-loop model only works at its trained depth.** looped-l2 loses 0.35 bpb with one loop fewer and
  0.09–0.14 with any number more.
- **The random-loop model degrades gracefully.** It stays within 0.01 bpb from 1 to 4 loops, including 4, which it
  never trained at, and loses only 0.015 at 6. At 1 loop it runs at the baseline's cost and is 0.008 bpb behind the
  baseline (0.8565 vs 0.8480), while one checkpoint also serves the 2- and 3-loop budgets.
- **More loops at eval time do not help.** The best loop count is the trained mean (2). So random-loop training
  gives depth *flexibility*, not free test-time compute, at this scale and budget. Getting the extra loops to
  pay off would likely need what recurrent-depth papers use: many more loops, a heavy-tailed loop distribution,
  and backprop truncated to the last few iterations.

## Reading the results

- **Looping at the same parameter count (looped-l2) gives the biggest gain at equal tokens**, −0.010 bpb and
  +0.014 CORE, but it pays with 68% more compute. Spending that compute on more tokens instead (baseline-long)
  does better by another 0.010 bpb, so the gain comes from compute, not from looping (see above).
- **Diff V2 is the strongest of Carmen's four at equal tokens** (−0.0065 bpb, +0.012 CORE, clear at every
  checkpoint), but it also uses 19% more FLOPs and 8% more weights. Corrected for compute, its bpb gain
  disappears and only a borderline CORE gain remains.
- **Depth-Attention and MGVR are ties on bpb** (−0.0013, smaller than what a different seed would plausibly
  change). Their CORE moves in opposite directions by about 0.01, which is within CORE's noise at this size.
- **xIPReLU is slightly worse** on both metrics.
- **Of the looped extensions, only the random loop count changes anything**: it trades 0.008 bpb at the trained
  depth for a checkpoint that works from 1 to 6 loops. The per-loop embedding is a tie (see above).
- **Looping at the same compute loses**: +0.022 bpb with 42% fewer weights (u7), +0.008 with 25% fewer (u9). At d6 the equivalent
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
| looped-u9 (×2) | 40.2 min | 51.8 min | 545k | 41.9% | 27.6 GB | $3.41 each |
| baseline-long (×2) | 68.5 min | 85.0 min | 537k | 41.3% | 28.1 GB | $5.60 each |
| xIPReLU | 41.7 min | 53.5 min | 527k | 40.5% | 28.1 GB | $3.52 |
| Diff V2 | 51.3 min | 64.1 min | 427k | 39.0% | 32.8 GB | $4.22 |
| Depth-Attention | 43.1 min | 55.9 min | 508k | 39.1% | 29.8 GB | $3.68 |
| MGVR | 48.4 min | 61.2 min | 453k | 34.9% | 31.7 GB | $4.03 |
| looped-l2-embed | 67.8 min | 84.5 min | 323k | 41.6% | 45.1 GB | $5.57 |
| looped-rand13 | 69.1 min | 90.2 min | 355k | 45.6%* | 62.0 GB | $5.94 |
| **Total** (13 runs) | | **≈14.6 H100-hours** | | | | **≈$57.58** |

\* rand13's per-step MFU is computed against the *mean* FLOPs of 1–3 loops, so 1-loop steps read low and 3-loop
steps high; the average is not comparable to the other rows. Its peak memory is set by the 3-loop steps.

The loop-count sweep (10 evals of bpb on 21M tokens + CORE, one H100 each, in parallel) cost about $5 on top
(estimated from eval times; Modal reports no per-call cost). The two extension runs had their own $0.48 smoke test.

Add $0.87 for the 30-step smoke test of every config, run before the full runs. The first seven runs ran in
parallel (about 1.5 hours wall-clock); looped-u9 and baseline-long ran later as a second batch (about 1.4 hours), and
looped-l2-embed and looped-rand13 as a third (about 1.5 hours).

MGVR runs 12% slower than its FLOPs imply (35% MFU vs ~41%). Carmen's PR replicates the AdamW update for
MGVR's unevenly shaped matrices instead of sharding it. That is a cost of the implementation, not of the idea.

## Caveats

- **One seed per run.** Gaps under about 0.003 bpb, and CORE gaps under about 0.01, should be treated as ties.
- **Run-to-run noise, measured.** Modal preempted the CPU launcher ~11 minutes into the second batch, and on
  restart it spawned looped-u9 and baseline-long again, so both ran twice with the same seed. GPU
  nondeterminism alone moved val bpb by under 0.0002, but CORE by 0.005–0.007. So bpb differences are reliable
  at this size, and CORE differences under ~0.01 are not. (The duplicate cost ≈$9.)
- **looped-u7/u9 weight decay:** nanochat scales weight decay with the parameter count, so the runs with fewer
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
  offline-run-20261103_040507-y6nvh3l6 offline-run-20261103_143326-8q6ma4j8 \
  offline-run-20261103_143326-brihic8u offline-run-20261103_144732-hnvy3kv8 \
  offline-run-20261103_144732-ys0w0wws offline-run-20261103_144744-cqcpx9lb   offline-run-20261103_144743-3uaqex4a
```

These are, in order: baseline, looped-l2, looped-u7, xIPReLU, Diff V2, Depth-Attention, MGVR, looped-u9 (a),
baseline-long (a), looped-u9 (b), baseline-long (b), looped-l2-embed and looped-rand13. Only these full runs are committed; the smoke-test runs
and a stopped launch were left out. `out/logs` and `out/results` hold u9 (b) and baseline-long (a), since the
repeat of each wrote to the same volume path; baseline-long (b)'s metrics are in `bc_results.json`.
