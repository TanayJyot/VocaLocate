# A3 Part 2: d12 picochat ablations

Thirteen d12 configurations (twenty-one runs), each on one Modal H100 (80 GB, Flash Attention 3), trained 2026-11-02 to 11-05.
Script: [train_d12.py](train_d12.py). Per-run logs and result JSONs are in `out/logs/` and `out/results/`.

## Setup (identical for every run)

| | |
|---|---|
| Model | nanochat d12, width 768, head dim 128, 2048-token context, `SSSL` sliding windows |
| Data | ClimbMix shards 0–31 (the same order for every run), tokenizer: 32,768 vocab |
| Budget | 2,520 steps × 524,288 tokens = **1.32B tokens** (12 tokens per scaling param of the baseline, nanochat's default) |
| Batch | 524,288 tokens/step (nanochat's d12 reference), device batch 32 × 2048 |
| Seed | 42 (nanochat default), one run per config; the baseline, looped-l2 and looped-l2-embed were also run at seeds 1 and 2 (`NANOCHAT_SEED`, init only) |
| Eval | val bpb on 42M held-out tokens every 250 steps; CORE (22 tasks, ≤500 examples each) at the end |
| Code | VocaLocate PR #27 (`feat/a3-other-ablation-ideas`, `04ab375`); each run turns on one change with `--experiment=`. `--loop-embed`, `--n-loops-min/max` and the `xsa`/`mudd` experiments were added on this branch (PR #28) |

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
| **XSA** | each head removes its output's component along its own value (learned gate, zero-init) | 84.9M (+72 scalars) | 7.60e8 | **0.8462** | **−0.0018** | 0.1595 | −0.0013 |
| MUDD-lite | token-dependent mixing of earlier hidden states on the last 2 layers | 84.9M (+0.1M other) | 7.60e8 (+0.08%) | 0.8473 | −0.0007 | 0.1510 | −0.0098 |
| baseline, seed 1 | — (different init) | 84.9M | 7.60e8 | 0.8479 | −0.0001 | 0.1565 | −0.0043 |
| baseline, seed 2 | — (different init) | 84.9M | 7.60e8 | 0.8469 | −0.0011 | 0.1439 | −0.0169 |
| looped-l2, seed 1 | — (different init) | 84.9M | 12.74e8 (+68%) | 0.8375 | −0.0105 | 0.1740 | +0.0132 |
| looped-l2, seed 2 | — (different init) | 84.9M | 12.74e8 (+68%) | 0.8375 | −0.0105 | 0.1576 | −0.0032 |
| looped-l2-embed, seed 1 | — (different init) | 84.9M (+1.5k other) | 12.74e8 (+68%) | 0.8373 | −0.0107 | 0.1781 | +0.0173 |
| looped-l2-embed, seed 2 | — (different init) | 84.9M (+1.5k other) | 12.74e8 (+68%) | 0.8375 | −0.0105 | 0.1703 | +0.0095 |

† Mean of two identical runs (same config and seed; a launcher preemption ran each twice, see Caveats).
looped-u9: 0.8559 / 0.8561 bpb, CORE 0.1456 / 0.1526. baseline-long: 0.8280 / 0.8279 bpb, CORE 0.1690 / 0.1643.
All Δ are against the seed-42 baseline; the seed-1/2 rows show how far a different init alone moves it.

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
  large as looped-l2's original gain over the baseline. Its seed-42 CORE edge (+0.009) did not hold up:
  over seeds 42, 1 and 2 looped-l2 averages 0.1689, only +0.002 over baseline-long's 0.1667 (see the looped extensions).
- **Diff V2's bpb gain is fully explained by its extra compute.** Its CORE gain (≈+0.010) survives the
  correction but is at the edge of noise. The interpolation is an estimate; an exact control would be a
  baseline run at ~3,000 steps.
- **The same-compute looped runs lose less as sharing shrinks.** With 25% fewer weights (u9) the loss is
  +0.008 bpb, with 42% fewer (u7) +0.022. Neither ties the baseline, so at this token budget weight sharing
  costs quality rather than saving it for free.

## Same-compute candidates from the 2026 speedrun records: XSA and MUDD-lite

Two changes taken from modded-nanogpt records that Karpathy never ran in this form, both with no meaningful
extra compute, plus the baseline at two more seeds to measure how much the init alone moves the result:

- **XSA (`--experiment=xsa`)**, Exclusive Self-Attention (arXiv 2603.09078) in the learnable form of speedrun
  record #82: after attention, each head subtracts a fraction tanh(α) of its output's component along the token's
  own value vector. α is one scalar per head and layer, zero-initialised, so training starts as the baseline.
  Karpathy's nanochat log only tried a fixed version on the last 3 layers.
- **MUDD-lite (`--experiment=mudd`)**, after speedrun record #81: on the last two layers only, a small MLP
  (768→64→6) on the hidden state gives per-token weights over the input embedding, the mid-network residual
  (nanochat's backout point) and the layer's own output. Layer 10 adds the mix to the residual and to layer 11's
  values; layer 11 adds a second mix to the final residual. Output weights start at zero, so it starts as the
  baseline. It differs from Depth-Attention, which mixed values across all layers with no learned weights.

Both runs use seed 42, and every weight they share with the baseline starts identical (checked in
`tests/ablations/test_xsa_mudd.py`).

| Run | Val bpb | Δ vs seed-42 baseline | Δ vs 3-seed baseline mean | CORE |
|---|---|---|---|---|
| baseline, seeds 42 / 1 / 2 | 0.8480 / 0.8479 / 0.8469 | — | mean **0.8476**, sd 0.0006 | 0.1608 / 0.1565 / 0.1439 (mean 0.1537, sd 0.009) |
| **XSA** | **0.8462** | **−0.0018** | **−0.0014** | 0.1595 |
| Depth-Attention | 0.8467 | −0.0013 | −0.0010 | 0.1510 |
| MGVR | 0.8467 | −0.0013 | −0.0009 | 0.1637 |
| MUDD-lite | 0.8473 | −0.0007 | −0.0003 | 0.1510 |

Validation bpb during training:

| Step | baseline s42 | baseline s1 | baseline s2 | XSA | MUDD-lite |
|---|---|---|---|---|---|
| 500 | 1.0062 | 1.0076 | 1.0056 | 1.0034 | 1.0030 |
| 1000 | 0.9491 | 0.9472 | 0.9474 | 0.9467 | 0.9469 |
| 1500 | 0.9049 | 0.9046 | 0.9040 | 0.9030 | 0.9036 |
| 2000 | 0.8715 | 0.8712 | 0.8703 | 0.8693 | 0.8705 |
| 2520 | 0.8480 | 0.8479 | 0.8469 | 0.8462 | 0.8473 |

- **The seed matters more than the earlier same-seed repeats suggested.** A different init moves final val bpb
  by up to 0.0011 (sd 0.0006), against under 0.0002 for a same-seed rerun. The seed-42 baseline happens to be the
  worst of the three, so every Δ against it reads about 0.0004 too favourable.
- **XSA is the best same-compute change so far, but only just.** It is 0.0007 below the best baseline seed and
  0.0014 below the mean, ahead at every checkpoint from step 500. That is about two baseline standard deviations,
  from one XSA seed. It beats Depth-Attention and MGVR by 0.0005, which is inside the seed noise.
- **MUDD-lite is a tie.** It sits inside the baseline's seed range (−0.0003 vs the mean). Its early lead
  (step 500) is gone by step 2000. Restricting cross-layer mixing to the last two layers and learning the
  weights did not rescue Depth-Attention's idea at this size.
- **CORE cannot rank these runs.** The three baseline seeds alone span 0.017 CORE (sd 0.009), wider than every
  same-compute CORE gap in the table.

## Looped extensions: per-loop embedding and random loop count

Both start from looped-l2 (1 prelude + 10 core blocks + 1 coda, same tokens, same seed) and change one thing:

- **looped-l2-embed (`--loop-embed`)**: a learned 768-dim vector per core pass is added to the residual stream at
  the start of that pass, so the shared blocks can tell pass 1 from pass 2. It is zero-initialised, so the run
  starts exactly as looped-l2. Each executed step already had its own `resid_lambdas`/`x0_lambdas` scalars;
  this adds a direction, not just a scale.
- **looped-rand13 (`--n-loops-min=1 --n-loops-max=3`)**: every optimizer step draws its loop count uniformly from
  {1, 2, 3}, so the expected compute equals looped-l2's. Validation during training and the final CORE use 2
  loops. Per-step scalars are sized for 3 passes; more passes than that reuse the last pass's scalars.

**The per-loop embedding ties on bpb, and its CORE lead is not conclusive.** Both models were run at seeds 42, 1
and 2 (init only; the data order is unchanged):

| Seed | looped-l2 bpb | embed bpb | looped-l2 CORE | embed CORE | CORE gap (embed − looped-l2) |
|---|---|---|---|---|---|
| 42 | 0.8380 | 0.8378 | 0.1752 | 0.1806 | +0.0054 |
| 1 | 0.8375 | 0.8373 | 0.1740 | 0.1781 | +0.0041 |
| 2 | 0.8375 | 0.8375 | 0.1576 | 0.1703 | +0.0127 |
| **Mean** | **0.8377** | **0.8375** | **0.1689** | **0.1763** | **+0.0074** |

- **bpb is a tie.** The mean gap is 0.0001. At seed 42, l2-embed tracked looped-l2 to within 0.0003 at every
  checkpoint. The per-step scalars already tell the passes apart, which may be why bpb does not move.
- **CORE favours the embedding on all three seeds, but three seeds cannot settle it** (mean +0.0074, paired
  t-test p ≈ 0.11).
- **Both looped models keep their gain over the baseline.** Against the 3-seed baseline mean (0.8476, 0.1537),
  looped-l2 is −0.010 bpb and +0.015 CORE, and l2-embed is −0.010 bpb and +0.023 CORE. Looped-l2's seed-2 CORE
  (0.1576) is still inside the baseline's seed range.
- **Against baseline-long (0.8280, 0.1667; two same-seed runs),** both lose clearly on bpb. Looped-l2's mean CORE
  is too close to call (+0.002). L2-embed's is +0.010, but baseline-long has only been run at one seed.

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
  does better by another 0.010 bpb, so the gain comes from compute, not from looping (see above). Across three seeds, looped-l2 averages −0.010 bpb and +0.015 CORE against the baseline's 3-seed mean.
- **Diff V2 is the strongest of Carmen's four at equal tokens** (−0.0065 bpb, +0.012 CORE, clear at every
  checkpoint), but it also uses 19% more FLOPs and 8% more weights. Corrected for compute, its bpb gain
  disappears and only a borderline CORE gain remains.
- **Depth-Attention and MGVR are small gains or ties on bpb**: −0.0013 against the seed-42 baseline, but only
  −0.0009/−0.0010 against the 3-seed mean, which is about 1.5 baseline standard deviations. Their CORE moves in
  opposite directions by about 0.01, inside CORE's seed noise (sd 0.009).
- **XSA is the strongest same-compute change** (−0.0014 vs the baseline mean, beyond the baseline's seed
  range), and MUDD-lite is a tie (see above). XSA's margin over Depth-Attention and MGVR is within seed noise.
- **xIPReLU is slightly worse** on both metrics.
- **Of the looped extensions, only the random loop count changes anything**: it trades 0.008 bpb at the trained
  depth for a checkpoint that works from 1 to 6 loops. The per-loop embedding ties on bpb, and its CORE lead over three seeds is not conclusive (see above).
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
| XSA | 42.2 min | 55.6 min | 520k | 39.9% | 30.4 GB | $3.66 |
| MUDD-lite | 42.2 min | 54.8 min | 520k | 40.0% | 28.6 GB | $3.60 |
| baseline, seed 1 | 40.3 min | 52.1 min | 543k | 41.7% | 28.1 GB | $3.43 |
| baseline, seed 2 | 40.9 min | 53.6 min | 536k | 41.2% | 28.1 GB | $3.53 |
| looped-l2, seed 1 | 67.5 min | 82.9 min | 325k | 41.8% | 45.1 GB | $5.46 |
| looped-l2, seed 2 | 66.9 min | 84.4 min | 328k | 42.2% | 45.1 GB | $5.55 |
| looped-l2-embed, seed 1 | 67.1 min | 84.2 min | 327k | 42.1% | 45.1 GB | $5.54 |
| looped-l2-embed, seed 2 | 67.9 min | 85.9 min | 323k | 41.6% | 45.1 GB | $5.65 |
| **Total** (21 runs) | | **≈23.8 H100-hours** | | | | **≈$94.00** |

\* rand13's per-step MFU is computed against the *mean* FLOPs of 1–3 loops, so 1-loop steps read low and 3-loop
steps high; the average is not comparable to the other rows. Its peak memory is set by the 3-loop steps.

The loop-count sweep (10 evals of bpb on 21M tokens + CORE, one H100 each, in parallel) cost about $5 on top
(estimated from eval times; Modal reports no per-call cost). The two extension runs had their own $0.48 smoke test.

Add $0.87 for the 30-step smoke test of every config, run before the full runs. The first seven runs ran in
parallel (about 1.5 hours wall-clock); looped-u9 and baseline-long ran later as a second batch (about 1.4 hours), and
looped-l2-embed and looped-rand13 as a third (about 1.5 hours), and XSA, MUDD-lite and the two baseline seeds as
a fourth (about 0.9 hours, after a $0.45 smoke test). XSA and MUDD-lite train about 4% slower per second than the
baseline (elementwise work outside the matmuls); at a fixed step count that changes cost, not results.

The four seeded looped runs ran in parallel as their own batch (about 1.4 hours), with no smoke test and no
preemption.

MGVR runs 12% slower than its FLOPs imply (35% MFU vs ~41%). Carmen's PR replicates the AdamW update for
MGVR's unevenly shaped matrices instead of sharding it. That is a cost of the implementation, not of the idea.

## Caveats

- **One seed per run, except the baseline, looped-l2 and looped-l2-embed (three seeds each).** For the baseline, a
  different init moves val bpb by sd 0.0006 (range 0.0011) and CORE by sd 0.009 (range 0.017). So single-run bpb gaps
  under about 0.0012 and CORE gaps under about 0.02 are ties. The looped models move similarly on CORE (ranges 0.018
  and 0.010) and less on bpb (range 0.0005 for both).
- **Same-seed run-to-run noise, measured.** Modal preempted the CPU launcher ~11 minutes into the second batch, and on
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

All 21 runs are uploaded to the W&B project
[vocalocate-a3](https://forge.coreweave.com/wandb/tanayjyot-university-of-toronto/vocalocate-a3/workspace), tagged
`baseline`, `pr27`, `looped`, `extension`, `speedrun`, `control`, `repeat` or `seed`. The figures in `figures/` are rebuilt from the committed logs
with `python plots.py`.

The runs were logged with `WANDB_MODE=offline` and saved in `out/wandb/wandb/`. To upload them to another W&B
account (run `wandb login` first):

```bash
cd out/wandb/wandb
wandb sync offline-run-20261103_040506-2msa7208 offline-run-20261103_040529-e4cyz84k \
  offline-run-20261103_042935-cmrws97h offline-run-20261103_040531-cxowofgs \
  offline-run-20261103_040506-u4asolds offline-run-20261103_040507-kn36s4nh \
  offline-run-20261103_040507-y6nvh3l6 offline-run-20261103_143326-8q6ma4j8 \
  offline-run-20261103_143326-brihic8u offline-run-20261103_144732-hnvy3kv8 \
  offline-run-20261103_144732-ys0w0wws offline-run-20261103_144744-cqcpx9lb \
  offline-run-20261103_144743-3uaqex4a \
  offline-run-20261103_190310-hwxzvtrl offline-run-20261103_190311-t9k0bjol \
  offline-run-20261103_190312-24wheiew offline-run-20261103_190316-cgwmsfqi \
  offline-run-20261106_002625-pzodxqp5 offline-run-20261106_002630-qvumz6vt \
  offline-run-20261106_002642-w6cud06w offline-run-20261106_002643-efl63fy0
```

These are, in order: baseline, looped-l2, looped-u7, xIPReLU, Diff V2, Depth-Attention, MGVR, looped-u9 (a),
baseline-long (a), looped-u9 (b), baseline-long (b), looped-l2-embed, looped-rand13, XSA, MUDD-lite, baseline
seed 1, baseline seed 2, looped-l2 seeds 1 and 2, and looped-l2-embed seeds 1 and 2. The seeded looped runs' metrics are
in `seeds_l2_results.json`; their run folders are not committed under `out/` yet. Only full runs are committed; the smoke-test runs
and a stopped launch were left out. `out/logs` and `out/results` hold u9 (b) and baseline-long (a), since the
repeat of each wrote to the same volume path; baseline-long (b)'s metrics are in `bc_results.json`.
