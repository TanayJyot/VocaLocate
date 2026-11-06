# A3 Part 2: d12 picochat ablations

Baseline, Diff V2 and looped-l2-embed at d12, each on one Modal H100 with Flash Attention 3.
Script: [train_d12.py](train_d12.py). Every other experiment is on the `a3-archive` branch and W&B project.

| | |
|---|---|
| Model | nanochat d12, width 768, head dim 128, 2048-token context, `SSSL` sliding windows |
| Data | ClimbMix shards 0-31, the same order for every run, 32,768-token vocabulary |
| Budget | 2,520 steps x 524,288 tokens = 1.32B tokens |
| Seeds | 42, plus 1 and 2 for the baseline and looped-l2-embed |
| Eval | val bpb on 42M held-out tokens every 250 steps, CORE (22 tasks) at the end |

| Run | What changes | FLOPs/token | Val bpb | CORE |
|---|---|---|---|---|
| baseline, seed 42 | none | 7.60e8 | 0.8480 | 0.1608 |
| baseline, seed 1 | different init | 7.60e8 | 0.8479 | 0.1565 |
| baseline, seed 2 | different init | 7.60e8 | 0.8469 | 0.1439 |
| baseline-long | 4,224 steps, looped-l2-embed's total compute | 7.60e8 | 0.8280 | 0.1667 |
| Diff V2 | paired queries, learned subtraction | 9.02e8 | 0.8415 | 0.1727 |
| looped-l2-embed, seed 42 | 10 middle blocks run twice, with a learned vector per pass | 12.74e8 | 0.8378 | 0.1806 |
| looped-l2-embed, seed 1 | different init | 12.74e8 | 0.8373 | 0.1781 |
| looped-l2-embed, seed 2 | different init | 12.74e8 | 0.8375 | 0.1703 |

W&B: https://forge.coreweave.com/wandb/tanayjyot-university-of-toronto/a3-final
