# Part 4: Reward experiments on GSM8K

Two rule-based rewards extend Part 3's correctness-only RL experiment: complete final-answer formatting and coverage of reference intermediate calculations. Each new run starts independently from the same Standard SFT checkpoint, `d20-baseline-long-sft` at step 932. The existing Part 3 RL model is the evaluation baseline.

## Results

All models use the same 1319 GSM8K test questions, greedy decoding, a 512-token completion limit and the original correctness scorer.

| Training reward | Correct | Accuracy | Difference from baseline |
| --- | ---: | ---: | ---: |
| Correctness only, from Part 3 | 265 | 20.09% | - |
| Correctness + format | 260 | 19.71% | -0.38 pp |
| Correctness + calculations | 270 | 20.47% | +0.38 pp |
| Correctness + both | 264 | 20.02% | -0.08 pp |

All three paired 95% bootstrap intervals include zero. The rewards change answer extraction and which questions are solved, without establishing a final accuracy advantage over the baseline. `results/final_analysis.json` contains aggregate results and `results/final_per_question.csv` contains paired predictions and error categories.

## Reward definitions

`C` is Part 3's original correctness score. `F` is 1 when the response has exactly one `####` marker and its last nonempty line is a complete `#### <number>` answer, and 0 otherwise. The format check never changes the original evaluation scorer.

`S` is the fraction of distinct, valid reference calculations matched by valid generated calculator calls before the first answer marker, or in the entire response if there is no marker. Reference calculations whose result equals the final answer are excluded. `S` is 0 when no reference calculations remain. Addition and multiplication accept reordered and regrouped operands. Subtraction and division retain operand order. Duplicate calculations count once.

| Run | Training reward |
| --- | --- |
| Correctness only | `C` |
| Format | `C + 0.1 (1 - C) F` |
| Calculations | `C + 0.1 (1 - C) S` |
| Combined | `C + 0.1 (1 - C) (F + S)` |

Correct responses always receive reward 1. Incorrect responses receive at most 0.1 with either individual addition and at most 0.2 with both. The arithmetic parser uses a bounded syntax tree without executing generated code. Matching calculations does not establish correct use of quantities or recognition of alternative solution methods.

## Code and saved evidence

- `../part-3-rl/rewards.py`: reward functions, restricted arithmetic parsing and component registration.
- `runs.json`: initialization, training and evaluation configuration, and the three reward specifications.
- `modal_part4.py` and `run_protocol.py`: sequential training and evaluation, checkpoint continuation and output validation. Part 3's training and evaluation implementations are reused.
- `audit_rewards.py`: training-only replay of the reward rules on saved Part 3 responses.
- `analyze_results.py`: raw-output checks, shared error categories, calculation coverage and paired bootstrap comparisons.
- `plot_results.py`: the seven report figures from saved result tables.
- `results/`: per-step training measurements, periodic evaluations, per-question final results, selected model outputs and input verification records.
- `figures/`: generated report figures.

The replay audit covers 59776 training responses in 3736 complete groups. Of 1637 all-wrong groups, format assigns differing scores within 80.0%, calculations within 65.1%, and both within 91.3%. The input hashes and counts are recorded in `results/reward_audit.json`. This audit measures the availability of a reward signal, not the effect of training on test accuracy.

Each new full run uses 467 optimizer updates, 16 questions per update and 16 sampled responses per question. Sampling uses temperature 1, top-k 50 and at most 256 new tokens. The data order, tokenizer, optimizer settings and learning-rate schedule match Part 3. Periodic tests use the same 400 questions with eight samples each. The format model continues from checkpoint 360 at step 361 with a fresh optimizer because optimizer state was not saved. Analysis retains each training step once. The calculation and combined runs complete without interruption.

## Reproduce figures and run checks

From the repository root:

```powershell
python -m pip install -r a4/part-4-rewards/requirements-analysis.txt
python -m unittest discover -s a4/part-4-rewards -p "test_*.py" -v
python a4/part-4-rewards/plot_results.py
```

These checks and figure generation use local code and the compact result files. They do not start training or require model weights. Bootstrap intervals use 20000 paired resamples of test-question results with seed 17011, keeping trained models fixed. They do not measure variability across training seeds.

Running `analyze_results.py` again requires the original raw outputs. Place the three runs' `rollouts/rank*.jsonl`, `final_eval/gsm8k_*rank*.jsonl` and `training.log` under `local/report_data/{format,reference,combined}/`. Also supply the original GSM8K training JSONL as `local/gsm8k_train.jsonl`, the cached test parquet under `local/artifacts/a3/sft_runs/d20-baseline-long-sft/task_data/openai--gsm8k/main/test/00000.parquet`, and Part 3's original training log as `local/tanay_baseline/training.log`. Part 3's compact CSV results are read from `../part-3-rl/results/`.

## Training launcher

The manifest records the workspace and volume used for the experiments. Training requires Part 3's training dependencies, access to that environment, the SFT checkpoint and tokenizer, the GSM8K cache, and a successful separate 10-step smoke run. Training commands below launch paid compute.

```powershell
$env:MODAL_PROFILE = "justinzhaoovo"
$env:RL_VOLUME = "vocalocate-a4-part4"
python -m modal run --detach a4/part-4-rewards/modal_part4.py::start --manifest a4/part-4-rewards/runs.json
```

The suite runs each experiment and its final evaluation sequentially and stops after a failed stage. A fresh launch refuses existing output tags. `--resume` continues an inspected failed suite, preserves completed experiments, archives partial records and resumes from the latest saved model. The manifest's experiment settings must match the saved run. Raw datasets, model weights, logs, report drafts and local analysis intermediates are ignored by Git.

## References

- DeepSeek-AI (2025), [DeepSeek-R1](https://arxiv.org/abs/2501.12948v1), Section 2.2.2: separate rule-based accuracy and format rewards.
- Lightman et al. (2023), [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050): motivation for feedback on intermediate reasoning. The reference-matching rule here is not a learned process verifier.
