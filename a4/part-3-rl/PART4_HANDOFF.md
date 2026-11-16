# Handoff: what Part 4 needs from Part 3

Part 4 asks for at least two new rewards based on the Part 3 patterns, using the same configuration as the
original RL run. Run #1 is the original reward (our Part 3 run, already done). Run #2 uses the new rewards
together. Runs #3 and #4 (or more) use each new reward on its own. Then compare the types of mistakes and
put everything in one summary table. Everything below is what you need to do that without redoing Part 3.

## 1. Run #1 (the baseline) is already done

Use the Part 3 run as the Part 4 baseline. Do not rerun it.

| | Run #1: original strict reward |
|---|---|
| Start checkpoint | d20 baseline-long SFT, step 932 |
| GSM8K full test, greedy | 2.73% (SFT) → **20.09%** (RL step 466) |
| pass@1 / pass@8 (400 Qs, T=1), step 420 | 0.190 / 0.335 |
| ARC-E / ARC-C / MMLU / HumanEval after RL | 57.79 / 46.42 / 35.44 / 11.59% |
| Per-question results | 246 fixed by RL, 17 broken, 19 both right, 1,037 both wrong |

All numbers and tables are in `results/` (`final_eval_rl.json`, `final_eval_sft.json`,
`fulltest_sft_vs_rl.csv`, `train_per_step.csv`, `eval_per_step.csv`, and more). Figures are in
`figures/`. The README explains each one.

## 2. Where everything lives

**Modal volume `vocalocate-a3`** (ask Tanay to invite you to the Modal workspace):

| What | Path |
|---|---|
| SFT start checkpoint (every Part 4 run starts here) | `/a3/sft_runs/d20-baseline-long-sft/chatsft_checkpoints/d20-baseline-long/model_000932.pt` (+ `meta_000932.json`) |
| Run #1 RL checkpoints (steps 60, 120, ..., 420, 466) | `/a3/rl_runs/d20-baseline-long-rl/chatrl_checkpoints/d20-baseline-long/` |
| Run #1 rollouts (every training and eval sample, 8 jsonl files, about 70 MB) | `/a3/rl_runs/d20-baseline-long-rl/rollouts/rank{0..7}.jsonl` |
| Run #1 training log | `/a3/logs/d20-baseline-long-rl.log` |
| Run #1 final eval (RL) and every greedy GSM8K answer | `/a3/rl_runs/d20-baseline-long-rl/final_eval/` |
| SFT final eval and every greedy GSM8K answer | `/a3/rl_runs/d20-baseline-long-rl-smoke/final_eval/` |
| Tokenizer | `/a3/tokenizer/` |

**Code:**
- Everything in this folder (`a4/part-3-rl/`).
- **nanochat** is the repo's own `a3/nanochat` on `main`: upstream karpathy/nanochat plus all of our A3
  changes. The launcher uses it by default, so `NANOCHAT_REPO` is only needed to point at another copy.
  This needs PR #44 merged: before #44, `main`'s copy could not load the baseline-long checkpoints. With
  #44, every checkpoint we trained loads with identical outputs (max logit difference 0.0).

## 3. How to launch a run

From `a4/part-3-rl/`, with the Modal CLI:

```bash
modal run --detach modal_chat_rl.py \
  --volume vocalocate-a3 --base-path a3 \
  --sft-tag d20-baseline-long-sft --model-tag d20-baseline-long --model-step 932 \
  --output-tag d20-baseline-long-rl-<name> --run-name d20-baseline-long-rl-<name>
```

- Use a **new `--output-tag`** for every run. The launcher resumes automatically if a tag already has
  checkpoints, so reusing a tag would continue the old run instead of starting a new one.
- Add `--smoke-steps 10` first for a 5-minute check (about $4).
- After training, run the same evaluation as run #1 on the final checkpoint:
  `... --output-tag <same tag> --run-name eval --final-eval rl:466`.
- The launcher starts the GPU job inside Modal and returns. Follow progress in
  `/a3/logs/<output-tag>.log` on the volume, which is saved every 5 minutes.

## 4. Adding a reward (the hook is ready)

Rewards are plugged in through `rewards.py`; nanochat itself is not edited.

1. Write a function in `rewards.py` and register it:
   ```python
   @register("my_format")
   def my_format(conversation, response, ctx):
       return float(ctx["pred"] is not None)  # ctx also has "correct", "ref", "ref_solution_parts"
   ```
2. Pick the training reward with `--reward` on the launcher. Components are summed, with optional weights:
   `--reward strict` (default, identical to run #1), `--reward strict,my_format:0.2`, `--reward my_partial`.
   An unknown name fails on your laptop before any GPU starts.

What the hook guarantees:
- **Only the training reward changes.** `GSM8K.evaluate` is untouched, so the in-training pass@k and the
  final `--final-eval` stay strict and comparable to run #1 and to Karpathy.
- **Logs keep both values.** Every training rollout records `correct` (the strict 0/1 check), `reward`
  (the shaped total) and `reward_parts` (each weighted component).
- **A run keeps its reward.** The reward spec is saved as `reward.txt` in the run folder, and an automatic
  resume refuses to continue a tag under a different `--reward`.
- `analysis/analyze_rollouts.py` reads the new fields. Its per-step table gains `accuracy` (strict),
  `reward` (shaped) and one `part_<name>` column per component. Zero-advantage groups are counted on the
  shaped reward, because the advantage is computed from it. The training-reward curve from
  `plot_curves.py` shows the shaped reward.

`example_has_marker` in `rewards.py` only shows the function shape and was used to test the hook. It is
not a proposed Part 4 reward.

## 5. What Part 3 found (the basis for the new rewards)

These are measurements from run #1. The reward ideas are suggestions; none were tested.

| Pattern in Part 3 | Number | A reward it suggests |
|---|---|---|
| Wrong answers end **without `####`**, stopping mid-reply on hard problems. Writing nothing and writing a wrong answer both score 0, so giving up costs nothing. | 34% of the RL model's wrong full-test answers; P(####) fell from 0.99 (step 30) to about 0.65 late | A small format reward for a well-formed `#### <number>`, or a penalty for ending without one |
| **Correct arithmetic, wrong quantity**: the final answer equals one of the model's own calculator results, or a reference intermediate value | 55% and 8% of the RL model's wrong full-test answers | Partial credit for calculator outputs that match the reference solution's `<<...>>` intermediate values (a denser, step-level reward) |
| **No gradient** from half the questions: all 16 samples scored the same, almost always all wrong | 50% of groups overall, 77% during the collapse | Any denser reward helps here, including the partial-credit reward above |
| **Collapse** at steps 60–100: up to 57% of rollouts contained non-ASCII junk and reward fell from 0.35 to 0.03 | non-ASCII share per step is in `results/train_per_step.csv` | A penalty for non-ASCII or repeated text, which may stabilize training |
| Difficulty is mainly **solution length** | full-test accuracy after RL: 2-step 39.8%, 4-step 11.4%, 6+-step 2.3% | Optionally scale the reward by the reference solution's step count |

Be careful with a format reward on its own. The model learned the format within 30 steps under the strict
reward anyway, and a reward that pays for format can be gamed by writing `####` with any number. Check
P(correct given ####), not just P(####).

## 6. Comparing mistakes across runs

`analysis/analyze_rollouts.py` produces the Part 3 figures for **any** run: the failure categories by
phase and eval step, solve rate by solution length, clusters, and the SFT-vs-RL full-test comparison. To
compare runs, fetch each run with `analysis/fetch_run.py <output-tag> <local dir>` and run the analysis on
each one. The categories are identical, so the CSVs (`train_failure_mix_by_phase.csv`,
`eval_failure_mix_by_step.csv`, `fulltest_sft_vs_rl.csv`) can be put side by side.

What is not built yet: a single plot that overlays several runs. `analysis/plot_curves.py` currently
plots one run against Karpathy and needs a small extension to accept several logs.

## 7. Cost, time and gotchas

- **Per full run:** about 70 minutes of training on 8×H100, about $40–45 at roughly $35 an hour. The
  final eval adds about 15 minutes, about $8.
- **Part 4 needs at least 3 new runs** (all rewards together, plus each reward alone): budget roughly
  **$150–180**. Part 3 itself cost about $90–100, because of a preemption and an aborted attempt.
- **Preemption:** Modal stopped run #1 at step 262. The launcher now resumes from the latest checkpoint
  (saved every 60 steps), but `chat_rl.py` never saves optimizer state, so a resume restarts the optimizer.
  Mention it in the write-up if it happens.
- **GPU capacity:** a resume once waited about 11 minutes for 8 free H100s.
- **Noise:** with 400 eval questions, one standard error on pass@1 near 0.15 is about ±0.018. Compare
  runs on the **full 1,319-question test** with the paired per-question comparison
  (`analyze_rollouts.py --final-eval` reports an exact McNemar test).
- **One seed:** run #1 collapsed and recovered on its own. Without repeated seeds we cannot tell how much
  run-to-run variation there is. Treat small differences between runs carefully.
- **Same configuration:** keep every `chat_rl.py` default (467 steps, 16 × 16 samples, LRs, temperature,
  256 new tokens). Only `--reward` should change.
- **Reward scale:** the advantage is reward minus the group mean, with no normalization. Adding components
  changes the size of the updates, so a reward that can reach 1.5 effectively raises the step size. Keep the
  total in a similar range to the strict reward (0–1), or say in the write-up that you did not.
