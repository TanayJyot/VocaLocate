# Part 3: Replicating Karpathy's GSM8K RL run on d20 baseline-long

We ran nanochat's stock `scripts/chat_rl.py` with **every hyperparameter at its default**: 467 steps,
16 questions × 16 samples per step, temperature 1.0, top-k 50, at most 256 new tokens, Muon/Adam LRs
0.02 / 0.2 / 0.004 at 5% of base with a linear ramp to zero, and pass@k evaluated every 60 steps on 400
test questions. The start point is our **d20 baseline-long SFT checkpoint** (step 932, see Part 2). The
run used 8×H100 on Modal, which is Karpathy's setup.

| | Ours (d20 baseline-long) | Karpathy d20 ("d20speed") |
|---|---|---|
| Base model | d20, 21,328 steps, 11.2B tokens (equal compute to our looped + diff-attention model) | d20 "$100 speedrun" |
| Mid-training before SFT | none (pretrain → SFT) | yes (pretrain → mid → SFT) |
| RL config | stock defaults | stock defaults |
| RL wall clock, 8×H100 | 63 min to the step-420 eval; 69.0 min for all 467 steps | about 87 min to the step-420 eval |
| GSM8K full test, greedy, before → after RL | **2.73% → 20.09%** | **4.55% → 7.58%** |
| In-training pass@1 (400 Qs, T=1.0), step 0 → 420 | **0.020 → 0.190** | about 0.023 → 0.062 |
| In-training pass@8, step 0 → 420 | **0.128 → 0.335** | about 0.148 → 0.210 |

### Full evaluation before and after RL

Stock `chat_eval.py` on the full test sets (greedy, 512 new tokens), the protocol of Karpathy's report
card, run on both checkpoints by the same launcher (`--final-eval`):

| Task | SFT (step 932) | RL (step 466) | Change |
|---|---|---|---|
| ARC-Easy | 58.75% | 57.79% | −0.96 |
| ARC-Challenge | 46.50% | 46.42% | −0.08 |
| MMLU | 36.26% | 35.44% | −0.82 |
| **GSM8K** (1,319) | **2.73%** | **20.09%** | **+17.36** |
| HumanEval (164) | 15.24% | 11.59% | −3.65 |
| ChatCORE | 0.2133 | 0.2357 | +0.022 |

RL is trained on GSM8K only. GSM8K improves about 7×; the multiple-choice tasks are unchanged within
noise; HumanEval drops 6 problems (25 → 19 of 164), which is a small sample but in the direction of
a small loss of coding ability.

## Where Karpathy's numbers come from

His raw logs are not public: the W&B projects are not readable and the Hugging Face repo holds weights
only. The only curves are a W&B screenshot of run `d20speed` in
[nanochat discussion #1](https://github.com/karpathy/nanochat/discussions/1), with three panels:
reward against step, pass@1 against step, and pass@8 against **minutes**. We digitized it
(`analysis/digitize_karpathy.py`): axes are calibrated from the gridlines and ticks detected in the
image, and the series colour is traced per pixel column. The overlay check is
`reference/karpathy_d20_digitize_check.png`. Reading error is about ±0.001 for reward and ±0.0003 for
pass@k. The reward panel is W&B's *smoothed* line; the raw series cannot be recovered. His pass@8
points are mapped in order to the evals at steps 0, 60, …, 420 when plotted against steps. His
full-test numbers (4.55% → 7.58%) come from the report card in the same discussion.

## Figures

| Figure | What it shows |
|---|---|
| `figures/fig_rl_reward.png` | training reward vs step, ours (raw + EMA) vs Karpathy (W&B-smoothed) |
| `figures/fig_rl_passk_step.png` | pass@1 and pass@8 vs step, both runs |
| `figures/fig_rl_pass8_minutes.png` | pass@8 vs wall-clock minutes, matching Karpathy's third panel |
| `figures/fig_rl_seqlen.png` | mean rollout length (ours only; Karpathy did not plot it) |
| `figures/fig_eda_train_dynamics.png` | reward = P(####) × P(correct given ####), completion length, degenerate rollouts, zero-advantage questions |
| `figures/fig_eda_eval_dynamics.png` | the same decomposition on the eval set, and question turnover between evals |
| `figures/fig_eda_failure_mix.png` | why answers are wrong, by training phase and by eval step |
| `figures/fig_eda_difficulty.png` | solve rate by reference solution length and by topic, early/mid/late |
| `figures/fig_eda_clusters.png` | TF-IDF k-means clusters of the questions × training phase |
| `figures/fig_eda_fulltest_sft_vs_rl.png` | greedy full test, SFT vs RL: failure mix and accuracy by solution length |

## 1. Our run vs Karpathy's

**Eval curves.** Both runs start at the same place (pass@1 0.020 vs about 0.023), but ours climbs about
3× further: pass@1 0.190 vs 0.062 at step 420, and pass@8 0.335 vs 0.210. Most of our gain is in the
first 60 steps (pass@1 0.020 → 0.108), where Karpathy's moved only 0.023 → 0.030.

**Reward curves.** Ours reaches a training reward of about 0.35 within 30 steps and ends around 0.34;
his rises slowly from 0.02 to about 0.06. The two are on very different scales, and the gap is
larger than the eval gap, so part of it is that the *training* questions are easier for our model than
the test questions (training reward 0.34 vs eval accuracy 0.17 at the end). Karpathy's training reward
and pass@1 are both about 0.06, so his model has no such gap.

**The collapse.** Our run is not monotone. Between steps 60 and 100 the training reward fell from 0.35
to 0.03, the share of rollouts that write `####` fell from 99% to 14%, and up to 57% of rollouts
contained non-ASCII junk (GSM8K solutions are plain ASCII). By step 87 most samples were gibberish. It
then recovered by itself: reward was back above 0.2 by step 110 and around 0.3 from step 200 on. The
evals at 60 and 120 straddle the collapse, so the pass@k curve barely shows it (pass@8 actually dips
from 0.340 to 0.303). Karpathy's run shows nothing similar, but other course teams reported collapses
with the same defaults, so it is not unique to our model.

**Likely causes of the differences.**
1. *Start point.* Our SFT model writes `####` in only 41% of eval samples at step 0, against 92% by
   step 60. A large share of the early gain is format learning. The reward requires the marker, so
   learning to write it is cheap reward, and that is exactly what the decomposition plot shows: P(####)
   goes from 0.74 to 0.99 in the first 30 steps while P(correct given ####) moves less.
2. *Lineage.* Karpathy's model had mid-training (which teaches the tool-use and conversation format)
   before SFT; ours went straight from pretraining to SFT, so it starts less fluent at the format but
   with more headroom. Our base model was also trained on twice the tokens of our own original d20
   baseline (val bpb 0.717 vs 0.736); we have no matching number for Karpathy's base model.
3. *Code version.* Since his run, upstream merged the optimizer into MuonAdamW and removed autocast. The
   RL hyperparameters are identical, but the numerics are not.
4. *The collapse.* Our hypothesis, which we did not test: `chat_rl.py` has no KL penalty and no
   clipping, and its LR is highest at the start of the ramp. Once almost every rollout writes `####`
   (by step 30), the remaining signal is correctness alone, and large unconstrained updates can push the
   model off its language distribution. Recovery fits the same picture: junk rollouts all score 0, so
   those questions give no gradient, while questions that still yield some correct samples pull the
   model back. Testing this would need a rerun with a lower LR or a KL term.
5. *Noise.* The in-training eval uses 400 questions; one standard error on pass@1 near 0.15 is about
   ±0.018.

**Wall clock.** Our step-420 eval finished at 63 minutes, his at about 87 (`fig_rl_pass8_minutes.png`);
our whole loop took 69 minutes. Our completions got short fast (175 → about 78 generated tokens), which
makes sampling cheaper.

**Preemption.** Modal preempted the run at step 262. The launcher resumed from the step-240 checkpoint
at step 241, with the same LR schedule and data position. `chat_rl.py` does not save optimizer state,
so the optimizer restarted at step 241; the training reward shows no visible discontinuity there. The
redone steps 241–261 from the first attempt were removed from the logs.

## 2. Which problems it gets right and wrong

All 119,552 training rollouts and 25,600 eval samples were logged (`rl_rollout_logger.py`). Training is
a single epoch, so each training question is sampled 16 times at one step only. Changes over time are
therefore read across questions (early: steps 0–154, mid: 155–310, late: 311–466), and per-question
trajectories come from the 400 eval questions, which are re-sampled at every eval.

**Difficulty is mostly solution length.** Solve rate peaks for 2-step reference solutions (0.42) and
falls steadily with more steps: 0.30 (3 steps), 0.20 (4), 0.11 (5), 0.03 (6+). One-step problems
(0.20) score lower than two-step ones; we did not investigate why. Problems whose reference solution
divides are harder (0.23 vs 0.31).

**Topic.** Percent questions are the hardest throughout (0.15 early → 0.27 late); "other" (counting,
totals) gains most (0.25 → 0.44). Every topic improves from the early to the late phase.

**Clusters.** TF-IDF k-means (k = 8) on the question text finds topic clusters (ages, time/distance,
weights, money, marbles, books, fruit, generic totals). Solve rate ranges only from 0.20 (age problems)
to 0.32 (fruit), a much weaker separation than solution length, so the wording of a problem matters less
than how many steps it needs.

**Why answers are wrong** (rule-based, first match wins; see the figure legend for the order):
* Early on, the largest category is *a wrong answer that equals one of the model's own calculator
  results* (44% of wrong answers): the arithmetic is right but it computed the wrong quantity or stopped
  at an intermediate step. A further 8% give one of the *reference's* intermediate values.
* Late in training the largest category becomes *the reply ends without `####`* (52%). These replies
  are short and mostly stop mid-solution: of the 5,685 such rollouts after step 400, 61% end on a
  letter, i.e. mid-word or mid-sentence (one is just "First first"). None reach the 256-token limit.
  This started with the collapse and never fully went away (P(####) settled near 0.65 instead of
  returning to 0.99). Writing nothing and writing a wrong answer both score 0, so ending early costs the
  policy nothing on questions it would get wrong anyway. Meanwhile P(correct given ####) rose from 0.07
  (step 0) to about 0.28 on eval and 0.50 on train.
* Truncation at 256 tokens almost disappears (16% of rollouts at step 0, about 0% after step 110).

**Zero-advantage groups.** 50% of the 16-sample groups had every reward equal (mostly all wrong;
all right grows to about 12% late), so half the questions gave no gradient. During the collapse this
reached 77%.

**Eval turnover.** The step 0 → 420 improvement in mean accuracy over 8 samples is +0.139 (paired
bootstrap 95% CI 0.112–0.167 on the same 400 questions). Learning is not a pure superset: between
consecutive evals the model gains 30–40 questions and loses 20–40 (pass@8). Of the 400 eval questions,
160 were never solved at any eval, and 83 were solved at 6 or more evals.

**Full test, greedy (SFT vs RL).** Per question, on all 1,319 test problems: RL **fixed 246** that SFT
got wrong and **broke 17** that SFT got right; 19 are right for both and 1,037 wrong for both (exact
McNemar p ≈ 3e-53). So the improvement is almost entirely new problems, not a reshuffle. The RL model
writes `####` on 72% of answers (SFT: 40%) and uses half the tokens (82 vs 173). Gains are largest on
2-step problems (2.8% → 39.8%) and shrink with length: 3 steps 3.8% → 18.7%, 4 steps 1.7% → 11.4%,
5 steps 2.2% → 7.2%, 6+ steps 0% → 2.3%. Of the RL model's wrong answers, 55% are a `####` answer equal
to one of its own calculator results (correct arithmetic, wrong quantity or an unfinished chain) and 34%
end without `####`. For SFT the largest group was ending without the marker (50%), and 8% hit the
512-token limit; RL eliminated truncation.

**Summary of patterns.**
1. RL's first gain is format: `####` appears in 41% of eval samples at step 0 and in 99% of training
   rollouts by step 30.
2. The lasting gain is on short problems (2–3 steps). Long chains stay mostly unsolved; 160 of the 400
   eval questions were never solved.
3. The dominant error is reasoning, not arithmetic: the calculator is used correctly, but on the wrong
   quantity or the chain stops early. Near-misses and unit/scale errors are under 1%.
4. After the collapse the policy learned to end replies early, without `####`, on questions it cannot
   solve. The strict reward cannot tell this apart from a wrong answer, which motivates the extra reward
   terms of Part 4 (format and length shaping). With half of all groups giving zero advantage, a reward
   that separates "wrong" from "gave up" would also add gradient signal.

## Files

| Path | What it is |
|---|---|
| `modal_chat_rl.py` | Modal launcher: training (with auto-resume after preemption), and `--final-eval` for the report-card evaluation |
| `rl_rollout_logger.py` | runs the stock `chat_rl.py` and logs every rollout; resume edits are applied in memory only |
| `rewards.py` | Part 4 hook: training-reward components selected with `--reward` (default `strict` = stock); evaluation stays strict |
| `PART4_HANDOFF.md` | what Part 4 needs from Part 3 |
| `eval_logger.py` | runs the stock `chat_eval.py` and logs every GSM8K completion |
| `analysis/digitize_karpathy.py` | digitizes the W&B screenshot |
| `analysis/plot_curves.py` | reward and eval curves vs Karpathy |
| `analysis/analyze_rollouts.py` | the right/wrong analysis, clustering, and SFT-vs-RL comparison |
| `analysis/fetch_run.py`, `analysis/run_all.sh` | download a run from the Modal volume and regenerate everything |
| `reference/` | Karpathy's screenshot, the digitized CSVs, and the overlay check |
| `figures/` | all figures above |

Run data on Modal volume `vocalocate-a3`: checkpoints at `/a3/rl_runs/d20-baseline-long-rl/chatrl_checkpoints/d20-baseline-long/`
(steps 60, 120, …, 420, 466), rollouts at `/a3/rl_runs/d20-baseline-long-rl/rollouts/`, the log at
`/a3/logs/d20-baseline-long-rl.log`, and the final evaluation at `/a3/rl_runs/d20-baseline-long-rl/final_eval/`
(the SFT evaluation is under `d20-baseline-long-rl-smoke/final_eval/`).
