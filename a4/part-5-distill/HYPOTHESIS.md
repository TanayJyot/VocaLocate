# Part 5 hypothesis (pre-registered before any distillation run)

Committed before the first Modal job of Part 5, including smoke tests. Results that do not match these
predictions will be reported as they are; this file will not be edited after the runs start.

## Method

We use on-policy distillation (Thinking Machines Lab, 2025), implemented in `chat_distill.py`. The student
starts from our d20 baseline-long SFT checkpoint (step 932, 2.73% GSM8K), the same start as the Part 3 and
Part 4 RL runs. At each step it samples 16 completions for each of 16 GSM8K training questions, exactly as
`chat_rl.py` does. A frozen teacher then scores the same tokens. Each generated token y_t gets the advantage
log p_teacher(y_t) - log p_student(y_t), so the update is the gradient of the per-token reverse KL
KL(student || teacher) on the student's own samples, with discount 0. The teacher is our Part 3 RL model
(step 466, 20.09% GSM8K). It is the only stronger model that shares our tokenizer.

Everything else matches the Part 3 RL run: sampling (T = 1.0, top-k 50, 256 new tokens), batch
(16 x 16), learning rates, data order, and the 400-question pass@k evaluation every 60 steps.
The final evaluation is the same: the full 1,319-question test, greedy, 512 tokens, plus ARC, MMLU and
HumanEval. One thing differs. Every distillation run is 200 steps, 43% of RL's 467 steps and samples, with
the same linear LR decay compressed into 200 steps. This tests the blog's central claim that dense
per-token supervision needs far fewer samples than RL.

## Hypothesis

RL gives one bit per completion, and in Part 3 half of all 16-sample groups gave no gradient at all,
because every sample scored the same. Distillation gives a signal on every token of every completion.
It also tells the student how the teacher would continue, not just whether the final number was right.
We therefore expect the student to recover most of the RL teacher's GSM8K accuracy with fewer than half
the samples, and to train more smoothly than RL did. Reverse KL is mode-seeking and bounded by the
teacher, so we expect the student to inherit the teacher's habits, including its failure modes, rather
than surpass it.

Pre-registered predictions for **D00** (teacher = RL step 466, pure distillation, 200 steps):

1. Full-test GSM8K (greedy) is **at least 17%**, within 3 points of the teacher's 20.09%, after 200 steps.
2. It does **not exceed the teacher** by more than 1 point.
3. In-training pass@1 at step 60 is **at least RL's 0.108** at the same step.
4. **No collapse.** The share of training rollouts with non-ASCII junk stays below 10% at every step
   (RL peaked at 57%), and the per-token reverse KL to the teacher falls by at least half from step 0 to
   the last step.
5. **The student's mistakes look like the teacher's.** The share of answers ending without `####` and
   the share giving one of its own intermediate results will be within 10 points of the RL model's
   full-test failure mix (34% and 55% of its wrong answers).
6. Its drop on ARC-E/ARC-C/MMLU/HumanEval is **no larger than RL's**.

## Ablations (each changes one thing relative to D00)

| Run | Change | Prediction |
|---|---|---|
| D01 | SDPO-style self-teacher (Hübotter et al., 2026). The teacher is the student itself (EMA, rate 0.05) with the GSM8K reference solution in its prompt; no separate model. | **Clearly worse than D00 and RL (≤ 10%).** A 3% model is a weak in-context learner, so the reference solution barely changes its next-token distribution. The SDPO paper also reports SDPO falling behind GRPO at the 0.6B to 1.5B scale. |
| D02 | Weaker teacher: RL step 120 (pass@1 0.13 vs 0.19 at step 420) | **Below D00**, roughly tracking the weaker teacher's own accuracy. The teacher bounds the student. |
| D03 | Distillation plus the RL reward: advantage = (r − mean r) + teacher log-ratio | **At least D00 + 1 point.** The reward can correct answers where the teacher is wrong, and the KL term keeps the policy close to fluent text. |

Every comparison uses one seed. With 1,319 test questions, a difference of about 2 points is at the edge
of what we can resolve. We will use the paired, per-question McNemar test from Part 3 rather than
compare raw accuracies.
