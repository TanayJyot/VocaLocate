# Part 5: On-policy distillation on GSM8K

The hypothesis and predictions are in [`HYPOTHESIS.md`](HYPOTHESIS.md). They were committed in `529fe48`,
before any Part 5 GPU run.

## Method

`chat_distill.py` is nanochat's `scripts/chat_rl.py` with one change: the per-token advantage. For every
token y_t of every sampled completion:

```
A_t = reward_coef * (r - mean_group(r)) + distill_coef * (log p_teacher(y_t | ctx_T, y<t) - log p_student(y_t | x, y<t))
```

The teacher term is under stop-gradient, with discount 0. The loss is `chat_rl.py`'s token-level policy
gradient, so `reward_coef=0, distill_coef=1` is the gradient of the per-token reverse KL to the teacher on
the student's own samples (Thinking Machines Lab, 2025). Sampling, data order, batch, learning rates,
pass@k evaluation and checkpointing are unchanged from the Part 3 RL run.

| Option | Meaning |
|---|---|
| `--teacher=checkpoint --teacher-dir D --teacher-step S` | frozen teacher model, same prompt as the student |
| `--teacher=self --teacher-ema a` | SDPO-style self-teacher: a copy of the student with the GSM8K reference solution in its prompt, updated toward the student at rate `a` after each step (0 = frozen) |
| `--distill-coef`, `--reward-coef` | weights of the two advantage terms (defaults 1 and 0) |
| `--adv-clip c` | clip the per-token log-ratio to [-c, c] (default off) |

Logged per step, to stdout and W&B: `reward` (strict correctness), `reverse_kl` (mean of
log p_S − log p_T over trained tokens), `log_ratio_std` and `clip_frac`. Every rollout goes to
`rollouts/rank*.jsonl` in the Part 3 format, so `../part-3-rl/analysis/analyze_rollouts.py` works on these
runs unchanged.

## Runs

Every run starts from the d20 baseline-long SFT checkpoint (step 932) and runs 200 steps on 8×H100. The
schedule is `--smoke-steps 200`: the first 3,200 GSM8K training questions, in the same order as RL,
with the LR decaying linearly to zero over 200 steps.

| Run | Output tag | Teacher and advantage |
|---|---|---|
| D00 | `a4-p5-d00-rl466` | RL step 466, distillation only |
| D01 | `a4-p5-d01-sdpo` | self-teacher with the reference solution, EMA 0.05 |
| D02 | `a4-p5-d02-rl120` | RL step 120, distillation only |
| D03 | `a4-p5-d03-rl466-reward` | RL step 466, distillation plus `(r − mean r)` |
| D01b (exploratory, added after D01 collapsed) | `a4-p5-d01b-sdpo-grpo-v2` | frozen self-teacher with the reference solution, `0.9 (r − mean r) + 0.1 ×` log-ratio |

Results: [`REPORT.md`](REPORT.md) and `results/`. `analysis/compare_runs.py` writes `p5_summary.csv`,
`p5_per_step.csv`, `p5_fulltest.csv` and the pass@k, training and full-test figures;
`analysis/extra_figures.py` then writes `p5_answers.csv` (every model's answer to each test question),
`p5_agreement.csv`, `p5_collapse.csv`, `p5_by_length.csv` and the collapse, solution-length and teacher
figures. Both read run folders fetched with `../part-3-rl/analysis/fetch_run.py`; `extra_figures.py` also
needs the RL run's `final_eval/` folder (below) in `<runs>/rl-teacher/final_eval`:

```bash
python analysis/compare_runs.py --runs <runs> --gsm8k <task_data>/openai--gsm8k/main --tokenizer <tokenizer.pkl> --out results
python analysis/extra_figures.py --runs <runs> --results results
```
 W&B project: `riyadvaliyev/a4-distill`. Teacher evaluations (RL steps 466 and 120) are in
`a3/rl_runs/d20-baseline-long-rl/final_eval/` on the volume.

The runs used volume `vocalocate-a4-p5` in the `riyadvaliyev` Modal workspace. It holds copies of the
files they need from `vocalocate-a3` (Tanay's workspace), in the same layout: the SFT checkpoint (step
932), the RL checkpoints at steps 466 and 120, the tokenizer and the cached eval data. Every run is on
8×H100, as Parts 3 and 4 were.

Launch commands, from `a4/part-3-rl/`, in bash (zsh does not word-split `$COMMON`):

```bash
export MODAL_PROFILE=riyadvaliyev RL_VOLUME=vocalocate-a4-p5
COMMON="--volume vocalocate-a4-p5 --base-path a3 --sft-tag d20-baseline-long-sft --model-tag d20-baseline-long \
  --model-step 932 --smoke-steps 200 --script chat_distill --wandb-project a4-distill"
RL_CKPT=/vol/a3/rl_runs/d20-baseline-long-rl/chatrl_checkpoints/d20-baseline-long

modal run --detach modal_chat_rl.py $COMMON --output-tag a4-p5-d00-rl466 --run-name a4-p5-d00-rl466 \
  --rl-args="--teacher-dir=$RL_CKPT --teacher-step=466"
modal run --detach modal_chat_rl.py $COMMON --output-tag a4-p5-d01-sdpo --run-name a4-p5-d01-sdpo \
  --rl-args="--teacher=self --teacher-ema=0.05"
modal run --detach modal_chat_rl.py $COMMON --output-tag a4-p5-d02-rl120 --run-name a4-p5-d02-rl120 \
  --rl-args="--teacher-dir=$RL_CKPT --teacher-step=120"
modal run --detach modal_chat_rl.py $COMMON --output-tag a4-p5-d03-rl466-reward --run-name a4-p5-d03-rl466-reward \
  --rl-args="--teacher-dir=$RL_CKPT --teacher-step=466 --reward-coef=1.0"
modal run --detach modal_chat_rl.py $COMMON --output-tag a4-p5-d01b-sdpo-grpo-v2 --run-name a4-p5-d01b-sdpo-grpo-v2 \
  --rl-args="--teacher=self --teacher-ema=0 --reward-coef=0.9 --distill-coef=0.1"

# Final evaluation (full test sets, greedy, 512 tokens), same protocol as Parts 3 and 4
modal run --detach modal_chat_rl.py --volume vocalocate-a4-p5 --base-path a3 --sft-tag d20-baseline-long-sft \
  --model-tag d20-baseline-long --model-step 932 --output-tag a4-p5-d00-rl466 --run-name eval --final-eval rl:199 --eval-gpus 2   # 2 GPUs fit next to an 8-GPU training job
```

## Tests

```bash
python a4/part-5-distill/local_smoke_test.py
```

This runs tiny random models on CPU through the same entry point the Modal launcher uses. It checks that
a teacher identical to the student gives a reverse KL of exactly 0, that a different teacher gives a
positive, finite one, and that the checkpoint, mixed and self-teacher modes all train and write
checkpoints and rollouts.
