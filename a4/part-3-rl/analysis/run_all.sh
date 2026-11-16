#!/usr/bin/env bash
# Post-run pipeline: fetch the run from the volume, plot curves vs Karpathy, run the EDA.
# Usage: bash run_all.sh <output-tag> <local-run-dir> <sft-eval-tag>
#   e.g. bash run_all.sh d20-baseline-long-rl ../../../runs_modal/d20-baseline-long-rl d20-baseline-long-rl-smoke
# Env: MODAL_PY (python with modal installed), PY (python with pandas/sklearn/matplotlib/tiktoken),
#      GSM8K_DIR (task_data/openai--gsm8k/main), TOKENIZER (tokenizer.pkl).
set -euo pipefail
TAG=$1; RUN=$2; SFT_EVAL_TAG=${3:-$TAG}
HERE=$(cd "$(dirname "$0")" && pwd)
"$MODAL_PY" -I "$HERE/fetch_run.py" "$TAG" "$RUN"
if [ "$SFT_EVAL_TAG" != "$TAG" ]; then
  "$MODAL_PY" -I "$HERE/fetch_run.py" "$SFT_EVAL_TAG" "$RUN/_sft_eval_src" >/dev/null
  cp "$RUN"/_sft_eval_src/final_eval/* "$RUN/final_eval/" 2>/dev/null || true
fi
"$PY" -I "$HERE/plot_curves.py" --log "$RUN/logs/$TAG.log" --ref "$RUN/reference" --out "$RUN/figures"
RL_STEP=$(ls "$RUN"/final_eval/gsm8k_rl_step*_rank0.jsonl 2>/dev/null | sed -E 's/.*gsm8k_rl_(step[0-9]+)_rank0.jsonl/\1/' | tail -1 || true)
FE=()
if [ -n "$RL_STEP" ]; then
  FE=(--final-eval "$RUN/final_eval/gsm8k_sft_step000932" "$RUN/final_eval/gsm8k_rl_$RL_STEP")
fi
LOKY_MAX_CPU_COUNT=8 "$PY" -I "$HERE/analyze_rollouts.py" --rollouts "$RUN/rollouts" --gsm8k "$GSM8K_DIR" \
  --tokenizer "$TOKENIZER" --out "$RUN/eda" "${FE[@]}"
