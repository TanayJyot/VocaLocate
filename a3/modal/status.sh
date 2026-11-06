#!/usr/bin/env bash
# Print one status line per d12 run, from Modal's live logs (filtered by each run's FunctionCall id) and the
# result JSONs the runs write to the vocalocate-a3 volume when they finish.
# Usage: status.sh APP_ID RUN_LOG [interval_seconds]
#   RUN_LOG is the local `modal run` output that contains the launcher's "spawned: {...}" line.
#   No interval = print once; with an interval = repeat until every run has finished.
MODAL="${MODAL:-modal}"
APP="$1"; RUN_LOG="$2"; INTERVAL="$3"
TMP=$(mktemp -d)
export MSYS_NO_PATHCONV=1 PYTHONUTF8=1

# "tag fc-id" pairs from the launcher's spawned line
PAIRS=$(grep -m1 "spawned:" "$RUN_LOG" | grep -oE "'d12-[a-z0-9-]+': 'fc-[A-Z0-9]+'" | tr -d "':")
[ -z "$PAIRS" ] && { echo "no spawned line in $RUN_LOG yet"; exit 1; }

check() {
  local total=0 done=0
  echo "[$(date +%H:%M)] d12 status"
  while read -r tag fc; do
    total=$((total + 1))
    if "$MODAL" volume get --force vocalocate-a3 "/a3/results/$tag.json" "$TMP/$tag.json" >/dev/null 2>&1; then
      done=$((done + 1))
      python -c "import json,sys;r=json.load(open(sys.argv[1]));m=r['mfu_pct'];print(f\"  {sys.argv[2]:20s} DONE rc={r['returncode']} val_bpb={r['val_bpb']} CORE={r['core']} train={r['train_minutes']}m mfu={m and round(m,1)}%\" + (' ERROR' if r['returncode'] else ''))" "$TMP/$tag.json" "$tag"
      continue
    fi
    "$MODAL" app logs "$APP" --function-call "$fc" --tail 400 > "$TMP/$tag.log" 2>&1
    step=$(grep -oE "step [0-9]+/[0-9]+ \([0-9.]+%\)" "$TMP/$tag.log" | tail -1)
    mfu=$(grep -oE "bf16_mfu: [0-9.]+" "$TMP/$tag.log" | tail -1)
    eta=$(grep -oE "eta: [0-9.]+m" "$TMP/$tag.log" | tail -1)
    bpb=$(grep -oE "Validation bpb: [0-9.]+" "$TMP/$tag.log" | tail -1)
    core=$(grep -oE "CORE metric: -?[0-9.]+" "$TMP/$tag.log" | tail -1)
    err=$(grep -m1 -E "Traceback|CUDA out of memory|RuntimeError|Killed|preempt" "$TMP/$tag.log")
    printf "  %-20s %s | %s | %s | %s%s%s\n" "$tag" "${step:-no step yet}" "$mfu" "$eta" "${bpb:-no val yet}" \
      "${core:+ | $core}" "${err:+ | ERROR: $err}"
  done <<< "$PAIRS"
  [ "$done" -eq "$total" ] && echo "ALL $total RUNS FINISHED" && return 0
  return 1
}

if [ -z "$INTERVAL" ]; then check; exit; fi
while true; do
  check && exit 0
  sleep "$INTERVAL"
done
