#!/usr/bin/env bash
# The GPU-2 held-out BCT-main pair starts as soon as RMCT's held-out worker
# releases it. This watcher starts only the remaining GPU-0 LogiQA pair after
# the independent RMCT train worker exits.
set -euo pipefail

RUN=/workspace/ctm-act-repair-20260731
RUNNER=$RUN/runners/ctm-bct-main-hf-recovery-20260802.sh
RUNNER_ROOT=$RUN/runners/bct-main-hf-recovery-20260802
REPO=/workspace/ctm-eval-none-20260801/repo
PAIR=heldout-logiqa
OUTPUT=$REPO/logs/evals/stage1-bct-main-hf-recovery-20260802/$PAIR
RMCT_TRAIN_PID=24300

test ! -e "$OUTPUT"
mkdir -p "$RUNNER_ROOT"
echo "waiting for RMCT-control train worker $RMCT_TRAIN_PID to release GPU 0"
while [ -r "/proc/$RMCT_TRAIN_PID/cmdline" ]; do
  command_line=$(tr '\0' ' ' < "/proc/$RMCT_TRAIN_PID/cmdline")
  case "$command_line" in
    *run_evals.py*rmct-control-b8-accelerated*) sleep 20 ;;
    *) break ;;
  esac
done

sleep 15
echo "RMCT-control train worker released; starting BCT-main $PAIR on GPU 0"
nohup env BCT_MAIN_MODE=raw-pair BCT_MAIN_PAIR=$PAIR BCT_MAIN_PAIR_GPU=0 \
  bash "$RUNNER" > "$RUNNER_ROOT/$PAIR.log" 2>&1 < /dev/null &
pair_pid=$!
wait "$pair_pid"
echo "BCT-main GPU-0 held-out raw pair complete"
