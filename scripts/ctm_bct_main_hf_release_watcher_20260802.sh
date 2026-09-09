#!/usr/bin/env bash
# Start the two held-out native-HF BCT-main raw pairs as soon as the RMCT
# control generator has released its two GPUs. This intentionally performs no
# grading or finalization on Vast: those stages run locally after archival.
set -euo pipefail

RUN=/workspace/ctm-act-repair-20260731
RUNNER=$RUN/runners/ctm-bct-main-hf-recovery-20260802.sh
RUNNER_ROOT=$RUN/runners/bct-main-hf-recovery-20260802
REPO=/workspace/ctm-eval-none-20260801/repo
EVAL_ROOT=$REPO/logs/evals/stage1-bct-main-hf-recovery-20260802

for pair in heldout-logiqa heldout-hellaswag; do
  if [ -e "$EVAL_ROOT/$pair" ]; then
    echo "refusing to overwrite existing BCT-main raw pair: $pair" >&2
    exit 1
  fi
done

mkdir -p "$RUNNER_ROOT"
echo "waiting for RMCT-control evaluation workers to release GPUs 0 and 2"
# These are the two observed long-lived RMCT evaluation parents. Check their
# command lines rather than matching arbitrary processes, so the watcher
# cannot reuse a GPU while either still owns its model context.
rmct_worker_alive() {
  local pid cmdline
  for pid in 24300 24304; do
    if [ -r "/proc/$pid/cmdline" ]; then
      cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
      if [[ "$cmdline" == *"run_evals.py"* && "$cmdline" == *"rmct-control-b8-accelerated"* ]]; then
        return 0
      fi
    fi
  done
  return 1
}
while rmct_worker_alive; do
  sleep 20
done

# Allow CUDA contexts from the completed RMCT workers to disappear before the
# two new workers reserve their devices.
sleep 15
echo "RMCT-control workers released; starting held-out BCT-main pairs"

nohup env BCT_MAIN_MODE=raw-pair BCT_MAIN_PAIR=heldout-logiqa BCT_MAIN_PAIR_GPU=0 \
  bash "$RUNNER" > "$RUNNER_ROOT/heldout-logiqa.log" 2>&1 < /dev/null &
logiqa_pid=$!

nohup env BCT_MAIN_MODE=raw-pair BCT_MAIN_PAIR=heldout-hellaswag BCT_MAIN_PAIR_GPU=2 \
  bash "$RUNNER" > "$RUNNER_ROOT/heldout-hellaswag.log" 2>&1 < /dev/null &
hellaswag_pid=$!

wait "$logiqa_pid"
wait "$hellaswag_pid"
echo "BCT-main held-out raw pairs complete"
