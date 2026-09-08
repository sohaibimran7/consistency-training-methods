#!/usr/bin/env bash
# Poll only for final Stage 2 OOD launcher markers and hand each completed
# condition to the locked, hash-bound postprocess script.  The child script
# takes per-condition and host-wide locks, so repeated polling is safe.
set -euo pipefail

RUN=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
SCRIPT=${CTM_OOD_POSTPROCESS_SCRIPT:-"$RUN/repo/scripts/ctm_stage2_ood_postprocess_condition_20260802.sh"}
ANALYSIS_SCRIPT=${CTM_OOD_ANALYSIS_SCRIPT:-"$RUN/repo/scripts/ctm_stage2_ood_analyze_20260802.sh"}
INTERVAL_SECONDS=${CTM_OOD_POSTPROCESS_INTERVAL_SECONDS:-30}
LOG_DIR="$RUN/postprocess-logs"

case "$INTERVAL_SECONDS" in
  ''|*[!0-9]*|0)
    echo "CTM_OOD_POSTPROCESS_INTERVAL_SECONDS must be a positive integer" >&2
    exit 2
    ;;
esac
test -x "$SCRIPT"
test -x "$ANALYSIS_SCRIPT"
mkdir -p "$LOG_DIR"

CONDITIONS=(
  base-vllm
  act-vllm-compat
  attct-vllm-compat
  mlpct-vllm-compat
  opct-vllm-compat
  rmct-vllm-compat
  rmct-hf-peft
  bct-hf-peft
  bct-control-hf-peft
  rmct-control-hf-peft
)

while :; do
  for CONDITION in "${CONDITIONS[@]}"; do
    STATE="$RUN/artifacts/postprocess-state-v1/${CONDITION}.complete"
    if [ -f "$STATE" ]; then
      continue
    fi
    # The child exits harmlessly until the exact completion marker is present;
    # its per-condition flock suppresses duplicate work from prior polls.
    "$SCRIPT" "$CONDITION" >>"$LOG_DIR/$CONDITION.log" 2>&1 &
  done
  "$ANALYSIS_SCRIPT" >>"$LOG_DIR/analysis.log" 2>&1 &
  sleep "$INTERVAL_SECONDS"
done
