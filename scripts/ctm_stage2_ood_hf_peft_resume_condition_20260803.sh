#!/usr/bin/env bash
# Resume one partially completed native-HF/PEFT Qwen3.5 Stage 2 OOD matrix
# without regenerating any header-validated successful EvalLog.
#
# This is deliberately generation-only: it makes no Luna/OpenRouter request.
# The existing postprocess script remains the authoritative all-21-cell
# preflight/stage/grade handoff.  Per-cell local handoff receipts merely allow
# an approved incremental grader to begin from a completed biased cell once
# all clean references are present.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1}
RUN_ROOT=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
CONDITION=${CTM_OOD_CONDITION:?CTM_OOD_CONDITION is required}
GPU_LIST=${CTM_OOD_GPUS:?CTM_OOD_GPUS is required (one to four physical GPU indices)}
CHECKPOINT=${CTM_OOD_CHECKPOINT:?CTM_OOD_CHECKPOINT is required}

REPO=$(cd "$REPO" && pwd -P)
PY="$(cd "$(dirname "$PY")" && pwd -P)/$(basename "$PY")"
FROZEN=$(cd "$FROZEN" && pwd -P)
CHECKPOINT=$(cd "$CHECKPOINT" && pwd -P)
if [ "${RUN_ROOT#/}" = "$RUN_ROOT" ]; then
  RUN_ROOT="$(pwd -P)/$RUN_ROOT"
fi

readonly BASE_MODEL=Qwen/Qwen3.5-9B
readonly TASK_FACTORY=experiments.stage2_ood_hle.tasks:ood_tasks
readonly RAW_LOG_DIR="$RUN_ROOT/raw-no-luna/$CONDITION"
readonly RUNNER_DIR="$RUN_ROOT/runners"
readonly RUN_ID="resume-$(date -u +%Y%m%dT%H%M%SZ)-$$"
readonly RUNNER_LOG="$RUNNER_DIR/raw-no-luna-$CONDITION-$RUN_ID.log"
readonly CONTRACT="$RAW_LOG_DIR/resume-contract.json"
readonly ARCHIVE_DIR="$RUN_ROOT/_archive/$CONDITION/$RUN_ID"
readonly HANDOFF_ROOT="$RUN_ROOT/luna-incremental-staged-v1"
readonly PREFLIGHT_REPORT="$RUN_ROOT/artifacts/preflight-v1/$CONDITION.json"
readonly CLEAN_MARKER="$RAW_LOG_DIR/clean.resume.$RUN_ID.complete"
readonly MODEL_ARGS='{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}'
readonly MAX_CONNECTIONS=8

case "$CONDITION" in
  bct-hf-peft)
    DEFAULT_EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct
    ;;
  bct-control-hf-peft)
    DEFAULT_EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct-control
    ;;
  opct-phase2-hf-peft)
    DEFAULT_EXPECTED_CHECKPOINT=/workspace/ctm-opct-stage2-20260804/checkpoint/rmct_paper_isambard_phase2_qwen3_5_9b_opct_rng_repair_4gpu_20260803_opct-lr-1e-4
    ;;
  *) echo "unsupported native-HF/PEFT resume condition: $CONDITION" >&2; exit 2 ;;
esac
# The existing completed-condition postprocessor is deliberately pinned to
# these native raw-adapter paths.  Refuse a byte-identical copy at another
# location too: its header would not satisfy that postprocessor's exact
# expected-checkpoint check.  The synthetic override is deliberately gated
# behind an explicit test-only flag; it cannot change a production launch.
if [ "${CTM_OOD_TEST_MODE:-0}" = 1 ]; then
  EXPECTED_CHECKPOINT=${CTM_OOD_EXPECTED_CHECKPOINT:?CTM_OOD_EXPECTED_CHECKPOINT is required in test mode}
else
  EXPECTED_CHECKPOINT=$DEFAULT_EXPECTED_CHECKPOINT
fi
EXPECTED_CHECKPOINT=$(cd "$EXPECTED_CHECKPOINT" && pwd -P)
if [ "$CHECKPOINT" != "$EXPECTED_CHECKPOINT" ]; then
  echo "CTM_OOD_CHECKPOINT must be the exact postprocess-compatible raw adapter for $CONDITION: $EXPECTED_CHECKPOINT" >&2
  exit 2
fi
test -n "$RUN_ROOT"
test "$RUN_ROOT" != /
test -x "$PY"
test -f "$FROZEN/manifest.json"
test -d "$CHECKPOINT"

# Bash-3-compatible parsing.  One worker owns each physical GPU for the
# entirety of a phase, and its Inspect children see only cuda:0.
IFS=, read -r -a GPUS <<< "$GPU_LIST"
GPU_COUNT=${#GPUS[@]}
if [ "$GPU_COUNT" -lt 1 ] || [ "$GPU_COUNT" -gt 4 ]; then
  echo "CTM_OOD_GPUS must contain one, two, three, or four physical GPU indices" >&2
  exit 2
fi
for index in "${!GPUS[@]}"; do
  gpu=${GPUS[$index]}
  case "$gpu" in
    ''|*[!0-9]*) echo "CTM_OOD_GPUS contains an invalid GPU index: $gpu" >&2; exit 2 ;;
  esac
  for ((prior_index = 0; prior_index < index; prior_index++)); do
    if [ "$gpu" = "${GPUS[$prior_index]}" ]; then
      echo "CTM_OOD_GPUS contains duplicate GPU index: $gpu" >&2
      exit 2
    fi
  done
done

mkdir -p "$RAW_LOG_DIR" "$RUNNER_DIR"
test ! -e "$RUNNER_LOG"

export PYTHONPATH="$REPO"
export HF_HOME=${CTM_OOD_HF_HOME:-/workspace/hf-cache-direct}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA
cd "$REPO"

log() {
  printf '%s\n' "$*" | tee -a "$RUNNER_LOG"
}

resume_command() {
  "$PY" -m experiments.stage2_ood_hle.hf_peft_resume "$@" \
    --condition "$CONDITION" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$FROZEN/manifest.json" \
    --raw-log-dir "$RAW_LOG_DIR" \
    --contract "$CONTRACT"
}

state_field() {
  local field=$1
  "$PY" -c '
import json, sys
document = json.loads(sys.stdin.read())
value = document[sys.argv[1]]
if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
    raise SystemExit(f"invalid resume state field: {sys.argv[1]}")
print(" ".join(str(item) for item in value))
' "$field"
}

# This fails closed if a successful historical EvalLog has a wrong condition,
# checkpoint/runtime, frozen task identity, or decode header.  Only
# non-success/unreadable/partial `.eval` files are moved, recoverably, to the
# condition-specific archive directory.
PREPARE_JSON=$(resume_command prepare --archive-dir "$ARCHIVE_DIR")
log "resume preparation: $PREPARE_JSON"

TASK_ARGS=$(
  "$PY" -c 'import json, sys; print(json.dumps({"manifest": sys.argv[1], "unbiased_log": sys.argv[2], "prompt_style": "none", "include_bias_acknowledged": False}, sort_keys=True))' \
    "$FROZEN/manifest.json" "$RAW_LOG_DIR"
)
GENERATION_CONFIG='{"max_connections":8,"max_tokens":20480,"temperature":1.0,"top_k":20,"top_p":0.95}'
ARGS=(
  scripts/run_evals.py
  --task-factory "$TASK_FACTORY"
  --local-checkpoint "$CHECKPOINT"
  --base-model "$BASE_MODEL"
  --task-args "$TASK_ARGS"
  --model-args "$MODEL_ARGS"
  --generation-config "$GENERATION_CONFIG"
  --log-dir "$RAW_LOG_DIR"
  --max-tasks 1
  --isolate-tasks
  --yes
)

declare -a PIDS=()
launch_worker() {
  local phase=$1
  local gpu=$2
  shift 2
  local task_index
  local child_log
  [ "$#" -gt 0 ] || return 0
  (
    for task_index in "$@"; do
      child_log="$RUNNER_DIR/raw-no-luna-$CONDITION-$RUN_ID-$phase-gpu${gpu}-task${task_index}.log"
      test ! -e "$child_log" || { echo "refusing to overwrite worker log: $child_log" >&2; exit 2; }
      printf '%s\n' "started phase=$phase gpu=$gpu task_index=$task_index max_connections=$MAX_CONNECTIONS" >>"$RUNNER_LOG"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${ARGS[@]}" --task-index "$task_index" >"$child_log" 2>&1
      if [ "$phase" = biased ]; then
        # Other physical-GPU workers can still be writing their own retryable
        # EvalLogs. This call validates and stages *this* successful biased
        # cell while tolerating those in-flight partials; the final all-cell
        # handoff below remains strict.
        resume_command handoff --output-root "$HANDOFF_ROOT" --task-index "$task_index" --allow-inflight-partials >>"$child_log" 2>&1
      fi
      printf '%s\n' "completed phase=$phase gpu=$gpu task_index=$task_index" >>"$RUNNER_LOG"
    done
  ) &
  PIDS+=("$!")
}

wait_for_phase() {
  local phase=$1
  local pid
  local status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  PIDS=()
  if [ "$status" -ne 0 ]; then
    log "Stage 2 OOD native-HF/PEFT $phase resume workers failed; successful EvalLogs remain in place and failed/partial logs will be archived on the next resume."
    exit "$status"
  fi
  log "Stage 2 OOD native-HF/PEFT $phase resume workers complete"
}

launch_missing_phase() {
  local phase=$1
  shift
  local -a missing=("$@")
  local slot task_index group task_index_index
  local -a groups=()
  for ((slot = 0; slot < GPU_COUNT; slot++)); do groups[$slot]=''; done
  for ((task_index_index = 0; task_index_index < ${#missing[@]}; task_index_index++)); do
    slot=$((task_index_index % GPU_COUNT))
    groups[$slot]="${groups[$slot]} ${missing[$task_index_index]}"
  done
  for ((slot = 0; slot < GPU_COUNT; slot++)); do
    group=${groups[$slot]}
    [ -n "${group// /}" ] || continue
    set -- $group
    launch_worker "$phase" "${GPUS[$slot]}" "$@"
  done
  if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_for_phase "$phase"
  fi
}

clean_text=$(printf '%s' "$PREPARE_JSON" | state_field missing_clean_task_indices)
declare -a missing_clean=()
for task_index in $clean_text; do missing_clean+=("$task_index"); done
if [ "${#missing_clean[@]}" -gt 0 ]; then
  log "resuming missing clean task indices: ${missing_clean[*]}"
  launch_missing_phase clean "${missing_clean[@]}"
fi

# This is a semantic Inspect-header/runtime barrier.  It is intentionally
# checked even when all clean logs were inherited rather than newly generated.
CLEAN_JSON=$(resume_command verify-clean)
printf '%s\n' "$CLEAN_JSON" >"$CLEAN_MARKER"
log "Stage 2 OOD clean resume barrier verified: $CLEAN_JSON"

# Existing successful biased cells may now be safely staged for an approved
# incremental grader.  This performs a full paired-score validation and local
# copy only; it makes no API request.  The same call after each new biased
# task below makes newly completed cells available without waiting for 21/21.
HANDOFF_JSON=$(resume_command handoff --output-root "$HANDOFF_ROOT" --all-biased)
log "incremental local handoff after clean barrier: $HANDOFF_JSON"

STATUS_JSON=$(resume_command status)
biased_text=$(printf '%s' "$STATUS_JSON" | state_field missing_biased_task_indices)
declare -a missing_biased=()
for task_index in $biased_text; do missing_biased+=("$task_index"); done
if [ "${#missing_biased[@]}" -gt 0 ]; then
  log "resuming missing biased task indices: ${missing_biased[*]}"
  launch_missing_phase biased "${missing_biased[@]}"
fi

FINAL_HANDOFF_JSON=$(resume_command handoff --output-root "$HANDOFF_ROOT" --all-biased)
log "incremental local handoff after biased workers: $FINAL_HANDOFF_JSON"

# Do not emit the existing completion marker until the authoritative complete
# matrix preflight validates all 3 clean + 18 biased cells, their matching
# clean-log resolution, runtime identity, and frozen sample IDs.
"$PY" -m experiments.stage2_ood_hle.raw_preflight \
  --raw-log-root "$RAW_LOG_DIR" \
  --manifest "$FROZEN/manifest.json" \
  --condition "$CONDITION" \
  --runtime-profile hf-peft \
  --expected-checkpoint "$CHECKPOINT" \
  --expected-max-connections "$MAX_CONNECTIONS" \
  --output "$PREFLIGHT_REPORT" | tee -a "$RUNNER_LOG"

log "Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION"
