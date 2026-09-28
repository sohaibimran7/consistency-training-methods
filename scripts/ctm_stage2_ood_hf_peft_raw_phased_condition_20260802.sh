#!/usr/bin/env bash
# Separate native-HF/PEFT Stage 2 OOD launcher. It is raw/no-Luna only and
# never changes the existing serial launcher or any in-flight condition.
#
# CTM_OOD_GPUS is a comma-separated physical-GPU list. One through four GPUs
# are supported. Tasks 1..3 (clean) complete and pass a semantic Inspect-header
# barrier before any biased task 4..21 starts. With four GPUs, the fourth is
# intentionally idle during the three-cell clean barrier and joins the biased
# phase after that barrier is verified.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1}
RUN_ROOT=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
CONDITION=${CTM_OOD_CONDITION:?CTM_OOD_CONDITION is required}
GPU_LIST=${CTM_OOD_GPUS:?CTM_OOD_GPUS is required (comma-separated physical GPU indices)}
CHECKPOINT=${CTM_OOD_CHECKPOINT:?CTM_OOD_CHECKPOINT is required}
MAX_CONNECTIONS=${CTM_OOD_HF_MAX_CONNECTIONS:-8}

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
readonly RUNNER_LOG="$RUNNER_DIR/raw-no-luna-$CONDITION-phased.log"
readonly LAUNCH_CONTRACT="$RAW_LOG_DIR/launch-contract.json"
readonly CLEAN_MARKER="$RAW_LOG_DIR/clean.complete"
readonly MODEL_ARGS='{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}'

case "$CONDITION" in
  bct-hf-peft|bct-control-hf-peft|rmct-hf-peft|rmct-control-hf-peft|opct-phase2-hf-peft) ;;
  *) echo "unknown native-HF/PEFT Stage 2 OOD condition: $CONDITION" >&2; exit 2 ;;
esac
case "$MAX_CONNECTIONS" in
  ''|*[!0-9]*|0) echo "CTM_OOD_HF_MAX_CONNECTIONS must be a positive integer" >&2; exit 2 ;;
esac
test -n "$RUN_ROOT"
test "$RUN_ROOT" != /
test -x "$PY"
test -f "$FROZEN/manifest.json"
test -d "$CHECKPOINT"

# Bash-3-compatible list parsing. One model process is ever visible on each
# listed GPU in a phase; the model still sees that device as cuda:0.
IFS=, read -r -a GPUS <<< "$GPU_LIST"
GPU_COUNT=${#GPUS[@]}
if [ "$GPU_COUNT" -lt 1 ] || [ "$GPU_COUNT" -gt 4 ]; then
  echo "CTM_OOD_GPUS must contain one, two, three, or four GPU indices" >&2
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

# A fresh condition never mixes retries. Archival is recoverable and moves all
# condition-specific runner logs because the postprocess watcher reads them.
shopt -s nullglob
EXISTING_RUNNER_LOGS=("$RUNNER_DIR"/raw-no-luna-"$CONDITION"*.log)
if [ -e "$RAW_LOG_DIR" ] || [ "${#EXISTING_RUNNER_LOGS[@]}" -gt 0 ]; then
  if [ "${CTM_OOD_ARCHIVE_EXISTING:-0}" != 1 ]; then
    echo "existing raw/no-Luna output detected; do not mix retries in place." >&2
    echo "Set CTM_OOD_ARCHIVE_EXISTING=1 to archive this exact condition before a fresh rerun:" >&2
    echo "  $RAW_LOG_DIR" >&2
    exit 2
  fi
  ARCHIVE_DIR="$RUN_ROOT/_archive/${CONDITION}-$(date -u +%Y%m%dT%H%M%SZ)"
  test ! -e "$ARCHIVE_DIR"
  mkdir -p "$ARCHIVE_DIR/runner-logs"
  if [ -e "$RAW_LOG_DIR" ]; then
    mv "$RAW_LOG_DIR" "$ARCHIVE_DIR/raw-no-luna"
  fi
  for old_log in "${EXISTING_RUNNER_LOGS[@]}"; do
    mv "$old_log" "$ARCHIVE_DIR/runner-logs/"
  done
  printf '%s\n' "Archived prior raw/no-Luna condition to: $ARCHIVE_DIR"
fi
mkdir -p "$RAW_LOG_DIR" "$RUNNER_DIR"
test ! -e "$CLEAN_MARKER"

export PYTHONPATH="$REPO"
export HF_HOME=${CTM_OOD_HF_HOME:-/workspace/hf-cache-direct}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA
cd "$REPO"

# This binds the original raw PEFT bytes, immutable Stage-2 manifest, exact
# no-CoT task args, native HF model args, and 20k decode contract.
"$PY" -m experiments.stage2_ood_hle.hf_peft_runner \
  --condition "$CONDITION" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$FROZEN/manifest.json" \
  --raw-log-dir "$RAW_LOG_DIR" \
  --max-connections "$MAX_CONNECTIONS" \
  --output "$LAUNCH_CONTRACT"

TASK_ARGS=$("$PY" -c 'import json, sys; print(json.dumps({"manifest": sys.argv[1], "unbiased_log": sys.argv[2], "prompt_style": "none", "include_bias_acknowledged": False}, sort_keys=True))' "$FROZEN/manifest.json" "$RAW_LOG_DIR")
GENERATION_CONFIG=$(printf '{"max_connections": %s, "max_tokens": 20480, "temperature": 1.0, "top_k": 20, "top_p": 0.95}' "$MAX_CONNECTIONS")
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
  # `max_tasks=1` is a per-child Inspect budget. Without isolation, a group
  # such as 4..12 silently evaluates only task 4. The runner's isolated mode
  # executes each selected index in a fresh child and preserves every EvalLog.
  --isolate-tasks
  --yes
)

log() {
  printf '%s\n' "$*" | tee -a "$RUNNER_LOG"
}

declare -a PIDS=()
launch_group() {
  local phase=$1
  local gpu=$2
  local task_index
  local label=
  local child_log
  shift 2
  local -a index_args=()
  for task_index in "$@"; do
    case "$task_index" in
      ''|*[!0-9]*) echo "internal error: non-numeric task index" >&2; return 2 ;;
    esac
    label=${label:+$label-}$task_index
    index_args+=(--task-index "$task_index")
  done
  [ -n "$label" ] || { echo "internal error: empty task group" >&2; return 2; }
  child_log="$RUNNER_DIR/raw-no-luna-$CONDITION-phased-$phase-gpu${gpu}-tasks${label}.log"
  test ! -e "$child_log" || { echo "refusing to overwrite runner log: $child_log" >&2; return 2; }
  log "started phase=$phase gpu=$gpu task_indices=$label runner_log=$child_log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${ARGS[@]}" "${index_args[@]}"
  ) >"$child_log" 2>&1 &
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
    log "Stage 2 OOD native-HF/PEFT $phase phase failed; completed raw logs have been preserved for diagnosis."
    exit "$status"
  fi
  log "Stage 2 OOD native-HF/PEFT $phase phase process groups complete"
}

# A zero exit from a worker is insufficient for pairing. Verify exactly the
# three expected clean headers before biased processes are permitted to start.
verify_clean_phase() {
  "$PY" - "$FROZEN/manifest.json" "$RAW_LOG_DIR" <<'PY'
from pathlib import Path
import sys

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest = Path(sys.argv[1]).resolve()
raw_root = Path(sys.argv[2]).resolve()
expected = raw_preflight._expected_cell_specs(ood_task_specs(manifest))
clean = {identity: spec for identity, spec in expected.items() if spec.kind == "unbiased"}
seen = {}
for path in raw_preflight._discover_eval_log_paths(raw_root):
    log = raw_preflight._read_eval_log(path, header_only=True)
    if raw_preflight._attribute(log, "status") != "success":
        raise SystemExit(f"clean phase contains a non-success EvalLog: {path}")
    evaluation = raw_preflight._attribute(log, "eval")
    task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
    if task_name != raw_preflight.TASK_UNBIASED:
        raise SystemExit(f"biased or unrelated EvalLog appeared before the clean barrier: {path}")
    identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
    spec = clean.get(identity)
    if spec is None:
        raise SystemExit(f"unexpected clean EvalLog before the barrier: {path}")
    raw_preflight._validate_header(log, path=path, spec=spec, raw_root=raw_root)
    if identity in seen:
        raise SystemExit(f"duplicate successful clean EvalLog before the barrier: {path} and {seen[identity]}")
    seen[identity] = path
if set(seen) != set(clean):
    missing = sorted(set(clean) - set(seen))
    raise SystemExit(f"clean phase is incomplete before the biased barrier; missing={missing}")
print(f"verified Stage 2 clean barrier: {len(seen)} exact clean cells")
PY
}

# Clean tasks are round-robin grouped: one GPU runs 1,2,3 serially; two run
# 1+3 and 2; three run 1, 2, 3; with four, the fourth waits for the barrier.
# No GPU receives concurrent model processes.
declare -a CLEAN_GROUPS=()
for ((slot = 0; slot < GPU_COUNT; slot++)); do CLEAN_GROUPS[$slot]=''; done
for ((task_index = 1; task_index <= 3; task_index++)); do
  slot=$(((task_index - 1) % GPU_COUNT))
  CLEAN_GROUPS[$slot]="${CLEAN_GROUPS[$slot]} $task_index"
done
for ((slot = 0; slot < GPU_COUNT; slot++)); do
  set -- ${CLEAN_GROUPS[$slot]}
  [ "$#" -gt 0 ] || continue
  launch_group clean "${GPUS[$slot]}" "$@"
done
wait_for_phase clean
verify_clean_phase | tee -a "$RUNNER_LOG"
touch "$CLEAN_MARKER"
log "Stage 2 OOD clean phase verified: $CLEAN_MARKER"

# Balanced contiguous biased groups amortize each HF model load: 18, 9+9,
# 6+6+6, or 5+5+4+4 selected tasks for one through four visible GPUs.
next_task=4
base_group_size=$((18 / GPU_COUNT))
extra_groups=$((18 % GPU_COUNT))
for ((slot = 0; slot < GPU_COUNT; slot++)); do
  group_size=$base_group_size
  if [ "$slot" -lt "$extra_groups" ]; then group_size=$((group_size + 1)); fi
  declare -a group_indices=()
  for ((offset = 0; offset < group_size; offset++)); do
    group_indices+=("$next_task")
    next_task=$((next_task + 1))
  done
  launch_group biased "${GPUS[$slot]}" "${group_indices[@]}"
done
[ "$next_task" -eq 22 ] || { echo "internal error: biased task assignment is incomplete" >&2; exit 2; }
wait_for_phase biased

printf '%s\n' "Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION" | tee -a "$RUNNER_LOG"
