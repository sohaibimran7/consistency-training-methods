#!/usr/bin/env bash
# Redistribute RMCT's remaining held-out-bias cells after task 13 finishes.
#
# The original two-GPU held-out launcher was intentionally stopped at its
# outer scheduler while its task-13 children continue. Once the task-4 GPUs
# are free, this script verifies task 4 and task 13 success, kills only those
# stopped outer scheduler PIDs (never their process groups), and starts the
# remaining eight cells as two matched four-cell queues per condition.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-ood-hle-20260802/env/bin/python}
RUN=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json}
POLL_SECONDS=${CTM_RMCT_SPLIT_POLL_SECONDS:-30}

MAIN_PARENT=${CTM_RMCT_MAIN_HELDOUT_PARENT:-20180}
CONTROL_PARENT=${CTM_RMCT_CONTROL_HELDOUT_PARENT:-20181}
MAIN_CHECKPOINT=${CTM_RMCT_MAIN_CHECKPOINT:-$RUN/adapters/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801_rate-matching-lr-1e-4}
CONTROL_CHECKPOINT=${CTM_RMCT_CONTROL_CHECKPOINT:-$RUN/adapters/raw-final-adapter}

case "$POLL_SECONDS" in
  ''|*[!0-9]*|0) echo "CTM_RMCT_SPLIT_POLL_SECONDS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$POLL_SECONDS" -gt 60 ]; then
  echo "CTM_RMCT_SPLIT_POLL_SECONDS must be at most 60 seconds" >&2
  exit 2
fi
for file in "$FROZEN"; do test -f "$file"; done
for directory in "$REPO" "$MAIN_CHECKPOINT" "$CONTROL_CHECKPOINT"; do test -d "$directory"; done
test -x "$PY"

readonly MAIN_RAW="$RUN/raw-no-luna/rmct-hf-peft"
readonly CONTROL_RAW="$RUN/raw-no-luna/rmct-control-hf-peft"
readonly RUNNER_DIR="$RUN/runners"
readonly MAIN_GPU0_LOG="$RUNNER_DIR/rmct-hf-peft-split-heldout-gpu0-tasks14-15-17-19.log"
readonly MAIN_GPU1_LOG="$RUNNER_DIR/rmct-hf-peft-split-heldout-gpu1-tasks16-18-20-21.log"
readonly CONTROL_GPU2_LOG="$RUNNER_DIR/rmct-control-hf-peft-split-heldout-gpu2-tasks14-15-17-19.log"
readonly CONTROL_GPU3_LOG="$RUNNER_DIR/rmct-control-hf-peft-split-heldout-gpu3-tasks16-18-20-21.log"
readonly MODEL_ARGS='{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}'
readonly GENERATION_CONFIG='{"max_connections":8,"max_tokens":20480,"temperature":1.0,"top_k":20,"top_p":0.95}'

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

# This protects the handoff boundary: a stopped parent with a live child is
# expected, but a vanished/stale parent or an unsuccessful task is not safe to
# redistribute automatically.
boundary_ready() {
  cd "$REPO"
  if ! PYTHONPATH=. "$PY" - "$FROZEN" "$MAIN_RAW" "$CONTROL_RAW" <<'PY'
from pathlib import Path
import sys

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest, main_root, control_root = map(Path, sys.argv[1:])
specs = ood_task_specs(manifest)
target = {raw_preflight._task_identity(specs[index - 1]) for index in (4, 13)}
for raw_root in (main_root, control_root):
    seen = set()
    for path in raw_preflight._discover_eval_log_paths(raw_root):
        try:
            header = raw_preflight._read_eval_log(path, header_only=True)
        except Exception:
            continue
        if raw_preflight._attribute(header, "status") != "success":
            continue
        evaluation = raw_preflight._attribute(header, "eval")
        task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
        if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
            continue
        identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
        if identity in target:
            seen.add(identity)
    if seen != target:
        raise SystemExit(f"waiting for exact successful task-4/task-13 boundary cells in {raw_root}: {len(seen)}/2")
PY
  then
    return 1
  fi

  for parent in "$MAIN_PARENT" "$CONTROL_PARENT"; do
    if [ "$(ps -p "$parent" -o state= | tr -d ' ')" != T ]; then
      echo "waiting for stopped held-out scheduler parent $parent" >&2
      return 1
    fi
    args=$(ps -p "$parent" -o args=)
    case "$args" in
      *'--task-index 13'*) ;;
      *) echo "refusing to act on unexpected stopped parent $parent" >&2; return 2 ;;
    esac
    if ps -eo ppid= | awk -v parent="$parent" '$1 == parent { found=1 } END { exit found ? 0 : 1 }'; then
      echo "waiting for child of stopped parent $parent to exit" >&2
      return 1
    fi
  done
  return 0
}

while :; do
  if boundary_ready; then
    break
  else
    status=$?
  fi
  if [ "$status" -eq 2 ]; then
    exit 2
  fi
  log "waiting for RMCT task-4/task-13 split boundary"
  sleep "$POLL_SECONDS"
done

# The checks above prove the task-13 children are gone. These signals can
# therefore never interrupt a live decoder; do not signal the shared PGID.
kill -KILL "$MAIN_PARENT"
kill -KILL "$CONTROL_PARENT"
log "removed stopped held-out scheduler parents after exact task-13 success"

mkdir -p "$RUNNER_DIR"
for runner_log in "$MAIN_GPU0_LOG" "$MAIN_GPU1_LOG" "$CONTROL_GPU2_LOG" "$CONTROL_GPU3_LOG"; do
  test ! -e "$runner_log" || { echo "refusing to overwrite runner log: $runner_log" >&2; exit 2; }
done

cd "$REPO"
MAIN_TASK_ARGS=$("$PY" -c 'import json,sys; print(json.dumps({"manifest":sys.argv[1],"unbiased_log":sys.argv[2],"prompt_style":"none","include_bias_acknowledged":False},sort_keys=True))' "$FROZEN" "$MAIN_RAW")
CONTROL_TASK_ARGS=$("$PY" -c 'import json,sys; print(json.dumps({"manifest":sys.argv[1],"unbiased_log":sys.argv[2],"prompt_style":"none","include_bias_acknowledged":False},sort_keys=True))' "$FROZEN" "$CONTROL_RAW")

launch() {
  local gpu=$1
  local checkpoint=$2
  local task_args=$3
  local raw_root=$4
  local runner_log=$5
  shift 5
  local -a task_indexes=()
  local index
  for index in "$@"; do task_indexes+=(--task-index "$index"); done
  nohup env -u VLLM_BASE_URL -u VLLM_API_KEY -u CTM_PERSISTENT_VLLM_SERVER_METADATA \
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO" HF_HOME=/workspace/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    "$PY" scripts/run_evals.py \
      --task-factory experiments.stage2_ood_hle.tasks:ood_tasks \
      --local-checkpoint "$checkpoint" \
      --base-model Qwen/Qwen3.5-9B \
      --task-args "$task_args" \
      --model-args "$MODEL_ARGS" \
      --generation-config "$GENERATION_CONFIG" \
      --log-dir "$raw_root" \
      --max-tasks 1 \
      --isolate-tasks \
      --yes \
      "${task_indexes[@]}" >"$runner_log" 2>&1 &
  printf '%s\n' "$!"
}

MAIN_GPU0_PID=$(launch 0 "$MAIN_CHECKPOINT" "$MAIN_TASK_ARGS" "$MAIN_RAW" "$MAIN_GPU0_LOG" 14 15 17 19)
MAIN_GPU1_PID=$(launch 1 "$MAIN_CHECKPOINT" "$MAIN_TASK_ARGS" "$MAIN_RAW" "$MAIN_GPU1_LOG" 16 18 20 21)
CONTROL_GPU2_PID=$(launch 2 "$CONTROL_CHECKPOINT" "$CONTROL_TASK_ARGS" "$CONTROL_RAW" "$CONTROL_GPU2_LOG" 14 15 17 19)
CONTROL_GPU3_PID=$(launch 3 "$CONTROL_CHECKPOINT" "$CONTROL_TASK_ARGS" "$CONTROL_RAW" "$CONTROL_GPU3_LOG" 16 18 20 21)
log "launched matched RMCT split queues: main=$MAIN_GPU0_PID,$MAIN_GPU1_PID control=$CONTROL_GPU2_PID,$CONTROL_GPU3_PID"
