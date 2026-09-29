#!/usr/bin/env bash
# Run one original raw Qwen3.5 PEFT checkpoint through Inspect's native HF
# provider over the frozen 21-task Stage 2 OOD matrix. This launcher makes no
# OpenRouter/Luna calls: it writes raw/no-Luna EvalLogs only.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1}
RUN_ROOT=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
CONDITION=${CTM_OOD_CONDITION:?CTM_OOD_CONDITION is required}
GPU=${CTM_OOD_GPU:?CTM_OOD_GPU is required}
CHECKPOINT=${CTM_OOD_CHECKPOINT:?CTM_OOD_CHECKPOINT is required}
MAX_CONNECTIONS=${CTM_OOD_HF_MAX_CONNECTIONS:-1}

# Resolve all existing inputs before changing directory. This preserves the
# exact launch contract even when a caller starts this script from elsewhere.
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
readonly RUNNER_LOG="$RUN_ROOT/runners/raw-no-luna-$CONDITION.log"
readonly LAUNCH_CONTRACT="$RAW_LOG_DIR/launch-contract.json"
readonly MODEL_ARGS='{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}'

case "$CONDITION" in
  bct-hf-peft|bct-control-hf-peft|rmct-hf-peft|rmct-control-hf-peft) ;;
  *)
    echo "unknown native-HF/PEFT Stage 2 OOD condition: $CONDITION" >&2
    exit 2
    ;;
esac
case "$GPU" in
  ''|*[!0-9]*)
    echo "CTM_OOD_GPU must be one non-negative GPU index" >&2
    exit 2
    ;;
esac
case "$MAX_CONNECTIONS" in
  ''|*[!0-9]*|0)
    echo "CTM_OOD_HF_MAX_CONNECTIONS must be a positive integer" >&2
    exit 2
    ;;
esac
test -n "$RUN_ROOT"
test "$RUN_ROOT" != /
test -x "$PY"
test -f "$FROZEN/manifest.json"
test -d "$CHECKPOINT"

# Never merge or overwrite a prior condition. By default fail with a clear
# resume instruction. Set CTM_OOD_ARCHIVE_EXISTING=1 to move only the exact
# old raw condition directory and runner log to a timestamped recovery area,
# then begin a fresh, fully paired matrix.
if [ -e "$RAW_LOG_DIR" ] || [ -e "$RUNNER_LOG" ]; then
  if [ "${CTM_OOD_ARCHIVE_EXISTING:-0}" != 1 ]; then
    echo "existing raw/no-Luna output detected; do not mix retries in place." >&2
    echo "Set CTM_OOD_ARCHIVE_EXISTING=1 to archive this exact condition before a fresh rerun:" >&2
    echo "  $RAW_LOG_DIR" >&2
    echo "  $RUNNER_LOG" >&2
    exit 2
  fi
  ARCHIVE_DIR="$RUN_ROOT/_archive/${CONDITION}-$(date -u +%Y%m%dT%H%M%SZ)"
  test ! -e "$ARCHIVE_DIR"
  mkdir -p "$ARCHIVE_DIR"
  if [ -e "$RAW_LOG_DIR" ]; then
    mv "$RAW_LOG_DIR" "$ARCHIVE_DIR/raw-no-luna"
  fi
  if [ -e "$RUNNER_LOG" ]; then
    mv "$RUNNER_LOG" "$ARCHIVE_DIR/runner.log"
  fi
  printf '%s\n' "Archived prior raw/no-Luna condition to: $ARCHIVE_DIR"
fi
mkdir -p "$RAW_LOG_DIR" "$RUN_ROOT/runners"

export PYTHONPATH="$REPO"
export HF_HOME=${CTM_OOD_HF_HOME:-/workspace/hf-cache-direct}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA

# The launcher may be started under nohup from an arbitrary working directory.
# Anchor the relative run_evals path in the isolated repository.
# `--isolate-tasks` is required with this runner's `--max-tasks 1`: it
# explicitly invokes one child per selected task rather than asking Inspect to
# truncate a multi-task selection to its first task. `--local-checkpoint` plus
# provider=hf invokes the PEFT path in ctm.evals.local_model; do not substitute
# a vLLM compatibility adapter.
cd "$REPO"

"$PY" -m experiments.stage2_ood_hle.hf_peft_runner \
  --condition "$CONDITION" \
  --checkpoint "$CHECKPOINT" \
  --manifest "$FROZEN/manifest.json" \
  --raw-log-dir "$RAW_LOG_DIR" \
  --max-connections "$MAX_CONNECTIONS" \
  --output "$LAUNCH_CONTRACT"

# Avoid a heredoc nested inside a command substitution here. Bash 3 accepts
# that form, while the Bash 5 runtime used on Vast rejects this particular
# pair of substitutions at parse time. A short Python expression preserves
# proper JSON escaping for arbitrary absolute paths without shell quoting.
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
  --isolate-tasks
  --yes
)

printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "$PY" "${ARGS[@]}"
printf '\n'
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${ARGS[@]}" >"$RUNNER_LOG" 2>&1
printf '%s\n' "Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION"
