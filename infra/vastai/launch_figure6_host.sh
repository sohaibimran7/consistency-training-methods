#!/usr/bin/env bash
# Launch the fixed Figure 6 model plan for one Vast.ai host.
set -euo pipefail
set +x

usage() {
    echo "Usage: $0 {A|B} {canary|full}" >&2
    exit 2
}

[[ $# -eq 2 ]] || usage
HOST=$1
MODE=$2
case "$HOST" in
    A|B) ;;
    *) usage ;;
esac
case "$MODE" in
    canary)
        if [[ -n "${FIGURE6_LIMIT_CONDITIONS:-}" && "$FIGURE6_LIMIT_CONDITIONS" != 100 ]]; then
            echo "ERROR: canary mode requires FIGURE6_LIMIT_CONDITIONS=100 when it is already set." >&2
            exit 2
        fi
        export FIGURE6_LIMIT_CONDITIONS=100
        ;;
    full)
        unset FIGURE6_LIMIT_CONDITIONS
        ;;
    *) usage ;;
esac
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: do not pre-set CUDA_VISIBLE_DEVICES; the fixed host plan assigns physical GPUs." >&2
    exit 2
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must not be present on a Vast generation host." >&2
    exit 2
fi
if [[ -n "${HF_TOKEN_FILE:-}" && -n "${HF_TOKEN:-}" ]]; then
    echo "ERROR: set only one of HF_TOKEN_FILE or HF_TOKEN." >&2
    exit 2
fi
if [[ "$HOST" == A && -z "${HF_TOKEN_FILE:-}" && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: host A requires HF_TOKEN_FILE or HF_TOKEN for gated model llama33." >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd -P)
FIGURE6_VENV=${FIGURE6_VENV:-$REPO_DIR/.venv-figure6}
MODELS_CONFIG="$REPO_DIR/experiments/eval_awareness/figure6/models.yaml"
RUNNER="$SCRIPT_DIR/run_figure6_model.sh"

for name in HF_HOME FIGURE6_ARTIFACT FIGURE6_OUTPUT_ROOT; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable $name is not set." >&2
        exit 2
    fi
    if [[ "${!name}" != /* ]]; then
        echo "ERROR: $name must be an absolute path." >&2
        exit 2
    fi
done
if [[ ! -x "$FIGURE6_VENV/bin/python" ]]; then
    echo "ERROR: missing Figure 6 venv; run infra/vastai/provision_figure6.sh first." >&2
    exit 2
fi
if [[ ! -x "$RUNNER" ]]; then
    echo "ERROR: Figure 6 Vast model runner is not executable." >&2
    exit 2
fi
if [[ ! -f "$FIGURE6_ARTIFACT" || ! -f "$FIGURE6_ARTIFACT.manifest.json" ]]; then
    echo "ERROR: FIGURE6_ARTIFACT and its .manifest.json sidecar must exist." >&2
    exit 2
fi
if [[ "$HOST" == A ]]; then
    for name in PAPER_NATURAL_PROMPT_PATH EXPLICIT_SCRATCHPAD_PROMPT_PATH; do
        if [[ -z "${!name:-}" || "${!name}" != /* || ! -f "${!name}" ]]; then
            echo "ERROR: host A requires $name to be an existing absolute file." >&2
            exit 2
        fi
    done
else
    if [[ -z "${EXPLICIT_SCRATCHPAD_PROMPT_PATH:-}" || "$EXPLICIT_SCRATCHPAD_PROMPT_PATH" != /* || ! -f "$EXPLICIT_SCRATCHPAD_PROMPT_PATH" ]]; then
        echo "ERROR: host B requires EXPLICIT_SCRATCHPAD_PROMPT_PATH to be an existing absolute file." >&2
        exit 2
    fi
fi

source "$FIGURE6_VENV/bin/activate"
DETECTED_GPU_COUNT=$(python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is unavailable on this host.")
print(torch.cuda.device_count())
PY
)
mapfile -t PLAN_LINES < <(
    python "$SCRIPT_DIR/figure6_plan.py" \
        --models "$MODELS_CONFIG" \
        --host "$HOST" \
        --detected-gpu-count "$DETECTED_GPU_COUNT"
)
if [[ ${#PLAN_LINES[@]} -eq 0 ]]; then
    echo "ERROR: validated host plan is empty." >&2
    exit 2
fi

mkdir -p "$FIGURE6_OUTPUT_ROOT/_logs" "$FIGURE6_OUTPUT_ROOT/_status"
STATUS_LOG="$FIGURE6_OUTPUT_ROOT/_status/host-${HOST}.jsonl"
LOCK_FILE="$FIGURE6_OUTPUT_ROOT/_status/host-${HOST}.lock"
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: required image command is unavailable: flock" >&2
    exit 2
fi
exec 9>> "$LOCK_FILE"
if ! flock -n 9; then
    echo "ERROR: another Figure 6 launcher already holds the host $HOST lock." >&2
    exit 2
fi

RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-host${HOST}-$$"
export FIGURE6_RUN_TAG=$RUN_TAG
LIMIT_FOR_STATUS=${FIGURE6_LIMIT_CONDITIONS:-0}

record_status() {
    local model_key=$1
    local gpu_list=$2
    local port=$3
    local phase=$4
    local process_id=$5
    local exit_code=$6
    python - "$STATUS_LOG" "$RUN_TAG" "$HOST" "$MODE" "$LIMIT_FOR_STATUS" \
        "$model_key" "$gpu_list" "$port" "$phase" "$process_id" "$exit_code" <<'PY'
import datetime
import json
import os
import sys

(
    path,
    run_tag,
    host,
    mode,
    limit_conditions,
    model_key,
    gpu_list,
    port,
    phase,
    process_id,
    exit_code,
) = sys.argv[1:]
record = {
    "schema": "ctm.eval_awareness.figure6_vast_status.v1",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    "run_tag": run_tag,
    "host": host,
    "mode": mode,
    "limit_conditions": int(limit_conditions),
    "model_key": model_key,
    "gpu_list": gpu_list,
    "port": int(port),
    "phase": phase,
    "pid": None if process_id == "-" else int(process_id),
    "exit_code": None if exit_code == "-" else int(exit_code),
}
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

PIDS=()
MODEL_KEYS=()
GPU_LISTS=()
PORTS=()
ACTIVE=()

cleanup_children() {
    local status=$?
    trap - EXIT
    for ((index = 0; index < ${#PIDS[@]}; index++)); do
        if [[ "${ACTIVE[$index]:-0}" == 1 ]]; then
            if kill -0 "${PIDS[$index]}" 2>/dev/null; then
                kill "${PIDS[$index]}" 2>/dev/null || true
                wait "${PIDS[$index]}" 2>/dev/null || true
            fi
            record_status "${MODEL_KEYS[$index]}" "${GPU_LISTS[$index]}" "${PORTS[$index]}" aborted "${PIDS[$index]}" "$status" || true
            ACTIVE[$index]=0
        fi
    done
    exit "$status"
}
trap cleanup_children EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for line in "${PLAN_LINES[@]}"; do
    IFS=$'\t' read -r model_key gpu_list port extra <<< "$line"
    if [[ -z "$model_key" || -z "$gpu_list" || -z "$port" || -n "${extra:-}" ]]; then
        echo "ERROR: invalid validated plan row." >&2
        exit 2
    fi
    runner_log="$FIGURE6_OUTPUT_ROOT/_logs/${model_key}.runner.log"
    printf '\n[%s] launching mode=%s GPUs=%s port=%s\n' "$RUN_TAG" "$MODE" "$gpu_list" "$port" >> "$runner_log"
    "$RUNNER" "$model_key" "$gpu_list" "$port" >> "$runner_log" 2>&1 &
    process_id=$!
    PIDS+=("$process_id")
    MODEL_KEYS+=("$model_key")
    GPU_LISTS+=("$gpu_list")
    PORTS+=("$port")
    ACTIVE+=(1)
    record_status "$model_key" "$gpu_list" "$port" started "$process_id" -
    echo "Launched $model_key on GPUs $gpu_list (status log: $STATUS_LOG)."
done

OVERALL_STATUS=0
for ((index = 0; index < ${#PIDS[@]}; index++)); do
    if wait "${PIDS[$index]}"; then
        child_status=0
    else
        child_status=$?
        OVERALL_STATUS=1
    fi
    ACTIVE[$index]=0
    record_status "${MODEL_KEYS[$index]}" "${GPU_LISTS[$index]}" "${PORTS[$index]}" completed "${PIDS[$index]}" "$child_status"
    echo "${MODEL_KEYS[$index]} finished with exit code $child_status."
done

if (( OVERALL_STATUS != 0 )); then
    echo "ERROR: one or more Figure 6 model runners failed; inspect append-only runner logs." >&2
    exit 1
fi
echo "Host $HOST $MODE run completed successfully."
