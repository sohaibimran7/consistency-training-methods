#!/usr/bin/env bash
# Serve one pinned Figure 6 model and resume its append-only generation log.
#
# Usage:
#   run_figure6_model.sh MODEL_KEY
#   run_figure6_model.sh qwen ARRAY_INDEX
#   run_figure6_model.sh llama ARRAY_INDEX
#
# Required environment:
#   REPO_DIR, HF_HOME, FIGURE6_ARTIFACT, FIGURE6_OUTPUT_ROOT
#   PAPER_NATURAL_PROMPT_PATH       for Qwen models
#   EXPLICIT_SCRATCHPAD_PROMPT_PATH for Llama models
#   HF_TOKEN                        for the gated Llama checkpoint
set -euo pipefail

usage() {
    echo "Usage: $0 MODEL_KEY | $0 {qwen|llama} ARRAY_INDEX" >&2
    exit 2
}

resolve_model_key() {
    if [[ $# -eq 1 ]]; then
        case "$1" in
            qwen36|qwen32|qwen_mo_mid|qwen_mo_post|llama33|llama_mo_mid|llama_mo_post)
                printf '%s\n' "$1"
                ;;
            *) usage ;;
        esac
        return
    fi
    [[ $# -eq 2 && "$2" =~ ^[0-9]+$ ]] || usage
    case "$1:$2" in
        qwen:0) printf '%s\n' qwen36 ;;
        qwen:1) printf '%s\n' qwen32 ;;
        qwen:2) printf '%s\n' qwen_mo_mid ;;
        qwen:3) printf '%s\n' qwen_mo_post ;;
        llama:0) printf '%s\n' llama33 ;;
        llama:1) printf '%s\n' llama_mo_mid ;;
        llama:2) printf '%s\n' llama_mo_post ;;
        *) usage ;;
    esac
}

require_env() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable $name is not set." >&2
        exit 2
    fi
}

MODEL_KEY=$(resolve_model_key "$@")
require_env REPO_DIR
require_env HF_HOME
require_env FIGURE6_ARTIFACT
require_env FIGURE6_OUTPUT_ROOT

for path_name in REPO_DIR HF_HOME FIGURE6_ARTIFACT FIGURE6_OUTPUT_ROOT; do
    if [[ "${!path_name}" != /* ]]; then
        echo "ERROR: $path_name must be an absolute path." >&2
        exit 2
    fi
done
shared_cache=false
if [[ -n "${PROJECTDIR:-}" && "$HF_HOME" == "$PROJECTDIR"/* ]]; then
    shared_cache=true
fi
if [[ -n "${SCRATCHDIR:-}" && "$HF_HOME" == "$SCRATCHDIR"/* ]]; then
    shared_cache=true
fi
if [[ "$shared_cache" != true ]]; then
    echo "ERROR: HF_HOME must be inside the shared project or scratch filesystem." >&2
    exit 2
fi
if [[ ! -f "$FIGURE6_ARTIFACT" || ! -f "$FIGURE6_ARTIFACT.manifest.json" ]]; then
    echo "ERROR: FIGURE6_ARTIFACT and its .manifest.json sidecar must exist." >&2
    exit 2
fi

cd "$REPO_DIR"
source .venv/bin/activate
source infra/isambard/activate_gpu_runtime.sh
mkdir -p "$HF_HOME" "$FIGURE6_OUTPUT_ROOT/$MODEL_KEY" "$FIGURE6_OUTPUT_ROOT/_logs"
export HF_HOME

MODELS_CONFIG="$REPO_DIR/experiments/eval_awareness/figure6/models.yaml"
mapfile -t MODEL_FIELDS < <(
    python - "$MODELS_CONFIG" "$MODEL_KEY" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    model = yaml.safe_load(handle)["models"][sys.argv[2]]
for field in (
    "model_id",
    "revision",
    "tensor_parallel_size",
    "dtype",
    "prompt_key",
    "reasoning_parser",
    "language_model_only",
    "gated",
):
    value = model[field]
    print(str(value).lower() if isinstance(value, bool) else value)
PY
)
if [[ ${#MODEL_FIELDS[@]} -ne 8 ]]; then
    echo "ERROR: could not resolve $MODEL_KEY from the model registry." >&2
    exit 2
fi

MODEL_ID=${MODEL_FIELDS[0]}
MODEL_REVISION=${MODEL_FIELDS[1]}
TENSOR_PARALLEL_SIZE=${MODEL_FIELDS[2]}
MODEL_DTYPE=${MODEL_FIELDS[3]}
PROMPT_KEY=${MODEL_FIELDS[4]}
REASONING_PARSER=${MODEL_FIELDS[5]}
LANGUAGE_MODEL_ONLY=${MODEL_FIELDS[6]}
MODEL_GATED=${MODEL_FIELDS[7]}

case "$PROMPT_KEY" in
    paper_natural)
        require_env PAPER_NATURAL_PROMPT_PATH
        PROMPT_PATH=$PAPER_NATURAL_PROMPT_PATH
        ;;
    explicit_scratchpad)
        require_env EXPLICIT_SCRATCHPAD_PROMPT_PATH
        PROMPT_PATH=$EXPLICIT_SCRATCHPAD_PROMPT_PATH
        ;;
    *)
        echo "ERROR: unsupported prompt key $PROMPT_KEY." >&2
        exit 2
        ;;
esac
if [[ "$PROMPT_PATH" != /* || ! -f "$PROMPT_PATH" ]]; then
    echo "ERROR: the selected external prompt path must be an existing absolute file." >&2
    exit 2
fi
if [[ "$MODEL_GATED" == true && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is required for the gated $MODEL_KEY checkpoint." >&2
    exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<< "$CUDA_VISIBLE_DEVICES"
    if [[ ${#visible_devices[@]} -ne "$TENSOR_PARALLEL_SIZE" ]]; then
        echo "ERROR: $MODEL_KEY needs $TENSOR_PARALLEL_SIZE visible GPU(s); found ${#visible_devices[@]}." >&2
        exit 2
    fi
fi

VLLM_PORT=${VLLM_PORT:-8000}
if [[ ! "$VLLM_PORT" =~ ^[0-9]+$ ]] || (( VLLM_PORT < 1024 || VLLM_PORT > 65535 )); then
    echo "ERROR: VLLM_PORT must be an integer from 1024 through 65535." >&2
    exit 2
fi
VLLM_STARTUP_TIMEOUT_SECONDS=${VLLM_STARTUP_TIMEOUT_SECONDS:-1800}
if [[ ! "$VLLM_STARTUP_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( VLLM_STARTUP_TIMEOUT_SECONDS < 1 )); then
    echo "ERROR: VLLM_STARTUP_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 2
fi

BASE_URL="http://127.0.0.1:$VLLM_PORT"
JOB_TAG=${SLURM_JOB_ID:-manual}
SERVER_LOG="$FIGURE6_OUTPUT_ROOT/_logs/${MODEL_KEY}-vllm-${JOB_TAG}.log"
GENERATION_LOG="$FIGURE6_OUTPUT_ROOT/$MODEL_KEY/generations.jsonl"
SERVER_PID=""

cleanup() {
    local status=$?
    trap - EXIT
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit "$status"
}
terminate() {
    exit 143
}
trap cleanup EXIT
trap terminate TERM
trap 'exit 130' INT

VLLM_ARGS=(
    serve "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --served-model-name "$MODEL_ID"
    --dtype "$MODEL_DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}"
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
    --host 127.0.0.1
    --port "$VLLM_PORT"
)
if [[ "$REASONING_PARSER" == qwen3 ]]; then
    VLLM_ARGS+=(--reasoning-parser qwen3)
fi
if [[ "$LANGUAGE_MODEL_ONLY" == true ]]; then
    VLLM_ARGS+=(--language-model-only)
fi

echo "Starting pinned $MODEL_KEY server (TP=$TENSOR_PARALLEL_SIZE, dtype=$MODEL_DTYPE)."
vllm "${VLLM_ARGS[@]}" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
until curl --fail --silent --show-error "$BASE_URL/health" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: vLLM exited before becoming healthy; see $SERVER_LOG." >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "ERROR: vLLM did not become healthy within $VLLM_STARTUP_TIMEOUT_SECONDS seconds; see $SERVER_LOG." >&2
        exit 1
    fi
    sleep 5
done

GENERATOR_ARGS=(
    --artifact "$FIGURE6_ARTIFACT"
    --output "$GENERATION_LOG"
    --model-key "$MODEL_KEY"
    --prompt-path "$PROMPT_PATH"
    --base-url "$BASE_URL/v1"
    --api-key-env FIGURE6_LOCAL_ENDPOINT_TOKEN
    --replicates 3
    --temperature 0.3
    --max-tokens 4096
    --max-concurrency "${FIGURE6_MAX_CONCURRENCY:-10}"
)
if [[ -n "${FIGURE6_LIMIT_CONDITIONS:-}" ]]; then
    if [[ ! "$FIGURE6_LIMIT_CONDITIONS" =~ ^[0-9]+$ ]] || (( FIGURE6_LIMIT_CONDITIONS < 1 )); then
        echo "ERROR: FIGURE6_LIMIT_CONDITIONS must be a positive integer." >&2
        exit 2
    fi
    GENERATOR_ARGS+=(--limit-conditions "$FIGURE6_LIMIT_CONDITIONS")
fi

echo "Server healthy; resuming generation output at $GENERATION_LOG."
python -m ctm_data.adapters.eval_awareness.figure6_generate "${GENERATOR_ARGS[@]}"
