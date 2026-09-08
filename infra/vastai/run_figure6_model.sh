#!/usr/bin/env bash
# Serve one pinned Figure 6 model on assigned Vast GPUs and resume its log.
set -euo pipefail
set +x

usage() {
    echo "Usage: $0 MODEL_KEY GPU_LIST PORT" >&2
    exit 2
}

require_env() {
    local name=$1
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable $name is not set." >&2
        exit 2
    fi
}

require_absolute_path() {
    local name=$1
    if [[ "${!name}" != /* ]]; then
        echo "ERROR: $name must be an absolute path." >&2
        exit 2
    fi
}

load_private_hf_token() {
    if [[ -n "${HF_TOKEN_FILE:-}" && -n "${HF_TOKEN:-}" ]]; then
        echo "ERROR: set only one of HF_TOKEN_FILE or HF_TOKEN." >&2
        exit 2
    fi
    if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
        if [[ "$HF_TOKEN_FILE" != /* ]]; then
            echo "ERROR: HF_TOKEN_FILE must be an absolute path." >&2
            exit 2
        fi
        python - "$HF_TOKEN_FILE" <<'PY'
import os
import stat
import sys

path = sys.argv[1]
metadata = os.lstat(path)
if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
    raise SystemExit("ERROR: HF_TOKEN_FILE must be a regular file, not a symlink.")
if stat.S_IMODE(metadata.st_mode) != 0o600:
    raise SystemExit("ERROR: HF_TOKEN_FILE permissions must be exactly 600.")
if metadata.st_uid != os.geteuid():
    raise SystemExit("ERROR: HF_TOKEN_FILE must be owned by the current user.")
with open(path, "rb") as handle:
    payload = handle.read()
token = payload.rstrip(b"\r\n")
valid_payloads = {token, token + b"\n", token + b"\r\n"}
if not token or payload not in valid_payloads or token != token.strip() or b"\n" in token or b"\r" in token:
    raise SystemExit("ERROR: HF_TOKEN_FILE must contain exactly one non-empty token line.")
PY
        IFS= read -r HF_TOKEN < "$HF_TOKEN_FILE" || true
        export HF_TOKEN
    fi
    if [[ -n "${HF_TOKEN:-}" ]]; then
        python - <<'PY'
import os

token = os.environ["HF_TOKEN"]
if not token or token != token.strip() or any(character.isspace() for character in token):
    raise SystemExit("ERROR: HF_TOKEN must be one non-empty token without whitespace.")
PY
    fi
}

[[ $# -eq 3 ]] || usage
MODEL_KEY=$1
GPU_LIST=$2
VLLM_PORT=$3

case "$MODEL_KEY" in
    qwen36|qwen32|qwen_mo_mid|qwen_mo_post|llama33|llama_mo_mid|llama_mo_post) ;;
    *) usage ;;
esac
if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: GPU_LIST must be a comma-separated list of physical GPU indices." >&2
    exit 2
fi
if [[ ! "$VLLM_PORT" =~ ^[0-9]+$ ]] || (( VLLM_PORT < 1024 || VLLM_PORT > 65535 )); then
    echo "ERROR: PORT must be an integer from 1024 through 65535." >&2
    exit 2
fi
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must not be present on a Vast generation host." >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd -P)
FIGURE6_VENV=${FIGURE6_VENV:-$REPO_DIR/.venv-figure6}
MODELS_CONFIG="$REPO_DIR/experiments/eval_awareness/figure6/models.yaml"

for name in HF_HOME FIGURE6_ARTIFACT FIGURE6_OUTPUT_ROOT; do
    require_env "$name"
    require_absolute_path "$name"
done
if [[ ! -x "$FIGURE6_VENV/bin/python" ]]; then
    echo "ERROR: missing Figure 6 venv; run infra/vastai/provision_figure6.sh first." >&2
    exit 2
fi
if [[ ! -f "$FIGURE6_ARTIFACT" || ! -f "$FIGURE6_ARTIFACT.manifest.json" ]]; then
    echo "ERROR: FIGURE6_ARTIFACT and its .manifest.json sidecar must exist." >&2
    exit 2
fi

source "$FIGURE6_VENV/bin/activate"
MODEL_FIELDS_TEXT=$(python - "$MODELS_CONFIG" "$MODEL_KEY" <<'PY'
import sys

import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    registry = yaml.safe_load(handle)
if not isinstance(registry, dict) or not isinstance(registry.get("models"), dict):
    raise SystemExit("ERROR: invalid Figure 6 model registry.")
try:
    model = registry["models"][sys.argv[2]]
except KeyError as exc:
    raise SystemExit(f"ERROR: model is absent from registry: {sys.argv[2]}") from exc
fields = (
    "model_id",
    "revision",
    "tensor_parallel_size",
    "dtype",
    "prompt_key",
    "reasoning_parser",
    "language_model_only",
    "gated",
)
missing = [field for field in fields if field not in model]
if missing:
    raise SystemExit(f"ERROR: model registry entry is missing fields: {missing}")
for field in fields:
    value = model[field]
    print(str(value).lower() if isinstance(value, bool) else value)
PY
)
mapfile -t MODEL_FIELDS <<< "$MODEL_FIELDS_TEXT"
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

if [[ ! "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: $MODEL_KEY does not have a full pinned revision." >&2
    exit 2
fi
if [[ "$MODEL_DTYPE" != bfloat16 ]]; then
    echo "ERROR: $MODEL_KEY must use bfloat16; registry has $MODEL_DTYPE." >&2
    exit 2
fi
if [[ ! "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]] || (( TENSOR_PARALLEL_SIZE < 1 )); then
    echo "ERROR: $MODEL_KEY has an invalid tensor_parallel_size." >&2
    exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "$GPU_LIST"
if [[ ${#GPU_IDS[@]} -ne "$TENSOR_PARALLEL_SIZE" ]]; then
    echo "ERROR: $MODEL_KEY needs $TENSOR_PARALLEL_SIZE GPU(s); assignment has ${#GPU_IDS[@]}." >&2
    exit 2
fi
for ((left = 0; left < ${#GPU_IDS[@]}; left++)); do
    for ((right = left + 1; right < ${#GPU_IDS[@]}; right++)); do
        if [[ "${GPU_IDS[$left]}" == "${GPU_IDS[$right]}" ]]; then
            echo "ERROR: GPU_LIST repeats physical GPU ${GPU_IDS[$left]}." >&2
            exit 2
        fi
    done
done

case "$PROMPT_KEY" in
    paper_natural)
        require_env PAPER_NATURAL_PROMPT_PATH
        require_absolute_path PAPER_NATURAL_PROMPT_PATH
        PROMPT_PATH=$PAPER_NATURAL_PROMPT_PATH
        ;;
    explicit_scratchpad)
        require_env EXPLICIT_SCRATCHPAD_PROMPT_PATH
        require_absolute_path EXPLICIT_SCRATCHPAD_PROMPT_PATH
        PROMPT_PATH=$EXPLICIT_SCRATCHPAD_PROMPT_PATH
        ;;
    *)
        echo "ERROR: unsupported prompt key $PROMPT_KEY." >&2
        exit 2
        ;;
esac
if [[ ! -f "$PROMPT_PATH" ]]; then
    echo "ERROR: selected external prompt path does not exist." >&2
    exit 2
fi

python - "$FIGURE6_ARTIFACT" "$MODEL_KEY" "$PROMPT_PATH" <<'PY'
import sys

from ctm_data.adapters.eval_awareness.figure6_materialize import load_figure6_artifact
from ctm_data.adapters.eval_awareness.figure6_spec import load_verified_model_prompt

rows, _ = load_figure6_artifact(sys.argv[1])
load_verified_model_prompt(sys.argv[2], sys.argv[3])
print(f"Verified immutable Figure 6 inputs: conditions={len(rows)}, model={sys.argv[2]}")
PY

load_private_hf_token
if [[ "$MODEL_GATED" == true && -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN_FILE or HF_TOKEN is required for gated model $MODEL_KEY." >&2
    exit 2
fi

if [[ -n "${FIGURE6_LIMIT_CONDITIONS:-}" ]]; then
    if [[ ! "$FIGURE6_LIMIT_CONDITIONS" =~ ^[0-9]+$ ]] || (( FIGURE6_LIMIT_CONDITIONS < 1 )); then
        echo "ERROR: FIGURE6_LIMIT_CONDITIONS must be a positive integer when set." >&2
        exit 2
    fi
fi
VLLM_STARTUP_TIMEOUT_SECONDS=${VLLM_STARTUP_TIMEOUT_SECONDS:-1800}
if [[ ! "$VLLM_STARTUP_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( VLLM_STARTUP_TIMEOUT_SECONDS < 1 )); then
    echo "ERROR: VLLM_STARTUP_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export HF_HOME
export FIGURE6_LOCAL_ENDPOINT_TOKEN=vast-local-vllm-dummy
mkdir -p "$HF_HOME" "$FIGURE6_OUTPUT_ROOT/$MODEL_KEY" "$FIGURE6_OUTPUT_ROOT/_logs"
for command_name in curl vllm; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required image command is unavailable: $command_name" >&2
        exit 2
    fi
done

python - "$TENSOR_PARALLEL_SIZE" <<'PY'
import sys

import torch
import vllm

expected_gpus = int(sys.argv[1])
assert vllm.__version__ == "0.26.0", f"expected vLLM 0.26.0, found {vllm.__version__}"
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == expected_gpus, (
    f"CUDA assignment exposes {torch.cuda.device_count()} GPUs; expected {expected_gpus}"
)
for gpu_id in range(expected_gpus):
    with torch.cuda.device(gpu_id):
        assert torch.cuda.is_bf16_supported(), f"CUDA device {gpu_id} does not support bfloat16"
print(f"Runtime verified for {expected_gpus} assigned CUDA GPU(s): vllm={vllm.__version__}, dtype=bfloat16")
PY

python - "$VLLM_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as probe:
    probe.bind(("127.0.0.1", port))
PY

BASE_URL="http://127.0.0.1:$VLLM_PORT"
GENERATION_LOG="$FIGURE6_OUTPUT_ROOT/$MODEL_KEY/generations.jsonl"
SERVER_LOG="$FIGURE6_OUTPUT_ROOT/_logs/${MODEL_KEY}.vllm.log"
RUN_TAG=${FIGURE6_RUN_TAG:-manual}
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._:-]+$ ]]; then
    echo "ERROR: FIGURE6_RUN_TAG contains unsafe characters." >&2
    exit 2
fi
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
    --max-model-len 8192
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
if [[ "$MODEL_KEY" == qwen36 ]]; then
    # Qwen3.6 has one recurrent-state cache block per decode sequence. vLLM's
    # H100 default of 1024 exceeds the blocks available after loading this
    # checkpoint; 256 remains far above the fixed generation concurrency 10.
    VLLM_ARGS+=(--max-num-seqs 256)
fi
if [[ "$MODEL_KEY" == llama_mo_post ]]; then
    # CUDA-graph capture for this checkpoint faults on the two H100 PCIe host;
    # eager execution keeps the same model and sampling protocol.
    VLLM_ARGS+=(--enforce-eager)
fi

printf '\n[%s] starting %s on GPUs %s, port %s\n' "$RUN_TAG" "$MODEL_KEY" "$GPU_LIST" "$VLLM_PORT" >> "$SERVER_LOG"
echo "Starting pinned $MODEL_KEY server on assigned GPUs $GPU_LIST (TP=$TENSOR_PARALLEL_SIZE)."
vllm "${VLLM_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
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
    GENERATOR_ARGS+=(--limit-conditions "$FIGURE6_LIMIT_CONDITIONS")
fi

echo "Server healthy; resuming append-only generation output at $GENERATION_LOG."
python -m ctm_data.adapters.eval_awareness.figure6_generate "${GENERATOR_ARGS[@]}"
