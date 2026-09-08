#!/usr/bin/env bash
# Produce one immutable Qwen3.5 worker-parity attestation for any local
# on-policy target.  This is a non-production bootstrap: it performs a short
# real LoRA update, then proves the translated vLLM adapter changes scores on
# every rollout worker and agrees with the HF/PEFT coordinator.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=<coordinator-plus-workers> \
    bash infra/vastai/preflight_qwen35_rollout_workers.sh \
      --experiment NAME --run NAME --worker-gpus LOGICAL_IDS \
      --worker-gpu-mem-util FRACTION --worker-max-model-len TOKENS \
      --worker-max-num-seqs N --worker-max-num-batched-tokens TOKENS \
      --worker-seed-base SEED \
      --target-logprob-chunk-size TOKENS [--worker-gdn-prefill-backend BACKEND] \
      [--label LABEL] [--resume-attestation] [--dry-run]

The status directory is always derived as
logs/NAME/RUN/rollout_workers, which is the default consumed by the training
CLI.  Use the exact worker options resolved by the future training target.
The preflight creates immutable evidence and refuses to replace a prior
attestation or preflight log in that namespace. `--resume-attestation` is the
only exception: it performs a no-model revalidation of existing immutable
evidence and skips the bootstrap. It refuses a namespace containing production
rollout sessions, which require a fresh training identity rather than an
unsafe rerun. Production sessions are direct ``session-*`` children of the
status directory; the bootstrap's nested ``workers/session-*`` tree is not a
production marker and remains resumable. A launcher-written training-started
marker and legacy root-level adapter snapshots also block unsafe reuse.

`--dry-run` prints the derived contract without probing CUDA or writing files.
EOF
}

model="Qwen/Qwen3.5-9B"
label="qwen35-onpolicy"
experiment=""
run_name=""
worker_gpus=""
worker_gpu_mem_util=""
worker_max_model_len=""
worker_max_num_seqs=""
worker_max_num_batched_tokens=""
worker_seed_base=""
worker_gdn_prefill_backend=""
target_logprob_chunk_size=""
dry_run=false
resume_attestation=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment|--run|--worker-gpus|--worker-gpu-mem-util|--worker-max-model-len|--worker-max-num-seqs|--worker-max-num-batched-tokens|--worker-seed-base|--worker-gdn-prefill-backend|--target-logprob-chunk-size|--label|--model)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: $1 requires a non-empty value" >&2
                exit 2
            fi
            case "$1" in
                --experiment) experiment="$2" ;;
                --run) run_name="$2" ;;
                --worker-gpus) worker_gpus="$2" ;;
                --worker-gpu-mem-util) worker_gpu_mem_util="$2" ;;
                --worker-max-model-len) worker_max_model_len="$2" ;;
                --worker-max-num-seqs) worker_max_num_seqs="$2" ;;
                --worker-max-num-batched-tokens) worker_max_num_batched_tokens="$2" ;;
                --worker-seed-base) worker_seed_base="$2" ;;
                --worker-gdn-prefill-backend) worker_gdn_prefill_backend="$2" ;;
                --target-logprob-chunk-size) target_logprob_chunk_size="$2" ;;
                --label) label="$2" ;;
                --model) model="$2" ;;
            esac
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --resume-attestation)
            resume_attestation=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

for required in experiment run_name worker_gpus worker_gpu_mem_util worker_max_model_len worker_max_num_seqs worker_max_num_batched_tokens worker_seed_base target_logprob_chunk_size; do
    if [[ -z "${!required}" ]]; then
        echo "ERROR: missing required option for $required" >&2
        usage >&2
        exit 2
    fi
done
if [[ ! "$worker_seed_base" =~ ^[0-9]+$ ]] || (( worker_seed_base > 2147483645 )); then
    echo "ERROR: --worker-seed-base must be an integer in [0, 2147483645] for three workers" >&2
    exit 2
fi
if [[ "$model" != "Qwen/Qwen3.5-9B" ]]; then
    echo "ERROR: this transport preflight is scoped to Qwen/Qwen3.5-9B; got $model" >&2
    exit 2
fi
if [[ ! "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ || ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "ERROR: --experiment and --run must be safe non-empty run identifiers" >&2
    exit 2
fi
if [[ ! "$worker_gpus" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "ERROR: --worker-gpus must be comma-separated logical GPU indices" >&2
    exit 2
fi
if [[ -n "$worker_gdn_prefill_backend" && "$worker_gdn_prefill_backend" != "flashinfer" && "$worker_gdn_prefill_backend" != "triton" ]]; then
    echo "ERROR: --worker-gdn-prefill-backend must be flashinfer or triton" >&2
    exit 2
fi
for numeric in worker_max_model_len worker_max_num_seqs worker_max_num_batched_tokens target_logprob_chunk_size; do
    if [[ ! "${!numeric}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --${numeric//_/-} must be a positive integer" >&2
        exit 2
    fi
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
status_dir="$repo_root/logs/$experiment/$run_name/rollout_workers"
attestation="$status_dir/qwen35-rollout-worker-parity-attestation.json"
log_path="$status_dir/qwen35-rollout-worker-parity-preflight.log"
training_started_marker="$status_dir/qwen35-onpolicy-training-started.json"

# ``RolloutWorkerPool`` creates its production state as
#   rollout_workers/session-*/
# before it creates either IPC or the adapter root. Looking only for
# ``rollout_workers/adapters`` (or even ``session-*/ipc``) therefore misses a
# process killed during construction and lets --resume-attestation relaunch it.
# The non-production bootstrap instead uses ``rollout_workers/workers/session-*``;
# deliberately inspect only direct session children so retained bootstrap
# evidence is not mistaken for a production run.
production_rollout_session() {
    local candidate
    for candidate in "$status_dir"/session-*; do
        [[ -d "$candidate" ]] || continue
        printf '%s\n' "$candidate"
        return 0
    done
    return 1
}

if [[ "$dry_run" == true ]]; then
    printf '%s\n' \
        "QWEN35_ROLLOUT_WORKER_PREFLIGHT_DRY_RUN=1" \
        "label=$label" \
        "model=$model" \
        "experiment_name=$experiment" \
        "run_name=$run_name" \
        "status_dir=$status_dir" \
        "attestation=$attestation" \
        "training_started_marker=$training_started_marker" \
        "coordinator_logical_gpu=0" \
        "rollout_worker_logical_gpus=$worker_gpus" \
        "worker_gpu_memory_utilization=$worker_gpu_mem_util" \
        "worker_max_model_len=$worker_max_model_len" \
        "worker_max_num_seqs=$worker_max_num_seqs" \
        "worker_max_num_batched_tokens=$worker_max_num_batched_tokens" \
        "worker_seed_base=$worker_seed_base" \
        "worker_gdn_prefill_backend=${worker_gdn_prefill_backend:-auto}" \
        "target_logprob_chunk_size=$target_logprob_chunk_size" \
        "resume_attestation=$resume_attestation" \
        "probe=1_prompt_x_12_rollouts_x_32_generated_tokens_with_cross_worker_rng_diversity"
    exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES to one coordinator plus the requested rollout workers." >&2
    exit 2
fi

production_session=""
if production_session="$(production_rollout_session)"; then
    :
fi

if [[ "$resume_attestation" == true ]]; then
    if [[ ! -f "$attestation" ]]; then
        echo "ERROR: --resume-attestation requires existing immutable evidence: $attestation" >&2
        exit 2
    fi
    if [[ -n "$production_session" ]]; then
        echo "ERROR: detected production rollout session: $production_session (direct session-* marker)." >&2
        echo "The local on-policy trainer cannot safely restart this namespace without a training-state resume; use a fresh experiment/run identity rather than overwriting or reusing v1." >&2
        exit 2
    fi
    if [[ -e "$training_started_marker" ]]; then
        echo "ERROR: detected immutable training-started marker: $training_started_marker" >&2
        echo "The target may have entered the runner or child before a rollout session was created; use a fresh experiment/run identity rather than attempting an unsafe restart." >&2
        exit 2
    fi
    if [[ -e "$status_dir/adapters" ]]; then
        echo "ERROR: $status_dir has legacy/root-level production adapter snapshots." >&2
        echo "Use a fresh experiment/run identity rather than reusing a namespace with production residue." >&2
        exit 2
    fi
elif [[ -e "$attestation" || -n "$production_session" || -e "$training_started_marker" || -e "$status_dir/adapters" || -e "$log_path" ]]; then
    echo "ERROR: refusing to replace or mix existing worker-parity evidence in $status_dir" >&2
    if [[ -n "$production_session" ]]; then
        echo "Detected production rollout session: $production_session (direct session-* marker)." >&2
    fi
    if [[ -e "$training_started_marker" ]]; then
        echo "Detected immutable training-started marker: $training_started_marker." >&2
    fi
    if [[ -e "$status_dir/adapters" ]]; then
        echo "Detected legacy/root-level production adapter snapshots: $status_dir/adapters." >&2
    fi
    echo "Use --resume-attestation only to validate a completed preflight with no production rollout session; otherwise use a fresh experiment/run namespace." >&2
    exit 2
fi

python_bin="${CTM_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="python3"
    fi
fi
if ! "$python_bin" -c 'import sys' >/dev/null 2>&1; then
    echo "ERROR: CTM_PYTHON is not an executable Python interpreter: $python_bin" >&2
    exit 2
fi

# Resolve logical worker identities and reject incompatible options before a
# model is initialized.  This does not query CUDA; it only validates the
# inherited allocation string and exact vLLM worker contract.
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
WORKER_GPUS="$worker_gpus" \
WORKER_GPU_MEM_UTIL="$worker_gpu_mem_util" \
WORKER_MAX_MODEL_LEN="$worker_max_model_len" \
WORKER_MAX_NUM_SEQS="$worker_max_num_seqs" \
WORKER_MAX_NUM_BATCHED_TOKENS="$worker_max_num_batched_tokens" \
WORKER_SEED_BASE="$worker_seed_base" \
TARGET_LOGPROB_CHUNK_SIZE="$target_logprob_chunk_size" \
"$python_bin" - <<'PY'
from __future__ import annotations

import math
import os

from ctm.backends.local.rollout_workers import resolve_rollout_gpus

memory = float(os.environ["WORKER_GPU_MEM_UTIL"])
if not math.isfinite(memory) or not 0 < memory <= 1:
    raise ValueError("worker GPU memory utilization must be finite and in (0, 1]")
for name in (
    "WORKER_MAX_MODEL_LEN",
    "WORKER_MAX_NUM_SEQS",
    "WORKER_MAX_NUM_BATCHED_TOKENS",
    "TARGET_LOGPROB_CHUNK_SIZE",
):
    if int(os.environ[name]) < 1:
        raise ValueError(f"{name} must be positive")
seed_base = int(os.environ["WORKER_SEED_BASE"])
if not 0 <= seed_base <= 2**31 - len(os.environ["WORKER_GPUS"].split(",")):
    raise ValueError("WORKER_SEED_BASE cannot cover every derived worker seed")
workers = resolve_rollout_gpus(
    os.environ["WORKER_GPUS"],
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    coordinator_device="cuda:0",
)
if not workers:
    raise ValueError("worker parity preflight requires at least one rollout worker")
print("QWEN35_ROLLOUT_WORKER_PREFLIGHT_TOPOLOGY=" + ",".join(worker.device_token for worker in workers))
PY

validate_existing_attestation() {
    STATUS_DIR="$status_dir" \
    WORKER_GPUS="$worker_gpus" \
    WORKER_GPU_MEM_UTIL="$worker_gpu_mem_util" \
    WORKER_MAX_MODEL_LEN="$worker_max_model_len" \
    WORKER_MAX_NUM_SEQS="$worker_max_num_seqs" \
    WORKER_MAX_NUM_BATCHED_TOKENS="$worker_max_num_batched_tokens" \
    WORKER_SEED_BASE="$worker_seed_base" \
    WORKER_GDN_PREFILL_BACKEND="$worker_gdn_prefill_backend" \
    "$python_bin" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from ctm.backends.local.qwen35_vllm_compat import (
    WORKER_PARITY_ATTESTATION_NAME,
    file_sha256,
    validate_qwen35_rollout_worker_parity_attestation,
)
from ctm.backends.local.rollout_workers import resolve_rollout_gpus

workers = resolve_rollout_gpus(
    os.environ["WORKER_GPUS"],
    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    coordinator_device="cuda:0",
)
engine_kwargs = {
    "gpu_memory_utilization": float(os.environ["WORKER_GPU_MEM_UTIL"]),
    "max_model_len": int(os.environ["WORKER_MAX_MODEL_LEN"]),
    "max_num_seqs": int(os.environ["WORKER_MAX_NUM_SEQS"]),
    "max_num_batched_tokens": int(os.environ["WORKER_MAX_NUM_BATCHED_TOKENS"]),
    "language_model_only": True,
    "logprobs_mode": "processed_logprobs",
    "tensor_parallel_size": 1,
    "seed": int(os.environ["WORKER_SEED_BASE"]),
}
if os.environ["WORKER_GDN_PREFILL_BACKEND"]:
    engine_kwargs["gdn_prefill_backend"] = os.environ["WORKER_GDN_PREFILL_BACKEND"]
path = Path(os.environ["STATUS_DIR"]) / WORKER_PARITY_ATTESTATION_NAME
document = validate_qwen35_rollout_worker_parity_attestation(
    path,
    expected_model="Qwen/Qwen3.5-9B",
    expected_worker_gpus=workers,
    expected_worker_engine_kwargs=engine_kwargs,
)
print("QWEN35_ROLLOUT_WORKER_PARITY_ATTESTATION=" + str(path.resolve()))
print("QWEN35_ROLLOUT_WORKER_PARITY_SHA256=" + file_sha256(path))
print("QWEN35_ROLLOUT_WORKER_COUNT=" + str(len(document["worker_gpus"])))
PY
}

if [[ "$resume_attestation" == true ]]; then
    validate_existing_attestation
    echo "QWEN35_ROLLOUT_WORKER_PREFLIGHT_RESUMED=1"
    exit 0
fi

mkdir -p "$status_dir"
echo "Qwen3.5 rollout-worker preflight label=$label experiment=$experiment run=$run_name" | tee "$log_path"
echo "status_dir=$status_dir" | tee -a "$log_path"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$log_path"

# Twelve short same-prompt rollouts exercise four matched lanes on every
# worker and fail if all lanes show the cloned-seed signature. The benchmark
# then duplicates a selected nonzero completion across all workers, so no
# individual engine can hide behind an aggregate score.
benchmark_args=(
    --model "$model"
    --worker-gpus "$worker_gpus"
    --output-dir "$status_dir"
    --prompt-count 1
    --rollouts-per-prompt 12
    --max-new-tokens 32
    --temperature 0.7
    --ignore-eos
    --target-logprob-chunk-size "$target_logprob_chunk_size"
    --worker-gpu-mem-util "$worker_gpu_mem_util"
    --worker-max-model-len "$worker_max_model_len"
    --worker-max-num-seqs "$worker_max_num_seqs"
    --worker-max-num-batched-tokens "$worker_max_num_batched_tokens"
    --worker-seed-base "$worker_seed_base"
    --post-update-min-effect 1e-5
)
if [[ -n "$worker_gdn_prefill_backend" ]]; then
    benchmark_args+=(--worker-gdn-prefill-backend "$worker_gdn_prefill_backend")
fi
"$python_bin" infra/vastai/benchmark_qwen35_opct_group.py "${benchmark_args[@]}" 2>&1 | tee -a "$log_path"

# Rebuild and validate the sidecar from its raw/translated adapter bytes and
# require the same options/topology the training CLI will resolve.
validate_existing_attestation 2>&1 | tee -a "$log_path"
