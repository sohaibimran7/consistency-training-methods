#!/usr/bin/env bash
# Produce the immutable fixed-token Qwen3.5 rollout-worker attestation for
# exactly one of the two paper-fidelity RMCT training targets.
#
# This is deliberately a non-production bootstrap harness. It performs one
# real LoRA update, publishes the nonzero v2 adapter to every vLLM worker, and
# compares the fixed teacher-forced policy-minus-base effect with HF/PEFT. A
# normal RMCT run cannot start its Qwen3.5 rollout workers without the sidecar
# this script writes at its own default status path.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: CUDA_VISIBLE_DEVICES=<eight-GPU-allocation> [CTM_PYTHON=/path/to/python] \
  bash infra/vastai/preflight_qwen35_rmct_rollout_workers.sh <rmct-main|rmct-control> [--dry-run]

The command must use the same eight-GPU allocation and checkout that will run
the named target. It writes immutable evidence below that target's normal
logs/<experiment>/<run>/rollout_workers/ directory; do not pass a custom
--local-rollout-status-dir to the subsequent training command.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

target="$1"
dry_run=false
if [[ $# -eq 2 ]]; then
    if [[ "$2" != "--dry-run" ]]; then
        echo "ERROR: the only optional argument is --dry-run; got $2" >&2
        usage >&2
        exit 2
    fi
    dry_run=true
fi
case "$target" in
    rmct-main)
        run_name="rate-matching-lr-1e-4"
        ;;
    rmct-control)
        run_name="rate-matching-control-lr-1e-4"
        ;;
    *)
        echo "ERROR: target must be rmct-main or rmct-control; got $target" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
experiment_name="rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803"
status_dir="$repo_root/logs/$experiment_name/$run_name/rollout_workers"
attestation="$status_dir/qwen35-rollout-worker-parity-attestation.json"

# The matching YAML uses local_device=cuda:0 and local_rollout_gpus=1..7.
# Keep these worker settings in one place with its post-run validation below.
worker_gpus="1,2,3,4,5,6,7"
worker_gpu_mem_util="0.75"
worker_max_num_batched_tokens="8192"
worker_max_model_len="32768"
worker_max_num_seqs="256"
target_logprob_chunk_size="2048"

if [[ "$dry_run" == true ]]; then
    printf '%s\n' \
        "QWEN35_RMCT_WORKER_PREFLIGHT_DRY_RUN=1" \
        "target=$target" \
        "experiment_name=$experiment_name" \
        "run_name=$run_name" \
        "status_dir=$status_dir" \
        "attestation=$attestation" \
        "coordinator_logical_gpu=0" \
        "rollout_worker_logical_gpus=$worker_gpus" \
        "worker_gpu_memory_utilization=$worker_gpu_mem_util" \
        "worker_max_model_len=$worker_max_model_len" \
        "worker_max_num_seqs=$worker_max_num_seqs" \
        "worker_max_num_batched_tokens=$worker_max_num_batched_tokens" \
        "target_logprob_chunk_size=$target_logprob_chunk_size" \
        "probe=3_prompts_x_1_rollout_x_32_generated_tokens"
    exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES to the coordinator plus seven rollout GPUs." >&2
    exit 2
fi

if [[ -e "$attestation" ]]; then
    echo "ERROR: refusing to replace existing worker-parity evidence: $attestation" >&2
    echo "Use the existing attestation for the matching run, or archive the entire failed run and use a new experiment identity." >&2
    exit 2
fi
if [[ -e "$status_dir/adapters" ]]; then
    echo "ERROR: $status_dir already has production adapter snapshots; refusing a preflight for a non-empty run." >&2
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

mkdir -p "$status_dir"
log_path="$status_dir/qwen35-rollout-worker-parity-preflight.log"

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

echo "Qwen3.5 RMCT rollout-worker preflight target=$target" | tee "$log_path"
echo "status_dir=$status_dir" | tee -a "$log_path"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES" | tee -a "$log_path"

# Fourteen same-prompt rollouts first check two matched RNG lanes across all
# seven workers. The harness then holds a selected completion fixed and
# duplicates it across every engine for the LoRA transport check.
"$python_bin" infra/vastai/benchmark_qwen35_opct_group.py \
    --model Qwen/Qwen3.5-9B \
    --worker-gpus "$worker_gpus" \
    --output-dir "$status_dir" \
    --prompt-count 1 \
    --rollouts-per-prompt 14 \
    --max-new-tokens 32 \
    --temperature 0.7 \
    --ignore-eos \
    --target-logprob-chunk-size "$target_logprob_chunk_size" \
    --worker-gpu-mem-util "$worker_gpu_mem_util" \
    --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
    --worker-seed-base 42 \
    --post-update-min-effect 1e-5 2>&1 | tee -a "$log_path"

# Do not trust success solely because the non-production benchmark exited 0:
# prove that the sidecar has the exact normal-training worker topology and
# options that the paper-fidelity YAML resolves.
STATUS_DIR="$status_dir" \
WORKER_GPUS="$worker_gpus" \
WORKER_GPU_MEM_UTIL="$worker_gpu_mem_util" \
WORKER_MAX_NUM_BATCHED_TOKENS="$worker_max_num_batched_tokens" \
WORKER_MAX_MODEL_LEN="$worker_max_model_len" \
WORKER_MAX_NUM_SEQS="$worker_max_num_seqs" \
"$python_bin" - 2>&1 <<'PY' | tee -a "$log_path"
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
    "seed": 42,
}
path = Path(os.environ["STATUS_DIR"]) / WORKER_PARITY_ATTESTATION_NAME
document = validate_qwen35_rollout_worker_parity_attestation(
    path,
    expected_model="Qwen/Qwen3.5-9B",
    expected_worker_gpus=workers,
    expected_worker_engine_kwargs=engine_kwargs,
)
print("QWEN35_RMCT_WORKER_PARITY_ATTESTATION=" + str(path.resolve()))
print("QWEN35_RMCT_WORKER_PARITY_SHA256=" + file_sha256(path))
print("QWEN35_RMCT_WORKER_COUNT=" + str(len(document["worker_gpus"])))
PY
