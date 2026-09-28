#!/usr/bin/env bash
# Run the generic, non-production Qwen3.5 phase-shared preflight on one
# Isambard-AI GH200 node. Submit with an explicit durable checkout:
#
#   sbatch --export=ALL,REPO_DIR=/project/path/consistency-training-methods \
#     infra/isambard/preflight_qwen35_phase_shared.sbatch
#
# The generic preflight owns the evidence directory and refuses to reuse it.
# This wrapper only prepares the allocated node and keeps launcher records in
# the adjacent, fresh run root.
#SBATCH --job-name=ctm-qwen35-phase-shared-preflight
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-qwen35-phase-shared-preflight-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" ]]; then
    echo "ERROR: submit with an explicit durable REPO_DIR checkout path." >&2
    exit 2
fi
if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR does not exist: $REPO_DIR" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"

if [[ -z "${SLURM_JOB_ID:-}" || ! "$SLURM_JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: this wrapper must run as a Slurm batch job with a numeric SLURM_JOB_ID." >&2
    exit 2
fi
if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR must name an existing shared scratch directory." >&2
    exit 2
fi

export CTM_PYTHON="$repo_root/.venv/bin/python"
if [[ ! -x "$CTM_PYTHON" ]]; then
    echo "ERROR: missing GPU-validated repository virtual environment: $CTM_PYTHON" >&2
    echo "Run infra/isambard/setup_gpu_env.sh in an allocated GH200 session first." >&2
    exit 2
fi

preflight_wrapper="$repo_root/infra/vastai/preflight_qwen35_phase_shared.sh"
if [[ ! -x "$preflight_wrapper" ]]; then
    echo "ERROR: generic phase-shared preflight wrapper is not executable: $preflight_wrapper" >&2
    exit 2
fi
setup_script="$repo_root/infra/isambard/setup_gpu_env.sh"
if [[ ! -x "$setup_script" ]]; then
    echo "ERROR: Isambard GPU environment setup script is not executable: $setup_script" >&2
    exit 2
fi

# Persist reusable downloads on scratch, never beside transient node-local
# caches. Deliberately override inherited cache homes so this job has one
# reproducible cache location.
scratch_root="$(cd -- "$SCRATCHDIR" && pwd -P)"
export HF_HOME="$scratch_root/ctm/huggingface"
export UV_CACHE_DIR="$scratch_root/ctm/uv-cache"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
if [[ ! -w "$HF_HOME" || ! -w "$UV_CACHE_DIR" ]]; then
    echo "ERROR: scratch cache directories are not writable." >&2
    exit 2
fi

# Isambard nodes can inherit absent local cache paths. Make a newly-created,
# private namespace for this job instead of sharing state with another job.
ctm_job_tmp="$(mktemp -d "/tmp/ctm-phase-shared-${USER:-user}-${SLURM_JOB_ID}-XXXXXX")"
mkdir -p \
    "$ctm_job_tmp/tmp" \
    "$ctm_job_tmp/xdg-cache" \
    "$ctm_job_tmp/torchinductor" \
    "$ctm_job_tmp/triton" \
    "$ctm_job_tmp/cuda-cache"
chmod 700 \
    "$ctm_job_tmp" \
    "$ctm_job_tmp/tmp" \
    "$ctm_job_tmp/xdg-cache" \
    "$ctm_job_tmp/torchinductor" \
    "$ctm_job_tmp/triton" \
    "$ctm_job_tmp/cuda-cache"
export TMPDIR="$ctm_job_tmp/tmp"
export XDG_CACHE_HOME="$ctm_job_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$ctm_job_tmp/torchinductor"
export TRITON_CACHE_DIR="$ctm_job_tmp/triton"
export CUDA_CACHE_PATH="$ctm_job_tmp/cuda-cache"

# Qwen3.5 is BF16 here. On the Isambard GH200 vLLM build, disable the
# unavailable DeepGEMM warmup route before either the health probe or vLLM
# imports run.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

# Keep all evidence, the immutable launch receipt, and the streamed task log
# in a fresh durable directory under the explicit project checkout. The
# generic harness receives a child that does not exist yet and will reject any
# unexpected reuse itself.
durable_parent="$repo_root/artifacts/qwen35-phase-shared-preflight"
mkdir -p "$durable_parent"
run_root="$(mktemp -d "$durable_parent/isambard-${SLURM_JOB_ID}-XXXXXX")"
chmod 700 "$run_root"
output_dir="$run_root/evidence"
metadata_path="$run_root/launch-metadata.json"
log_path="$run_root/preflight.log"
if [[ -e "$output_dir" || -e "$metadata_path" || -e "$log_path" ]]; then
    echo "ERROR: fresh Isambard preflight run paths unexpectedly already exist: $run_root" >&2
    exit 2
fi

# Record only the reproducibility paths and Slurm identity, not the inherited
# environment. The temporary file plus rename makes a partially written
# receipt impossible to mistake for a completed one.
"$CTM_PYTHON" - "$metadata_path" "$repo_root" "$run_root" "$output_dir" "$log_path" "$SLURM_JOB_ID" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
repository_root, run_root, evidence_dir, log_path, job_id = sys.argv[2:]
payload = {
    "schema": "qwen35-isambard-phase-shared-launch-v1",
    "slurm_job_id": job_id,
    "repository_root": repository_root,
    "run_root": run_root,
    "evidence_dir": evidence_dir,
    "log_path": log_path,
    "preflight_wrapper": str(Path(repository_root) / "infra/vastai/preflight_qwen35_phase_shared.sh"),
}
temporary = metadata_path.with_name(f".{metadata_path.name}.partial-{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, metadata_path)
PY

printf 'qwen35 phase-shared preflight job=%s evidence=%s\n' "$SLURM_JOB_ID" "$output_dir" > "$log_path"
cd "$repo_root"

# Do not assign CUDA_VISIBLE_DEVICES here: Slurm owns that mapping. One task
# receives the complete four-GPU step and checks the inherited allocation
# before handing it unchanged to the generic wrapper.
if srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash -c '
        set -euo pipefail
        setup_script=$1
        preflight_wrapper=$2
        shift 2
        if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
            echo "ERROR: Slurm did not provide a usable CUDA_VISIBLE_DEVICES allocation to the four-GPU task." >&2
            exit 2
        fi
        IFS=, read -r -a ctm_visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
        if [[ "${#ctm_visible_gpus[@]}" -ne 4 ]]; then
            echo "ERROR: expected exactly four Slurm-visible GPUs in the task, got: $CUDA_VISIBLE_DEVICES" >&2
            exit 2
        fi
        for ctm_visible_gpu in "${ctm_visible_gpus[@]}"; do
            if [[ -z "$ctm_visible_gpu" ]]; then
                echo "ERROR: Slurm supplied an empty CUDA-visible GPU token." >&2
                exit 2
            fi
        done
        if ! command -v flock >/dev/null 2>&1; then
            echo "ERROR: flock is required to serialize repository virtual-environment setup." >&2
            exit 2
        fi
        repo_root="$(cd -- "$(dirname -- "$setup_script")/../.." && pwd -P)"
        venv_lock="$repo_root/.venv/.ctm-isambard-phase-shared-setup.lock"
        # Keep this descriptor open across exec so another job cannot upgrade
        # this shared repository virtual environment during the preflight.
        exec 9>"$venv_lock"
        flock -x 9
        bash "$setup_script"
        exec "$preflight_wrapper" "$@"
    ' bash \
    "$setup_script" \
    "$preflight_wrapper" \
    --output-dir "$output_dir" \
    --training-gpus all \
    --rollout-gpus all \
    --topology-contract-sizes 2 4 8 \
    --worker-gdn-prefill-backend triton \
    --worker-gpu-mem-util 0.35 \
    --start-timeout-seconds 1800 \
    --request-timeout-seconds 7200 \
    --shutdown-timeout-seconds 60 \
    --packing-budgets 20480 40960 49152 \
    2>&1 | tee -a "$log_path"; then
    pipeline_status=(0 0)
else
    pipeline_status=("${PIPESTATUS[@]}")
fi

if (( pipeline_status[0] != 0 )); then
    echo "ERROR: Isambard phase-shared preflight task failed; inspect $log_path and $output_dir." >&2
    exit "${pipeline_status[0]}"
fi
if (( pipeline_status[1] != 0 )); then
    echo "ERROR: unable to persist the Isambard phase-shared preflight log: $log_path" >&2
    exit "${pipeline_status[1]}"
fi
