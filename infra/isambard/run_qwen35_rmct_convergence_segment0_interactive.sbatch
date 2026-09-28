#!/usr/bin/env bash
# Allocate one four-GH200 Isambard node for the first sealed RMCT-convergence
# window.  This file intentionally never waits for source synchronization: use
# submit_qwen35_rmct_convergence_segment0_when_ready.sh on a login node when
# the immutable source-ready sentinel has not appeared yet.
#
# Once allocated, this job performs exactly one srun step, with this order:
#   source-ready identity -> offline snapshot -> worker parity -> readiness
#   capture/verify -> launcher segment 0.
# It never invokes training before all of those gates pass.
#SBATCH --job-name=ctm-rmct-convergence-s000
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-rmct-convergence-s000-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" ]]; then
    echo "ERROR: submit with REPO_DIR pointing to the durable production checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
source_ready="$repo_root/.rmct-convergence-production-ready"
parity_dir="$repo_root/artifacts/rmct-convergence-worker-parity-20260813"
parity_helper="$repo_root/infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
verify_script="$repo_root/infra/isambard/verify_rmct_convergence_production_ready.py"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_deadline.sh"
ready_receipt="$repo_root/artifacts/rmct-convergence-production-ready.json"

if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
    echo "ERROR: missing GPU-validated Isambard Python: $repo_root/.venv/bin/python" >&2
    exit 2
fi
export CTM_PYTHON="$repo_root/.venv/bin/python"
for required in "$parity_helper" "$verify_script" "$launcher"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required protected production helper is missing or linked: $required" >&2
        exit 2
    fi
done
if [[ ! -x "$launcher" ]]; then
    echo "ERROR: production launcher is not executable: $launcher" >&2
    exit 2
fi
if [[ ! -f "$source_ready" || -L "$source_ready" ]]; then
    echo "ERROR: immutable source-ready sentinel is absent: $source_ready" >&2
    echo "Submit via submit_qwen35_rmct_convergence_segment0_when_ready.sh before allocation." >&2
    exit 2
fi
"$CTM_PYTHON" "$verify_script" verify-source-ready --repository "$repo_root" --receipt "$source_ready"

source_ready_sha="$($CTM_PYTHON - "$source_ready" <<PY
import hashlib
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"
if [[ -n "${CTM_RMCT_SOURCE_READY_SHA256:-}" && "$CTM_RMCT_SOURCE_READY_SHA256" != "$source_ready_sha" ]]; then
    echo "ERROR: source-ready sentinel changed after login-node submission; refuse allocated execution" >&2
    exit 2
fi
export CTM_RMCT_SOURCE_READY_SHA256="$source_ready_sha"

if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR must name the durable shared scratch directory" >&2
    exit 2
fi
scratch_root="$(cd -- "$SCRATCHDIR" && pwd -P)"
export HF_HOME="$scratch_root/ctm/huggingface"
export UV_CACHE_DIR="$scratch_root/ctm/uv-cache"
export UV_LINK_MODE=copy
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
if [[ ! -w "$HF_HOME" || ! -w "$UV_CACHE_DIR" ]]; then
    echo "ERROR: durable Hugging Face or uv cache is not writable" >&2
    exit 2
fi

# Node-local cache paths are deliberately job-private and disposable.  They do
# not contain source, model snapshots, checkpoints, receipts, or attestations.
ctm_job_tmp="$(mktemp -d "/tmp/ctm-rmct-convergence-${USER:-user}-${SLURM_JOB_ID:-manual}-XXXXXX")"
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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

# A single four-GPU Slurm step retains the allocation unchanged through every
# proof and segment-0 execution phase.  No later segment can be reached here.
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash -c '
        set -euo pipefail
        repo_root=$1
        expected_source_ready_sha=$2
        python_bin=$3
        source_ready="$repo_root/.rmct-convergence-production-ready"
        parity_dir="$repo_root/artifacts/rmct-convergence-worker-parity-20260813"
        parity_helper="$repo_root/infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
        verify_script="$repo_root/infra/isambard/verify_rmct_convergence_production_ready.py"
        launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_deadline.sh"
        ready_receipt="$repo_root/artifacts/rmct-convergence-production-ready.json"

        if [[ ! -f "$source_ready" || -L "$source_ready" ]]; then
            echo "ERROR: source-ready sentinel disappeared before the four-GPU step" >&2
            exit 2
        fi
        "$python_bin" "$verify_script" verify-source-ready --repository "$repo_root" --receipt "$source_ready"
        actual_source_ready_sha="$($python_bin - "$source_ready" <<PY
import hashlib
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"
        if [[ "$actual_source_ready_sha" != "$expected_source_ready_sha" ]]; then
            echo "ERROR: source-ready sentinel changed before the four-GPU step" >&2
            exit 2
        fi
        $python_bin - <<PY
import os

tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
if len(tokens) != 4 or any(not value or value in {"-1", "NoDevFiles"} for value in tokens) or len(set(tokens)) != 4:
    raise SystemExit("error: segment 0 requires exactly four distinct Slurm-visible GPUs")
if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
    raise SystemExit("error: offline model resolution must be forced before worker parity")
PY
        model_snapshot="$($python_bin - <<PY
import os
from pathlib import Path

revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
home = os.environ.get("HF_HOME")
if not home:
    raise SystemExit("error: HF_HOME is required for the pinned offline snapshot")
snapshot = Path(home).resolve() / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / revision
if snapshot.is_symlink() or not snapshot.is_dir() or not (snapshot / "config.json").is_file():
    raise SystemExit(f"error: pinned offline Qwen snapshot is absent/incomplete: {snapshot}")
print(snapshot)
PY
)"

        # The helper creates a fresh global sidecar once.  Thereafter --resume
        # only validates the immutable attestation and SUCCESS marker.
        if [[ -e "$parity_dir" || -L "$parity_dir" ]]; then
            $python_bin "$parity_helper" --output-dir "$parity_dir" --model-snapshot "$model_snapshot" --resume
        else
            $python_bin "$parity_helper" --output-dir "$parity_dir" --model-snapshot "$model_snapshot"
        fi

        # Readiness binds the exact source tree, plan, immutable shared data,
        # pinned snapshot, environment, controller, and parity-helper source.
        $python_bin "$verify_script" capture --repository "$repo_root" --output "$ready_receipt"
        $python_bin "$verify_script" verify --repository "$repo_root" --receipt "$ready_receipt"
        exec "$launcher" --ready-receipt "$ready_receipt" --segment-index 0 --yes
    ' bash "$repo_root" "$CTM_RMCT_SOURCE_READY_SHA256" "$CTM_PYTHON"
