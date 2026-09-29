#!/usr/bin/env bash
# Dependency-safe one-window executor for the fresh r4 RMCT recovery.
# Submit only with --dependency=afterok:<sealed-predecessor-job>.  r4 begins
# at logical segment 4 after r3 s004; failed job 6031786 is never a state
# parent.  This job never calls sbatch and cannot choose an implicit index.
#SBATCH --job-name=ctm-rmct-convergence-r4-recovery
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct-convergence-r4-recovery-%j.out

set -euo pipefail
umask 077

segment_index="${CTM_RMCT_SEGMENT_INDEX:-}"
if [[ ! "$segment_index" =~ ^[0-9]+$ ]] || (( 10#$segment_index < 4 || 10#$segment_index > 31 )); then
    echo "ERROR: set CTM_RMCT_SEGMENT_INDEX to exactly one r4 successor index in [4, 31]" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" ]]; then
    echo "ERROR: submit with REPO_DIR pointing to the durable r4 recovery checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
python_bin="$repo_root/.venv/bin/python"
source_ready="$repo_root/.rmct-convergence-gcall-r2-mb40960-r4-production-source-ready"
ready_receipt="$repo_root/artifacts/rmct-convergence-gcall-r2-mb40960-r4-production-ready.json"
parity_dir="$repo_root/artifacts/rmct-convergence-worker-parity-20260813"
parity_helper="$repo_root/infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
verify_script="$repo_root/infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_r4_recovery_deadline.sh"

if [[ ! -x "$python_bin" || ! -x "$launcher" ]]; then
    echo "ERROR: r4 recovery Python or launcher is unavailable/executable" >&2
    exit 2
fi
for required in "$parity_helper" "$verify_script"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required r4 recovery helper is missing or linked: $required" >&2
        exit 2
    fi
done
if [[ ! -f "$source_ready" || -L "$source_ready" ]]; then
    echo "ERROR: r4 recovery source-ready receipt is absent" >&2
    exit 2
fi
if [[ ! -d "$parity_dir" || -L "$parity_dir" ]]; then
    echo "ERROR: inherited immutable rollout worker-parity sidecar is absent" >&2
    exit 2
fi
"$python_bin" "$verify_script" verify-source-ready --repository "$repo_root" --receipt "$source_ready"
source_ready_sha="$("$python_bin" - "$source_ready" <<'PY'
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

if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR must name durable shared scratch" >&2
    exit 2
fi
scratch_root="$(cd -- "$SCRATCHDIR" && pwd -P)"
export CTM_PYTHON="$python_bin"
export HF_HOME="$scratch_root/ctm/huggingface"
export UV_CACHE_DIR="$scratch_root/ctm/uv-cache"
export UV_LINK_MODE=copy
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
if [[ ! -w "$HF_HOME" || ! -w "$UV_CACHE_DIR" ]]; then
    echo "ERROR: durable Hugging Face or uv cache is not writable" >&2
    exit 2
fi

ctm_job_tmp="$(mktemp -d "/tmp/ctm-rmct-convergence-r4-s${segment_index}-${USER:-user}-${SLURM_JOB_ID:-manual}-XXXXXX")"
mkdir -p "$ctm_job_tmp/tmp" "$ctm_job_tmp/xdg-cache" "$ctm_job_tmp/torchinductor" "$ctm_job_tmp/triton" "$ctm_job_tmp/cuda-cache"
chmod 700 "$ctm_job_tmp" "$ctm_job_tmp/tmp" "$ctm_job_tmp/xdg-cache" "$ctm_job_tmp/torchinductor" "$ctm_job_tmp/triton" "$ctm_job_tmp/cuda-cache"
export TMPDIR="$ctm_job_tmp/tmp"
export XDG_CACHE_HOME="$ctm_job_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$ctm_job_tmp/torchinductor"
export TRITON_CACHE_DIR="$ctm_job_tmp/triton"
export CUDA_CACHE_PATH="$ctm_job_tmp/cuda-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash -c '
        set -euo pipefail
        repo_root=$1
        expected_source_ready_sha=$2
        python_bin=$3
        segment_index=$4
        source_ready="$repo_root/.rmct-convergence-gcall-r2-mb40960-r4-production-source-ready"
        ready_receipt="$repo_root/artifacts/rmct-convergence-gcall-r2-mb40960-r4-production-ready.json"
        parity_dir="$repo_root/artifacts/rmct-convergence-worker-parity-20260813"
        parity_helper="$repo_root/infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
        verify_script="$repo_root/infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py"
        launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_r4_recovery_deadline.sh"

        "$python_bin" "$verify_script" verify-source-ready --repository "$repo_root" --receipt "$source_ready"
        actual_source_ready_sha="$("$python_bin" - "$source_ready" <<PY
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
            echo "ERROR: r4 recovery source-ready receipt changed before GPU execution" >&2
            exit 2
        fi
        "$python_bin" - <<PY
import os

tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
if len(tokens) != 4 or any(not value or value in {"-1", "NoDevFiles"} for value in tokens) or len(set(tokens)) != 4:
    raise SystemExit("error: r4 recovery requires exactly four distinct Slurm-visible GPUs")
PY
        model_snapshot="$("$python_bin" - <<PY
import os
from pathlib import Path

revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise SystemExit("error: HF_HOME is required for the pinned offline snapshot")
snapshot = Path(hf_home).resolve() / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / revision
if snapshot.is_symlink() or not snapshot.is_dir() or not (snapshot / "config.json").is_file():
    raise SystemExit(f"error: pinned offline Qwen snapshot is absent/incomplete: {snapshot}")
print(snapshot)
PY
)"
        if [[ ! -d "$parity_dir" || -L "$parity_dir" ]]; then
            echo "ERROR: immutable rollout worker-parity sidecar is absent" >&2
            exit 2
        fi
        "$python_bin" "$parity_helper" --output-dir "$parity_dir" --model-snapshot "$model_snapshot" --resume
        # Capture is immutable/resumable.  It binds r3 s004, the failed-attempt
        # provenance artifact, and the 40,960-token contract before r4 writes.
        "$python_bin" "$verify_script" capture --repository "$repo_root" --output "$ready_receipt"
        "$python_bin" "$verify_script" verify --repository "$repo_root" --receipt "$ready_receipt"
        exec "$launcher" --ready-receipt "$ready_receipt" --segment-index "$segment_index" --yes
    ' bash "$repo_root" "$source_ready_sha" "$python_bin" "$segment_index"
