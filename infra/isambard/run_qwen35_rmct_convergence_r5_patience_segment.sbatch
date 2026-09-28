#!/usr/bin/env bash
# Run one legacy-capped window, scheduling another only after a continue receipt.
#SBATCH --job-name=ctm-rmct-patience20k
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct-convergence-r5-patience-%j.out

set -euo pipefail
umask 077

ctm_r5_repo="${REPO_DIR:?REPO_DIR is required}"
ctm_r5_scratch="${SCRATCHDIR:?SCRATCHDIR is required}"
ctm_r5_segment="${CTM_RMCT_SEGMENT_INDEX:?CTM_RMCT_SEGMENT_INDEX is required}"
if [[ ! "$ctm_r5_segment" =~ ^[0-9]+$ ]] || (( 10#$ctm_r5_segment < 11 )); then
    echo "ERROR: CTM_RMCT_SEGMENT_INDEX must be at least 11" >&2
    exit 2
fi
cd "$ctm_r5_repo"
ctm_r5_repo="$(pwd -P)"
ctm_r5_python="${CTM_RMCT_PYTHON:-$ctm_r5_repo/.venv/bin/python}"
ctm_r5_amendment="$ctm_r5_repo/artifacts/rmct-capped-patience-20260910/amendment.json"
ctm_r5_runner="$ctm_r5_repo/infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.py"
ctm_r5_parity="$ctm_r5_repo/artifacts/rmct-convergence-worker-parity-20260813"
ctm_r5_parity_helper="$ctm_r5_repo/infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
for ctm_r5_required in "$ctm_r5_python" "$ctm_r5_runner" "$ctm_r5_amendment"; do
    if [[ ! -f "$ctm_r5_required" || -L "$ctm_r5_required" ]]; then
        echo "ERROR: required r5 input is missing or linked: $ctm_r5_required" >&2
        exit 2
    fi
done

export HF_HOME="$ctm_r5_scratch/ctm/huggingface"
export UV_CACHE_DIR="$ctm_r5_scratch/ctm/uv-cache"
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH="$ctm_r5_repo${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

ctm_r5_tmp="$(mktemp -d "/tmp/ctm-rmct-r5-s${ctm_r5_segment}-${USER:-user}-${SLURM_JOB_ID:-manual}-XXXXXX")"
mkdir -p "$ctm_r5_tmp/tmp" "$ctm_r5_tmp/xdg" "$ctm_r5_tmp/torchinductor" "$ctm_r5_tmp/triton" "$ctm_r5_tmp/cuda"
chmod 700 "$ctm_r5_tmp" "$ctm_r5_tmp/tmp" "$ctm_r5_tmp/xdg" "$ctm_r5_tmp/torchinductor" "$ctm_r5_tmp/triton" "$ctm_r5_tmp/cuda"
export TMPDIR="$ctm_r5_tmp/tmp"
export XDG_CACHE_HOME="$ctm_r5_tmp/xdg"
export TORCHINDUCTOR_CACHE_DIR="$ctm_r5_tmp/torchinductor"
export TRITON_CACHE_DIR="$ctm_r5_tmp/triton"
export CUDA_CACHE_PATH="$ctm_r5_tmp/cuda"

"$ctm_r5_python" -m experiments.rmct_convergence.patience verify-amendment \
    --repository "$ctm_r5_repo" --amendment "$ctm_r5_amendment" >/dev/null

srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash -c '
        set -euo pipefail
        ctm_r5_repo=$1
        ctm_r5_python=$2
        ctm_r5_segment=$3
        ctm_r5_amendment=$4
        ctm_r5_parity=$5
        ctm_r5_parity_helper=$6
        ctm_r5_snapshot="$HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        "$ctm_r5_python" "$ctm_r5_parity_helper" --output-dir "$ctm_r5_parity" \
            --model-snapshot "$ctm_r5_snapshot" --resume
        exec "$ctm_r5_python" "$ctm_r5_repo/infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.py" \
            --repository "$ctm_r5_repo" --segment-index "$ctm_r5_segment" \
            --amendment "$ctm_r5_amendment" --yes
    ' bash "$ctm_r5_repo" "$ctm_r5_python" "$ctm_r5_segment" "$ctm_r5_amendment" "$ctm_r5_parity" "$ctm_r5_parity_helper"

# Replays the sealed decision; submits nothing once patience is exhausted.
bash "$ctm_r5_repo/infra/isambard/submit_qwen35_rmct_convergence_r5_patience_chain.sh" --yes
