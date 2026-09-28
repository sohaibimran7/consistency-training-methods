#!/usr/bin/env bash
# One static 16-update RMCT-256 convergence segment on one Isambard-AI Phase 2
# node: four GH200s (one coordinator and three rollout workers). The 16-hour
# budget leaves material margin below the queue's 24-hour limit; continuation
# is a new static Slurm job with a final-boundary receipt, never a requeue.
#
# Example:
#   sbatch --export=ALL,REPO_DIR=/project/.../consistency-training-methods \
#     infra/isambard/run_qwen35_rmct256_convergence_segment.sbatch \
#     --segment-index 0 --yes
#SBATCH --job-name=ctm-rmct256-seg
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=16:00:00
#SBATCH --output=slurm-rmct256-convergence-%j.out

set -euo pipefail

if [[ -n "${REPO_DIR:-}" ]]; then
    repo_root="$REPO_DIR"
elif [[ -n "${PROJECTDIR:-}" ]]; then
    repo_root="$PROJECTDIR/$USER/consistency-training-methods"
else
    echo "ERROR: set REPO_DIR to the durable Isambard checkout before submitting." >&2
    exit 2
fi
if [[ ! -d "$repo_root" ]]; then
    echo "ERROR: REPO_DIR does not exist: $repo_root" >&2
    exit 2
fi
if [[ ! -x "$repo_root/.venv/bin/python" ]]; then
    echo "ERROR: missing GPU-validated Isambard virtual environment: $repo_root/.venv/bin/python" >&2
    echo "Run infra/isambard/setup_gpu_env.sh in an allocated GH200 session first." >&2
    exit 2
fi

cd "$repo_root"

# Torch, Triton and Dynamo may inherit a nonexistent /local path. Keep cache
# writes job-private and node-local; these are disposable caches, never model,
# proof, checkpoint or receipt paths.
ctm_job_tmp="/tmp/ctm-$USER-$SLURM_JOB_ID"
mkdir -p \
    "$ctm_job_tmp/tmp" \
    "$ctm_job_tmp/xdg-cache" \
    "$ctm_job_tmp/torchinductor" \
    "$ctm_job_tmp/triton" \
    "$ctm_job_tmp/cuda-cache"
chmod 700 "$ctm_job_tmp"
export TMPDIR="$ctm_job_tmp/tmp"
export XDG_CACHE_HOME="$ctm_job_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$ctm_job_tmp/torchinductor"
export TRITON_CACHE_DIR="$ctm_job_tmp/triton"
export CUDA_CACHE_PATH="$ctm_job_tmp/cuda-cache"

if [[ -z "${SCRATCHDIR:-}" && -n "${SCRATCH:-}" ]]; then
    export SCRATCHDIR="$SCRATCH"
fi
if [[ -n "${SCRATCHDIR:-}" && -z "${HF_HOME:-}" ]]; then
    export HF_HOME="$SCRATCHDIR/ctm/huggingface"
fi
if [[ -n "${HF_HOME:-}" ]]; then
    mkdir -p "$HF_HOME"
fi
export CTM_PYTHON="$repo_root/.venv/bin/python"

# One Slurm step retains the exact four-device allocation through the offline
# base-model gate, source proof, worker preflight, target attestation, runner,
# receipt sealer and metric extractor. No --gpus option reaches the sequential
# runner, which must inherit this allocation unchanged.
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash "$repo_root/infra/isambard/run_qwen35_rmct256_convergence_segment.sh" "$@"
