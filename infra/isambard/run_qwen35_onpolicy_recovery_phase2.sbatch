#!/usr/bin/env bash
# Submit one fail-closed Qwen3.5 on-policy target on a single Isambard-AI
# Phase 2 node: four GH200s, one coordinator, and three rollout workers.
#
# Example:
#   sbatch --export=ALL,REPO_DIR=/project/path/consistency-training-methods \
#     infra/isambard/run_qwen35_onpolicy_recovery_phase2.sbatch opct
#
# ``workq`` permits at most 24 hours. A preflight-only submission can override
# this downward (for example ``sbatch --time=01:00:00 ... --preflight-only``).
# RMCT is expected to be substantially slower than the seven-worker Vast
# topology, so a production target may need a continuation design if it cannot
# complete within that queue limit.
#SBATCH --job-name=ctm-qwen35-onpolicy
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-qwen35-onpolicy-%j.out

set -euo pipefail

if [ -n "$(printenv REPO_DIR || true)" ]; then
    repo_root=$REPO_DIR
elif [ -n "$(printenv PROJECTDIR || true)" ]; then
    repo_root=$PROJECTDIR/$USER/consistency-training-methods
else
    echo "ERROR: set REPO_DIR to the durable Isambard checkout before submitting." >&2
    exit 2
fi
if [ ! -d "$repo_root" ]; then
    echo "ERROR: REPO_DIR does not exist: $repo_root" >&2
    exit 2
fi
if [ ! -x "$repo_root/.venv/bin/python" ]; then
    echo "ERROR: missing GPU-validated Isambard virtual environment: $repo_root/.venv/bin/python" >&2
    echo "Run infra/isambard/setup_gpu_env.sh in an allocated GH200 session first." >&2
    exit 2
fi

cd "$repo_root"

# Some Phase 2 jobs inherit TMPDIR/XDG paths under /local/user/<uid> even
# when that directory is absent or not writable on the allocated node. Torch
# creates its Dynamo/Inductor caches during ordinary Transformers imports, so
# fail closed onto a private node-local cache namespace before importing it.
ctm_job_tmp=/tmp/ctm-$USER-$SLURM_JOB_ID
mkdir -p \
    "$ctm_job_tmp/tmp" \
    "$ctm_job_tmp/xdg-cache" \
    "$ctm_job_tmp/torchinductor" \
    "$ctm_job_tmp/triton" \
    "$ctm_job_tmp/cuda-cache"
chmod 700 "$ctm_job_tmp"
export TMPDIR=$ctm_job_tmp/tmp
export XDG_CACHE_HOME=$ctm_job_tmp/xdg-cache
export TORCHINDUCTOR_CACHE_DIR=$ctm_job_tmp/torchinductor
export TRITON_CACHE_DIR=$ctm_job_tmp/triton
export CUDA_CACHE_PATH=$ctm_job_tmp/cuda-cache

if [ -n "$(printenv SCRATCHDIR || true)" ] && [ -z "$(printenv HF_HOME || true)" ]; then
    export HF_HOME=$SCRATCHDIR/ctm/huggingface
fi
if [ -n "$(printenv HF_HOME || true)" ]; then
    mkdir -p "$HF_HOME"
fi
export CTM_PYTHON=$repo_root/.venv/bin/python

# A single Slurm step owns all four GPUs, preserving the allocation into the
# source proof, worker parity bootstrap, target attestation, and training child.
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash "$repo_root/infra/isambard/run_qwen35_onpolicy_recovery_phase2.sh" "$@"
