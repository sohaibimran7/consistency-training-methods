#!/usr/bin/env bash
# Diagnostic only: no model outputs from this job enter the base evaluation.
# Wall time bounds the diagnostic allocation, not model output length. A
# scheduler-interrupted response must remain incomplete, never EOS-successful.
#SBATCH --job-name=gemma4-eos-debug
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-gemma4-eos-debug-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must identify the isolated Gemma recovery checkout" >&2
    exit 2
fi
cd "$REPO_DIR"
if [[ ! -f infra/isambard/debug_gemma_eos.py ]]; then
    echo "ERROR: Gemma EOS diagnostic source is missing" >&2
    exit 2
fi

export CTM_MUSE_PYTHON="${CTM_GEMMA4_EVAL_PYTHON:-/projects/a5v/sohaib.a5v/consistency-training-methods/.venv-muse-cu129/bin/python}"
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
export CTM_HF_EOS_ONLY_EXPECTED_INSPECT=0.3.260
export CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS=5.15.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA

diagnostic_cache=$(mktemp -d "/tmp/ctm-gemma-eos-j${SLURM_JOB_ID:-manual}-XXXXXX")
mkdir -p "$diagnostic_cache/tmp" "$diagnostic_cache/xdg" "$diagnostic_cache/torch" "$diagnostic_cache/triton" "$diagnostic_cache/cuda"
export TMPDIR="$diagnostic_cache/tmp" TMP="$diagnostic_cache/tmp" TEMP="$diagnostic_cache/tmp"
export XDG_CACHE_HOME="$diagnostic_cache/xdg" TORCHINDUCTOR_CACHE_DIR="$diagnostic_cache/torch"
export TRITON_CACHE_DIR="$diagnostic_cache/triton" CUDA_CACHE_PATH="$diagnostic_cache/cuda"
source infra/isambard/muse_glimmer_cuda129_runtime_env.sh

exec srun --label --exclusive --exact --nodes=1 --ntasks=1 \
    --gpus-per-task=1 --cpus-per-task=16 \
    "$CTM_MUSE_RUNTIME_PYTHON" infra/isambard/debug_gemma_eos.py "$@"
