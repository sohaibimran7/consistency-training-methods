#!/usr/bin/env bash
# One fresh one-bias Qwen RMCT window (at most 64 sampled batches = 256 encounters).
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
umask 077
repo=${1:?canonical repository}
py=${2:?canonical Python}
shift 2
cd "$repo"
export PYTHONPATH="$repo" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$py"):$PATH"
unset PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 "$py" -m experiments.rmct_restart_20260928.qwen_one_bias_rmct "$@"
# Validation of this window is a separate submission; never self-advance.
