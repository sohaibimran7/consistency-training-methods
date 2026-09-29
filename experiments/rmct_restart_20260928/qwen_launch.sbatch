#!/usr/bin/env bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
umask 077
# Explicit absolute canonical repository and environment; no legacy defaults.
repo=${1:?canonical repository required}
py=${2:?canonical environment Python required}
shift 2
cd "$repo"
export PYTHONPATH="$repo" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 "$py" \
  "$repo/experiments/rmct_restart_20260928/qwen_launch.py" "$@"
# No automatic segment chaining or evaluation submission.
