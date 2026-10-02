#!/usr/bin/env bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
set -euo pipefail
umask 077
repo=${1:?canonical repository}
py=${2:?canonical Python}
folder=${3:?fresh one-bias validation output}
manifest=${4:?frozen validation manifest}
shift 4
cd "$repo"
export PYTHONPATH="$repo" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$py"):$PATH"
unset PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
srun --nodes=1 --ntasks=1 --gpus-per-task=1 "$py" -m experiments.rmct_restart_20260928.qwen_one_bias_validation prepare \
  --folder "$folder" --manifest "$manifest" "$@"
srun --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=16 \
  bash -c 'exec "$1" -m experiments.rmct_restart_20260928.qwen_validation_server --folder "$2" --manifest "$3" --rank "$SLURM_PROCID" --port "$((18789 + SLURM_PROCID))"' _ "$py" "$folder" "$manifest"
# Score externally only AFTER this job is COMPLETED0:0. Never self-advance.
