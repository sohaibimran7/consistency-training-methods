#!/usr/bin/env bash
# RLCT training job on Isambard-AI (one GH200 node, LocalBackend).
#
# Usage (extra args are forwarded to scripts/train_rlct.py):
#   sbatch infra/isambard/train_rlct.sbatch \
#       --model openai/gpt-oss-20b \
#       --setting-factory ctm_data.adapters.mcq_bias:create_setting \
#       --setting-config '{"data_paths":["/path/to/train.jsonl"]}' \
#       --n-datapoints 64 \
#       --experiment-name rl_wfs_local --run-name gh200-rlct-wfs
# This job is non-interactive and supplies -y below. Run the identical command
# with scripts/train_rlct.py --dry-run and obtain approval before submitting.
#
# EDIT-ME before first use: account, partition, time, and REPO_DIR.
#SBATCH --job-name=rlct-local
#SBATCH --account=EDIT-ME            # your Isambard project code
#SBATCH --partition=EDIT-ME          # GH200 partition on your allocation
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/consistency-training-methods}"
cd "$REPO_DIR"

# Isambard-AI is module-based; load your verified CUDA stack here if the venv's
# torch needs it (often unnecessary with pip wheels — they bundle CUDA libs).
# module load ...

source .venv/bin/activate
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"   # keep model downloads on project storage

srun python scripts/train_rlct.py \
    --backend local \
    --local-dtype bfloat16 \
    --local-sampler vllm \
    --local-gpu-mem-util 0.45 \
    -y \
    "$@"
