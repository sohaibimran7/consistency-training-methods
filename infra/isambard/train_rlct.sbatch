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
# The project association and default workq partition come from the Clifton
# project login. Override time or REPO_DIR at submission time when needed.
#SBATCH --job-name=rlct-local
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${PROJECTDIR:-}" ]]; then
        REPO_DIR="$PROJECTDIR/$USER/consistency-training-methods"
    else
        REPO_DIR="$HOME/consistency-training-methods"
    fi
fi
cd "$REPO_DIR"

# Isambard-AI is module-based; load your verified CUDA stack here if the venv's
# torch needs it (often unnecessary with pip wheels — they bundle CUDA libs).
# module load ...

source .venv/bin/activate
if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -n "${SCRATCHDIR:-}" ]]; then
        HF_HOME="$SCRATCHDIR/ctm/huggingface"
    else
        HF_HOME="$REPO_DIR/.hf_cache"
    fi
fi
export HF_HOME
mkdir -p "$HF_HOME"

srun python scripts/train_rlct.py \
    --backend local \
    --local-dtype bfloat16 \
    --local-sampler vllm \
    --local-gpu-mem-util 0.45 \
    -y \
    "$@"
