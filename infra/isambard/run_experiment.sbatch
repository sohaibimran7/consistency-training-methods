#!/usr/bin/env bash
# Run a selected experiment target on one Isambard GH200 node.
# Configure account, partition, time, and REPO_DIR before first use.
# Usage: sbatch infra/isambard/run_experiment.sbatch CONFIG.yaml --target isambard --yes
#SBATCH --job-name=ctm-experiment
#SBATCH --account=EDIT-ME
#SBATCH --partition=EDIT-ME
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/consistency-training-methods}"
cd "$REPO_DIR"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"

srun python scripts/run_experiment.py "$@"
