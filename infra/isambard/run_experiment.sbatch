#!/usr/bin/env bash
# Run a selected experiment target on one Isambard GH200 node.
# The project association and default workq partition come from the Clifton
# project login. Override time or REPO_DIR at submission time when needed.
# Usage: sbatch infra/isambard/run_experiment.sbatch CONFIG.yaml --target isambard --yes
#SBATCH --job-name=ctm-experiment
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

srun python scripts/run_experiment.py "$@"
