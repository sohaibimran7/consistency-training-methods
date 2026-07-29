#!/usr/bin/env bash
# Three two-GH200 Figure 6 Llama jobs, one immutable checkpoint per array task.
# Required paths and credentials are documented in run_figure6_model.sh.
#SBATCH --job-name=f6-llama
#SBATCH --nodes=1
#SBATCH --array=0-2%3
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=72
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

if [[ -z "${REPO_DIR:-}" ]]; then
    echo "ERROR: REPO_DIR must be exported before submission." >&2
    exit 2
fi
cd "$REPO_DIR"
source .venv/bin/activate

srun --kill-on-bad-exit=1 --unbuffered \
    bash infra/isambard/run_figure6_model.sh llama "$SLURM_ARRAY_TASK_ID"
