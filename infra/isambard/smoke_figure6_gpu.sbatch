#!/usr/bin/env bash
# Validate the pinned AArch64 GPU serving environment before any model job is
# eligible to run. This job downloads no model weights and performs no eval.
#
# Required environment:
#   REPO_DIR    absolute repository checkout
#SBATCH --job-name=f6-gpu-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=72
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ -z "${REPO_DIR:-}" || "$REPO_DIR" != /* ]]; then
    echo "ERROR: REPO_DIR must be an absolute exported path." >&2
    exit 2
fi
if [[ ! -f "$REPO_DIR/infra/isambard/setup_gpu_env.sh" ]]; then
    echo "ERROR: REPO_DIR does not contain the Isambard GPU setup script." >&2
    exit 2
fi

cd "$REPO_DIR"
bash infra/isambard/setup_gpu_env.sh
echo "Figure 6 GPU environment smoke check passed."
