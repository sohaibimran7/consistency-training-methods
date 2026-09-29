#!/usr/bin/env bash
# Real, isolated segment-0 readiness proof before submitting the 16-job chain.
# It is intentionally a separate Slurm identity from every production segment.
#
# Example (after the one-GPU setup_gpu_env job has made the validated venv):
#   sbatch --export=ALL,REPO_DIR=/project/.../consistency-training-methods \
#     infra/isambard/preflight_qwen35_rmct256_convergence.sbatch
#SBATCH --job-name=ctm-rmct256-preflight
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-rmct256-preflight-%j.out

set -euo pipefail

if [[ -z "${REPO_DIR:-}" && -z "${PROJECTDIR:-}" ]]; then
    echo "ERROR: set REPO_DIR to the durable Isambard checkout before submitting." >&2
    exit 2
fi

# Reuse the production allocation/bootstrap wrapper so the probe gets the
# exact same four-GH200 srun, SCRATCHDIR/HF_HOME normalization, Python
# interpreter, and private caches. The inner launcher exits before its marker
# and runner, and writes all probe evidence only under the dedicated preflight
# experiment/run namespace.
if [[ -n "${REPO_DIR:-}" ]]; then
    repo_root="$REPO_DIR"
else
    repo_root="$PROJECTDIR/$USER/consistency-training-methods"
fi
exec bash "$repo_root/infra/isambard/run_qwen35_rmct256_convergence_segment.sbatch" \
    --segment-index 0 --preflight-only --yes
