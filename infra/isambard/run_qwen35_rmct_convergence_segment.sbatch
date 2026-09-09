#!/usr/bin/env bash
# Submit one exact sealed RMCT convergence segment after a predecessor job.
# This is optional normal queue integration; the deadline broker instead calls
# the allocated-node launcher directly.  Neither path launches without --yes.
#SBATCH --job-name=ctm-rmct-convergence-segment
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct-convergence-segment-%j.out

set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]] || (( 10#$1 > 31 )); then
    echo "Usage: sbatch .../run_qwen35_rmct_convergence_segment.sbatch <segment-index 0..31>" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" ]]; then
    echo "ERROR: submit with REPO_DIR pointing to the durable repository checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_deadline.sh"
ready="$repo_root/artifacts/rmct-convergence-production-ready.json"
if [[ ! -x "$launcher" ]]; then
    echo "ERROR: production launcher is not executable: $launcher" >&2
    exit 2
fi
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    "$launcher" --ready-receipt "$ready" --segment-index "$1" --yes
