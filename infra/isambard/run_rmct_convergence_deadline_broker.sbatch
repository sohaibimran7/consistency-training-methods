#!/usr/bin/env bash
# Deadline broker for the approved RMCT-convergence run.  This job may reserve
# the node while the immutable source/data/readiness receipt is transferred,
# but it cannot enter the production launcher until that receipt verifies.
#SBATCH --job-name=ctm-rmct-convergence
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct-convergence-%j.out

set -euo pipefail
umask 077

# Readiness proves an offline, exact snapshot contract before the broker
# permits production entry; no verifier path may silently obtain a new model.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

repo_root="${REPO_DIR:?submit with an explicit durable REPO_DIR}"
repo_root="$(cd -- "$repo_root" && pwd -P)"
ready_receipt="$repo_root/artifacts/rmct-convergence-production-ready.json"
verify_script="$repo_root/infra/isambard/verify_rmct_convergence_production_ready.py"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_deadline.sh"

wait_seconds="${CTM_RMCT_READY_WAIT_SECONDS:-7200}"
case "$wait_seconds" in
    ''|*[!0-9]*) echo "ERROR: CTM_RMCT_READY_WAIT_SECONDS must be an integer." >&2; exit 2 ;;
esac
deadline=$(( $(date +%s) + wait_seconds ))

echo "RMCT-convergence broker job=$SLURM_JOB_ID waiting for a verified production receipt."
while true; do
    if [[ -f "$ready_receipt" && -f "$verify_script" && -x "$launcher" ]]; then
        if "$repo_root/.venv/bin/python" "$verify_script" \
            --receipt "$ready_receipt" \
            --repository "$repo_root"; then
            break
        fi
        echo "Readiness receipt exists but is not yet valid; refusing to launch." >&2
    fi
    if (( $(date +%s) >= deadline )); then
        echo "ERROR: timed out before a valid RMCT-convergence readiness receipt appeared." >&2
        exit 3
    fi
    # Wait without holding GPU time.  A valid receipt triggers requeue; the
    # subsequent allocation performs production.  This broker can therefore
    # be submitted early without burning a four-GPU node during assembly.
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "Production is not ready; requeueing instead of idling on GPUs."
        scontrol requeue "$SLURM_JOB_ID"
        exit 0
    fi
    sleep 15
done

echo "Verified readiness receipt; entering the four-GPU production launcher."
cd "$repo_root"
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    "$launcher" --ready-receipt "$ready_receipt" --yes
