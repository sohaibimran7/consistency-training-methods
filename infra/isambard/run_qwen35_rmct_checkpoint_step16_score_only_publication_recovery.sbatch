#!/usr/bin/env bash
# Receipt-bound, CPU-only publication recovery for the failed r002 step-16
# finalizer.  This performs no generation or model inference: it can only
# derive six corrected switch-score EvalLogs in a new immutable namespace.
#
# This file is deliberately an execution *draft*.  It is not a submission
# helper and contains no scheduler-client call.  An operator must explicitly
# acknowledge independent review before a scheduler submits it.

#SBATCH --job-name=ctm-rmct-s16-score-recovery
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-rmct-s16-score-recovery-%j.out

set -euo pipefail
umask 077

if [[ "${CTM_RMCT_SCORE_ONLY_RECOVERY_APPROVED:-}" != "1" ]]; then
    echo "ERROR: set CTM_RMCT_SCORE_ONLY_RECOVERY_APPROVED=1 only after independent review" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable early-checkpoint evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
program="$repo_root/infra/isambard/rmct_checkpoint_step16_score_only_publication_recovery.py"
frozen_launcher="$repo_root/infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py"
training_repository="${CTM_RMCT_TRAINING_REPOSITORY:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo}"
if [[ ! -d "$training_repository" || -L "$training_repository" ]]; then
    echo "ERROR: CTM_RMCT_TRAINING_REPOSITORY must name the durable sealed-training checkout" >&2
    exit 2
fi
training_repository="$(cd -- "$training_repository" && pwd -P)"

python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"
if [[ ! -x "$python_requested" || ! -f "$python_requested" || ! -f "$program" || -L "$program" || ! -f "$frozen_launcher" || -L "$frozen_launcher" ]]; then
    echo "ERROR: evaluation Python, recovery program, or frozen launcher is unavailable or linked" >&2
    exit 2
fi
# Retain the venv spelling: resolving its Python symlink would silently select
# the base interpreter and drop the Inspect package selected for r002.
python_directory="$(cd -L -- "$(dirname -- "$python_requested")" && pwd -L)"
python_bin="$python_directory/$(basename -- "$python_requested")"
if [[ ! -x "$python_bin" || ! -f "$python_bin" ]]; then
    echo "ERROR: selected evaluation Python is unavailable or not executable" >&2
    exit 2
fi
export PATH="$python_directory:$PATH"
export PYTHONDONTWRITEBYTECODE=1

"$python_bin" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: CTM_RMCT_EVAL_PYTHON must preserve the evaluation venv")
import inspect_ai  # noqa: F401
PY

# The recovery program fails closed if either setting differs.  It reads and
# writes local EvalLogs only; no model/runtime service may be contacted.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

evaluation_parent="${CTM_RMCT_EVAL_PARENT_ROOT:-$repo_root/logs/rmct-evaluations}"
campaign_root="${CTM_RMCT_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/rmct-convergence-step16-step64-16gpu-v1-r002}"
step16_root="$evaluation_parent/rmct-convergence-step016-two-bias-v1-r002"
recovery_root="$campaign_root/recoveries/step16-score-only-publication-r002"
phase_one_receipt="$campaign_root/phases/phase-001.json"
clean_gate_receipt="$step16_root/stage2/paired-clean-ready/clean-gate-receipt.json"

if [[ ! -d "$evaluation_parent" || -L "$evaluation_parent" || ! -d "$campaign_root" || -L "$campaign_root" || ! -d "$step16_root" || -L "$step16_root" || ! -f "$phase_one_receipt" || -L "$phase_one_receipt" || ! -f "$clean_gate_receipt" || -L "$clean_gate_receipt" ]]; then
    echo "ERROR: r002 evaluation parent/campaign/step16 root, phase-one receipt, or sealed clean gate is missing or linked" >&2
    exit 2
fi
if [[ -e "$recovery_root" && ( -L "$recovery_root" || ! -d "$recovery_root" ) ]]; then
    echo "ERROR: recovery root exists but is not a regular directory" >&2
    exit 2
fi

exec "$python_bin" "$program" \
    --step16-root "$step16_root" \
    --campaign-root "$campaign_root" \
    --recovery-root "$recovery_root" \
    --frozen-launcher "$frozen_launcher" \
    --write
