#!/usr/bin/env bash
# One method per GPU. Submit five methods independently; no existing jobs are altered.
#SBATCH --job-name=ctm-two-bias-methods
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --signal=USR1@3600
#SBATCH --output=slurm-two-bias-methods-%j.out

set -euo pipefail
umask 077
ctm_methods_repo="${REPO_DIR:?REPO_DIR is required}"
ctm_methods_method="${CTM_METHOD:?CTM_METHOD is required}"
ctm_methods_plan="${CTM_METHODS_PLAN:?CTM_METHODS_PLAN is required}"
ctm_methods_runs="${CTM_METHODS_RUN_ROOT:?CTM_METHODS_RUN_ROOT is required}"
ctm_methods_python="${CTM_METHODS_PYTHON:?CTM_METHODS_PYTHON is required}"
ctm_methods_script="$ctm_methods_repo/infra/isambard/run_methods_two_bias_convergence.sbatch"
cd "$ctm_methods_repo"
export PYTHONPATH="$ctm_methods_repo${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${SCRATCHDIR:?SCRATCHDIR is required}/ctm/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=16
ctm_methods_snapshot="$HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ctm_methods_alignment=()
case "$ctm_methods_method" in
    act|attct|mlpct)
        ctm_methods_alignment=(--alignment-audit "${CTM_METHODS_ALIGNMENT_AUDIT:?passing alignment audit is required}")
        ;;
esac
"$ctm_methods_python" -m experiments.methods_two_bias_convergence.plan verify \
    --repository "$ctm_methods_repo" --plan "$ctm_methods_plan"
ctm_methods_updates=128
ctm_methods_amended=0
if "$ctm_methods_python" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p.get("execution_amendment") and sys.argv[2] in ("bct", "opct") else 1)' \
    "$ctm_methods_plan" "$ctm_methods_method"; then
    ctm_methods_amended=1
    ctm_methods_updates=16
    "$ctm_methods_python" -m experiments.methods_two_bias_convergence.continuation prepare
    # Queue before srun: Slurm can kill the entire allocation at walltime.
    # The successor permits only completion or timeout with verified progress.
    "$ctm_methods_python" -m experiments.methods_two_bias_convergence.continuation queue
fi
srun --nodes=1 --ntasks=1 --gpus=1 --cpus-per-task=16 \
    "$ctm_methods_python" -m experiments.methods_two_bias_convergence.train \
    --repository "$ctm_methods_repo" --method "$ctm_methods_method" \
    --plan "$ctm_methods_plan" --model-snapshot "$ctm_methods_snapshot" \
    --run-root "$ctm_methods_runs" --updates-this-job "$ctm_methods_updates" "${ctm_methods_alignment[@]}"

if [[ "$ctm_methods_amended" == 1 ]]; then
    exit 0
fi

# The next slice is queued only after the previous one exits successfully and
# its sealed state says training is unfinished. An error is never auto-retried.
if "$ctm_methods_python" -c 'import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s["convergence"]["decision"] == "continue" else 1)' \
    "$ctm_methods_runs/$ctm_methods_method/state.json"; then
    sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" \
        --job-name="ctm-two-bias-$ctm_methods_method" "$ctm_methods_script"
fi
