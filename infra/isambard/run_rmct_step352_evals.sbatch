#!/usr/bin/env bash
#SBATCH --job-name=rmct352-eval16
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct352-eval-%j.out
set -euo pipefail
umask 077
: "${REPO_DIR:?}" "${EVAL_ROOT:?}" "${CTM_RMCT_EVAL_PYTHON:?}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$CTM_RMCT_EVAL_PYTHON"):$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
unset SLURM_CPUS_PER_GPU
worker="$REPO_DIR/infra/isambard/run_rmct_step352_eval_worker.sh"
srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=4 --cpus-per-task=64 \
    bash "$worker" parity
if [[ "${CTM_EVAL_DEBUG_ONLY:-0}" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --gpus-per-task=1 \
        --cpus-per-task=16 --gpu-bind=verbose,per_task:1 bash "$worker" smoke
    exit
fi
for phase in 1 2; do
    srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus=16 --gpus-per-task=1 \
        --cpus-per-task=16 --gpu-bind=verbose,per_task:1 --distribution=block:block \
        bash "$worker" "$phase"
done
"$CTM_RMCT_EVAL_PYTHON" -m experiments.rmct_two_bias_eval.step352 score --root "$EVAL_ROOT"
