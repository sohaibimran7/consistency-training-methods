#!/usr/bin/env bash
#SBATCH --job-name=rmct352-aita16
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct352-aita-%j.out
set -euo pipefail
umask 077
: "${REPO_DIR:?}" "${AITA_ROOT:?}" "${CTM_RMCT_EVAL_PYTHON:?}"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR" PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$CTM_RMCT_EVAL_PYTHON"):$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1 CTM_DISABLE_CUDNN_SDP=0
unset SLURM_CPUS_PER_GPU
worker="$REPO_DIR/infra/isambard/run_rmct_step352_aita_worker.sh"
if [[ "${CTM_AITA_DEBUG_ONLY:-0}" == 1 ]]; then
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=16 \
        --gpu-bind=verbose,per_task:1 bash "$worker" smoke
    exit
fi
for mode in probe worker; do
    if [[ "$mode" == worker ]]; then
        "$CTM_RMCT_EVAL_PYTHON" -m experiments.elephant_aita_ntaflip.step352 verify-topology --root "$AITA_ROOT"
    fi
    srun --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus=16 --gpus-per-task=1 --cpus-per-task=16 \
        --gpu-bind=verbose,per_task:1 --distribution=block:block bash "$worker" "$mode"
done
"$CTM_RMCT_EVAL_PYTHON" -u -m experiments.elephant_aita_ntaflip.step352 finalize --root "$AITA_ROOT"
