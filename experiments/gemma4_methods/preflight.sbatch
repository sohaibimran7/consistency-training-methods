#!/usr/bin/env bash
#SBATCH --job-name=gemma-method-gate
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
set -euo pipefail
: "${GEMMA_DEPLOY_REPO:?}" "${GEMMA_METHODS_ROOT:?}" "${GEMMA_MODEL_SNAPSHOT:?}"
: "${GEMMA_CPU_RECEIPT:?}" "${GEMMA_GATE_OUTPUT:?}"
export GEMMA_LAUNCH_PROFILE=generation VLLM_WORKER_MULTIPROC_METHOD=spawn
source "$GEMMA_DEPLOY_REPO/experiments/gemma4_methods/deployment_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
methods=(act attct mlpct)
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-2]$ ]] || exit 2
gemma_probe_method="${methods[$SLURM_ARRAY_TASK_ID]}"
gemma_probe_output="$GEMMA_GATE_OUTPUT/internal-$SLURM_ARRAY_JOB_ID-$gemma_probe_method"
for gemma_probe_stage in update restore; do
    "$CTM_MUSE_RUNTIME_PYTHON" -B -m experiments.gemma4_methods.native_method_probe "$gemma_probe_stage" \
        --repository "$GEMMA_DEPLOY_REPO" --commit "$GEMMA_DEPLOY_COMMIT" \
        --cpu-receipt "$GEMMA_CPU_RECEIPT" --data-root "$GEMMA_METHODS_ROOT" \
        --model "$GEMMA_MODEL_SNAPSHOT" --method "$gemma_probe_method" --output "$gemma_probe_output"
done
