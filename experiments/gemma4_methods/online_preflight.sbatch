#!/usr/bin/env bash
#SBATCH --job-name=gemma-online-gate
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --time=02:00:00
set -euo pipefail
: "${GEMMA_DEPLOY_REPO:?}" "${GEMMA_METHODS_ROOT:?}" "${GEMMA_MODEL_SNAPSHOT:?}"
: "${GEMMA_CPU_RECEIPT:?}" "${GEMMA_GATE_OUTPUT:?}" "${GEMMA_METHOD:?}"
case "$GEMMA_METHOD" in bct|opct) ;; *) exit 2;; esac
export GEMMA_LAUNCH_PROFILE=generation VLLM_WORKER_MULTIPROC_METHOD=spawn
source "$GEMMA_DEPLOY_REPO/experiments/gemma4_methods/deployment_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
export CTM_EXCLUDE_LENGTH_TERMINATED=1 CTM_ROLLOUT_DIAGNOSTICS=1 CTM_ROLLOUT_PROGRESS_TIMEOUT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
gemma_probe_output="$GEMMA_GATE_OUTPUT/online-$SLURM_JOB_ID-$GEMMA_METHOD"
for gemma_probe_stage in update restore; do
    srun --ntasks=1 --gpus=4 --cpus-per-task=288 --cpu-bind=none \
        "$CTM_MUSE_RUNTIME_PYTHON" -m ctm.backends.local.gpu_affinity \
        --output "$GEMMA_GATE_OUTPUT/placement-$SLURM_JOB_ID-$gemma_probe_stage.json" -- \
        "$CTM_MUSE_RUNTIME_PYTHON" -B -m experiments.gemma4_methods.native_method_probe "$gemma_probe_stage" \
        --repository "$GEMMA_DEPLOY_REPO" --commit "$GEMMA_DEPLOY_COMMIT" \
        --cpu-receipt "$GEMMA_CPU_RECEIPT" --data-root "$GEMMA_METHODS_ROOT" \
        --model "$GEMMA_MODEL_SNAPSHOT" --method "$GEMMA_METHOD" --output "$gemma_probe_output"
done
