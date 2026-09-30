#!/usr/bin/env bash
# One independent method per allocation; online methods use one four-GPU node.
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --time=12:00:00
set -euo pipefail
umask 077
: "${GEMMA_METHODS_ROOT:?}" "${GEMMA_METHOD:?}" "${GEMMA_GPUS:?}"
: "${GEMMA_DEPLOY_REPO:?}" "${GEMMA_MODEL_SNAPSHOT:?}"
: "${GEMMA_SELECTION_CONTRACT:?}" "${GEMMA_SELECTION_FOLDER:?}" "${GEMMA_VERIFIER_FACTORY:?}"
case "$GEMMA_METHOD:$GEMMA_GPUS" in
    bct:4|opct:4|act:1|attct:1|mlpct:1) ;;
    *) echo 'Method/GPU count differs from reviewed recipe.' >&2; exit 2;;
esac
export GEMMA_LAUNCH_PROFILE=generation
source "$GEMMA_DEPLOY_REPO/experiments/gemma4_methods/deployment_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
export CTM_EXCLUDE_LENGTH_TERMINATED=1 CTM_ROLLOUT_DIAGNOSTICS=1 CTM_ROLLOUT_PROGRESS_TIMEOUT=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
unset SLURM_CPUS_PER_GPU SLURM_TRES_PER_TASK SRUN_CPUS_PER_TASK
cpus=16
if [[ "$GEMMA_GPUS" == 4 ]]; then cpus=288; fi
placement=()
if [[ "$GEMMA_GPUS" == 4 ]]; then
 placement=("$CTM_MUSE_RUNTIME_PYTHON" -m ctm.backends.local.gpu_affinity --output "$GEMMA_METHODS_ROOT/placement-$SLURM_JOB_ID.json" --)
fi
srun --ntasks=1 --gpus="$GEMMA_GPUS" --cpus-per-task="$cpus" --cpu-bind=none "${placement[@]}" \
 "$CTM_MUSE_RUNTIME_PYTHON" -m experiments.gemma4_methods.train train \
 --root "$GEMMA_METHODS_ROOT" --method "$GEMMA_METHOD" --updates=16 \
 --model "$GEMMA_MODEL_SNAPSHOT" \
 --selection-contract "$GEMMA_SELECTION_CONTRACT" --selection-folder "$GEMMA_SELECTION_FOLDER" \
 --verifier-factory "$GEMMA_VERIFIER_FACTORY"
# No autonomous historical loss-patience successor. Shared validation controller
# owns continuation and exact actual-update/attempted-QID cursor verification.
