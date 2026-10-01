#!/usr/bin/env bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
# Controller overrides --array=0-(workers-1) for larger reviewed manifests.
set -euo pipefail
umask 077
: "${GEMMA_DEPLOY_REPO:?}" "${GEMMA_EVAL_ROOT:?}"
: "${GEMMA_EVAL_WORKERS:?Explicit prepared-manifest worker count required}"
case "$GEMMA_EVAL_WORKERS" in 4|8|16|32) ;; *) exit 2;; esac
[[ "${SLURM_ARRAY_TASK_COUNT:?}" == "$GEMMA_EVAL_WORKERS" \
   && "${SLURM_ARRAY_TASK_MIN:?}" == 0 \
   && "${SLURM_ARRAY_TASK_MAX:?}" == "$((GEMMA_EVAL_WORKERS-1))" \
   && "${SLURM_ARRAY_TASK_STEP:?}" == 1 ]] || {
    echo 'Scheduled array does not cover the reviewed manifest worker count.' >&2; exit 2;
}
export GEMMA_LAUNCH_PROFILE=generation
source "$GEMMA_DEPLOY_REPO/experiments/gemma4_methods/deployment_env.sh"
export VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset CTM_HF_EOS_ONLY_NO_TOKEN_CAP VLLM_BASE_URL VLLM_API_KEY VLLM_DEFAULT_SERVER_ARGS
task_cache=$(mktemp -d /tmp/gemma-method-eval-XXXXXX)
export TMPDIR="$task_cache" TMP="$task_cache" TEMP="$task_cache" XDG_CACHE_HOME="$task_cache/cache"
exec "$CTM_MUSE_RUNTIME_PYTHON" -m experiments.gemma4_methods.evaluate run \
 --output "$GEMMA_EVAL_ROOT" --rank "$SLURM_ARRAY_TASK_ID" --workers "$GEMMA_EVAL_WORKERS"
