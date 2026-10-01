#!/usr/bin/env bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
set -euo pipefail
umask 077
: "${GEMMA_DEPLOY_REPO:?}" "${GEMMA_EVAL_ROOT:?}" "${GEMMA_GRADING_ENV_FILE:?}"
export GEMMA_LAUNCH_PROFILE=grading
source "$GEMMA_DEPLOY_REPO/experiments/gemma4_methods/deployment_env.sh"
"$GEMMA_RUNTIME_PYTHON" -m experiments.gemma4_methods.score merge --root "$GEMMA_EVAL_ROOT" --output "$GEMMA_EVAL_ROOT/scored"
credential_file="$GEMMA_GRADING_ENV_FILE"
[[ -f "$credential_file" && ! -L "$credential_file" ]]
case "$(stat -c '%a' "$credential_file")" in 400|600) ;; *) exit 2 ;; esac
set -a
source "$credential_file"
set +a
: "${OPENROUTER_API_KEY:?}"
exec "$GEMMA_RUNTIME_PYTHON" -m experiments.gemma4_methods.score grade --output "$GEMMA_EVAL_ROOT/scored"
