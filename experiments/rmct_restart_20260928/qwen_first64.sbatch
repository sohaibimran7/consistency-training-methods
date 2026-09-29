#!/usr/bin/env bash
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
umask 077
# Submit only after independently verifying actual incorporation of this exact
# commit and fetching the real origin/main into this clean deployment.
repo=${1:?exact incorporated repository}
py=${2:?pinned candidate Python}
plan=${3:?fresh bound plan}
evidence=${4:?fresh evidence root}
manifest=${5:?frozen validation prompts}
commit=${6:?exact incorporated commit}
campaign=${7:?fresh production campaign outside source}
cd "$repo"
test "$(git rev-parse HEAD)" = "$commit"
test -z "$(git status --porcelain)"
git merge-base --is-ancestor "$commit" origin/main
export PYTHONPATH="$repo" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$(dirname "$py"):$PATH"
unset PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 "$py" -m experiments.rmct_restart_20260928.regression_gate \
  --source-root "$repo" --source-commit "$commit" --output "$evidence/regressions"
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 "$py" -m experiments.rmct_restart_20260928.qwen_launch \
  --plan "$plan" --output "$evidence/preflight" --validation-manifest "$manifest"
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 "$py" -m experiments.rmct_restart_20260928.native_rl_gate \
  --plan "$plan" --preflight "$evidence/preflight/preflight-results.json" --output "$evidence/disposable-rl"
srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 "$py" -m experiments.rmct_restart_20260928.qwen_train_window \
  --plan "$plan" --campaign "$campaign" --target 64 --validation-manifest "$manifest" \
  --preflight "$evidence/preflight/preflight-results.json" --rl-gate "$evidence/disposable-rl/receipt.json" \
  --regression "$evidence/regressions/receipt.json"
# Stop at64. No evaluation submission and no later training window here.
# Never resume a checkpoint under evidence/disposable-rl.
