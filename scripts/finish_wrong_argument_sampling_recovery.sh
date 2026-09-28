#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/ctm-qwen-stage1-20260731
REPO_ROOT="$RUN_ROOT/repo"
RECOVERY_PID=84852

while kill -0 "$RECOVERY_PID" 2>/dev/null; do
  sleep 30
done

cd "$REPO_ROOT"
set -a
source .env
set +a

COMMON_ENV=(
  env
  PYTHONPATH="$REPO_ROOT"
  MCQ_BIAS_DATA_DIR="$RUN_ROOT/mcq-bias-data"
  HF_HOME=/workspace/huggingface
  HF_HUB_OFFLINE=1
  WANDB_MODE=offline
  TOKENIZERS_PARALLELISM=false
  TMPDIR="$RUN_ROOT/tmp"
)
PYTHON=/workspace/ctm-qwen-validation/env/bin/python

"${COMMON_ENV[@]}" "$PYTHON" -m ctm_data.adapters.mcq_bias.materialize_eval \
  --bias-types \
  suggested_answer \
  distractor_fact \
  wrong_argument \
  post_hoc \
  spurious_few_shot_squares \
  wrong_few_shot \
  --datasets artifacts/rmct-hle-dense-models-shared/data/hle-text-mc.jsonl \
  --prompt-style none \
  --n-questions 100 \
  --min-n-questions 92 \
  --seed 42 \
  --argument-model openrouter/google/gemma-4-31b-it \
  --argument-generation-rounds 1 \
  --dataset-dir artifacts/rmct-hle-dense-models-shared/mcq-bias-evaluation \
  --yes \
  >"$RUN_ROOT/logs/shared-eval-sampling-freeze.log" 2>&1

"${COMMON_ENV[@]}" "$PYTHON" scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/shared_data.yaml \
  --stages data_preparation \
  --yes \
  >"$RUN_ROOT/logs/shared-eval-sampling-register.log" 2>&1

echo "evaluation suite sampling recovery, freeze, and registration complete"
