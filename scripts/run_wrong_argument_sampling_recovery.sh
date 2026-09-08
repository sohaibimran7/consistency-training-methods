#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/workspace/ctm-qwen-stage1-20260731
REPO_ROOT="$RUN_ROOT/repo"

cd "$REPO_ROOT"
set -a
source .env
set +a

exec env \
  PYTHONPATH="$REPO_ROOT" \
  MCQ_BIAS_DATA_DIR="$RUN_ROOT/mcq-bias-data" \
  HF_HOME=/workspace/huggingface \
  HF_HUB_OFFLINE=1 \
  WANDB_MODE=offline \
  TOKENIZERS_PARALLELISM=false \
  TMPDIR="$RUN_ROOT/tmp" \
  /workspace/ctm-qwen-validation/env/bin/python \
  scripts/recover_wrong_arguments_sampling.py
