#!/usr/bin/env bash
# Invoke inside the coordinator's approved allocation; never submits a job.
set -euo pipefail
: "${GEMMA_RESTART_PYTHON:?}" "${GEMMA_RESTART_MODEL:?}" "${GEMMA_RESTART_CPU_RECEIPT:?}" "${GEMMA_RESTART_PARITY_OUTPUT:?}"
: "${GEMMA_RESTART_SCORING_TAIL_APPROVAL_REFERENCE:?Explicit per-run one-token diagnostic approval required}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for stage in hf vllm; do
  "$GEMMA_RESTART_PYTHON" -m experiments.rmct_restart_20260928.gemma_launch_parity "$stage" \
    --model "$GEMMA_RESTART_MODEL" --cpu-receipt "$GEMMA_RESTART_CPU_RECEIPT" \
    --output "$GEMMA_RESTART_PARITY_OUTPUT" \
    --scoring-tail-approval-reference "$GEMMA_RESTART_SCORING_TAIL_APPROVAL_REFERENCE"
done
