#!/usr/bin/env bash
# Re-attest the archived no-CoT supervised adapters with fixed-token HF/vLLM
# probes. This is provenance-only: it neither generates benchmark responses
# nor modifies the production PEFT adapters.
set -euo pipefail

REPO=/workspace/ctm-eval-none-20260801/repo
RUN=/workspace/ctm-act-repair-20260731
PY=$RUN/env/bin/python
FROZEN=$REPO/artifacts/stage1-iid-diagnostic-none-20260801
ROOT=$REPO/artifacts/stage1-supervised-parity-recovery-20260802
RAW_ROOT=$ROOT/raw-adapters
COMPAT_ROOT=$ROOT/vllm-compat-adapters
PARITY_ROOT=$ROOT/runtime-parity

CONDITION=${CTM_SUPERVISED_PARITY_CONDITION:?set CTM_SUPERVISED_PARITY_CONDITION to repaired-act, mlpct, or attct}
GPU=${CTM_SUPERVISED_PARITY_GPU:?set CTM_SUPERVISED_PARITY_GPU to an unused physical GPU index}
PORT=${CTM_SUPERVISED_PARITY_PORT:-8794}
# A failed amplified diagnostic can safely resume from an immutable completed
# primary report. It never reuses a partial amplified report.
RESUME_COMPOSITE=${CTM_SUPERVISED_PARITY_RESUME_COMPOSITE:-0}

case "$CONDITION" in
  repaired-act|mlpct|attct) ;;
  *)
    echo "unknown supervised parity condition: $CONDITION" >&2
    exit 2
    ;;
esac
case "$RESUME_COMPOSITE" in
  0|1) ;;
  *)
    echo "CTM_SUPERVISED_PARITY_RESUME_COMPOSITE must be 0 or 1" >&2
    exit 2
    ;;
esac

RAW=$RAW_ROOT/$CONDITION
COMPAT=$COMPAT_ROOT/$CONDITION
DATA=$FROZEN/train-eval-n200.jsonl
for path in "$RAW/adapter_model.safetensors" "$RAW/adapter_config.json" "$RAW/manifest.json" "$DATA"; do
  test -f "$path"
done
if [ "$RESUME_COMPOSITE" = 1 ]; then
  test -d "$COMPAT"
else
  test ! -e "$COMPAT"
fi

cd "$REPO"
if [ "$RESUME_COMPOSITE" != 1 ]; then
  PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
    --source "$RAW" \
    --destination "$COMPAT"
fi

run_parity() {
  local hf_adapter=$1
  local vllm_adapter=$2
  local output=$3
  local port=$4
  shift 4
  test ! -e "$output"
  CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HOME="$RUN/hf-cache" \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    PYTHONPATH=. \
    "$PY" -m experiments.act_repair_gate.runtime_parity \
      --model Qwen/Qwen3.5-9B \
      --adapter "$vllm_adapter" \
      --hf-adapter "$hf_adapter" \
      --data "$DATA" \
      --output-dir "$output" \
      --samples 8 \
      --top-token-count 16 \
      --hf-batch-size 2 \
      --vllm-port "$port" \
      --max-model-len 32768 \
      --vllm-memory-utilization 0.90 \
      --enforce-eager \
      --isolate-vllm-variants \
      "$@"
}

if [ "$CONDITION" = attct ]; then
  PRIMARY=$PARITY_ROOT/attct-primary
else
  PRIMARY=$PARITY_ROOT/$CONDITION
fi
if [ "$RESUME_COMPOSITE" = 1 ]; then
  test -f "$PRIMARY/report.json"
else
  run_parity "$RAW" "$COMPAT" "$PRIMARY" "$PORT"
fi

# Prefer the ordinary four-variant attestation. The composite path is a
# narrow fallback only if this exact primary report has the documented
# nonzero-but-mismatched self-attention signal.
if PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --adapter "$COMPAT" \
  --parity-report "$PRIMARY/report.json"; then
  echo "supervised direct parity re-audit complete: $CONDITION"
  exit 0
fi

# The direct attestor is fail-closed and does not write on a failed parity
# report. Verify the *specific* documented weak-signal shape before spending
# GPU time on the separately bound composite exception below; this is not a
# generic fallback for an arbitrary direct-attestation failure.
PYTHONPATH=. "$PY" - "$PRIMARY/report.json" <<'PY'
import json
import sys
from pathlib import Path

from ctm.evals.qwen35_vllm_attestation import (
    COMPOSITE_PRIMARY_PASSING_VARIANTS,
    COMPOSITE_WEAK_SIGNAL_VARIANT,
    PASSING_VERDICT,
    _primary_weak_signal_result_is_documented,
)

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
results = report.get("results") if isinstance(report, dict) else None
if not isinstance(results, dict):
    raise SystemExit("primary parity report has no result mapping for composite fallback")
if any(
    not isinstance(results.get(variant), dict)
    or results[variant].get("verdict") != PASSING_VERDICT
    for variant in COMPOSITE_PRIMARY_PASSING_VARIANTS
):
    raise SystemExit("primary parity report is not eligible for composite fallback")
if not _primary_weak_signal_result_is_documented(results.get(COMPOSITE_WEAK_SIGNAL_VARIANT)):
    raise SystemExit("primary parity report lacks the documented weak self-attention signal")
PY

AMPLIFIED_HF=$ROOT/amplified-self-attn-hf/$CONDITION
AMPLIFIED_VLLM=$ROOT/amplified-self-attn-vllm/$CONDITION
AMPLIFIED_PARITY=$PARITY_ROOT/${CONDITION}-self-attn-amplified
test ! -e "$AMPLIFIED_HF"
test ! -e "$AMPLIFIED_VLLM"
PYTHONPATH=. "$PY" -m experiments.act_repair_gate.amplify_self_attention \
  --source "$RAW" \
  --destination "$AMPLIFIED_HF"
PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --source "$AMPLIFIED_HF" \
  --destination "$AMPLIFIED_VLLM"
run_parity "$AMPLIFIED_HF" "$AMPLIFIED_VLLM" "$AMPLIFIED_PARITY" "$((PORT + 1))" \
  --variants self_attn_only
PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --adapter "$COMPAT" \
  --primary-parity-report "$PRIMARY/report.json" \
  --amplified-self-attn-report "$AMPLIFIED_PARITY/report.json"
echo "supervised composite parity re-audit complete: $CONDITION"
