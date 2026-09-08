#!/usr/bin/env bash
# Assemble the fully graded Stage 2 OOD matrix and render its dedicated
# four-column figures.  This script has no model or API path: it waits for
# every condition's successful postprocess sentinel, then consumes only the
# immutable Luna-derived EvalLogs.
set -euo pipefail

RUN=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
REPO=${CTM_OOD_REPO:-$RUN/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}

readonly STATE_DIR="$RUN/artifacts/postprocess-state-v1"
readonly ANALYSIS_DIR="$RUN/artifacts/analysis-v1"
readonly FIGURE_DIR="$RUN/artifacts/figures-v1"
readonly ANALYSIS="$ANALYSIS_DIR/stage2-ood-hle-four-column.json"
readonly STATE="$RUN/artifacts/analysis-state-v1/four-column.complete"
readonly LOCK="$RUN/.stage2-ood-analysis.lock"
readonly MANIFEST="$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json"

test -x "$PY"
test -f "$MANIFEST"

# The analysis names are intentionally stable presentation labels, while the
# source directories retain their backend-specific raw condition identities.
SOURCE_CONDITIONS=(
  base-vllm
  bct-hf-peft
  bct-control-hf-peft
  opct-vllm-compat
  rmct-hf-peft
  rmct-control-hf-peft
  act-vllm-compat
  attct-vllm-compat
  mlpct-vllm-compat
)
for SOURCE in "${SOURCE_CONDITIONS[@]}"; do
  if [ ! -f "$STATE_DIR/${SOURCE}.complete" ]; then
    exit 3
  fi
done

mkdir -p "$(dirname "$STATE")"
exec 8>"$LOCK"
if ! flock -n 8; then
  exit 0
fi
if [ -f "$STATE" ]; then
  echo "Stage 2 OOD analysis already complete"
  exit 0
fi

export PYTHONPATH="$REPO"
"$PY" -m experiments.stage2_ood_hle.analyze \
  --run "untrained=$RUN/luna-derived-v1/base-vllm" \
  --run "bct=$RUN/luna-derived-v1/bct-hf-peft" \
  --run "bct-control=$RUN/luna-derived-v1/bct-control-hf-peft" \
  --run "opct=$RUN/luna-derived-v1/opct-vllm-compat" \
  --run "rmct=$RUN/luna-derived-v1/rmct-hf-peft" \
  --run "rmct-control=$RUN/luna-derived-v1/rmct-control-hf-peft" \
  --run "act=$RUN/luna-derived-v1/act-vllm-compat" \
  --run "attct=$RUN/luna-derived-v1/attct-vllm-compat" \
  --run "mlpct=$RUN/luna-derived-v1/mlpct-vllm-compat" \
  --stage2-manifest "$MANIFEST" \
  --output "$ANALYSIS"
"$PY" -m experiments.stage2_ood_hle.plot \
  --analysis "$ANALYSIS" \
  --output-dir "$FIGURE_DIR"

STATE_TEMP="$STATE.$$"
printf '%s\n' "stage2-ood-analysis-v1 completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATE_TEMP"
mv "$STATE_TEMP" "$STATE"
printf '%s\n' "Stage 2 OOD four-column analysis complete"
