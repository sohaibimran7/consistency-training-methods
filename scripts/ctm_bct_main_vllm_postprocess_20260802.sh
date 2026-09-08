#!/usr/bin/env bash
# Resume only the post-generation half of the BCT-main recovery after raw
# logs and their immutable preflight report already exist. It never invokes a
# model sampler or rewrites raw EvalLogs.
set -euo pipefail

REPO=/workspace/ctm-eval-none-20260801/repo
PY=/workspace/ctm-act-repair-20260731/env/bin/python
FROZEN=$REPO/artifacts/stage1-iid-diagnostic-none-20260801
STAGE=$REPO/artifacts/stage1-bct-main-vllm-recovery-20260802
CONDITION=bct-main-vllm-compat

cd "$REPO"
test -f "$REPO/.env"
test -f "$FROZEN/manifest.json"
test -f "$STAGE/provenance/$CONDITION.raw-preflight.json"
test -d "$STAGE/raw/$CONDITION"
test ! -e "$STAGE/graded"
test ! -e "$STAGE/analysis"

set -a
. "$REPO/.env"
set +a
PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic_none.grade_luna \
  --raw-log-root "$STAGE/raw" \
  --preflight-report "$STAGE/provenance/$CONDITION.raw-preflight.json" \
  --output-root "$STAGE/graded" \
  --workers 5 \
  --connections-per-worker 100 \
  --grader-max-tokens 1024

mkdir -p "$STAGE/analysis"
PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic.analyze \
  --graded-root "$STAGE/graded" \
  --manifest "$FROZEN/manifest.json" \
  --grader-max-tokens 1024 \
  --output "$STAGE/analysis/$CONDITION.json"

echo "BCT-main recovery postprocess complete: $STAGE/analysis/$CONDITION.json"
