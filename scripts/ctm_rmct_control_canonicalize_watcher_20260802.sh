#!/usr/bin/env bash
# Wait for the active RMCT-control runner to publish its immutable alias
# analysis, then canonicalize it once. This is deliberately CPU-only and never
# alters raw logs, graded logs, or the runner's own alias report.
set -euo pipefail

REPO=/workspace/ctm-eval-none-20260801/repo
PY=/workspace/ctm-act-repair-20260731/env/bin/python
FROZEN=$REPO/artifacts/stage1-iid-diagnostic-none-20260801
STAGE=$REPO/artifacts/stage1-rmct-control-b8-accelerated-20260802
ALIAS=$STAGE/analysis/rmct-control-b8-accelerated.json
OUTPUT=$STAGE/analysis/rmct-control-canonical.json
PROVENANCE=$STAGE/analysis/rmct-control-canonicalization.json

if [ -e "$OUTPUT" ] || [ -e "$PROVENANCE" ]; then
  if [ -f "$OUTPUT" ] && [ -f "$PROVENANCE" ]; then
    echo "RMCT-control canonicalization already published; nothing to do."
    exit 0
  fi
  echo "refusing partial existing RMCT-control canonicalization state" >&2
  exit 1
fi

for _ in $(seq 1 720); do
  if [ -f "$ALIAS" ]; then
    cd "$REPO"
    PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic_none.canonicalize_rmct_control \
      --alias-analysis "$ALIAS" \
      --graded-root "$STAGE/graded" \
      --manifest "$FROZEN/manifest.json" \
      --output "$OUTPUT" \
      --provenance-output "$PROVENANCE"
    echo "RMCT-control canonicalization complete: $OUTPUT"
    exit 0
  fi
  sleep 30
done

echo "timed out waiting for RMCT-control alias analysis: $ALIAS" >&2
exit 1
