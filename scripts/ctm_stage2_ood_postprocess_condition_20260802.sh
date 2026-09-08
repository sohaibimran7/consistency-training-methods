#!/usr/bin/env bash
# Hash-bound Stage 2 OOD handoff for one *completed* condition.
#
# This script is intentionally separate from generation.  It refuses to send
# anything to Luna until the raw 21-cell generation matrix has a launcher
# completion marker, has passed `raw_preflight`, and has been staged with
# SHA-256 verification.  A host-wide flock serializes all Luna invocations so
# the approved 5 x 100 connection limit remains global, not per condition.
set -euo pipefail

RUN=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
REPO=${CTM_OOD_REPO:-$RUN/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}
ENV_FILE=${CTM_OOD_ENV_FILE:-/workspace/ctm-eval-none-20260801/repo/.env}

if [ "$#" -ne 1 ]; then
  echo "usage: $0 CONDITION" >&2
  exit 2
fi
CONDITION=$1

readonly FROZEN="$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json"
readonly PREFLIGHT_DIR="$RUN/artifacts/preflight-v1"
readonly STAGED_ROOT="$RUN/luna-staged-v1"
readonly DERIVED_ROOT="$RUN/luna-derived-v1"
readonly GRADE_LOCK="$RUN/.luna-grade.lock"
readonly CONDITION_LOCK="$RUN/.luna-postprocess-${CONDITION}.lock"
readonly STATE_DIR="$RUN/artifacts/postprocess-state-v1"
readonly STATE_FILE="$STATE_DIR/${CONDITION}.complete"

RAW_ROOT=
RUNTIME_PROFILE=
EXPECTED_CHECKPOINT=
EXPECTED_MAX_CONNECTIONS=
COMPLETE_MARKER=
case "$CONDITION" in
  base-vllm)
    RAW_ROOT="$RUN/logs/$CONDITION"
    RUNTIME_PROFILE=vllm
    COMPLETE_MARKER="Stage 2 OOD vLLM raw condition complete: $CONDITION"
    ;;
  act-vllm-compat)
    RAW_ROOT="$RUN/logs/$CONDITION"
    RUNTIME_PROFILE=vllm
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-supervised-parity-recovery-20260802/vllm-compat-adapters/repaired-act
    COMPLETE_MARKER="Stage 2 OOD vLLM raw condition complete: $CONDITION"
    ;;
  attct-vllm-compat)
    RAW_ROOT="$RUN/logs/$CONDITION"
    RUNTIME_PROFILE=vllm
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-supervised-parity-recovery-20260802/vllm-compat-adapters/attct
    COMPLETE_MARKER="Stage 2 OOD vLLM raw condition complete: $CONDITION"
    ;;
  mlpct-vllm-compat)
    RAW_ROOT="$RUN/logs/$CONDITION"
    RUNTIME_PROFILE=vllm
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-supervised-parity-recovery-20260802/vllm-compat-adapters/mlpct
    COMPLETE_MARKER="Stage 2 OOD vLLM raw condition complete: $CONDITION"
    ;;
  opct-vllm-compat)
    RAW_ROOT="$RUN/logs/$CONDITION"
    RUNTIME_PROFILE=vllm
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-opct-handoff-20260802/vllm-compat-adapter
    COMPLETE_MARKER="Stage 2 OOD vLLM raw condition complete: $CONDITION"
    ;;
  rmct-vllm-compat)
    RAW_ROOT="$RUN/logs/$CONDITION"
    RUNTIME_PROFILE=vllm
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-rmct-main-source-b8-20260802/vllm-compat-adapter
    COMPLETE_MARKER="Stage 2 OOD vLLM raw condition complete: $CONDITION"
    ;;
  rmct-hf-peft)
    RAW_ROOT="$RUN/raw-no-luna/$CONDITION"
    RUNTIME_PROFILE=hf-peft
    # The native-HF OOD matrix uses the hash-pinned raw adapter staged under
    # this run root.  It is deliberately distinct from the separately
    # translated, vLLM-attested compatibility adapter.
    EXPECTED_CHECKPOINT="$RUN/adapters/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801_rate-matching-lr-1e-4"
    EXPECTED_MAX_CONNECTIONS=8
    COMPLETE_MARKER="Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION"
    ;;
  bct-hf-peft)
    RAW_ROOT="$RUN/raw-no-luna/$CONDITION"
    RUNTIME_PROFILE=hf-peft
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct
    EXPECTED_MAX_CONNECTIONS=8
    COMPLETE_MARKER="Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION"
    ;;
  bct-control-hf-peft)
    RAW_ROOT="$RUN/raw-no-luna/$CONDITION"
    RUNTIME_PROFILE=hf-peft
    EXPECTED_CHECKPOINT=/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct-control
    EXPECTED_MAX_CONNECTIONS=8
    COMPLETE_MARKER="Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION"
    ;;
  rmct-control-hf-peft)
    RAW_ROOT="$RUN/raw-no-luna/$CONDITION"
    RUNTIME_PROFILE=hf-peft
    # As above, bind the finalizer to the exact raw control adapter used by
    # the native-HF workers, rather than a historical staging location.
    EXPECTED_CHECKPOINT="$RUN/adapters/raw-final-adapter"
    EXPECTED_MAX_CONNECTIONS=8
    COMPLETE_MARKER="Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: $CONDITION"
    ;;
  *)
    echo "unknown Stage 2 OOD condition: $CONDITION" >&2
    exit 2
    ;;
esac

test -x "$PY"
test -f "$FROZEN"
test -f "$ENV_FILE"

# This per-condition lock makes a daemon retry harmless.  It also prevents a
# second operator from creating a competing preflight/stage/grade handoff.
mkdir -p "$STATE_DIR"
exec 8>"$CONDITION_LOCK"
if ! flock -n 8; then
  exit 0
fi
if [ -f "$STATE_FILE" ]; then
  exit 0
fi

# A directory can contain a partially-written EvalLog while generation is
# active.  Only the launcher's exact final marker authorizes the handoff.
# Vast's lean image does not include ripgrep, so use Bash's nullglob plus
# grep over the two small launcher/runner log sets.
shopt -s nullglob
COMPLETION_LOGS=("$RUN"/launchers/*.log "$RUN"/runners/*.log)
if [ "${#COMPLETION_LOGS[@]}" -eq 0 ] || ! grep -F -l -- "$COMPLETE_MARKER" "${COMPLETION_LOGS[@]}" >/dev/null 2>&1; then
  exit 3
fi
test -d "$RAW_ROOT"

REPORT="$PREFLIGHT_DIR/$CONDITION.json"
PREFLIGHT_ARGS=(
  --raw-log-root "$RAW_ROOT"
  --manifest "$FROZEN"
  --condition "$CONDITION"
  --runtime-profile "$RUNTIME_PROFILE"
  --output "$REPORT"
)
if [ -n "$EXPECTED_CHECKPOINT" ]; then
  PREFLIGHT_ARGS+=(--expected-checkpoint "$EXPECTED_CHECKPOINT")
fi
if [ -n "$EXPECTED_MAX_CONNECTIONS" ]; then
  PREFLIGHT_ARGS+=(--expected-max-connections "$EXPECTED_MAX_CONNECTIONS")
fi

export PYTHONPATH="$REPO"
"$PY" -m experiments.stage2_ood_hle.raw_preflight "${PREFLIGHT_ARGS[@]}"
"$PY" -m experiments.stage2_ood_hle.stage_luna \
  --preflight-report "$REPORT" \
  --output-root "$STAGED_ROOT"

# The connection cap in grade_luna applies to one process.  The host-wide
# lock extends it to every completed Stage 2 condition.  Once acquired,
# restage all published reports (byte-identical resumes only) so a prior
# interrupted handoff cannot cause a missing staged input in a later batch.
exec 9>"$GRADE_LOCK"
flock 9

REPORTS=("$PREFLIGHT_DIR"/*.json)
if [ ! -f "${REPORTS[0]}" ]; then
  echo "no Stage 2 OOD preflight reports were found" >&2
  exit 1
fi
for READY_REPORT in "${REPORTS[@]}"; do
  "$PY" -m experiments.stage2_ood_hle.stage_luna \
    --preflight-report "$READY_REPORT" \
    --output-root "$STAGED_ROOT"
done

GRADE_ARGS=(
  --staged-raw-root "$STAGED_ROOT"
  --output-root "$DERIVED_ROOT"
  --workers 5
  --connections-per-worker 100
)
for READY_REPORT in "${REPORTS[@]}"; do
  GRADE_ARGS+=(--preflight-report "$READY_REPORT")
done

# Do not print exported credentials.  The scorer is the only statement in
# this script that can make an OpenRouter request.
set -a
. "$ENV_FILE"
set +a
# The user environment may define PYTHONPATH for an unrelated checkout.
# Restore the immutable staged repository after importing only the approved
# grader credentials, including for the spawned Luna worker processes.
export PYTHONPATH="$REPO"
cd "$REPO"
"$PY" -m experiments.stage2_ood_hle.grade_luna "${GRADE_ARGS[@]}"

STATE_TEMP="$STATE_FILE.$$"
printf '%s\n' "stage2-ood-postprocess-v1 condition=$CONDITION completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$STATE_TEMP"
mv "$STATE_TEMP" "$STATE_FILE"
printf '%s\n' "Stage 2 OOD postprocess complete: $CONDITION"
