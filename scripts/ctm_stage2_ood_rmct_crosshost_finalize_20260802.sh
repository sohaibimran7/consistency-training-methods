#!/usr/bin/env bash
# Finalize the native-HF RMCT Stage-2 matrix after its two matched halves have
# completed on separate already-rented hosts. This script never starts a
# generation or a grader. It waits for successful EvalLogs, transfers only the
# completed H200 biased logs, runs the normal fail-closed raw preflight on the
# primary host, and then writes the usual completion marker for its existing
# postprocess daemon.
set -euo pipefail

H200_HOST=${CTM_RMCT_H200_HOST:?CTM_RMCT_H200_HOST is required}
H200_PORT=${CTM_RMCT_H200_PORT:?CTM_RMCT_H200_PORT is required}
H200_KNOWN_HOSTS=${CTM_RMCT_H200_KNOWN_HOSTS:?CTM_RMCT_H200_KNOWN_HOSTS is required}
PRIMARY_HOST=${CTM_RMCT_PRIMARY_HOST:?CTM_RMCT_PRIMARY_HOST is required}
PRIMARY_PORT=${CTM_RMCT_PRIMARY_PORT:?CTM_RMCT_PRIMARY_PORT is required}
PRIMARY_KNOWN_HOSTS=${CTM_RMCT_PRIMARY_KNOWN_HOSTS:?CTM_RMCT_PRIMARY_KNOWN_HOSTS is required}
HANDOFF_STAGE=${CTM_RMCT_HANDOFF_STAGE:?CTM_RMCT_HANDOFF_STAGE is required}
POLL_SECONDS=${CTM_RMCT_POLL_SECONDS:-30}

RUN=/workspace/ctm-ood-hle-20260802
REPO=$RUN/repo
H200_PY=${CTM_RMCT_H200_PY:-$RUN/env/bin/python}
PRIMARY_PY=${CTM_RMCT_PRIMARY_PY:-/workspace/ctm-hf-accelerated-20260802/env/bin/python}
FROZEN=$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json

case "$POLL_SECONDS" in
  ''|*[!0-9]*|0) echo "CTM_RMCT_POLL_SECONDS must be a positive integer" >&2; exit 2 ;;
esac
if [ "$POLL_SECONDS" -gt 60 ]; then
  echo "CTM_RMCT_POLL_SECONDS must be at most 60 seconds" >&2
  exit 2
fi
for file in "$H200_KNOWN_HOSTS" "$PRIMARY_KNOWN_HOSTS"; do
  test -f "$file"
done
test -n "$HANDOFF_STAGE"
test "$HANDOFF_STAGE" != /
mkdir -p "$HANDOFF_STAGE"

h200_ssh() {
  ssh -T -p "$H200_PORT" \
    -o "UserKnownHostsFile=$H200_KNOWN_HOSTS" \
    -o StrictHostKeyChecking=yes \
    "root@$H200_HOST" "$@"
}

primary_ssh() {
  ssh -T -p "$PRIMARY_PORT" \
    -o "UserKnownHostsFile=$PRIMARY_KNOWN_HOSTS" \
    -o StrictHostKeyChecking=yes \
    "root@$PRIMARY_HOST" "$@"
}

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

# The H200 owns three clean logs, its original task-4 core log, and tasks
# 13..21. The primary owns the copied clean logs and isolated tasks 5..12.
# Requiring exact success counts here merely authorizes the copy; the full raw
# preflight below proves identities, pairing, bytes, and the 21-cell topology.
h200_complete() {
  h200_ssh "cd '$REPO' && PYTHONPATH=. '$H200_PY' -" <<'PY'
from pathlib import Path
from inspect_ai.log import read_eval_log

for condition in ("rmct-hf-peft", "rmct-control-hf-peft"):
    root = Path("/workspace/ctm-ood-hle-20260802/raw-no-luna") / condition
    clean = biased = 0
    for path in root.glob("*.eval"):
        try:
            log = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        if getattr(log, "status", None) != "success":
            continue
        task = str(getattr(getattr(log, "eval", None), "task", ""))
        if task.endswith("stage2_ood_unbiased"):
            clean += 1
        elif task.endswith("stage2_ood_biased"):
            biased += 1
    if (clean, biased) != (3, 10):
        raise SystemExit(f"{condition}: waiting for H200 success logs, got clean={clean}, biased={biased}")
PY
}

primary_complete() {
  primary_ssh "cd '$REPO' && PYTHONPATH=. '$PRIMARY_PY' -" <<'PY'
from pathlib import Path
from inspect_ai.log import read_eval_log

for condition in ("rmct-hf-peft", "rmct-control-hf-peft"):
    root = Path("/workspace/ctm-ood-hle-20260802/raw-no-luna") / condition
    clean = biased = 0
    for path in root.glob("*.eval"):
        try:
            log = read_eval_log(str(path), header_only=True)
        except Exception:
            continue
        if getattr(log, "status", None) != "success":
            continue
        task = str(getattr(getattr(log, "eval", None), "task", ""))
        if task.endswith("stage2_ood_unbiased"):
            clean += 1
        elif task.endswith("stage2_ood_biased"):
            biased += 1
    if (clean, biased) != (3, 8):
        raise SystemExit(f"{condition}: waiting for primary success logs, got clean={clean}, biased={biased}")
PY
}

wait_for() {
  local label=$1
  local check=$2
  until "$check"; do
    log "waiting for $label"
    sleep "$POLL_SECONDS"
  done
  log "$label is complete"
}

copy_biased_logs() {
  local condition=$1
  local source="$RUN/raw-no-luna/$condition"
  local destination="$HANDOFF_STAGE/$condition"
  local name
  mkdir -p "$destination"
  while IFS= read -r name; do
    test -n "$name"
    scp -P "$H200_PORT" \
      -o "UserKnownHostsFile=$H200_KNOWN_HOSTS" \
      -o StrictHostKeyChecking=yes \
      "root@$H200_HOST:$source/$name" "$destination/$name"
    scp -P "$PRIMARY_PORT" \
      -o "UserKnownHostsFile=$PRIMARY_KNOWN_HOSTS" \
      -o StrictHostKeyChecking=yes \
      "$destination/$name" "root@$PRIMARY_HOST:$source/$name"
  done < <(h200_ssh "find '$source' -maxdepth 1 -type f -name '*stage2-ood-biased*.eval' -printf '%f\\n' | sort")
  test "$(find "$destination" -maxdepth 1 -type f -name '*stage2-ood-biased*.eval' | wc -l | tr -d ' ')" = 10
}

run_final_preflight() {
  primary_ssh "cd '$REPO' && PYTHONPATH=. '$PRIMARY_PY' -m experiments.stage2_ood_hle.raw_preflight --raw-log-root '$RUN/raw-no-luna/rmct-hf-peft' --manifest '$FROZEN' --condition rmct-hf-peft --runtime-profile hf-peft --expected-checkpoint '$RUN/adapters/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801_rate-matching-lr-1e-4' --expected-max-connections 8 --output '$RUN/artifacts/preflight-v1/rmct-hf-peft.json'"
  primary_ssh "cd '$REPO' && PYTHONPATH=. '$PRIMARY_PY' -m experiments.stage2_ood_hle.raw_preflight --raw-log-root '$RUN/raw-no-luna/rmct-control-hf-peft' --manifest '$FROZEN' --condition rmct-control-hf-peft --runtime-profile hf-peft --expected-checkpoint '$RUN/adapters/raw-final-adapter' --expected-max-connections 8 --output '$RUN/artifacts/preflight-v1/rmct-control-hf-peft.json'"
}

publish_markers() {
  primary_ssh "mkdir -p '$RUN/runners'; printf '%s\\n' 'Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: rmct-hf-peft' > '$RUN/runners/rmct-hf-peft-crosshost-finalizer.log'; printf '%s\\n' 'Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: rmct-control-hf-peft' > '$RUN/runners/rmct-control-hf-peft-crosshost-finalizer.log'"
}

wait_for "H200 RMCT logs" h200_complete
wait_for "primary RMCT core logs" primary_complete
copy_biased_logs rmct-hf-peft
copy_biased_logs rmct-control-hf-peft
run_final_preflight
publish_markers
log "RMCT cross-host handoff complete; primary postprocess daemon may now grade the hash-bound raw logs"
