#!/usr/bin/env bash
# One-shot safety handoff for the legacy H200 RMCT task-4 scheduler.
#
# It never starts an evaluation. Once the exact task-4 EvalLog is successful,
# it suspends and removes only the attested legacy scheduler before it can
# advance into task 5, which is already owned by the primary host.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-ood-hle-20260802/env/bin/python}
RUN=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json}
PARENT_PID=${CTM_RMCT_MAIN_CORE_PARENT:-16190}
CHECKPOINT=${CTM_RMCT_MAIN_CHECKPOINT:-$RUN/adapters/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801_rate-matching-lr-1e-4}
POLL_SECONDS=${CTM_RMCT_TASK4_GUARD_POLL_SECONDS:-1}

readonly RAW_ROOT="$RUN/raw-no-luna/rmct-hf-peft"
readonly LOCK_PATH="$RUN/runners/.rmct-h200-task4-guard.lock"
readonly CLAIM_PATH="$RUN/runners/rmct-h200-task4-guard.claim.json"
readonly RESULT_PATH="$RUN/runners/rmct-h200-task4-guard.result.json"

case "$PARENT_PID" in ''|*[!0-9]*|0) echo "CTM_RMCT_MAIN_CORE_PARENT must be a positive PID" >&2; exit 2 ;; esac
case "$POLL_SECONDS" in ''|*[!0-9]*|0) echo "CTM_RMCT_TASK4_GUARD_POLL_SECONDS must be a positive integer" >&2; exit 2 ;; esac
for path in "$REPO" "$CHECKPOINT" "$RAW_ROOT"; do test -d "$path"; done
test -f "$FROZEN"
test -x "$PY"
command -v flock >/dev/null
mkdir -p "$RUN/runners"

manifest_sha256() {
  "$PY" - "$FROZEN" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

readonly FROZEN_SHA256="$(manifest_sha256)"

assert_frozen_manifest() {
  test "$(manifest_sha256)" = "$FROZEN_SHA256" || {
    echo "frozen Stage 2 manifest changed while task-4 guard was waiting; refusing signal" >&2
    exit 2
  }
}

exec 9>"$LOCK_PATH"
flock -n 9 || { echo "another RMCT task-4 guard owns $LOCK_PATH" >&2; exit 2; }
test ! -e "$CLAIM_PATH" || { echo "refusing stale or duplicate task-4 guard claim: $CLAIM_PATH" >&2; exit 2; }
test ! -e "$RESULT_PATH" || { echo "refusing stale or duplicate task-4 guard result: $RESULT_PATH" >&2; exit 2; }

attest_parent() {
  "$PY" - "$PARENT_PID" "$CHECKPOINT" "$RAW_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

pid, checkpoint, raw_root = sys.argv[1:]
proc = Path("/proc") / pid
if not proc.is_dir():
    raise SystemExit(f"attested parent PID vanished: {pid}")
stat = (proc / "stat").read_text(encoding="utf-8")
tail = stat.rsplit(")", 1)[1].split()
if len(tail) < 20:
    raise SystemExit(f"cannot parse /proc/{pid}/stat")
tokens = [item.decode("utf-8", "surrogateescape") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
if len(tokens) < 2 or not tokens[0].endswith("/python") or tokens[1] != "scripts/run_evals.py":
    raise SystemExit(f"PID {pid} is not the expected run_evals Python process: {tokens[:2]!r}")

def values(flag: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 >= len(tokens):
                raise SystemExit(f"PID {pid} has incomplete {flag}")
            values.append(tokens[index + 1])
    return values

expected_indices = [str(index) for index in range(4, 13)]
if values("--task-index") != expected_indices:
    raise SystemExit(f"PID {pid} has unexpected task-index tokens: {values('--task-index')!r}")
if values("--local-checkpoint") != [checkpoint]:
    raise SystemExit("PID checkpoint token does not match the attested RMCT main checkpoint")
if values("--log-dir") != [raw_root]:
    raise SystemExit("PID log-dir token does not match the attested RMCT main raw root")
if values("--task-factory") != ["experiments.stage2_ood_hle.tasks:ood_tasks"]:
    raise SystemExit("PID task-factory token does not match the Stage 2 OOD factory")
if "--isolate-tasks" in tokens:
    raise SystemExit("legacy task-4 parent unexpectedly uses isolate-tasks")

ppid_by_pid: dict[int, int] = {}
for candidate in Path("/proc").iterdir():
    if not candidate.name.isdigit():
        continue
    try:
        candidate_tail = (candidate / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except (FileNotFoundError, PermissionError, IndexError):
        continue
    if len(candidate_tail) > 1:
        ppid_by_pid[int(candidate.name)] = int(candidate_tail[1])
descendants: set[int] = set()
frontier = [int(pid)]
while frontier:
    ancestor = frontier.pop()
    children = [child for child, parent in ppid_by_pid.items() if parent == ancestor and child not in descendants]
    descendants.update(children)
    frontier.extend(children)
if descendants:
    raise SystemExit(f"PID {pid} has live descendants: {sorted(descendants)}")

print(json.dumps({
    "pid": int(pid),
    "state": tail[0],
    "start_ticks": int(tail[19]),
    "cmdline_sha256": hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest(),
    "cmdline_tokens": tokens,
}, sort_keys=True))
PY
}

matrix_state() {
  cd "$REPO"
  PYTHONPATH=. "$PY" - "$FROZEN" "$FROZEN_SHA256" "$RAW_ROOT" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest = Path(sys.argv[1])
expected_manifest_sha256 = sys.argv[2]
raw_root = Path(sys.argv[3])
if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_manifest_sha256:
    raise SystemExit("frozen Stage 2 manifest changed while task-4 guard was waiting")
specs = ood_task_specs(manifest)
task1_spec = specs[0]
task1 = raw_preflight._task_identity(task1_spec)
task4_spec = specs[3]
task4 = raw_preflight._task_identity(task4_spec)
forbidden = {raw_preflight._task_identity(specs[index - 1]) for index in range(5, 13)}
seen: dict[tuple[object, ...], tuple[Path, object, object]] = {}
for path in raw_preflight._discover_eval_log_paths(raw_root):
    try:
        header = raw_preflight._read_eval_log(path, header_only=True)
    except Exception as exc:
        raise SystemExit(f"could not read EvalLog header under {raw_root}: {path}: {exc}") from exc
    evaluation = raw_preflight._attribute(header, "eval")
    task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
    if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
        continue
    identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
    if identity in seen:
        raise SystemExit(f"duplicate relevant EvalLog identity in {raw_root}: {path} and {seen[identity][0]}")
    seen[identity] = (path, raw_preflight._attribute(header, "status"), header)

if task4 not in seen or seen[task4][1] != "success":
    print("waiting")
elif forbidden & set(seen):
    matches = sorted(str(seen[identity][0]) for identity in forbidden & set(seen))
    raise SystemExit("unsafe: legacy parent created later task output: " + ", ".join(matches))
else:
    if seen.get(task1, (None, None, None))[1] != "success":
        raise SystemExit("task 4 is successful but its exact LogiQA clean reference is not")
    clean_path, _, clean_header = seen[task1]
    task4_path, _, task4_header = seen[task4]
    clean_created = raw_preflight._validate_header(clean_header, path=clean_path, spec=task1_spec, raw_root=raw_root)
    task4_created = raw_preflight._validate_header(task4_header, path=task4_path, spec=task4_spec, raw_root=raw_root)
    try:
        clean_full = raw_preflight._read_eval_log(clean_path, header_only=False)
        task4_full = raw_preflight._read_eval_log(task4_path, header_only=False)
    except Exception as exc:
        raise SystemExit(f"could not fully read task-4 boundary logs: {exc}") from exc
    clean_loaded = raw_preflight.LoadedTaskLog(task1_spec, clean_path, clean_created, clean_header, "", {})
    task4_loaded = raw_preflight.LoadedTaskLog(task4_spec, task4_path, task4_created, task4_header, "", {})
    raw_preflight._validate_samples(clean_full, loaded=clean_loaded, clean_paths={})
    raw_preflight._validate_samples(
        task4_full,
        loaded=task4_loaded,
        clean_paths={(task4_spec.population, task4_spec.dataset): clean_path},
    )
    print("ready")
PY
}

write_json_once() {
  local destination=$1
  local state=$2
  local attestation=$3
  "$PY" - "$destination" "$state" "$attestation" "$FROZEN" "$FROZEN_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

destination, state, attestation, manifest, expected_manifest_sha256 = sys.argv[1:]
if hashlib.sha256(Path(manifest).read_bytes()).hexdigest() != expected_manifest_sha256:
    raise SystemExit("frozen Stage 2 manifest changed before task-4 handoff record write")
path = Path(destination)
payload = {
    "schema": "rmct-h200-task4-guard-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "manifest": manifest,
    "manifest_sha256": expected_manifest_sha256,
    "attestation": json.loads(attestation),
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

same_process_identity() {
  "$PY" - "$1" "$2" <<'PY'
from __future__ import annotations

import json
import sys

before, after = (json.loads(item) for item in sys.argv[1:])
keys = ("pid", "start_ticks", "cmdline_sha256", "cmdline_tokens")
raise SystemExit(0 if all(before.get(key) == after.get(key) for key in keys) else 1)
PY
}

initial_attestation=$(attest_parent)
while :; do
  state=$(matrix_state) || exit $?
  if [ "$state" = ready ]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

# Commit an immutable claim before changing the legacy scheduler. Re-attest
# immediately before and after STOP so PID reuse cannot redirect the signal.
assert_frozen_manifest
write_json_once "$CLAIM_PATH" "claimed_before_stop" "$initial_attestation"
before_stop=$(attest_parent)
same_process_identity "$initial_attestation" "$before_stop" || {
  echo "legacy parent identity changed since initial attestation; refusing to signal it" >&2
  exit 2
}
kill -STOP -- "$PARENT_PID"
after_stop=$(attest_parent)
same_process_identity "$before_stop" "$after_stop" || {
  echo "legacy parent identity changed while applying SIGSTOP; refusing removal" >&2
  exit 2
}
case "$after_stop" in *'"state": "T"'*) ;; *) echo "legacy parent is not stopped after SIGSTOP" >&2; exit 2 ;; esac

# A newly-created task-5 log after SIGSTOP is evidence of a race. Leave the
# process stopped rather than silently mixing it with the primary task-5 run.
state=$(matrix_state) || exit $?
test "$state" = ready || {
  echo "later task output appeared while stopping the legacy parent; it remains stopped for inspection" >&2
  exit 2
}
assert_frozen_manifest
kill -KILL -- "$PARENT_PID"
for _ in $(seq 1 20); do
  test ! -e "/proc/$PARENT_PID" && break
  sleep 1
done
test ! -e "/proc/$PARENT_PID" || { echo "legacy parent survived SIGKILL: $PARENT_PID" >&2; exit 2; }
write_json_once "$RESULT_PATH" "removed_after_task4_success" "$after_stop"
printf '%s\n' "removed exact legacy H200 RMCT task-4 scheduler PID $PARENT_PID after successful task 4"
