#!/usr/bin/env bash
# Finish RMCT H200 held-out queues after the two legacy task-13 decoders end.
#
# The legacy outer schedulers are deliberately SIGSTOP'd. This guard waits for
# their exact task-13 cells, proves no decoder descends from either scheduler,
# records an immutable handoff claim, kills only those stopped PIDs, and then
# launches non-overlapping tasks 14..21 on H200 GPUs 1 (main) and 3 (control).
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-ood-hle-20260802/env/bin/python}
RUN=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json}
MAIN_PARENT=${CTM_RMCT_MAIN_HELDOUT_PARENT:-20180}
CONTROL_PARENT=${CTM_RMCT_CONTROL_HELDOUT_PARENT:-20181}
MAIN_CHECKPOINT=${CTM_RMCT_MAIN_CHECKPOINT:-$RUN/adapters/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801_rate-matching-lr-1e-4}
CONTROL_CHECKPOINT=${CTM_RMCT_CONTROL_CHECKPOINT:-$RUN/adapters/raw-final-adapter}
POLL_SECONDS=${CTM_RMCT_HELDOUT_GUARD_POLL_SECONDS:-10}

readonly MAIN_RAW="$RUN/raw-no-luna/rmct-hf-peft"
readonly CONTROL_RAW="$RUN/raw-no-luna/rmct-control-hf-peft"
readonly LOCK_PATH="$RUN/runners/.rmct-h200-heldout-guard.lock"
readonly CLAIM_PATH="$RUN/runners/rmct-h200-heldout-guard.claim.json"
readonly RESULT_PATH="$RUN/runners/rmct-h200-heldout-guard.result.json"
readonly MAIN_LOG="$RUN/runners/rmct-h200-main-heldout-tasks14-21.log"
readonly CONTROL_LOG="$RUN/runners/rmct-h200-control-heldout-tasks14-21.log"
readonly MODEL_ARGS='{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}'
readonly GENERATION_CONFIG='{"max_connections":8,"max_tokens":20480,"temperature":1.0,"top_k":20,"top_p":0.95}'

for pid in "$MAIN_PARENT" "$CONTROL_PARENT"; do
  case "$pid" in ''|*[!0-9]*|0) echo "held-out scheduler PID must be positive: $pid" >&2; exit 2 ;; esac
done
case "$POLL_SECONDS" in ''|*[!0-9]*|0) echo "CTM_RMCT_HELDOUT_GUARD_POLL_SECONDS must be a positive integer" >&2; exit 2 ;; esac
if [ "$POLL_SECONDS" -gt 60 ]; then echo "held-out guard polling must be at most 60 seconds" >&2; exit 2; fi
for path in "$REPO" "$MAIN_CHECKPOINT" "$CONTROL_CHECKPOINT" "$MAIN_RAW" "$CONTROL_RAW"; do test -d "$path"; done
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

# The manifest is an input to task construction.  Pin its bytes before this
# guard starts waiting and prove they did not change before any handoff action.
readonly FROZEN_SHA256="$(manifest_sha256)"

assert_frozen_manifest() {
  test "$(manifest_sha256)" = "$FROZEN_SHA256" || {
    echo "frozen Stage 2 manifest changed while the held-out guard was waiting; refusing handoff" >&2
    exit 2
  }
}

exec 9>"$LOCK_PATH"
flock -n 9 || { echo "another H200 held-out guard owns $LOCK_PATH" >&2; exit 2; }
for path in "$CLAIM_PATH" "$RESULT_PATH" "$MAIN_LOG" "$CONTROL_LOG"; do
  test ! -e "$path" || { echo "refusing stale or existing held-out handoff artifact: $path" >&2; exit 2; }
done

attest_scheduler() {
  local pid=$1
  local checkpoint=$2
  local raw_root=$3
  local require_quiescent=$4
  "$PY" - "$pid" "$checkpoint" "$raw_root" "$require_quiescent" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

pid, checkpoint, raw_root, require_quiescent = sys.argv[1:]
proc = Path("/proc") / pid
if not proc.is_dir():
    raise SystemExit(f"attested held-out scheduler PID vanished: {pid}")
stat = (proc / "stat").read_text(encoding="utf-8")
tail = stat.rsplit(")", 1)[1].split()
if len(tail) < 20:
    raise SystemExit(f"cannot parse /proc/{pid}/stat")
tokens = [item.decode("utf-8", "surrogateescape") for item in (proc / "cmdline").read_bytes().split(b"\0") if item]
if len(tokens) < 2 or not tokens[0].endswith("/python") or tokens[1] != "scripts/run_evals.py":
    raise SystemExit(f"PID {pid} is not the expected run_evals Python process: {tokens[:2]!r}")

def values(flag: str) -> list[str]:
    result: list[str] = []
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 >= len(tokens):
                raise SystemExit(f"PID {pid} has incomplete {flag}")
            result.append(tokens[index + 1])
    return result

if tail[0] != "T":
    raise SystemExit(f"PID {pid} is not an intentionally stopped scheduler: state={tail[0]}")
if values("--task-index") != [str(index) for index in range(13, 22)]:
    raise SystemExit(f"PID {pid} has unexpected task-index tokens: {values('--task-index')!r}")
if values("--local-checkpoint") != [checkpoint]:
    raise SystemExit(f"PID {pid} checkpoint token does not match the expected condition")
if values("--log-dir") != [raw_root]:
    raise SystemExit(f"PID {pid} log-dir token does not match the expected condition")
if values("--task-factory") != ["experiments.stage2_ood_hle.tasks:ood_tasks"]:
    raise SystemExit(f"PID {pid} task-factory token does not match Stage 2 OOD")
if "--isolate-tasks" not in tokens:
    raise SystemExit(f"PID {pid} does not use the required isolated-task scheduler")

ppid_by_pid: dict[int, int] = {}
for candidate in Path("/proc").iterdir():
    if not candidate.name.isdigit():
        continue
    try:
        fields = (candidate / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except (FileNotFoundError, PermissionError, IndexError):
        continue
    if len(fields) > 1:
        ppid_by_pid[int(candidate.name)] = int(fields[1])
descendants: set[int] = set()
frontier = [int(pid)]
while frontier:
    ancestor = frontier.pop()
    children = [child for child, parent in ppid_by_pid.items() if parent == ancestor and child not in descendants]
    descendants.update(children)
    frontier.extend(children)
if require_quiescent == "1" and descendants:
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

boundary_state() {
  cd "$REPO"
  PYTHONPATH=. "$PY" - "$FROZEN" "$FROZEN_SHA256" "$MAIN_RAW" "$CONTROL_RAW" "$MAIN_CHECKPOINT" "$CONTROL_CHECKPOINT" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest = Path(sys.argv[1])
expected_manifest_sha256 = sys.argv[2]
main_root, control_root, main_checkpoint, control_checkpoint = map(Path, sys.argv[3:])
actual_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
if actual_manifest_sha256 != expected_manifest_sha256:
    raise SystemExit("frozen Stage 2 manifest changed while waiting; refusing held-out handoff")
specs = ood_task_specs(manifest)
task13_spec = specs[12]
task13 = raw_preflight._task_identity(task13_spec)
task1_spec = specs[0]
task1 = raw_preflight._task_identity(task1_spec)
targets = {raw_preflight._task_identity(specs[index - 1]): index for index in range(14, 22)}
for root, checkpoint in ((main_root, main_checkpoint), (control_root, control_checkpoint)):
    seen: dict[tuple[object, ...], tuple[Path, object, object]] = {}
    for path in raw_preflight._discover_eval_log_paths(root):
        try:
            header = raw_preflight._read_eval_log(path, header_only=True)
        except Exception as exc:
            raise SystemExit(f"could not read EvalLog header under {root}: {path}: {exc}") from exc
        evaluation = raw_preflight._attribute(header, "eval")
        task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
        if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
            continue
        identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
        if identity in seen:
            raise SystemExit(f"duplicate relevant EvalLog identity in {root}: {path} and {seen[identity][0]}")
        seen[identity] = (path, raw_preflight._attribute(header, "status"), header)
    if seen.get(task13, (None, None, None))[1] != "success":
        print("waiting")
        raise SystemExit(0)
    if seen.get(task1, (None, None, None))[1] != "success":
        raise SystemExit(f"{root}: task 13 is successful but its exact LogiQA clean reference is not")
    clean_path, _, clean_header = seen[task1]
    task13_path, _, task13_header = seen[task13]
    clean_created = raw_preflight._validate_header(clean_header, path=clean_path, spec=task1_spec, raw_root=root)
    task13_created = raw_preflight._validate_header(task13_header, path=task13_path, spec=task13_spec, raw_root=root)
    try:
        clean_full = raw_preflight._read_eval_log(clean_path, header_only=False)
        task13_full = raw_preflight._read_eval_log(task13_path, header_only=False)
    except Exception as exc:
        raise SystemExit(f"could not fully read task-13 boundary logs under {root}: {exc}") from exc
    clean_loaded = raw_preflight.LoadedTaskLog(task1_spec, clean_path, clean_created, clean_header, "", {})
    task13_loaded = raw_preflight.LoadedTaskLog(task13_spec, task13_path, task13_created, task13_header, "", {})
    raw_preflight._validate_samples(clean_full, loaded=clean_loaded, clean_paths={})
    raw_preflight._validate_samples(
        task13_full,
        loaded=task13_loaded,
        clean_paths={(task13_spec.population, task13_spec.dataset): clean_path},
    )
    runtime = raw_preflight._validate_runtime_contract(
        runtime_profile="hf-peft",
        expected_base_model="Qwen/Qwen3.5-9B",
        expected_checkpoint=str(checkpoint),
        expected_max_connections=8,
        require_vllm_adapter_attestation=True,
    )
    raw_preflight._assert_runtime(clean_path, runtime=runtime)
    raw_preflight._assert_runtime(task13_path, runtime=runtime)
    collisions = {index: seen[identity][0] for identity, index in targets.items() if identity in seen}
    if collisions:
        raise SystemExit(f"{root}: refuses existing task 14..21 cells: {collisions}")
print("ready")
PY
}

gpu_idle() {
  local gpu=$1
  local pids
  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')
  test -z "$pids" || { echo "GPU $gpu is not idle: $pids" >&2; return 1; }
}

write_json_once() {
  local destination=$1
  local state=$2
  local main_attestation=$3
  local control_attestation=$4
  "$PY" - "$destination" "$state" "$main_attestation" "$control_attestation" "$FROZEN" "$FROZEN_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

destination, state, main_attestation, control_attestation, manifest, expected_manifest_sha256 = sys.argv[1:]
actual_manifest_sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
if actual_manifest_sha256 != expected_manifest_sha256:
    raise SystemExit("frozen Stage 2 manifest changed before handoff claim; refusing write")
payload = {
    "schema": "rmct-h200-heldout-guard-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "manifest": manifest,
    "manifest_sha256": expected_manifest_sha256,
    "main_scheduler": json.loads(main_attestation),
    "control_scheduler": json.loads(control_attestation),
    "assignments": {
        "rmct-hf-peft": {"gpu": 1, "task_indices": list(range(14, 22))},
        "rmct-control-hf-peft": {"gpu": 3, "task_indices": list(range(14, 22))},
    },
}
path = Path(destination)
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

attest_worker() {
  local pid=$1
  local checkpoint=$2
  local raw_root=$3
  local gpu=$4
  "$PY" - "$pid" "$PY" "$checkpoint" "$raw_root" "$gpu" "$FROZEN" "$FROZEN_SHA256" "$REPO" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

pid, expected_python, checkpoint, raw_root, gpu, manifest, manifest_sha256, repo = sys.argv[1:]
proc = Path("/proc") / pid
if not proc.is_dir():
    raise SystemExit(f"launched worker PID vanished: {pid}")
stat = (proc / "stat").read_text(encoding="utf-8")
tail = stat.rsplit(")", 1)[1].split()
if len(tail) < 20 or tail[0] in {"Z", "X", "T", "t"}:
    raise SystemExit(f"launched worker PID is not live: {pid}")
cmdline = (proc / "cmdline").read_bytes()
tokens = [item.decode("utf-8", "surrogateescape") for item in cmdline.split(b"\0") if item]
if len(tokens) < 2 or tokens[0] != expected_python or tokens[1] != "scripts/run_evals.py":
    raise SystemExit(f"launched PID {pid} is not the exact expected run_evals worker: {tokens[:2]!r}")

def values(flag: str) -> list[str]:
    result: list[str] = []
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 >= len(tokens):
                raise SystemExit(f"PID {pid} has incomplete {flag}")
            result.append(tokens[index + 1])
    return result

if values("--task-index") != [str(index) for index in range(14, 22)]:
    raise SystemExit(f"PID {pid} has unexpected task indices: {values('--task-index')!r}")
if values("--local-checkpoint") != [checkpoint]:
    raise SystemExit(f"PID {pid} checkpoint does not match its claimed condition")
if values("--log-dir") != [raw_root]:
    raise SystemExit(f"PID {pid} log root does not match its claimed condition")
if values("--task-factory") != ["experiments.stage2_ood_hle.tasks:ood_tasks"]:
    raise SystemExit(f"PID {pid} does not use the Stage 2 OOD task factory")
if "--isolate-tasks" not in tokens or values("--max-tasks") != ["1"]:
    raise SystemExit(f"PID {pid} does not use the required isolated-task scheduler")
if values("--base-model") != ["Qwen/Qwen3.5-9B"]:
    raise SystemExit(f"PID {pid} has an unexpected base model")
try:
    model_args = json.loads(values("--model-args")[0])
    generation = json.loads(values("--generation-config")[0])
    task_args = json.loads(values("--task-args")[0])
except (IndexError, json.JSONDecodeError) as exc:
    raise SystemExit(f"PID {pid} has unreadable JSON task/model arguments") from exc
if model_args != {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"}:
    raise SystemExit(f"PID {pid} has unexpected HF model args: {model_args!r}")
if generation != {"max_connections": 8, "max_tokens": 20480, "temperature": 1.0, "top_k": 20, "top_p": 0.95}:
    raise SystemExit(f"PID {pid} has unexpected generation contract: {generation!r}")
if task_args != {"include_bias_acknowledged": False, "manifest": manifest, "prompt_style": "none", "unbiased_log": raw_root}:
    raise SystemExit(f"PID {pid} task args do not bind the frozen manifest and exact raw root: {task_args!r}")
if hashlib.sha256(Path(manifest).read_bytes()).hexdigest() != manifest_sha256:
    raise SystemExit("frozen Stage 2 manifest changed during held-out launch")
environment = {}
for item in (proc / "environ").read_bytes().split(b"\0"):
    if b"=" in item:
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
expected_environment = {
    "CUDA_VISIBLE_DEVICES": gpu,
    "PYTHONPATH": repo,
    "HF_HOME": "/workspace/hf-cache",
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_XET": "1",
}
for key, value in expected_environment.items():
    if environment.get(key) != value:
        raise SystemExit(f"PID {pid} environment {key} does not match handoff contract")
print(json.dumps({
    "pid": int(pid),
    "state": tail[0],
    "start_ticks": int(tail[19]),
    "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    "cmdline_tokens": tokens,
}, sort_keys=True))
PY
}

initial_main=$(attest_scheduler "$MAIN_PARENT" "$MAIN_CHECKPOINT" "$MAIN_RAW" 0)
initial_control=$(attest_scheduler "$CONTROL_PARENT" "$CONTROL_CHECKPOINT" "$CONTROL_RAW" 0)
while :; do
  state=$(boundary_state) || exit $?
  if [ "$state" = ready ]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

# All launch inputs and destinations are resolved before either old scheduler
# is removed. The re-attestation below proves these are still the same PIDs.
assert_frozen_manifest
gpu_idle 1
gpu_idle 3
MAIN_TASK_ARGS=$("$PY" -c 'import json,sys; print(json.dumps({"manifest":sys.argv[1],"unbiased_log":sys.argv[2],"prompt_style":"none","include_bias_acknowledged":False},sort_keys=True))' "$FROZEN" "$MAIN_RAW")
CONTROL_TASK_ARGS=$("$PY" -c 'import json,sys; print(json.dumps({"manifest":sys.argv[1],"unbiased_log":sys.argv[2],"prompt_style":"none","include_bias_acknowledged":False},sort_keys=True))' "$FROZEN" "$CONTROL_RAW")
before_main=$(attest_scheduler "$MAIN_PARENT" "$MAIN_CHECKPOINT" "$MAIN_RAW" 1)
before_control=$(attest_scheduler "$CONTROL_PARENT" "$CONTROL_CHECKPOINT" "$CONTROL_RAW" 1)
test "$before_main" = "$initial_main" || { echo "main held-out scheduler identity changed; refusing handoff" >&2; exit 2; }
test "$before_control" = "$initial_control" || { echo "control held-out scheduler identity changed; refusing handoff" >&2; exit 2; }
state=$(boundary_state) || exit $?
test "$state" = ready || { echo "held-out boundary is no longer ready; refusing handoff" >&2; exit 2; }
assert_frozen_manifest
write_json_once "$CLAIM_PATH" "claimed_before_parent_removal" "$before_main" "$before_control"
test "$(attest_scheduler "$MAIN_PARENT" "$MAIN_CHECKPOINT" "$MAIN_RAW" 1)" = "$before_main"
test "$(attest_scheduler "$CONTROL_PARENT" "$CONTROL_CHECKPOINT" "$CONTROL_RAW" 1)" = "$before_control"
kill -KILL -- "$MAIN_PARENT"
kill -KILL -- "$CONTROL_PARENT"
for _ in $(seq 1 20); do
  test ! -e "/proc/$MAIN_PARENT" && test ! -e "/proc/$CONTROL_PARENT" && break
  sleep 1
done
test ! -e "/proc/$MAIN_PARENT" && test ! -e "/proc/$CONTROL_PARENT" || {
  echo "one stopped held-out scheduler survived SIGKILL" >&2
  exit 2
}
state=$(boundary_state) || exit $?
test "$state" = ready || { echo "held-out boundary changed after scheduler removal; refusing launch" >&2; exit 2; }
assert_frozen_manifest
gpu_idle 1
gpu_idle 3

nohup env -u VLLM_BASE_URL -u VLLM_API_KEY -u CTM_PERSISTENT_VLLM_SERVER_METADATA \
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$REPO" HF_HOME=/workspace/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  "$PY" scripts/run_evals.py --task-factory experiments.stage2_ood_hle.tasks:ood_tasks \
  --local-checkpoint "$MAIN_CHECKPOINT" --base-model Qwen/Qwen3.5-9B \
  --task-args "$MAIN_TASK_ARGS" --model-args "$MODEL_ARGS" --generation-config "$GENERATION_CONFIG" \
  --log-dir "$MAIN_RAW" --max-tasks 1 --isolate-tasks --yes \
  --task-index 14 --task-index 15 --task-index 16 --task-index 17 --task-index 18 --task-index 19 --task-index 20 --task-index 21 \
  >"$MAIN_LOG" 2>&1 9>&- &
main_pid=$!
nohup env -u VLLM_BASE_URL -u VLLM_API_KEY -u CTM_PERSISTENT_VLLM_SERVER_METADATA \
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$REPO" HF_HOME=/workspace/hf-cache HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  "$PY" scripts/run_evals.py --task-factory experiments.stage2_ood_hle.tasks:ood_tasks \
  --local-checkpoint "$CONTROL_CHECKPOINT" --base-model Qwen/Qwen3.5-9B \
  --task-args "$CONTROL_TASK_ARGS" --model-args "$MODEL_ARGS" --generation-config "$GENERATION_CONFIG" \
  --log-dir "$CONTROL_RAW" --max-tasks 1 --isolate-tasks --yes \
  --task-index 14 --task-index 15 --task-index 16 --task-index 17 --task-index 18 --task-index 19 --task-index 20 --task-index 21 \
  >"$CONTROL_LOG" 2>&1 9>&- &
control_pid=$!

# Do not record a launch handoff merely because nohup returned.  Both direct
# workers must still be live and exactly match the immutable launch contract.
sleep 2
main_worker=$(attest_worker "$main_pid" "$MAIN_CHECKPOINT" "$MAIN_RAW" 1)
control_worker=$(attest_worker "$control_pid" "$CONTROL_CHECKPOINT" "$CONTROL_RAW" 3)

"$PY" - "$RESULT_PATH" "$CLAIM_PATH" "$main_worker" "$control_worker" "$FROZEN" "$FROZEN_SHA256" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

result, claim, main_worker, control_worker, manifest, expected_manifest_sha256 = sys.argv[1:]
actual_manifest_sha256 = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
if actual_manifest_sha256 != expected_manifest_sha256:
    raise SystemExit("frozen Stage 2 manifest changed before result write; refusing write")
payload = {
    "schema": "rmct-h200-heldout-guard-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "state": "launched",
    "claim": claim,
    "manifest": manifest,
    "manifest_sha256": expected_manifest_sha256,
    "main_worker": json.loads(main_worker),
    "control_worker": json.loads(control_worker),
}
descriptor = os.open(Path(result), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
printf 'launched H200 held-out queues: main=%s control=%s\n' "$main_pid" "$control_pid"
