#!/usr/bin/env bash
# Launch one previously unowned Stage-2 OOD cell on one physical GPU.
#
# This is deliberately a narrow fan-out primitive: it validates the matching
# clean reference already staged in the condition-local root, creates an
# immutable claim, starts exactly one task index, and records that the direct
# worker is live with the declared runtime contract.  It never overwrites an
# EvalLog or a handoff record.
set -euo pipefail

REPO=${CTM_FANOUT_REPO:?Set CTM_FANOUT_REPO to the mirrored Stage-2 checkout.}
RUN=${CTM_FANOUT_RUN_ROOT:?Set CTM_FANOUT_RUN_ROOT to the mirrored Stage-2 run root.}
PY=${CTM_FANOUT_PY:?Set CTM_FANOUT_PY to the native-HF Python executable.}
CONDITION=${CTM_FANOUT_CONDITION:?Set CTM_FANOUT_CONDITION.}
CHECKPOINT=${CTM_FANOUT_CHECKPOINT:?Set CTM_FANOUT_CHECKPOINT.}
GPU=${CTM_FANOUT_GPU:?Set CTM_FANOUT_GPU to one physical GPU index.}
TASK_INDEX=${CTM_FANOUT_TASK_INDEX:?Set CTM_FANOUT_TASK_INDEX to one Stage-2 task index.}
HF_HOME_DIR=${CTM_FANOUT_HF_HOME:?Set CTM_FANOUT_HF_HOME to the complete HF cache.}
FROZEN=${CTM_FANOUT_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json}
DISABLE_CUDNN_SDP=${CTM_FANOUT_DISABLE_CUDNN_SDP:-0}

case "$CONDITION" in ''|*/*|.*) echo "CTM_FANOUT_CONDITION must be a single safe path component" >&2; exit 2 ;; esac
case "$GPU" in ''|*[!0-9]*) echo "CTM_FANOUT_GPU must be a non-negative integer" >&2; exit 2 ;; esac
case "$TASK_INDEX" in ''|*[!0-9]*|0) echo "CTM_FANOUT_TASK_INDEX must be a positive integer" >&2; exit 2 ;; esac
if [ "$TASK_INDEX" -gt 21 ]; then echo "CTM_FANOUT_TASK_INDEX must be at most 21" >&2; exit 2; fi
case "$DISABLE_CUDNN_SDP" in 0|1) ;; *) echo "CTM_FANOUT_DISABLE_CUDNN_SDP must be 0 or 1" >&2; exit 2 ;; esac
for path in "$REPO" "$RUN" "$CHECKPOINT" "$HF_HOME_DIR"; do test -d "$path"; done
test -x "$PY"
test -f "$FROZEN"
command -v nvidia-smi >/dev/null

RAW_ROOT="$RUN/raw-no-luna/$CONDITION"
CLAIM="$RUN/runners/fanout-$CONDITION-task$TASK_INDEX-gpu$GPU.claim.json"
RESULT="$RUN/runners/fanout-$CONDITION-task$TASK_INDEX-gpu$GPU.result.json"
LOG="$RUN/runners/fanout-$CONDITION-task$TASK_INDEX-gpu$GPU.log"
# The condition root is an intentional prerequisite: its matching clean EvalLog
# must have been staged and validated before this launcher may create a claim.
# The runner ledger itself is local scheduler metadata, so make it idempotently
# rather than failing a fresh fan-out host before any work has begun.
test -d "$RAW_ROOT"
mkdir -p "$RUN/runners"
for path in "$CLAIM" "$RESULT" "$LOG"; do
  test ! -e "$path" || { echo "refusing existing fan-out artifact: $path" >&2; exit 2; }
done

gpu_pids=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')
test -z "$gpu_pids" || { echo "GPU $GPU is not idle: $gpu_pids" >&2; exit 2; }

cd "$REPO"
PYTHONPATH=. "$PY" - "$FROZEN" "$RAW_ROOT" "$CHECKPOINT" "$TASK_INDEX" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

from experiments.stage1_iid_diagnostic.raw_preflight import _assert_expected_hf_peft_model
from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest, raw_root, checkpoint = map(Path, sys.argv[1:4])
task_index = int(sys.argv[4])
specs = ood_task_specs(manifest)
target = specs[task_index - 1]
if target.kind != "biased":
    raise SystemExit("the fan-out launcher is only for biased Stage-2 cells")

# A biased cell depends only on the clean generation for its own
# (population, dataset) pair.  Requiring the other two clean references here
# would turn an independent LogiQA/HellaSwag launch into an unnecessary HLE
# barrier.  We still validate that one reference completely, including its
# absolute-path binding and native-HF/PEFT runtime contract.
matching_clean_specs = [
    spec
    for spec in specs[:3]
    if (spec.population, spec.dataset) == (target.population, target.dataset)
]
if len(matching_clean_specs) != 1:
    raise SystemExit(
        "could not determine the unique clean reference for target "
        f"{raw_preflight._task_identity(target)}"
    )
clean_spec = matching_clean_specs[0]
seen: dict[tuple[object, ...], tuple[Path, object, object]] = {}
for path in raw_preflight._discover_eval_log_paths(raw_root):
    try:
        header = raw_preflight._read_eval_log(path, header_only=True)
    except Exception as exc:
        raise SystemExit(f"could not read staged EvalLog header: {path}: {exc}") from exc
    evaluation = raw_preflight._attribute(header, "eval")
    task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
    if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
        continue
    identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
    if identity in seen:
        raise SystemExit(f"duplicate staged task identity {identity}: {path} and {seen[identity][0]}")
    seen[identity] = (path, raw_preflight._attribute(header, "status"), header)

target_identity = raw_preflight._task_identity(target)
if target_identity in seen:
    raise SystemExit(f"target task identity already exists in this root: {target_identity}")
clean_identity = raw_preflight._task_identity(clean_spec)
if seen.get(clean_identity, (None, None, None))[1] != "success":
    raise SystemExit(f"missing successful staged clean reference: {clean_identity}")
path, _, header = seen[clean_identity]
created = raw_preflight._validate_header(header, path=path, spec=clean_spec, raw_root=raw_root)
full = raw_preflight._read_eval_log(path, header_only=False)
loaded = raw_preflight.LoadedTaskLog(clean_spec, path, created, header, "", {})
raw_preflight._validate_samples(full, loaded=loaded, clean_paths={})
_assert_expected_hf_peft_model(
    path,
    base_model="Qwen/Qwen3.5-9B",
    checkpoint=str(checkpoint),
    max_connections=8,
)
print(
    "fanout_preflight_ok "
    f"task_index={task_index} target={target_identity} clean_reference={clean_identity}"
)
PY

"$PY" - "$CLAIM" "$FROZEN" "$RAW_ROOT" "$CHECKPOINT" "$CONDITION" "$TASK_INDEX" "$GPU" "$DISABLE_CUDNN_SDP" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

claim, frozen, raw_root, checkpoint, condition, task_index, gpu, disable_cudnn_sdp = sys.argv[1:]
payload = {
    "schema": "stage2-ood-singleton-fanout-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "condition": condition,
    "task_index": int(task_index),
    "gpu": int(gpu),
    "manifest": frozen,
    "manifest_sha256": hashlib.sha256(Path(frozen).read_bytes()).hexdigest(),
    "raw_root": raw_root,
    "checkpoint": checkpoint,
    "generation": {"max_connections": 8, "max_tokens": 20480, "temperature": 1.0, "top_k": 20, "top_p": 0.95},
    "attention_backend": {"cudnn_sdp_disabled": disable_cudnn_sdp == "1"},
}
descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

TASK_ARGS=$("$PY" -c 'import json,sys; print(json.dumps({"manifest":sys.argv[1],"unbiased_log":sys.argv[2],"prompt_style":"none","include_bias_acknowledged":False},sort_keys=True))' "$FROZEN" "$RAW_ROOT")
nohup env -u VLLM_BASE_URL -u VLLM_API_KEY -u CTM_PERSISTENT_VLLM_SERVER_METADATA \
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO" HF_HOME="$HF_HOME_DIR" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 CTM_DISABLE_CUDNN_SDP="$DISABLE_CUDNN_SDP" \
  "$PY" scripts/run_evals.py --task-factory experiments.stage2_ood_hle.tasks:ood_tasks \
  --local-checkpoint "$CHECKPOINT" --base-model Qwen/Qwen3.5-9B \
  --task-args "$TASK_ARGS" --model-args '{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}' \
  --generation-config '{"max_connections":8,"max_tokens":20480,"temperature":1.0,"top_k":20,"top_p":0.95}' \
  --log-dir "$RAW_ROOT" --max-tasks 1 --yes --task-index "$TASK_INDEX" >"$LOG" 2>&1 &
worker_pid=$!
sleep 2

worker_attestation=$("$PY" - "$worker_pid" "$PY" "$CHECKPOINT" "$RAW_ROOT" "$FROZEN" "$TASK_INDEX" "$GPU" "$REPO" "$HF_HOME_DIR" "$DISABLE_CUDNN_SDP" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

pid, expected_python, checkpoint, raw_root, manifest, task_index, gpu, repo, hf_home, disable_cudnn_sdp = sys.argv[1:]
proc = Path("/proc") / pid
if not proc.is_dir():
    raise SystemExit(f"fan-out worker vanished: {pid}")
stat = (proc / "stat").read_text(encoding="utf-8")
tail = stat.rsplit(")", 1)[1].split()
if len(tail) < 20 or tail[0] in {"Z", "X", "T", "t"}:
    raise SystemExit(f"fan-out worker is not live: {pid}")
cmdline = (proc / "cmdline").read_bytes()
tokens = [item.decode("utf-8", "surrogateescape") for item in cmdline.split(b"\0") if item]
if len(tokens) < 2 or tokens[0] != expected_python or not (tokens[1] == "scripts/run_evals.py" or tokens[1].endswith("/scripts/run_evals.py")):
    raise SystemExit(f"fan-out worker command does not match: {tokens[:2]!r}")

def values(flag: str) -> list[str]:
    found: list[str] = []
    for index, token in enumerate(tokens):
        if token == flag:
            if index + 1 >= len(tokens):
                raise SystemExit(f"worker has incomplete {flag}")
            found.append(tokens[index + 1])
    return found

if values("--task-index") != [task_index] or values("--local-checkpoint") != [checkpoint] or values("--log-dir") != [raw_root]:
    raise SystemExit("fan-out worker task/checkpoint/raw-root contract differs from the claim")
if values("--task-factory") != ["experiments.stage2_ood_hle.tasks:ood_tasks"] or values("--max-tasks") != ["1"]:
    raise SystemExit("fan-out worker task factory or max-tasks differs from the contract")
try:
    model_args = json.loads(values("--model-args")[0])
    generation = json.loads(values("--generation-config")[0])
    task_args = json.loads(values("--task-args")[0])
except (IndexError, json.JSONDecodeError) as exc:
    raise SystemExit("fan-out worker has unreadable JSON arguments") from exc
if model_args != {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"}:
    raise SystemExit(f"unexpected model args: {model_args!r}")
if generation != {"max_connections": 8, "max_tokens": 20480, "temperature": 1.0, "top_k": 20, "top_p": 0.95}:
    raise SystemExit(f"unexpected generation args: {generation!r}")
if task_args != {"include_bias_acknowledged": False, "manifest": manifest, "prompt_style": "none", "unbiased_log": raw_root}:
    raise SystemExit(f"unexpected task args: {task_args!r}")
environment = {}
for item in (proc / "environ").read_bytes().split(b"\0"):
    if b"=" in item:
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
expected_environment = {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": repo, "HF_HOME": hf_home, "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_XET": "1", "CTM_DISABLE_CUDNN_SDP": disable_cudnn_sdp}
for key, value in expected_environment.items():
    if environment.get(key) != value:
        raise SystemExit(f"fan-out worker environment {key} differs from the contract")
print(json.dumps({"pid": int(pid), "state": tail[0], "start_ticks": int(tail[19]), "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(), "cmdline_tokens": tokens}, sort_keys=True))
PY
)

"$PY" - "$RESULT" "$CLAIM" "$worker_attestation" <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

result, claim, worker = sys.argv[1:]
payload = {
    "schema": "stage2-ood-singleton-fanout-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "state": "launched",
    "claim": claim,
    "worker": json.loads(worker),
}
descriptor = os.open(result, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
printf 'fanout singleton launched: condition=%s task=%s gpu=%s worker=%s\n' "$CONDITION" "$TASK_INDEX" "$GPU" "$worker_pid"
