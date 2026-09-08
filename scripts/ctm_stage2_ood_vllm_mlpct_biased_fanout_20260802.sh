#!/usr/bin/env bash
# Finish the already-started MLPCT Stage-2 OOD condition after its three clean
# cells are complete.  The original persistent-vLLM parent is deliberately
# held stopped at the clean barrier; this launcher owns three *new* servers on
# otherwise idle GPUs for disjoint biased task groups.  It never regenerates a
# clean cell, changes a prompt, or emits the canonical completion marker until
# the complete merged 21-cell tree has passed raw preflight.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1}
RUN_ROOT=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
CONDITION=${CTM_OOD_CONDITION:-mlpct-vllm-compat}
ADAPTER=${CTM_OOD_ADAPTER:-/workspace/ctm-eval-none-20260801/repo/artifacts/stage1-supervised-parity-recovery-20260802/vllm-compat-adapters/mlpct}
ORIGINAL_PARENT_PID=${CTM_OOD_ORIGINAL_PARENT_PID:?CTM_OOD_ORIGINAL_PARENT_PID is required}
ORIGINAL_PGID=${CTM_OOD_ORIGINAL_PGID:?CTM_OOD_ORIGINAL_PGID is required}

readonly BASE_MODEL=Qwen/Qwen3.5-9B
readonly TASK_FACTORY=experiments.stage2_ood_hle.tasks:ood_tasks
readonly RAW_ROOT="$RUN_ROOT/logs/$CONDITION"
readonly SHARD_ROOT="$RAW_ROOT/shards"
readonly RUNNER_DIR="$RUN_ROOT/runners"
readonly LAUNCH_LOG="$RUNNER_DIR/$CONDITION-biased-fanout.log"
readonly PREFLIGHT_OUTPUT="$RUN_ROOT/artifacts/fanout-preflight-v1/$CONDITION.json"
readonly MODEL_ARGS='{"provider":"vllm","gpu_memory_utilization":0.9,"language_model_only":true,"max_model_len":32768,"max_num_seqs":256}'
readonly GENERATION_CONFIG='{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"extra_body":{"top_k":20}}'

fail() {
  printf '%s\n' "ERROR: $*" >&2
  exit 2
}

log() {
  printf '%s\n' "$*" | tee -a "$LAUNCH_LOG"
}

case "$CONDITION" in
  mlpct-vllm-compat) ;;
  *) fail "this recovery launcher is pinned to mlpct-vllm-compat" ;;
esac
case "$ORIGINAL_PARENT_PID" in ''|*[!0-9]*) fail "CTM_OOD_ORIGINAL_PARENT_PID must be numeric" ;; esac
case "$ORIGINAL_PGID" in ''|*[!0-9]*) fail "CTM_OOD_ORIGINAL_PGID must be numeric" ;; esac

test -x "$PY"
test -f "$FROZEN/manifest.json"
test -d "$ADAPTER"
test -d "$RAW_ROOT"
test ! -e "$LAUNCH_LOG"
test ! -e "$PREFLIGHT_OUTPUT"
mkdir -p "$RUNNER_DIR" "$SHARD_ROOT" "$(dirname "$PREFLIGHT_OUTPUT")"

export PYTHONPATH="$REPO"
export HF_HOME=${CTM_OOD_HF_HOME:-/workspace/ctm-act-repair-20260731/hf-cache}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
unset VLLM_BASE_URL VLLM_API_KEY
cd "$REPO"

# The original parent is blocked in subprocess.run() on clean task 3.  Its
# stopped state is a reversible, race-free barrier: the child and its vLLM
# server continue, but the parent cannot spawn task 4 while shards are staged.
test -d "/proc/$ORIGINAL_PARENT_PID" || fail "original MLPCT parent is no longer live"
PARENT_STATE=$(ps -o stat= -p "$ORIGINAL_PARENT_PID" | tr -d '[:space:]')
case "$PARENT_STATE" in T*) ;; *) fail "original MLPCT parent is not stopped at the clean barrier (state=$PARENT_STATE)" ;; esac
OBSERVED_PGID=$(ps -o pgid= -p "$ORIGINAL_PARENT_PID" | tr -d '[:space:]')
[ "$OBSERVED_PGID" = "$ORIGINAL_PGID" ] || fail "original parent PGID changed: expected=$ORIGINAL_PGID observed=$OBSERVED_PGID"
SELF_PGID=$(ps -o pgid= -p "$$" | tr -d '[:space:]')
[ "$SELF_PGID" != "$ORIGINAL_PGID" ] || fail "fanout launcher unexpectedly shares the original condition process group"

# Revalidate both the frozen matrix and the exact immutable compatibility
# adapter before allocating the idle GPUs.
"$PY" - "$FROZEN/manifest.json" "$ADAPTER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from ctm.evals.qwen35_vllm_attestation import is_verified_qwen35_vllm_compat_adapter
from experiments.stage2_ood_hle.materialize import validate_manifest
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest_path, adapter = sys.argv[1:]
manifest = validate_manifest(manifest_path)
specs = ood_task_specs(manifest_path)
if len(specs) != 21 or sum(spec.kind == "unbiased" for spec in specs) != 3:
    raise SystemExit("unexpected Stage 2 OOD task matrix")
if not is_verified_qwen35_vllm_compat_adapter(adapter):
    raise SystemExit(f"adapter lacks valid immutable Qwen3.5 vLLM parity evidence: {adapter}")
print(json.dumps({
    "condition": "mlpct-vllm-compat",
    "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
    "tasks": len(specs),
    "clean_tasks": sum(spec.kind == "unbiased" for spec in specs),
    "biased_tasks": sum(spec.kind == "biased" for spec in specs),
}, sort_keys=True))
PY

# Do not infer the barrier from a filename: Inspect must have written three
# successful, exact clean headers and no biased output yet.
"$PY" - "$FROZEN/manifest.json" "$RAW_ROOT" <<'PY'
from pathlib import Path
import sys

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest = Path(sys.argv[1]).resolve()
raw_root = Path(sys.argv[2]).resolve()
expected = raw_preflight._expected_cell_specs(ood_task_specs(manifest))
clean = {identity: spec for identity, spec in expected.items() if spec.kind == "unbiased"}
seen = {}
for path in raw_preflight._discover_eval_log_paths(raw_root):
    log = raw_preflight._read_eval_log(path, header_only=True)
    if raw_preflight._attribute(log, "status") != "success":
        raise SystemExit(f"clean barrier contains a non-success EvalLog: {path}")
    evaluation = raw_preflight._attribute(log, "eval")
    task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
    if task_name != raw_preflight.TASK_UNBIASED:
        raise SystemExit(f"biased EvalLog appeared before the MLPCT fanout barrier: {path}")
    identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
    spec = clean.get(identity)
    if spec is None:
        raise SystemExit(f"unexpected clean EvalLog before fanout: {path}")
    raw_preflight._validate_header(log, path=path, spec=spec, raw_root=raw_root)
    if identity in seen:
        raise SystemExit(f"duplicate clean EvalLog before fanout: {path} and {seen[identity]}")
    seen[identity] = path
if set(seen) != set(clean):
    missing = sorted(set(clean) - set(seen))
    raise SystemExit(f"clean barrier is incomplete; missing={missing}")
print(f"verified MLPCT clean barrier: {len(seen)} exact clean cells")
PY

# These groups contain six distinct task indices each and two HLE-biased cells
# apiece.  They are balanced by the frozen 100-question cell size, while the
# HLE-heavy decode tail is spread over all three persistent-vLLM owners.
declare -a GPUS=(3 4 6)
declare -a GROUP_3=(6 17 4 7 8 9)
declare -a GROUP_4=(18 19 5 10 11 12)
declare -a GROUP_6=(20 21 13 14 15 16)

declare -a ALL_TASKS=("${GROUP_3[@]}" "${GROUP_4[@]}" "${GROUP_6[@]}")
[ "${#ALL_TASKS[@]}" -eq 18 ] || fail "internal task fanout has the wrong size"
declare -A SEEN_TASKS=()
for task_index in "${ALL_TASKS[@]}"; do
  case "$task_index" in 4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21) ;; *) fail "invalid biased task index $task_index" ;; esac
  [ -z "${SEEN_TASKS[$task_index]+x}" ] || fail "duplicate biased task index $task_index"
  SEEN_TASKS[$task_index]=1
done
[ "${#SEEN_TASKS[@]}" -eq 18 ] || fail "not all biased task indices were assigned"

for gpu in "${GPUS[@]}"; do
  running=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  if [ -n "$running" ] && [ "$running" != "No running processes found" ]; then
    fail "GPU $gpu is not idle; refusing to collide with an existing compute process: $running"
  fi
done

TASK_ARGS=$("$PY" - "$FROZEN/manifest.json" "$RAW_ROOT" <<'PY'
import json
import sys

print(json.dumps({
    "manifest": sys.argv[1],
    "unbiased_log": sys.argv[2],
    "prompt_style": "none",
    "include_bias_acknowledged": False,
}, sort_keys=True))
PY
)

BASE_ARGS=(
  scripts/run_evals.py
  --task-factory "$TASK_FACTORY"
  --local-checkpoint "$ADAPTER"
  --base-model "$BASE_MODEL"
  --task-args "$TASK_ARGS"
  --model-args "$MODEL_ARGS"
  --generation-config "$GENERATION_CONFIG"
  --max-tasks 1
  --isolate-tasks
  --persistent-vllm-server
  --yes
)

declare -a PIDS=()
launch_shard() {
  local gpu=$1
  shift
  local label=
  local task_index
  local shard_dir="$SHARD_ROOT/gpu$gpu"
  local shard_log="$RUNNER_DIR/$CONDITION-biased-fanout-gpu$gpu.log"
  local -a task_args=()
  for task_index in "$@"; do
    label=${label:+$label-}$task_index
    task_args+=(--task-index "$task_index")
  done
  test ! -e "$shard_log" || fail "refusing to overwrite shard runner log: $shard_log"
  mkdir -p "$shard_dir"
  if find "$shard_dir" -type f -name '*.eval' -print -quit | grep -q .; then
    fail "refusing to reuse nonempty shard log directory: $shard_dir"
  fi
  log "started MLPCT biased fanout gpu=$gpu task_indices=$label log_dir=$shard_dir runner_log=$shard_log"
  (
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${BASE_ARGS[@]}" --log-dir "$shard_dir" "${task_args[@]}"
  ) >"$shard_log" 2>&1 &
  PIDS+=("$!")
}

launch_shard 3 "${GROUP_3[@]}"
launch_shard 4 "${GROUP_4[@]}"
launch_shard 6 "${GROUP_6[@]}"

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  log "MLPCT biased fanout failed; completed raw EvalLogs are preserved and no completion marker was written."
  exit "$status"
fi
log "all MLPCT biased fanout shards exited successfully"

# A complete raw-preflight gate proves that the three original clean logs and
# all 18 nested shard logs form exactly one valid frozen 21-cell matrix before
# either the old owner is terminated or downstream Luna work is authorized.
"$PY" -m experiments.stage2_ood_hle.raw_preflight \
  --raw-log-root "$RAW_ROOT" \
  --manifest "$FROZEN/manifest.json" \
  --condition "$CONDITION" \
  --runtime-profile vllm \
  --expected-checkpoint "$ADAPTER" \
  --output "$PREFLIGHT_OUTPUT"
log "verified complete MLPCT fanout raw-preflight: $PREFLIGHT_OUTPUT"

# The old stopped process group owns GPU 5 and would otherwise remain a latent
# task-4 producer.  Re-check its identity *after* the complete raw-preflight:
# a SIGTERM sent to a stopped parent remains pending until SIGCONT, which would
# let that parent spawn task 4.  SIGKILL takes effect while stopped, so it
# tears down only the verified old condition group without resuming it.
test -d "/proc/$ORIGINAL_PARENT_PID" || fail "original MLPCT parent disappeared before SIGKILL"
PARENT_STATE=$(ps -o stat= -p "$ORIGINAL_PARENT_PID" | tr -d '[:space:]')
case "$PARENT_STATE" in T*) ;; *) fail "original MLPCT parent is no longer stopped before SIGKILL (state=$PARENT_STATE)" ;; esac
OBSERVED_PGID=$(ps -o pgid= -p "$ORIGINAL_PARENT_PID" | tr -d '[:space:]')
[ "$OBSERVED_PGID" = "$ORIGINAL_PGID" ] || fail "original parent PGID changed before SIGKILL: expected=$ORIGINAL_PGID observed=$OBSERVED_PGID"
SELF_PGID=$(ps -o pgid= -p "$$" | tr -d '[:space:]')
[ "$SELF_PGID" != "$ORIGINAL_PGID" ] || fail "fanout launcher unexpectedly shares the original condition process group before SIGKILL"
kill -KILL -- "-$ORIGINAL_PGID"
for ((attempt = 0; attempt < 30; attempt++)); do
  if ! ps -eo pgid= | awk -v group="$ORIGINAL_PGID" '$1 == group { found=1 } END { exit !found }'; then
    break
  fi
  sleep 1
done
if ps -eo pgid= | awk -v group="$ORIGINAL_PGID" '$1 == group { found=1 } END { exit !found }'; then
  fail "original MLPCT process group $ORIGINAL_PGID survived SIGKILL; completion marker withheld"
fi
gpu5_running=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 5 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
if [ -n "$gpu5_running" ] && [ "$gpu5_running" != "No running processes found" ]; then
  fail "GPU 5 still has a compute process after old MLPCT teardown: $gpu5_running"
fi

printf '%s\n' "Stage 2 OOD vLLM raw condition complete: $CONDITION" | tee -a "$LAUNCH_LOG"
