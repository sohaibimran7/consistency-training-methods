#!/usr/bin/env bash
# Run the completed RMCT-main checkpoint independently on the unused half of
# the RMCT-control host. This is deliberately fail-closed: raw Qwen3.5 LoRA
# weights are used by vLLM only after all four HF↔vLLM parity variants attest.
set -euo pipefail

REPO=/workspace/ctm-eval-none-20260801/repo
PY=/workspace/ctm-act-repair-20260731/env/bin/python
RUN=/workspace/ctm-act-repair-20260731
CONDITION=rmct
ROOT="$REPO/artifacts/stage1-rmct-main-source-b8-20260802"
RAW="$RUN/repo/logs/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801/rate-matching-lr-1e-4/checkpoints/rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801_rate-matching-lr-1e-4"
COMPAT="$ROOT/vllm-compat-adapter"
PARITY="$ROOT/runtime-parity-v1"
EVAL_ROOT="logs/evals/stage1-onpolicy-final-20260802/rmct-source-b8"
STAGE="artifacts/stage1-rmct-main-source-b8-20260802"
FROZEN="artifacts/stage1-iid-diagnostic-none-20260801"

cd "$REPO"
test ! -e "$ROOT"
test ! -e "$EVAL_ROOT"
test -f "$RAW/adapter_model.safetensors"
test -f "$RAW/adapter_config.json"
test -f "$RAW/manifest.json"

# Bind this run to the exact adapter that was handoff-verified on the
# evaluator host. Any unexpected source mutation fails before a GPU is used.
test "$(sha256sum "$RAW/adapter_model.safetensors" | awk '{print $1}')" = \
  f0fb88018b9862e65d4060437d51e186d01703e5d0ae7b0810e131b9d520b02b
test "$(sha256sum "$RAW/adapter_config.json" | awk '{print $1}')" = \
  9a4029043a6b456640b7f229082eca9002fb39dc69e8302c76d8d02e391e177d
test "$(sha256sum "$RAW/manifest.json" | awk '{print $1}')" = \
  1dbe2b297e1474aae23423a84b17eb84683c67194a3f859db815abe5917fc189

mkdir -p "$ROOT" "$STAGE/provenance" "$STAGE/raw"

PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --source "$RAW" \
  --destination "$COMPAT"

CUDA_VISIBLE_DEVICES=4 HF_HOME="$RUN/hf-cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
  "$PY" -m experiments.act_repair_gate.runtime_parity \
    --model Qwen/Qwen3.5-9B \
    --adapter "$COMPAT" \
    --hf-adapter "$RAW" \
    --data "$FROZEN/train-eval-n200.jsonl" \
    --output-dir "$PARITY" \
    --samples 8 \
    --top-token-count 16 \
    --hf-batch-size 2 \
    --vllm-port 8812 \
    --max-model-len 32768 \
    --vllm-memory-utilization 0.90 \
    --enforce-eager \
    --isolate-vllm-variants

if PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --adapter "$COMPAT" \
  --parity-report "$PARITY/report.json"; then
  MODE=vllm
  CHECKPOINT="$COMPAT"
  echo "RMCT main source-b8: final vLLM attestation passed"
else
  test -f "$PARITY/report.json"
  MODE=hf-peft
  CHECKPOINT="$RAW"
  echo "RMCT main source-b8: final vLLM parity mismatch; using native HF/PEFT"
fi

launch_pair() {
  local gpu=$1
  local pair=$2
  local split=$3
  local clean_index=$4
  local biased_index=$5
  local task_args
  task_args=$(printf '{"manifest":"%s","split":"%s","unbiased_log":"%s","prompt_style":"none","include_bias_acknowledged":false}' \
    "$FROZEN/manifest.json" "$split" "$EVAL_ROOT/$pair")

  if [ "$MODE" = vllm ]; then
    nohup env CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$RUN/hf-cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
      "$PY" scripts/run_evals.py \
        --task-factory experiments.stage1_iid_diagnostic_none.tasks:diagnostic_tasks \
        --local-checkpoint "$CHECKPOINT" \
        --base-model Qwen/Qwen3.5-9B \
        --task-args "$task_args" \
        --model-args '{"provider":"vllm","gpu_memory_utilization":0.90,"language_model_only":true,"max_model_len":32768,"max_num_seqs":256}' \
        --generation-config '{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"extra_body":{"top_k":20}}' \
        --log-dir "$EVAL_ROOT/$pair" --limit 100 --max-tasks 1 --isolate-tasks --persistent-vllm-server \
        --task-index "$clean_index" --task-index "$biased_index" --yes \
        > "$RUN/runners/rmct-main-source-b8-vllm-${pair}-20260802.log" 2>&1 < /dev/null &
  else
    nohup env CUDA_VISIBLE_DEVICES="$gpu" HF_HOME="$RUN/hf-cache" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 PYTHONPATH=. \
      "$PY" scripts/run_evals.py \
        --task-factory experiments.stage1_iid_diagnostic_none.tasks:diagnostic_tasks \
        --local-checkpoint "$CHECKPOINT" \
        --base-model Qwen/Qwen3.5-9B \
        --task-args "$task_args" \
        --model-args '{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}' \
        --generation-config '{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"max_connections":8}' \
        --log-dir "$EVAL_ROOT/$pair" --limit 100 --max-tasks 1 --isolate-tasks \
        --task-index "$clean_index" --task-index "$biased_index" --yes \
        > "$RUN/runners/rmct-main-source-b8-hf-${pair}-20260802.log" 2>&1 < /dev/null &
  fi
}

launch_pair 4 train-logiqa train_eval 1 3
launch_pair 5 train-hellaswag train_eval 2 4
launch_pair 6 heldout-logiqa heldout_in_domain 1 3
launch_pair 7 heldout-hellaswag heldout_in_domain 2 4

completed=false
for attempt in $(seq 1 1800); do
  if "$PY" - "$EVAL_ROOT" <<'PY'
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log

root = Path(sys.argv[1])
pairs = ("train-logiqa", "train-hellaswag", "heldout-logiqa", "heldout-hellaswag")
complete = True
for pair in pairs:
    logs = sorted((root / pair).glob("*.eval"))
    observed = []
    for path in logs:
        try:
            log = read_eval_log(path)
            observed.append((log.status, len(log.samples or [])))
        except Exception as exc:
            observed.append((type(exc).__name__, 0))
    print(f"{pair}={observed}", file=sys.stderr)
    if any(status in {"error", "cancelled"} for status, _ in observed):
        raise RuntimeError(f"failed RMCT-main cell: {pair}={observed}")
    if len(logs) != 2 or any(status != "success" or count != 100 for status, count in observed):
        complete = False
raise SystemExit(0 if complete else 1)
PY
  then
    completed=true
    echo "raw generation complete: $CONDITION ($MODE)"
    break
  fi
  sleep 20
done
if [ "$completed" != true ]; then
  echo "timed out waiting for RMCT-main raw generation" >&2
  exit 1
fi

PREFLIGHT="$STAGE/provenance/$CONDITION.raw-preflight.json"
PREFLIGHT_ARGS=(
  --raw-log-root "$EVAL_ROOT"
  --manifest "$FROZEN/manifest.json"
  --split-file "train_eval=$FROZEN/train-eval-n200.jsonl"
  --split-file "heldout_in_domain=$FROZEN/heldout-in-domain-n200.jsonl"
  --condition "$CONDITION"
  --expected-base-model Qwen/Qwen3.5-9B
  --expected-checkpoint "$CHECKPOINT"
  --runtime-profile "$MODE"
  --output "$PREFLIGHT"
)
if [ "$MODE" = hf-peft ]; then
  PREFLIGHT_ARGS+=(--expected-max-connections 8)
fi
PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic_none.raw_preflight "${PREFLIGHT_ARGS[@]}"

"$PY" - "$PREFLIGHT" "$STAGE/raw" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

report_path, raw_root = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
for source in report["sources"]:
    origin = Path(source["raw_log"])
    expected = source["raw_log_sha256"]
    if hashlib.sha256(origin.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"source changed after preflight: {origin}")
    destination = raw_root / report["condition"] / source["split"] / origin.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise FileExistsError(f"refusing to replace mismatching staged log: {destination}")
    else:
        shutil.copy2(origin, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"staged raw log hash mismatch: {destination}")
    print(destination)
PY

set -a
. "$RUN/repo/.env"
set +a
PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic_none.grade_luna \
  --raw-log-root "$STAGE/raw" \
  --preflight-report "$PREFLIGHT" \
  --output-root "$STAGE/graded-rmct" \
  --workers 5 --connections-per-worker 100 --grader-max-tokens 1024

PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic.analyze \
  --graded-root "$STAGE/graded-rmct" \
  --manifest "$FROZEN/manifest.json" \
  --grader-max-tokens 1024 \
  --output "$STAGE/analysis/rmct-final.json"
