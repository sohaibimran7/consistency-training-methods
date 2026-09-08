#!/usr/bin/env bash
# Rebuild the missing BCT-control no-CoT evidence using the native HF/PEFT
# evaluator. Unlike BCT-main, this adapter's translated vLLM effects did not
# match HF, so this is deliberately the authoritative backend for the control.
set -euo pipefail

# A fresh host may use a different staging root. Keep every location
# overrideable, but require absolute paths below so the paired-log contract
# cannot accidentally construct a doubled or relative path.
HOST_ROOT=${CTM_BCT_HOST_ROOT:-/workspace}
REPO=${BCT_CONTROL_REPO:-"$HOST_ROOT/ctm-eval-none-20260801/repo"}
RUN=${BCT_CONTROL_RUN:-"$HOST_ROOT/ctm-act-repair-20260731"}
PY=${BCT_CONTROL_PY:-"$RUN/env/bin/python"}
FROZEN=${BCT_CONTROL_FROZEN:-"$REPO/artifacts/stage1-iid-diagnostic-none-20260801"}
CHECKPOINT=${BCT_CONTROL_CHECKPOINT:-"$REPO/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct-control"}
CONDITION=bct-control-hf-peft
# Exact original PEFT checkpoint identity. The control is intentionally
# evaluated natively because its translated vLLM effect did not match HF.
ADAPTER_MODEL_SHA256=84efdc5b38f488ad49ef3659518408ec4f93cf624bf6f30b535cbcf897ad19f5
ADAPTER_CONFIG_SHA256=a3814737831811f7b46f7df475cb220bec20c80f803f1af6c0ce9cd7d98f88ac
CHECKPOINT_MANIFEST_SHA256=8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d
HF_HOME_ROOT=${BCT_CONTROL_HF_HOME:-"$RUN/hf-cache"}
EVAL_ROOT=${BCT_CONTROL_EVAL_ROOT:-"$REPO/logs/evals/stage1-bct-control-hf-recovery-20260802"}
STAGE=${BCT_CONTROL_STAGE:-"$REPO/artifacts/stage1-bct-control-hf-recovery-20260802"}
RUNNER_ROOT=${BCT_CONTROL_RUNNER_ROOT:-"$RUN/runners/bct-control-hf-recovery-20260802"}
MODE=${BCT_CONTROL_MODE:-all}

GPU_TRAIN_LOGIQA=${BCT_CONTROL_GPU_TRAIN_LOGIQA:-0}
GPU_TRAIN_HELLASWAG=${BCT_CONTROL_GPU_TRAIN_HELLASWAG:-1}
GPU_HELDOUT_LOGIQA=${BCT_CONTROL_GPU_HELDOUT_LOGIQA:-2}
GPU_HELDOUT_HELLASWAG=${BCT_CONTROL_GPU_HELDOUT_HELLASWAG:-3}

for path in "$REPO" "$RUN" "$PY" "$FROZEN" "$CHECKPOINT" "$HF_HOME_ROOT" "$EVAL_ROOT" "$STAGE" "$RUNNER_ROOT"; do
  case "$path" in
    /*) ;;
    *) echo "BCT-control paths must be absolute; got $path" >&2; exit 2 ;;
  esac
done

cd "$REPO"
for path in \
  "$CHECKPOINT/adapter_model.safetensors" \
  "$CHECKPOINT/adapter_config.json" \
  "$CHECKPOINT/manifest.json" \
  "$FROZEN/manifest.json" \
  "$FROZEN/train-eval-n200.jsonl" \
  "$FROZEN/heldout-in-domain-n200.jsonl"; do
  test -f "$path"
done
case "$MODE" in
  all|raw-pair|finalize) ;;
  *)
    echo "BCT_CONTROL_MODE must be all, raw-pair, or finalize; got $MODE" >&2
    exit 2
    ;;
esac

launch_pair() {
  local pair=$1
  local gpu=$2
  local split=$3
  local indices=$4
  local task_args
  task_args=$(printf '{"manifest":"%s/manifest.json","split":"%s","unbiased_log":"%s/%s","prompt_style":"none","include_bias_acknowledged":false}' "$FROZEN" "$split" "$EVAL_ROOT" "$pair")
  CUDA_VISIBLE_DEVICES="$gpu" \
    HF_HOME="$HF_HOME_ROOT" \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    PYTHONPATH=. \
    "$PY" scripts/run_evals.py \
      --task-factory experiments.stage1_iid_diagnostic_none.tasks:diagnostic_tasks \
      --local-checkpoint "$CHECKPOINT" \
      --base-model Qwen/Qwen3.5-9B \
      --task-args "$task_args" \
      --model-args '{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}' \
      --generation-config '{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"max_connections":8}' \
      --log-dir "$EVAL_ROOT/$pair" \
      --limit 100 \
      --max-tasks 1 \
      --isolate-tasks \
      $indices \
      --yes \
      > "$RUNNER_ROOT/$pair.log" 2>&1 < /dev/null &
  LAUNCHED_PID=$!
}

if [ "$MODE" = all ]; then
  test ! -e "$EVAL_ROOT"
  test ! -e "$STAGE"
  mkdir -p "$RUNNER_ROOT"
  launch_pair train-logiqa "$GPU_TRAIN_LOGIQA" train_eval '--task-index 1 --task-index 3'
  train_logiqa_pid=$LAUNCHED_PID
  launch_pair train-hellaswag "$GPU_TRAIN_HELLASWAG" train_eval '--task-index 2 --task-index 4'
  train_hellaswag_pid=$LAUNCHED_PID
  launch_pair heldout-logiqa "$GPU_HELDOUT_LOGIQA" heldout_in_domain '--task-index 1 --task-index 3'
  heldout_logiqa_pid=$LAUNCHED_PID
  launch_pair heldout-hellaswag "$GPU_HELDOUT_HELLASWAG" heldout_in_domain '--task-index 2 --task-index 4'
  heldout_hellaswag_pid=$LAUNCHED_PID
  printf '%s\n' "train-logiqa=$train_logiqa_pid train-hellaswag=$train_hellaswag_pid heldout-logiqa=$heldout_logiqa_pid heldout-hellaswag=$heldout_hellaswag_pid"

  failed=0
  for pid in "$train_logiqa_pid" "$train_hellaswag_pid" "$heldout_logiqa_pid" "$heldout_hellaswag_pid"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "one or more BCT-control HF raw-generation workers failed; logs are preserved" >&2
    exit 1
  fi
elif [ "$MODE" = raw-pair ]; then
  PAIR=${BCT_CONTROL_PAIR:?BCT_CONTROL_PAIR is required with BCT_CONTROL_MODE=raw-pair}
  GPU=${BCT_CONTROL_PAIR_GPU:?BCT_CONTROL_PAIR_GPU is required with BCT_CONTROL_MODE=raw-pair}
  test ! -e "$EVAL_ROOT/$PAIR"
  mkdir -p "$RUNNER_ROOT"
  case "$PAIR" in
    train-logiqa)
      launch_pair "$PAIR" "$GPU" train_eval '--task-index 1 --task-index 3'
      ;;
    train-hellaswag)
      launch_pair "$PAIR" "$GPU" train_eval '--task-index 2 --task-index 4'
      ;;
    heldout-logiqa)
      launch_pair "$PAIR" "$GPU" heldout_in_domain '--task-index 1 --task-index 3'
      ;;
    heldout-hellaswag)
      launch_pair "$PAIR" "$GPU" heldout_in_domain '--task-index 2 --task-index 4'
      ;;
    *)
      echo "unknown BCT control raw pair: $PAIR" >&2
      exit 2
      ;;
  esac
  pair_pid=$LAUNCHED_PID
  printf '%s\n' "$PAIR=$pair_pid"
  wait "$pair_pid"
  echo "BCT-control raw pair complete: $PAIR"
  exit 0
else
  test -d "$EVAL_ROOT"
  test ! -e "$STAGE"
fi

mkdir -p "$STAGE/provenance"
PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic_none.raw_preflight \
  --raw-log-root "$EVAL_ROOT" \
  --manifest "$FROZEN/manifest.json" \
  --split-file "train_eval=$FROZEN/train-eval-n200.jsonl" \
  --split-file "heldout_in_domain=$FROZEN/heldout-in-domain-n200.jsonl" \
  --condition "$CONDITION" \
  --expected-base-model Qwen/Qwen3.5-9B \
  --expected-checkpoint "$CHECKPOINT" \
  --runtime-profile hf-peft \
  --expected-max-connections 8 \
  --expected-adapter-model-sha256 "$ADAPTER_MODEL_SHA256" \
  --expected-adapter-config-sha256 "$ADAPTER_CONFIG_SHA256" \
  --expected-checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA256" \
  --output "$STAGE/provenance/$CONDITION.raw-preflight.json"

"$PY" - "$STAGE/provenance/$CONDITION.raw-preflight.json" "$STAGE/raw" <<'PY'
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

echo "BCT-control HF recovery analysis complete: $STAGE/analysis/$CONDITION.json"
