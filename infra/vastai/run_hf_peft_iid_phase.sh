#!/usr/bin/env bash
# Run one phase of the frozen Stage-1 IID diagnostic through Transformers/PEFT.
#
# The control adapter cannot satisfy the independent vLLM runtime-parity gate,
# so this intentionally uses the raw local LoRA checkpoint.  Four logical
# cells run on four isolated GPUs.  Clean cells always precede biased cells so
# each paired-switch scorer sees a complete, unambiguous clean log set.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_hf_peft_iid_phase.sh \
    --checkpoint PATH --log-dir PATH --phase clean|biased [--connections N]
EOF
}

checkpoint=""
log_dir=""
phase=""
connections=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      checkpoint=${2:?missing value for --checkpoint}
      shift 2
      ;;
    --log-dir)
      log_dir=${2:?missing value for --log-dir}
      shift 2
      ;;
    --phase)
      phase=${2:?missing value for --phase}
      shift 2
      ;;
    --connections)
      connections=${2:?missing value for --connections}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[[ -n "$checkpoint" && -n "$log_dir" ]] || { usage >&2; exit 2; }
[[ "$phase" == "clean" || "$phase" == "biased" ]] || { usage >&2; exit 2; }
[[ "$connections" =~ ^[1-9][0-9]*$ ]] || { echo "--connections must be a positive integer" >&2; exit 2; }

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${CTM_PYTHON:-"$repo/../env/bin/python"}
[[ -x "$python_bin" ]] || { echo "missing Python environment: $python_bin" >&2; exit 2; }
[[ -d "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }

cd "$repo"
mkdir -p "$log_dir/runner-logs"

if [[ "$phase" == "clean" ]]; then
  if find "$log_dir" -maxdepth 1 -type f -name '*.eval' -print -quit | grep -q .; then
    echo "refusing to rerun clean phase into an existing log directory: $log_dir" >&2
    exit 1
  fi
  indices=(1 2 3 4)
else
  [[ -f "$log_dir/clean.complete" ]] || { echo "biased phase requires a successful clean phase marker" >&2; exit 1; }
  clean_count=$(find "$log_dir" -maxdepth 1 -type f -name '*stage1-iid-unbiased*.eval' | wc -l | tr -d ' ')
  biased_count=$(find "$log_dir" -maxdepth 1 -type f -name '*stage1-iid-biased*.eval' | wc -l | tr -d ' ')
  [[ "$clean_count" == "4" ]] || { echo "biased phase requires exactly four completed clean logs, found $clean_count" >&2; exit 1; }
  [[ "$biased_count" == "0" ]] || { echo "refusing to rerun biased phase with $biased_count existing biased log(s)" >&2; exit 1; }
  indices=(5 6 7 8)
fi

task_args=$(printf '{"manifest":"artifacts/act-repair-gate-20260731/data/manifest.json","unbiased_log":"%s","include_bias_acknowledged":false}' "$log_dir")
model_args='{"provider":"hf","device":"cuda:0","dtype":"bfloat16"}'
generation_config=$(printf '{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"max_connections":%s}' "$connections")

declare -a pids=()
for offset in "${!indices[@]}"; do
  task_index=${indices[$offset]}
  gpu=$offset
  stdout="$log_dir/runner-logs/${phase}-task-${task_index}.stdout.log"
  (
    CUDA_VISIBLE_DEVICES=$gpu \
      HF_HOME=${HF_HOME:-/workspace/hf-cache-direct} \
      HF_HUB_OFFLINE=1 \
      HF_HUB_DISABLE_XET=1 \
      PYTHONUNBUFFERED=1 \
      PYTHONPATH=. \
      "$python_bin" scripts/run_evals.py \
        --task-factory experiments.stage1_iid_diagnostic.tasks:diagnostic_matrix_tasks \
        --local-checkpoint "$checkpoint" \
        --base-model Qwen/Qwen3.5-9B \
        --task-args "$task_args" \
        --model-args "$model_args" \
        --generation-config "$generation_config" \
        --log-dir "$log_dir" \
        --limit 100 \
        --max-tasks 1 \
        --task-index "$task_index" \
        --yes
  ) >"$stdout" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "started $phase task $task_index on GPU $gpu (pid $pid)"
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "$phase phase failed; completed logs have been preserved for diagnosis." >&2
  exit "$status"
fi

touch "$log_dir/${phase}.complete"
echo "$phase phase complete: $log_dir"
