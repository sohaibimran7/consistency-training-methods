#!/usr/bin/env bash
# Rebuild the missing immutable BCT-main no-CoT evaluation evidence on the
# existing Stage-1 host. The trained raw adapter is never edited: vLLM receives
# a translated copy only after four-variant HF/vLLM parity attestation.
set -euo pipefail

REPO=/workspace/ctm-eval-none-20260801/repo
PY=/workspace/ctm-act-repair-20260731/env/bin/python
RUN=/workspace/ctm-act-repair-20260731
FROZEN=$REPO/artifacts/stage1-iid-diagnostic-none-20260801
RAW=$REPO/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct
STAGE=$REPO/artifacts/stage1-bct-main-vllm-recovery-20260802
COMPAT=$STAGE/vllm-compat-adapter
PARITY=$STAGE/runtime-parity
EVAL_ROOT=logs/evals/stage1-bct-main-vllm-recovery-20260802
CONDITION=bct-main-vllm-compat
GPU=${BCT_MAIN_GPU:-4}
VLLM_PORT=${BCT_MAIN_VLLM_PORT:-8804}

cd "$REPO"
for path in \
  "$RAW/adapter_model.safetensors" \
  "$RAW/adapter_config.json" \
  "$RAW/manifest.json" \
  "$FROZEN/manifest.json" \
  "$FROZEN/train-eval-n200.jsonl" \
  "$FROZEN/heldout-in-domain-n200.jsonl"; do
  test -f "$path"
done
test ! -e "$COMPAT"
test ! -e "$PARITY"
test ! -e "$EVAL_ROOT"
test ! -e "$STAGE/raw"
test ! -e "$STAGE/graded"
test ! -e "$STAGE/analysis"

mkdir -p "$STAGE"
PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --source "$RAW" \
  --destination "$COMPAT"

CUDA_VISIBLE_DEVICES="$GPU" \
  HF_HOME="$RUN/hf-cache" \
  HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_XET=1 \
  PYTHONPATH=. \
  "$PY" -m experiments.act_repair_gate.runtime_parity \
    --model Qwen/Qwen3.5-9B \
    --adapter "$COMPAT" \
    --hf-adapter "$RAW" \
    --data "$FROZEN/train-eval-n200.jsonl" \
    --output-dir "$PARITY" \
    --samples 8 \
    --top-token-count 16 \
    --hf-batch-size 2 \
    --vllm-port "$VLLM_PORT" \
    --max-model-len 32768 \
    --vllm-memory-utilization 0.90 \
    --enforce-eager \
    --isolate-vllm-variants

PYTHONPATH=. "$PY" -m experiments.act_repair_gate.vllm_compat_adapter \
  --adapter "$COMPAT" \
  --parity-report "$PARITY/report.json"
test -f "$COMPAT/vllm-parity-attestation.json"

unset VLLM_BASE_URL CTM_PERSISTENT_VLLM_SERVER_METADATA
CUDA_VISIBLE_DEVICES="$GPU" \
  HF_HOME="$RUN/hf-cache" \
  HF_HUB_OFFLINE=1 \
  HF_HUB_DISABLE_XET=1 \
  PYTHONPATH=. \
  "$PY" scripts/run_evals.py \
    --task-factory experiments.stage1_iid_diagnostic_none.tasks:diagnostic_matrix_tasks \
    --local-checkpoint "$COMPAT" \
    --base-model Qwen/Qwen3.5-9B \
    --task-args "{\"manifest\":\"$FROZEN/manifest.json\",\"unbiased_log\":\"$REPO/$EVAL_ROOT\",\"prompt_style\":\"none\",\"include_bias_acknowledged\":false}" \
    --model-args '{"provider":"vllm","gpu_memory_utilization":0.9,"max_model_len":32768,"language_model_only":true,"max_num_seqs":256}' \
    --generation-config '{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"extra_body":{"top_k":20}}' \
    --log-dir "$EVAL_ROOT" \
    --limit 100 \
    --max-tasks 1 \
    --isolate-tasks \
    --persistent-vllm-server \
    --yes

mkdir -p "$STAGE/provenance"
PYTHONPATH=. "$PY" -m experiments.stage1_iid_diagnostic_none.raw_preflight \
  --raw-log-root "$EVAL_ROOT" \
  --manifest "$FROZEN/manifest.json" \
  --split-file "train_eval=$FROZEN/train-eval-n200.jsonl" \
  --split-file "heldout_in_domain=$FROZEN/heldout-in-domain-n200.jsonl" \
  --condition "$CONDITION" \
  --expected-base-model Qwen/Qwen3.5-9B \
  --expected-checkpoint "$COMPAT" \
  --runtime-profile vllm \
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

echo "BCT-main recovery analysis complete: $STAGE/analysis/$CONDITION.json"
