#!/usr/bin/env bash
# Run one parity-attested Qwen3.5 vLLM condition over the full, fresh Stage 2
# IID/HLE 2×2 OOD suite.  This is raw generation only: it makes no OpenRouter
# calls and leaves Luna grading to the hash-bound offline postprocess phase.
set -euo pipefail

REPO=${CTM_OOD_REPO:-/workspace/ctm-ood-hle-20260802/repo}
PY=${CTM_OOD_PY:-/workspace/ctm-act-repair-20260731/env/bin/python}
FROZEN=${CTM_OOD_FROZEN:-$REPO/artifacts/stage2-ood-hle-2x2-20260802-r1}
RUN_ROOT=${CTM_OOD_RUN_ROOT:-/workspace/ctm-ood-hle-20260802}
CONDITION=${CTM_OOD_CONDITION:?CTM_OOD_CONDITION is required}
GPU=${CTM_OOD_GPU:?CTM_OOD_GPU is required}
ADAPTER=${CTM_OOD_ADAPTER:-}

readonly BASE_MODEL=Qwen/Qwen3.5-9B
readonly TASK_FACTORY=experiments.stage2_ood_hle.tasks:ood_tasks
readonly EXPECTED_TASKS=21
readonly MODEL_ARGS='{"provider":"vllm","gpu_memory_utilization":0.9,"language_model_only":true,"max_model_len":32768,"max_num_seqs":256}'
readonly BASE_MODEL_ARGS='{"gpu_memory_utilization":0.9,"language_model_only":true,"max_model_len":32768,"max_num_seqs":256}'
readonly GENERATION_CONFIG='{"max_tokens":20480,"temperature":1.0,"top_p":0.95,"top_k":20,"extra_body":{"top_k":20}}'

case "$CONDITION" in
  base-vllm)
    [ -z "$ADAPTER" ] || { echo "base-vllm must not receive CTM_OOD_ADAPTER" >&2; exit 2; }
    ;;
  act-vllm-compat|attct-vllm-compat|mlpct-vllm-compat|opct-vllm-compat|rmct-vllm-compat)
    [ -n "$ADAPTER" ] || { echo "$CONDITION requires CTM_OOD_ADAPTER" >&2; exit 2; }
    ;;
  *)
    echo "unknown vLLM OOD condition: $CONDITION" >&2
    exit 2
    ;;
esac

test -x "$PY"
test -f "$FROZEN/manifest.json"
test ! -e "$RUN_ROOT/logs/$CONDITION"
test ! -e "$RUN_ROOT/runners/$CONDITION.log"
mkdir -p "$RUN_ROOT/logs/$CONDITION" "$RUN_ROOT/runners"

export PYTHONPATH="$REPO"
export HF_HOME=${CTM_OOD_HF_HOME:-/workspace/ctm-act-repair-20260731/hf-cache}
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
unset VLLM_BASE_URL VLLM_API_KEY

# The launcher can be invoked by nohup from an arbitrary working directory.
# Keep the relative run_evals entrypoint and all child-process paths anchored
# in the isolated staged repository rather than the caller's home directory.
cd "$REPO"

# Validate the frozen task contract before allocating a GPU.  A translated
# adapter is accepted only with the full immutable parity evidence that guards
# against Qwen3.5's previous silently-inactive vLLM LoRA path.
"$PY" - "$FROZEN/manifest.json" "$ADAPTER" "$CONDITION" "$EXPECTED_TASKS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from experiments.stage2_ood_hle.materialize import validate_manifest
from experiments.stage2_ood_hle.tasks import ood_task_specs

manifest_path, adapter, condition, expected_tasks = sys.argv[1:]
manifest = validate_manifest(manifest_path)
specs = ood_task_specs(manifest_path)
if len(specs) != int(expected_tasks) or sum(spec.kind == "unbiased" for spec in specs) != 3:
    raise SystemExit("unexpected Stage 2 OOD task matrix")
if condition != "base-vllm":
    from ctm.evals.qwen35_vllm_attestation import is_verified_qwen35_vllm_compat_adapter

    if not is_verified_qwen35_vllm_compat_adapter(adapter):
        raise SystemExit(f"adapter lacks valid immutable Qwen3.5 vLLM parity evidence: {adapter}")
payload = Path(manifest_path).read_bytes()
print(json.dumps({
    "condition": condition,
    "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    "tasks": len(specs),
    "clean_tasks": sum(spec.kind == "unbiased" for spec in specs),
    "biased_tasks": sum(spec.kind == "biased" for spec in specs),
}, sort_keys=True))
PY

TASK_ARGS=$("$PY" - "$FROZEN/manifest.json" "$RUN_ROOT/logs/$CONDITION" <<'PY'
import json
import sys
print(json.dumps({"manifest": sys.argv[1], "unbiased_log": sys.argv[2], "prompt_style": "none", "include_bias_acknowledged": False}, sort_keys=True))
PY
)

ARGS=(
  scripts/run_evals.py
  --task-factory "$TASK_FACTORY"
  --task-args "$TASK_ARGS"
  --generation-config "$GENERATION_CONFIG"
  --log-dir "$RUN_ROOT/logs/$CONDITION"
  --max-tasks 1
  --isolate-tasks
  --persistent-vllm-server
  --yes
)
if [ "$CONDITION" = base-vllm ]; then
  ARGS+=(--model "vllm/$BASE_MODEL" --model-args "$BASE_MODEL_ARGS")
else
  ARGS+=(--local-checkpoint "$ADAPTER" --base-model "$BASE_MODEL" --model-args "$MODEL_ARGS")
fi

printf '%q ' env "CUDA_VISIBLE_DEVICES=$GPU" "$PY" "${ARGS[@]}"
printf '\n'
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "${ARGS[@]}" >"$RUN_ROOT/runners/$CONDITION.log" 2>&1
printf '%s\n' "Stage 2 OOD vLLM raw condition complete: $CONDITION"
