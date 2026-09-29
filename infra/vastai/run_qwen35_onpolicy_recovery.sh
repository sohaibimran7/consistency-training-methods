#!/usr/bin/env bash
# Fail-closed launcher for the two RMCT paper-fidelity targets and the fresh
# OPCT-only recovery.  It is intentionally the only documented production
# entry point: each run must prove its recovered no-CoT source before the
# non-production worker parity bootstrap initializes a model, and only then
# start the actual target-scoped training command.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    bash infra/vastai/run_qwen35_onpolicy_recovery.sh <rmct-main|rmct-control|opct> [--dry-run|--resume-attestation]

Each target owns a distinct experiment/run namespace.  A normal invocation:
  1. validates and records the exact no-CoT recovered source + its manifest;
  2. creates a fixed-token, all-worker Qwen3.5 LoRA transport attestation;
  3. writes an immutable target sidecar binding the authored YAML, compiled
     entry, exact child argv, source proof, and worker proof;
  4. runs only the named training target on the same eight-GPU allocation.

The wrapper refuses a non-numeric or non-eight-GPU CUDA_VISIBLE_DEVICES list,
because scripts/run_experiment.py assigns a concrete exclusive eight-GPU
bundle and the training target uses logical coordinator 0 plus workers 1..7.
`--dry-run` is a no-GPU/no-write command preview.
`--resume-attestation` validates an already-complete immutable source and
worker preflight, then starts training without regenerating the transport
probe. It is safe only before the immutable training-started marker, a
production rollout session, or production adapter snapshots exist.
EOF
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

target="$1"
shift
dry_run=false
resume_attestation=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            if [[ "$dry_run" == true ]]; then
                echo "ERROR: --dry-run was supplied more than once" >&2
                exit 2
            fi
            dry_run=true
            ;;
        --resume-attestation)
            if [[ "$resume_attestation" == true ]]; then
                echo "ERROR: --resume-attestation was supplied more than once" >&2
                exit 2
            fi
            resume_attestation=true
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done
if [[ "$dry_run" == true && "$resume_attestation" == true ]]; then
    echo "ERROR: --dry-run and --resume-attestation are mutually exclusive" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl"
source_manifest_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json"
worker_gpus="1,2,3,4,5,6,7"
worker_gpu_mem_util="0.75"
worker_max_model_len="32768"
worker_max_num_seqs="256"
worker_max_num_batched_tokens="8192"
worker_seed_base="42"
target_logprob_chunk_size="2048"

case "$target" in
    rmct-main)
        plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_paper_fidelity_20260803.yaml"
        experiment="rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803"
        run_name="rate-matching-lr-1e-4"
        label="rmct-paper-fidelity-main"
        ;;
    rmct-control)
        plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_paper_fidelity_20260803.yaml"
        experiment="rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803"
        run_name="rate-matching-control-lr-1e-4"
        label="rmct-paper-fidelity-control"
        ;;
    opct)
        plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_opct_recovery_20260803.yaml"
        experiment="rmct_paper_vast_dense_qwen3_5_9b_opct_recovery_20260803"
        run_name="opct-lr-1e-4"
        label="opct-recovery"
        ;;
    *)
        echo "ERROR: target must be rmct-main, rmct-control, or opct; got $target" >&2
        usage >&2
        exit 2
        ;;
esac

plan="$repo_root/$plan_rel"
source="$repo_root/$source_rel"
source_manifest="$repo_root/$source_manifest_rel"
source_attestation="$repo_root/logs/$experiment/$run_name/preflight/qwen35-recovered-none-source-attestation.json"
worker_parity_attestation="$repo_root/logs/$experiment/$run_name/rollout_workers/qwen35-rollout-worker-parity-attestation.json"
target_attestation="$repo_root/logs/$experiment/$run_name/preflight/qwen35-onpolicy-target-attestation.json"
rollout_status_dir="$repo_root/logs/$experiment/$run_name/rollout_workers"
training_started_marker="$rollout_status_dir/qwen35-onpolicy-training-started.json"
if [[ ! -f "$plan" ]]; then
    echo "ERROR: expected immutable plan is missing: $plan" >&2
    exit 2
fi

python_bin="${CTM_PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="python3"
    fi
fi
if ! "$python_bin" -c 'import sys' >/dev/null 2>&1; then
    echo "ERROR: CTM_PYTHON is not an executable Python interpreter: $python_bin" >&2
    exit 2
fi

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

verify_target_contract() {
    PLAN_PATH="$plan" \
    TARGET_NAME="$target" \
    SOURCE_PATH="$source" \
    SOURCE_MANIFEST_PATH="$source_manifest" \
    EXPERIMENT_NAME="$experiment" \
    RUN_NAME="$run_name" \
    WORKER_GPUS="$worker_gpus" \
    WORKER_GPU_MEM_UTIL="$worker_gpu_mem_util" \
    WORKER_MAX_MODEL_LEN="$worker_max_model_len" \
    WORKER_MAX_NUM_SEQS="$worker_max_num_seqs" \
    WORKER_MAX_NUM_BATCHED_TOKENS="$worker_max_num_batched_tokens" \
    WORKER_SEED_BASE="$worker_seed_base" \
    TARGET_LOGPROB_CHUNK_SIZE="$target_logprob_chunk_size" \
    "$python_bin" - <<'PY'
from __future__ import annotations

import json
import os

from experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight import (
    verify_onpolicy_target_contract,
)

report = verify_onpolicy_target_contract(
    plan=os.environ["PLAN_PATH"],
    target=os.environ["TARGET_NAME"],
    source=os.environ["SOURCE_PATH"],
    source_manifest=os.environ["SOURCE_MANIFEST_PATH"],
    experiment_name=os.environ["EXPERIMENT_NAME"],
    run_name=os.environ["RUN_NAME"],
    worker_gpus=os.environ["WORKER_GPUS"],
    worker_gpu_mem_util=float(os.environ["WORKER_GPU_MEM_UTIL"]),
    worker_max_model_len=int(os.environ["WORKER_MAX_MODEL_LEN"]),
    worker_max_num_seqs=int(os.environ["WORKER_MAX_NUM_SEQS"]),
    worker_max_num_batched_tokens=int(os.environ["WORKER_MAX_NUM_BATCHED_TOKENS"]),
    worker_seed_base=int(os.environ["WORKER_SEED_BASE"]),
    target_logprob_chunk_size=int(os.environ["TARGET_LOGPROB_CHUNK_SIZE"]),
)
print("QWEN35_ONPOLICY_TARGET_CONTRACT=" + json.dumps(report, sort_keys=True))
PY
}

if [[ "$dry_run" == true ]]; then
    gpu_bundle="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    printf '%s\n' \
        "QWEN35_ONPOLICY_RECOVERY_LAUNCH_DRY_RUN=1" \
        "target=$target" \
        "plan=$plan" \
        "experiment_name=$experiment" \
        "run_name=$run_name" \
        "source=$source" \
        "source_manifest=$source_manifest" \
        "source_attestation=$source_attestation" \
        "worker_parity_attestation=$worker_parity_attestation" \
        "target_attestation=$target_attestation" \
        "training_started_marker=$training_started_marker" \
        "source_gate=python -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight --source $source --source-manifest $source_manifest --output $source_attestation" \
        "target_gate=python -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation --plan $plan --target $target --source $source --source-manifest $source_manifest --source-attestation $source_attestation --worker-parity-attestation $worker_parity_attestation --output $target_attestation" \
        "gpu_bundle=$gpu_bundle"
    verify_target_contract
    bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" \
        --experiment "$experiment" \
        --run "$run_name" \
        --worker-gpus "$worker_gpus" \
        --worker-gpu-mem-util "$worker_gpu_mem_util" \
        --worker-max-model-len "$worker_max_model_len" \
        --worker-max-num-seqs "$worker_max_num_seqs" \
        --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
        --worker-seed-base "$worker_seed_base" \
        --target-logprob-chunk-size "$target_logprob_chunk_size" \
        --dry-run
    "$python_bin" scripts/run_experiment.py "$plan_rel" --stages training --target "$target" \
        --parallel 8 --gpus "$gpu_bundle" --dry-run --yes
    exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == "-1" || "${CUDA_VISIBLE_DEVICES}" == "NoDevFiles" ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES to the exact eight-GPU allocation before launching." >&2
    exit 2
fi
gpu_bundle="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
if [[ ! "$gpu_bundle" =~ ^[0-9]+(,[0-9]+){7}$ ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must name exactly eight distinct numeric GPU ids; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 2
fi
IFS=',' read -r -a gpu_parts <<< "$gpu_bundle"
if [[ "${#gpu_parts[@]}" -ne 8 ]]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES must contain exactly eight GPUs" >&2
    exit 2
fi
for ((i = 0; i < ${#gpu_parts[@]}; i++)); do
    for ((j = i + 1; j < ${#gpu_parts[@]}; j++)); do
        if [[ "${gpu_parts[$i]}" == "${gpu_parts[$j]}" ]]; then
            echo "ERROR: CUDA_VISIBLE_DEVICES contains a duplicate GPU id: ${gpu_parts[$i]}" >&2
            exit 2
        fi
    done
done
if [[ ! -f "$source" || ! -f "$source_manifest" ]]; then
    echo "ERROR: frozen recovered-source input or manifest is missing; refusing model initialization." >&2
    echo "  source=$source" >&2
    echo "  source_manifest=$source_manifest" >&2
    exit 2
fi
preflight_resume_args=()
if [[ "$resume_attestation" == true ]]; then
    preflight_resume_args=(--resume-attestation)
fi

# P0 source gate: it invokes the exact content-level legacy-G4 conversion
# verifier, and records both input identities before worker/model startup.
verify_target_contract
"$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight \
    --source "$source" \
    --source-manifest "$source_manifest" \
    --output "$source_attestation"

# P0 transport gate: this is the only model initialization before training.
bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
    --label "$label" \
    --experiment "$experiment" \
    --run "$run_name" \
    --worker-gpus "$worker_gpus" \
    --worker-gpu-mem-util "$worker_gpu_mem_util" \
    --worker-max-model-len "$worker_max_model_len" \
    --worker-max-num-seqs "$worker_max_num_seqs" \
    --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
    --worker-seed-base "$worker_seed_base" \
    --target-logprob-chunk-size "$target_logprob_chunk_size" \
    "${preflight_resume_args[@]}"

# Bind both completed preflights to exactly the child argv that the experiment
# runner will launch. The child independently rechecks this sidecar before it
# touches data or initializes its backend.
"$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation \
    --plan "$plan" \
    --target "$target" \
    --source "$source" \
    --source-manifest "$source_manifest" \
    --source-attestation "$source_attestation" \
    --worker-parity-attestation "$worker_parity_attestation" \
    --output "$target_attestation" \
    --experiment-name "$experiment" \
    --run-name "$run_name" \
    --worker-gpus "$worker_gpus" \
    --worker-gpu-mem-util "$worker_gpu_mem_util" \
    --worker-max-model-len "$worker_max_model_len" \
    --worker-max-num-seqs "$worker_max_num_seqs" \
    --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
    --worker-seed-base "$worker_seed_base" \
    --target-logprob-chunk-size "$target_logprob_chunk_size"

# This intentionally comes immediately before the runner.  OPCT creates its
# log directory before worker/backend setup, so a process killed in that small
# interval has no session-* or adapter marker.  The immutable marker makes a
# later --resume-attestation fail closed instead of relaunching that target.
TARGET_ATTESTATION_PATH="$target_attestation" \
TRAINING_STARTED_MARKER_PATH="$training_started_marker" \
CUDA_VISIBLE_DEVICES="$gpu_bundle" \
"$python_bin" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

target = Path(os.environ["TARGET_ATTESTATION_PATH"]).resolve()
marker = Path(os.environ["TRAINING_STARTED_MARKER_PATH"]).resolve()
if not target.is_file():
    raise SystemExit(f"error: target attestation is missing before training marker: {target}")
try:
    target_document = json.loads(target.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"error: target attestation is invalid before training marker: {target}") from exc
if not isinstance(target_document, dict) or target_document.get("schema") != "qwen35-onpolicy-target-attestation-v1":
    raise SystemExit(f"error: target attestation has unexpected schema before training marker: {target}")
payload = (json.dumps(
    {
        "schema": "qwen35-onpolicy-training-started-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_attestation": {
            "path": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        },
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    },
    indent=2,
    sort_keys=True,
) + "\n").encode("utf-8")
marker.parent.mkdir(parents=True, exist_ok=True)
try:
    with marker.open("xb") as handle:
        handle.write(payload)
except FileExistsError as exc:
    raise SystemExit(f"error: refusing to overwrite training-started marker: {marker}") from exc
print("QWEN35_ONPOLICY_TRAINING_STARTED_MARKER=" + str(marker))
PY

exec "$python_bin" scripts/run_experiment.py "$plan_rel" --stages training --target "$target" \
    --parallel 8 --gpus "$gpu_bundle" --onpolicy-target-attestation "$target_attestation" --yes
