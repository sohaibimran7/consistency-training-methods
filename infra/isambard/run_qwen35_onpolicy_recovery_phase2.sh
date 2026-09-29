#!/usr/bin/env bash
# Fail-closed inner launcher for one Isambard-AI Phase 2 node.  It uses
# logical GPU 0 as the Qwen3.5 training coordinator and GPUs 1--3 as
# independent vLLM rollout workers.  The outer .sbatch wrapper must invoke
# this script inside one four-GPU Slurm step.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash infra/isambard/run_qwen35_onpolicy_recovery_phase2.sh \
    <rmct-main|rmct-control|opct> \
    [--dry-run|--preflight-only|--resume-attestation]

The caller must provide exactly four Slurm-visible devices.  The launcher:
  1. proves the immutable recovered no-CoT source;
  2. runs a real non-production HF/PEFT-to-vLLM transport probe on all three
     workers;
  3. seals the exact child command to those proofs; and
  4. starts only the selected training target, inheriting the same allocation.

`--preflight-only` performs the real source proof, three-worker LoRA transport
probe, and target attestation, then exits before creating a training marker or
starting production rollouts. Use --resume-attestation only after that
completed preflight, with no training-started marker or production rollout
session. A failed/incomplete training run must use a fresh namespace.
EOF
}

if [ "$#" -lt 1 ]; then
    usage >&2
    exit 2
fi

target=$1
shift
dry_run=false
preflight_only=false
resume_attestation=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dry-run)
            if [ "$dry_run" = true ]; then
                echo "ERROR: --dry-run was supplied more than once" >&2
                exit 2
            fi
            dry_run=true
            ;;
        --preflight-only)
            if [ "$preflight_only" = true ]; then
                echo "ERROR: --preflight-only was supplied more than once" >&2
                exit 2
            fi
            preflight_only=true
            ;;
        --resume-attestation)
            if [ "$resume_attestation" = true ]; then
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
if [ "$dry_run" = true ] && { [ "$preflight_only" = true ] || [ "$resume_attestation" = true ]; }; then
    echo "ERROR: --dry-run cannot be combined with --preflight-only or --resume-attestation" >&2
    exit 2
fi
if [ "$preflight_only" = true ] && [ "$resume_attestation" = true ]; then
    echo "ERROR: --preflight-only and --resume-attestation are mutually exclusive" >&2
    exit 2
fi

repo_root=$(cd -- "$(dirname "$0")/../.." && pwd -P)
source_rel=artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl
source_manifest_rel=artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json

# The official vLLM 0.21 aarch64 wheel contains a vendored DeepGEMM namespace
# but its GH200 build does not expose the FP8 symbols that vLLM's automatic
# kernel warmup probes.  Qwen3.5-9B is loaded here as unquantized BF16
# (quantization=None), so DeepGEMM is neither part of the scientific contract
# nor needed by the rollout path.  Disable it globally instead of merely
# skipping warmup: that forces the supported BF16 kernels and prevents a later
# request from accidentally reaching the same unavailable FP8 backend.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
worker_gpus=1,2,3
worker_gpu_mem_util=0.75
worker_max_model_len=32768
worker_max_num_seqs=256
worker_max_num_batched_tokens=8192
worker_seed_base=42
worker_gdn_prefill_backend=triton
target_logprob_chunk_size=2048

case "$target" in
    rmct-main)
        plan_rel=experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_paper_fidelity_isambard_phase2_4gpu_20260803.yaml
        experiment=rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803
        run_name=rate-matching-lr-1e-4
        label=isambard-phase2-rmct-main
        ;;
    rmct-control)
        plan_rel=experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_paper_fidelity_isambard_phase2_4gpu_20260803.yaml
        experiment=rmct_paper_isambard_phase2_qwen3_5_9b_rng_repair_4gpu_20260803
        run_name=rate-matching-control-lr-1e-4
        label=isambard-phase2-rmct-control
        ;;
    opct)
        plan_rel=experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_opct_recovery_isambard_phase2_4gpu_20260803.yaml
        experiment=rmct_paper_isambard_phase2_qwen3_5_9b_opct_rng_repair_4gpu_20260803
        run_name=opct-lr-1e-4
        label=isambard-phase2-opct
        ;;
    *)
        echo "ERROR: target must be rmct-main, rmct-control, or opct; got $target" >&2
        usage >&2
        exit 2
        ;;
esac

plan=$repo_root/$plan_rel
source_path=$repo_root/$source_rel
source_manifest=$repo_root/$source_manifest_rel
source_attestation=$repo_root/logs/$experiment/$run_name/preflight/qwen35-recovered-none-source-attestation.json
worker_parity_attestation=$repo_root/logs/$experiment/$run_name/rollout_workers/qwen35-rollout-worker-parity-attestation.json
target_attestation=$repo_root/logs/$experiment/$run_name/preflight/qwen35-onpolicy-target-attestation.json
training_started_marker=$repo_root/logs/$experiment/$run_name/rollout_workers/qwen35-onpolicy-training-started.json

if [ ! -f "$plan" ]; then
    echo "ERROR: expected immutable plan is missing: $plan" >&2
    exit 2
fi

python_bin=$(printenv CTM_PYTHON || true)
if [ -z "$python_bin" ]; then
    if [ -x "$repo_root/.venv/bin/python" ]; then
        python_bin=$repo_root/.venv/bin/python
    else
        python_bin=python3
    fi
fi
if ! "$python_bin" -c 'import sys' >/dev/null 2>&1; then
    echo "ERROR: CTM_PYTHON is not an executable Python interpreter: $python_bin" >&2
    exit 2
fi

previous_pythonpath=$(printenv PYTHONPATH || true)
cd "$repo_root"
export PYTHONPATH=$repo_root:$previous_pythonpath

verify_target_contract() {
    PLAN_PATH=$plan TARGET_NAME=$target SOURCE_PATH=$source_path SOURCE_MANIFEST_PATH=$source_manifest \
    EXPERIMENT_NAME=$experiment RUN_NAME=$run_name WORKER_GPUS=$worker_gpus \
    WORKER_GPU_MEM_UTIL=$worker_gpu_mem_util WORKER_MAX_MODEL_LEN=$worker_max_model_len \
    WORKER_MAX_NUM_SEQS=$worker_max_num_seqs WORKER_MAX_NUM_BATCHED_TOKENS=$worker_max_num_batched_tokens \
    WORKER_SEED_BASE=$worker_seed_base \
    WORKER_GDN_PREFILL_BACKEND=$worker_gdn_prefill_backend \
    TARGET_LOGPROB_CHUNK_SIZE=$target_logprob_chunk_size "$python_bin" - <<'PY'
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
    worker_gdn_prefill_backend=os.environ["WORKER_GDN_PREFILL_BACKEND"],
)
print("QWEN35_ONPOLICY_TARGET_CONTRACT=" + json.dumps(report, sort_keys=True))
PY
}

validate_four_visible_gpus() {
    CUDA_VISIBLE_DEVICES=$1 "$python_bin" - <<'PY'
from __future__ import annotations

import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
tokens = [token.strip() for token in visible.split(",")]
if len(tokens) != 4 or any(not token or token in {"-1", "NoDevFiles"} for token in tokens):
    raise SystemExit(
        "error: Isambard Phase 2 target requires exactly four non-empty CUDA_VISIBLE_DEVICES tokens; "
        f"got {visible!r}"
    )
if len(set(tokens)) != 4:
    raise SystemExit(f"error: CUDA_VISIBLE_DEVICES contains duplicate tokens: {visible!r}")
print("QWEN35_ISAMBARD_PHASE2_GPU_ALLOCATION=" + ",".join(tokens))
PY
}

if [ "$dry_run" = true ]; then
    preview_visible=$(printenv CUDA_VISIBLE_DEVICES || true)
    if [ -z "$preview_visible" ]; then
        preview_visible=0,1,2,3
    fi
    validate_four_visible_gpus "$preview_visible"
    printf '%s\n' \
        QWEN35_ISAMBARD_PHASE2_ONPOLICY_DRY_RUN=1 \
        target=$target \
        plan=$plan \
        experiment_name=$experiment \
        run_name=$run_name \
        source=$source_path \
        source_manifest=$source_manifest \
        source_attestation=$source_attestation \
        worker_parity_attestation=$worker_parity_attestation \
        worker_gdn_prefill_backend=$worker_gdn_prefill_backend \
        worker_seed_base=$worker_seed_base \
        target_attestation=$target_attestation \
        training_started_marker=$training_started_marker \
        cuda_visible_devices=$preview_visible
    verify_target_contract
    CUDA_VISIBLE_DEVICES=$preview_visible bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" --experiment "$experiment" --run "$run_name" \
        --worker-gpus "$worker_gpus" --worker-gpu-mem-util "$worker_gpu_mem_util" \
        --worker-max-model-len "$worker_max_model_len" --worker-max-num-seqs "$worker_max_num_seqs" \
        --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
        --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend "$worker_gdn_prefill_backend" \
        --target-logprob-chunk-size "$target_logprob_chunk_size" --dry-run
    CUDA_VISIBLE_DEVICES=$preview_visible "$python_bin" scripts/run_experiment.py "$plan_rel" \
        --stages training --target "$target" --parallel 1 --dry-run --yes
    exit 0
fi

cuda_visible=$(printenv CUDA_VISIBLE_DEVICES || true)
if [ -z "$cuda_visible" ]; then
    echo "ERROR: no Slurm CUDA_VISIBLE_DEVICES allocation is visible" >&2
    exit 2
fi
validate_four_visible_gpus "$cuda_visible"
export CUDA_VISIBLE_DEVICES=$cuda_visible
if [ ! -f "$source_path" ] || [ ! -f "$source_manifest" ]; then
    echo "ERROR: frozen recovered no-CoT source or manifest is missing; refusing model initialization." >&2
    echo "  source=$source_path" >&2
    echo "  source_manifest=$source_manifest" >&2
    exit 2
fi

verify_target_contract
"$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight \
    --source "$source_path" --source-manifest "$source_manifest" --output "$source_attestation"

if [ "$resume_attestation" = true ]; then
    bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" --experiment "$experiment" --run "$run_name" \
        --worker-gpus "$worker_gpus" --worker-gpu-mem-util "$worker_gpu_mem_util" \
        --worker-max-model-len "$worker_max_model_len" --worker-max-num-seqs "$worker_max_num_seqs" \
        --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
        --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend "$worker_gdn_prefill_backend" \
        --target-logprob-chunk-size "$target_logprob_chunk_size" --resume-attestation
else
    bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" --experiment "$experiment" --run "$run_name" \
        --worker-gpus "$worker_gpus" --worker-gpu-mem-util "$worker_gpu_mem_util" \
        --worker-max-model-len "$worker_max_model_len" --worker-max-num-seqs "$worker_max_num_seqs" \
        --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
        --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend "$worker_gdn_prefill_backend" \
        --target-logprob-chunk-size "$target_logprob_chunk_size"
fi

"$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation \
    --plan "$plan" --target "$target" --source "$source_path" --source-manifest "$source_manifest" \
    --source-attestation "$source_attestation" --worker-parity-attestation "$worker_parity_attestation" \
    --output "$target_attestation" --experiment-name "$experiment" --run-name "$run_name" \
    --worker-gpus "$worker_gpus" --worker-gpu-mem-util "$worker_gpu_mem_util" \
    --worker-max-model-len "$worker_max_model_len" --worker-max-num-seqs "$worker_max_num_seqs" \
    --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
    --worker-seed-base "$worker_seed_base" \
    --worker-gdn-prefill-backend "$worker_gdn_prefill_backend" \
    --target-logprob-chunk-size "$target_logprob_chunk_size"

if [ "$preflight_only" = true ]; then
    printf '%s\n' \
        QWEN35_ISAMBARD_PHASE2_PREFLIGHT_COMPLETE=1 \
        target=$target \
        source_attestation=$source_attestation \
        worker_parity_attestation=$worker_parity_attestation \
        target_attestation=$target_attestation
    exit 0
fi

TARGET_ATTESTATION_PATH=$target_attestation TRAINING_STARTED_MARKER_PATH=$training_started_marker \
CUDA_VISIBLE_DEVICES=$cuda_visible "$python_bin" - <<'PY'
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
document = json.loads(target.read_text(encoding="utf-8"))
if not isinstance(document, dict) or document.get("schema") != "qwen35-onpolicy-target-attestation-v1":
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

# Do not pass --gpus: a sequential child inherits Slurm's complete, possibly
# UUID-valued, allocation unchanged.  The launcher above has already proven it
# contains exactly four visible devices.
exec "$python_bin" scripts/run_experiment.py "$plan_rel" --stages training --target "$target" \
    --parallel 1 --onpolicy-target-attestation "$target_attestation" --yes
