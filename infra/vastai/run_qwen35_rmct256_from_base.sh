#!/usr/bin/env bash
# Fail-closed launcher for the immutable RMCT-256 main-condition experiment.
# A profile selects the full logical coordinator/worker topology; arbitrary GPU
# counts and control targets are intentionally not accepted.  The target
# attestation seals the selected profile, complete worker allocation, selected
# 256-row input, and its full provenance chain before production rollouts.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=<allocated GPUs> \
    bash infra/vastai/run_qwen35_rmct256_from_base.sh \
      --topology-profile <four-gpu|eight-gpu> [--dry-run|--preflight-only|--resume-attestation] [--yes]

This launcher owns only the fresh RMCT main condition in the immutable
`rmct256-from-base-20260804` namespace. It has no control target.

The required profile chooses an authored logical topology:
  four-gpu   coordinator cuda:0 and rollout workers cuda:1,2,3
  eight-gpu  coordinator cuda:0 and rollout workers cuda:1,2,3,4,5,6,7

For a real invocation, CUDA_VISIBLE_DEVICES must contain exactly the selected
profile's number of unique non-empty tokens (numeric IDs or scheduler UUIDs).
The profile and that exact allocation are sealed into the target sidecar.

`--dry-run` performs no writes or model initialization. `--preflight-only`
performs the source proof, real worker transport probe, and target attestation,
then exits before the immutable training-start marker or production rollout.
Both normal training and --preflight-only require --yes. `--resume-attestation`
only revalidates a completed worker preflight before a new production start; it
does not assert an exact on-policy continuation because rollout/RNG state is
not restored by optimizer state alone.
EOF
}

topology_profile=""
dry_run=false
preflight_only=false
resume_attestation=false
yes=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --topology-profile)
            if [[ -n "$topology_profile" || $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --topology-profile requires exactly one non-empty value" >&2
                exit 2
            fi
            topology_profile="$2"
            shift 2
            ;;
        --dry-run)
            if [[ "$dry_run" == true ]]; then
                echo "ERROR: --dry-run was supplied more than once" >&2
                exit 2
            fi
            dry_run=true
            shift
            ;;
        --preflight-only)
            if [[ "$preflight_only" == true ]]; then
                echo "ERROR: --preflight-only was supplied more than once" >&2
                exit 2
            fi
            preflight_only=true
            shift
            ;;
        --resume-attestation)
            if [[ "$resume_attestation" == true ]]; then
                echo "ERROR: --resume-attestation was supplied more than once" >&2
                exit 2
            fi
            resume_attestation=true
            shift
            ;;
        --yes)
            yes=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$topology_profile" ]]; then
    echo "ERROR: --topology-profile is required" >&2
    usage >&2
    exit 2
fi
if [[ "$dry_run" == true && ( "$preflight_only" == true || "$resume_attestation" == true || "$yes" == true ) ]]; then
    echo "ERROR: --dry-run cannot be combined with --preflight-only, --resume-attestation, or --yes" >&2
    exit 2
fi
if [[ "$preflight_only" == true && "$resume_attestation" == true ]]; then
    echo "ERROR: --preflight-only and --resume-attestation are mutually exclusive" >&2
    exit 2
fi
if [[ "$dry_run" == false && "$yes" == false ]]; then
    echo "ERROR: real preflight or training requires --yes" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_256_from_base_20260804.yaml"
source_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl"
source_manifest_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json"
selection_rel="artifacts/rmct-256-training-20260804/rmct-256-training-7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a.jsonl"
selection_manifest_rel="artifacts/rmct-256-training-20260804/rmct-256-training-manifest-5c18fa31ddaaca76256bef15cc54ddfa0f103287b0f5c940fd0a87cc8e2e179e.json"
canonical_source_rel="artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-20260801/data/canonical-consistency-pairs-n2048.jsonl"
canonical_source_manifest_rel="artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-20260801/data/canonical-consistency-pairs-n2048.manifest.json"
original64_reference_manifest_rel="artifacts/stage1-iid-diagnostic-none-20260801/manifest.json"
stage2_manifest_rel="artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json"

worker_gpu_mem_util="0.75"
worker_max_model_len="32768"
worker_max_num_seqs="256"
worker_max_num_batched_tokens="8192"
worker_seed_base="42"
worker_gdn_prefill_backend="triton"
target_logprob_chunk_size="2048"
experiment="rmct256-from-base-20260804"

case "$topology_profile" in
    four-gpu)
        target="rmct-256-4gpu"
        run_name="rate-matching-lr-1e-4-4gpu"
        worker_gpus="1,2,3"
        gpu_count="4"
        label="rmct256-from-base-four-gpu"
        preview_visible="0,1,2,3"
        ;;
    eight-gpu)
        target="rmct-256-8gpu"
        run_name="rate-matching-lr-1e-4-8gpu"
        worker_gpus="1,2,3,4,5,6,7"
        gpu_count="8"
        label="rmct256-from-base-eight-gpu"
        preview_visible="0,1,2,3,4,5,6,7"
        ;;
    *)
        echo "ERROR: --topology-profile must be four-gpu or eight-gpu; got $topology_profile" >&2
        exit 2
        ;;
esac

plan="$repo_root/$plan_rel"
source_path="$repo_root/$source_rel"
source_manifest="$repo_root/$source_manifest_rel"
selection_path="$repo_root/$selection_rel"
selection_manifest="$repo_root/$selection_manifest_rel"
canonical_source="$repo_root/$canonical_source_rel"
canonical_source_manifest="$repo_root/$canonical_source_manifest_rel"
original64_reference_manifest="$repo_root/$original64_reference_manifest_rel"
stage2_manifest="$repo_root/$stage2_manifest_rel"
source_attestation="$repo_root/logs/$experiment/$run_name/preflight/qwen35-recovered-none-source-attestation.json"
worker_parity_attestation="$repo_root/logs/$experiment/$run_name/rollout_workers/qwen35-rollout-worker-parity-attestation.json"
target_attestation="$repo_root/logs/$experiment/$run_name/preflight/qwen35-onpolicy-target-attestation.json"
training_started_marker="$repo_root/logs/$experiment/$run_name/rollout_workers/qwen35-onpolicy-training-started.json"

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
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

validate_visible_gpu_count() {
    CUDA_VISIBLE_DEVICES="$1" EXPECTED_GPU_COUNT="$gpu_count" "$python_bin" - <<'PY'
from __future__ import annotations

import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
expected = int(os.environ["EXPECTED_GPU_COUNT"])
tokens = [token.strip() for token in visible.split(",")]
if len(tokens) != expected or any(not token or token in {"-1", "NoDevFiles"} for token in tokens):
    raise SystemExit(
        f"error: profile requires exactly {expected} non-empty CUDA_VISIBLE_DEVICES tokens; got {visible!r}"
    )
if len(set(tokens)) != expected:
    raise SystemExit(f"error: CUDA_VISIBLE_DEVICES contains duplicate tokens: {visible!r}")
print("QWEN35_RMCT256_GPU_ALLOCATION=" + ",".join(tokens))
PY
}

verify_target_contract() {
    PLAN_PATH="$plan" \
    TARGET_NAME="$target" \
    SOURCE_PATH="$source_path" \
    SOURCE_MANIFEST_PATH="$source_manifest" \
    RMCT256_SELECTION_PATH="$selection_path" \
    RMCT256_SELECTION_MANIFEST_PATH="$selection_manifest" \
    RMCT256_CANONICAL_SOURCE_PATH="$canonical_source" \
    RMCT256_CANONICAL_SOURCE_MANIFEST_PATH="$canonical_source_manifest" \
    RMCT256_ORIGINAL64_REFERENCE_MANIFEST_PATH="$original64_reference_manifest" \
    RMCT256_STAGE2_MANIFEST_PATH="$stage2_manifest" \
    EXPERIMENT_NAME="$experiment" \
    RUN_NAME="$run_name" \
    TOPOLOGY_PROFILE="$topology_profile" \
    WORKER_GPUS="$worker_gpus" \
    WORKER_GPU_MEM_UTIL="$worker_gpu_mem_util" \
    WORKER_MAX_MODEL_LEN="$worker_max_model_len" \
    WORKER_MAX_NUM_SEQS="$worker_max_num_seqs" \
    WORKER_MAX_NUM_BATCHED_TOKENS="$worker_max_num_batched_tokens" \
    WORKER_SEED_BASE="$worker_seed_base" \
    WORKER_GDN_PREFILL_BACKEND="$worker_gdn_prefill_backend" \
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
    target_logprob_chunk_size=int(os.environ["TARGET_LOGPROB_CHUNK_SIZE"]),
    worker_gdn_prefill_backend=os.environ["WORKER_GDN_PREFILL_BACKEND"],
    worker_seed_base=int(os.environ["WORKER_SEED_BASE"]),
    topology_profile=os.environ["TOPOLOGY_PROFILE"],
    rmct256_selection=os.environ["RMCT256_SELECTION_PATH"],
    rmct256_selection_manifest=os.environ["RMCT256_SELECTION_MANIFEST_PATH"],
    rmct256_canonical_source=os.environ["RMCT256_CANONICAL_SOURCE_PATH"],
    rmct256_canonical_source_manifest=os.environ["RMCT256_CANONICAL_SOURCE_MANIFEST_PATH"],
    rmct256_original64_reference_manifest=os.environ["RMCT256_ORIGINAL64_REFERENCE_MANIFEST_PATH"],
    rmct256_stage2_manifest=os.environ["RMCT256_STAGE2_MANIFEST_PATH"],
)
print("QWEN35_RMCT256_TARGET_CONTRACT=" + json.dumps(report, sort_keys=True))
PY
}

worker_preflight() {
    local mode="$1"
    local -a resume_args=()
    if [[ "$mode" == "resume" ]]; then
        resume_args=(--resume-attestation)
    elif [[ "$mode" == "dry-run" ]]; then
        resume_args=(--dry-run)
    fi
    bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" --experiment "$experiment" --run "$run_name" \
        --worker-gpus "$worker_gpus" --worker-gpu-mem-util "$worker_gpu_mem_util" \
        --worker-max-model-len "$worker_max_model_len" --worker-max-num-seqs "$worker_max_num_seqs" \
        --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
        --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend "$worker_gdn_prefill_backend" \
        --target-logprob-chunk-size "$target_logprob_chunk_size" \
        "${resume_args[@]}"
}

if [[ "$dry_run" == true ]]; then
    cuda_visible="${CUDA_VISIBLE_DEVICES:-$preview_visible}"
    validate_visible_gpu_count "$cuda_visible"
    printf '%s\n' \
        "QWEN35_RMCT256_FROM_BASE_DRY_RUN=1" \
        "topology_profile=$topology_profile" \
        "target=$target" \
        "plan=$plan" \
        "experiment_name=$experiment" \
        "run_name=$run_name" \
        "gpu_count=$gpu_count" \
        "worker_gpus=$worker_gpus" \
        "source=$source_path" \
        "source_manifest=$source_manifest" \
        "selection=$selection_path" \
        "selection_manifest=$selection_manifest" \
        "canonical_source=$canonical_source" \
        "canonical_source_manifest=$canonical_source_manifest" \
        "original64_reference_manifest=$original64_reference_manifest" \
        "stage2_manifest=$stage2_manifest" \
        "source_attestation=$source_attestation" \
        "worker_parity_attestation=$worker_parity_attestation" \
        "target_attestation=$target_attestation" \
        "training_started_marker=$training_started_marker" \
        "cuda_visible_devices=$cuda_visible"
    verify_target_contract
    CUDA_VISIBLE_DEVICES="$cuda_visible" worker_preflight dry-run
    CUDA_VISIBLE_DEVICES="$cuda_visible" "$python_bin" scripts/run_experiment.py "$plan_rel" \
        --topology-profile "$topology_profile" --stages training --target "$target" --parallel 1 --dry-run --yes
    exit 0
fi

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$cuda_visible" ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES to the selected profile's complete allocation before launching." >&2
    exit 2
fi
validate_visible_gpu_count "$cuda_visible"
export CUDA_VISIBLE_DEVICES="$cuda_visible"

for frozen_input in \
    "$source_path" "$source_manifest" "$selection_path" "$selection_manifest" \
    "$canonical_source" "$canonical_source_manifest" "$original64_reference_manifest" "$stage2_manifest"; do
    if [[ ! -f "$frozen_input" ]]; then
        echo "ERROR: required immutable RMCT-256 input is missing; refusing model initialization: $frozen_input" >&2
        exit 2
    fi
done

# This full content proof runs before the recovered-source sidecar and before
# any worker/model startup; it binds the exact selected data to its six-source
# provenance chain and the selected topology to the compiled target argv.
verify_target_contract
"$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight \
    --source "$source_path" --source-manifest "$source_manifest" --output "$source_attestation"

if [[ "$resume_attestation" == true ]]; then
    worker_preflight resume
else
    worker_preflight regular
fi

"$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation \
    --plan "$plan" --target "$target" --topology-profile "$topology_profile" \
    --source "$source_path" --source-manifest "$source_manifest" \
    --source-attestation "$source_attestation" --worker-parity-attestation "$worker_parity_attestation" \
    --rmct256-selection "$selection_path" --rmct256-selection-manifest "$selection_manifest" \
    --rmct256-canonical-source "$canonical_source" \
    --rmct256-canonical-source-manifest "$canonical_source_manifest" \
    --rmct256-original64-reference-manifest "$original64_reference_manifest" \
    --rmct256-stage2-manifest "$stage2_manifest" \
    --output "$target_attestation" --experiment-name "$experiment" --run-name "$run_name" \
    --worker-gpus "$worker_gpus" --worker-gpu-mem-util "$worker_gpu_mem_util" \
    --worker-max-model-len "$worker_max_model_len" --worker-max-num-seqs "$worker_max_num_seqs" \
    --worker-max-num-batched-tokens "$worker_max_num_batched_tokens" \
    --worker-seed-base "$worker_seed_base" \
    --worker-gdn-prefill-backend "$worker_gdn_prefill_backend" \
    --target-logprob-chunk-size "$target_logprob_chunk_size"

if [[ "$preflight_only" == true ]]; then
    printf '%s\n' \
        "QWEN35_RMCT256_FROM_BASE_PREFLIGHT_COMPLETE=1" \
        "topology_profile=$topology_profile" \
        "target=$target" \
        "source_attestation=$source_attestation" \
        "worker_parity_attestation=$worker_parity_attestation" \
        "target_attestation=$target_attestation"
    exit 0
fi

TARGET_ATTESTATION_PATH="$target_attestation" \
TRAINING_STARTED_MARKER_PATH="$training_started_marker" \
EXPECTED_TARGET="$target" \
EXPECTED_PROFILE="$topology_profile" \
CUDA_VISIBLE_DEVICES="$cuda_visible" \
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
    document = json.loads(target.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"error: target attestation is invalid before training marker: {target}") from exc
if not isinstance(document, dict) or document.get("schema") != "qwen35-onpolicy-target-attestation-v1":
    raise SystemExit(f"error: target attestation has unexpected schema before training marker: {target}")
if document.get("target") != os.environ["EXPECTED_TARGET"]:
    raise SystemExit("error: target attestation target differs from the selected RMCT-256 target")
if document.get("topology_profile") != os.environ["EXPECTED_PROFILE"]:
    raise SystemExit("error: target attestation topology profile differs from the selected profile")
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

# A sequential runner must inherit the attested complete allocation unchanged.
# Do not pass --gpus: that option is reserved for parallel host-side splitting.
exec "$python_bin" scripts/run_experiment.py "$plan_rel" \
    --topology-profile "$topology_profile" --stages training --target "$target" \
    --parallel 1 --onpolicy-target-attestation "$target_attestation" --yes
