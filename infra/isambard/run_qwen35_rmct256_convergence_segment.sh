#!/usr/bin/env bash
# Execute exactly one static RMCT-256 convergence segment on one Isambard
# Phase-2 GH200 node.  This is intentionally narrower than the legacy
# Phase-2 wrapper: it accepts a global segment index only, derives the target,
# namespace, slice, parent, and worker seeds from the immutable plan, and
# never offers a same-namespace retry or an ambient checkpoint override.
set -euo pipefail

condition="rmct256-convergence-isambard-4x64-20260811"
plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct256_convergence_isambard_4x64_20260811.yaml"
selection_sha256="7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a"

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=<four allocated devices> \
    bash infra/isambard/run_qwen35_rmct256_convergence_segment.sh \
      --segment-index <0..15> [--dry-run|--preflight-only|--yes]

The segment index is global across four ordered passes of four 64-question
blocks. The launcher derives its static target, unique run namespace, exact
row offset, parent final checkpoint, and rollout worker seeds from the
authored RMCT-256 convergence plan. It deliberately rejects arbitrary target,
checkpoint, resume, and topology arguments.

`--dry-run` performs no writes or model initialization. `--preflight-only`
is a real four-GPU segment-0 transport/attestation probe under a dedicated
non-production namespace; it requires `--yes`, creates no production
training-start marker or target output state, and returns before the runner.
A normal real invocation requires `--yes`, one Slurm-visible four-GPU
allocation, the pinned offline Qwen3.5 snapshot, and a clean unique namespace
(or a valid completion receipt for an idempotent no-op).
EOF
}

segment_raw=""
dry_run=false
preflight_only=false
yes=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --segment-index)
            if [[ -n "$segment_raw" || $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --segment-index requires one integer in [0, 15]" >&2
                exit 2
            fi
            segment_raw="$2"
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
        --yes)
            if [[ "$yes" == true ]]; then
                echo "ERROR: --yes was supplied more than once" >&2
                exit 2
            fi
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

if [[ -z "$segment_raw" ]]; then
    echo "ERROR: --segment-index is required" >&2
    usage >&2
    exit 2
fi
segment_index=$((10#$segment_raw))
if (( segment_index < 0 || segment_index >= 16 )); then
    echo "ERROR: --segment-index must be in [0, 15]" >&2
    exit 2
fi
if [[ "$dry_run" == true && "$preflight_only" == true ]]; then
    echo "ERROR: --dry-run and --preflight-only are mutually exclusive" >&2
    exit 2
fi
if [[ "$preflight_only" == true && "$segment_index" -ne 0 ]]; then
    echo "ERROR: --preflight-only is defined only for static segment index 0" >&2
    exit 2
fi
if [[ "$dry_run" == false && "$yes" == false ]]; then
    echo "ERROR: real production work requires --yes" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
plan="$repo_root/$plan_rel"
contract="$repo_root/infra/isambard/rmct256_convergence_segment_contract.py"
metric_extractor="$repo_root/infra/isambard/extract_rmct256_convergence_metrics.py"
lora_fingerprint_helper="$repo_root/infra/isambard/rmct256_convergence_lora_fingerprint.py"
runtime_policy_helper="$repo_root/infra/isambard/rmct256_runtime_policy_receipt.py"
if [[ ! -f "$plan" || ! -f "$contract" || ! -f "$metric_extractor" || ! -f "$lora_fingerprint_helper" || ! -f "$runtime_policy_helper" ]]; then
    echo "ERROR: expected static plan or RMCT256 convergence helper is missing" >&2
    echo "  plan=$plan" >&2
    echo "  contract=$contract" >&2
    echo "  metric_extractor=$metric_extractor" >&2
    echo "  lora_fingerprint_helper=$lora_fingerprint_helper" >&2
    echo "  runtime_policy_helper=$runtime_policy_helper" >&2
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

pass_index=$((segment_index / 4 + 1))
block_index=$((segment_index % 4))
printf -v pass_label '%02d' "$pass_index"
printf -v block_label '%02d' "$((block_index + 1))"
target="rmct256-convergence-p${pass_label}-s${block_label}"
run_name="${condition}-p${pass_label}-s${block_label}"
worker_seed_base=$((42 + 3 * segment_index))
row_offset=$((64 * block_index))
label="isambard-rmct256-convergence-p${pass_label}-s${block_label}"

source_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl"
source_manifest_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json"
selection_rel="artifacts/rmct-256-training-20260804/rmct-256-training-7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a.jsonl"
selection_manifest_rel="artifacts/rmct-256-training-20260804/rmct-256-training-manifest-5c18fa31ddaaca76256bef15cc54ddfa0f103287b0f5c940fd0a87cc8e2e179e.json"
canonical_source_rel="artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-20260801/data/canonical-consistency-pairs-n2048.jsonl"
canonical_source_manifest_rel="artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-20260801/data/canonical-consistency-pairs-n2048.manifest.json"
original64_reference_manifest_rel="artifacts/stage1-iid-diagnostic-none-20260801/manifest.json"
stage2_manifest_rel="artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json"

source_path="$repo_root/$source_rel"
source_manifest="$repo_root/$source_manifest_rel"
selection_path="$repo_root/$selection_rel"
selection_manifest="$repo_root/$selection_manifest_rel"
canonical_source="$repo_root/$canonical_source_rel"
canonical_source_manifest="$repo_root/$canonical_source_manifest_rel"
original64_reference_manifest="$repo_root/$original64_reference_manifest_rel"
stage2_manifest="$repo_root/$stage2_manifest_rel"
run_root="$repo_root/logs/$condition/$run_name"
source_attestation="$run_root/preflight/qwen35-recovered-none-source-attestation.json"
worker_parity_attestation="$run_root/rollout_workers/qwen35-rollout-worker-parity-attestation.json"
target_attestation="$run_root/preflight/qwen35-onpolicy-target-attestation.json"
training_started_marker="$run_root/rollout_workers/qwen35-onpolicy-training-started.json"
lora_fingerprint="$run_root/preflight/qwen35-lora-fingerprint-attestation.json"
runtime_policy_receipt="$run_root/preflight/qwen35-runtime-policy-receipt.json"
preflight_experiment="${condition}-preflight"
preflight_run_name="${preflight_experiment}-p01-s01"
preflight_root="$repo_root/logs/$preflight_experiment/$preflight_run_name"
preflight_plan="$preflight_root/preflight/rmct256-convergence-segment0-preflight.yaml"
preflight_source_attestation="$preflight_root/preflight/qwen35-recovered-none-source-attestation.json"
preflight_base_snapshot_attestation="$preflight_root/preflight/qwen35-base-snapshot-attestation.json"
preflight_worker_parity_attestation="$preflight_root/rollout_workers/qwen35-rollout-worker-parity-attestation.json"
preflight_target_attestation="$preflight_root/preflight/qwen35-onpolicy-target-attestation.json"
preflight_lora_fingerprint="$preflight_root/preflight/qwen35-lora-fingerprint-attestation.json"
preflight_runtime_policy_receipt="$preflight_root/preflight/qwen35-runtime-policy-receipt.json"
preflight_success_receipt="$preflight_root/preflight/rmct256-convergence-preflight-success-receipt.json"
preflight_label="isambard-rmct256-convergence-preflight-p01-s01"
production_output_state="$repo_root/logs/experiments/$condition/targets/$target/outputs.json"

# The coordinator and all three worker processes use the model identifier's
# default `main` ref. The contract helper proves that ref resolves locally to
# this exact immutable snapshot before preflight; network fallback is disabled.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_RUN_GROUP="$condition"
export WANDB_JOB_TYPE="rmct256-segment"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"

validate_four_visible_gpus() {
    CUDA_VISIBLE_DEVICES="$1" "$python_bin" - <<'PY'
from __future__ import annotations

import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
tokens = [token.strip() for token in visible.split(",")]
if len(tokens) != 4 or any(not token or token in {"-1", "NoDevFiles"} for token in tokens):
    raise SystemExit(
        "error: RMCT256 convergence requires exactly four non-empty Slurm-visible CUDA devices; "
        f"got {visible!r}"
    )
if len(set(tokens)) != 4:
    raise SystemExit(f"error: CUDA_VISIBLE_DEVICES contains duplicate tokens: {visible!r}")
print("RMCT256_CONVERGENCE_GPU_ALLOCATION=" + ",".join(tokens))
PY
}

verify_target_contract() {
    local contract_plan="${1:-$plan}"
    local contract_target="${2:-$target}"
    local contract_experiment="${3:-$condition}"
    local contract_run_name="${4:-$run_name}"
    PLAN_PATH="$contract_plan" \
    TARGET_NAME="$contract_target" \
    SOURCE_PATH="$source_path" \
    SOURCE_MANIFEST_PATH="$source_manifest" \
    RMCT256_SELECTION_PATH="$selection_path" \
    RMCT256_SELECTION_MANIFEST_PATH="$selection_manifest" \
    RMCT256_CANONICAL_SOURCE_PATH="$canonical_source" \
    RMCT256_CANONICAL_SOURCE_MANIFEST_PATH="$canonical_source_manifest" \
    RMCT256_ORIGINAL64_REFERENCE_MANIFEST_PATH="$original64_reference_manifest" \
    RMCT256_STAGE2_MANIFEST_PATH="$stage2_manifest" \
    EXPERIMENT_NAME="$contract_experiment" \
    RUN_NAME="$contract_run_name" \
    WORKER_SEED_BASE="$worker_seed_base" \
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
    worker_gpus="1,2,3",
    worker_gpu_mem_util=0.75,
    worker_max_model_len=32768,
    worker_max_num_seqs=256,
    worker_max_num_batched_tokens=8192,
    target_logprob_chunk_size=2048,
    worker_gdn_prefill_backend="triton",
    worker_seed_base=int(os.environ["WORKER_SEED_BASE"]),
    topology_profile="four-gpu",
    rmct256_selection=os.environ["RMCT256_SELECTION_PATH"],
    rmct256_selection_manifest=os.environ["RMCT256_SELECTION_MANIFEST_PATH"],
    rmct256_canonical_source=os.environ["RMCT256_CANONICAL_SOURCE_PATH"],
    rmct256_canonical_source_manifest=os.environ["RMCT256_CANONICAL_SOURCE_MANIFEST_PATH"],
    rmct256_original64_reference_manifest=os.environ["RMCT256_ORIGINAL64_REFERENCE_MANIFEST_PATH"],
    rmct256_stage2_manifest=os.environ["RMCT256_STAGE2_MANIFEST_PATH"],
)
print("RMCT256_CONVERGENCE_TARGET_CONTRACT=" + json.dumps(report, sort_keys=True))
PY
}

if [[ "$dry_run" == true ]]; then
    preview_visible="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
    validate_four_visible_gpus "$preview_visible"
    "$python_bin" "$contract" validate-plan --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index"
    printf '%s\n' \
        "RMCT256_CONVERGENCE_DRY_RUN=1" \
        "condition=$condition" \
        "segment_index=$segment_index" \
        "pass_index=$pass_index" \
        "block_index=$block_index" \
        "target=$target" \
        "run_name=$run_name" \
        "row_offset=$row_offset" \
        "worker_seed_base=$worker_seed_base" \
        "plan=$plan" \
        "wandb_run_group=$WANDB_RUN_GROUP" \
        "wandb_job_type=$WANDB_JOB_TYPE" \
        "hf_hub_offline=$HF_HUB_OFFLINE" \
        "cuda_visible_devices=$preview_visible"
    verify_target_contract
    CUDA_VISIBLE_DEVICES="$preview_visible" bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" --experiment "$condition" --run "$run_name" \
        --worker-gpus 1,2,3 --worker-gpu-mem-util 0.75 \
        --worker-max-model-len 32768 --worker-max-num-seqs 256 \
        --worker-max-num-batched-tokens 8192 --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend triton --target-logprob-chunk-size 2048 --dry-run
    CUDA_VISIBLE_DEVICES="$preview_visible" "$python_bin" scripts/run_experiment.py "$plan_rel" \
        --topology-profile four-gpu --stages training --target "$target" --parallel 1 --dry-run --yes
    exit 0
fi

cuda_visible="${CUDA_VISIBLE_DEVICES:-}"
if [[ -z "$cuda_visible" ]]; then
    echo "ERROR: no Slurm CUDA_VISIBLE_DEVICES allocation is visible" >&2
    exit 2
fi
validate_four_visible_gpus "$cuda_visible"
export CUDA_VISIBLE_DEVICES="$cuda_visible"

verify_frozen_inputs() {
    local frozen_input
    for frozen_input in \
        "$source_path" "$source_manifest" "$selection_path" "$selection_manifest" \
        "$canonical_source" "$canonical_source_manifest" "$original64_reference_manifest" "$stage2_manifest"; do
        if [[ ! -f "$frozen_input" ]]; then
            echo "ERROR: required immutable RMCT256 input is missing: $frozen_input" >&2
            return 2
        fi
    done
}

if [[ "$preflight_only" == true ]]; then
    # The worker probe uses a real short LoRA transport bootstrap, but all of
    # its evidence is below this isolated namespace.  The protected production
    # segment-0 namespace and runner output state must remain absent.
    if [[ -e "$run_root" || -L "$run_root" || -e "$production_output_state" || -L "$production_output_state" ]]; then
        echo "ERROR: production segment-0 state already exists; refuse to mix a preflight with its namespace." >&2
        echo "  run_root=$run_root" >&2
        echo "  target_output_state=$production_output_state" >&2
        exit 2
    fi
    preflight_lock="$preflight_root/preflight/rmct256-convergence-preflight.lock"
    mkdir -p "$(dirname -- "$preflight_lock")"
    exec 9>"$preflight_lock"
    if ! flock -n 9; then
        echo "ERROR: another job holds the RMCT256 dedicated preflight lock: $preflight_lock" >&2
        exit 2
    fi

    verify_frozen_inputs
    "$python_bin" "$contract" validate-plan --repo-root "$repo_root" --plan "$plan" --segment-index 0
    "$python_bin" "$contract" preflight-plan --repo-root "$repo_root" --plan "$plan" --segment-index 0 --output "$preflight_plan"

    # This production-plan check plus the generated-plan factory check proves
    # the isolated namespace has not changed the segment-0 data/model/worker
    # command.  Only experiment/run names differ so no production residue is
    # possible.
    verify_target_contract "$plan" "$target" "$condition" "$run_name"
    verify_target_contract "$preflight_plan" "$target" "$preflight_experiment" "$preflight_run_name"

    # Fail closed on the pinned cache before any worker model process starts;
    # the sidecar is written only in the dedicated preflight namespace.
    "$python_bin" "$contract" base-snapshot --repo-root "$repo_root" --plan "$plan" --segment-index 0 \
        --output "$preflight_base_snapshot_attestation"
    "$python_bin" "$lora_fingerprint_helper" capture \
        --repo-root "$repo_root" --plan "$plan" --segment-index 0 \
        --base-snapshot-attestation "$preflight_base_snapshot_attestation" \
        --output "$preflight_lora_fingerprint"
    "$python_bin" -m infra.isambard.rmct256_runtime_policy_receipt capture \
        --repo-root "$repo_root" --plan "$plan" --segment-index 0 \
        --output "$preflight_runtime_policy_receipt"
    "$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight \
        --source "$source_path" --source-manifest "$source_manifest" --output "$preflight_source_attestation"
    bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$preflight_label" --experiment "$preflight_experiment" --run "$preflight_run_name" \
        --worker-gpus 1,2,3 --worker-gpu-mem-util 0.75 \
        --worker-max-model-len 32768 --worker-max-num-seqs 256 \
        --worker-max-num-batched-tokens 8192 --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend triton --target-logprob-chunk-size 2048
    "$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation \
        --plan "$preflight_plan" --target "$target" --topology-profile four-gpu \
        --source "$source_path" --source-manifest "$source_manifest" \
        --source-attestation "$preflight_source_attestation" \
        --worker-parity-attestation "$preflight_worker_parity_attestation" \
        --rmct256-selection "$selection_path" --rmct256-selection-manifest "$selection_manifest" \
        --rmct256-canonical-source "$canonical_source" \
        --rmct256-canonical-source-manifest "$canonical_source_manifest" \
        --rmct256-original64-reference-manifest "$original64_reference_manifest" \
        --rmct256-stage2-manifest "$stage2_manifest" \
        --output "$preflight_target_attestation" --experiment-name "$preflight_experiment" --run-name "$preflight_run_name" \
        --worker-gpus 1,2,3 --worker-gpu-mem-util 0.75 \
        --worker-max-model-len 32768 --worker-max-num-seqs 256 \
        --worker-max-num-batched-tokens 8192 --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend triton --target-logprob-chunk-size 2048
    "$python_bin" -m infra.isambard.rmct256_runtime_policy_receipt validate \
        --repo-root "$repo_root" --receipt "$preflight_runtime_policy_receipt" --segment-index 0
    "$python_bin" "$lora_fingerprint_helper" validate \
        --repo-root "$repo_root" --plan "$plan" --segment-index 0 \
        --base-snapshot-attestation "$preflight_base_snapshot_attestation" \
        --receipt "$preflight_lora_fingerprint"
    "$python_bin" "$contract" seal-preflight \
        --repo-root "$repo_root" --plan "$plan" --segment-index 0
    printf '%s\n' \
        "RMCT256_CONVERGENCE_PREFLIGHT_COMPLETE=1" \
        "preflight_experiment=$preflight_experiment" \
        "preflight_run_name=$preflight_run_name" \
        "preflight_plan=$preflight_plan" \
        "base_snapshot_attestation=$preflight_base_snapshot_attestation" \
        "lora_fingerprint=$preflight_lora_fingerprint" \
        "runtime_policy_receipt=$preflight_runtime_policy_receipt" \
        "preflight_success_receipt=$preflight_success_receipt" \
        "source_attestation=$preflight_source_attestation" \
        "worker_parity_attestation=$preflight_worker_parity_attestation" \
        "target_attestation=$preflight_target_attestation"
    exit 0
fi

# A durable lock serializes a static target even if someone accidentally
# submits a duplicate Slurm job. It lives outside any segment run namespace,
# so it cannot make a missing receipt look like a resumable attempt.
chain_lock="$repo_root/logs/$condition/chain-locks/segment-${segment_index}.lock"
mkdir -p "$(dirname -- "$chain_lock")"
exec 9>"$chain_lock"
if ! flock -n 9; then
    echo "ERROR: another job holds the RMCT256 convergence segment lock: $chain_lock" >&2
    exit 2
fi

# The plateau controller is deliberately evaluated before all successor
# preflight/model work. Its `continue` decision permits work; `converged` and
# `capped` are successful no-ops; a malformed or failed decision is fatal.
plateau_metrics_dir="$repo_root/logs/$condition/plateau-metrics"
plateau_decision_dir="$repo_root/logs/$condition/plateau-decisions"
plateau_decision() {
    local target_pass="$1"
    local output action
    output="$("$python_bin" -m experiments.rmct_256_convergence.plateau guard \
        --metrics-directory "$plateau_metrics_dir" --output-directory "$plateau_decision_dir" \
        --target-pass "$target_pass" --target-segment 3 --selection-sha256 "$selection_sha256")"
    action="$(PLATEAU_OUTPUT="$output" "$python_bin" - <<'PY'
from __future__ import annotations

import json
import os

document = json.loads(os.environ["PLATEAU_OUTPUT"])
decision = document.get("decision")
afterok = document.get("afterok")
if decision not in {"continue", "converged", "capped", "fail"}:
    raise SystemExit("invalid plateau decision")
if not isinstance(afterok, dict) or afterok.get("successor_action") not in {"launch", "no_op", "block"}:
    raise SystemExit("invalid plateau afterok contract")
print(decision)
PY
)"
    printf '%s\n' "$output"
    printf '%s\n' "$action"
}

if (( pass_index > 1 )); then
    previous_action="$(plateau_decision "$((pass_index - 1))" | tail -n 1)"
    case "$previous_action" in
        continue)
            ;;
        converged|capped)
            printf '%s\n' "RMCT256_CONVERGENCE_NOOP_STOP=1" "segment_index=$segment_index" "decision=$previous_action"
            exit 0
            ;;
        *)
            echo "ERROR: previous pass has no safe continue decision: $previous_action" >&2
            exit 2
            ;;
    esac
fi

guard_json="$("$python_bin" "$contract" guard --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index")"
guard_action="$(GUARD_JSON="$guard_json" "$python_bin" - <<'PY'
from __future__ import annotations

import json
import os

value = json.loads(os.environ["GUARD_JSON"])
action = value.get("action")
if action not in {"proceed", "completed"}:
    raise SystemExit("invalid segment guard action")
print(action)
PY
)"
printf '%s\n' "RMCT256_CONVERGENCE_SEGMENT_GUARD=$guard_json"

if [[ "$guard_action" == "proceed" ]]; then
    for frozen_input in \
        "$source_path" "$source_manifest" "$selection_path" "$selection_manifest" \
        "$canonical_source" "$canonical_source_manifest" "$original64_reference_manifest" "$stage2_manifest"; do
        if [[ ! -f "$frozen_input" ]]; then
            echo "ERROR: required immutable RMCT256 input is missing: $frozen_input" >&2
            exit 2
        fi
    done

    # This writes a unique immutable, secret-free cache-resolution sidecar and
    # fails before the worker transport probe if main cannot resolve locally to
    # the pinned Qwen snapshot commit.
    "$python_bin" "$contract" base-snapshot --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index"
    "$python_bin" "$lora_fingerprint_helper" capture \
        --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index" \
        --base-snapshot-attestation "$run_root/preflight/qwen35-base-snapshot-attestation.json" \
        --output "$lora_fingerprint"
    "$python_bin" -m infra.isambard.rmct256_runtime_policy_receipt capture \
        --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index" \
        --output "$runtime_policy_receipt" --reference-receipt "$preflight_runtime_policy_receipt"
    verify_target_contract
    "$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight \
        --source "$source_path" --source-manifest "$source_manifest" --output "$source_attestation"
    bash "$repo_root/infra/vastai/preflight_qwen35_rollout_workers.sh" \
        --label "$label" --experiment "$condition" --run "$run_name" \
        --worker-gpus 1,2,3 --worker-gpu-mem-util 0.75 \
        --worker-max-model-len 32768 --worker-max-num-seqs 256 \
        --worker-max-num-batched-tokens 8192 --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend triton --target-logprob-chunk-size 2048
    "$python_bin" -m experiments.rmct_paper_vast_dense_models.stage1.onpolicy_target_attestation \
        --plan "$plan" --target "$target" --topology-profile four-gpu \
        --source "$source_path" --source-manifest "$source_manifest" \
        --source-attestation "$source_attestation" --worker-parity-attestation "$worker_parity_attestation" \
        --rmct256-selection "$selection_path" --rmct256-selection-manifest "$selection_manifest" \
        --rmct256-canonical-source "$canonical_source" \
        --rmct256-canonical-source-manifest "$canonical_source_manifest" \
        --rmct256-original64-reference-manifest "$original64_reference_manifest" \
        --rmct256-stage2-manifest "$stage2_manifest" \
        --output "$target_attestation" --experiment-name "$condition" --run-name "$run_name" \
        --worker-gpus 1,2,3 --worker-gpu-mem-util 0.75 \
        --worker-max-model-len 32768 --worker-max-num-seqs 256 \
        --worker-max-num-batched-tokens 8192 --worker-seed-base "$worker_seed_base" \
        --worker-gdn-prefill-backend triton --target-logprob-chunk-size 2048

    # The parent was validated before this potentially lengthy worker probe.
    # Re-hash it immediately before the irreversible training-start marker so
    # a changed receipt/checkpoint cannot slip through an otherwise valid
    # target attestation.
    "$python_bin" "$contract" validate-parent --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index"
    "$python_bin" "$lora_fingerprint_helper" validate \
        --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index" \
        --base-snapshot-attestation "$run_root/preflight/qwen35-base-snapshot-attestation.json" \
        --receipt "$lora_fingerprint"
    "$python_bin" -m infra.isambard.rmct256_runtime_policy_receipt validate \
        --repo-root "$repo_root" --receipt "$runtime_policy_receipt" --segment-index "$segment_index"

    TARGET_ATTESTATION_PATH="$target_attestation" \
    BASE_SNAPSHOT_ATTESTATION_PATH="$run_root/preflight/qwen35-base-snapshot-attestation.json" \
    LORA_FINGERPRINT_PATH="$lora_fingerprint" \
    RUNTIME_POLICY_RECEIPT_PATH="$runtime_policy_receipt" \
    TRAINING_STARTED_MARKER_PATH="$training_started_marker" \
    EXPECTED_TARGET="$target" EXPECTED_RUN_NAME="$run_name" EXPECTED_PROFILE="four-gpu" \
    CUDA_VISIBLE_DEVICES="$cuda_visible" "$python_bin" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

target = Path(os.environ["TARGET_ATTESTATION_PATH"]).resolve()
base = Path(os.environ["BASE_SNAPSHOT_ATTESTATION_PATH"]).resolve()
lora = Path(os.environ["LORA_FINGERPRINT_PATH"]).resolve()
runtime_policy = Path(os.environ["RUNTIME_POLICY_RECEIPT_PATH"]).resolve()
marker = Path(os.environ["TRAINING_STARTED_MARKER_PATH"]).resolve()
if not target.is_file() or target.is_symlink():
    raise SystemExit(f"error: target attestation is missing before training marker: {target}")
if not base.is_file() or base.is_symlink():
    raise SystemExit(f"error: base snapshot attestation is missing before training marker: {base}")
if not lora.is_file() or lora.is_symlink():
    raise SystemExit(f"error: LoRA fingerprint is missing before training marker: {lora}")
if not runtime_policy.is_file() or runtime_policy.is_symlink():
    raise SystemExit(f"error: runtime-policy receipt is missing before training marker: {runtime_policy}")
try:
    document = json.loads(target.read_text(encoding="utf-8"))
except json.JSONDecodeError as exc:
    raise SystemExit(f"error: target attestation is invalid before training marker: {target}") from exc
if not isinstance(document, dict) or document.get("schema") != "qwen35-onpolicy-target-attestation-v1":
    raise SystemExit(f"error: target attestation has unexpected schema before training marker: {target}")
for key, expected in {
    "target": os.environ["EXPECTED_TARGET"],
    "run_name": os.environ["EXPECTED_RUN_NAME"],
    "topology_profile": os.environ["EXPECTED_PROFILE"],
}.items():
    if document.get(key) != expected:
        raise SystemExit(f"error: target attestation {key} differs from the static segment contract")
payload = (json.dumps(
    {
        "schema": "rmct256-convergence-training-started-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_attestation": {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()},
        "base_snapshot_attestation": {"path": str(base), "sha256": hashlib.sha256(base.read_bytes()).hexdigest()},
        "lora_fingerprint_attestation": {"path": str(lora), "sha256": hashlib.sha256(lora.read_bytes()).hexdigest()},
        "runtime_policy_receipt": {"path": str(runtime_policy), "sha256": hashlib.sha256(runtime_policy.read_bytes()).hexdigest()},
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
print("RMCT256_CONVERGENCE_TRAINING_STARTED_MARKER=" + str(marker))
PY

    # Do not exec: a zero-returning runner must be followed by receipt sealing.
    # If this process is killed between the two, the marker/output residue but
    # no receipt makes every later segment fail closed before model startup.
    "$python_bin" scripts/run_experiment.py "$plan_rel" \
        --topology-profile four-gpu --stages training --target "$target" --parallel 1 \
        --onpolicy-target-attestation "$target_attestation" --yes
    "$python_bin" "$contract" seal --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index"
else
    printf '%s\n' "RMCT256_CONVERGENCE_NOOP_COMPLETED=1" "segment_index=$segment_index"
fi

# Metrics are extracted only from the trainer's authoritative JSONL after a
# valid final completion receipt exists. The controller re-reads the sealed
# normalized file and its raw-source/receipt hashes; no fixed /96 denominator
# or synthetic success-count reconstruction is used here.
metric_source_receipt_json="$("$python_bin" "$metric_extractor" --repo-root "$repo_root" --plan "$plan" --segment-index "$segment_index")"
metric_source_receipt="$(METRIC_SOURCE_RECEIPT_JSON="$metric_source_receipt_json" "$python_bin" - <<'PY'
from __future__ import annotations

import json
import os

document = json.loads(os.environ["METRIC_SOURCE_RECEIPT_JSON"])
path = document.get("source_receipt")
if not isinstance(path, str) or not path:
    raise SystemExit("metric extractor did not publish a source receipt path")
print(path)
PY
)"
printf '%s\n' "RMCT256_CONVERGENCE_METRIC_SOURCE=$metric_source_receipt_json"
"$python_bin" -m experiments.rmct_256_convergence.plateau publish-metrics \
    --input "$metric_source_receipt" --output-directory "$plateau_metrics_dir"

# A pass decision must be materialized only after all four segment receipts.
# Valid stop outcomes deliberately return success so every already-submitted
# afterok successor reaches its own CPU-only no-op guard rather than running.
if (( block_index == 3 )); then
    boundary_action="$(plateau_decision "$pass_index" | tail -n 1)"
    case "$boundary_action" in
        continue|converged|capped)
            printf '%s\n' "RMCT256_CONVERGENCE_PASS_DECISION=$boundary_action" "pass_index=$pass_index"
            ;;
        *)
            echo "ERROR: plateau controller did not issue a safe boundary decision: $boundary_action" >&2
            exit 2
            ;;
    esac
fi
