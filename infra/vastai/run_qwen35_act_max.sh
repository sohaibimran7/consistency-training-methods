#!/usr/bin/env bash
# Fail-closed launcher for the profile-bound ACT-Max data-scaling condition.
# It deliberately rents/provisions nothing: use it only on an already selected
# host after copying the immutable, hash-bound input bundle.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=<one allocated GPU> \
    bash infra/vastai/run_qwen35_act_max.sh \
      --topology-profile single-gpu [--dry-run|--yes]

ACT-Max starts from base Qwen3.5-9B. It uses 2,800 canonical none-style
wrong-argument pairs (1,400 LogiQA + 1,400 HellaSwag), with the frozen Stage-2
held-out IID n=200 population excluded. The launcher first replays the full
offline selection proof, then invokes the target-scoped CPU verifier before
the native Transformers/PEFT ACT training command can initialise a model.

--dry-run performs no write or model initialisation. A real launch requires
--yes and exactly one logical CUDA_VISIBLE_DEVICES token.
EOF
}

topology_profile=""
dry_run=false
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

if [[ "$topology_profile" != "single-gpu" ]]; then
    echo "ERROR: ACT-Max requires --topology-profile single-gpu" >&2
    exit 2
fi
if [[ "$dry_run" == true && "$yes" == true ]]; then
    echo "ERROR: --dry-run cannot be combined with --yes" >&2
    exit 2
fi
if [[ "$dry_run" == false && "$yes" == false ]]; then
    echo "ERROR: a real ACT-Max launch requires --yes" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_act_max_from_base_20260804.yaml"
target="act-max-1gpu"
run_name="act-max-none-n2800-5600steps-lr-1e-4"
experiment="act-max-from-base-20260804"

recovered_none_source_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl"
recovered_none_manifest_rel="artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json"
canonical_prefix_rel="artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-20260801/data/canonical-consistency-pairs-n2048.jsonl"
canonical_prefix_manifest_rel="artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-20260801/data/canonical-consistency-pairs-n2048.manifest.json"
iid_reference_manifest_rel="artifacts/stage1-iid-diagnostic-none-20260801/manifest.json"
iid_heldout_rel="artifacts/stage1-iid-diagnostic-none-20260801/heldout-in-domain-n200.jsonl"
stage2_manifest_rel="artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json"
selection_rel="artifacts/act-max-training-20260804/act-max-training-cec56e4d33531a9f997740850a654e7ceaf16b6c2d4830108861df903660cbd5.jsonl"
selection_manifest_rel="artifacts/act-max-training-20260804/act-max-training-manifest-18b8f70b8260c3d2deaf3345efcfddd41da3e9b7c61ed50d52664bbca56bd04c.json"

plan="$repo_root/$plan_rel"
recovered_none_source="$repo_root/$recovered_none_source_rel"
recovered_none_manifest="$repo_root/$recovered_none_manifest_rel"
canonical_prefix="$repo_root/$canonical_prefix_rel"
canonical_prefix_manifest="$repo_root/$canonical_prefix_manifest_rel"
iid_reference_manifest="$repo_root/$iid_reference_manifest_rel"
iid_heldout="$repo_root/$iid_heldout_rel"
stage2_manifest="$repo_root/$stage2_manifest_rel"
selection="$repo_root/$selection_rel"
selection_manifest="$repo_root/$selection_manifest_rel"

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

validate_visible_gpu() {
    CUDA_VISIBLE_DEVICES="$1" "$python_bin" - <<'PY'
from __future__ import annotations

import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
tokens = [token.strip() for token in visible.split(",")]
if len(tokens) != 1 or not tokens[0] or tokens[0] in {"-1", "NoDevFiles"}:
    raise SystemExit(f"error: ACT-Max single-gpu profile requires exactly one CUDA_VISIBLE_DEVICES token; got {visible!r}")
print("QWEN35_ACT_MAX_GPU_ALLOCATION=" + tokens[0])
PY
}

verify_selection_contract() {
    "$python_bin" -m experiments.act_max.selection verify \
        --recovered-none-source "$recovered_none_source" \
        --recovered-none-manifest "$recovered_none_manifest" \
        --canonical-prefix "$canonical_prefix" \
        --canonical-prefix-manifest "$canonical_prefix_manifest" \
        --iid-reference-manifest "$iid_reference_manifest" \
        --iid-heldout "$iid_heldout" \
        --stage2-manifest "$stage2_manifest" \
        --selection "$selection" \
        --selection-manifest "$selection_manifest"
}

if [[ "$dry_run" == true ]]; then
    cuda_visible="${CUDA_VISIBLE_DEVICES:-0}"
    validate_visible_gpu "$cuda_visible"
    for input in "$plan" "$recovered_none_source" "$recovered_none_manifest" "$canonical_prefix" "$canonical_prefix_manifest" "$iid_reference_manifest" "$iid_heldout" "$stage2_manifest" "$selection" "$selection_manifest"; do
        if [[ ! -f "$input" ]]; then
            echo "ERROR: expected ACT-Max input is missing: $input" >&2
            exit 2
        fi
    done
    printf '%s\n' \
        "QWEN35_ACT_MAX_DRY_RUN=1" \
        "topology_profile=$topology_profile" \
        "target=$target" \
        "experiment_name=$experiment" \
        "run_name=$run_name" \
        "plan=$plan" \
        "selection=$selection" \
        "selection_manifest=$selection_manifest" \
        "cuda_visible_devices=$cuda_visible"
    verify_selection_contract
    CUDA_VISIBLE_DEVICES="$cuda_visible" "$python_bin" scripts/run_experiment.py "$plan_rel" \
        --topology-profile "$topology_profile" --stages data_preparation,training --target "$target" --parallel 1 --dry-run --yes
    exit 0
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: set CUDA_VISIBLE_DEVICES to the single allocated GPU before launching ACT-Max" >&2
    exit 2
fi
validate_visible_gpu "$CUDA_VISIBLE_DEVICES"
for input in "$plan" "$recovered_none_source" "$recovered_none_manifest" "$canonical_prefix" "$canonical_prefix_manifest" "$iid_reference_manifest" "$iid_heldout" "$stage2_manifest" "$selection" "$selection_manifest"; do
    if [[ ! -f "$input" ]]; then
        echo "ERROR: required ACT-Max input is missing: $input" >&2
        exit 2
    fi
done

# This is a complete pre-model proof. The data-preparation command repeats it
# within the resolved target plan, so the run record cannot omit the gate.
verify_selection_contract
exec "$python_bin" scripts/run_experiment.py "$plan_rel" \
    --topology-profile "$topology_profile" --stages data_preparation,training --target "$target" --parallel 1 --yes
