#!/usr/bin/env bash
# Execute only the fresh r4 RMCT recovery windows inside an already allocated
# four-GPU node.  r4 begins at logical segment 4 from sealed r3 s004 (step 64).
# Failed job 6031786 is provenance only: no state from it is ever resumed.
# The only training-runtime delta from r3 is the 49,152 -> 40,960 padded-token
# forward-microbatch cap; all scientific/controller/optimizer arguments stay
# in the frozen compiler.
set -euo pipefail
umask 077

condition="rmct-convergence"
run_prefix="rmct-convergence-gcall-r2-mb40960-r4"
parent_run_prefix="rmct-convergence-gcall-r2-mb49152-r3"
parent_run_name="${parent_run_prefix}-s004"
plan_rel="experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_convergence_gcall_r2_mb40960_r4_isambard_20260818.yaml"
ready_rel="artifacts/rmct-convergence-gcall-r2-mb40960-r4-production-ready.json"
worker_parity_rel="artifacts/rmct-convergence-worker-parity-20260813"
start_segment_index=4
last_segment_index=31

usage() {
    cat <<'EOF'
Usage:
  CUDA_VISIBLE_DEVICES=<four allocated devices> \
    infra/isambard/run_qwen35_rmct_convergence_r4_recovery_deadline.sh --yes \
      [--ready-receipt PATH] [--segment-index 4..31] [--max-segments N] [--dry-run]

This is the immutable fresh r4 recovery.  It begins only at logical segment 4,
resuming sealed r3 s004 at optimizer step 64.  Failed job 6031786 is never a
state parent.  Without --segment-index it starts at the first unsealed r4
segment and may run up to --max-segments (default 28); with --segment-index it
executes at most one sealed 16-optimizer-step window.  Existing verified
decisions are reused.  A terminal predecessor or controller decision is a
successful no-op.  This entrypoint does not submit jobs; --yes is required for
real work, and the pinned model is offline-only.
EOF
}

yes=false
dry_run=false
ready_receipt=""
segment_raw=""
max_segments=28
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes)
            [[ "$yes" == false ]] || { echo "ERROR: --yes was supplied more than once" >&2; exit 2; }
            yes=true; shift ;;
        --dry-run)
            [[ "$dry_run" == false ]] || { echo "ERROR: --dry-run was supplied more than once" >&2; exit 2; }
            dry_run=true; shift ;;
        --ready-receipt)
            [[ -z "$ready_receipt" && $# -ge 2 ]] || { echo "ERROR: --ready-receipt requires one path" >&2; exit 2; }
            ready_receipt="$2"; shift 2 ;;
        --segment-index)
            [[ -z "$segment_raw" && $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --segment-index requires an integer in [4, 31]" >&2; exit 2; }
            segment_raw="$2"; shift 2 ;;
        --max-segments)
            [[ $# -ge 2 && "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --max-segments requires a positive integer" >&2; exit 2; }
            max_segments="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$dry_run" == false && "$yes" == false ]]; then
    echo "ERROR: real production work requires --yes" >&2
    exit 2
fi
if [[ -n "$segment_raw" && "$max_segments" != 28 ]]; then
    echo "ERROR: --segment-index and --max-segments cannot be combined" >&2
    exit 2
fi
if [[ -n "$segment_raw" ]] && (( 10#$segment_raw < start_segment_index || 10#$segment_raw > last_segment_index )); then
    echo "ERROR: --segment-index must be in [4, 31]; r4 never replays an earlier segment" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
plan="$repo_root/$plan_rel"
verify_ready="$repo_root/infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py"
boundary="$repo_root/infra/isambard/rmct_convergence_segment_boundary.py"
worker_parity_helper="$repo_root/infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
worker_parity_dir="$repo_root/$worker_parity_rel"
parent_decisions="$repo_root/logs/$condition/$parent_run_name/decisions"
if [[ -z "$ready_receipt" ]]; then
    ready_receipt="$repo_root/$ready_rel"
fi
for required in "$plan" "$verify_ready" "$boundary" "$worker_parity_helper"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: r4 recovery plan or protected helper is missing or linked: $required" >&2
        exit 2
    fi
done

python_bin="${CTM_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: expected Python interpreter is not executable: $python_bin" >&2
    exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to serialize sealed r4 recovery segments" >&2
    exit 2
fi

validate_four_visible_gpus() {
    "$python_bin" - <<'PY'
import os

tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
if len(tokens) != 4 or any(not value or value in {"-1", "NoDevFiles"} for value in tokens) or len(set(tokens)) != 4:
    raise SystemExit("error: r4 RMCT recovery requires exactly four distinct Slurm-visible GPUs")
PY
}

snapshot_path() {
    "$python_bin" - <<'PY'
import os
from pathlib import Path

revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
hf_home = os.environ.get("HF_HOME")
if not hf_home:
    raise SystemExit("error: HF_HOME must be set to the verified offline Hugging Face cache")
path = Path(hf_home).expanduser().resolve() / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / revision
if path.is_symlink() or not path.is_dir() or not (path / "config.json").is_file():
    raise SystemExit(f"error: pinned offline Qwen snapshot is absent/incomplete: {path}")
print(path)
PY
}

ensure_worker_parity() {
    if [[ ! -d "$worker_parity_dir" || -L "$worker_parity_dir" ]]; then
        echo "ERROR: inherited immutable rollout worker-parity sidecar is absent" >&2
        exit 2
    fi
    "$python_bin" "$worker_parity_helper" \
        --output-dir "$worker_parity_dir" --model-snapshot "$model_snapshot" --resume
}

if [[ "$dry_run" == true ]]; then
    "$python_bin" "$verify_ready" --repository "$repo_root" --receipt "$ready_receipt"
    "$python_bin" -m experiments.rmct_convergence_r4_recovery.plan render \
        --repository "$repo_root" --plan "$plan" --segment-index "$start_segment_index" \
        --model-snapshot "$(snapshot_path)" \
        --output "$repo_root/logs/$condition/${run_prefix}-dry-run-command.json"
    printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_DRY_RUN=1" "run_prefix=$run_prefix" \
        "logical_start_segment_index=$start_segment_index" "local_forward_microbatch_max_tokens=40960"
    exit 0
fi

validate_four_visible_gpus
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
model_snapshot="$(snapshot_path)"
ensure_worker_parity
"$python_bin" "$verify_ready" --repository "$repo_root" --receipt "$ready_receipt"

decision_for() {
    local index="$1"
    local run="$run_prefix-s$(printf '%03d' "$((index + 1))")"
    local decisions="$repo_root/logs/$condition/$run/decisions"
    [[ -d "$decisions" && ! -L "$decisions" ]] || return 1
    "$python_bin" "$boundary" decision --directory "$decisions" --segment-index "$index"
}

decision_path_exists() {
    local index="$1"
    local run="$run_prefix-s$(printf '%03d' "$((index + 1))")"
    local decisions="$repo_root/logs/$condition/$run/decisions"
    [[ -d "$decisions" && ! -L "$decisions" ]] && compgen -G "$decisions/checkpoint-window-decision-s$(printf '%03d' "$index")-*.json" >/dev/null
}

parent_predecessor() {
    "$python_bin" "$boundary" decision --directory "$parent_decisions" --segment-index "$((start_segment_index - 1))"
}

next_unsealed_segment() {
    local index output
    for (( index=start_segment_index; index<=last_segment_index; index++ )); do
        if output="$(decision_for "$index" 2>/dev/null)"; then
            if [[ "$output" == *'"successor_action": "no_op"'* ]]; then
                printf '%s\n' "terminal:$index"
                return 0
            fi
            continue
        fi
        if decision_path_exists "$index"; then
            echo "ERROR: r4 segment $index has an invalid or ambiguous persisted controller receipt" >&2
            return 2
        fi
        printf '%s\n' "$index"
        return 0
    done
    printf '%s\n' "terminal:$last_segment_index"
}

run_segment() {
    local index="$1"
    if (( index < start_segment_index || index > last_segment_index )); then
        echo "ERROR: r4 launcher accepts logical segments [4, 31] only" >&2
        return 2
    fi
    local run="${run_prefix}-s$(printf '%03d' "$((index + 1))")"
    local run_dir="$repo_root/logs/$condition/$run"
    local segment_dir="$run_dir/segment"
    local decisions="$run_dir/decisions"
    local marker="$segment_dir/training-started.json"
    local command_attestation="$segment_dir/training-command.json"
    local checkpoint_receipt="$segment_dir/checkpoint-receipt.json"
    local completion_receipt="$segment_dir/completion-receipt.json"
    local source_metrics="$segment_dir/rmct-convergence-source-metrics.json"
    local checkpoint="$run_dir/checkpoints/${condition}_${run}"
    local predecessor policy

    if policy="$(decision_for "$index" 2>/dev/null)"; then
        printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_SEGMENT_REUSED=1" "segment_index=$index" "$policy"
        return 0
    fi
    if decision_path_exists "$index"; then
        echo "ERROR: r4 segment $index has an invalid or ambiguous persisted controller receipt" >&2
        return 2
    fi
    if (( index == start_segment_index )); then
        predecessor="$(parent_predecessor)"
        if [[ "$predecessor" == *'"successor_action": "no_op"'* ]]; then
            printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_PARENT_NOOP=1" "segment_index=$index" "$predecessor"
            return 0
        fi
    else
        local previous=$((index - 1))
        local previous_run="${run_prefix}-s$(printf '%03d' "$((previous + 1))")"
        local previous_decisions="$repo_root/logs/$condition/$previous_run/decisions"
        predecessor="$("$python_bin" "$boundary" decision --directory "$previous_decisions" --segment-index "$previous")"
        if [[ "$predecessor" == *'"successor_action": "no_op"'* ]]; then
            printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_SUCCESSOR_NOOP=1" "segment_index=$index" "$predecessor"
            return 0
        fi
    fi
    predecessor="$(printf '%s\n' "$predecessor" | "$python_bin" -c 'import json,sys; print(json.load(sys.stdin)["path"])')"

    mkdir -p "$segment_dir"
    local lock="$segment_dir/segment.lock"
    exec 9>"$lock"
    if ! flock -n 9; then
        echo "ERROR: another process owns sealed r4 recovery segment $index" >&2
        return 2
    fi

    # A directory alone is not a valid recovery checkpoint.  It must prove the
    # r4 command, marker, and readiness receipt before it can be sealed.
    if [[ -L "$checkpoint" ]]; then
        echo "ERROR: r4 segment $index checkpoint path is linked; refuse recovery" >&2
        return 2
    elif [[ -e "$checkpoint" ]]; then
        "$python_bin" "$verify_ready" verify-sealed-segment-custody \
            --repository "$repo_root" --ready-receipt "$ready_receipt" --segment-index "$index"
        "$python_bin" "$boundary" seal --checkpoint "$checkpoint" --segment-index "$index" \
            --checkpoint-receipt "$checkpoint_receipt" --completion-receipt "$completion_receipt"
    elif [[ -e "$marker" || -L "$marker" ]]; then
        echo "ERROR: r4 segment $index has a start marker but no sealed final checkpoint; refuse ambiguous replay" >&2
        return 2
    else
        "$python_bin" -m experiments.rmct_convergence_r4_recovery.plan render \
            --repository "$repo_root" --plan "$plan" --segment-index "$index" \
            --model-snapshot "$model_snapshot" --output "$command_attestation"
        "$python_bin" "$boundary" mark --path "$marker" --segment-index "$index" \
            --command-attestation "$command_attestation" --ready-receipt "$ready_receipt"
        "$python_bin" -m experiments.rmct_convergence_r4_recovery.plan execute \
            --repository "$repo_root" --plan "$plan" --segment-index "$index" \
            --model-snapshot "$model_snapshot" --output "$command_attestation" --yes
        "$python_bin" "$verify_ready" verify-sealed-segment-custody \
            --repository "$repo_root" --ready-receipt "$ready_receipt" --segment-index "$index"
        "$python_bin" "$boundary" seal --checkpoint "$checkpoint" --segment-index "$index" \
            --checkpoint-receipt "$checkpoint_receipt" --completion-receipt "$completion_receipt"
    fi

    "$python_bin" -m experiments.rmct_convergence.controller extract-source \
        --metrics-jsonl "$run_dir/metrics.jsonl" --segment-index "$index" --output "$source_metrics"
    "$python_bin" -m experiments.rmct_convergence.controller guard \
        --source-metrics "$source_metrics" --checkpoint-receipt "$checkpoint_receipt" \
        --completion-receipt "$completion_receipt" --output-directory "$decisions" \
        --predecessor-receipt "$predecessor" --max-optimizer-steps 512
    "$python_bin" "$boundary" decision --directory "$decisions" --segment-index "$index"
}

if [[ -n "$segment_raw" ]]; then
    run_segment "$((10#$segment_raw))"
    exit 0
fi

for (( consumed=0; consumed<max_segments; consumed++ )); do
    next="$(next_unsealed_segment)"
    if [[ "$next" == terminal:* ]]; then
        printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_CHAIN_NOOP=1" "reason=controller_terminal_or_cap" "state=$next"
        exit 0
    fi
    run_segment "$next"
    if policy="$(decision_for "$next")" && [[ "$policy" == *'"successor_action": "no_op"'* ]]; then
        printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_CHAIN_NOOP=1" "reason=controller_terminal_or_cap" "$policy"
        exit 0
    fi
done

printf '%s\n' "RMCT_CONVERGENCE_R4_RECOVERY_CHAIN_LIMIT_REACHED=1" "max_segments=$max_segments"
