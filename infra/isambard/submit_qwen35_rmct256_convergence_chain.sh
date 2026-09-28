#!/usr/bin/env bash
# Submit the fixed 16-job RMCT-256 Isambard convergence chain. Slurm afterok
# provides ordering only: every job independently revalidates its immediate
# parent receipt, and terminal plateau outcomes turn all scheduled successors
# into successful CPU-only no-ops before they initialize a model.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash infra/isambard/submit_qwen35_rmct256_convergence_chain.sh \
    --repo-dir /durable/isambard/consistency-training-methods [--dry-run|--yes]

Submits the authored 16 segment indices (0 through 15), each dependent on the
previous one with afterok. `--dry-run` prints the fixed dependency graph and
does not call sbatch. Real submission requires `--yes` and a currently valid,
immutable four-GH200 dedicated-preflight success receipt.

The submitted jobs derive all targets, unique namespaces, slices, parents and
worker seeds internally; this submitter accepts no target/checkpoint/pass
override. It never cancels a submitted job on a later submission failure.
EOF
}

repo_dir=""
dry_run=false
yes=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir)
            if [[ -n "$repo_dir" || $# -lt 2 || -z "$2" ]]; then
                echo "ERROR: --repo-dir requires exactly one non-empty value" >&2
                exit 2
            fi
            repo_dir="$2"
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

if [[ -z "$repo_dir" ]]; then
    echo "ERROR: --repo-dir is required" >&2
    usage >&2
    exit 2
fi
repo_dir="$(cd -- "$repo_dir" && pwd -P)"
plan="$repo_dir/experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct256_convergence_isambard_4x64_20260811.yaml"
sbatch_script="$repo_dir/infra/isambard/run_qwen35_rmct256_convergence_segment.sbatch"
contract="$repo_dir/infra/isambard/rmct256_convergence_segment_contract.py"
if [[ ! -f "$plan" || ! -f "$sbatch_script" || ! -f "$contract" ]]; then
    echo "ERROR: --repo-dir is not the durable checkout with the static RMCT256 convergence chain" >&2
    echo "  plan=$plan" >&2
    echo "  sbatch=$sbatch_script" >&2
    echo "  contract=$contract" >&2
    exit 2
fi
if [[ "$dry_run" == false && "$yes" == false ]]; then
    echo "ERROR: real Slurm submission requires --yes" >&2
    exit 2
fi
if [[ "$dry_run" == false ]]; then
    python_bin="${CTM_PYTHON:-$repo_dir/.venv/bin/python}"
    if [[ ! -x "$python_bin" ]]; then
        echo "ERROR: real chain submission requires the durable GPU-validated Python interpreter: $python_bin" >&2
        exit 2
    fi
    # This CPU-safe verifier rehashes the production and isolated plans, all
    # dedicated-preflight sidecars, the runtime source manifest, and the
    # launcher/submitter code identities. It must complete before even the
    # segment-0 sbatch call; Slurm afterok alone is not scientific custody.
    preflight_gate_json="$("$python_bin" "$contract" validate-preflight \
        --repo-root "$repo_dir" --plan "$plan" --segment-index 0)" || {
        echo "ERROR: refusing to submit RMCT256 convergence jobs without a current dedicated-preflight success receipt." >&2
        exit 2
    }
    printf '%s\n' "RMCT256_CONVERGENCE_PREFLIGHT_GATE=$preflight_gate_json"
fi
if [[ "$dry_run" == false ]] && ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch is not available on PATH" >&2
    exit 2
fi

previous_job_id=""
for segment_index in {0..15}; do
    if [[ "$dry_run" == true ]]; then
        if [[ -z "$previous_job_id" ]]; then
            printf '%s\n' "RMCT256_CONVERGENCE_SUBMIT_DRY_RUN=1 segment_index=$segment_index dependency=none"
        else
            printf '%s\n' "RMCT256_CONVERGENCE_SUBMIT_DRY_RUN=1 segment_index=$segment_index dependency=afterok:$previous_job_id"
        fi
        previous_job_id="dryrun-$segment_index"
        continue
    fi

    submit_args=(--parsable --export="ALL,REPO_DIR=$repo_dir")
    dependency_label="none"
    if [[ -n "$previous_job_id" ]]; then
        submit_args+=("--dependency=afterok:$previous_job_id")
        dependency_label="afterok:$previous_job_id"
    fi
    submit_args+=("$sbatch_script" --segment-index "$segment_index" --yes)
    raw_job_id="$(sbatch "${submit_args[@]}")" || {
        echo "ERROR: sbatch failed while submitting segment $segment_index; earlier jobs remain submitted and are not cancelled." >&2
        exit 2
    }
    if [[ ! "$raw_job_id" =~ ^([0-9]+)(\;[A-Za-z0-9._-]+)?$ ]]; then
        echo "ERROR: sbatch returned an unparseable job id for segment $segment_index: $raw_job_id" >&2
        exit 2
    fi
    previous_job_id="${BASH_REMATCH[1]}"
    printf '%s\n' \
        "RMCT256_CONVERGENCE_SUBMITTED=1" \
        "segment_index=$segment_index" \
        "job_id=$previous_job_id" \
        "dependency=$dependency_label"
done
