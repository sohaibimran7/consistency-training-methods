#!/usr/bin/env bash
# Submit the 32 sealed RMCT-convergence segments with afterok dependencies.
# This does not submit a topology benchmark: phase-shared four-lane execution
# is the approved deadline configuration. The direct deadline broker uses the
# allocated-node launcher instead of this submitter.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  REPO_DIR=/durable/checkout \
    infra/isambard/submit_qwen35_rmct_convergence_chain.sh --yes [--from-segment 0..31]

The command only queues work; each segment still verifies the immutable
readiness receipt and controller predecessor decision before model startup.
EOF
}

yes=false
start=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes) [[ "$yes" == false ]] || { echo "ERROR: duplicate --yes" >&2; exit 2; }; yes=true; shift ;;
        --from-segment) [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo "ERROR: --from-segment needs 0..31" >&2; exit 2; }; start="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ "$yes" == false ]]; then
    echo "ERROR: submission requires --yes" >&2
    exit 2
fi
if (( 10#$start > 31 )); then
    echo "ERROR: --from-segment must be in [0, 31]" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" ]]; then
    echo "ERROR: set REPO_DIR to a durable repository checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
verify="$repo_root/infra/isambard/verify_rmct_convergence_production_ready.py"
receipt="$repo_root/artifacts/rmct-convergence-production-ready.json"
sbatch_script="$repo_root/infra/isambard/run_qwen35_rmct_convergence_segment.sbatch"
python_bin="${CTM_PYTHON:-$repo_root/.venv/bin/python}"
"$python_bin" "$verify" --repository "$repo_root" --receipt "$receipt"

previous_job=""
for (( index=10#$start; index<32; index++ )); do
    args=(--parsable --export="ALL,REPO_DIR=$repo_root")
    if [[ -n "$previous_job" ]]; then
        args+=(--dependency="afterok:$previous_job")
    fi
    job="$(sbatch "${args[@]}" "$sbatch_script" "$index")"
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "ERROR: sbatch did not return a numeric job ID for segment $index: $job" >&2
        exit 2
    fi
    printf '%s\n' "RMCT_CONVERGENCE_SUBMITTED_SEGMENT=$index" "job_id=$job"
    previous_job="$job"
done
