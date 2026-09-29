#!/usr/bin/env bash
# Login-node submission gate for the one-shot, four-GPU RMCT segment-0 job.
# It waits for source synchronization *before* asking Slurm for GPUs, so an
# incomplete checkout can never idle an allocated four-GH200 node.
set -euo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage:
  REPO_DIR=/durable/checkout \
    infra/isambard/submit_qwen35_rmct_convergence_segment0_when_ready.sh --yes

The login-node gate waits at most 20 minutes (override with
CTM_RMCT_SOURCE_READY_WAIT_SECONDS) for the repository-root regular marker
`.rmct-convergence-production-ready`. Its SHA-256 is passed to the allocated
job, which rechecks it before parity/readiness/training. No GPUs are requested
until the marker exists.
EOF
}

yes=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes) [[ "$yes" == false ]] || { echo "ERROR: duplicate --yes" >&2; exit 2; }; yes=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ "$yes" == false ]]; then
    echo "ERROR: submission requires --yes" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" ]]; then
    echo "ERROR: set REPO_DIR to the durable production checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
python_bin="${CTM_PYTHON:-$repo_root/.venv/bin/python}"
sbatch_script="$repo_root/infra/isambard/run_qwen35_rmct_convergence_segment0_interactive.sbatch"
source_ready="$repo_root/.rmct-convergence-production-ready"
verify_script="$repo_root/infra/isambard/verify_rmct_convergence_production_ready.py"
if [[ ! -x "$python_bin" || ! -f "$sbatch_script" || -L "$sbatch_script" || ! -f "$verify_script" || -L "$verify_script" ]]; then
    echo "ERROR: production submission helper or Isambard Python is absent" >&2
    exit 2
fi

wait_seconds="${CTM_RMCT_SOURCE_READY_WAIT_SECONDS:-1200}"
case "$wait_seconds" in
    ''|*[!0-9]*) echo "ERROR: CTM_RMCT_SOURCE_READY_WAIT_SECONDS must be a nonnegative integer" >&2; exit 2 ;;
esac
deadline=$(( $(date +%s) + wait_seconds ))
while true; do
    if [[ -f "$source_ready" && ! -L "$source_ready" ]]; then
        "$python_bin" "$verify_script" verify-source-ready --repository "$repo_root" --receipt "$source_ready"
        source_ready_sha="$($python_bin - "$source_ready" <<PY
import hashlib
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
PY
)"
        break
    fi
    if (( $(date +%s) >= deadline )); then
        echo "ERROR: source-ready sentinel did not appear within ${wait_seconds}s: $source_ready" >&2
        exit 3
    fi
    sleep 15
done

job="$(sbatch --parsable --export="ALL,REPO_DIR=$repo_root,CTM_RMCT_SOURCE_READY_SHA256=$source_ready_sha" "$sbatch_script")"
if [[ ! "$job" =~ ^[0-9]+([;][A-Za-z0-9_-]+)?$ ]]; then
    echo "ERROR: sbatch did not return a Slurm job ID: $job" >&2
    exit 2
fi
printf '%s\n' "RMCT_CONVERGENCE_SEGMENT0_SUBMITTED=1" "job_id=$job" "source_ready_sha256=$source_ready_sha"
