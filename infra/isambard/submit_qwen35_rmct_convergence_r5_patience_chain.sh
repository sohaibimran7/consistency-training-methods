#!/usr/bin/env bash
# Submit the next required window; each completed window extends the chain.
set -euo pipefail
umask 077

ctm_r5_yes=false
if [[ "${1:-}" == "--yes" && $# -eq 1 ]]; then
    ctm_r5_yes=true
elif [[ $# -ne 0 ]]; then
    echo "Usage: $0 [--yes]" >&2
    exit 2
fi

ctm_r5_repo="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
ctm_r5_scratch="${SCRATCHDIR:?SCRATCHDIR is required}"
cd "$ctm_r5_repo"
ctm_r5_repo="$(pwd -P)"
ctm_r5_python="${CTM_RMCT_PYTHON:-$ctm_r5_repo/.venv/bin/python}"
ctm_r5_amendment="$ctm_r5_repo/artifacts/rmct-capped-patience-20260910/amendment.json"
ctm_r5_sbatch="$ctm_r5_repo/infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.sbatch"

"$ctm_r5_python" -m experiments.rmct_convergence.patience verify-amendment \
    --repository "$ctm_r5_repo" --amendment "$ctm_r5_amendment" >/dev/null

exec 9>"$ctm_r5_repo/artifacts/rmct-capped-patience-20260910/submission.lock"
flock -n 9 || { echo 'Another submission is in progress' >&2; exit 2; }

ctm_r5_next="$($ctm_r5_python - "$ctm_r5_repo" <<'PY'
import json
import sys
from pathlib import Path

from experiments.rmct_convergence import patience
from experiments.rmct_convergence_r5_patience import plan

root = Path(sys.argv[1]).resolve()
index = plan.START_SEGMENT_INDEX
while True:
    directory = root / "logs" / plan.CONDITION_NAME / plan.run_name(index) / "patience-decisions"
    if not directory.exists():
        print(json.dumps({"action": "submit", "segment_index": index}, sort_keys=True))
        break
    decision = patience.decision_in_directory(root, directory, index)
    if decision["decision"] != "continue":
        print(json.dumps({"action": "complete", "segment_index": index, **decision}, sort_keys=True))
        break
    index += 1
PY
)"
printf '%s\n' "$ctm_r5_next"
ctm_r5_action="$($ctm_r5_python -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "$ctm_r5_next")"
if [[ "$ctm_r5_action" != "submit" ]]; then
    exit 0
fi
ctm_r5_start="$($ctm_r5_python -c 'import json,sys; print(json.loads(sys.argv[1])["segment_index"])' "$ctm_r5_next")"

ctm_r5_existing="$(squeue -h -u "$USER" -n ctm-rmct-patience20k -o '%i' | awk -v current="${SLURM_JOB_ID:-none}" '$1 != current {print $1}' | tr '\n' ' ')"
if [[ -n "${ctm_r5_existing// /}" ]]; then
    echo "ERROR: r5 patience jobs already queued or running: $ctm_r5_existing" >&2
    exit 2
fi

ctm_r5_exports="ALL,REPO_DIR=$ctm_r5_repo,SCRATCHDIR=$ctm_r5_scratch,CTM_RMCT_PYTHON=$ctm_r5_python,CTM_RMCT_SEGMENT_INDEX=$ctm_r5_start"
ctm_r5_scheduler=("--kill-on-invalid-dep=no")
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    ctm_r5_scheduler+=("--dependency=afterok:$SLURM_JOB_ID")
fi
sbatch --test-only "${ctm_r5_scheduler[@]}" --export="$ctm_r5_exports" "$ctm_r5_sbatch" >/dev/null
if [[ "$ctm_r5_yes" == false ]]; then
    printf 'DRY_RUN_RMCT_R5_SEGMENT=%s\n' "$ctm_r5_start"
    exit 0
fi
ctm_r5_job="$(sbatch --parsable "${ctm_r5_scheduler[@]}" --export="$ctm_r5_exports" "$ctm_r5_sbatch")"
printf 'CTM_RMCT_R5_SEGMENT_JOB=%s segment_index=%s optimizer_step=%s\n' \
    "$ctm_r5_job" "$ctm_r5_start" "$(((ctm_r5_start + 1) * 16))"
printf '%s %s\n' "$ctm_r5_start" "$ctm_r5_job" >> "$ctm_r5_repo/artifacts/rmct-capped-patience-20260910/submissions.log"
