#!/usr/bin/env bash
# Submit the remaining r5 windows as a strictly sequential afterok chain.
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
ctm_r5_python="$ctm_r5_repo/.venv/bin/python"
ctm_r5_amendment="$ctm_r5_repo/artifacts/rmct-convergence-gcall-r2-mb40960-uncapped-patience-r5-20260825/amendment-v2.json"
ctm_r5_sbatch="$ctm_r5_repo/infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.sbatch"

"$ctm_r5_python" -m experiments.rmct_convergence.patience verify-amendment \
    --repository "$ctm_r5_repo" --amendment "$ctm_r5_amendment" >/dev/null

ctm_r5_next="$($ctm_r5_python - "$ctm_r5_repo" <<'PY'
import json
import sys
from pathlib import Path

from experiments.rmct_convergence import patience
from experiments.rmct_convergence_r5_patience import plan

root = Path(sys.argv[1]).resolve()
for index in range(plan.START_SEGMENT_INDEX, plan.TOTAL_SEGMENTS):
    directory = root / "logs" / plan.CONDITION_NAME / plan.run_name(index) / "patience-decisions"
    if not directory.exists():
        print(json.dumps({"action": "submit", "segment_index": index}, sort_keys=True))
        break
    decision = patience.decision_in_directory(root, directory, index)
    if decision["decision"] != "continue":
        print(json.dumps({"action": "complete", "segment_index": index, **decision}, sort_keys=True))
        break
else:
    print(json.dumps({"action": "complete", "segment_index": plan.TOTAL_SEGMENTS - 1}, sort_keys=True))
PY
)"
printf '%s\n' "$ctm_r5_next"
ctm_r5_action="$($ctm_r5_python -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "$ctm_r5_next")"
if [[ "$ctm_r5_action" != "submit" ]]; then
    exit 0
fi
ctm_r5_start="$($ctm_r5_python -c 'import json,sys; print(json.loads(sys.argv[1])["segment_index"])' "$ctm_r5_next")"

ctm_r5_existing="$(squeue -h -u "$USER" -n ctm-rmct-r5-patience -o '%i' | tr '\n' ' ')"
if [[ -n "${ctm_r5_existing// /}" ]]; then
    echo "ERROR: r5 patience jobs already queued or running: $ctm_r5_existing" >&2
    exit 2
fi

ctm_r5_dependency=""
ctm_r5_jobs=()
for ((ctm_r5_index=ctm_r5_start; ctm_r5_index<=31; ctm_r5_index++)); do
    ctm_r5_exports="ALL,REPO_DIR=$ctm_r5_repo,SCRATCHDIR=$ctm_r5_scratch,CTM_RMCT_SEGMENT_INDEX=$ctm_r5_index"
    ctm_r5_scheduler=()
    if [[ "$ctm_r5_yes" == true && -n "$ctm_r5_dependency" ]]; then
        ctm_r5_scheduler+=("--dependency=afterok:$ctm_r5_dependency")
    fi
    sbatch --test-only "${ctm_r5_scheduler[@]}" --export="$ctm_r5_exports" "$ctm_r5_sbatch" >/dev/null
    if [[ "$ctm_r5_yes" == false ]]; then
        printf 'DRY_RUN_RMCT_R5_SEGMENT=%s dependency=%s\n' "$ctm_r5_index" "${ctm_r5_dependency:-none}"
        continue
    fi
    ctm_r5_job="$(sbatch --parsable "${ctm_r5_scheduler[@]}" --export="$ctm_r5_exports" "$ctm_r5_sbatch")"
    ctm_r5_jobs+=("$ctm_r5_job")
    ctm_r5_dependency="$ctm_r5_job"
    printf 'CTM_RMCT_R5_SEGMENT_JOB=%s segment_index=%s optimizer_step=%s\n' \
        "$ctm_r5_job" "$ctm_r5_index" "$(((ctm_r5_index + 1) * 16))"
done

if [[ "$ctm_r5_yes" == true ]]; then
    printf 'CTM_RMCT_R5_CHAIN=%s\n' "$(IFS=,; echo "${ctm_r5_jobs[*]}")"
fi
