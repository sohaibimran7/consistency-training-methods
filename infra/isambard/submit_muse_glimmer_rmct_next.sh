#!/usr/bin/env bash
# Submit exactly the next unsealed Muse RMCT segment after the parity gate.
set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd -P)}"
ctm_muse_scratch="${SCRATCHDIR:?SCRATCHDIR is required for the Phase-2 Muse runtime}"
ctm_muse_reservation="${MUSE_SLURM_RESERVATION:-interactive}"
cd "$ctm_muse_repo"
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="${HF_HOME:-$PROJECTDIR/$USER/cache/muse-glimmer-rmct/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ctm_muse_scratch/ctm/uv-cache-muse-cu129}"
source infra/isambard/muse_glimmer_cuda129_runtime_env.sh

ctm_muse_python="$CTM_MUSE_RUNTIME_PYTHON"
ctm_muse_next="$($ctm_muse_python - "$ctm_muse_repo" <<'PY'
import json
import sys
from pathlib import Path

from experiments.muse_glimmer_rmct_replication import plan
from infra.isambard import muse_glimmer_rmct_segment_contract as contract

root = Path(sys.argv[1]).resolve()
contract.validate_preflight(root)
next_index = None
first_passing = None
last_sealed = None
for index in range(plan.TOTAL_SEGMENTS):
    receipt = contract.receipt_path(root, index)
    if not receipt.exists():
        next_index = index
        break
    contract.validate_receipt(root, index)
    convergence = contract.validate_convergence(root, index)
    last_sealed = index
    if first_passing is None and convergence["passed"] is True:
        first_passing = index

completed_step = 0 if last_sealed is None else (last_sealed + 1) * plan.UPDATES_PER_SEGMENT
if first_passing is not None and completed_step >= plan.MINIMUM_COMPARISON_OPTIMIZER_STEP:
    print(json.dumps({
        "action": "complete",
        "final_segment": first_passing,
        "comparison_terminal_segment": last_sealed,
        "comparison_terminal_optimizer_step": completed_step,
    }, sort_keys=True))
    raise SystemExit(0)
if next_index is None:
    print(json.dumps({
        "action": "complete",
        "final_segment": first_passing if first_passing is not None else plan.TOTAL_SEGMENTS - 1,
        "comparison_terminal_segment": plan.TOTAL_SEGMENTS - 1,
        "comparison_terminal_optimizer_step": plan.HARD_CAP_OPTIMIZER_STEPS,
    }, sort_keys=True))
    raise SystemExit(0)

guard = contract.guard(root, next_index)
if guard["action"] != "proceed":
    print(json.dumps(guard, sort_keys=True))
else:
    print(json.dumps({"action": "submit", "segment_index": next_index}, sort_keys=True))
PY
)"
printf '%s\n' "$ctm_muse_next"

ctm_muse_action="$($ctm_muse_python -c 'import json,sys; print(json.loads(sys.argv[1])["action"])' "$ctm_muse_next")"
if [[ "$ctm_muse_action" != "submit" ]]; then
    exit 0
fi
ctm_muse_segment="$($ctm_muse_python -c 'import json,sys; print(json.loads(sys.argv[1])["segment_index"])' "$ctm_muse_next")"

ctm_muse_existing="$(squeue -h -u "$USER" -n muse-rmct -o '%i' | tr '\n' ' ')"
if [[ -n "${ctm_muse_existing// /}" ]]; then
    echo "ERROR: refusing to submit while a Muse RMCT segment job already exists: $ctm_muse_existing" >&2
    exit 2
fi
ctm_muse_existing_interactive="$(squeue -h -u "$USER" -t PENDING,RUNNING -o '%i|%v' | awk -F'|' -v reservation="$ctm_muse_reservation" '$2 == reservation {print $1}' | tr '\n' ' ')"
if [[ -n "${ctm_muse_existing_interactive// /}" ]]; then
    echo "ERROR: the Phase-2 interactive reservation permits only one queued/running job per user: $ctm_muse_existing_interactive" >&2
    exit 2
fi

ctm_muse_exports="ALL,REPO_DIR=$ctm_muse_repo,SCRATCHDIR=$ctm_muse_scratch,HF_HOME=$HF_HOME,UV_CACHE_DIR=$UV_CACHE_DIR,SEGMENT_INDEX=$ctm_muse_segment"
ctm_muse_scheduler=(--reservation="$ctm_muse_reservation" --time=08:00:00)
sbatch --test-only "${ctm_muse_scheduler[@]}" --export="$ctm_muse_exports" \
    infra/isambard/run_muse_glimmer_rmct_segment.sbatch
ctm_muse_job_id="$(sbatch --parsable "${ctm_muse_scheduler[@]}" --export="$ctm_muse_exports" \
    infra/isambard/run_muse_glimmer_rmct_segment.sbatch)"
printf 'CTM_MUSE_SEGMENT_JOB=%s\n' "$ctm_muse_job_id"
printf 'CTM_MUSE_SEGMENT_INDEX=%s\n' "$ctm_muse_segment"
