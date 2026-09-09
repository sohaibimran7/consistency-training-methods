#!/usr/bin/env bash
# Lightweight login-node coordinator; all training runs inside Slurm allocations.
# Never cancels or changes any scheduled job. Each window gets a new allocation.
set -euo pipefail
umask 077

if (( $# != 3 )); then
    echo "usage: $0 <runtime-python> <isolated-repository> <isolated-amendment>" >&2
    exit 2
fi
ctm_interactive_python=$1
ctm_interactive_repo=$2
ctm_interactive_amendment=$3
for ctm_interactive_file in "$ctm_interactive_python" "$ctm_interactive_amendment"; do
    test -f "$ctm_interactive_file" || exit 2
done
test -d "$ctm_interactive_repo" && test ! -L "$ctm_interactive_repo"
test -x "$ctm_interactive_python"
cd "$ctm_interactive_repo"
ctm_interactive_repo=$(pwd -P)
ctm_interactive_source=$("$ctm_interactive_python" - "$ctm_interactive_repo" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = root / "artifacts/rmct-r5-interactive-duplicate-20260908/preparation.json"
document = json.loads(receipt.read_text())
assert document["duplicate_repository"] == str(root)
source = Path(document["source_repository"])
assert source != root and source.parent == root.parent
print(source)
PY
)
export PYTHONPATH="$ctm_interactive_repo"
export HF_HOME="${SCRATCHDIR:?SCRATCHDIR is required}/ctm/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
ctm_interactive_snapshot="$HF_HOME/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
exec 9>"$ctm_interactive_repo/interactive-broker.lock"
flock -n 9 || { echo "Another interactive broker owns this isolated repository" >&2; exit 2; }

"$ctm_interactive_python" -m experiments.rmct_convergence.patience verify-amendment \
    --repository "$ctm_interactive_repo" --amendment "$ctm_interactive_amendment" >/dev/null

for ctm_interactive_index in {11..31}; do
    # Resume only from a sealed decision. Partial training fails closed in the runner.
    ctm_interactive_state=$("$ctm_interactive_python" - "$ctm_interactive_repo" "$ctm_interactive_index" <<'PY'
import sys
from pathlib import Path
from experiments.rmct_convergence import patience
from experiments.rmct_convergence_r5_patience import plan

root = Path(sys.argv[1])
index = int(sys.argv[2])
directory = root / "logs" / plan.CONDITION_NAME / plan.run_name(index) / "patience-decisions"
if directory.exists():
    decision = patience.decision_in_directory(root, directory, index)
    print("continue" if decision["decision"] == "continue" else "terminal")
else:
    print("run")
PY
)
    if [[ "$ctm_interactive_state" == terminal ]]; then
        echo "CTM_INTERACTIVE_TERMINAL_SEGMENT=$ctm_interactive_index"
        exit 0
    elif [[ "$ctm_interactive_state" == continue ]]; then
        continue
    elif [[ "$ctm_interactive_state" != run ]]; then
        echo "Unknown sealed-decision state" >&2
        exit 2
    fi
    echo "CTM_INTERACTIVE_REQUEST_SEGMENT=$ctm_interactive_index"
    # No explicit interactive partition or QOS: Isambard exposes a reservation.
    srun --reservation=interactive --nodes=1 --ntasks=1 --gpus=4 \
        --cpus-per-task=64 --mem=200G --time=08:00:00 \
        --job-name=ctm-rmct-r5-interactive --unbuffered \
        bash -c '
            set -euo pipefail
            ctm_gpu_tmp=$(mktemp -d /tmp/ctm-rmct-r5-interactive-XXXXXX)
            mkdir -p "$ctm_gpu_tmp/tmp" "$ctm_gpu_tmp/xdg" "$ctm_gpu_tmp/inductor" "$ctm_gpu_tmp/triton" "$ctm_gpu_tmp/cuda"
            export TMPDIR="$ctm_gpu_tmp/tmp" XDG_CACHE_HOME="$ctm_gpu_tmp/xdg"
            export TORCHINDUCTOR_CACHE_DIR="$ctm_gpu_tmp/inductor" TRITON_CACHE_DIR="$ctm_gpu_tmp/triton" CUDA_CACHE_PATH="$ctm_gpu_tmp/cuda"
            echo "CTM_INTERACTIVE_JOB_ID=$SLURM_JOB_ID SEGMENT=$3"
            exec "$1" "$2/infra/isambard/prepare_qwen35_rmct_r5_interactive_duplicate.py" run \
                --source-repository "$5" --duplicate-repository "$2" --runtime-python "$1" \
                --model-snapshot "$6" --max-segments 1 --yes
        ' bash "$ctm_interactive_python" "$ctm_interactive_repo" "$ctm_interactive_index" "$ctm_interactive_amendment" "$ctm_interactive_source" "$ctm_interactive_snapshot"
    ctm_interactive_decision=$("$ctm_interactive_python" - "$ctm_interactive_repo" "$ctm_interactive_index" <<'PY'
import sys
from pathlib import Path
from experiments.rmct_convergence import patience
from experiments.rmct_convergence_r5_patience import plan

root = Path(sys.argv[1])
index = int(sys.argv[2])
directory = root / "logs" / plan.CONDITION_NAME / plan.run_name(index) / "patience-decisions"
print(patience.decision_in_directory(root, directory, index)["decision"])
PY
)
    echo "CTM_INTERACTIVE_DECISION=$ctm_interactive_decision SEGMENT=$ctm_interactive_index"
    if [[ "$ctm_interactive_decision" != continue ]]; then
        exit 0
    fi
done
