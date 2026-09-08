#!/usr/bin/env bash
# One rank in the all-HF/PEFT matched two-bias campaign.
#
# The sbatch wrapper owns phase sequencing.  This wrapper maps a local Slurm
# rank to exactly one frozen condition/task cell in the clean wave or one of
# five balanced biased waves, and deliberately has no retry/scheduler
# behaviour of its own.

set -euo pipefail
umask 077

if (( $# != 8 )); then
    echo "usage: $0 <launcher.py> <python> <clean|biased-1|biased-2|biased-3|biased-4|biased-5> <campaign-root> <training-repository> <source-stage2-manifest> <stage2-artifact-root> <worker-cache-parent>" >&2
    exit 2
fi

launcher=$1
python_bin=$2
phase=$3
campaign_root=$4
training_repository=$5
source_manifest=$6
artifact_root=$7
cache_parent=$8

if [[ ! -f "$launcher" || -L "$launcher" || ! -e "$python_bin" || ! -x "$python_bin" ]]; then
    echo "ERROR: all-HF launcher or Python is unavailable" >&2
    exit 2
fi
if [[ ! "$cache_parent" = /* || ! -d "$cache_parent" ]]; then
    echo "ERROR: worker cache parent must be an existing absolute directory" >&2
    exit 2
fi
case "$phase" in
    clean|biased-1|biased-2|biased-3|biased-4|biased-5) ;;
    *) echo "ERROR: phase must be clean or biased-1 through biased-5" >&2; exit 2 ;;
esac

rank=${SLURM_PROCID:-}
if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( 10#$rank > 15 )); then
    echo "ERROR: worker requires a Slurm rank in [0, 15]" >&2
    exit 2
fi

condition=''
task_index=''
case "$phase:$rank" in
    clean:0) condition=base; task_index=1 ;;
    clean:1) condition=base; task_index=2 ;;
    clean:2) condition=base; task_index=3 ;;
    clean:3) condition=step016; task_index=1 ;;
    clean:4) condition=step016; task_index=2 ;;
    clean:5) condition=step016; task_index=3 ;;
    clean:6) condition=step064; task_index=1 ;;
    clean:7) condition=step064; task_index=2 ;;
    clean:8) condition=step064; task_index=3 ;;
    clean:9) condition=step176; task_index=1 ;;
    clean:10) condition=step176; task_index=2 ;;
    clean:11) condition=step176; task_index=3 ;;
    clean:*) exit 0 ;;

    biased-1:0) condition=base; task_index=6 ;;
    biased-1:1) condition=base; task_index=17 ;;
    biased-1:2) condition=step016; task_index=6 ;;
    biased-1:3) condition=step016; task_index=17 ;;
    biased-1:4) condition=step064; task_index=6 ;;
    biased-1:5) condition=step064; task_index=17 ;;
    biased-1:6) condition=step176; task_index=6 ;;
    biased-1:7) condition=step176; task_index=17 ;;
    biased-1:8) condition=base; task_index=18 ;;
    biased-1:9) condition=base; task_index=19 ;;
    biased-1:10) condition=base; task_index=20 ;;
    biased-1:11) condition=base; task_index=21 ;;
    biased-1:12) condition=step064; task_index=18 ;;
    biased-1:13) condition=step064; task_index=19 ;;
    biased-1:14) condition=step064; task_index=20 ;;
    biased-1:15) condition=step064; task_index=21 ;;

    biased-2:0) condition=base; task_index=4 ;;
    biased-2:1) condition=base; task_index=5 ;;
    biased-2:2) condition=step016; task_index=4 ;;
    biased-2:3) condition=step016; task_index=5 ;;
    biased-2:4) condition=step064; task_index=4 ;;
    biased-2:5) condition=step064; task_index=5 ;;
    biased-2:6) condition=step176; task_index=4 ;;
    biased-2:7) condition=step176; task_index=5 ;;
    biased-2:8) condition=step016; task_index=18 ;;
    biased-2:9) condition=step016; task_index=19 ;;
    biased-2:10) condition=step016; task_index=20 ;;
    biased-2:11) condition=step016; task_index=21 ;;
    biased-2:12) condition=step176; task_index=18 ;;
    biased-2:13) condition=step176; task_index=19 ;;
    biased-2:14) condition=step176; task_index=20 ;;
    biased-2:15) condition=step176; task_index=21 ;;

    biased-3:0) condition=base; task_index=7 ;;
    biased-3:1) condition=base; task_index=8 ;;
    biased-3:2) condition=step016; task_index=7 ;;
    biased-3:3) condition=step016; task_index=8 ;;
    biased-3:4) condition=step064; task_index=7 ;;
    biased-3:5) condition=step064; task_index=8 ;;
    biased-3:6) condition=step176; task_index=7 ;;
    biased-3:7) condition=step176; task_index=8 ;;
    biased-3:8) condition=base; task_index=13 ;;
    biased-3:9) condition=base; task_index=15 ;;
    biased-3:10) condition=step016; task_index=13 ;;
    biased-3:11) condition=step016; task_index=15 ;;
    biased-3:12) condition=step064; task_index=13 ;;
    biased-3:13) condition=step064; task_index=15 ;;
    biased-3:14) condition=step176; task_index=13 ;;
    biased-3:15) condition=step176; task_index=15 ;;

    biased-4:0) condition=base; task_index=9 ;;
    biased-4:1) condition=base; task_index=10 ;;
    biased-4:2) condition=step016; task_index=9 ;;
    biased-4:3) condition=step016; task_index=10 ;;
    biased-4:4) condition=step064; task_index=9 ;;
    biased-4:5) condition=step064; task_index=10 ;;
    biased-4:6) condition=step176; task_index=9 ;;
    biased-4:7) condition=step176; task_index=10 ;;
    biased-4:8) condition=base; task_index=14 ;;
    biased-4:9) condition=base; task_index=16 ;;
    biased-4:10) condition=step016; task_index=14 ;;
    biased-4:11) condition=step016; task_index=16 ;;
    biased-4:12) condition=step064; task_index=14 ;;
    biased-4:13) condition=step064; task_index=16 ;;
    biased-4:14) condition=step176; task_index=14 ;;
    biased-4:15) condition=step176; task_index=16 ;;

    biased-5:0) condition=base; task_index=11 ;;
    biased-5:1) condition=base; task_index=12 ;;
    biased-5:2) condition=step016; task_index=11 ;;
    biased-5:3) condition=step016; task_index=12 ;;
    biased-5:4) condition=step064; task_index=11 ;;
    biased-5:5) condition=step064; task_index=12 ;;
    biased-5:6) condition=step176; task_index=11 ;;
    biased-5:7) condition=step176; task_index=12 ;;
    biased-5:*) exit 0 ;;
esac

# Replace hostile inherited compiler/cache paths before any Python process can
# import Torch, Transformers, PEFT, or Inspect.  This rank-local cache is
# intentionally retained after failure and cannot use a global CUDA token:
# every node may expose its local device as CUDA_VISIBLE_DEVICES=0.
worker_tmp=$(mktemp -d "$cache_parent/ctm-rmct-all-hf-r002-${phase}-${condition}-j${SLURM_JOB_ID:-manual}-r${rank}-XXXXXX")
mkdir -p "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
chmod 700 "$worker_tmp" "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
export TMPDIR="$worker_tmp/tmp"
export XDG_CACHE_HOME="$worker_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$worker_tmp/torchinductor"
export TRITON_CACHE_DIR="$worker_tmp/triton"
export CUDA_CACHE_PATH="$worker_tmp/cuda-cache"

"$python_bin" - <<'PY'
import os
from importlib.metadata import PackageNotFoundError, version

tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(any(char.isspace() for char in token) for token in tokens):
    raise SystemExit("error: every all-HF worker must receive exactly one local Slurm-visible GPU")
expected = {
    "inspect-ai": "0.3.258",
    "torch": "2.11.0+cu129",
    "transformers": "5.5.4",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
}
try:
    installed = {distribution: version(distribution) for distribution in expected}
except PackageNotFoundError as exc:
    raise SystemExit(f"error: all-HF worker has a missing pinned evaluator package: {exc}") from exc
if installed != expected:
    raise SystemExit(f"error: all-HF worker evaluator versions differ from pinned remote runtime: got={installed!r}, expected={expected!r}")
if os.environ.get("CTM_DISABLE_CUDNN_SDP") != "0":
    raise SystemExit("error: all-HF workers require CTM_DISABLE_CUDNN_SDP=0")
PY

run_task() {
    local selected_task=$1
    "$python_bin" "$launcher" worker \
        --campaign-root "$campaign_root" \
        --training-repository "$training_repository" \
        --source-stage2-manifest "$source_manifest" \
        --stage2-artifact-root "$artifact_root" \
        --condition "$condition" \
        --task-index "$selected_task" \
        --python "$python_bin"
}

run_task "$task_index"
