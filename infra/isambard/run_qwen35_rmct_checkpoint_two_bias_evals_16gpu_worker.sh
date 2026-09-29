#!/usr/bin/env bash
# One rank of the sealed 16-GPU early-checkpoint evaluation campaign.
#
# This file is invoked once on every rank of a single 16-rank srun phase.  It
# deliberately maps rank -> (checkpoint step, task index) itself, rather than
# deriving output names from CUDA_VISIBLE_DEVICES: local GPU token "0" is valid
# simultaneously on all four nodes.

set -euo pipefail
umask 077

if (( $# != 7 )); then
    echo "usage: $0 <launcher.py> <python> <phase> <campaign-root> <step16-root> <step64-root> <worker-cache-parent>" >&2
    exit 2
fi

launcher=$1
python_bin=$2
phase=$3
campaign_root=$4
step16_root=$5
step64_root=$6
cache_parent=$7

if [[ ! -f "$launcher" || -L "$launcher" ]]; then
    echo "ERROR: worker launcher is missing or linked: $launcher" >&2
    exit 2
fi
# The approved evaluation venv's bin/python is normally a symlink to its
# interpreter.  Preserve that spelling so sys.prefix remains the venv, while
# still rejecting missing/non-executable Python paths.
if [[ ! -e "$python_bin" || ! -x "$python_bin" ]]; then
    echo "ERROR: worker Python is missing or not executable: $python_bin" >&2
    exit 2
fi
if [[ ! "$phase" =~ ^[123]$ ]]; then
    echo "ERROR: phase must be exactly 1, 2, or 3" >&2
    exit 2
fi

rank=${SLURM_PROCID:-}
if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( 10#$rank > 15 )); then
    echo "ERROR: worker requires a Slurm rank in [0, 15]" >&2
    exit 2
fi

# Exactly fourteen ranks carry cells.  Ranks 14/15 intentionally exit with no
# work: their allocation slots remain unassigned by the phase plan, preserving
# two-GPU headroom without inventing an automatic retry policy.
step=''
task_index=''
case "$phase:$rank" in
    1:0) step=16; task_index=1 ;;  1:1) step=16; task_index=2 ;;
    1:2) step=16; task_index=3 ;;  1:3) step=16; task_index=4 ;;
    1:4) step=16; task_index=5 ;;  1:5) step=16; task_index=6 ;;
    1:6) step=16; task_index=7 ;;  1:7) step=16; task_index=8 ;;
    1:8) step=16; task_index=9 ;;  1:9) step=16; task_index=10 ;;
    1:10) step=16; task_index=11 ;; 1:11) step=16; task_index=12 ;;
    1:12) step=16; task_index=13 ;; 1:13) step=16; task_index=14 ;;

    2:0) step=16; task_index=15 ;; 2:1) step=16; task_index=16 ;;
    2:2) step=16; task_index=17 ;; 2:3) step=16; task_index=18 ;;
    2:4) step=16; task_index=19 ;; 2:5) step=16; task_index=20 ;;
    2:6) step=16; task_index=21 ;; 2:7) step=64; task_index=1 ;;
    2:8) step=64; task_index=2 ;; 2:9) step=64; task_index=3 ;;
    2:10) step=64; task_index=4 ;; 2:11) step=64; task_index=5 ;;
    2:12) step=64; task_index=6 ;; 2:13) step=64; task_index=7 ;;

    3:0) step=64; task_index=8 ;;  3:1) step=64; task_index=9 ;;
    3:2) step=64; task_index=10 ;; 3:3) step=64; task_index=11 ;;
    3:4) step=64; task_index=12 ;; 3:5) step=64; task_index=13 ;;
    3:6) step=64; task_index=14 ;; 3:7) step=64; task_index=15 ;;
    3:8) step=64; task_index=16 ;; 3:9) step=64; task_index=17 ;;
    3:10) step=64; task_index=18 ;; 3:11) step=64; task_index=19 ;;
    3:12) step=64; task_index=20 ;; 3:13) step=64; task_index=21 ;;
    *) exit 0 ;;
esac

"$python_bin" - <<'PY'
import os

tokens = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(character.isspace() for character in tokens[0]):
    raise SystemExit("error: every campaign worker rank must receive exactly one local Slurm-visible GPU")
if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
    raise SystemExit("error: VLLM_USE_FLASHINFER_SAMPLER must be exactly 0")
PY

# Every rank gets node/job/rank-local writable compiler/cache paths.  No cache
# path is keyed by the CUDA token, because local token spellings repeat across
# nodes.  We intentionally leave this local evidence in place on failure.
mkdir -p "$cache_parent"
worker_tmp=$(mktemp -d "$cache_parent/ctm-rmct-eval-p${phase}-s${step}-t${task_index}-j${SLURM_JOB_ID:-manual}-r${rank}-XXXXXX")
mkdir -p "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
chmod 700 "$worker_tmp" "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
export TMPDIR="$worker_tmp/tmp"
export XDG_CACHE_HOME="$worker_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$worker_tmp/torchinductor"
export TRITON_CACHE_DIR="$worker_tmp/triton"
export CUDA_CACHE_PATH="$worker_tmp/cuda-cache"

if (( step == 16 )); then
    output_root=$step16_root
else
    output_root=$step64_root
fi

exec "$python_bin" "$launcher" worker \
    --step "$step" \
    --output-root "$output_root" \
    --task-index "$task_index" \
    --python "$python_bin" \
    --phase "$phase" \
    --campaign-root "$campaign_root" \
    --step16-root "$step16_root" \
    --step64-root "$step64_root"
