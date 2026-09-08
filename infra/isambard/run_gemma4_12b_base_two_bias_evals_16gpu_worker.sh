#!/usr/bin/env bash
# One rank of the 16-way source-ID-sharded Gemma 4 base evaluation.

set -euo pipefail
umask 077

if (( $# != 4 )); then
    echo "usage: $0 <launcher.py> <python> <campaign-root> <cache-parent>" >&2
    exit 2
fi

launcher=$1
python_bin=$2
campaign_root=$3
cache_parent=$4

if [[ ! -f "$launcher" || -L "$launcher" || ! -x "$python_bin" ]]; then
    echo "ERROR: Gemma worker launcher or Python is unavailable" >&2
    exit 2
fi
if [[ ! "$cache_parent" = /* || ! -d "$cache_parent" || -L "$cache_parent" ]]; then
    echo "ERROR: Gemma worker cache parent must be an existing regular absolute directory" >&2
    exit 2
fi

rank=${SLURM_PROCID:-}
if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( 10#$rank > 15 )); then
    echo "ERROR: Gemma worker requires a Slurm rank in [0, 15]" >&2
    exit 2
fi

"$python_bin" - <<'PY'
import os

tokens = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(character.isspace() for character in tokens[0]):
    raise SystemExit("error: every Gemma worker must receive exactly one local Slurm-visible GPU")
PY

worker_tmp=$(mktemp -d "$cache_parent/ctm-gemma4-12b-base-j${SLURM_JOB_ID:-manual}-r${rank}-XXXXXX")
mkdir -p "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
chmod 700 "$worker_tmp" "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
export TMPDIR="$worker_tmp/tmp"
export TMP="$worker_tmp/tmp"
export TEMP="$worker_tmp/tmp"
export XDG_CACHE_HOME="$worker_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$worker_tmp/torchinductor"
export TRITON_CACHE_DIR="$worker_tmp/triton"
export CUDA_CACHE_PATH="$worker_tmp/cuda-cache"

# A one-device CUDA_VISIBLE_DEVICES value does not establish distinct physical
# GPUs. All sixteen ranks must attest unique CUDA UUIDs before weights load.
"$python_bin" "${launcher%/*}/gemma_gpu_binding.py" --campaign-root "$campaign_root"

exec "$python_bin" "$launcher" worker \
    --campaign-root "$campaign_root" \
    --rank "$rank" \
    --python "$python_bin"
