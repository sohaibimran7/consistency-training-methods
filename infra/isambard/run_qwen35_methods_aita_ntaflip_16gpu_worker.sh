#!/usr/bin/env bash
# One rank in the two-phase BCT/ACT/AttCT/MLPCT/OPCT AITA replication.
# The sbatch wrapper owns all phase sequencing and scheduler interaction.

set -euo pipefail
umask 077

if (( $# != 7 )); then
    echo "usage: $0 <launcher.py> <python> <phase-1|phase-2> <campaign-root> <official-source> <checkpoint-directory> <worker-cache-parent>" >&2
    exit 2
fi

launcher=$1
python_bin=$2
phase=$3
campaign_root=$4
official_source_dir=$5
checkpoint_directory=$6
cache_parent=$7

if [[ ! -f "$launcher" || -L "$launcher" || ! -e "$python_bin" || ! -x "$python_bin" ]]; then
    echo "ERROR: methods AITA launcher or Python is unavailable" >&2
    exit 2
fi
if [[ ! -d "$checkpoint_directory" || -L "$checkpoint_directory" ]]; then
    echo "ERROR: methods AITA checkpoint registry is unavailable" >&2
    exit 2
fi
if [[ ! "$cache_parent" = /* || ! -d "$cache_parent" ]]; then
    echo "ERROR: worker cache parent must be an existing absolute directory" >&2
    exit 2
fi
if [[ "${CTM_DISABLE_CUDNN_SDP:-0}" != "0" ]]; then
    echo "ERROR: every methods AITA worker requires CTM_DISABLE_CUDNN_SDP=0" >&2
    exit 2
fi
export CTM_DISABLE_CUDNN_SDP=0
if [[ "${CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP:-}" != "1" ]]; then
    echo "ERROR: every methods AITA worker requires CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1" >&2
    exit 2
fi
export CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1

rank=${SLURM_PROCID:-}
if [[ ! "$rank" =~ ^[0-9]+$ ]]; then
    echo "ERROR: worker requires a numeric Slurm rank" >&2
    exit 2
fi

condition=''
shard_index=''
case "$phase:$rank" in
    phase-1:0) condition=bct; shard_index=0 ;; phase-1:1) condition=bct; shard_index=1 ;;
    phase-1:2) condition=bct; shard_index=2 ;; phase-1:3) condition=bct; shard_index=3 ;;
    phase-1:4) condition=act; shard_index=0 ;; phase-1:5) condition=act; shard_index=1 ;;
    phase-1:6) condition=act; shard_index=2 ;; phase-1:7) condition=act; shard_index=3 ;;
    phase-1:8) condition=attct; shard_index=0 ;; phase-1:9) condition=attct; shard_index=1 ;;
    phase-1:10) condition=attct; shard_index=2 ;; phase-1:11) condition=attct; shard_index=3 ;;
    phase-1:12) condition=mlpct; shard_index=0 ;; phase-1:13) condition=mlpct; shard_index=1 ;;
    phase-1:14) condition=mlpct; shard_index=2 ;; phase-1:15) condition=mlpct; shard_index=3 ;;
    phase-2:0) condition=opct; shard_index=0 ;; phase-2:1) condition=opct; shard_index=1 ;;
    phase-2:2) condition=opct; shard_index=2 ;; phase-2:3) condition=opct; shard_index=3 ;;
    phase-1:*|phase-2:*)
        echo "ERROR: rank $rank is outside the assigned $phase cell plan" >&2
        exit 2
        ;;
    *)
        echo "ERROR: phase must be phase-1 or phase-2" >&2
        exit 2
        ;;
esac

"$python_bin" - <<'PY'
import os

tokens = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(any(char.isspace() for char in item) for item in tokens):
    raise SystemExit("error: every methods AITA worker rank must receive exactly one local Slurm-visible GPU")
PY

# Cache state is rank-local and intentionally retained after errors.  Do not
# key it by CUDA token: each isolated node can correctly expose local GPU 0.
worker_tmp=$(mktemp -d "$cache_parent/ctm-methods-aita-ntaflip-r006-${phase}-${condition}-j${SLURM_JOB_ID:-manual}-r${rank}-XXXXXX")
mkdir -p "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
chmod 700 "$worker_tmp" "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
export TMPDIR="$worker_tmp/tmp"
export XDG_CACHE_HOME="$worker_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$worker_tmp/torchinductor"
export TRITON_CACHE_DIR="$worker_tmp/triton"
export CUDA_CACHE_PATH="$worker_tmp/cuda-cache"

exec "$python_bin" "$launcher" worker \
    --campaign-root "$campaign_root" \
    --official-source-dir "$official_source_dir" \
    --checkpoint-directory "$checkpoint_directory" \
    --condition "$condition" \
    --shard-index "$shard_index" \
    --python "$python_bin"
