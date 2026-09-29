#!/usr/bin/env bash
# One condition/shard worker in the 16-GPU Muse AITA-NTA-FLIP campaign.

set -euo pipefail
umask 077

if (( $# != 6 )); then
    echo "usage: $0 <launcher.py> <python> <campaign-root> <official-source> <training-repository> <cache-parent>" >&2
    exit 2
fi

launcher=$1
python_bin=$2
campaign_root=$3
official_source_dir=$4
training_repository=$5
cache_parent=$6

if [[ ! -f "$launcher" || -L "$launcher" || ! -x "$python_bin" ]]; then
    echo "ERROR: Muse AITA launcher or Python is unavailable" >&2
    exit 2
fi
if [[ ! "$cache_parent" = /* || ! -d "$cache_parent" ]]; then
    echo "ERROR: Muse AITA worker cache parent must be an existing absolute directory" >&2
    exit 2
fi
if [[ "${CTM_DISABLE_CUDNN_SDP:-}" != 0 ]]; then
    echo "ERROR: Muse AITA worker requires CTM_DISABLE_CUDNN_SDP=0" >&2
    exit 2
fi
if [[ "${CTM_HF_EOS_ONLY_NO_TOKEN_CAP:-}" != 1 || \
      "${CTM_HF_EOS_ONLY_EXPECTED_INSPECT:-}" != 0.3.260 || \
      "${CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS:-}" != 5.15.1 ]]; then
    echo "ERROR: Muse AITA worker lacks the exact generic EOS-only/no-cap runtime markers" >&2
    exit 2
fi

rank=${SLURM_PROCID:-}
if [[ ! "$rank" =~ ^[0-9]+$ ]] || (( 10#$rank > 15 )); then
    echo "ERROR: Muse AITA worker requires Slurm rank [0,15]" >&2
    exit 2
fi

case "$rank" in
    0) condition=base; shard_index=0 ;; 1) condition=base; shard_index=1 ;;
    2) condition=base; shard_index=2 ;; 3) condition=base; shard_index=3 ;;
    4) condition=step016; shard_index=0 ;; 5) condition=step016; shard_index=1 ;;
    6) condition=step016; shard_index=2 ;; 7) condition=step016; shard_index=3 ;;
    8) condition=step064; shard_index=0 ;; 9) condition=step064; shard_index=1 ;;
    10) condition=step064; shard_index=2 ;; 11) condition=step064; shard_index=3 ;;
    12) condition=final; shard_index=0 ;; 13) condition=final; shard_index=1 ;;
    14) condition=final; shard_index=2 ;; 15) condition=final; shard_index=3 ;;
esac

"$python_bin" - <<'PY'
import os

tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
if len(tokens) != 1 or tokens[0] in {"-1", "NoDevFiles"} or any(character.isspace() for character in tokens[0]):
    raise SystemExit("error: every Muse AITA rank must receive exactly one Slurm-visible GPU")
PY

worker_tmp=$(mktemp -d "$cache_parent/ctm-muse-aita-j${SLURM_JOB_ID:-manual}-r${rank}-XXXXXX")
mkdir -p "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
chmod 700 "$worker_tmp" "$worker_tmp/tmp" "$worker_tmp/xdg-cache" "$worker_tmp/torchinductor" "$worker_tmp/triton" "$worker_tmp/cuda-cache"
export TMPDIR="$worker_tmp/tmp"
export TMP="$worker_tmp/tmp"
export TEMP="$worker_tmp/tmp"
export XDG_CACHE_HOME="$worker_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$worker_tmp/torchinductor"
export TRITON_CACHE_DIR="$worker_tmp/triton"
export CUDA_CACHE_PATH="$worker_tmp/cuda-cache"

exec "$python_bin" "$launcher" worker \
    --campaign-root "$campaign_root" \
    --official-source-dir "$official_source_dir" \
    --training-repository "$training_repository" \
    --condition "$condition" \
    --shard-index "$shard_index" \
    --python "$python_bin"
