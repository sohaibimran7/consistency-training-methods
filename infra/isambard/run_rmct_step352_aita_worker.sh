#!/usr/bin/env bash
set -euo pipefail
aita_work_tmp=$(mktemp -d /tmp/ctm-aita352-XXXXXX)
export TMPDIR="$aita_work_tmp/tmp" XDG_CACHE_HOME="$aita_work_tmp/cache" XDG_DATA_HOME="$aita_work_tmp/data"
export TRITON_CACHE_DIR="$aita_work_tmp/triton" TORCHINDUCTOR_CACHE_DIR="$aita_work_tmp/inductor" CUDA_CACHE_PATH="$aita_work_tmp/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$REPO_DIR"
exec "$CTM_RMCT_EVAL_PYTHON" -u -m experiments.elephant_aita_ntaflip.step352 "$1" --root "$AITA_ROOT" --rank "${SLURM_PROCID:-0}"
