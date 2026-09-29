#!/usr/bin/env bash
set -euo pipefail
mode=$1
index=0
case "$mode" in
    parity) ;;
    smoke) index=4 ;;
    1) index=$((SLURM_PROCID + 1)) ;;
    2) index=$((SLURM_PROCID + 17)); if (( index > 21 )); then exit 0; fi ;;
    *) exit 2 ;;
esac
ctm_eval_tmp=$(mktemp -d /tmp/ctm-rmct352-XXXXXX)
export TMPDIR="$ctm_eval_tmp/tmp" XDG_CACHE_HOME="$ctm_eval_tmp/xdg"
export XDG_DATA_HOME="$ctm_eval_tmp/data"
export TRITON_CACHE_DIR="$ctm_eval_tmp/triton" TORCHINDUCTOR_CACHE_DIR="$ctm_eval_tmp/inductor"
export CUDA_CACHE_PATH="$ctm_eval_tmp/cuda"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$REPO_DIR"
if [[ "$mode" == parity ]]; then
    exec "$CTM_RMCT_EVAL_PYTHON" -m experiments.rmct_two_bias_eval.step352 parity --root "$EVAL_ROOT"
fi
if [[ "$mode" == smoke ]]; then
    exec "$CTM_RMCT_EVAL_PYTHON" -m experiments.rmct_two_bias_eval.step352 worker --root "$EVAL_ROOT" --task-index "$index" --smoke
fi
exec "$CTM_RMCT_EVAL_PYTHON" -m experiments.rmct_two_bias_eval.step352 worker --root "$EVAL_ROOT" --task-index "$index"
