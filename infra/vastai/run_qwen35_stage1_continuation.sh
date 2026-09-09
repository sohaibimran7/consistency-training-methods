#!/usr/bin/env bash
set -euo pipefail

# One-off node allocation for the shared 8x H100 Vast host used on 2026-07-30.
# GPUs 1,2,3 are reserved by the concurrent Figure 6 run. This node owns only
# data preparation, the five one-GPU training methods, and the four-GPU OPCT
# target. The two eight-GPU RMCT targets and central publication run elsewhere.
CTM_DENSE_ROOT=/workspace/ctm-dense-qwen-20260730
CTM_DENSE_REPO="$CTM_DENSE_ROOT/repo"
CTM_DENSE_PYTHON="$CTM_DENSE_ROOT/env/bin/python"
CTM_GPU_IDS=0,4,5,6,7
CTM_OPCT_GPU_IDS=0,4,5,6
CTM_STAGE1_PLAN=experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b.yaml
CTM_LAUNCH_DIR="$CTM_DENSE_ROOT/stage1-continuation"
CTM_STATUS="$CTM_LAUNCH_DIR/status.tsv"
CTM_LOG="$CTM_LAUNCH_DIR/master.log"
CTM_CUDA_HOME="${CTM_CUDA_HOME:-/usr/local/cuda}"

# Qwen3.5's Gated-DeltaNet backward uses FLA/TileLang on Hopper. TileLang
# falls back to its pip-packaged nvcc when a sanitized launcher PATH hides the
# host compiler. That can pair a newer compiler with older wheel headers, so
# select and validate one coherent host toolkit before Python imports FLA.
if [[ ! -x "$CTM_CUDA_HOME/bin/nvcc" || ! -f "$CTM_CUDA_HOME/include/cuda.h" ]]; then
    echo "ERROR: expected a complete CUDA toolkit at CTM_CUDA_HOME=$CTM_CUDA_HOME" >&2
    exit 2
fi

mkdir -p "$CTM_LAUNCH_DIR"
mkdir -p "$CTM_DENSE_ROOT/cache/tilelang-cuda130"
mkdir "$CTM_DENSE_ROOT/stage1-continuation.lock"
exec > >(tee -a "$CTM_LOG") 2>&1

export CUDA_HOME="$CTM_CUDA_HOME"
export CUDA_PATH="$CTM_CUDA_HOME"
export PATH="$CTM_DENSE_ROOT/env/bin:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$CTM_DENSE_REPO"
export HF_HOME=/workspace/huggingface
export MCQ_BIAS_DATA_DIR="$CTM_DENSE_ROOT/mcq-bias-data"
export TMPDIR="$CTM_DENSE_ROOT/tmp"
export TILELANG_CACHE_DIR="$CTM_DENSE_ROOT/cache/tilelang-cuda130"
export CUDA_VISIBLE_DEVICES="$CTM_GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$CTM_DENSE_REPO"

record_status() {
    local state="$1"
    local label="$2"
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$state" "$label" | tee -a "$CTM_STATUS"
}

run_plan() {
    local label="$1"
    shift
    record_status START "$label"
    "$@"
    record_status PASS "$label"
}

on_exit() {
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
        record_status COMPLETE shared-five-gpu-node
    else
        record_status "FAILED:$rc" shared-five-gpu-node
    fi
}
trap on_exit EXIT

record_status START shared-five-gpu-node

run_plan shared-data-full \
    "$CTM_DENSE_PYTHON" scripts/run_experiment.py \
    experiments/rmct_paper_vast_dense_models/shared_data.yaml \
    --stages data_preparation --yes

run_plan qwen3.5-9b-stage1-data-preparation \
    "$CTM_DENSE_PYTHON" scripts/run_experiment.py \
    "$CTM_STAGE1_PLAN" --stages data_preparation --target data-preparation \
    --parallel 5 --gpus "$CTM_GPU_IDS" --yes

run_plan qwen3.5-9b-stage1-short-training \
    "$CTM_DENSE_PYTHON" scripts/run_experiment.py \
    "$CTM_STAGE1_PLAN" --stages training --target short-training \
    --parallel 5 --gpus "$CTM_GPU_IDS" --yes

run_plan qwen3.5-9b-stage1-opct \
    "$CTM_DENSE_PYTHON" scripts/run_experiment.py \
    "$CTM_STAGE1_PLAN" --stages training --target opct \
    --parallel 4 --gpus "$CTM_OPCT_GPU_IDS" --yes

record_status WAITING "rmct-main,rmct-control,central-publication,evaluation,analysis,rendering"
