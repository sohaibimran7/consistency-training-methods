#!/usr/bin/env bash
# Isolated 4-GPU GPT-OSS training-memory matrix.
# Each array task is a fresh process for one (sequence length, checkpointing)
# pair, preventing a prior output/failed graph from contaminating later peaks.
#SBATCH --job-name=ctm-mem-isolated
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=00:25:00
#SBATCH --array=0-5
#SBATCH --output=slurm-mem-isolated-%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:?submit with --export=ALL,REPO_DIR=/absolute/repository/path}"
cd "$REPO_DIR"
VENV_DIR="${MAIN_VENV_DIR:-$REPO_DIR/.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "missing Python environment at $VENV_DIR" >&2
    exit 1
fi

CASES=(
    "4096 0"
    "4096 1"
    "8192 0"
    "8192 1"
    "9728 0"
    "9728 1"
)
CASE_INDEX="${SLURM_ARRAY_TASK_ID:?this launcher must run as a Slurm array}"
if (( CASE_INDEX < 0 || CASE_INDEX >= ${#CASES[@]} )); then
    echo "invalid array index: $CASE_INDEX" >&2
    exit 2
fi
read -r SEQUENCE_LENGTH GRADIENT_CHECKPOINTING <<< "${CASES[$CASE_INDEX]}"

JOB_TMPDIR="/tmp/ctm-mem-${USER}-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export TMPDIR="$JOB_TMPDIR"
export XDG_CACHE_HOME="$JOB_TMPDIR/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$JOB_TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$JOB_TMPDIR/triton"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

export VLLM_MXFP4_USE_MARLIN=0
HARMONY_CACHE_DIR="$REPO_DIR/.cache/tiktoken-encodings"
if [[ ! -f "$HARMONY_CACHE_DIR/o200k_base.tiktoken" ]]; then
    echo "missing harmony vocab cache at $HARMONY_CACHE_DIR" >&2
    exit 1
fi
export TIKTOKEN_ENCODINGS_BASE="$HARMONY_CACHE_DIR"
export TIKTOKEN_RS_CACHE_DIR="$HARMONY_CACHE_DIR"

if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -n "${SCRATCHDIR:-}" ]]; then
        HF_HOME="$SCRATCHDIR/ctm/huggingface"
    else
        HF_HOME="$REPO_DIR/.hf_cache"
    fi
fi
export HF_HOME

echo "case: sequence_length=$SEQUENCE_LENGTH gradient_checkpointing=$GRADIENT_CHECKPOINTING"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

PROBE_ARGS=(
    --model "${PROBE_MODEL:-openai/gpt-oss-20b}"
    --batch-size 1
    --sequence-length "$SEQUENCE_LENGTH"
    --device-map auto
    --max-memory-per-gpu "${PROBE_MAX_MEM_PER_GPU:-45GiB}"
    --rank 8
    --alpha 16
    --dropout 0.0
    --seed 42
    --optimizer-step
)
if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
    PROBE_ARGS+=(--gradient-checkpointing)
fi

srun --ntasks=1 "$VENV_DIR/bin/python" -u infra/isambard/probe_training_memory.py "${PROBE_ARGS[@]}"
