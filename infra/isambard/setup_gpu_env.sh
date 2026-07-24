#!/usr/bin/env bash
# Finish Isambard-AI setup from inside a one-GPU Slurm allocation.
#
# First obtain an interactive allocation, then run this script from the repo:
#   srun --nodes=1 --gpus=1 --time=00:30:00 --pty /bin/bash --login
#   bash infra/isambard/setup_gpu_env.sh
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "ERROR: setup_gpu_env.sh must run inside a Slurm GPU allocation." >&2
    exit 2
fi

cd "$(dirname "$0")/../.."

if [[ -n "${SCRATCHDIR:-}" ]]; then
    export HF_HOME="${HF_HOME:-$SCRATCHDIR/ctm/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache}"
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
fi
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

source .venv/bin/activate

# BriCS's documented AArch64 recipe. --torch-backend=auto must run where the
# GH200 is visible so uv selects the compatible CUDA-enabled PyTorch wheels.
uv pip install --upgrade "vllm==0.10.2" \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/0.10.2/vllm \
    --constraint requirements.txt \
    --constraint infra/isambard/vllm-constraints.txt

uv pip check
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'device:', torch.cuda.get_device_name(0))"
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
