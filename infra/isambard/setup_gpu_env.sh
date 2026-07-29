#!/usr/bin/env bash
# Finish Isambard-AI setup from inside a one-GPU Slurm allocation. The
# allocation may be interactive or the Figure 6 prefetch batch job.
#
# First obtain an interactive allocation, then run this script from the repo:
#   srun --nodes=1 --gpus-per-node=1 --time=01:00:00 --pty /bin/bash --login
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

# vLLM 0.26.0 publishes a standard PyPI cp38-abi3 AArch64 wheel.  Run this
# inside the GH200 allocation so uv's automatic PyTorch backend selects a
# compatible CUDA build of the vLLM-pinned PyTorch 2.11 release.
uv pip install --upgrade "vllm==0.26.0" \
    --torch-backend=auto \
    --constraint requirements.txt \
    --constraint infra/isambard/vllm-constraints.txt

uv pip check
python - <<'PY'
import torch
import vllm

assert torch.__version__.split("+", 1)[0].startswith("2.11."), torch.__version__
assert torch.cuda.is_available(), "CUDA is not visible inside the allocation"
assert torch.cuda.is_bf16_supported(), "the allocated GPU must support bfloat16"
assert vllm.__version__ == "0.26.0", vllm.__version__
print(
    "GPU serving smoke check:",
    f"torch={torch.__version__}",
    f"vllm={vllm.__version__}",
    f"device={torch.cuda.get_device_name(0)}",
    f"capability={torch.cuda.get_device_capability(0)}",
)
PY
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
