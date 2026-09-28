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

# vLLM 0.10.2 predates Qwen3.5 and has no Qwen3_5 model implementation. Use
# the released CUDA 12.9 Arm64 wheel from the first stable line that supports
# Qwen3.5. ``--torch-backend=auto`` must run where the GH200 is visible so uv
# resolves matching CUDA-enabled PyTorch wheels. The post-install import gate
# below is deliberately necessary but insufficient: the four-GPU transport
# preflight still proves the real language_model_only + translated-LoRA path.
VLLM_VERSION="${CTM_ISAMBARD_VLLM_VERSION:-0.21.0}"
VLLM_WHEEL="${CTM_ISAMBARD_VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_34_aarch64.whl}"
uv pip install --upgrade "$VLLM_WHEEL" \
    --torch-backend=auto \
    --constraint requirements.txt \
    --constraint infra/isambard/vllm-constraints.txt

ctm_uv_check_output=
if ! ctm_uv_check_output=$(uv pip check 2>&1); then
    # uv 0.8.16 misclassifies NVIDIA's cuSPARSELt Arm wheel from the official
    # PyTorch CUDA index even though the installed shared object is AArch64.
    # Accept exactly that one metadata false positive only after proving the
    # binary architecture; every other dependency conflict remains fatal.
    ctm_uv_check_unexpected=$(printf '%s\n' "$ctm_uv_check_output" | sed -E \
        -e '/^Checked [0-9]+ packages in /d' \
        -e '/^Found 1 incompatibilit(y|ies)$/d' \
        -e '/^The package `nvidia-cusparselt-cu12` was built for a different platform$/d' \
        -e '/^[[:space:]]*$/d')
    if [[ -n "$ctm_uv_check_unexpected" ]]; then
        printf '%s\n' "$ctm_uv_check_output" >&2
        exit 1
    fi
    ctm_cusparselt_so=$(find .venv/lib/python*/site-packages/nvidia/cusparselt \
        -type f -name 'libcusparseLt.so.0' -print -quit 2>/dev/null || true)
    if [[ -z "$ctm_cusparselt_so" ]]; then
        echo "ERROR: uv reported the cuSPARSELt platform mismatch but its shared object is missing." >&2
        exit 1
    fi
    ctm_cusparselt_file=$(file "$ctm_cusparselt_so")
    if [[ "$ctm_cusparselt_file" != *"ELF 64-bit LSB shared object, ARM aarch64"* ]]; then
        echo "ERROR: cuSPARSELt is not an AArch64 shared object: $ctm_cusparselt_file" >&2
        exit 1
    fi
    printf '%s\n' "$ctm_uv_check_output" >&2
    echo "Accepted the verified AArch64 cuSPARSELt metadata false positive." >&2
else
    printf '%s\n' "$ctm_uv_check_output"
fi
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'device:', torch.cuda.get_device_name(0))"
CTM_EXPECTED_VLLM_VERSION="$VLLM_VERSION" python - <<'PY'
import os

import transformers
import vllm
from vllm import LLM, SamplingParams  # noqa: F401
from vllm.inputs import TokensPrompt  # noqa: F401
from vllm.lora.request import LoRARequest  # noqa: F401
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: F401

expected = os.environ["CTM_EXPECTED_VLLM_VERSION"]
observed = vllm.__version__.split("+", 1)[0]
assert observed == expected, (observed, expected)
print("vllm", vllm.__version__, "transformers", transformers.__version__, "qwen3_5=available")
PY
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
