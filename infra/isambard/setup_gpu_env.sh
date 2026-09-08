#!/usr/bin/env bash
# Finish Isambard-AI setup from inside a one-GPU Slurm allocation.
#
# First obtain an interactive allocation, then run this script from the repo:
#   srun --nodes=1 --gpus=1 --time=00:30:00 --pty /bin/bash --login
#   bash infra/isambard/setup_gpu_env.sh
set -euo pipefail

# Keep training and Figure 6 serving stacks explicit. Use separate clean
# checkouts/environments for these profiles; their vLLM/CUDA builds differ.
ctm_gpu_profile=training
if [[ $# -gt 0 ]]; then
    if [[ $# -ne 2 || "$1" != --profile || ( "$2" != training && "$2" != figure6 ) ]]; then
        echo "Usage: setup_gpu_env.sh [--profile training|figure6]" >&2
        exit 2
    fi
    ctm_gpu_profile=$2
fi


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

if [[ -f .venv/.ctm-gpu-profile && "$(cat .venv/.ctm-gpu-profile)" != "$ctm_gpu_profile" ]]; then
    echo "ERROR: this environment belongs to another GPU profile; use a fresh checkout/environment." >&2
    exit 2
fi

if [[ "$ctm_gpu_profile" == figure6 ]]; then
# vLLM 0.26.0 publishes a standard PyPI cp38-abi3 AArch64 wheel.  Run this
# inside the GH200 allocation so uv's automatic PyTorch backend selects a
# compatible CUDA build of the vLLM-pinned PyTorch 2.11 release.
uv pip install --upgrade "vllm==0.26.0" \
    --torch-backend=auto \
    --constraint requirements.txt \
    --constraint infra/isambard/vllm-figure6-constraints.txt

else
# Training uses the released Qwen3.5-capable CUDA 12.9 Arm64 wheel. ``--torch-backend=auto`` must run where the GH200 is visible so uv
# resolves matching CUDA-enabled PyTorch wheels. The post-install import gate
# below is deliberately necessary but insufficient: the four-GPU transport
# preflight still proves the real language_model_only + translated-LoRA path.
VLLM_VERSION="${CTM_ISAMBARD_VLLM_VERSION:-0.21.0}"
VLLM_WHEEL="${CTM_ISAMBARD_VLLM_WHEEL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_34_aarch64.whl}"
uv pip install --upgrade "$VLLM_WHEEL" \
    --torch-backend=auto \
    --constraint requirements.txt \
    --constraint infra/isambard/vllm-constraints.txt

fi

# The AArch64 vLLM wheel uses NVIDIA's `manylinux2014_sbsa` tag for
# cusparseLt. uv 0.9 currently reports that installed ARM library as a
# different-platform package, even though the ELF object is AArch64.  Keep the
# dependency check strict for every other incompatibility, and verify the
# known false positive against the installed binary before allowing it.
PIP_CHECK_OUTPUT=""
if ! PIP_CHECK_OUTPUT=$(uv pip check 2>&1); then
    printf '%s\n' "$PIP_CHECK_OUTPUT"
    export PIP_CHECK_OUTPUT
    python - <<'PY'
import os
import re

lines = [line.replace("Found 1 incompatibilities", "Found 1 incompatibility")
         for line in os.environ["PIP_CHECK_OUTPUT"].splitlines() if line]
allowed = {
    "Found 1 incompatibility",
    "The package `nvidia-cusparselt-cu12` was built for a different platform",
}
unexpected = [
    line
    for line in lines
    if line not in allowed and not re.fullmatch(r"Checked [0-9]+ packages in .+", line)
]
if unexpected:
    raise SystemExit(f"unexpected dependency incompatibilities: {unexpected}")
if not allowed.issubset(lines):
    raise SystemExit(f"unexpected uv pip check output: {lines}")
PY
    CUSPARSELT_LIBRARY=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cusparselt/lib/libcusparseLt.so.0")')
    if [[ ! -f "$CUSPARSELT_LIBRARY" ]] || ! file "$CUSPARSELT_LIBRARY" | grep -q 'ARM aarch64'; then
        echo "ERROR: the allowed cusparseLt metadata mismatch is not an AArch64 library." >&2
        exit 2
    fi
    echo "Accepted verified AArch64 cusparseLt sbsa-tag metadata mismatch."
fi
unset PIP_CHECK_OUTPUT

if [[ "$ctm_gpu_profile" == figure6 ]]; then
source infra/isambard/activate_gpu_runtime.sh
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
else
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
fi
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
printf '%s\n' "$ctm_gpu_profile" > .venv/.ctm-gpu-profile
