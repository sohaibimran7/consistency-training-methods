#!/usr/bin/env bash
# Provision a Vast.ai instance for LocalBackend training. Idempotent.
#
# Works on either:
#   - the image defined by infra/vastai/Dockerfile, with dependencies installed;
#   - a generic PyTorch/vLLM template, which installs dependencies during setup.
#
# Usage (inside the instance):
#   REPO_URL=git@github.com:<you>/consistency-training-methods.git bash provision.sh
#
# Qwen3.5 requires development versions newer than the default image. Opt in
# explicitly; ordinary installs retain the stable/image-provided stack:
#   QWEN35_COMPAT=1 REPO_URL=... bash provision.sh
#
# The Transformers Qwen3.5 implementation has a very slow Torch fallback for
# its DeltaNet layers. Installing its optional CUDA kernels is a separate,
# explicit choice because they must compile/import against the resolved Torch:
#   QWEN35_COMPAT=1 QWEN35_INSTALL_FAST_KERNELS=1 REPO_URL=... bash provision.sh
set -euo pipefail

REPO_URL="${REPO_URL:?Set REPO_URL to the git remote (use a deploy key or https token)}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$WORKDIR/consistency-training-methods"
QWEN35_COMPAT="${QWEN35_COMPAT:-0}"
QWEN35_INSTALL_FAST_KERNELS="${QWEN35_INSTALL_FAST_KERNELS:-0}"

case "$QWEN35_COMPAT" in
    0|1) ;;
    *) echo "ERROR: QWEN35_COMPAT must be 0 or 1." >&2; exit 2 ;;
esac
case "$QWEN35_INSTALL_FAST_KERNELS" in
    0|1) ;;
    *) echo "ERROR: QWEN35_INSTALL_FAST_KERNELS must be 0 or 1." >&2; exit 2 ;;
esac
if [[ "$QWEN35_INSTALL_FAST_KERNELS" == "1" && "$QWEN35_COMPAT" != "1" ]]; then
    echo "ERROR: QWEN35_INSTALL_FAST_KERNELS=1 requires QWEN35_COMPAT=1." >&2
    exit 2
fi

cd "$WORKDIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only

# Install dependencies that are not already present in the selected image.
if [[ "$QWEN35_COMPAT" != "1" ]] && ! python -c "import vllm" 2>/dev/null; then
    pip install --no-cache-dir vllm
fi
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -e . --no-deps

if [[ "$QWEN35_COMPAT" == "1" ]]; then
    # Qwen3.5 currently needs Transformers main and vLLM main/nightly. uv's
    # torch-backend=auto selects a Torch wheel compatible with the visible
    # NVIDIA driver instead of retaining an arbitrary image/base Torch build.
    python -m pip install --no-cache-dir --upgrade uv
    uv pip install --system --upgrade vllm \
        --torch-backend=auto \
        --extra-index-url https://wheels.vllm.ai/nightly

    if [[ "$QWEN35_INSTALL_FAST_KERNELS" == "1" ]]; then
        # Install only after vLLM resolves Torch. Import/version checks in
        # preflight.sh catch a missing or ABI-incompatible CUDA extension.
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                build-essential git ninja-build
        fi
        uv pip install --system --upgrade ninja packaging wheel
        for build_tool in git cc c++ ninja nvcc; do
            if ! command -v "$build_tool" >/dev/null 2>&1; then
                echo "ERROR: Qwen3.5 fast-kernel installation requires '$build_tool'." >&2
                echo "       Use infra/vastai/Dockerfile or a CUDA devel image with build tools." >&2
                exit 2
            fi
        done
        CTM_CUDA_HOME="$(python -c 'from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or "")')"
        if [[ -z "$CTM_CUDA_HOME" || ! -f "$CTM_CUDA_HOME/include/cuda.h" ]]; then
            echo "ERROR: CUDA development headers were not found through torch.utils.cpp_extension.CUDA_HOME." >&2
            echo "       A runtime-only CUDA image cannot build a missing fast-kernel wheel." >&2
            exit 2
        fi
        echo "Qwen3.5 kernel build prerequisites: CUDA_HOME=$CTM_CUDA_HOME"
        nvcc --version
        uv pip install --system --upgrade flash-linear-attention
        uv pip install --system --upgrade --no-build-isolation causal-conv1d
    else
        echo "NOTE: Qwen3.5 fast kernels were not installed. Set QWEN35_INSTALL_FAST_KERNELS=1"
        echo "      and rerun provisioning before a Qwen3.5 smoke; preflight.sh rejects the slow fallback."
    fi

    # Install Transformers main last so an optional kernel dependency cannot
    # replace it with a stable release lacking the Qwen3.5 compatibility map.
    uv pip install --system --upgrade \
        "transformers @ git+https://github.com/huggingface/transformers.git@main"
    python -m pip check
fi

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

if [ ! -f .env ]; then
    echo "NOTE: create $REPO_DIR/.env with grader-provider credentials before training. Add WANDB_API_KEY only when using --wandb-project."
fi

echo
echo "Provisioned. Sanity check:"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "from importlib.metadata import version; print('transformers', version('transformers'), 'vllm', version('vllm'), 'peft', version('peft'))"
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
if [[ "$QWEN35_COMPAT" == "1" ]]; then
    echo "Qwen3.5 compatibility stack selected. Run: bash infra/vastai/preflight.sh"
fi
echo
echo "Train inside a tmux session:"
echo "  python scripts/train_rlct.py --backend local ... "
