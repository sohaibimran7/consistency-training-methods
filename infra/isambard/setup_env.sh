#!/usr/bin/env bash
# One-time environment bootstrap for Isambard-AI (GH200, aarch64 Linux).
# Run from the repo root on a login node (or inside an interactive job).
#
# Notes:
# - Every compiled dependency must support AArch64. Pin system-verified versions
#   if the standard dependency resolver cannot select compatible wheels.
# - Put grader-provider keys in .env at the repo root. Add WANDB_API_KEY only
#   when the training command explicitly enables W&B with --wandb-project.
#   Tinker credentials are NOT needed for --backend local runs.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repository root

# Keep downloads and package caches off the smaller home filesystem when the
# Isambard storage variables are available.
if [[ -n "${SCRATCHDIR:-}" ]]; then
    export HF_HOME="${HF_HOME:-$SCRATCHDIR/ctm/huggingface}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache}"
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
fi

# Login nodes are cgroup-limited. Keep wheel extraction conservative so large
# CUDA packages do not exhaust the session while the environment is created.
export UV_CONCURRENT_BUILDS="${UV_CONCURRENT_BUILDS:-1}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-1}"

# Isambard's supported vLLM recipe is tested with this uv release.
UV_VERSION="${UV_VERSION:-0.8.16}"
if ! command -v uv >/dev/null 2>&1 || [[ "$(uv --version)" != "uv $UV_VERSION" ]]; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 || uv venv   # use the system Python if 3.12 is unavailable
fi

# Complete repository environment (the top-level file includes both layers).
# Use CPU PyTorch on the login node; setup_gpu_env.sh replaces it with the
# compatible CUDA build after uv can see the allocated GH200.
uv pip install -r requirements.txt --torch-backend=cpu
uv pip install -e . --no-deps
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo
echo "Login-node sanity check:"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
uv run python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
echo
echo "Base environment installed. Install and validate the Isambard-supported vLLM stack"
echo "inside an approved GPU allocation with: bash infra/isambard/setup_gpu_env.sh"
