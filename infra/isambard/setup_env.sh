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

# uv (static binary, works on aarch64)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 || uv venv   # use the system Python if 3.12 is unavailable

# Complete repository environment (the top-level file includes both layers).
uv pip install -r requirements.txt
uv pip install -e . --no-deps
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# vLLM: aarch64 wheels exist but are version-sensitive on GH200 — install last and
# verify. If the generic wheel fails, use the NGC pytorch container or a source build.
uv pip install vllm || echo "WARNING: vllm install failed — HF sampler (--local-sampler hf) still works; fix vllm before production runs."

echo
echo "Sanity check:"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
uv run python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
