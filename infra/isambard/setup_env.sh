#!/usr/bin/env bash
# One-time environment bootstrap for Isambard-AI (GH200, aarch64 Linux).
# Run from the repo root on a login node (or inside an interactive job).
#
# Notes:
# - Everything must be aarch64: torch, vllm, flash-attn etc. ship arm64 wheels,
#   but pin versions you have verified on the system if resolution misbehaves.
# - Put API keys in .env at the repo root (OPENAI_API_KEY for LLM judges, WANDB_API_KEY).
#   Tinker credentials are NOT needed for --backend local runs.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

# uv (static binary, works on aarch64)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.12 || uv venv   # fall back to system python if 3.12 unavailable

# Repo deps, following CLAUDE.md (grugstream excluded — fails to build, unused)
grep -v '^grugstream' requirements.txt | uv pip install -r /dev/stdin
uv pip install inspect-ai
uv pip install -e . --no-deps
uv pip install -e '.[local]'          # peft (LoRA for LocalBackend)
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# vLLM: aarch64 wheels exist but are version-sensitive on GH200 — install last and
# verify. If the generic wheel fails, use the NGC pytorch container or a source build.
uv pip install vllm || echo "WARNING: vllm install failed — HF sampler (--local-sampler hf) still works; fix vllm before production runs."

echo
echo "Sanity check:"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
uv run python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
