#!/usr/bin/env bash
# Provision a Vast.ai instance for LocalBackend training. Idempotent.
#
# Works on either:
#   - the image defined by infra/vastai/Dockerfile, with dependencies installed;
#   - a generic PyTorch/vLLM template, which installs dependencies during setup.
#
# Usage (inside the instance):
#   REPO_URL=git@github.com:<you>/consistency-training-methods.git bash provision.sh
set -euo pipefail

REPO_URL="${REPO_URL:?Set REPO_URL to the git remote (use a deploy key or https token)}"
WORKDIR="${WORKDIR:-/workspace}"
REPO_DIR="$WORKDIR/consistency-training-methods"

cd "$WORKDIR"
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only || true

# Install dependencies that are not already present in the selected image.
if ! python -c "import vllm" 2>/dev/null; then
    pip install --no-cache-dir vllm
fi
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -e . --no-deps
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

if [ ! -f .env ]; then
    echo "NOTE: create $REPO_DIR/.env with grader-provider credentials before training. Add WANDB_API_KEY only when using --wandb-project."
fi

echo
echo "Provisioned. Sanity check:"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
echo
echo "Train inside a tmux session:"
echo "  python scripts/train_rlct.py --backend local ... "
