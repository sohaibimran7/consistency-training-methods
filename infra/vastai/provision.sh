#!/usr/bin/env bash
# Provision a Vast.ai instance for LocalBackend training. Idempotent.
#
# Works on either:
#   - the baked image (infra/vastai/Dockerfile): deps preinstalled, this just clones;
#   - a generic pytorch/vllm template: installs deps too (slower first boot).
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

# Deps (no-ops on the baked image; installs on a generic template)
if ! python -c "import vllm" 2>/dev/null; then
    pip install --no-cache-dir vllm
fi
grep -v '^grugstream' requirements.txt | pip install --no-cache-dir -r /dev/stdin
pip install --no-cache-dir inspect-ai peft
pip install --no-cache-dir -e . --no-deps
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

if [ ! -f .env ]; then
    echo "NOTE: create $REPO_DIR/.env with OPENAI_API_KEY (judges) + WANDB_API_KEY before training."
fi

# LFS-budget workaround: training jsonl may be pointer files on a fresh clone.
if head -c 40 dataset_dumps/test/*/*.jsonl 2>/dev/null | grep -q "git-lfs"; then
    echo "WARNING: dataset_dumps contains LFS pointer files — rsync real data from a working machine:"
    echo "  rsync -av <local>:consistency-training-methods/dataset_dumps/ $REPO_DIR/dataset_dumps/"
fi

echo
echo "Provisioned. Sanity check:"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import ctm.backends.local.engine as e; print('LocalBackend importable, peft:', e.HAS_PEFT)"
echo
echo "Train (inside tmux!):"
echo "  python scripts/tinker_training/train_rl.py --backend local ... "
