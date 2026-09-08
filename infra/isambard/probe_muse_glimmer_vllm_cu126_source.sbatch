#!/usr/bin/env bash
#SBATCH --job-name=muse-vllm-cu126
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-muse-vllm-cu126-%j.out

# Build the exact post-Muse-LoRA-fix vLLM source against Isambard Phase 2's
# CUDA 12.6 toolkit and the already pinned torch 2.13.0+cu129 installation.
# This is an isolated compatibility probe: it never mutates .venv-muse.
set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
ctm_muse_source="${MUSE_VLLM_SOURCE:-$SCRATCHDIR/ctm/vllm-source-probe-8c2bbe00}"
ctm_muse_venv="$ctm_muse_repo/.venv-muse-cu126"
ctm_muse_commit="8c2bbe00d58a930c6c09a80495728b26b79d9200"

cd "$ctm_muse_repo"
module load cudatoolkit/24.11_12.6
module load gcc-native/13.2
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDA_HOME="${CUDA_HOME:?CUDA toolkit module did not set CUDA_HOME}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE=cuda
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse-cu126}"
export UV_LINK_MODE=copy

if [[ ! -x .venv-muse/bin/python ]]; then
    echo "ERROR: pinned base Muse environment is absent." >&2
    exit 2
fi
if [[ -e "$ctm_muse_venv" || -L "$ctm_muse_venv" ]]; then
    echo "ERROR: refusing to replace existing compatibility-probe environment: $ctm_muse_venv" >&2
    exit 2
fi
if [[ ! -d "$ctm_muse_source/.git" ]]; then
    echo "ERROR: exact vLLM source checkout is absent: $ctm_muse_source" >&2
    exit 2
fi
if [[ "$(git -C "$ctm_muse_source" rev-parse HEAD)" != "$ctm_muse_commit" ]]; then
    echo "ERROR: vLLM source checkout is not the pinned commit." >&2
    exit 2
fi
if [[ -n "$(git -C "$ctm_muse_source" status --short)" ]]; then
    echo "ERROR: vLLM source checkout is not clean." >&2
    exit 2
fi

cp -a --reflink=auto .venv-muse "$ctm_muse_venv"
ctm_muse_python="$ctm_muse_venv/bin/python"
uv pip uninstall --python "$ctm_muse_python" vllm
"$ctm_muse_python" "$ctm_muse_source/use_existing_torch.py"
uv pip install --python "$ctm_muse_python" -r "$ctm_muse_source/requirements/build/cuda.txt"
uv pip install --python "$ctm_muse_python" --no-build-isolation --no-deps -e "$ctm_muse_source"

"$ctm_muse_python" - <<'PY'
import importlib.metadata
from pathlib import Path

import torch
import vllm
from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM

mapping = MuseGlimmerForCausalLM.get_mm_mapping(object.__new__(MuseGlimmerForCausalLM))
assert mapping.language_model == ["model"], mapping
root = Path(vllm.__file__).resolve().parent
extensions = sorted(root.glob("_C*.so"))
assert extensions, root
print("torch", torch.__version__, "torch_cuda", torch.version.cuda)
print("vllm", importlib.metadata.version("vllm"), "source", vllm.__file__)
print("extension", extensions[0])
print("device", torch.cuda.get_device_name(0))
PY

ctm_muse_extension="$($ctm_muse_python - <<'PY'
from pathlib import Path
import vllm
print(sorted(Path(vllm.__file__).resolve().parent.glob("_C*.so"))[0])
PY
)"
ldd "$ctm_muse_extension"
