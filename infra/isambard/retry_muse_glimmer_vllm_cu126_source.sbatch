#!/usr/bin/env bash
#SBATCH --job-name=muse-vllm-cu126-r2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-muse-vllm-cu126-r2-%j.out

# Continue the isolated Phase-2 source-build probe after creating the standard
# CUDA layout expected by PyTorch's CMake package.  The NVIDIA HPC SDK stores
# AArch64 headers/libraries under targets/sbsa-linux rather than include/lib64.
set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
ctm_muse_source="${MUSE_VLLM_SOURCE:-$SCRATCHDIR/ctm/vllm-source-probe-8c2bbe00}"
ctm_muse_venv="$ctm_muse_repo/.venv-muse-cu126"
ctm_muse_commit="8c2bbe00d58a930c6c09a80495728b26b79d9200"

cd "$ctm_muse_repo"
module load cudatoolkit/24.11_12.6
ctm_muse_vendor_cuda="${CUDA_HOME:?CUDA toolkit module did not set CUDA_HOME}"
module load gcc-native/13.2
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
ctm_muse_cuda_shim="$SCRATCHDIR/ctm/cuda-12.6-sbsa-shim"
mkdir -p "$ctm_muse_cuda_shim"

ensure_link() {
    local source=$1
    local destination=$2
    if [[ -L "$destination" ]]; then
        if [[ "$(readlink -f "$destination")" != "$(readlink -f "$source")" ]]; then
            echo "ERROR: existing CUDA shim link has a different target: $destination" >&2
            exit 2
        fi
    elif [[ -e "$destination" ]]; then
        echo "ERROR: CUDA shim path exists and is not a link: $destination" >&2
        exit 2
    else
        ln -s "$source" "$destination"
    fi
}

ensure_link "$ctm_muse_vendor_cuda/bin" "$ctm_muse_cuda_shim/bin"
ensure_link "$ctm_muse_vendor_cuda/targets/sbsa-linux/include" "$ctm_muse_cuda_shim/include"
ensure_link "$ctm_muse_vendor_cuda/targets/sbsa-linux/lib" "$ctm_muse_cuda_shim/lib64"
ensure_link "$ctm_muse_vendor_cuda/nvvm" "$ctm_muse_cuda_shim/nvvm"
ensure_link "$ctm_muse_vendor_cuda/targets" "$ctm_muse_cuda_shim/targets"

export CUDA_HOME="$ctm_muse_cuda_shim"
export CUDAToolkit_ROOT="$ctm_muse_cuda_shim"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE=cuda
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse-cu126}"
export UV_LINK_MODE=copy

ctm_muse_python="$ctm_muse_venv/bin/python"
if [[ ! -x "$ctm_muse_python" || ! -d "$ctm_muse_source/.git" ]]; then
    echo "ERROR: first source-build attempt did not leave its isolated inputs." >&2
    exit 2
fi
if [[ "$(git -C "$ctm_muse_source" rev-parse HEAD)" != "$ctm_muse_commit" ]] \
    || [[ -n "$(git -C "$ctm_muse_source" status --short)" ]]; then
    echo "ERROR: vLLM source checkout is not the clean pinned commit." >&2
    exit 2
fi
if "$ctm_muse_python" -c 'import importlib.metadata; importlib.metadata.version("vllm")' 2>/dev/null; then
    echo "ERROR: isolated retry environment unexpectedly already contains vLLM." >&2
    exit 2
fi

"$CUDA_HOME/bin/nvcc" --version
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
