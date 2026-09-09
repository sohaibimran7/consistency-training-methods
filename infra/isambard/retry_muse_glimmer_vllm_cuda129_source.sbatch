#!/usr/bin/env bash
#SBATCH --job-name=muse-vllm-cu129-r2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-muse-vllm-cu129-r2-%j.out

# Resume the exact CUDA-12.9 source build after the first attempt stopped at
# CMake compiler identification because Slurm's inherited TMPDIR was not
# writable. The installed toolkit, environment, and clean source checkout are
# replay-validated before setting a fresh scratch TMPDIR.
set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
ctm_muse_source="${MUSE_VLLM_SOURCE:-$SCRATCHDIR/ctm/vllm-source-probe-8c2bbe00}"
ctm_muse_venv="$ctm_muse_repo/.venv-muse-cu129"
ctm_muse_commit="8c2bbe00d58a930c6c09a80495728b26b79d9200"
ctm_muse_version="0.26.1rc1.dev1136+g8c2bbe00d"
ctm_cuda_home="$SCRATCHDIR/ctm/cuda-12.9.1"
ctm_cuda_installer="$SCRATCHDIR/ctm/cuda-installers/cuda_12.9.1_575.57.08_linux_sbsa.run"

cd "$ctm_muse_repo"
module load gcc-native/13.2
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse-cu129}"
export UV_LINK_MODE=copy

if [[ ! -x "$ctm_muse_venv/bin/python" ]]; then
    echo "ERROR: preserved CUDA-12.9 environment is absent." >&2
    exit 2
fi
if [[ ! -x "$ctm_cuda_home/bin/nvcc" ]] \
    || [[ "$($ctm_cuda_home/bin/nvcc --version)" != *"release 12.9, V12.9.86"* ]]; then
    echo "ERROR: preserved CUDA toolkit is not the pinned CUDA 12.9.1 compiler." >&2
    exit 2
fi
if [[ "$(stat -c %s "$ctm_cuda_installer")" != "5074335372" ]] \
    || [[ "$(md5sum "$ctm_cuda_installer" | awk '{print $1}')" != "824d4aea22be39358bb809b4f738488b" ]] \
    || [[ "$(sha256sum "$ctm_cuda_installer" | awk '{print $1}')" != "64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6" ]]; then
    echo "ERROR: preserved CUDA installer identity changed." >&2
    exit 2
fi
if [[ ! -d "$ctm_muse_source/.git" ]] \
    || [[ "$(git -C "$ctm_muse_source" rev-parse HEAD)" != "$ctm_muse_commit" ]] \
    || [[ -n "$(git -C "$ctm_muse_source" status --short)" ]]; then
    echo "ERROR: vLLM source is not the clean pinned commit." >&2
    exit 2
fi
ctm_muse_python="$ctm_muse_venv/bin/python"
if "$ctm_muse_python" -c 'import importlib.metadata; importlib.metadata.version("vllm")' 2>/dev/null; then
    echo "ERROR: preserved environment unexpectedly contains an installed vLLM." >&2
    exit 2
fi

ctm_build_tmp="$(mktemp -d "$SCRATCHDIR/ctm/vllm-cu129-build-tmp.XXXXXX")"
export TMPDIR="$ctm_build_tmp"
export TMP="$ctm_build_tmp"
export TEMP="$ctm_build_tmp"
export CUDA_HOME="$ctm_cuda_home"
export CUDAToolkit_ROOT="$ctm_cuda_home"
export CMAKE_CUDA_COMPILER="$ctm_cuda_home/bin/nvcc"
export PATH="$ctm_cuda_home/bin:$PATH"
export LD_LIBRARY_PATH="$ctm_cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_CACHE_PATH="$ctm_build_tmp/cuda-cache"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE=cuda
export VLLM_VERSION_OVERRIDE="$ctm_muse_version"

"$CUDA_HOME/bin/nvcc" --version
uv pip install --python "$ctm_muse_python" --no-build-isolation --no-deps -e "$ctm_muse_source"

CTM_MUSE_EXPECTED_VERSION="$ctm_muse_version" \
CTM_MUSE_EXPECTED_SOURCE="$ctm_muse_source" \
"$ctm_muse_python" - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import torch
import vllm
from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM

source = Path(os.environ["CTM_MUSE_EXPECTED_SOURCE"]).resolve()
assert importlib.metadata.version("vllm") == os.environ["CTM_MUSE_EXPECTED_VERSION"]
assert Path(vllm.__file__).resolve().is_relative_to(source)
mapping = MuseGlimmerForCausalLM.get_mm_mapping(object.__new__(MuseGlimmerForCausalLM))
assert mapping.language_model == ["model"], mapping
extensions = sorted(Path(vllm.__file__).resolve().parent.glob("_C*.so"))
assert extensions
assert torch.version.cuda == "12.9" and torch.cuda.is_available()
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
ctm_muse_ldd="$(ldd "$ctm_muse_extension")"
printf '%s\n' "$ctm_muse_ldd"
if [[ "$ctm_muse_ldd" == *"libcudart.so.13"* ]] \
    || [[ "$ctm_muse_ldd" == *"not found"* ]] \
    || [[ "$ctm_muse_ldd" != *"libcudart.so.12"* ]]; then
    echo "ERROR: built extension did not resolve exclusively against CUDA 12." >&2
    exit 2
fi

printf 'CTM_MUSE_CUDA_HOME=%s\n' "$ctm_cuda_home"
printf 'CTM_MUSE_BUILD_TMPDIR=%s\n' "$ctm_build_tmp"
printf 'CTM_MUSE_VLLM_COMMIT=%s\n' "$ctm_muse_commit"
printf 'CTM_MUSE_VLLM_VERSION=%s\n' "$ctm_muse_version"
printf 'CTM_MUSE_VLLM_SOURCE=%s\n' "$ctm_muse_source"
printf 'CTM_MUSE_VLLM_ENVIRONMENT=%s\n' "$ctm_muse_venv"
printf 'CTM_MUSE_VLLM_EXTENSION_SHA256=%s\n' "$(sha256sum "$ctm_muse_extension" | awk '{print $1}')"
