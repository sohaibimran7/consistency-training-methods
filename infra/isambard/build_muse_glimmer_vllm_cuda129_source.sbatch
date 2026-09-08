#!/usr/bin/env bash
#SBATCH --job-name=muse-vllm-cu129
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-muse-vllm-cu129-%j.out

# Install the SHA-pinned NVIDIA CUDA 12.9.1 SBSA toolkit without a driver and
# build the unchanged post-Muse-LoRA-fix vLLM commit against it.  This uses a
# fresh environment and never mutates the CUDA-13 wheel runtime or prior CUDA
# 12.6 probes.
set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
ctm_muse_source="${MUSE_VLLM_SOURCE:-$SCRATCHDIR/ctm/vllm-source-probe-8c2bbe00}"
ctm_muse_venv="$ctm_muse_repo/.venv-muse-cu129"
ctm_muse_commit="8c2bbe00d58a930c6c09a80495728b26b79d9200"
ctm_muse_version="0.26.1rc1.dev1136+g8c2bbe00d"
ctm_cuda_name="cuda_12.9.1_575.57.08_linux_sbsa.run"
ctm_cuda_installer="$SCRATCHDIR/ctm/cuda-installers/$ctm_cuda_name"
ctm_cuda_expected_size="5074335372"
ctm_cuda_expected_md5="824d4aea22be39358bb809b4f738488b"
ctm_cuda_expected_sha256="64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6"
ctm_cuda_home="$SCRATCHDIR/ctm/cuda-12.9.1"

cd "$ctm_muse_repo"
module load gcc-native/13.2
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse-cu129}"
export UV_LINK_MODE=copy

if [[ ! -f "$ctm_cuda_installer" ]]; then
    echo "ERROR: staged CUDA installer is absent: $ctm_cuda_installer" >&2
    exit 2
fi
if [[ "$(stat -c %s "$ctm_cuda_installer")" != "$ctm_cuda_expected_size" ]] \
    || [[ "$(md5sum "$ctm_cuda_installer" | awk '{print $1}')" != "$ctm_cuda_expected_md5" ]] \
    || [[ "$(sha256sum "$ctm_cuda_installer" | awk '{print $1}')" != "$ctm_cuda_expected_sha256" ]]; then
    echo "ERROR: CUDA installer failed its pinned size/MD5/SHA-256 identity." >&2
    exit 2
fi
if [[ -e "$ctm_cuda_home" || -L "$ctm_cuda_home" ]]; then
    echo "ERROR: refusing to install over an existing CUDA target: $ctm_cuda_home" >&2
    exit 2
fi
ctm_cuda_tmp="$(mktemp -d "$SCRATCHDIR/ctm/cuda-12.9.1-install.XXXXXX")"
sh "$ctm_cuda_installer" \
    --silent \
    --toolkit \
    --toolkitpath="$ctm_cuda_home" \
    --tmpdir="$ctm_cuda_tmp" \
    --no-man-page

if [[ ! -x "$ctm_cuda_home/bin/nvcc" ]]; then
    echo "ERROR: CUDA 12.9.1 installer did not produce nvcc." >&2
    exit 2
fi
ctm_nvcc_version="$($ctm_cuda_home/bin/nvcc --version)"
if [[ "$ctm_nvcc_version" != *"release 12.9, V12.9.86"* ]]; then
    echo "ERROR: installed nvcc is not the pinned CUDA 12.9.1 compiler." >&2
    printf '%s\n' "$ctm_nvcc_version" >&2
    exit 2
fi

if [[ ! -x .venv-muse/bin/python ]]; then
    echo "ERROR: pinned base Muse environment is absent." >&2
    exit 2
fi
if [[ -e "$ctm_muse_venv" || -L "$ctm_muse_venv" ]]; then
    echo "ERROR: refusing to replace existing CUDA-12.9 build environment: $ctm_muse_venv" >&2
    exit 2
fi
if [[ ! -d "$ctm_muse_source/.git" ]] \
    || [[ "$(git -C "$ctm_muse_source" rev-parse HEAD)" != "$ctm_muse_commit" ]] \
    || [[ -n "$(git -C "$ctm_muse_source" status --short)" ]]; then
    echo "ERROR: vLLM source is not the clean pinned commit." >&2
    exit 2
fi

cp -a --reflink=auto .venv-muse "$ctm_muse_venv"
ctm_muse_python="$ctm_muse_venv/bin/python"
uv pip uninstall --python "$ctm_muse_python" vllm
"$ctm_muse_python" "$ctm_muse_source/use_existing_torch.py"
uv pip install --python "$ctm_muse_python" -r "$ctm_muse_source/requirements/build/cuda.txt"

export CUDA_HOME="$ctm_cuda_home"
export CUDAToolkit_ROOT="$ctm_cuda_home"
export CMAKE_CUDA_COMPILER="$ctm_cuda_home/bin/nvcc"
export PATH="$ctm_cuda_home/bin:$PATH"
export LD_LIBRARY_PATH="$ctm_cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS="${MAX_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
export VLLM_TARGET_DEVICE=cuda
export VLLM_VERSION_OVERRIDE="$ctm_muse_version"

"$CUDA_HOME/bin/nvcc" --version
uv pip install --python "$ctm_muse_python" --no-build-isolation --no-deps -e "$ctm_muse_source"

CTM_MUSE_EXPECTED_COMMIT="$ctm_muse_commit" \
CTM_MUSE_EXPECTED_VERSION="$ctm_muse_version" \
CTM_MUSE_EXPECTED_SOURCE="$ctm_muse_source" \
"$ctm_muse_python" - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import torch
import vllm
from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM

expected_source = Path(os.environ["CTM_MUSE_EXPECTED_SOURCE"]).resolve()
assert importlib.metadata.version("vllm") == os.environ["CTM_MUSE_EXPECTED_VERSION"]
assert Path(vllm.__file__).resolve().is_relative_to(expected_source)
mapping = MuseGlimmerForCausalLM.get_mm_mapping(object.__new__(MuseGlimmerForCausalLM))
assert mapping.language_model == ["model"], mapping
root = Path(vllm.__file__).resolve().parent
extensions = sorted(root.glob("_C*.so"))
assert extensions, root
assert torch.cuda.is_available()
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

printf 'CTM_MUSE_CUDA_INSTALLER_SHA256=%s\n' "$ctm_cuda_expected_sha256"
printf 'CTM_MUSE_CUDA_HOME=%s\n' "$ctm_cuda_home"
printf 'CTM_MUSE_VLLM_COMMIT=%s\n' "$ctm_muse_commit"
printf 'CTM_MUSE_VLLM_VERSION=%s\n' "$ctm_muse_version"
printf 'CTM_MUSE_VLLM_SOURCE=%s\n' "$ctm_muse_source"
printf 'CTM_MUSE_VLLM_ENVIRONMENT=%s\n' "$ctm_muse_venv"
printf 'CTM_MUSE_VLLM_EXTENSION_SHA256=%s\n' "$(sha256sum "$ctm_muse_extension" | awk '{print $1}')"
