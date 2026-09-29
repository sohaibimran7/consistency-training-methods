#!/usr/bin/env bash
# Export and validate the user-local CUDA-12.9 source-build Muse runtime.
# Source this file from every Phase-2 preflight, training, and evaluation job.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source muse_glimmer_cuda129_runtime_env.sh; do not execute it." >&2
    exit 2
fi

ctm_muse_env_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ctm_muse_env_python="${CTM_MUSE_PYTHON:-$ctm_muse_env_repo/.venv-muse-cu129/bin/python}"
ctm_muse_env_cuda_home="${CTM_MUSE_CUDA_HOME:-${SCRATCHDIR:?SCRATCHDIR is required}/ctm/cuda-12.9.1}"
ctm_muse_env_source="${CTM_MUSE_VLLM_SOURCE:-${SCRATCHDIR}/ctm/vllm-source-probe-8c2bbe00}"
ctm_muse_env_commit="8c2bbe00d58a930c6c09a80495728b26b79d9200"
ctm_muse_env_version="0.26.1rc1.dev1136+g8c2bbe00d"

if [[ ! -x "$ctm_muse_env_python" ]]; then
    echo "ERROR: Muse CUDA-12.9 runtime Python is absent: $ctm_muse_env_python" >&2
    return 2
fi
ctm_muse_env_torch_lib="$($ctm_muse_env_python - <<'PY'
import sysconfig
from pathlib import Path

print(Path(sysconfig.get_paths()["purelib"]) / "torch/lib")
PY
)"
if [[ ! -f "$ctm_muse_env_torch_lib/libtorch.so" ]] \
    || [[ ! -f "$ctm_muse_env_torch_lib/libtorch_cpu.so" ]] \
    || [[ ! -f "$ctm_muse_env_torch_lib/libtorch_cuda.so" ]]; then
    echo "ERROR: pinned Muse runtime torch libraries are absent: $ctm_muse_env_torch_lib" >&2
    return 2
fi
if [[ ! -x "$ctm_muse_env_cuda_home/bin/nvcc" ]] \
    || [[ "$($ctm_muse_env_cuda_home/bin/nvcc --version)" != *"release 12.9, V12.9.86"* ]]; then
    echo "ERROR: pinned user-local CUDA 12.9.1 toolkit is absent or changed." >&2
    return 2
fi
if [[ ! -d "$ctm_muse_env_source/.git" ]] \
    || [[ "$(git -C "$ctm_muse_env_source" rev-parse HEAD)" != "$ctm_muse_env_commit" ]] \
    || [[ -n "$(git -C "$ctm_muse_env_source" status --short)" ]]; then
    echo "ERROR: Muse vLLM source is not the clean pinned commit." >&2
    return 2
fi

export CUDA_HOME="$ctm_muse_env_cuda_home"
export CUDAToolkit_ROOT="$ctm_muse_env_cuda_home"
export PATH="$ctm_muse_env_cuda_home/bin:$PATH"
export LD_LIBRARY_PATH="$ctm_muse_env_cuda_home/lib64:$ctm_muse_env_torch_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CTM_MUSE_RUNTIME_PYTHON="$ctm_muse_env_python"
export CTM_MUSE_CUDA_HOME="$ctm_muse_env_cuda_home"
export CTM_MUSE_VLLM_SOURCE="$ctm_muse_env_source"

if [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ctm_muse_env_probe="$($ctm_muse_env_python - "$ctm_muse_env_source" "$ctm_muse_env_version" <<'PY'
import ctypes
import importlib.metadata
import subprocess
import sys
from pathlib import Path

import torch
import vllm
from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM

source = Path(sys.argv[1]).resolve()
expected_version = sys.argv[2]
if importlib.metadata.version("vllm") != expected_version:
    raise SystemExit("pinned vLLM version changed")
if not Path(vllm.__file__).resolve().is_relative_to(source):
    raise SystemExit("vLLM is not imported from the pinned source checkout")
if torch.version.cuda != "12.9" or not torch.cuda.is_available():
    raise SystemExit(f"torch CUDA runtime is {torch.version.cuda!r} or CUDA is unavailable")
mapping = MuseGlimmerForCausalLM.get_mm_mapping(object.__new__(MuseGlimmerForCausalLM))
if mapping.language_model != ["model"]:
    raise SystemExit(f"Muse language-model mapping changed: {mapping}")
extensions = sorted(Path(vllm.__file__).resolve().parent.glob("_C*.so"))
if not extensions:
    raise SystemExit("vLLM CUDA extension is absent")
linked = subprocess.run(["ldd", str(extensions[0])], check=True, capture_output=True, text=True).stdout
if "libcudart.so.13" in linked or "not found" in linked or "libcudart.so.12" not in linked:
    raise SystemExit("vLLM extension is not resolved exclusively against CUDA 12")
driver = ctypes.CDLL("libcuda.so.1")
if driver.cuInit(0) != 0:
    raise SystemExit("CUDA driver initialization failed")
version = ctypes.c_int()
if driver.cuDriverGetVersion(ctypes.byref(version)) != 0 or version.value < 12000:
    raise SystemExit(f"CUDA driver API is incompatible: {version.value}")
print(version.value)
PY
)" || return 2
    if [[ ! "$ctm_muse_env_probe" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Muse CUDA-12.9 runtime probe returned an invalid driver API." >&2
        return 2
    fi
    export CTM_MUSE_CUDA_DRIVER_API_VERSION="$ctm_muse_env_probe"
    unset ctm_muse_env_probe
fi

unset ctm_muse_env_repo ctm_muse_env_python ctm_muse_env_cuda_home
unset ctm_muse_env_source ctm_muse_env_commit ctm_muse_env_version ctm_muse_env_torch_lib
