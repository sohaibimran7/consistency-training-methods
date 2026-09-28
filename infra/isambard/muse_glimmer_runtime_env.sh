#!/usr/bin/env bash
# Export native-library paths owned by the exact Muse vLLM runtime.
# This file is sourced by every setup, preflight, training, and evaluation job.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "ERROR: source muse_glimmer_runtime_env.sh; do not execute it." >&2
    exit 2
fi

ctm_muse_env_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
ctm_muse_env_python="$ctm_muse_env_repo/.venv-muse/bin/python"
if [[ ! -x "$ctm_muse_env_python" ]]; then
    echo "ERROR: Muse runtime Python is absent: $ctm_muse_env_python" >&2
    return 2
fi

ctm_muse_env_site_packages="$($ctm_muse_env_python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
ctm_muse_env_cu13_lib="$ctm_muse_env_site_packages/nvidia/cu13/lib"
if [[ ! -f "$ctm_muse_env_cu13_lib/libcudart.so.13" ]]; then
    echo "ERROR: pinned vLLM CUDA 13 runtime is absent: $ctm_muse_env_cu13_lib/libcudart.so.13" >&2
    return 2
fi

export LD_LIBRARY_PATH="$ctm_muse_env_cu13_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CTM_MUSE_CU13_LIBRARY_DIR="$ctm_muse_env_cu13_lib"

# The pinned post-Muse-fix vLLM wheel is linked against CUDA 13.  Importing
# torch alone is not a sufficient compatibility probe: a CUDA 12.x driver can
# initialize torch 2.13 but then fail when vLLM requests a CUDA-13 UVA mapping.
# Fail every allocated GPU job before model loading unless the driver API
# itself reports CUDA 13 or newer.  Login-node submitters intentionally skip
# this check because they own no allocated device.
if [[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    ctm_muse_env_driver_api="$($ctm_muse_env_python - <<'PY'
import ctypes

driver = ctypes.CDLL("libcuda.so.1")
if driver.cuInit(0) != 0:
    raise SystemExit("CUDA driver initialization failed")
version = ctypes.c_int()
if driver.cuDriverGetVersion(ctypes.byref(version)) != 0:
    raise SystemExit("CUDA driver version query failed")
print(version.value)
PY
)" || return 2
    if [[ ! "$ctm_muse_env_driver_api" =~ ^[0-9]+$ ]] || (( ctm_muse_env_driver_api < 13000 )); then
        echo "ERROR: pinned Muse vLLM requires CUDA driver API >=13000; got ${ctm_muse_env_driver_api:-<invalid>}." >&2
        return 2
    fi
    export CTM_MUSE_CUDA_DRIVER_API_VERSION="$ctm_muse_env_driver_api"
    unset ctm_muse_env_driver_api
fi

unset ctm_muse_env_repo ctm_muse_env_python ctm_muse_env_site_packages ctm_muse_env_cu13_lib
