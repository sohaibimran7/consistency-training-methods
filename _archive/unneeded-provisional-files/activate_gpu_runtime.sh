#!/usr/bin/env bash
# Expose the shared libraries bundled by the AArch64 vLLM and PyTorch wheels.
#
# vLLM 0.26.0's stable-libtorch AArch64 extension links against the CUDA 13
# runtime shipped in site-packages/nvidia/cu13, while PyTorch 2.11 ships its
# own CUDA 12.9 libraries.  Neither wheel directory is on Isambard's default
# loader path, so every Figure 6 process must source this after activating the
# virtual environment and before importing vLLM.

if [[ -z "${VIRTUAL_ENV:-}" || ! -x "$VIRTUAL_ENV/bin/python" ]]; then
    echo "ERROR: activate the project virtual environment before sourcing activate_gpu_runtime.sh." >&2
    return 2 2>/dev/null || exit 2
fi

GPU_SITE_PACKAGES=$(
    "$VIRTUAL_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
)
GPU_RUNTIME_DIRS=(
    "$GPU_SITE_PACKAGES/nvidia/cu13/lib"
    "$GPU_SITE_PACKAGES/torch/lib"
    "$GPU_SITE_PACKAGES/nvidia/cusparselt/lib"
)

for GPU_RUNTIME_DIR in "${GPU_RUNTIME_DIRS[@]}"; do
    if [[ ! -d "$GPU_RUNTIME_DIR" ]]; then
        echo "ERROR: required GPU runtime directory is missing: $GPU_RUNTIME_DIR" >&2
        return 2 2>/dev/null || exit 2
    fi
done

GPU_RUNTIME_PREFIX=$(IFS=:; printf '%s' "${GPU_RUNTIME_DIRS[*]}")
export LD_LIBRARY_PATH="$GPU_RUNTIME_PREFIX${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

unset GPU_RUNTIME_DIR GPU_RUNTIME_DIRS GPU_RUNTIME_PREFIX GPU_SITE_PACKAGES
