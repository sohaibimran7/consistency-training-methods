#!/usr/bin/env bash
#SBATCH --job-name=stage-cuda129-sbsa
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-stage-cuda129-sbsa-%j.out

# Stage the official NVIDIA CUDA 12.9.1 SBSA toolkit installer.  Installation
# is deliberately a second, SHA-256-pinned action after this receipt is read.
set -euo pipefail

ctm_cuda_name="cuda_12.9.1_575.57.08_linux_sbsa.run"
ctm_cuda_url="https://developer.download.nvidia.com/compute/cuda/12.9.1/local_installers/$ctm_cuda_name"
ctm_cuda_expected_size=5074335372
ctm_cuda_expected_md5="824d4aea22be39358bb809b4f738488b"
ctm_cuda_stage="$SCRATCHDIR/ctm/cuda-installers"
ctm_cuda_file="$ctm_cuda_stage/$ctm_cuda_name"
ctm_cuda_partial="$ctm_cuda_file.partial"

mkdir -p "$ctm_cuda_stage"
if [[ -e "$ctm_cuda_file" && ! -f "$ctm_cuda_file" ]]; then
    echo "ERROR: staged CUDA installer is not a regular file." >&2
    exit 2
fi
if [[ ! -f "$ctm_cuda_file" ]]; then
    curl --fail --location --continue-at - --output "$ctm_cuda_partial" "$ctm_cuda_url"
    if [[ "$(stat -c %s "$ctm_cuda_partial")" != "$ctm_cuda_expected_size" ]]; then
        echo "ERROR: downloaded CUDA installer has the wrong byte size." >&2
        exit 2
    fi
    if [[ "$(md5sum "$ctm_cuda_partial" | awk '{print $1}')" != "$ctm_cuda_expected_md5" ]]; then
        echo "ERROR: downloaded CUDA installer differs from NVIDIA's published MD5." >&2
        exit 2
    fi
    mv "$ctm_cuda_partial" "$ctm_cuda_file"
fi

if [[ "$(stat -c %s "$ctm_cuda_file")" != "$ctm_cuda_expected_size" ]] \
    || [[ "$(md5sum "$ctm_cuda_file" | awk '{print $1}')" != "$ctm_cuda_expected_md5" ]]; then
    echo "ERROR: staged CUDA installer failed immutable size/MD5 replay." >&2
    exit 2
fi
printf 'CTM_CUDA_129_INSTALLER=%s\n' "$ctm_cuda_file"
printf 'CTM_CUDA_129_SIZE=%s\n' "$ctm_cuda_expected_size"
printf 'CTM_CUDA_129_NVIDIA_MD5=%s\n' "$ctm_cuda_expected_md5"
printf 'CTM_CUDA_129_SHA256=%s\n' "$(sha256sum "$ctm_cuda_file" | awk '{print $1}')"
