#!/usr/bin/env bash
#SBATCH --job-name=image-suite-final
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --array=0-15%16
set -euo pipefail
root=${IMAGE_SUITE_ROOT:-/projects/a5v/sohaib.a5v/ctm-image-suite-20260921}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_HOME=/scratch/a5v/sohaib.a5v/ctm/cuda-12.9.1
image_cache=$(mktemp -d /tmp/ctm-image-suite-XXXXXX)
export TMPDIR="$image_cache" XDG_CACHE_HOME="$image_cache/cache" XDG_DATA_HOME="$image_cache/data"
export TRITON_CACHE_DIR="$image_cache/triton" TORCHINDUCTOR_CACHE_DIR="$image_cache/inductor" CUDA_CACHE_PATH="$image_cache/cuda"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"
unset SLURM_CPUS_PER_GPU
py=/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo/.venv/bin/python
if (( SLURM_ARRAY_TASK_ID >= 14 )); then
    py=/projects/a5v/sohaib.a5v/consistency-training-methods/.venv-muse-cu129/bin/python
fi
export PATH="${py%/python}:$CUDA_HOME/bin:$PATH"
cd "$root"
exec "$py" -u full_worker.py --root "$root" --index "$SLURM_ARRAY_TASK_ID"
