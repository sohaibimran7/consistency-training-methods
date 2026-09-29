#!/bin/bash
#SBATCH --job-name=qwen-cap64k-564
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
set -euo pipefail
umask 077
root=/scratch/a5v/sohaib.a5v/ctm/cap64k-rerun-20260920
runtime=/scratch/a5v/sohaib.a5v/ctm/methods-three-terminal-eval-20260913-compat-r2
py=/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo/.venv/bin/python
export PATH="${py%/python}:$PATH"
export PYTHONPATH="$root:$runtime/vendor:$runtime/repo"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/a5v/sohaib.a5v/ctm/huggingface
exec 9>"$root/campaign.lock"
flock -n 9 || exit 1
srun --exclusive --exact --kill-on-bad-exit=0 --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 "$py" "$root/worker.py"
