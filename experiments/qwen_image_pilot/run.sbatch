#!/usr/bin/env bash
#SBATCH --job-name=qwen-image-pilot
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=03:00:00
set -euo pipefail
root=${PILOT_ROOT:-/projects/a5v/sohaib.a5v/ctm-qwen-image-pilot-20260920}
py=/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo/.venv/bin/python
model=/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
adapter=/projects/a5v/sohaib.a5v/ctm-rmct-step352-eval-20260913/eval-r2/adapter
export PATH="${py%/python}:$PATH"
export PYTHONPATH="$root/vendor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/scratch/a5v/sohaib.a5v/ctm/cuda-12.9.1
export PATH="$CUDA_HOME/bin:$PATH"
unset SLURM_CPUS_PER_GPU
cd "$root"
"$py" "$root/verify_adapter.py" --root "$root" --adapter "$adapter"
"$py" "$root/pilot.py" prepare --root "$root"
"$py" "$root/pilot.py" audit --root "$root" --model "$model"
"$py" -m vllm.entrypoints.openai.api_server \
  --model "$model" --served-model-name base --host 127.0.0.1 --port 8123 \
  --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 --enable-lora --max-lora-rank 16 \
  --additional-config '{"gdn_prefill_backend":"triton"}' \
  --lora-modules "rmct=$adapter" --limit-mm-per-prompt '{"image":8,"video":0}' \
  --reasoning-parser qwen3 --generation-config vllm > "$root/server-${SLURM_JOB_ID}.log" 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT
for attempt in $(seq 1 180); do
  if curl -fsS http://127.0.0.1:8123/health >/dev/null; then break; fi
  kill -0 "$server" || { tail -100 "$root/server-${SLURM_JOB_ID}.log"; exit 1; }
  sleep 5
done
curl -fsS http://127.0.0.1:8123/v1/models > "$root/served-models.json"
"$py" "$root/pilot.py" evaluate --root "$root"
"$py" "$root/report.py" --root "$root"
