#!/usr/bin/env bash
#SBATCH --job-name=muse-preflight
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-muse-preflight-%j.out

set -euo pipefail

ctm_muse_repo="${REPO_DIR:-$PROJECTDIR/$USER/consistency-training-methods}"
cd "$ctm_muse_repo"
export HF_HOME="${HF_HOME:-$SCRATCHDIR/ctm/huggingface-muse}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
ctm_muse_job_tmp="$SCRATCHDIR/ctm/tmp/muse-preflight-$SLURM_JOB_ID"
mkdir -p "$ctm_muse_job_tmp"
export TMPDIR="$ctm_muse_job_tmp"
export TMP="$ctm_muse_job_tmp"
export TEMP="$ctm_muse_job_tmp"
export XDG_CACHE_HOME="$ctm_muse_job_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$ctm_muse_job_tmp/torchinductor"
export TRITON_CACHE_DIR="$ctm_muse_job_tmp/triton"
export CUDA_CACHE_PATH="$ctm_muse_job_tmp/cuda-cache"
source infra/isambard/muse_glimmer_cuda129_runtime_env.sh

ctm_muse_python="$CTM_MUSE_RUNTIME_PYTHON"
ctm_muse_revision="a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
ctm_muse_output="$ctm_muse_repo/artifacts/muse-glimmer-rmct-preflight-20260824"
ctm_muse_runtime_receipts="$ctm_muse_repo/artifacts/muse-glimmer-runtime-cu129-20260824"
ctm_muse_snapshot="$($ctm_muse_python - "$ctm_muse_runtime_receipts/model-snapshot.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = Path(sys.argv[1])
document = json.loads(receipt.read_text(encoding="utf-8"))
snapshot = Path(document["snapshot_path"]).resolve()
if snapshot.is_symlink() or not (snapshot / "config.json").is_file():
    raise SystemExit("attested Muse model snapshot is absent or linked")
print(snapshot)
PY
)"

if [[ ! -x "$ctm_muse_python" || "$ctm_muse_snapshot" != */"$ctm_muse_revision" || ! -f "$ctm_muse_snapshot/config.json" ]]; then
    echo "ERROR: pinned Muse runtime/snapshot is absent; run setup_muse_glimmer_runtime.sbatch first." >&2
    exit 2
fi
if [[ -e "$ctm_muse_output" || -L "$ctm_muse_output" ]]; then
    echo "ERROR: refusing to overwrite Muse preflight output: $ctm_muse_output" >&2
    exit 2
fi

exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    "$ctm_muse_python" infra/isambard/preflight_muse_glimmer_rmct.py \
    --model-snapshot "$ctm_muse_snapshot" \
    --output-dir "$ctm_muse_output" \
    --runtime-receipt-dir "$ctm_muse_runtime_receipts" \
    --worker-gpus 1,2,3 \
    --worker-gpu-memory-utilization 0.90 \
    --worker-max-model-len 131072 \
    --worker-max-num-seqs 128 \
    --worker-max-num-batched-tokens 8192 \
    --seed 42
