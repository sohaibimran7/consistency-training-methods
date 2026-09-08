#!/usr/bin/env bash
# One resumable 4-node / 16-GPU Muse Glimmer AITA-NTA-FLIP campaign.

#SBATCH --job-name=muse-rmct-aita
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-muse-rmct-aita-%j.out

set -euo pipefail
umask 077

if [[ "${SLURM_JOB_RESERVATION:-}" != interactive ]]; then
    echo "ERROR: submit Muse AITA with --reservation=interactive" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable Muse repository" >&2
    exit 2
fi

repo_root=$(cd -- "$REPO_DIR" && pwd -P)
cd "$repo_root"
launcher="$repo_root/infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu.py"
worker="$repo_root/infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu_worker.sh"
if [[ ! -f "$launcher" || -L "$launcher" || ! -x "$worker" || -L "$worker" ]]; then
    echo "ERROR: Muse AITA launcher or worker is unavailable" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-$PROJECTDIR/$USER/cache/muse-glimmer-rmct/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCHDIR/ctm/uv-cache-muse-cu129}"
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export CTM_DISABLE_CUDNN_SDP=0
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
export CTM_HF_EOS_ONLY_EXPECTED_INSPECT=0.3.260
export CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS=5.15.1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP
source infra/isambard/muse_glimmer_cuda129_runtime_env.sh
python_bin="$CTM_MUSE_RUNTIME_PYTHON"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: pinned Muse runtime Python is unavailable" >&2
    exit 2
fi
"$python_bin" - <<'PY'
import importlib.metadata
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: Muse AITA evaluator must preserve its pinned venv")
expected = {
    "inspect-ai": "0.3.260",
    "torch": "2.13.0+cu129",
    "transformers": "5.15.1",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
}
installed = {name: importlib.metadata.version(name) for name in expected}
if installed != expected:
    raise SystemExit(f"error: Muse AITA runtime differs: got={installed!r}, expected={expected!r}")
PY

training_repository="${CTM_MUSE_TRAINING_REPOSITORY:-$repo_root}"
official_source_dir="${CTM_AITA_NTAFLIP_SOURCE_DIR:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-two-bias-eval-20260820/repo/artifacts/elephant-aita-nta-flip-20260821-source}"
evaluation_parent="${CTM_MUSE_AITA_EVAL_PARENT_ROOT:-$repo_root/logs/muse-glimmer-rmct-evaluations}"
campaign_root="${CTM_MUSE_AITA_CAMPAIGN_ROOT:-$evaluation_parent/muse-glimmer-rmct-aita-nta-flip-16gpu-v1}"

for required_dir in "$training_repository" "$official_source_dir"; do
    if [[ ! -d "$required_dir" || -L "$required_dir" ]]; then
        echo "ERROR: required Muse AITA directory is missing or linked: $required_dir" >&2
        exit 2
    fi
done
mkdir -p "$evaluation_parent"

coordinator_tmp=$(mktemp -d "/tmp/ctm-muse-aita-coordinator-j${SLURM_JOB_ID:-manual}-XXXXXX")
mkdir -p "$coordinator_tmp/tmp" "$coordinator_tmp/xdg-cache" "$coordinator_tmp/torchinductor" "$coordinator_tmp/triton" "$coordinator_tmp/cuda-cache"
chmod 700 "$coordinator_tmp" "$coordinator_tmp/tmp" "$coordinator_tmp/xdg-cache" "$coordinator_tmp/torchinductor" "$coordinator_tmp/triton" "$coordinator_tmp/cuda-cache"
export TMPDIR="$coordinator_tmp/tmp"
export TMP="$coordinator_tmp/tmp"
export TEMP="$coordinator_tmp/tmp"
export XDG_CACHE_HOME="$coordinator_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$coordinator_tmp/torchinductor"
export TRITON_CACHE_DIR="$coordinator_tmp/triton"
export CUDA_CACHE_PATH="$coordinator_tmp/cuda-cache"

campaign_args=(
    --campaign-root "$campaign_root"
    --official-source-dir "$official_source_dir"
    --training-repository "$training_repository"
)

# Seal the exact 16-rank/16-GPU mapping before any model generation.
srun --label --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=16 \
    --gpus=16 --gpus-per-task=1 --gpu-bind=verbose,per_task:1 --distribution=block:block \
    "$python_bin" "$launcher" probe-gpu-topology --campaign-root "$campaign_root"
"$python_bin" "$launcher" seal-gpu-topology-probe --campaign-root "$campaign_root"

"$python_bin" "$launcher" prepare "${campaign_args[@]}" --yes

# Each smoke selects two source rows only; their generations remain uncapped
# and EOS-only.  The validator accepts reasoning only when the verdict is the
# exact output tail after one </think> boundary (or an exact direct verdict).
for smoke_condition in base step016 step064 final; do
    srun --exclusive --exact --kill-on-bad-exit=1 \
        --nodes=1 --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=16 \
        --gpu-bind=verbose,per_task:1 --distribution=block:block \
        "$python_bin" "$launcher" smoke "${campaign_args[@]}" \
        --condition "$smoke_condition" --shard-index 0 --python "$python_bin"
done

# Four conditions x four pair-preserving shards = sixteen one-GPU cells.
if ! srun --exclusive --exact --kill-on-bad-exit=0 \
    --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=16 \
    --gpus=16 --gpus-per-task=1 --gpu-bind=verbose,per_task:1 --distribution=block:block \
    "$worker" "$launcher" "$python_bin" "$campaign_root" "$official_source_dir" \
    "$training_repository" /tmp; then
    echo "ERROR: one or more Muse AITA workers failed; all attempts are preserved" >&2
    "$python_bin" "$launcher" finalize "${campaign_args[@]}" || true
    exit 1
fi

exec "$python_bin" "$launcher" finalize "${campaign_args[@]}"
