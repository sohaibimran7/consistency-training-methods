#!/usr/bin/env bash
# One non-chaining 4-node / 16-GPU all-HF/PEFT matched RMCT two-bias campaign.
#
# The allocation evaluates the base snapshot and strict raw checkpoints at
# steps 16, 64, and 176 under one identical mixed 1,400-generation protocol.
# One twelve-cell clean wave runs across all conditions first.  The four
# per-condition clean-gate receipts are then sealed before five global biased
# waves are admitted.  The wrapper has no scheduler-client calls and does not
# retry a failed rank or start a successor job.

#SBATCH --job-name=ctm-rmct-all-hf-2bias-r002
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-rmct-all-hf-two-bias-r002-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu.py"
worker_wrapper="$repo_root/infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu_worker.sh"
training_repository="${CTM_RMCT_TRAINING_REPOSITORY:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo}"
if [[ ! -d "$training_repository" || -L "$training_repository" ]]; then
    echo "ERROR: CTM_RMCT_TRAINING_REPOSITORY must name the durable sealed-training checkout" >&2
    exit 2
fi
training_repository="$(cd -- "$training_repository" && pwd -P)"

python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"
if [[ ! -x "$python_requested" || ! -f "$launcher" || -L "$launcher" || ! -x "$worker_wrapper" || -L "$worker_wrapper" ]]; then
    echo "ERROR: evaluation Python, all-HF launcher, or worker wrapper is unavailable" >&2
    exit 2
fi
# Preserve the venv spelling; resolving bin/python would silently drop the
# selected Inspect/PEFT environment on installations where it is a symlink.
python_directory="$(cd -L -- "$(dirname -- "$python_requested")" && pwd -L)"
python_bin="$python_directory/$(basename -- "$python_requested")"
if [[ ! -x "$python_bin" || ! -f "$python_bin" ]]; then
    echo "ERROR: resolved evaluation Python is unavailable or not a regular executable" >&2
    exit 2
fi

# Replace hostile inherited compiler/cache paths before any Python process can
# import Torch, Transformers, PEFT, or Inspect.  This coordinator-node cache
# deliberately survives a failed allocation as forensic state; rank workers
# receive separate cache roots below.
coordinator_tmp=$(mktemp -d "/tmp/ctm-rmct-all-hf-r002-coordinator-j${SLURM_JOB_ID:-manual}-XXXXXX")
mkdir -p "$coordinator_tmp/tmp" "$coordinator_tmp/xdg-cache" "$coordinator_tmp/torchinductor" "$coordinator_tmp/triton" "$coordinator_tmp/cuda-cache"
chmod 700 "$coordinator_tmp" "$coordinator_tmp/tmp" "$coordinator_tmp/xdg-cache" "$coordinator_tmp/torchinductor" "$coordinator_tmp/triton" "$coordinator_tmp/cuda-cache"
export TMPDIR="$coordinator_tmp/tmp"
export XDG_CACHE_HOME="$coordinator_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$coordinator_tmp/torchinductor"
export TRITON_CACHE_DIR="$coordinator_tmp/triton"
export CUDA_CACHE_PATH="$coordinator_tmp/cuda-cache"
export PATH="$python_directory:$PATH"

"$python_bin" - <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: CTM_RMCT_EVAL_PYTHON must preserve the evaluation venv")

expected = {
    "inspect-ai": "0.3.258",
    "torch": "2.11.0+cu129",
    "transformers": "5.5.4",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
}
try:
    installed = {distribution: version(distribution) for distribution in expected}
except PackageNotFoundError as exc:
    raise SystemExit(f"error: matched all-HF campaign has a missing pinned evaluator package: {exc}") from exc
if installed != expected:
    raise SystemExit(
        f"error: matched all-HF campaign evaluator versions differ from the pinned remote runtime: "
        f"got={installed!r}, expected={expected!r}"
    )
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA
if [[ "${CTM_DISABLE_CUDNN_SDP:-0}" != "0" ]]; then
    echo "ERROR: matched all-HF campaign requires CTM_DISABLE_CUDNN_SDP=0" >&2
    exit 2
fi
export CTM_DISABLE_CUDNN_SDP=0

if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR must name durable shared scratch" >&2
    exit 2
fi
scratch_root="$(cd -- "$SCRATCHDIR" && pwd -P)"
export HF_HOME="${HF_HOME:-$scratch_root/ctm/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$scratch_root/ctm/uv-cache}"
export UV_LINK_MODE=copy
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
if [[ ! -w "$HF_HOME" || ! -w "$UV_CACHE_DIR" ]]; then
    echo "ERROR: durable Hugging Face or uv cache is not writable" >&2
    exit 2
fi

source_manifest="${CTM_RMCT_EVAL_SOURCE_STAGE2_MANIFEST:-$repo_root/artifacts/stage2-ood-hle-2x2-20260802-r1-source/manifest.json}"
artifact_root="${CTM_RMCT_EVAL_STAGE2_ARTIFACT_ROOT:-${source_manifest%/manifest.json}}"
evaluation_parent="${CTM_RMCT_EVAL_PARENT_ROOT:-$repo_root/logs/rmct-evaluations}"
campaign_root="${CTM_RMCT_ALL_HF_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/rmct-convergence-all-hf-peft-two-bias-16gpu-v1-r002}"

for required in "$source_manifest" "$launcher" "$worker_wrapper"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required all-HF input is missing or linked: $required" >&2
        exit 2
    fi
done
if [[ ! -d "$artifact_root" || -L "$artifact_root" ]]; then
    echo "ERROR: CTM_RMCT_EVAL_STAGE2_ARTIFACT_ROOT must be a regular staged artifact directory" >&2
    exit 2
fi
if [[ -e "$campaign_root" && ( -L "$campaign_root" || ! -d "$campaign_root" ) ]]; then
    echo "ERROR: all-HF campaign root is linked or not a directory: $campaign_root" >&2
    exit 2
fi
mkdir -p "$evaluation_parent"

campaign_args=(
    --campaign-root "$campaign_root"
    --training-repository "$training_repository"
    --source-stage2-manifest "$source_manifest"
    --stage2-artifact-root "$artifact_root"
)

# This validates the pinned base snapshot and every complete, resumable raw
# checkpoint before model generation.  It is idempotent only for byte-identical
# contracts and receipts; it never revives a foreign condition tree.
"$python_bin" "$launcher" prepare "${campaign_args[@]}" --yes

run_phase() {
    local phase=$1
    if srun --exclusive --exact --kill-on-bad-exit=0 \
        --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 \
        "$worker_wrapper" "$launcher" "$python_bin" "$phase" "$campaign_root" "$training_repository" \
        "$source_manifest" "$artifact_root" /tmp; then
        return 0
    fi
    # Do not promote, seal, retry, or run any successor after a worker failure.
    # Its attempt directory is durable forensic evidence for an explicit later
    # recovery decision.
    echo "ERROR: phase=$phase has failed ranks; preserving all evidence and stopping" >&2
    return 1
}

# The first wave owns the three clean cells for every one of the four
# conditions. Ranks 12..15 exit without generation: duplicate clean output is
# not an acceptable way to claim full utilization.
run_phase clean

# Every condition must have a durable receipt-attested clean reference before
# the first biased worker starts.  This makes the switch scorer's frozen
# 3,600-second fallback wait path unreachable.
for condition in base step016 step064 step176; do
    "$python_bin" "$launcher" seal-clean-gate "${campaign_args[@]}" --condition "$condition"
done

# All four conditions use the same native-HF runtime and proceed in balanced
# global waves.  The fifth wave has eight assigned cells; the other ranks exit
# cleanly rather than generating duplicates.
for phase in biased-1 biased-2 biased-3 biased-4 biased-5; do
    run_phase "$phase"
done

for condition in base step016 step064 step176; do
    "$python_bin" "$launcher" finalize-condition "${campaign_args[@]}" --condition "$condition"
done

exec "$python_bin" "$launcher" finalize "${campaign_args[@]}"
