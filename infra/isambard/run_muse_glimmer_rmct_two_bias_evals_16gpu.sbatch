#!/usr/bin/env bash
# One resumable 4-node / 16-GPU Muse Glimmer two-bias evaluation campaign.

#SBATCH --job-name=muse-rmct-2bias
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-muse-rmct-two-bias-%j.out

set -euo pipefail
umask 077

if [[ "${SLURM_JOB_RESERVATION:-}" != interactive ]]; then
    echo "ERROR: submit Muse evaluation with --reservation=interactive" >&2
    exit 2
fi
if [[ "${MUSE_EVAL_GROUP:-}" != early && "${MUSE_EVAL_GROUP:-}" != late ]]; then
    echo "ERROR: MUSE_EVAL_GROUP must be early or late" >&2
    exit 2
fi
if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable Muse repository" >&2
    exit 2
fi

repo_root=$(cd -- "$REPO_DIR" && pwd -P)
cd "$repo_root"
launcher="$repo_root/infra/isambard/run_muse_glimmer_rmct_two_bias_evals_16gpu.py"
worker="$repo_root/infra/isambard/run_muse_glimmer_rmct_two_bias_evals_16gpu_worker.sh"
if [[ ! -f "$launcher" || -L "$launcher" || ! -x "$worker" || -L "$worker" ]]; then
    echo "ERROR: Muse evaluation launcher or worker is unavailable" >&2
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
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA
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
    raise SystemExit("error: Muse evaluator must preserve its pinned venv")
expected = {
    "inspect-ai": "0.3.260",
    "torch": "2.13.0+cu129",
    "transformers": "5.15.1",
    "peft": "0.19.1",
    "safetensors": "0.8.0",
}
installed = {name: importlib.metadata.version(name) for name in expected}
if installed != expected:
    raise SystemExit(f"error: Muse evaluator runtime differs: got={installed!r}, expected={expected!r}")
PY

training_repository="${CTM_MUSE_TRAINING_REPOSITORY:-$repo_root}"
source_manifest="${CTM_MUSE_EVAL_SOURCE_STAGE2_MANIFEST:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-two-bias-eval-20260820/repo/artifacts/stage2-ood-hle-2x2-20260802-r1-source/manifest.json}"
artifact_root="${CTM_MUSE_EVAL_STAGE2_ARTIFACT_ROOT:-${source_manifest%/manifest.json}}"
evaluation_parent="${CTM_MUSE_EVAL_PARENT_ROOT:-$repo_root/logs/muse-glimmer-rmct-evaluations}"
campaign_root="${CTM_MUSE_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/muse-glimmer-rmct-two-bias-${MUSE_EVAL_GROUP}-16gpu-v1}"

for required in "$source_manifest" "$launcher" "$worker"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required Muse evaluation input is missing or linked: $required" >&2
        exit 2
    fi
done
for required_dir in "$training_repository" "$artifact_root"; do
    if [[ ! -d "$required_dir" || -L "$required_dir" ]]; then
        echo "ERROR: required Muse evaluation directory is missing or linked: $required_dir" >&2
        exit 2
    fi
done
mkdir -p "$evaluation_parent"

coordinator_tmp=$(mktemp -d "/tmp/ctm-muse-two-bias-${MUSE_EVAL_GROUP}-coordinator-j${SLURM_JOB_ID:-manual}-XXXXXX")
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
    --training-repository "$training_repository"
    --source-stage2-manifest "$source_manifest"
    --stage2-artifact-root "$artifact_root"
)

"$python_bin" "$launcher" --group "$MUSE_EVAL_GROUP" prepare "${campaign_args[@]}" --yes

run_phase() {
    local phase=$1
    if ! srun --label --exclusive --exact --kill-on-bad-exit=0 \
        --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=16 \
        --gpus=16 --gpus-per-task=1 --gpu-bind=verbose,per_task:1 --distribution=block:block \
        "$worker" "$launcher" "$python_bin" "$MUSE_EVAL_GROUP" "$phase" "$campaign_root" \
        "$training_repository" "$source_manifest" "$artifact_root" /tmp; then
        echo "ERROR: Muse evaluation phase $phase has failed workers; preserving attempts" >&2
        return 1
    fi
    "$python_bin" "$launcher" --group "$MUSE_EVAL_GROUP" seal-phase \
        "${campaign_args[@]}" --phase "$phase"
}

run_phase 1
run_phase 2
run_phase 3

if [[ "$MUSE_EVAL_GROUP" == early ]]; then
    conditions=(base step016)
else
    conditions=(step064 final)
fi
for condition in "${conditions[@]}"; do
    "$python_bin" "$launcher" --group "$MUSE_EVAL_GROUP" seal-clean-gate \
        "${campaign_args[@]}" --condition "$condition"
    "$python_bin" "$launcher" --group "$MUSE_EVAL_GROUP" finalize-condition \
        "${campaign_args[@]}" --condition "$condition"
done

exec "$python_bin" "$launcher" --group "$MUSE_EVAL_GROUP" finalize "${campaign_args[@]}"
