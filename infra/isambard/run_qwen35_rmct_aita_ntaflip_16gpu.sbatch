#!/usr/bin/env bash
# One isolated 4-node / 16-GPU ELEPHANT AITA-NTA-FLIP evaluation campaign.
#
# This wrapper has no scheduler-client calls.  Submit it with
# ``sbatch --reservation=interactive``: on Isambard, ``interactive`` is a
# reservation rather than a partition, so this file deliberately contains no
# partition directive.  Every ``srun`` below inherits that one allocation;
# none makes a separate scheduler request.  All four conditions and their
# four pair-preserving shards are launched in one 16-rank step.  A rank
# failure leaves its attempt evidence in place and stops the campaign rather
# than silently resampling that shard.

#SBATCH --job-name=ctm-rmct-aita-ntaflip-r006
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-rmct-aita-ntaflip-r006-%j.out

set -euo pipefail
umask 077

# ``interactive`` is a reservation, not a partition/QOS.  The allocation is
# intentionally selected by the operator's sbatch invocation, then inherited
# by the smoke and full srun steps below.  Fail before any preparation if an
# accidental normal-queue allocation reaches this r006 runner.
if [[ "${SLURM_JOB_RESERVATION:-}" != "interactive" ]]; then
    echo "ERROR: submit this r006 AITA rerun with --reservation=interactive; got SLURM_JOB_RESERVATION=${SLURM_JOB_RESERVATION:-<unset>}" >&2
    exit 2
fi

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.py"
worker_wrapper="$repo_root/infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu_worker.sh"
training_repository="${CTM_RMCT_TRAINING_REPOSITORY:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo}"
if [[ ! -d "$training_repository" || -L "$training_repository" ]]; then
    echo "ERROR: CTM_RMCT_TRAINING_REPOSITORY must name the durable sealed-training checkout" >&2
    exit 2
fi
training_repository="$(cd -- "$training_repository" && pwd -P)"

python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"
if [[ ! -x "$python_requested" || ! -f "$launcher" || -L "$launcher" || ! -x "$worker_wrapper" || -L "$worker_wrapper" ]]; then
    echo "ERROR: evaluation Python, AITA launcher, or worker wrapper is unavailable" >&2
    exit 2
fi
# Retain the venv spelling: resolving its Python symlink would silently select
# the base interpreter and drop the native-HF/Inspect packages.
python_directory="$(cd -L -- "$(dirname -- "$python_requested")" && pwd -L)"
python_bin="$python_directory/$(basename -- "$python_requested")"
if [[ ! -x "$python_bin" || ! -f "$python_bin" ]]; then
    echo "ERROR: resolved evaluation Python is unavailable or not a regular executable" >&2
    exit 2
fi
export PATH="$python_directory:$PATH"

# Do this before any package import or campaign preparation.  In particular,
# do not inherit a coordinator-node TORCHINDUCTOR_CACHE_DIR under another
# user's local directory.  This cache is deliberately retained after the job
# for failure evidence; every worker receives its own equivalent cache below.
coordinator_tmp=$(mktemp -d "/tmp/ctm-aita-ntaflip-r006-coordinator-j${SLURM_JOB_ID:-manual}-XXXXXX")
mkdir -p "$coordinator_tmp/tmp" "$coordinator_tmp/xdg-cache" "$coordinator_tmp/torchinductor" "$coordinator_tmp/triton" "$coordinator_tmp/cuda-cache"
chmod 700 "$coordinator_tmp" "$coordinator_tmp/tmp" "$coordinator_tmp/xdg-cache" "$coordinator_tmp/torchinductor" "$coordinator_tmp/triton" "$coordinator_tmp/cuda-cache"
export TMPDIR="$coordinator_tmp/tmp"
export XDG_CACHE_HOME="$coordinator_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$coordinator_tmp/torchinductor"
export TRITON_CACHE_DIR="$coordinator_tmp/triton"
export CUDA_CACHE_PATH="$coordinator_tmp/cuda-cache"

"$python_bin" - <<'PY'
import importlib.metadata
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: CTM_RMCT_EVAL_PYTHON must preserve the evaluation venv")

expected = {
    "inspect-ai": "0.3.258",
    "torch": "2.11.0+cu129",
    "transformers": "5.5.4",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
}
for distribution, required in expected.items():
    try:
        distribution_version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(f"error: AITA evaluation lacks {distribution!r}") from exc
    if distribution_version != required:
        raise SystemExit(
            f"error: AITA evaluation requires {distribution} {required}, "
            f"got distribution={distribution_version!r}"
        )
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# The r006 launch uses the frozen r005 EOS-only sampler before model
# construction.  Do not permit an absent or conflicting inherited marker to
# fall back to Inspect's native-HF max-token default.
if [[ -n "${CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP:-}" && "${CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP}" != "1" ]]; then
    echo "ERROR: AITA r006 requires the frozen CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1 sampler marker" >&2
    exit 2
fi
export CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1
if [[ "${CTM_DISABLE_CUDNN_SDP:-0}" != "0" ]]; then
    echo "ERROR: AITA evaluation requires CTM_DISABLE_CUDNN_SDP=0; refusing inherited nonzero attention policy" >&2
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

evaluation_parent="${CTM_AITA_EVAL_PARENT_ROOT:-$repo_root/logs/rmct-evaluations}"
campaign_root="${CTM_AITA_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/elephant-aita-nta-flip-qwen35-rmct-16gpu-v3-r006}"
official_source_dir="${CTM_AITA_NTAFLIP_SOURCE_DIR:-}"

if [[ -z "$official_source_dir" || ! -d "$official_source_dir" || -L "$official_source_dir" ]]; then
    echo "ERROR: CTM_AITA_NTAFLIP_SOURCE_DIR must name the checked official ELEPHANT AITA-NTA-FLIP source directory" >&2
    exit 2
fi
official_source_dir="$(cd -- "$official_source_dir" && pwd -P)"
if [[ -e "$campaign_root" && ( -L "$campaign_root" || ! -d "$campaign_root" ) ]]; then
    echo "ERROR: AITA campaign root is linked or not a directory: $campaign_root" >&2
    exit 2
fi
mkdir -p "$evaluation_parent"

# r005 showed that ``--gpus-per-task=1`` alone can still expose every local
# GPU to a singleton Slurm step.  On Isambard, the proven correction requires
# both an explicit step-wide GPU count and documented per-task binding.  This
# bounded 16-rank step has the same four-node topology as the full evaluation,
# but runs only UUID/visibility checks.  The coordinator seals its immutable
# receipt before source preparation, smoke checks, or model generation.  Do
# not set CUDA_VISIBLE_DEVICES manually: Slurm owns the mapping.
srun --label --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=16 \
    --gpus=16 --gpus-per-task=1 \
    --gpu-bind=verbose,per_task:1 --distribution=block:block \
    "$python_bin" "$launcher" probe-gpu-topology \
    --campaign-root "$campaign_root"
"$python_bin" "$launcher" seal-gpu-topology-probe \
    --campaign-root "$campaign_root"

# This intentionally replays raw training-checkpoint custody before allocating
# any GPU.  The former behavioral-evaluation output trees are not inputs.
"$python_bin" "$launcher" prepare \
    --campaign-root "$campaign_root" \
    --official-source-dir "$official_source_dir" \
    --training-repository "$training_repository" \
    --yes

# Verify direct output format from each model-loading route before consuming
# the full 16-GPU evaluation step.  ``--limit 2`` is used only inside the
# launcher as a two-source-sample smoke selection; every selected completion
# remains EOS-terminated with no generation/output token limit.
for smoke_condition in base step016 step064 step176; do
    srun --exclusive --exact --kill-on-bad-exit=1 \
        --nodes=1 --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=16 \
        --gpu-bind=verbose,per_task:1 --distribution=block:block \
        "$python_bin" "$launcher" smoke \
        --campaign-root "$campaign_root" \
        --official-source-dir "$official_source_dir" \
        --training-repository "$training_repository" \
        --condition "$smoke_condition" \
        --shard-index 0 \
        --python "$python_bin"
done

# Exactly 16 condition/shard cells: 4 conditions x 4 pair-preserving shards.
# --kill-on-bad-exit=0 preserves siblings' completed attempt evidence; the
# explicit finalizer reports missing cells but never automatically retries.
if ! srun --exclusive --exact --kill-on-bad-exit=0 \
    --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus=16 --gpus-per-task=1 --cpus-per-task=16 \
    --gpu-bind=verbose,per_task:1 --distribution=block:block \
    "$worker_wrapper" "$launcher" "$python_bin" "$campaign_root" "$official_source_dir" \
    "$training_repository" /tmp; then
    echo "ERROR: one or more AITA workers failed; preserving all attempts and checking seals" >&2
    "$python_bin" "$launcher" finalize \
        --campaign-root "$campaign_root" \
        --official-source-dir "$official_source_dir" \
        --training-repository "$training_repository" || true
    exit 1
fi

exec "$python_bin" "$launcher" finalize \
    --campaign-root "$campaign_root" \
    --official-source-dir "$official_source_dir" \
    --training-repository "$training_repository"
