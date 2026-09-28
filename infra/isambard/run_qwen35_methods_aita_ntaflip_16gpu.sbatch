#!/usr/bin/env bash
# One isolated 4-node / 16-GPU AITA-NTA-FLIP replication for five methods.
#
# Submit this file into the already-authorized interactive reservation.  It
# inherits that one allocation and only uses srun steps within it: a topology
# probe, five small output-format smokes, a 16-cell phase, and a four-cell
# phase.  It deliberately does not submit, cancel, retry, or chain jobs.

#SBATCH --job-name=ctm-methods-aita-ntaflip-r006
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-methods-aita-ntaflip-r006-%j.out

set -euo pipefail
umask 077

if [[ "${SLURM_JOB_RESERVATION:-}" != "interactive" ]]; then
    echo "ERROR: submit the methods AITA r006 replication with --reservation=interactive; got SLURM_JOB_RESERVATION=${SLURM_JOB_RESERVATION:-<unset>}" >&2
    exit 2
fi

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu.py"
worker_wrapper="$repo_root/infra/isambard/run_qwen35_methods_aita_ntaflip_16gpu_worker.sh"

# The registry must be a direct, regular directory hierarchy:
#   <root>/{bct,act,attct,mlpct,opct}/{adapter_model.safetensors,adapter_config.json,manifest.json,README.md}
#   <root>/provenance/{bct,act,attct,mlpct,opct}.json
# ``act`` is the canonical repaired-ACT endpoint.  The Python launcher checks
# every supplied SHA-256, final loop state, and Stage-2 lineage record.
checkpoint_directory="${CTM_AITA_METHODS_CHECKPOINT_DIRECTORY:-}"
if [[ -z "$checkpoint_directory" || ! -d "$checkpoint_directory" || -L "$checkpoint_directory" ]]; then
    echo "ERROR: CTM_AITA_METHODS_CHECKPOINT_DIRECTORY must name the regular five-adapter registry root" >&2
    exit 2
fi
checkpoint_directory="$(cd -- "$checkpoint_directory" && pwd -P)"

python_requested="${CTM_AITA_EVAL_PYTHON:-${CTM_RMCT_EVAL_PYTHON:-$repo_root/.venv/bin/python}}"
if [[ ! -x "$python_requested" || ! -f "$launcher" || -L "$launcher" || ! -x "$worker_wrapper" || -L "$worker_wrapper" ]]; then
    echo "ERROR: evaluation Python, methods AITA launcher, or worker wrapper is unavailable" >&2
    exit 2
fi
# Retain the venv spelling: resolving bin/python can select an ambient base
# interpreter and drop the pinned native-HF/PEFT evaluator packages.
python_directory="$(cd -L -- "$(dirname -- "$python_requested")" && pwd -L)"
python_bin="$python_directory/$(basename -- "$python_requested")"
if [[ ! -x "$python_bin" || ! -f "$python_bin" ]]; then
    echo "ERROR: resolved evaluation Python is unavailable or not a regular executable" >&2
    exit 2
fi
export PATH="$python_directory:$PATH"

# Replace inherited cache locations before importing any GPU package.  These
# directories are intentionally retained after a failure as forensic state.
coordinator_tmp=$(mktemp -d "/tmp/ctm-methods-aita-ntaflip-r006-coordinator-j${SLURM_JOB_ID:-manual}-XXXXXX")
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
    raise SystemExit("error: CTM_AITA_EVAL_PYTHON must preserve the evaluation venv")

expected = {
    "inspect-ai": "0.3.258",
    "torch": "2.11.0+cu129",
    "transformers": "5.5.4",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
}
for distribution, required in expected.items():
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(f"error: methods AITA evaluation lacks {distribution!r}") from exc
    if installed != required:
        raise SystemExit(
            f"error: methods AITA evaluation requires {distribution} {required}, got {installed!r}"
        )
PY

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
if [[ -n "${CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP:-}" && "${CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP}" != "1" ]]; then
    echo "ERROR: methods AITA r006 requires CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1" >&2
    exit 2
fi
export CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1
if [[ "${CTM_DISABLE_CUDNN_SDP:-0}" != "0" ]]; then
    echo "ERROR: methods AITA evaluation requires CTM_DISABLE_CUDNN_SDP=0" >&2
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
campaign_root="${CTM_AITA_METHODS_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/elephant-aita-nta-flip-qwen35-methods-16gpu-v1-r006}"
official_source_dir="${CTM_AITA_NTAFLIP_SOURCE_DIR:-}"
if [[ -z "$official_source_dir" || ! -d "$official_source_dir" || -L "$official_source_dir" ]]; then
    echo "ERROR: CTM_AITA_NTAFLIP_SOURCE_DIR must name the checked official ELEPHANT AITA-NTA-FLIP source directory" >&2
    exit 2
fi
official_source_dir="$(cd -- "$official_source_dir" && pwd -P)"
if [[ -e "$campaign_root" && ( -L "$campaign_root" || ! -d "$campaign_root" ) ]]; then
    echo "ERROR: methods AITA campaign root is linked or not a directory: $campaign_root" >&2
    exit 2
fi
mkdir -p "$evaluation_parent"

# Establish all sixteen physical UUID bindings before source preparation or
# any model construction.  Do not set CUDA_VISIBLE_DEVICES manually: Slurm
# owns the one-device mapping for each rank.
srun --label --exclusive --exact --kill-on-bad-exit=1 \
    --nodes=4 --ntasks=16 --ntasks-per-node=4 --cpus-per-task=16 \
    --gpus=16 --gpus-per-task=1 \
    --gpu-bind=verbose,per_task:1 --distribution=block:block \
    "$python_bin" "$launcher" probe-gpu-topology \
    --campaign-root "$campaign_root"
"$python_bin" "$launcher" seal-gpu-topology-probe \
    --campaign-root "$campaign_root"

campaign_args=(
    --campaign-root "$campaign_root"
    --official-source-dir "$official_source_dir"
    --checkpoint-directory "$checkpoint_directory"
)
"$python_bin" "$launcher" prepare "${campaign_args[@]}" --yes

# The smoke selection is a pair of source rows checked by the inherited r006
# pipeline; it is not a completion-length setting.  Full cells sample to EOS.
for smoke_condition in bct act attct mlpct opct; do
    srun --exclusive --exact --kill-on-bad-exit=1 \
        --nodes=1 --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=16 \
        --gpu-bind=verbose,per_task:1 --distribution=block:block \
        "$python_bin" "$launcher" smoke "${campaign_args[@]}" \
        --condition "$smoke_condition" --shard-index 0 --python "$python_bin"
done

run_phase() {
    local phase=$1
    local node_count=$2
    local task_count=$3
    local gpu_count=$4
    if srun --exclusive --exact --kill-on-bad-exit=0 \
        --nodes="$node_count" --ntasks="$task_count" --ntasks-per-node=4 \
        --gpus="$gpu_count" --gpus-per-task=1 --cpus-per-task=16 \
        --gpu-bind=verbose,per_task:1 --distribution=block:block \
        "$worker_wrapper" "$launcher" "$python_bin" "$phase" "$campaign_root" \
        "$official_source_dir" "$checkpoint_directory" /tmp; then
        return 0
    fi
    echo "ERROR: methods AITA $phase has failed ranks; preserving attempts and stopping" >&2
    return 1
}

# Phase 1: BCT, canonical repaired-ACT, AttCT, and MLPCT (4 x 4 shards).
run_phase phase-1 4 16 16

# Phase 2: OPCT's four pair-preserving shards.  The allocation remains the
# same 4-node/16-GPU allocation; this final srun deliberately consumes four.
run_phase phase-2 1 4 4

exec "$python_bin" "$launcher" finalize "${campaign_args[@]}"
