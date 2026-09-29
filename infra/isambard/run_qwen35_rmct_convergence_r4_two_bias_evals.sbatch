#!/usr/bin/env bash
# One non-chaining Stage-2 evaluation window for the terminal r4 checkpoint.
# Submit this exact file directly.  It never invokes the scheduler client, so
# the operator remains the only owner of interactive/workq placement and retries.
# The directives below are a workq default; an interactive reservation may
# override --partition and --time at submission without changing this file.
#SBATCH --job-name=ctm-rmct-r4-s011-eval
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct-r4-s011-two-bias-eval-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable two-bias evaluation source checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.py"
training_repository="${CTM_RMCT_TRAINING_REPOSITORY:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo}"
if [[ ! -d "$training_repository" || -L "$training_repository" ]]; then
    echo "ERROR: CTM_RMCT_TRAINING_REPOSITORY must name the durable sealed-training checkout" >&2
    exit 2
fi
training_repository="$(cd -- "$training_repository" && pwd -P)"

# The source/evaluation checkout may intentionally have no environment.  An
# operator may point this at the proven training-checkout interpreter without
# making either checkout mutable.
python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"
if [[ ! -x "$python_requested" ]]; then
    echo "ERROR: CTM_RMCT_EVAL_PYTHON must name an executable evaluation interpreter" >&2
    exit 2
fi
# Keep the venv's ``bin/python`` spelling.  Resolving its final symlink would
# invoke the base uv interpreter and silently drop the venv site-packages.
python_directory="$(cd -L -- "$(dirname -- "$python_requested")" && pwd -L)"
python_bin="$python_directory/$(basename -- "$python_requested")"

if [[ -z "$python_bin" || ! -x "$python_bin" || ! -f "$python_bin" || ! -f "$launcher" || -L "$launcher" ]]; then
    echo "ERROR: r4 two-bias evaluation Python or launcher is unavailable" >&2
    exit 2
fi
"$python_bin" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: CTM_RMCT_EVAL_PYTHON must preserve an activated venv, not its base interpreter")
import inspect_ai  # noqa: F401
import peft  # noqa: F401
import safetensors  # noqa: F401
import torch  # noqa: F401
import transformers  # noqa: F401
import vllm

if vllm.__version__ != "0.21.0":
    raise SystemExit(f"error: r005 requires vLLM 0.21.0, got {vllm.__version__!r}")
PY
# ``runtime_parity`` starts native vLLM through ``shutil.which('vllm')``;
# calling the interpreter by absolute path does not otherwise activate its
# sibling command-line entry points.
export PATH="$python_directory:$PATH"
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: the selected evaluation venv does not expose the vllm executable" >&2
    exit 2
fi
# vLLM 0.21.0 selects FlashInfer sampling by default.  On this Isambard
# image that attempts a JIT build requiring nvcc, which is intentionally not
# present.  Treat a conflicting caller override as an ABI error and export
# the exact native-sampler setting inherited by parity and persistent eval.
if [[ "${VLLM_USE_FLASHINFER_SAMPLER:-0}" != "0" ]]; then
    echo "ERROR: VLLM_USE_FLASHINFER_SAMPLER must be exactly 0 for r005" >&2
    exit 2
fi
export VLLM_USE_FLASHINFER_SAMPLER=0

# The user-facing scheduler choice is intentionally external to this wrapper:
# provide the desired partition and safe time-window when submitting it.
# This optional marker catches accidental use of an unrelated reservation but
# never changes a scheduler request or creates a duplicate job.
scheduler_mode="${CTM_RMCT_EVAL_SCHEDULER_MODE:-workq}"
case "$scheduler_mode" in
    workq|interactive) ;;
    *) echo "ERROR: CTM_RMCT_EVAL_SCHEDULER_MODE must be workq or interactive" >&2; exit 2 ;;
esac

source_manifest="${CTM_RMCT_EVAL_SOURCE_STAGE2_MANIFEST:-$repo_root/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json}"
artifact_root="${CTM_RMCT_EVAL_STAGE2_ARTIFACT_ROOT:-${source_manifest%/manifest.json}}"
parity_manifest="${CTM_RMCT_EVAL_PARITY_MANIFEST:-$repo_root/artifacts/stage1-iid-diagnostic-none-20260801-source/manifest.json}"
parity_data="${CTM_RMCT_EVAL_PARITY_DATA:-$repo_root/artifacts/stage1-iid-diagnostic-none-20260801-source/train-eval-n200.jsonl}"
# r001--r004 contain preserved failed or superseded parity attempts and are
# deliberately immutable.  r005 starts a fresh write-once launch contract
# with the exact processed-logprob, allowed-token restricted-softmax protocol
# and the explicit 29-score vLLM server admission cap.
evaluation_root="${CTM_RMCT_EVAL_OUTPUT_ROOT:-$repo_root/logs/rmct-evaluations/rmct-convergence-gcall-r2-mb40960-r4-s011-two-bias-v1-r005}"

for required in "$source_manifest" "$parity_manifest" "$parity_data" "$launcher"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required evaluation input is missing or linked: $required" >&2
        exit 2
    fi
done
if [[ ! -d "$artifact_root" || -L "$artifact_root" ]]; then
    echo "ERROR: CTM_RMCT_EVAL_STAGE2_ARTIFACT_ROOT must be a regular staged artifact directory: $artifact_root" >&2
    exit 2
fi
if [[ -e "$evaluation_root" && ( -L "$evaluation_root" || ! -d "$evaluation_root" ) ]]; then
    echo "ERROR: evaluation output root is linked or not a directory: $evaluation_root" >&2
    exit 2
fi

if [[ -z "${SCRATCHDIR:-}" || ! -d "$SCRATCHDIR" ]]; then
    echo "ERROR: SCRATCHDIR must name durable shared scratch" >&2
    exit 2
fi
scratch_root="$(cd -- "$SCRATCHDIR" && pwd -P)"
export HF_HOME="${HF_HOME:-$scratch_root/ctm/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$scratch_root/ctm/uv-cache}"
export UV_LINK_MODE=copy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
if [[ ! -w "$HF_HOME" || ! -w "$UV_CACHE_DIR" ]]; then
    echo "ERROR: durable Hugging Face or uv cache is not writable" >&2
    exit 2
fi

ctm_eval_tmp="$(mktemp -d "/tmp/ctm-rmct-r4-s011-eval-${USER:-user}-${SLURM_JOB_ID:-manual}-XXXXXX")"
mkdir -p "$ctm_eval_tmp/tmp" "$ctm_eval_tmp/xdg-cache" "$ctm_eval_tmp/torchinductor" "$ctm_eval_tmp/triton" "$ctm_eval_tmp/cuda-cache"
chmod 700 "$ctm_eval_tmp" "$ctm_eval_tmp/tmp" "$ctm_eval_tmp/xdg-cache" "$ctm_eval_tmp/torchinductor" "$ctm_eval_tmp/triton" "$ctm_eval_tmp/cuda-cache"
export TMPDIR="$ctm_eval_tmp/tmp"
export XDG_CACHE_HOME="$ctm_eval_tmp/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$ctm_eval_tmp/torchinductor"
export TRITON_CACHE_DIR="$ctm_eval_tmp/triton"
export CUDA_CACHE_PATH="$ctm_eval_tmp/cuda-cache"

exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 \
    bash -c '
        set -euo pipefail
        training_repository=$1
        python_bin=$2
        launcher=$3
        source_manifest=$4
        artifact_root=$5
        parity_manifest=$6
        parity_data=$7
        evaluation_root=$8

        gpu_tokens="$("$python_bin" - <<PY
import os

tokens = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
if (
    len(tokens) != 4
    or any(not value or value in {"-1", "NoDevFiles"} or any(character.isspace() for character in value) for value in tokens)
    or len(set(tokens)) != 4
):
    raise SystemExit("error: r4 two-bias evaluation requires exactly four distinct Slurm-visible GPUs")
print(",".join(tokens))
PY
        )"

        exec "$python_bin" "$launcher" \
            --repository "$training_repository" \
            --source-stage2-manifest "$source_manifest" \
            --stage2-artifact-root "$artifact_root" \
            --parity-manifest "$parity_manifest" \
            --parity-data "$parity_data" \
            --output-root "$evaluation_root" \
            --gpus "$gpu_tokens" \
            --python "$python_bin" \
            --yes
    ' bash "$training_repository" "$python_bin" "$launcher" "$source_manifest" "$artifact_root" "$parity_manifest" "$parity_data" "$evaluation_root"
