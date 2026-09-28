#!/usr/bin/env bash
# One non-chaining 4-node / 16-GPU RMCT early-checkpoint evaluation campaign.
#
# The allocation is deliberately a single job: it proves step-16 parity,
# seals phase 1, proves step-64 parity, then seals phases 2 and 3.  It never
# submits, queues, holds, or cancels another Slurm job.  Operators choose an
# interactive reservation or workq when submitting this file; the directives
# below are a workq default only.
#
# Three exact phases use one 16-rank srun each.  Fourteen ranks carry cells and
# ranks 14/15 are intentionally idle headroom.  --kill-on-bad-exit=0 lets all
# concurrently started ranks retain any successful outputs/attempt evidence;
# the explicit phase finalizer then refuses to seal a partial phase, requiring
# an explicit later resubmission rather than an automatic retry.

#SBATCH --job-name=ctm-rmct-s16-s64-2bias
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-rmct-s16-s64-two-bias-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable early-checkpoint evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py"
worker_wrapper="$repo_root/infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu_worker.sh"
training_repository="${CTM_RMCT_TRAINING_REPOSITORY:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo}"
if [[ ! -d "$training_repository" || -L "$training_repository" ]]; then
    echo "ERROR: CTM_RMCT_TRAINING_REPOSITORY must name the durable sealed-training checkout" >&2
    exit 2
fi
training_repository="$(cd -- "$training_repository" && pwd -P)"

python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"
if [[ ! -x "$python_requested" || ! -f "$python_requested" || ! -f "$launcher" || -L "$launcher" || ! -x "$worker_wrapper" || -L "$worker_wrapper" ]]; then
    echo "ERROR: evaluation Python/launcher/worker wrapper is unavailable or linked" >&2
    exit 2
fi
# Preserve the venv spelling rather than resolving bin/python to its base
# interpreter; vLLM/Inspect live in this selected environment.
python_directory="$(cd -L -- "$(dirname -- "$python_requested")" && pwd -L)"
python_bin="$python_directory/$(basename -- "$python_requested")"
if [[ ! -x "$python_bin" || ! -f "$python_bin" ]]; then
    echo "ERROR: resolved evaluation Python is unavailable or not a regular executable" >&2
    exit 2
fi
export PATH="$python_directory:$PATH"

"$python_bin" - <<'PY'
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: CTM_RMCT_EVAL_PYTHON must preserve the evaluation venv")
import inspect_ai  # noqa: F401
import peft  # noqa: F401
import safetensors  # noqa: F401
import torch  # noqa: F401
import transformers  # noqa: F401
import vllm

if vllm.__version__ != "0.21.0":
    raise SystemExit(f"error: frozen evaluation requires vLLM 0.21.0, got {vllm.__version__!r}")
PY
if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: selected evaluation venv does not expose vllm" >&2
    exit 2
fi
if [[ "${VLLM_USE_FLASHINFER_SAMPLER:-0}" != "0" ]]; then
    echo "ERROR: VLLM_USE_FLASHINFER_SAMPLER must be exactly 0" >&2
    exit 2
fi
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

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
    echo "ERROR: durable HF/uv cache is not writable" >&2
    exit 2
fi

source_manifest="${CTM_RMCT_EVAL_SOURCE_STAGE2_MANIFEST:-$repo_root/artifacts/stage2-ood-hle-2x2-20260802-r1-source/manifest.json}"
artifact_root="${CTM_RMCT_EVAL_STAGE2_ARTIFACT_ROOT:-${source_manifest%/manifest.json}}"
parity_manifest="${CTM_RMCT_EVAL_PARITY_MANIFEST:-$repo_root/artifacts/stage1-iid-diagnostic-none-20260801-source/manifest.json}"
parity_data="${CTM_RMCT_EVAL_PARITY_DATA:-$repo_root/artifacts/stage1-iid-diagnostic-none-20260801-source/train-eval-n200.jsonl}"
evaluation_parent="${CTM_RMCT_EVAL_PARENT_ROOT:-$repo_root/logs/rmct-evaluations}"
# r001 is preserved as failed pre-parity evidence.  r002 has wholly separate
# campaign and condition roots, so no recovery can overwrite or reuse it.
campaign_root="${CTM_RMCT_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/rmct-convergence-step16-step64-16gpu-v1-r002}"
step16_root="$evaluation_parent/rmct-convergence-step016-two-bias-v1-r002"
step64_root="$evaluation_parent/rmct-convergence-step064-two-bias-v1-r002"

for required in "$source_manifest" "$parity_manifest" "$parity_data" "$launcher" "$worker_wrapper"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required evaluation input is missing or linked: $required" >&2
        exit 2
    fi
done
if [[ ! -d "$artifact_root" || -L "$artifact_root" ]]; then
    echo "ERROR: CTM_RMCT_EVAL_STAGE2_ARTIFACT_ROOT must be a regular staged artifact directory" >&2
    exit 2
fi
for root in "$evaluation_parent" "$campaign_root" "$step16_root" "$step64_root"; do
    if [[ -e "$root" && ( -L "$root" || ! -d "$root" ) ]]; then
        echo "ERROR: output path is linked or not a directory: $root" >&2
        exit 2
    fi
done
mkdir -p "$evaluation_parent" "$campaign_root"

prepare_step() {
    local step=$1
    local output_root=$2
    local phase_one_receipt=${3:-}
    srun --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=4 --cpus-per-task=64 \
        bash -c '
            set -euo pipefail
            launcher=$1
            python_bin=$2
            training_repository=$3
            step=$4
            source_manifest=$5
            artifact_root=$6
            parity_data=$7
            parity_manifest=$8
            output_root=$9
            campaign_root=${10}
            step16_root=${11}
            step64_root=${12}
            phase_one_receipt=${13}
            args=(
                prepare
                --repository "$training_repository"
                --step "$step"
                --source-stage2-manifest "$source_manifest"
                --stage2-artifact-root "$artifact_root"
                --parity-data "$parity_data"
                --parity-manifest "$parity_manifest"
                --output-root "$output_root"
                --parity-gpus "$CUDA_VISIBLE_DEVICES"
                --python "$python_bin"
                --yes
            )
            if [[ "$step" == "64" ]]; then
                args+=(
                    --campaign-root "$campaign_root"
                    --step16-root "$step16_root"
                    --step64-root "$step64_root"
                    --step16-phase-receipt "$phase_one_receipt"
                )
            fi
            exec "$python_bin" "$launcher" "${args[@]}"
        ' bash "$launcher" "$python_bin" "$training_repository" "$step" "$source_manifest" "$artifact_root" "$parity_data" "$parity_manifest" "$output_root" "$campaign_root" "$step16_root" "$step64_root" "$phase_one_receipt"
}

run_phase() {
    local phase=$1
    # One real 4-node Slurm step per phase.  Ranks 0..13 are mapped by the
    # worker wrapper; ranks 14/15 intentionally have no task.  A nonzero
    # A worker failure does not discard sibling successes: each preserved
    # attempt is promoted if complete, and seal-phase reports missing receipts.
    # The installed switch scorer owns its frozen 3600s clean-log poll policy.
    if srun --exclusive --exact --kill-on-bad-exit=0 \
        --nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16 \
        "$worker_wrapper" "$launcher" "$python_bin" "$phase" "$campaign_root" "$step16_root" "$step64_root" /tmp; then
        return 0
    fi
    # Keep every rank's attempt directory, then make one read-only sealing
    # attempt so the job output names the missing task receipts before this
    # allocation exits.  It never retries cells or invokes a scheduler client.
    echo "ERROR: phase $phase has failed worker ranks; preserving attempts and checking receipt completeness" >&2
    seal_phase "$phase" || true
    return 1
}

seal_phase() {
    local phase=$1
    "$python_bin" "$launcher" seal-phase \
        --phase "$phase" \
        --campaign-root "$campaign_root" \
        --step16-root "$step16_root" \
        --step64-root "$step64_root"
}

# Step 16: strict custody + PEFT/vLLM translation/parity, then 14 cells 1..14.
prepare_step 16 "$step16_root"
run_phase 1
seal_phase 1

# Step 64 cannot even begin parity until phase 1's exact task receipts and
# paired-clean gate seal.  Phase 2 interleaves the seven remaining step-16
# cells with the first seven step-64 cells (including clean and biased cells).
phase_one_receipt="$campaign_root/phases/phase-001.json"
prepare_step 64 "$step64_root" "$phase_one_receipt"
run_phase 2
seal_phase 2

# Step 16 is complete after phase 2; finalize its 21 receipts and native
# two-bias preflight before the final step-64 phase begins.
"$python_bin" "$launcher" finalize --step 16 --output-root "$step16_root"

# Step 64 tasks 8..21 are allowed only once phase 2 is sealed.
run_phase 3
seal_phase 3
"$python_bin" "$launcher" finalize --step 64 --output-root "$step64_root"
