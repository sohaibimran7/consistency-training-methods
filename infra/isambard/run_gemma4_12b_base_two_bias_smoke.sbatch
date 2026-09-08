#!/usr/bin/env bash
# A separate one-GPU Gemma 4 compatibility smoke. Its two clean samples are
# discarded evidence: the 16-GPU campaign worker refuses to run until this
# receipt exists, but never merges these samples into the 1,050-generation
# matrix. This wrapper does not submit another job.

#SBATCH --job-name=gemma4-12b-smoke
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-gemma4-12b-smoke-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the new durable Gemma evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
if [[ "$repo_root" == "/projects/a5v/sohaib.a5v/consistency-training-methods" ]]; then
    echo "ERROR: REPO_DIR must not be the existing dirty shared checkout" >&2
    exit 2
fi
cd "$repo_root"

launcher="$repo_root/infra/isambard/run_gemma4_12b_base_two_bias_evals_16gpu.py"
runtime_env="$repo_root/infra/isambard/muse_glimmer_cuda129_runtime_env.sh"
for required in "$launcher" "$runtime_env"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required Gemma launcher/runtime file is missing or linked: $required" >&2
        exit 2
    fi
done

export CTM_MUSE_PYTHON="${CTM_GEMMA4_EVAL_PYTHON:-/projects/a5v/sohaib.a5v/consistency-training-methods/.venv-muse-cu129/bin/python}"
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
export CTM_HF_EOS_ONLY_EXPECTED_INSPECT=0.3.260
export CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS=5.15.1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
unset VLLM_BASE_URL VLLM_API_KEY CTM_PERSISTENT_VLLM_SERVER_METADATA

# The pinned runtime imports Torch/vLLM during its GPU probe.  Give that probe
# job-owned writable caches before sourcing it; node defaults can point at a
# protected /local/user directory.
job_cache_root="$(mktemp -d "/tmp/ctm-gemma4-smoke-j${SLURM_JOB_ID:-manual}-XXXXXX")"
mkdir -p "$job_cache_root/tmp" "$job_cache_root/xdg-cache" "$job_cache_root/torchinductor" "$job_cache_root/triton" "$job_cache_root/cuda-cache"
chmod 700 "$job_cache_root" "$job_cache_root/tmp" "$job_cache_root/xdg-cache" "$job_cache_root/torchinductor" "$job_cache_root/triton" "$job_cache_root/cuda-cache"
export TMPDIR="$job_cache_root/tmp"
export TMP="$job_cache_root/tmp"
export TEMP="$job_cache_root/tmp"
export XDG_CACHE_HOME="$job_cache_root/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$job_cache_root/torchinductor"
export TRITON_CACHE_DIR="$job_cache_root/triton"
export CUDA_CACHE_PATH="$job_cache_root/cuda-cache"
source "$runtime_env"
python_bin="$CTM_MUSE_RUNTIME_PYTHON"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: pinned Gemma evaluation Python is unavailable" >&2
    exit 2
fi
"$python_bin" - <<'PY'
import importlib.metadata
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: Gemma evaluator must preserve its pinned virtual environment")
expected = {"inspect-ai": "0.3.260", "torch": "2.13.0+cu129", "transformers": "5.15.1"}
installed = {name: importlib.metadata.version(name) for name in expected}
if installed != expected:
    raise SystemExit(f"error: Gemma evaluator runtime differs: got={installed!r}, expected={expected!r}")
PY

source_manifest="${CTM_GEMMA4_EVAL_SOURCE_STAGE2_MANIFEST:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-two-bias-eval-20260820/repo/artifacts/stage2-ood-hle-2x2-20260802-r1-source/manifest.json}"
artifact_root="${CTM_GEMMA4_EVAL_STAGE2_ARTIFACT_ROOT:-${source_manifest%/manifest.json}}"
evaluation_parent="${CTM_GEMMA4_EVAL_PARENT_ROOT:-$repo_root/logs/gemma4-12b-base-evaluations}"
campaign_root="${CTM_GEMMA4_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/gemma4-12b-base-two-bias-50x21-16gpu-v1}"

if [[ -z "${CTM_GEMMA4_12B_SNAPSHOT:-}" || ! -d "$CTM_GEMMA4_12B_SNAPSHOT" || -L "$CTM_GEMMA4_12B_SNAPSHOT" ]]; then
    echo "ERROR: CTM_GEMMA4_12B_SNAPSHOT must name the pinned, materialized Gemma snapshot" >&2
    exit 2
fi
for required in "$source_manifest"; do
    if [[ ! -f "$required" || -L "$required" ]]; then
        echo "ERROR: required Gemma input is missing or linked: $required" >&2
        exit 2
    fi
done
for required in "$artifact_root"; do
    if [[ ! -d "$required" || -L "$required" ]]; then
        echo "ERROR: required Gemma directory is missing or linked: $required" >&2
        exit 2
    fi
done
for path in "$evaluation_parent" "$campaign_root"; do
    if [[ -e "$path" && ( -L "$path" || ! -d "$path" ) ]]; then
        echo "ERROR: Gemma output path is linked or not a directory: $path" >&2
        exit 2
    fi
done
mkdir -p "$evaluation_parent"

"$python_bin" "$launcher" prepare \
    --campaign-root "$campaign_root" \
    --source-stage2-manifest "$source_manifest" \
    --stage2-artifact-root "$artifact_root" \
    --yes

exec srun --label --exclusive --exact --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 \
    "$python_bin" "$launcher" discarded-smoke \
    --campaign-root "$campaign_root" \
    --python "$python_bin"
