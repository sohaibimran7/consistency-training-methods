#!/usr/bin/env bash
# Source-validate the complete Gemma base campaign, then grade parsed outputs
# with the Luna pool fixed in the campaign configuration at 500 connections.
# This wrapper does not submit another job.

#SBATCH --job-name=gemma4-12b-luna
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-gemma4-12b-luna-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the new durable Gemma evaluation checkout" >&2
    exit 2
fi
mode="${CTM_GEMMA4_LUNA_MODE:-grade-and-publish}"
case "$mode" in
    grade|publish|grade-and-publish) ;;
    *) echo "ERROR: CTM_GEMMA4_LUNA_MODE must be grade, publish, or grade-and-publish" >&2; exit 2 ;;
esac
if [[ "$mode" != publish ]]; then
    if [[ -z "${CTM_RMCT_LUNA_ENV_FILE:-}" || ! -f "$CTM_RMCT_LUNA_ENV_FILE" || -L "$CTM_RMCT_LUNA_ENV_FILE" ]]; then
        echo "ERROR: CTM_RMCT_LUNA_ENV_FILE must name an existing protected OpenRouter environment file" >&2
        exit 2
    fi
    case "$(stat -c '%a' "$CTM_RMCT_LUNA_ENV_FILE")" in
        400|600) ;;
        *) echo "ERROR: Luna credential file must have mode 400 or 600" >&2; exit 2 ;;
    esac
fi

repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
if [[ "$repo_root" == "/projects/a5v/sohaib.a5v/consistency-training-methods" ]]; then
    echo "ERROR: REPO_DIR must not be the existing dirty shared checkout" >&2
    exit 2
fi
cd "$repo_root"

runtime_env="$repo_root/infra/isambard/muse_glimmer_cuda129_runtime_env.sh"
if [[ ! -f "$runtime_env" || -L "$runtime_env" ]]; then
    echo "ERROR: Gemma runtime environment file is unavailable" >&2
    exit 2
fi
export CTM_MUSE_PYTHON="${CTM_GEMMA4_EVAL_PYTHON:-/projects/a5v/sohaib.a5v/consistency-training-methods/.venv-muse-cu129/bin/python}"
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
export CTM_HF_EOS_ONLY_EXPECTED_INSPECT=0.3.260
export CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS=5.15.1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

job_cache_root="$(mktemp -d "/tmp/ctm-gemma4-luna-j${SLURM_JOB_ID:-manual}-XXXXXX")"
mkdir -p "$job_cache_root/tmp" "$job_cache_root/xdg-cache" "$job_cache_root/torchinductor" "$job_cache_root/triton"
chmod 700 "$job_cache_root" "$job_cache_root/tmp" "$job_cache_root/xdg-cache" "$job_cache_root/torchinductor" "$job_cache_root/triton"
export TMPDIR="$job_cache_root/tmp"
export TMP="$job_cache_root/tmp"
export TEMP="$job_cache_root/tmp"
export XDG_CACHE_HOME="$job_cache_root/xdg-cache"
export TORCHINDUCTOR_CACHE_DIR="$job_cache_root/torchinductor"
export TRITON_CACHE_DIR="$job_cache_root/triton"
source "$runtime_env"
python_bin="${CTM_GEMMA4_LUNA_PYTHON:-$CTM_MUSE_RUNTIME_PYTHON}"

if [[ ! -x "$python_bin" ]]; then
    echo "ERROR: Gemma Luna Python is unavailable" >&2
    exit 2
fi
"$python_bin" - <<'PY'
import importlib.metadata
import sys

if sys.prefix == sys.base_prefix:
    raise SystemExit("error: Gemma Luna worker must preserve its virtual environment")
expected = {"inspect-ai": "0.3.260", "transformers": "5.15.1"}
installed = {name: importlib.metadata.version(name) for name in expected}
if installed != expected:
    raise SystemExit(f"error: Gemma Luna runtime differs: got={installed!r}, expected={expected!r}")
PY

evaluation_parent="${CTM_GEMMA4_EVAL_PARENT_ROOT:-$repo_root/logs/gemma4-12b-base-evaluations}"
campaign_root="${CTM_GEMMA4_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/gemma4-12b-base-two-bias-50x21-16gpu-v1}"
output_root="${CTM_GEMMA4_LUNA_OUTPUT_ROOT:-$campaign_root/derived-luna-no-cap-v1}"
publication_dir="${CTM_GEMMA4_PUBLICATION_DIR:-$output_root/publication}"
for path in "$campaign_root"; do
    if [[ ! -d "$path" || -L "$path" ]]; then
        echo "ERROR: required Gemma campaign directory is missing or linked: $path" >&2
        exit 2
    fi
done
for path in "$output_root" "$publication_dir"; do
    if [[ -e "$path" && ( -L "$path" || ! -d "$path" ) ]]; then
        echo "ERROR: Gemma derived-output path is linked or not a directory: $path" >&2
        exit 2
    fi
done

# No credential is loaded while receipt and merged-log custody is checked.
"$python_bin" -m experiments.gemma4_12b_base_eval.postprocess preflight \
    --campaign-root "$campaign_root"

if [[ "$mode" == grade || "$mode" == grade-and-publish ]]; then
    set -a
    . "$CTM_RMCT_LUNA_ENV_FILE"
    set +a
    if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
        echo "ERROR: Luna credential file did not provide OPENROUTER_API_KEY" >&2
        exit 2
    fi
    "$python_bin" -m experiments.gemma4_12b_base_eval.postprocess grade \
        --campaign-root "$campaign_root" \
        --output-root "$output_root"
fi

if [[ "$mode" == publish || "$mode" == grade-and-publish ]]; then
    exec "$python_bin" -m experiments.gemma4_12b_base_eval.postprocess publish \
        --campaign-root "$campaign_root" \
        --derived-root "$output_root" \
        --output-dir "$publication_dir"
fi
