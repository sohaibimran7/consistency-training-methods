#!/usr/bin/env bash
# Source-validate with the pinned Muse runtime, then grade parsed outputs with
# one uncapped Luna client pool whose max_connections is exactly 500.

#SBATCH --job-name=muse-rmct-luna
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-muse-rmct-luna-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable Muse repository" >&2
    exit 2
fi
if [[ "${MUSE_LUNA_GROUP:-}" != early && "${MUSE_LUNA_GROUP:-}" != late ]]; then
    echo "ERROR: MUSE_LUNA_GROUP must be early or late" >&2
    exit 2
fi
if [[ "${MUSE_LUNA_MODE:-}" != smoke && "${MUSE_LUNA_MODE:-}" != full ]]; then
    echo "ERROR: MUSE_LUNA_MODE must be smoke or full" >&2
    exit 2
fi
if [[ -z "${CTM_RMCT_LUNA_ENV_FILE:-}" || ! -f "$CTM_RMCT_LUNA_ENV_FILE" || -L "$CTM_RMCT_LUNA_ENV_FILE" ]]; then
    echo "ERROR: CTM_RMCT_LUNA_ENV_FILE must name the existing mode-400/600 OpenRouter environment file" >&2
    exit 2
fi
case "$(stat -c '%a' "$CTM_RMCT_LUNA_ENV_FILE")" in
    400|600) ;;
    *) echo "ERROR: Luna credential file must have mode 400 or 600" >&2; exit 2 ;;
esac

repo_root=$(cd -- "$REPO_DIR" && pwd -P)
cd "$repo_root"
source infra/isambard/muse_glimmer_cuda129_runtime_env.sh
muse_python="$CTM_MUSE_RUNTIME_PYTHON"
grader_python="${CTM_MUSE_LUNA_GRADER_PYTHON:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo/.venv/bin/python}"
if [[ ! -x "$muse_python" || ! -x "$grader_python" ]]; then
    echo "ERROR: Muse validation or Luna grader Python is unavailable" >&2
    exit 2
fi

export PYTHONPATH="$repo_root"
export PYTHONDONTWRITEBYTECODE=1
export CTM_DISABLE_CUDNN_SDP=0
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
export CTM_HF_EOS_ONLY_EXPECTED_INSPECT=0.3.260
export CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS=5.15.1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

training_repository="${CTM_MUSE_TRAINING_REPOSITORY:-$repo_root}"
source_manifest="${CTM_MUSE_EVAL_SOURCE_STAGE2_MANIFEST:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-two-bias-eval-20260820/repo/artifacts/stage2-ood-hle-2x2-20260802-r1-source/manifest.json}"
artifact_root="${CTM_MUSE_EVAL_STAGE2_ARTIFACT_ROOT:-${source_manifest%/manifest.json}}"
evaluation_parent="${CTM_MUSE_EVAL_PARENT_ROOT:-$repo_root/logs/muse-glimmer-rmct-evaluations}"
campaign_root="${CTM_MUSE_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/muse-glimmer-rmct-two-bias-${MUSE_LUNA_GROUP}-16gpu-v1}"
output_root="${CTM_MUSE_LUNA_OUTPUT_ROOT:-$repo_root/logs/muse-glimmer-rmct-luna-no-cap-v1}"

for path in "$training_repository" "$artifact_root" "$campaign_root"; do
    if [[ ! -d "$path" || -L "$path" ]]; then
        echo "ERROR: required Muse Luna directory is missing or linked: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$source_manifest" || -L "$source_manifest" ]]; then
    echo "ERROR: Muse Luna source manifest is missing or linked" >&2
    exit 2
fi
mkdir -p "$output_root"

# No credential is loaded while the pinned Muse evaluator replays raw-log,
# checkpoint, task-receipt, preflight, and completion custody.
"$muse_python" -m experiments.muse_glimmer_rmct_replication.luna_grade_no_cap stage \
    --group "$MUSE_LUNA_GROUP" \
    --campaign-root "$campaign_root" \
    --training-repository "$training_repository" \
    --source-stage2-manifest "$source_manifest" \
    --stage2-artifact-root "$artifact_root" \
    --output-root "$output_root"

handoff="$output_root/staging/$MUSE_LUNA_GROUP/handoff.json"
if [[ ! -f "$handoff" || -L "$handoff" ]]; then
    echo "ERROR: Muse Luna source-validation handoff was not published" >&2
    exit 2
fi

set -a
. "$CTM_RMCT_LUNA_ENV_FILE"
set +a
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: Luna credential file did not provide OPENROUTER_API_KEY" >&2
    exit 2
fi

# The scorer itself requires max_connections=500 and fails if Inspect or
# OpenRouter would add any output/reasoning/completion token limit.
exec "$grader_python" -m experiments.muse_glimmer_rmct_replication.luna_grade_no_cap grade \
    --handoff "$handoff" \
    --output-root "$output_root" \
    --mode "$MUSE_LUNA_MODE" \
    --smoke-samples 2
