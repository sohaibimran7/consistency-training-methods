#!/usr/bin/env bash
#SBATCH --job-name=ctm-rmct-r005-luna
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-rmct-r005-luna-%j.out

set -euo pipefail
umask 077

readonly REPO=/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-two-bias-eval-20260820/repo
readonly PY=/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo/.venv/bin/python
readonly EVAL_ROOT="$REPO/logs/rmct-evaluations/rmct-convergence-gcall-r2-mb40960-r4-s011-two-bias-v1-r005"
readonly RAW_ROOT="$EVAL_ROOT/stage2/raw"
readonly EVALUATION_RECEIPT="$EVAL_ROOT/runtime/evaluation-receipt.json"
readonly PREFLIGHT_REPORT="$EVAL_ROOT/stage2/preflight/rmct-convergence-r4-s011-two-bias-v1-r005.json"
readonly OUTPUT_ROOT=/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-two-bias-luna-20260821
readonly MODE=${CTM_RMCT_LUNA_MODE:?set CTM_RMCT_LUNA_MODE to smoke or full}
readonly ENV_FILE=${CTM_RMCT_LUNA_ENV_FILE:?set CTM_RMCT_LUNA_ENV_FILE to a mode-600 OpenRouter env file}

case "$MODE" in
  smoke|full) ;;
  *) printf 'unsupported CTM_RMCT_LUNA_MODE=%s\n' "$MODE" >&2; exit 2 ;;
esac

verify_source() {
  local relative=$1
  local expected=$2
  local path="$REPO/$relative"
  test -f "$path"
  test ! -L "$path"
  local actual
  actual=$(sha256sum "$path")
  actual=${actual%% *}
  if [ "$actual" != "$expected" ]; then
    printf 'source hash mismatch: %s\n' "$relative" >&2
    exit 2
  fi
}

verify_source experiments/rmct_two_bias_eval/luna_bridge.py 25f6b6ab7688645a3b07683995881db5073e077c8f4c9cb04b5febebc1df1b2d
verify_source experiments/rmct_two_bias_eval/contract.py 328d6cfc9ee2b2e026761743994a3e1edae406ccb2ded0ef73cade768042e132
verify_source experiments/rmct_two_bias_eval/raw_preflight.py fba028a544ce0faea059e737d6bbc3d4ff66e4f37b7ee79325534ac603b5b323
verify_source ctm_data/adapters/mcq_bias/luna_scorer.py d02d21ee37ba95ddbc4eb1d9685b3d881d394c14c412fce89c4ab47bb16bba43
verify_source ctm_data/adapters/mcq_bias/luna_config.py fa3e2ab9816e4e0cc4cd4859631dc21eb88ee62232b6aabee08825d63b749594

test -x "$PY"
test -d "$RAW_ROOT"
test -f "$EVALUATION_RECEIPT"
test -f "$PREFLIGHT_REPORT"
test -f "$ENV_FILE"
test ! -L "$ENV_FILE"
case "$(stat -c '%a' "$ENV_FILE")" in
  400|600) ;;
  *) printf 'credential file must have mode 400 or 600: %s\n' "$ENV_FILE" >&2; exit 2 ;;
esac

"$PY" -c 'import inspect_ai, mcq_bias, openai, sys; raise SystemExit(0 if sys.prefix != sys.base_prefix else 2)'

set -a
. "$ENV_FILE"
set +a
test -n "${OPENROUTER_API_KEY:-}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO"
cd "$REPO"

args=(
  --raw-log-root "$RAW_ROOT"
  --evaluation-receipt "$EVALUATION_RECEIPT"
  --preflight-report "$PREFLIGHT_REPORT"
  --output-root "$OUTPUT_ROOT"
)
if [ "$MODE" = smoke ]; then
  args+=(--smoke --smoke-samples 2)
else
  args+=(--full)
fi

printf 'RMCT_R005_LUNA_START mode=%s job=%s utc=%s\n' "$MODE" "${SLURM_JOB_ID:-none}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -m experiments.rmct_two_bias_eval.luna_bridge "${args[@]}"
printf 'RMCT_R005_LUNA_COMPLETE mode=%s job=%s utc=%s\n' "$MODE" "${SLURM_JOB_ID:-none}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
