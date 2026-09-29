#!/usr/bin/env bash
# Fail-closed, CPU-only launch wrapper for the fresh no-CoT repaired-ACT
# generated-answer evaluation.  It deliberately accepts the long evaluation
# command only after the immutable chain below has been created/revalidated:
# fresh target output -> raw adapter -> passed tiny native-HF gate -> vLLM
# compatibility source hash + live parity attestation.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=${CTM_REPAIRED_ACT_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
PY=${CTM_REPAIRED_ACT_PY:-python}
RUNTIME_PROFILE=${CTM_REPAIRED_ACT_RUNTIME_PROFILE:-vllm}

usage() {
  cat >&2 <<'EOF'
usage:
  Set the required CTM_REPAIRED_ACT_* paths below, then invoke this wrapper as
  
    scripts/ctm_repaired_act_long_eval_guard_20260803.sh -- LONG_EVAL_COMMAND [ARGS...]

required environment:
  CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE  target-scoped repaired-act outputs.json
  CTM_REPAIRED_ACT_RAW_CHECKPOINT         fresh raw LocalBackend adapter
  CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION  passed tiny native-HF gate
  CTM_REPAIRED_ACT_CHAIN_ATTESTATION      write-once/revalidatable chain output

for the default CTM_REPAIRED_ACT_RUNTIME_PROFILE=vllm additionally set:
  CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER    translated adapter with parity attestation

After generation, pass CTM_REPAIRED_ACT_CHAIN_ATTESTATION to
experiments.stage1_iid_diagnostic_none.raw_preflight via
--repaired-act-chain-attestation before Luna staging or grading.
EOF
  exit 2
}

if [ "${1:-}" != "--" ]; then
  usage
fi
shift
if [ "$#" -eq 0 ]; then
  usage
fi

: "${CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE:?set target-scoped fresh repaired-ACT outputs.json}"
: "${CTM_REPAIRED_ACT_RAW_CHECKPOINT:?set fresh raw repaired-ACT adapter path or file URI}"
: "${CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION:?set passed tiny native-HF gate attestation}"
: "${CTM_REPAIRED_ACT_CHAIN_ATTESTATION:?set write-once repaired-ACT chain attestation output}"

case "$RUNTIME_PROFILE" in
  vllm)
    : "${CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER:?set translated vLLM compatibility adapter}"
    COMPAT_ARGS=(--vllm-compat-adapter "$CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER")
    ;;
  hf-peft)
    if [ -n "${CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER:-}" ]; then
      echo "CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER applies only to vllm" >&2
      exit 2
    fi
    COMPAT_ARGS=()
    ;;
  *)
    echo "CTM_REPAIRED_ACT_RUNTIME_PROFILE must be vllm or hf-peft" >&2
    exit 2
    ;;
esac

test -d "$REPO"
test -f "$CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE"
test -f "$CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION"

cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# This command validates files and hashes only; it never initializes CUDA or
# starts vLLM.  A resumed identical attestation is revalidated, while a
# different pre-existing output is rejected.
"$PY" -m experiments.rmct_paper_vast_dense_models.stage1.repaired_act_evaluation_guard \
  --training-output-state "$CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE" \
  --checkpoint "$CTM_REPAIRED_ACT_RAW_CHECKPOINT" \
  --tiny-gate-attestation "$CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION" \
  --runtime-profile "$RUNTIME_PROFILE" \
  "${COMPAT_ARGS[@]}" \
  --output "$CTM_REPAIRED_ACT_CHAIN_ATTESTATION"

exec "$@"
