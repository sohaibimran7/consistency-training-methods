#!/usr/bin/env bash
# Receipt-only, fail-closed OPCT Stage 2 incremental Luna consumer.
#
# This process deliberately does *not* invoke hf_peft_resume handoff or touch
# raw EvalLogs.  The live native-HF resume owner remains the sole staging
# writer.  Once that owner has written an adjacent immutable handoff receipt,
# this consumer validates the receipt in dry-run mode and, only then, grades
# the staged immutable copy under the shared 5 x 100 = 500 connection lock.
set -euo pipefail

readonly RUN=/workspace/ctm-opct-stage2-20260804/run
readonly REPO=/workspace/ctm-opct-stage2-20260804/repo
readonly PY=/workspace/ctm-opct-stage2-20260804/env/bin/python
readonly CONDITION=opct-phase2-hf-peft
readonly CHECKPOINT=/workspace/ctm-opct-stage2-20260804/checkpoint/rmct_paper_isambard_phase2_qwen3_5_9b_opct_rng_repair_4gpu_20260803_opct-lr-1e-4
readonly CHECKPOINT_WEIGHTS="$CHECKPOINT/adapter_model.safetensors"
readonly BASE_MODEL=Qwen/Qwen3.5-9B
readonly HF_CACHE=/workspace/hf-cache-direct
readonly MANIFEST=/workspace/ctm-ood-hle-20260802/repo/artifacts/stage2-ood-hle-2x2-20260802-r1/manifest.json
readonly RAW_ROOT="$RUN/raw-no-luna/$CONDITION"
readonly RAW_CONTRACT="$RAW_ROOT/resume-contract.json"
readonly STAGED_ROOT="$RUN/luna-incremental-staged-v1"
readonly RECEIPT_ROOT="$STAGED_ROOT/$CONDITION"
readonly DERIVED_ROOT="$RUN/luna-incremental-derived-v1"
readonly CONSUMER_ROOT="$RUN/opct-incremental-luna-consumer-v1"
readonly SOURCE_ROOT="$CONSUMER_ROOT/source"
readonly SOURCE_CONTRACT="$CONSUMER_ROOT/consumer-contract.json"
readonly ENV_FILE="$REPO/.env"
readonly CONSUMER_LOCK="$RUN/.opct-incremental-luna-consumer-v1.lock"

readonly EXPECTED_CHECKPOINT_SHA256=d893a6c202e5c9b0a1d00358b1f656d21ad99168613758046e64749845d8a7dd
readonly EXPECTED_MANIFEST_SHA256=f13253b3bdc2536eb726100f419761b1a9be7db1f1cc6140f20f169f1a2d492e
readonly EXPECTED_BASE_SNAPSHOT=c202236235762e1c871ad0ccb60c8ee5ba337b9a
readonly EXPECTED_INCREMENTAL_SHA256=56209ae21e55189c44283d818c3905def3d33aa146245f3ba4d1500f1f39a9da
readonly EXPECTED_GRADE_SHA256=3cc92feca4e86f63a9328d6fffd18a14a5411334355eb2c33c3c9159a5f8938c
readonly EXPECTED_HF_PEFT_RESUME_SHA256=5af0dd4ecf72c6772ee9905655cc4e2b4b4527eb12ee45af216e54f070e27a63
readonly EXPECTED_HF_PEFT_RUNNER_SHA256=e3f139a9975be8df7036bfecf97e07624066e6e597543701fc8a55ce63c3455a
readonly EXPECTED_MATERIALIZE_SHA256=d58460994aef0792fd101480146cd66d1aeab9f3c4d968e0c2ef45afaeb514e4
readonly EXPECTED_TASKS_SHA256=81291139cab9305eadee281f7514661ece0f11ecb054995eb11b15c2b5b95d5e
readonly EXPECTED_RAW_PREFLIGHT_SHA256=837b47c4b246ed97a2b8f7c219cf720cdeac5e331f4d46877d971030f17798bc
readonly EXPECTED_STAGE_LUNA_SHA256=581c1e8636d78503f838d1537b0a0fbe24b25f8970ab66dd5db6e30c0bc74b7a
readonly EXPECTED_LUNA_SCORER_SHA256=d02d21ee37ba95ddbc4eb1d9685b3d881d394c14c412fce89c4ab47bb16bba43
readonly EXPECTED_LUNA_CONFIG_SHA256=fa3e2ab9816e4e0cc4cd4859631dc21eb88ee62232b6aabee08825d63b749594
readonly EXPECTED_GRADER_MODEL=openrouter/openai/gpt-5.6-luna-20260709
readonly POLL_SECONDS=45

# The deployed source overlay is a fixed, hash-attested input.  Avoid adding
# interpreter cache files beneath it while the consumer is running.
export PYTHONDONTWRITEBYTECODE=1
# Never permit an implicit base-model download or revision substitution while
# validating handoffs or spawning the isolated grading workers.
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

usage() {
  printf '%s\n' "usage: $0 [--once|--daemon] [--dry-run]" >&2
}

MODE=once
DRY_RUN=0
CONTRACT_VERIFIED=0
LAST_RECEIPT_FINGERPRINT=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --once) MODE=once ;;
    --daemon) MODE=daemon ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
  shift
done

test -x "$PY"
test -d "$REPO"
test -d "$RAW_ROOT"
test -f "$RAW_CONTRACT"
test -f "$SOURCE_CONTRACT"
test -d "$SOURCE_ROOT"
if [ "$STAGED_ROOT" = "$DERIVED_ROOT" ]; then
  printf '%s\n' "staged and derived roots must be distinct" >&2
  exit 2
fi

# A separate consumer lock prevents duplicate daemons.  The Python consumer
# also takes $RUN/.luna-grade.lock immediately before any scorer process is
# spawned, which serializes this process with the completed-matrix finalizer.
exec 8>"$CONSUMER_LOCK"
if ! flock -n 8; then
  printf '%s\n' "another OPCT incremental Luna consumer already owns $CONSUMER_LOCK" >&2
  exit 0
fi

verify_contract() {
  "$PY" - "$CHECKPOINT" "$CHECKPOINT_WEIGHTS" "$MANIFEST" "$RAW_ROOT" "$RAW_CONTRACT" "$REPO" "$HF_CACHE" "$BASE_MODEL" \
    "$SOURCE_ROOT" "$SOURCE_CONTRACT" "$CONDITION" "$EXPECTED_CHECKPOINT_SHA256" \
    "$EXPECTED_MANIFEST_SHA256" "$EXPECTED_BASE_SNAPSHOT" "$EXPECTED_INCREMENTAL_SHA256" \
    "$EXPECTED_GRADE_SHA256" "$EXPECTED_HF_PEFT_RESUME_SHA256" "$EXPECTED_HF_PEFT_RUNNER_SHA256" \
    "$EXPECTED_MATERIALIZE_SHA256" "$EXPECTED_TASKS_SHA256" "$EXPECTED_RAW_PREFLIGHT_SHA256" \
    "$EXPECTED_STAGE_LUNA_SHA256" "$EXPECTED_LUNA_SCORER_SHA256" "$EXPECTED_LUNA_CONFIG_SHA256" \
    "$EXPECTED_GRADER_MODEL" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    checkpoint,
    weights,
    manifest,
    raw_root,
    raw_contract,
    repo,
    hf_cache,
    base_model,
    source_root,
    source_contract,
    condition,
    expected_weights,
    expected_manifest,
    expected_base_snapshot,
    expected_incremental,
    expected_grade,
    expected_hf_peft_resume,
    expected_hf_peft_runner,
    expected_materialize,
    expected_tasks,
    expected_raw_preflight,
    expected_stage_luna,
    expected_luna_scorer,
    expected_luna_config,
    expected_model,
) = sys.argv[1:]

def require_regular(path_string: str, label: str) -> Path:
    path = Path(path_string)
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular file: {path}")
    return path.resolve()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object: {path}")
    return value

weights_path = require_regular(weights, "checkpoint weights")
manifest_path = require_regular(manifest, "frozen manifest")
raw_contract_path = require_regular(raw_contract, "raw resume contract")
source_contract_path = require_regular(source_contract, "consumer source contract")
if sha256(weights_path) != expected_weights:
    raise SystemExit("checkpoint adapter SHA-256 differs from the OPCT launch contract")
if sha256(manifest_path) != expected_manifest:
    raise SystemExit("frozen Stage 2 manifest SHA-256 differs from the OPCT launch contract")

raw = load_object(raw_contract_path, "raw resume contract")
launch = raw.get("launch_contract")
if not isinstance(launch, dict):
    raise SystemExit("raw resume contract has no launch contract")
stage2 = launch.get("stage2")
checkpoint_record = launch.get("checkpoint")
if not isinstance(stage2, dict) or not isinstance(checkpoint_record, dict):
    raise SystemExit("raw resume contract has malformed Stage 2/checkpoint bindings")
if raw.get("condition") != condition or raw.get("raw_log_dir") != raw_root:
    raise SystemExit("raw resume contract condition/raw root differs from this consumer")
if checkpoint_record.get("path") != checkpoint or checkpoint_record.get("adapter_model_sha256") != expected_weights:
    raise SystemExit("raw resume contract checkpoint binding differs from this consumer")
if stage2.get("manifest") != manifest or stage2.get("manifest_sha256") != expected_manifest:
    raise SystemExit("raw resume contract manifest binding differs from this consumer")
if stage2.get("grader_model") is not None:
    raise SystemExit("raw resume contract is not a no-Luna generation contract")

source = load_object(source_contract_path, "consumer source contract")
if source.get("schema") != "opct-stage2-incremental-luna-consumer-v1":
    raise SystemExit("consumer source contract has an unexpected schema")
base_binding = source.get("base_model")
if not isinstance(base_binding, dict):
    raise SystemExit("consumer source contract has no base-model binding")
if (
    base_binding.get("repo_id") != base_model
    or base_binding.get("snapshot") != expected_base_snapshot
    or base_binding.get("cache_root") != hf_cache
    or base_binding.get("offline_only") is not True
):
    raise SystemExit("consumer source contract has a different base-model snapshot binding")
if source.get("grader_model") != expected_model or source.get("aggregate_connection_limit") != 500:
    raise SystemExit("consumer source contract has an unexpected Luna policy")
expected_files = {
    "experiments/stage2_ood_hle/incremental_grade_luna.py": expected_incremental,
    "experiments/stage2_ood_hle/grade_luna.py": expected_grade,
    "experiments/stage2_ood_hle/hf_peft_resume.py": expected_hf_peft_resume,
    "experiments/stage2_ood_hle/hf_peft_runner.py": expected_hf_peft_runner,
    "experiments/stage2_ood_hle/materialize.py": expected_materialize,
    "experiments/stage2_ood_hle/tasks.py": expected_tasks,
    "experiments/stage2_ood_hle/raw_preflight.py": expected_raw_preflight,
    "experiments/stage2_ood_hle/stage_luna.py": expected_stage_luna,
}
recorded_files = source.get("source_files")
if not isinstance(recorded_files, dict):
    raise SystemExit("consumer source contract has no source-file digest map")
for relative, expected in expected_files.items():
    path = require_regular(str(Path(source_root) / relative), f"consumer source {relative}")
    actual = sha256(path)
    if actual != expected or recorded_files.get(relative) != expected:
        raise SystemExit(f"consumer source digest differs for {relative}")
for relative, expected in {
    "ctm_data/adapters/mcq_bias/luna_scorer.py": expected_luna_scorer,
    "ctm_data/adapters/mcq_bias/luna_config.py": expected_luna_config,
}.items():
    path = require_regular(str(Path(repo) / relative), f"repository scorer source {relative}")
    if sha256(path) != expected or recorded_files.get(relative) != expected:
        raise SystemExit(f"repository scorer digest differs for {relative}")

# c202… is the immutable Hugging Face base-model snapshot, not a repository
# commit.  Resolve both the explicit revision and the default offline ref, so
# a stale/misdirected cache cannot silently substitute another base model.
try:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig
except ImportError as exc:
    raise SystemExit("offline Hugging Face resolution dependencies are unavailable") from exc
cache = Path(hf_cache).resolve()
cache_ref = cache / f"models--{base_model.replace('/', '--')}" / "refs" / "main"
if cache_ref.is_symlink() or not cache_ref.is_file() or cache_ref.read_text(encoding="utf-8").strip() != expected_base_snapshot:
    raise SystemExit("offline Hugging Face main ref does not bind the expected base snapshot")
expected_snapshot = (cache / f"models--{base_model.replace('/', '--')}" / "snapshots" / expected_base_snapshot).resolve()
explicit_snapshot = Path(snapshot_download(
    repo_id=base_model,
    revision=expected_base_snapshot,
    cache_dir=str(cache),
    local_files_only=True,
)).resolve()
default_snapshot = Path(snapshot_download(
    repo_id=base_model,
    cache_dir=str(cache),
    local_files_only=True,
)).resolve()
if explicit_snapshot != expected_snapshot or default_snapshot != expected_snapshot:
    raise SystemExit("offline Hugging Face resolution did not produce the expected base snapshot")
config = AutoConfig.from_pretrained(
    base_model,
    revision=expected_base_snapshot,
    cache_dir=str(cache),
    local_files_only=True,
)
if getattr(config, "model_type", None) != "qwen3_5":
    raise SystemExit("offline base-model configuration is not Qwen3.5")

print(json.dumps({
    "checkpoint_sha256": expected_weights,
    "manifest_sha256": expected_manifest,
    "base_model": base_model,
    "base_model_snapshot": expected_base_snapshot,
    "base_model_resolution": "offline-cache",
    "grader_model": expected_model,
    "workers": 5,
    "connections_per_worker": 100,
    "aggregate_connection_limit": 500,
}, sort_keys=True))
PY
}

has_receipt() {
  [ -d "$RECEIPT_ROOT" ] && find "$RECEIPT_ROOT" -name '*.resume-handoff.json' -print -quit | grep -q .
}

receipt_fingerprint() {
  # Hash only the small, immutable receipt sidecars.  Do not repeatedly read
  # the large checkpoint or the active raw-log tree while workers are running.
  find "$RECEIPT_ROOT" -name '*.resume-handoff.json' -print | LC_ALL=C sort | while IFS= read -r receipt; do
    test -f "$receipt"
    test ! -L "$receipt"
    sha256sum "$receipt"
  done | sha256sum | awk '{print $1}'
}

run_incremental() {
  local dry_output
  if ! dry_output=$(PYTHONPATH="$SOURCE_ROOT:$REPO" "$PY" -m experiments.stage2_ood_hle.incremental_grade_luna \
    --staged-raw-root "$STAGED_ROOT" \
    --output-root "$DERIVED_ROOT" \
    --condition "$CONDITION" \
    --manifest "$MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --receipt-root "$RECEIPT_ROOT" \
    --dry-run 2>&1); then
    printf '%s\n' "incremental dry-run refused the staged receipts:" >&2
    printf '%s\n' "$dry_output" >&2
    return 1
  fi
  printf '%s\n' "$dry_output"
  if [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi

  # Keep credential import as close as possible to the only command that can
  # send a request.  Do not print environment values or inherit another
  # checkout's PYTHONPATH into the spawned scorer processes.
  test -f "$ENV_FILE"
  set -a
  . "$ENV_FILE"
  set +a
  if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    printf '%s\n' "approved OpenRouter credential is unavailable" >&2
    return 1
  fi
  export PYTHONPATH="$SOURCE_ROOT:$REPO"
  "$PY" -m experiments.stage2_ood_hle.incremental_grade_luna \
    --staged-raw-root "$STAGED_ROOT" \
    --output-root "$DERIVED_ROOT" \
    --condition "$CONDITION" \
    --manifest "$MANIFEST" \
    --checkpoint "$CHECKPOINT" \
    --receipt-root "$RECEIPT_ROOT"
}

run_once() {
  if ! has_receipt; then
    if [ "$CONTRACT_VERIFIED" -eq 0 ]; then
      verify_contract
      CONTRACT_VERIFIED=1
    fi
    printf '%s\n' "waiting: no immutable incremental handoff receipt exists under $RECEIPT_ROOT"
    return 0
  fi
  local current_receipt_fingerprint
  current_receipt_fingerprint=$(receipt_fingerprint)
  if [ "$current_receipt_fingerprint" = "$LAST_RECEIPT_FINGERPRINT" ]; then
    printf '%s\n' "waiting: all currently observed immutable handoff receipts were already consumed"
    return 0
  fi
  # Recheck the heavyweight immutable bindings exactly when new receipts need
  # action.  The Python consumer repeats its receipt/checkpoint verification
  # under the global grade lock immediately before a scorer can start.
  verify_contract
  CONTRACT_VERIFIED=1
  run_incremental
  LAST_RECEIPT_FINGERPRINT=$current_receipt_fingerprint
}

if [ "$MODE" = once ]; then
  run_once
  exit 0
fi

while true; do
  if ! run_once; then
    printf '%s\n' "consumer stopped after a fail-closed validation or grading error" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done
