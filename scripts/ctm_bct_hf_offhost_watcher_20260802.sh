#!/usr/bin/env bash
# Archive and postprocess the two backend-matched native-HF BCT evaluations
# locally, once their four remote raw-pair workers have each completed.
#
# This is deliberately a local watcher. Its only remote operations are SSH
# reads and rsync reads of the completed EvalLogs and runner evidence. It
# never runs BCT_*_MODE=finalize, never writes to Vast, and never transfers a
# remote .env. Luna uses the separately authorized local .env only after the
# raw evidence has been checksum-archived and strictly preflighted.
set -euo pipefail

SOURCE_ROOT=${CTM_OFFHOST_SOURCE_ROOT:-/Users/work/.codex/worktrees/d6d6/consistency-training-methods}
POSTPY=${CTM_OFFHOST_POSTPY:-/Users/work/consistency-training-methods/.venv-stage1-postprocess-20260802/bin/python}
LOCAL_ENV=${CTM_OFFHOST_LOCAL_ENV:-/Users/work/consistency-training-methods/.env}
FROZEN=${CTM_OFFHOST_FROZEN:-/Users/work/consistency-training-methods/artifacts/stage1-iid-diagnostic-none-20260801}
ARCHIVE_PARENT=${CTM_OFFHOST_ARCHIVE_PARENT:-/Users/work/consistency-training-methods/artifacts/_remote-archive-20260802/46525520}
ARCHIVE_LABEL=${CTM_OFFHOST_ARCHIVE_LABEL:-bct-hf-offhost-20260802}

REMOTE_HOST=${CTM_OFFHOST_REMOTE_HOST:-root@219.86.90.203}
REMOTE_PORT=${CTM_OFFHOST_REMOTE_PORT:-50694}
REMOTE_KNOWN_HOSTS=${CTM_OFFHOST_REMOTE_KNOWN_HOSTS:-/tmp/ctm-46525520.known_hosts}
REMOTE_REPO=${CTM_OFFHOST_REMOTE_REPO:-/workspace/ctm-eval-none-20260801/repo}
REMOTE_RUN=${CTM_OFFHOST_REMOTE_RUN:-/workspace/ctm-act-repair-20260731}
POLL_SECONDS=${CTM_OFFHOST_POLL_SECONDS:-30}

readonly BASE_MODEL=Qwen/Qwen3.5-9B
readonly GRADER_WORKERS=5
readonly GRADER_CONNECTIONS_PER_WORKER=100
readonly GRADER_MAX_TOKENS=1024
readonly RUNTIME_PROFILE=hf-peft
readonly EXPECTED_EVAL_MAX_CONNECTIONS=8
readonly LUNA_LOCK_NAME=.ctm-luna-grading-lock-20260802

SSH_CMD=(ssh -T -p "$REMOTE_PORT" -o "UserKnownHostsFile=$REMOTE_KNOWN_HOSTS" -o StrictHostKeyChecking=yes)
RSYNC_RSH="ssh -T -p $REMOTE_PORT -o UserKnownHostsFile=$REMOTE_KNOWN_HOSTS -o StrictHostKeyChecking=yes"
PAIRS=(train-logiqa train-hellaswag heldout-logiqa heldout-hellaswag)

die() {
  echo "off-host BCT watcher: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  ctm_bct_hf_offhost_watcher_20260802.sh --condition bct-main-hf-peft
  ctm_bct_hf_offhost_watcher_20260802.sh --condition bct-control-hf-peft
  ctm_bct_hf_offhost_watcher_20260802.sh --all

Waits read-only for all four remote raw-pair completion markers for the chosen
condition. It then checksum-archives the raw EvalLogs and runner evidence,
strictly validates/stages the immutable logs locally, grades with the
authorized local OpenRouter credential at 5 x 100 connections, and writes the
offline analysis report. It never invokes a remote finalizer or copies .env.
EOF
}

validate_positive_integer() {
  case "$1" in
    ''|*[!0-9]*) die "$2 must be a positive integer; got $1" ;;
  esac
  [ "$1" -gt 0 ] || die "$2 must be a positive integer; got $1"
}

validate_local_contract() {
  [ -x "$POSTPY" ] || die "missing local postprocess Python: $POSTPY"
  [ -d "$SOURCE_ROOT" ] || die "missing source tree: $SOURCE_ROOT"
  [ -f "$LOCAL_ENV" ] || die "missing authorized local .env: $LOCAL_ENV"
  [ -f "$FROZEN/manifest.json" ] || die "missing local frozen manifest"
  [ -f "$FROZEN/train-eval-n200.jsonl" ] || die "missing local frozen train split"
  [ -f "$FROZEN/heldout-in-domain-n200.jsonl" ] || die "missing local frozen held-out split"

  PYTHONPATH="$SOURCE_ROOT" "$POSTPY" - "$SOURCE_ROOT" "$FROZEN" <<'PY'
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
frozen = Path(sys.argv[2])
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"postprocess Python must be 3.12.x, got {sys.version}")
if metadata.version("inspect-ai") != "0.3.251":
    raise SystemExit("postprocess environment must pin inspect-ai==0.3.251")
if metadata.version("openai") != "2.48.0":
    raise SystemExit("postprocess environment must pin openai==2.48.0")
mcq_bias = metadata.distribution("mcq-bias")
direct_url = json.loads((Path(mcq_bias._path) / "direct_url.json").read_text())
if direct_url.get("vcs_info", {}).get("commit_id") != "1df2ea1ed8a1eeaf6ec5088c066db7c8c1049119":
    raise SystemExit("postprocess environment has the wrong mcq-bias commit")

expected = {
    "experiments/stage1_iid_diagnostic_none/raw_preflight.py": "3cb5f31b6bca317a6906515938808f94da2cfc2055f5fb5fc0e3053389dfba56",
    "experiments/stage1_iid_diagnostic_none/grade_luna.py": "0b27704811d31a9395898a34ca2124538dd2a5998b1e6d950b451a7b51e537e4",
    "experiments/stage1_iid_diagnostic/grade_luna.py": "40a65291b71498843a7ac7a2cbe7fdddce9c19f0850df7d6553035fb350947c3",
    "experiments/stage1_iid_diagnostic/analyze.py": "a5fdceeeeab5228b8a50f65f2bc82343d5b6784666c9a0c86f45988fc3b5703e",
    "ctm_data/adapters/mcq_bias/luna_scorer.py": "d02d21ee37ba95ddbc4eb1d9685b3d881d394c14c412fce89c4ab47bb16bba43",
    "ctm_data/adapters/mcq_bias/luna_config.py": "fa3e2ab9816e4e0cc4cd4859631dc21eb88ee62232b6aabee08825d63b749594",
}
for relative, expected_digest in expected.items():
    actual = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if actual != expected_digest:
        raise SystemExit(f"source hash mismatch for {relative}: {actual}")

from experiments.stage1_iid_diagnostic_none.prepare import validate_manifest
validate_manifest(frozen / "manifest.json", verify_source=True)
print("local postprocess contract verified")
PY
}

configure_condition() {
  CONDITION=$1
  case "$CONDITION" in
    bct-main-hf-peft)
      REMOTE_EVAL_DIR="$REMOTE_REPO/logs/evals/stage1-bct-main-hf-recovery-20260802"
      REMOTE_STAGE_DIR="$REMOTE_REPO/artifacts/stage1-bct-main-hf-recovery-20260802"
      REMOTE_RUNNER_ROOT="$REMOTE_RUN/runners/bct-main-hf-recovery-20260802"
      REMOTE_RUNNER_SCRIPT="$REMOTE_RUN/runners/ctm-bct-main-hf-recovery-20260802.sh"
      REMOTE_RELEASE_SCRIPT="$REMOTE_RUN/runners/ctm-bct-main-hf-release-watcher-20260802.sh"
      REMOTE_RELEASE_LOG="$REMOTE_RUN/runners/ctm-bct-main-hf-release-watcher-20260802.log"
      REMOTE_CHECKPOINT="$REMOTE_REPO/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct"
      EXPECTED_ADAPTER_MODEL_SHA256=b7b6d2545797894e9f976f497ee6e5c8b09c26a3b2185aa36be82e4c308285ac
      EXPECTED_ADAPTER_CONFIG_SHA256=b29d45fb65363175b99a9230d4bf33adefddf26e155e7b13d33d02c96e6319a9
      EXPECTED_CHECKPOINT_MANIFEST_SHA256=8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d
      COMPLETION_PREFIX="BCT-main raw pair complete"
      ;;
    bct-control-hf-peft)
      REMOTE_EVAL_DIR="$REMOTE_REPO/logs/evals/stage1-bct-control-hf-recovery-20260802"
      REMOTE_STAGE_DIR="$REMOTE_REPO/artifacts/stage1-bct-control-hf-recovery-20260802"
      REMOTE_RUNNER_ROOT="$REMOTE_RUN/runners/bct-control-hf-recovery-20260802"
      REMOTE_RUNNER_SCRIPT="$REMOTE_RUN/runners/ctm-bct-control-hf-recovery-20260802.sh"
      REMOTE_RELEASE_SCRIPT=""
      REMOTE_RELEASE_LOG=""
      REMOTE_CHECKPOINT="$REMOTE_REPO/artifacts/stage1-bct-recovery-20260802/raw-adapters/bct-control"
      EXPECTED_ADAPTER_MODEL_SHA256=84efdc5b38f488ad49ef3659518408ec4f93cf624bf6f30b535cbcf897ad19f5
      EXPECTED_ADAPTER_CONFIG_SHA256=a3814737831811f7b46f7df475cb220bec20c80f803f1af6c0ce9cd7d98f88ac
      EXPECTED_CHECKPOINT_MANIFEST_SHA256=8612a1651547ebc94348036d93bfdc71dc07f7e91ba9194c67fdc173182a5b2d
      COMPLETION_PREFIX="BCT-control raw pair complete"
      ;;
    *)
      die "unknown condition $CONDITION"
      ;;
  esac

  ARCHIVE_ROOT="$ARCHIVE_PARENT/$ARCHIVE_LABEL/$CONDITION"
  LOCAL_RAW_EVAL="$ARCHIVE_ROOT/raw-eval"
  LOCAL_RUNNER_ROOT="$ARCHIVE_ROOT/runner"
  LOCAL_STAGE="$ARCHIVE_ROOT/stage"
  PREFLIGHT="$LOCAL_STAGE/provenance/$CONDITION.raw-preflight.json"
}

# Return 0 once all raw-pair runners have written their successful completion
# markers. Return 10 while they are still pending, and 20 if somebody has
# created a remote finalization stage that would make a local handoff unsafe.
remote_condition_state() {
  "${SSH_CMD[@]}" "$REMOTE_HOST" bash -s -- \
    "$REMOTE_RUNNER_ROOT" "$REMOTE_STAGE_DIR" "$COMPLETION_PREFIX" "${PAIRS[@]}" <<'SH'
set -u
runner_root=$1
stage=$2
marker=$3
shift 3
if [ -e "$stage" ]; then
  exit 20
fi
for pair in "$@"; do
  log="$runner_root/$pair.log"
  if [ ! -f "$log" ] || ! grep -Fqx -- "$marker: $pair" "$log"; then
    exit 10
  fi
done
exit 0
SH
}

wait_for_remote_completion() {
  echo "$CONDITION: waiting read-only for four remote raw-pair completion markers"
  while :; do
    status=0
    remote_condition_state || status=$?
    case "$status" in
      0)
        echo "$CONDITION: all remote raw pairs completed; remote finalization stage is absent"
        return
        ;;
      10)
        sleep "$POLL_SECONDS"
        ;;
      20)
        die "$CONDITION: remote finalization stage already exists; refusing duplicate local postprocess"
        ;;
      255)
        # Vast occasionally resets an otherwise healthy SSH connection.  The
        # watcher is read-only at this point, so retrying only this transport
        # failure cannot duplicate a remote action or relax the completion
        # evidence required below.
        echo "$CONDITION: transient SSH transport failure; retrying read-only poll"
        sleep "$POLL_SECONDS"
        ;;
      *)
        die "$CONDITION: remote readiness check failed with status $status"
        ;;
    esac
  done
}

assert_remote_stage_absent() {
  "${SSH_CMD[@]}" "$REMOTE_HOST" test ! -e "$REMOTE_STAGE_DIR" \
    || die "$CONDITION: remote finalization stage appeared; refusing local postprocess"
}

sync_and_verify_tree() {
  remote_path=$1
  local_path=$2
  label=$3
  mkdir -p "$local_path"
  rsync -a --checksum --partial -q -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$remote_path/" "$local_path/"
  verification=$(rsync -aniq --checksum --out-format='%i %n%L' -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$remote_path/" "$local_path/")
  if [ -n "$verification" ]; then
    printf '%s\n' "$verification" >&2
    die "$CONDITION: checksum verification failed for $label"
  fi
  echo "$CONDITION: checksum-verified $label"
}

sync_and_verify_file() {
  remote_path=$1
  local_dir=$2
  label=$3
  mkdir -p "$local_dir"
  rsync -a --checksum --partial -q -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$remote_path" "$local_dir/"
  verification=$(rsync -aniq --checksum --out-format='%i %n%L' -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$remote_path" "$local_dir/")
  if [ -n "$verification" ]; then
    printf '%s\n' "$verification" >&2
    die "$CONDITION: checksum verification failed for $label"
  fi
  echo "$CONDITION: checksum-verified $label"
}

archive_remote_evidence() {
  # These are all read-only remote paths. In particular, no rsync source is a
  # repository root, so remote .env, caches, credentials, and Git metadata
  # are outside the archive by construction.
  assert_remote_stage_absent
  mkdir -p "$ARCHIVE_ROOT/provenance"
  sync_and_verify_tree "$REMOTE_EVAL_DIR" "$LOCAL_RAW_EVAL" "raw EvalLogs"
  sync_and_verify_tree "$REMOTE_RUNNER_ROOT" "$LOCAL_RUNNER_ROOT/logs" "runner logs"
  sync_and_verify_file "$REMOTE_RUNNER_SCRIPT" "$LOCAL_RUNNER_ROOT/scripts" "raw-pair runner script"
  if [ -n "$REMOTE_RELEASE_SCRIPT" ]; then
    sync_and_verify_file "$REMOTE_RELEASE_SCRIPT" "$LOCAL_RUNNER_ROOT/scripts" "GPU-release watcher script"
    sync_and_verify_file "$REMOTE_RELEASE_LOG" "$LOCAL_RUNNER_ROOT/logs" "GPU-release watcher log"
  fi
  assert_remote_stage_absent
}

stage_hash_bound_raw_logs() {
  "$POSTPY" - "$PREFLIGHT" "$LOCAL_STAGE/raw" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

report_path, raw_root = map(Path, sys.argv[1:])
report = json.loads(report_path.read_text())
for source in report["sources"]:
    origin = Path(source["raw_log"])
    expected = source["raw_log_sha256"]
    actual = hashlib.sha256(origin.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"source changed after preflight: {origin}")
    destination = raw_root / report["condition"] / source["split"] / origin.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise FileExistsError(f"refusing to replace mismatching staged log: {destination}")
    else:
        shutil.copy2(origin, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"staged raw log hash mismatch: {destination}")
PY
}

run_local_preflight_and_staging() {
  mkdir -p "$LOCAL_STAGE/provenance"
  PYTHONPATH="$SOURCE_ROOT" "$POSTPY" -m experiments.stage1_iid_diagnostic_none.raw_preflight \
    --raw-log-root "$LOCAL_RAW_EVAL" \
    --manifest "$FROZEN/manifest.json" \
    --split-file "train_eval=$FROZEN/train-eval-n200.jsonl" \
    --split-file "heldout_in_domain=$FROZEN/heldout-in-domain-n200.jsonl" \
    --condition "$CONDITION" \
    --expected-base-model "$BASE_MODEL" \
    --expected-checkpoint "$REMOTE_CHECKPOINT" \
    --runtime-profile "$RUNTIME_PROFILE" \
    --expected-max-connections "$EXPECTED_EVAL_MAX_CONNECTIONS" \
    --expected-adapter-model-sha256 "$EXPECTED_ADAPTER_MODEL_SHA256" \
    --expected-adapter-config-sha256 "$EXPECTED_ADAPTER_CONFIG_SHA256" \
    --expected-checkpoint-manifest-sha256 "$EXPECTED_CHECKPOINT_MANIFEST_SHA256" \
    --output "$PREFLIGHT"
  stage_hash_bound_raw_logs
}

LUNA_LOCK=""
LUNA_LOCK_HELD=0

release_luna_lock() {
  if [ "$LUNA_LOCK_HELD" -eq 1 ]; then
    rmdir "$LUNA_LOCK" 2>/dev/null || true
    LUNA_LOCK_HELD=0
  fi
}

on_exit() {
  release_luna_lock
}

acquire_luna_lock() {
  LUNA_LOCK="$ARCHIVE_PARENT/$ARCHIVE_LABEL/$LUNA_LOCK_NAME"
  mkdir -p "$(dirname "$LUNA_LOCK")"
  while ! mkdir "$LUNA_LOCK" 2>/dev/null; do
    echo "$CONDITION: waiting for the shared 500-connection local Luna slot"
    sleep "$POLL_SECONDS"
  done
  LUNA_LOCK_HELD=1
}

run_authorized_luna_and_analysis() {
  acquire_luna_lock
  # The user has authorized this local credential. Do not print it, copy it,
  # or send it to any host; Inspect uses it only for the approved OpenRouter
  # Luna scorer calls below.
  set -a
  . "$LOCAL_ENV"
  set +a
  [ -n "${OPENROUTER_API_KEY:-}" ] || die "authorized local .env has no OPENROUTER_API_KEY"
  export PYTHONPATH="$SOURCE_ROOT"

  "$POSTPY" -m experiments.stage1_iid_diagnostic_none.grade_luna \
    --raw-log-root "$LOCAL_STAGE/raw" \
    --preflight-report "$PREFLIGHT" \
    --output-root "$LOCAL_STAGE/graded" \
    --workers "$GRADER_WORKERS" \
    --connections-per-worker "$GRADER_CONNECTIONS_PER_WORKER" \
    --grader-max-tokens "$GRADER_MAX_TOKENS"
  release_luna_lock

  "$POSTPY" -m experiments.stage1_iid_diagnostic.analyze \
    --graded-root "$LOCAL_STAGE/graded" \
    --manifest "$FROZEN/manifest.json" \
    --grader-max-tokens "$GRADER_MAX_TOKENS" \
    --output "$LOCAL_STAGE/analysis/$CONDITION.json"
}

run_condition() {
  configure_condition "$1"
  validate_local_contract
  trap on_exit EXIT INT TERM
  wait_for_remote_completion
  archive_remote_evidence
  run_local_preflight_and_staging
  run_authorized_luna_and_analysis
  assert_remote_stage_absent
  echo "$CONDITION: local archive, strict preflight, Luna grading, and analysis complete"
}

main() {
  validate_positive_integer "$POLL_SECONDS" CTM_OFFHOST_POLL_SECONDS
  case "${1:-}" in
    --condition)
      [ "$#" -eq 2 ] || die "--condition requires exactly one condition name"
      run_condition "$2"
      ;;
    --all)
      [ "$#" -eq 1 ] || die "--all accepts no additional arguments"
      bash "$0" --condition bct-main-hf-peft &
      main_pid=$!
      bash "$0" --condition bct-control-hf-peft &
      control_pid=$!
      failed=0
      wait "$main_pid" || failed=1
      wait "$control_pid" || failed=1
      [ "$failed" -eq 0 ] || die "one or more off-host BCT watchers failed"
      ;;
    -h|--help)
      usage
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
