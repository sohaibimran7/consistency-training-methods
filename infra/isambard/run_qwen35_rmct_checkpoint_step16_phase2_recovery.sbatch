#!/usr/bin/env bash
# Isolated step-16-only recovery for the r002 early-checkpoint evaluation.
#
# This is deliberately not a replacement for the frozen 16-GPU campaign
# wrapper.  It replays that wrapper's phase-one receipt and its launch contract
# before running only phase-two ranks 0..6, which the frozen worker maps to
# step-16 tasks 15..21.  Rank 7 is never started, so no step-64 task can run.
#
# Do not call ``seal-phase 2`` here: the immutable original phase-two schema
# also requires step-64 tasks 1..7.  The frozen step-16 finalizer independently
# verifies all 21 task receipts and writes its native preflight/completion.
# After that, this file writes a separate immutable recovery-orchestration
# attestation outside the phase receipt namespace; it is not an official
# phase-two receipt and never changes step-64 state.

#SBATCH --job-name=ctm-rmct-s16-p2-recovery
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-rmct-s16-phase2-recovery-%j.out

set -euo pipefail
umask 077

if [[ -z "${REPO_DIR:-}" || ! -d "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    echo "ERROR: REPO_DIR must name the durable early-checkpoint evaluation checkout" >&2
    exit 2
fi
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
launcher="$repo_root/infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py"
worker_wrapper="$repo_root/infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu_worker.sh"
frozen_campaign_wrapper="$repo_root/infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.sbatch"
training_repository="${CTM_RMCT_TRAINING_REPOSITORY:-/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-convergence-gcall-r2-20260814/repo}"
if [[ ! -d "$training_repository" || -L "$training_repository" ]]; then
    echo "ERROR: CTM_RMCT_TRAINING_REPOSITORY must name the durable sealed-training checkout" >&2
    exit 2
fi
training_repository="$(cd -- "$training_repository" && pwd -P)"

python_requested="${CTM_RMCT_EVAL_PYTHON:-$training_repository/.venv/bin/python}"
if [[ ! -x "$python_requested" || ! -f "$python_requested" || ! -f "$launcher" || -L "$launcher" || ! -x "$worker_wrapper" || -L "$worker_wrapper" || ! -f "$frozen_campaign_wrapper" || -L "$frozen_campaign_wrapper" ]]; then
    echo "ERROR: frozen evaluation Python/launcher/worker/campaign wrapper is unavailable or linked" >&2
    exit 2
fi
# Preserve the venv spelling rather than resolving bin/python to its base
# interpreter; workers and the pre-existing receipts require this environment.
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

evaluation_parent="${CTM_RMCT_EVAL_PARENT_ROOT:-$repo_root/logs/rmct-evaluations}"
campaign_root="${CTM_RMCT_EVAL_CAMPAIGN_ROOT:-$evaluation_parent/rmct-convergence-step16-step64-16gpu-v1-r002}"
step16_root="$evaluation_parent/rmct-convergence-step016-two-bias-v1-r002"
# The frozen phase-one validator accepts this spelling only to reconstruct the
# original campaign topology.  This wrapper never runs a step-64 command,
# worker, finalizer, or phase seal.
step64_root="$evaluation_parent/rmct-convergence-step064-two-bias-v1-r002"
phase_one_receipt="$campaign_root/phases/phase-001.json"
recovery_receipt="$campaign_root/recoveries/step16-phase2-only-r001.json"

if [[ ! -d "$evaluation_parent" || -L "$evaluation_parent" || ! -d "$campaign_root" || -L "$campaign_root" || ! -d "$step16_root" || -L "$step16_root" || ! -f "$phase_one_receipt" || -L "$phase_one_receipt" ]]; then
    echo "ERROR: r002 evaluation parent/campaign/step16 root or phase-one receipt is missing or linked" >&2
    exit 2
fi

validate_recovery_boundary() {
    "$python_bin" - "$launcher" "$worker_wrapper" "$frozen_campaign_wrapper" "$campaign_root" "$step16_root" "$step64_root" <<'PY'
import importlib.util
import sys
from pathlib import Path

launcher_path, worker_path, campaign_wrapper_path, campaign_root, step16_root, step64_root = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("rmct_step16_phase2_recovery_frozen", launcher_path)
if spec is None or spec.loader is None:
    raise SystemExit("error: could not load frozen evaluator launcher")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

target = module._target(16)
paths = module._paths(step16_root, target=target)
phase_one = campaign_root / "phases" / "phase-001.json"
phase_two = campaign_root / "phases" / "phase-002.json"
if paths.completion.exists() or paths.completion.is_symlink():
    raise SystemExit("error: step-16 already has a completion receipt; no recovery allocation is permitted")
if phase_two.exists() or phase_two.is_symlink():
    raise SystemExit("error: an official phase-two receipt already exists; refusing an isolated subset recovery")

# This replays all fourteen phase-one receipts and the frozen source identities.
module.validate_phase_receipt(
    phase_one,
    phase=1,
    campaign_root=campaign_root,
    step16_root=step16_root,
    step64_root=step64_root,
)
launch, launch_sha256 = module._load_launch(paths, target=target)
_evaluation, evaluation_sha256 = module._load_evaluation_receipt(
    paths,
    target=target,
    launch=launch,
    launch_sha256=launch_sha256,
)
critical = launch.get("critical_sources")
if not isinstance(critical, dict):
    raise SystemExit("error: frozen launch contract has no critical-source identities")
for relative, path in {
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py": launcher_path,
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu_worker.sh": worker_path,
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.sbatch": campaign_wrapper_path,
}.items():
    if critical.get(relative) != module._identity(path, label=f"frozen recovery source {relative}"):
        raise SystemExit(f"error: {relative} differs from the deployed r002 launch contract")

# Existing completed subset cells are allowed to return as frozen-worker
# resumptions; every absent one is admitted only through phase two below.
for task_index in range(15, 22):
    module._load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
print("validated r002 phase-one custody and frozen launcher/worker bytes for step-16 tasks 15..21")
PY
}

write_recovery_orchestration_receipt() {
    "$python_bin" - "$launcher" "$worker_wrapper" "$frozen_campaign_wrapper" "$campaign_root" "$step16_root" "$step64_root" "$recovery_receipt" <<'PY'
import importlib.util
import sys
from pathlib import Path

(
    launcher_path,
    worker_path,
    campaign_wrapper_path,
    campaign_root,
    step16_root,
    step64_root,
    receipt_path,
) = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("rmct_step16_phase2_recovery_attestation", launcher_path)
if spec is None or spec.loader is None:
    raise SystemExit("error: could not load frozen evaluator launcher")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

target = module._target(16)
paths = module._paths(step16_root, target=target)
phase_one = campaign_root / "phases" / "phase-001.json"
phase_two = campaign_root / "phases" / "phase-002.json"
if phase_two.exists() or phase_two.is_symlink():
    raise SystemExit("error: official phase-two receipt appeared; refusing to call this a step-16-only recovery")
module.validate_phase_receipt(
    phase_one,
    phase=1,
    campaign_root=campaign_root,
    step16_root=step16_root,
    step64_root=step64_root,
)
launch, launch_sha256 = module._load_launch(paths, target=target)
_evaluation, evaluation_sha256 = module._load_evaluation_receipt(
    paths,
    target=target,
    launch=launch,
    launch_sha256=launch_sha256,
)
critical = launch.get("critical_sources")
if not isinstance(critical, dict):
    raise SystemExit("error: frozen launch contract has no critical-source identities")
source_bytes = {}
for relative, path in {
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py": launcher_path,
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu_worker.sh": worker_path,
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.sbatch": campaign_wrapper_path,
}.items():
    identity = module._identity(path, label=f"frozen recovery source {relative}")
    if critical.get(relative) != identity:
        raise SystemExit(f"error: {relative} differs from the deployed r002 launch contract")
    source_bytes[relative] = identity

task_records = []
for task_index in range(15, 22):
    if module._load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    ) is None:
        raise SystemExit(f"error: step-16 task-{task_index} is not sealed after recovery")
    task_records.append(
        {
            "task_index": task_index,
            "sample_count": module._sample_count_for_task(task_index),
            "receipt": module._identity(
                module._task_receipt_path(paths, task_index),
                label=f"step-16 recovery task-{task_index} receipt",
            ),
        }
    )

completion = module._read_json(paths.completion, label="step-16 recovery completion receipt")
if (
    completion.get("schema") != module.COMPLETION_SCHEMA
    or completion.get("condition") != target.condition
    or completion.get("target") != {"step": 16, "run_name": target.run_name, "segment_index": target.segment_index}
    or completion.get("generation") != module._sampling_contract()
):
    raise SystemExit("error: step-16 completion receipt does not retain the frozen r002 contract")
module.validate_native_preflight(paths.native_preflight)
document = {
    "schema": "rmct-checkpoint-two-bias-16gpu-step16-phase2-only-recovery-orchestration-v1",
    "campaign_root": str(campaign_root.resolve()),
    "phase_one_receipt": module._identity(phase_one, label="recovery phase-one receipt"),
    "official_phase_two_receipt": None,
    "recovered_step16": {
        "condition": target.condition,
        "output_root": str(paths.root),
        "task_indices": list(range(15, 22)),
        "launch_contract": module._identity(paths.contract, label="step-16 launch contract"),
        "evaluation_receipt": module._identity(paths.evaluation_receipt, label="step-16 evaluation receipt"),
        "task_receipts": task_records,
        "native_preflight": module._identity(paths.native_preflight, label="step-16 native preflight"),
        "completion": module._identity(paths.completion, label="step-16 completion receipt"),
    },
    "frozen_source_bytes": source_bytes,
    "orchestration": {
        "worker_phase": 2,
        "worker_ranks": list(range(0, 7)),
        "step64_worker_invocations": 0,
        "step64_finalizer_invocations": 0,
        "phase_two_seal": "not-written: frozen phase two requires step-64 tasks 1..7, which this recovery intentionally does not run",
    },
}
status = module._write_immutable_json(
    receipt_path,
    document,
    label="step-16-only recovery orchestration receipt",
)
if module._read_json(receipt_path, label="step-16-only recovery orchestration receipt") != document:
    raise SystemExit("error: recovery orchestration receipt differs after immutable write")
print({"status": status, "receipt": module._identity(receipt_path, label="step-16-only recovery orchestration receipt")})
PY
}

validate_recovery_boundary

# Exactly seven ranks are launched.  The frozen worker maps phase-two ranks
# 0..6 to step-16 tasks 15..21; rank 7 (the first step-64 cell) is absent.
# A worker failure preserves all attempts and exits this allocation; there is
# no cell retry, phase seal, scheduler operation, or step-64 action here.
if ! srun --exclusive --exact --kill-on-bad-exit=0 \
    --nodes=2 --ntasks=7 --ntasks-per-node=4 --distribution=block:block \
    --gpus-per-task=1 --cpus-per-task=16 \
    "$worker_wrapper" "$launcher" "$python_bin" 2 "$campaign_root" "$step16_root" "$step64_root" /tmp; then
    echo "ERROR: one or more step-16 recovery workers failed; preserved attempts remain unmodified" >&2
    exit 1
fi
if [[ -e "$campaign_root/phases/phase-002.json" || -L "$campaign_root/phases/phase-002.json" ]]; then
    echo "ERROR: an official phase-two receipt appeared during recovery; refusing step-16 finalization" >&2
    exit 2
fi

# This finalizer is custody-safe without an official phase-two receipt: it
# independently replays the immutable launch/evaluation records, requires all
# 21 receipt-selected raw logs, and writes the r002 native preflight/complete
# receipt.  It accepts only step 16 and never references step-64 artifacts.
"$python_bin" "$launcher" finalize --step 16 --output-root "$step16_root"
write_recovery_orchestration_receipt
