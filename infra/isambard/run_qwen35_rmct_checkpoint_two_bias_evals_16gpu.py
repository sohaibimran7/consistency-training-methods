#!/usr/bin/env python3
"""Isolated, resumable 16-GPU evaluation of the sealed RMCT step-16/64 states.

This entrypoint deliberately owns a *new* two-checkpoint campaign.  It never
opens, writes, or treats the terminal step-176/r005 evaluation tree as input.
Only the two reviewed continuation parents are admissible:

* the original gcall-r2 s001 boundary at optimizer step 16; and
* the r3 s004 boundary at optimizer step 64.

The Slurm wrapper calls this module in four small modes:

``prepare``
    validate one complete, resumable checkpoint; make a fresh PEFT-to-vLLM
    compatibility adapter; and require the exact four-server parity proof.
``worker``
    generate one of the 21 frozen Stage-2 cells on the one GPU granted by an
    exclusive Slurm step, then immediately promote a successful EvalLog into
    the shared canonical raw tree.
``seal-phase``
    fail closed unless every receipt for one of the fixed 14-cell phases is
    present.  The phase seal prevents phase 2/3 from starting after a partial
    predecessor.
``finalize``
    require all 21 task receipts, exact raw-log custody, and a fresh native
    two-seen/four-held-out preflight report.

The worker intentionally does *not* impose a clean-before-biased barrier:
all cells in a phase are independent ranks of one Slurm ``srun`` step.  Biased
task arguments name the shared canonical raw root and the installed
``switch_scorer`` waits for its matching clean log.  Promotion happens in the
clean worker immediately after that worker returns, so this is both safe and
maximally parallel within a phase.

This program does not call ``sbatch``, ``squeue``, ``scancel``, or any other
scheduler client.  Its shell wrapper is likewise a single non-chaining job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LAUNCH_SCHEMA = "rmct-checkpoint-two-bias-16gpu-launch-v2"
RUNTIME_SCHEMA = "rmct-checkpoint-two-bias-16gpu-runtime-v1"
EVALUATION_RECEIPT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-evaluation-receipt-v2"
TASK_RECEIPT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-task-receipt-v2"
PHASE_RECEIPT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-phase-receipt-v2"
COMPLETION_SCHEMA = "rmct-checkpoint-two-bias-16gpu-completion-v2"
NATIVE_PREFLIGHT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-native-preflight-v2"
CLEAN_GATE_RECEIPT_SCHEMA = "rmct-checkpoint-two-bias-16gpu-clean-gate-receipt-v2"

TRAINING_CONDITION = "rmct-convergence"
RUN_PREFIX_STEP16 = "rmct-convergence-gcall-r2"
RUN_PREFIX_STEP64 = "rmct-convergence-gcall-r2-mb49152-r3"
MODEL_SNAPSHOT = Path(
    "/lus/lfs1aip2/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
MODEL_ALIAS = "Qwen/Qwen3.5-9B"
TASK_FACTORY = "experiments.stage2_ood_hle.tasks:ood_tasks"
TASK_COUNT = 21
CLEAN_TASK_INDICES = (1, 2, 3)
BIASED_TASK_INDICES = tuple(range(4, TASK_COUNT + 1))
# r002 intentionally evaluates the two in-domain clean references and every
# in-domain biased cell on the same ordered 50-question pool.  HLE keeps its
# complete 100-question population.  These task-index identities are frozen:
# they are the task-factory execution order, not a derived scheduler mapping.
IID_CLEAN_TASK_INDICES = (1, 2)
HLE_CLEAN_TASK_INDICES = (3,)
IID_BIASED_TASK_INDICES = (4, 5, *range(7, 17))
HLE_BIASED_TASK_INDICES = (6, *range(17, 22))
IID_TASK_INDICES = (*IID_CLEAN_TASK_INDICES, *IID_BIASED_TASK_INDICES)
HLE_TASK_INDICES = (*HLE_CLEAN_TASK_INDICES, *HLE_BIASED_TASK_INDICES)
TASK_SAMPLE_COUNTS = {
    **{task_index: 50 for task_index in IID_TASK_INDICES},
    **{task_index: 100 for task_index in HLE_TASK_INDICES},
}
TOTAL_SAMPLES_PER_CHECKPOINT = sum(TASK_SAMPLE_COUNTS.values())
SEEN_BIASES = ("wrong_argument", "suggested_answer")
HELD_OUT_BIASES = ("distractor_fact", "post_hoc", "spurious_few_shot_squares", "wrong_few_shot")
ALL_BIASES = (*SEEN_BIASES, *HELD_OUT_BIASES)

# These are the frozen r005 generation/runtime controls.  The 40,960-token
# training microbatch cap is *not* an evaluator knob and is intentionally not
# copied here; target checkpoint custody binds the state being served.
GENERATION_CONFIG = {
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "extra_body": {"top_k": 20},
}
VLLM_MODEL_ARGS = {
    "provider": "vllm",
    "gpu_memory_utilization": 0.9,
    "max_model_len": 32768,
    "language_model_only": True,
    "max_num_seqs": 256,
    "gdn_prefill_backend": "triton",
}
VLLM_VERSION = "0.21.0"
VLLM_SAMPLER_ENVIRONMENT = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
PARITY_VARIANTS = ("full", "linear_only", "self_attn_only", "evaluator_path")
PARITY_TOP_TOKEN_COUNT = 16
PARITY_MAX_LOGPROBS = 29
PARITY_SAMPLING = {
    "max_tokens": 1,
    "min_tokens": 0,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
    "ignore_eos": True,
}
PARITY_LOGPROBS_MODE = "processed_logprobs"
PARITY_SCORE_TRANSPORT = "allowed_token_ids_restricted_softmax"
STAGE1_PARITY_MANIFEST_SHA256 = "8cdd4da0575a125b01b2b9b62d9c0e07176142eec6194e72e8abd04401ca2fec"
STAGE1_PARITY_MANIFEST_KIND = "stage1_iid_diagnostic_none_manifest"
STAGE1_PARITY_MANIFEST_SCHEMA_VERSION = 1
STAGE1_PARITY_TRAIN_EVAL_FILENAME = "train-eval-n200.jsonl"
STAGE1_PARITY_TRAIN_EVAL_ROWS = 200

# The campaign is intentionally three sealed 14-worker phases, rather than a
# clean phase followed by a biased phase.  Two of the 16 allocation slots are
# left free in every generation phase for scheduler/retry headroom.
PHASES: dict[int, dict[int, tuple[int, ...]]] = {
    1: {16: tuple(range(1, 15))},
    2: {16: tuple(range(15, 22)), 64: tuple(range(1, 8))},
    3: {64: tuple(range(8, 22))},
}

CRITICAL_SOURCES = (
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.py",
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu.sbatch",
    "infra/isambard/run_qwen35_rmct_checkpoint_two_bias_evals_16gpu_worker.sh",
    "scripts/run_evals.py",
    "ctm/evals/runner.py",
    "ctm/evals/local_model.py",
    "ctm/evals/qwen35_vllm_attestation.py",
    "ctm/training/resume_state.py",
    "experiments/act_repair_gate/runtime_parity.py",
    "experiments/act_repair_gate/vllm_compat_adapter.py",
    "experiments/stage2_ood_hle/tasks.py",
    "experiments/stage2_ood_hle/materialize.py",
    "experiments/stage2_ood_hle/prepare.py",
    "experiments/stage2_ood_hle/raw_preflight.py",
    "experiments/rmct_two_bias_eval/contract.py",
    "experiments/rmct_two_bias_eval/deployment.py",
    "experiments/rmct_two_bias_eval/raw_preflight.py",
    "experiments/rmct_convergence/controller.py",
)


class EvaluationError(ValueError):
    """A launch input, receipt, parity proof, or raw cell is unsafe."""


@dataclass(frozen=True)
class ApprovedTarget:
    """One exact sealed checkpoint accepted by this campaign."""

    step: int
    run_prefix: str
    run_name: str
    segment_index: int
    condition: str

    @property
    def checkpoint_relative(self) -> str:
        return f"logs/{TRAINING_CONDITION}/{self.run_name}/checkpoints/{TRAINING_CONDITION}_{self.run_name}"

    @property
    def segment_relative(self) -> str:
        return f"logs/{TRAINING_CONDITION}/{self.run_name}/segment"

    @property
    def checkpoint_receipt_relative(self) -> str:
        return f"{self.segment_relative}/checkpoint-receipt.json"

    @property
    def completion_receipt_relative(self) -> str:
        return f"{self.segment_relative}/completion-receipt.json"

    @property
    def decisions_relative(self) -> str:
        return f"logs/{TRAINING_CONDITION}/{self.run_name}/decisions"


APPROVED_TARGETS: dict[int, ApprovedTarget] = {
    16: ApprovedTarget(
        step=16,
        run_prefix=RUN_PREFIX_STEP16,
        run_name="rmct-convergence-gcall-r2-s001",
        segment_index=0,
        condition="rmct-convergence-step016-two-bias-v1-r002",
    ),
    64: ApprovedTarget(
        step=64,
        run_prefix=RUN_PREFIX_STEP64,
        run_name="rmct-convergence-gcall-r2-mb49152-r3-s004",
        segment_index=3,
        condition="rmct-convergence-step064-two-bias-v1-r002",
    ),
}


@dataclass(frozen=True)
class Paths:
    root: Path
    contract: Path
    deployment_manifest: Path
    raw: Path
    attempts: Path
    receipts: Path
    runtime: Path
    runtime_receipt: Path
    evaluation_receipt: Path
    clean_gate_receipt: Path
    native_preflight: Path
    completion: Path


def _target(step: int) -> ApprovedTarget:
    if isinstance(step, bool) or step not in APPROVED_TARGETS:
        raise EvaluationError("only the reviewed optimizer-step 16 and 64 checkpoints are admissible")
    return APPROVED_TARGETS[step]


def _sample_count_for_task(task_index: int) -> int:
    """Return the exact r002 source-sample cap for one frozen task index."""

    if isinstance(task_index, bool) or task_index not in TASK_SAMPLE_COUNTS:
        raise EvaluationError("worker task index is outside the frozen 21-cell sampling matrix")
    return TASK_SAMPLE_COUNTS[task_index]


def _task_sample_count_records() -> list[dict[str, int]]:
    """Use a JSON-stable list rather than a mapping whose integer keys drift."""

    return [
        {"task_index": task_index, "sample_count": _sample_count_for_task(task_index)}
        for task_index in range(1, TASK_COUNT + 1)
    ]


def _sampling_contract() -> dict[str, Any]:
    """The complete mixed-count plan bound in every r002 custody layer."""

    task_identities = [
        {
            "task_index": task_index,
            "kind": kind,
            "regime": regime,
            "population": population,
            "dataset": dataset,
            "bias_type": bias_type,
        }
        for task_index, (kind, regime, population, dataset, bias_type) in enumerate(
            _expected_r002_task_identities(), start=1
        )
    ]
    return {
        "task_sample_counts": _task_sample_count_records(),
        "task_identities": task_identities,
        "total_samples_per_checkpoint": TOTAL_SAMPLES_PER_CHECKPOINT,
        "iid_clean_task_indices": list(IID_CLEAN_TASK_INDICES),
        "hle_clean_task_indices": list(HLE_CLEAN_TASK_INDICES),
        "iid_biased_task_indices": list(IID_BIASED_TASK_INDICES),
        "hle_biased_task_indices": list(HLE_BIASED_TASK_INDICES),
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"{label} must be a regular file: {path}")
    size = path.stat().st_size
    if size < 1:
        raise EvaluationError(f"{label} must not be empty: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "size_bytes": size}


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvaluationError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must contain a JSON object: {path}")
    return value


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(f"{label} escapes expected root: {resolved}") from exc
    return resolved


def _resolve_unlinked(value: str | Path, *, label: str) -> Path:
    """Resolve an input only after rejecting a direct symlink spelling.

    Receipts record resolved absolute identities, but an operator-facing root
    or receipt argument must not smuggle a different location in through a
    direct symlink.  (Known Hugging Face *files inside* the pinned snapshot
    are separately handled by ``_validate_model_snapshot``.)
    """

    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise EvaluationError(f"{label} must not be a symlink: {candidate}")
    return candidate.resolve()


def _write_immutable_json(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    """Atomically create a JSON receipt, accepting byte-identical resume only."""

    payload = _canonical_json(value)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise EvaluationError(f"{label} parent must be a regular directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"{label} appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _paths(output_root: str | Path, *, target: ApprovedTarget) -> Paths:
    requested = Path(output_root).expanduser()
    if requested.is_symlink() or (requested.exists() and not requested.is_dir()):
        raise EvaluationError(f"evaluation output root must be a regular directory: {requested}")
    root = requested.resolve()
    if root.name != target.condition:
        raise EvaluationError(
            f"step-{target.step} output root must be named exactly {target.condition!r}; got {root.name!r}"
        )
    # This is a hard isolation guard, not merely a naming convention.  The
    # pre-existing r005 terminal evidence must remain off limits.
    if "s011-two-bias-v1-r005" in str(root):
        raise EvaluationError("the step-16/64 campaign must not use an r005 output/evidence root")
    return Paths(
        root=root,
        contract=root / "launch-contract.json",
        deployment_manifest=root / "input" / "stage2-deployment-manifest.json",
        # This is both the canonical raw tree and the receipt-aware clean gate
        # supplied to every biased switch scorer.  A clean `.eval` appears
        # here only after its immutable task receipt has been published.
        raw=root / "stage2" / "paired-clean-ready",
        attempts=root / "stage2" / "attempts",
        receipts=root / "stage2" / "receipts",
        runtime=root / "runtime",
        runtime_receipt=root / "runtime" / "runtime-receipt.json",
        evaluation_receipt=root / "runtime" / "evaluation-receipt.json",
        clean_gate_receipt=root / "stage2" / "paired-clean-ready" / "clean-gate-receipt.json",
        native_preflight=root / "stage2" / "preflight" / "native-two-bias.json",
        completion=root / "stage2" / "completion.json",
    )


def _critical_source_identities() -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative in CRITICAL_SOURCES:
        path = _under_root(PROJECT_ROOT / relative, PROJECT_ROOT, label=f"critical source {relative}")
        records[relative] = _identity(path, label=f"critical source {relative}")
    return records


def _validate_model_snapshot() -> dict[str, Any]:
    """Validate the fixed offline snapshot, including normal HF blob links."""

    snapshot = MODEL_SNAPSHOT
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.parent.name != "snapshots":
        raise EvaluationError(f"pinned Qwen3.5 snapshot is absent or incomplete: {snapshot}")
    logical_config = snapshot / "config.json"
    if not logical_config.exists():
        raise EvaluationError(f"pinned Qwen3.5 snapshot is absent or incomplete: {snapshot}")
    if logical_config.is_symlink():
        blobs = snapshot.parent.parent / "blobs"
        if blobs.is_symlink() or not blobs.is_dir():
            raise EvaluationError("pinned Qwen3.5 snapshot has no regular Hugging Face blob tree")
        resolved = logical_config.resolve()
        if resolved.is_symlink() or not resolved.is_file() or resolved.stat().st_size < 1:
            raise EvaluationError("pinned Qwen3.5 config resolves to an invalid blob")
        try:
            resolved.relative_to(blobs.resolve())
        except ValueError as exc:
            raise EvaluationError("pinned Qwen3.5 config escapes its own model blob tree") from exc
    else:
        _identity(logical_config, label="pinned Qwen3.5 config")
        resolved = logical_config.resolve()
    return {
        "path": str(snapshot),
        "config": {
            "logical_path": str(logical_config),
            "resolved_path": str(resolved),
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        },
    }


def _strict_checkpoint_identity(repository: Path, target: ApprovedTarget) -> dict[str, Any]:
    """Require every adapter, optimizer, manifest, and replicated RNG artifact.

    A weights-only LoRA is deliberately insufficient.  The evaluator never
    resumes training, but full resumability is the custody proof that makes a
    historical checkpoint eligible for comparison.
    """

    checkpoint = _under_root(repository / target.checkpoint_relative, repository, label="approved checkpoint")
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise EvaluationError(f"approved step-{target.step} checkpoint is absent or linked: {checkpoint}")
    try:
        from ctm.training.resume_state import load_strict_local_rl_resume_state
    except ImportError as exc:  # pragma: no cover - configured remote runtime
        raise EvaluationError("strict local-RL resume-state verifier is unavailable") from exc
    try:
        state = load_strict_local_rl_resume_state(checkpoint)
    except Exception as exc:
        raise EvaluationError(f"step-{target.step} checkpoint fails strict resumability validation: {exc}") from exc
    if state.checkpoint_dir != checkpoint or state.global_step != target.step or state.optimizer_step != target.step:
        raise EvaluationError(
            f"step-{target.step} checkpoint has inconsistent final state: "
            f"global={state.global_step}, optimizer={state.optimizer_step}"
        )
    required = {
        "adapter_config": "adapter_config.json",
        "adapter_model": "adapter_model.safetensors",
        "optimizer": "optimizer.pt",
        "manifest": "manifest.json",
        "replicated_training_manifest": "replicated_training_manifest.json",
        "replicated_training_rng": "replicated_training_rng.pt",
    }
    files = {name: _identity(checkpoint / filename, label=f"step-{target.step} checkpoint {name}") for name, filename in required.items()}
    manifest = _read_json(checkpoint / "manifest.json", label="checkpoint manifest")
    adapter = _read_json(checkpoint / "adapter_config.json", label="checkpoint adapter configuration")
    loop = manifest.get("loop_state")
    if (
        manifest.get("backend") != "local"
        or manifest.get("kind") != "both"
        or manifest.get("model") != str(MODEL_SNAPSHOT)
        or not isinstance(loop, Mapping)
        or loop.get("global_step") != target.step
        or loop.get("optimizer_step") != target.step
        or loop.get("step") != target.step
        or loop.get("accumulated_grads") != 0
        or loop.get("final") is not True
        or adapter.get("base_model_name_or_path") != str(MODEL_SNAPSHOT)
    ):
        raise EvaluationError("checkpoint manifest/adapter does not bind the exact sealed local Qwen3.5 boundary")
    replicated = _read_json(checkpoint / "replicated_training_manifest.json", label="replicated-training manifest")
    if (
        replicated.get("schema") != 1
        or replicated.get("checkpoint_kind") != "both"
        or replicated.get("world_size") != 4
        or replicated.get("train_logical_indices") != [0, 1, 2, 3]
        or replicated.get("process_group_backend") != "nccl"
        or replicated.get("device_type") != "cuda"
        or replicated.get("rng_state_file") != "replicated_training_rng.pt"
        or not _is_sha256(replicated.get("state_hash"))
    ):
        raise EvaluationError("checkpoint lacks the strict four-rank replicated optimizer/RNG custody bundle")
    return {
        "path": str(checkpoint),
        "checkpoint_relative": target.checkpoint_relative,
        "run_prefix": target.run_prefix,
        "run_name": target.run_name,
        "segment_index": target.segment_index,
        "global_step": state.global_step,
        "optimizer_step": state.optimizer_step,
        "completed_epochs": state.completed_epochs,
        "kind": "both",
        "full_resumability_required": True,
        "files": files,
        # Compatibility validation accepts this flattened identity and checks
        # it against the translation manifest/attestation later.
        "adapter_model_sha256": files["adapter_model"]["sha256"],
        "adapter_config_sha256": files["adapter_config"]["sha256"],
        "manifest": {"backend": manifest["backend"], "kind": manifest["kind"], "loop_state": dict(loop)},
        "replicated_training": {
            "schema": replicated.get("schema"),
            "checkpoint_kind": replicated["checkpoint_kind"],
            "world_size": replicated["world_size"],
            "train_logical_indices": replicated["train_logical_indices"],
            "process_group_backend": replicated["process_group_backend"],
            "device_type": replicated["device_type"],
            "rng_state_file": replicated["rng_state_file"],
            "state_hash": replicated["state_hash"],
        },
    }


def validate_approved_target(repository: str | Path, *, step: int) -> dict[str, Any]:
    """Replay the target's checkpoint, sealed receipts, and ``continue`` decision."""

    target = _target(step)
    root = _resolve_unlinked(repository, label="training repository")
    if not root.is_dir():
        raise EvaluationError(f"training repository must be a regular directory: {root}")
    checkpoint = _strict_checkpoint_identity(root, target)
    checkpoint_receipt = _identity(
        _under_root(root / target.checkpoint_receipt_relative, root, label="checkpoint receipt"),
        label=f"step-{step} sealed checkpoint receipt",
    )
    completion_receipt = _identity(
        _under_root(root / target.completion_receipt_relative, root, label="completion receipt"),
        label=f"step-{step} sealed completion receipt",
    )
    decisions = _under_root(root / target.decisions_relative, root, label="decision directory")
    if decisions.is_symlink() or not decisions.is_dir():
        raise EvaluationError(f"step-{step} decision directory is absent or linked: {decisions}")
    candidates = sorted(decisions.glob(f"checkpoint-window-decision-s{target.segment_index:03d}-*.json"))
    if len(candidates) != 1:
        raise EvaluationError(f"step-{step} requires exactly one sealed decision receipt; found {len(candidates)}")
    decision_path = candidates[0]
    if decision_path.is_symlink() or not decision_path.is_file():
        raise EvaluationError(f"step-{step} decision receipt must be a regular file: {decision_path}")
    try:
        from experiments.rmct_convergence import controller

        decision = controller.verify_decision_receipt(decision_path)
        controller.require_continue(decision_path)
    except Exception as exc:
        raise EvaluationError(f"step-{step} decision receipt is not replayable sealed continuation evidence: {exc}") from exc
    expected_target = {
        "segment_index": target.segment_index,
        "optimizer_step_start": target.step - 15,
        "optimizer_step_end": target.step,
        "optimizer_steps": 16,
    }
    if decision.get("target") != expected_target or decision.get("decision") != "continue":
        raise EvaluationError(f"step-{step} decision receipt does not bind the exact approved continuation window")
    # verify_decision_receipt replays and authenticates these records.  Bind
    # their absolute identities too so the evaluation receipt remains portable.
    decision_checkpoint = decision.get("checkpoint_receipt")
    decision_completion = decision.get("completion_receipt")
    if not isinstance(decision_checkpoint, Mapping) or not isinstance(decision_completion, Mapping):
        raise EvaluationError("replayed continuation decision lacks sealed checkpoint/completion receipt identities")
    if decision_checkpoint.get("path") != checkpoint_receipt["path"] or decision_completion.get("path") != completion_receipt["path"]:
        raise EvaluationError("continuation decision receipt does not bind this target's sealed checkpoint/completion receipts")
    return {
        "target": {
            "step": target.step,
            "condition": target.condition,
            "run_prefix": target.run_prefix,
            "run_name": target.run_name,
            "segment_index": target.segment_index,
        },
        "checkpoint": checkpoint,
        "checkpoint_receipt": checkpoint_receipt,
        "completion_receipt": completion_receipt,
        "decision_receipt": _identity(decision_path, label=f"step-{step} sealed continue decision"),
    }


def _validate_two_bias_substrate(manifest: Path) -> dict[str, Any]:
    """Bind the frozen 3-clean/18-biased matrix and two-bias science labels."""

    try:
        from experiments.rmct_two_bias_eval import contract
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - deployment environment only
        raise EvaluationError("two-bias Stage-2 contract/task factory is unavailable") from exc
    try:
        substrate = contract.validate_stage2_substrate(manifest)
        specs = list(ood_task_specs(manifest))
        _validate_r002_sampling_matrix(specs)
    except Exception as exc:
        raise EvaluationError(f"two-bias Stage-2 substrate validation failed: {exc}") from exc
    if (
        len(specs) != TASK_COUNT
        or sum(spec.kind == "unbiased" for spec in specs) != len(CLEAN_TASK_INDICES)
        or sum(spec.kind == "biased" for spec in specs) != len(BIASED_TASK_INDICES)
        or {spec.bias_type for spec in specs if spec.kind == "biased"} != set(ALL_BIASES)
        or tuple(getattr(contract, "SEEN_BIASES", ())) != SEEN_BIASES
        or tuple(getattr(contract, "HELD_OUT_BIASES", ())) != HELD_OUT_BIASES
    ):
        raise EvaluationError("frozen Stage-2 substrate is not the exact two-seen/four-held-out 21-cell matrix")
    return {
        "path": str(manifest.resolve()),
        "sha256": _sha256_file(manifest),
        "task_count": TASK_COUNT,
        "clean_task_count": len(CLEAN_TASK_INDICES),
        "biased_task_count": len(BIASED_TASK_INDICES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "contract": dict(substrate),
    }


def _materialize_deployment_manifest(source: str | Path, *, artifact_root: str | Path, output: Path) -> dict[str, Any]:
    """Use the existing portable-manifest bridge without changing JSONL data."""

    try:
        from experiments.rmct_two_bias_eval import deployment

        record = deployment.materialize_deployment_manifest(source, artifact_root, output)
    except Exception as exc:
        raise EvaluationError(f"could not materialize the verified Stage-2 deployment manifest: {exc}") from exc
    if not isinstance(record, Mapping) or Path(str(record.get("manifest_path", ""))).resolve() != output.resolve():
        raise EvaluationError("deployment-manifest bridge did not bind the requested immutable output")
    result = dict(record)
    # Whether the byte-identical immutable file was written or resumed is not
    # provenance and must never make a later launch contract differ.
    result.pop("status", None)
    return result


def _replay_deployment_manifest(
    *,
    paths: Paths,
    source_manifest: Path,
    source_identity: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Reopen the deployed Stage-2 tree without materializing or rewriting it."""

    try:
        from experiments.rmct_two_bias_eval import deployment

        provenance = deployment.validate_deployment_manifest(paths.deployment_manifest)
    except Exception as exc:
        raise EvaluationError(f"could not replay the deployed Stage-2 manifest: {exc}") from exc
    if not isinstance(provenance, Mapping):
        raise EvaluationError("deployed Stage-2 manifest is missing immutable deployment provenance")
    if source_identity.get("sha256") != getattr(deployment, "CANONICAL_SOURCE_MANIFEST_SHA256", None):
        raise EvaluationError("source Stage-2 manifest no longer matches the pinned canonical identity")
    expected_source = {"path": str(source_manifest), "sha256": source_identity["sha256"]}
    if provenance.get("source_manifest") != expected_source or provenance.get("artifact_root") != str(artifact_root):
        raise EvaluationError("deployed Stage-2 manifest no longer binds the launch source/artifact roots")
    copied = provenance.get("copied_artifacts")
    if not isinstance(copied, Mapping):
        raise EvaluationError("deployed Stage-2 manifest has no copied-artifact custody")
    deployed_identity = _identity(paths.deployment_manifest, label="deployed Stage-2 manifest")
    return {
        "schema": deployment.DEPLOYMENT_SCHEMA,
        "manifest_path": str(paths.deployment_manifest),
        "manifest_sha256": deployed_identity["sha256"],
        "source_manifest_sha256": source_identity["sha256"],
        "artifact_root": str(artifact_root),
        "artifacts": dict(copied),
    }


def _required_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EvaluationError(f"{label} must be a lower-case SHA-256")
    return value


def _parity_data(parity_data: str | Path, parity_manifest: str | Path) -> dict[str, Any]:
    """Hash-bind the same frozen Stage-1 transport diagnostic as r005."""

    manifest_path = _resolve_unlinked(parity_manifest, label="canonical Stage-1 parity manifest")
    manifest_identity = _identity(manifest_path, label="canonical Stage-1 parity manifest")
    if manifest_identity["sha256"] != STAGE1_PARITY_MANIFEST_SHA256:
        raise EvaluationError("Stage-1 parity manifest differs from the frozen no-CoT transport artifact")
    document = _read_json(manifest_path, label="canonical Stage-1 parity manifest")
    if (
        document.get("schema_version") != STAGE1_PARITY_MANIFEST_SCHEMA_VERSION
        or document.get("kind") != STAGE1_PARITY_MANIFEST_KIND
        or not isinstance(document.get("splits"), Mapping)
        or not isinstance(document["splits"].get("train_eval"), Mapping)
    ):
        raise EvaluationError("Stage-1 parity manifest has the wrong frozen schema")
    entry = document["splits"]["train_eval"]
    declared_path = entry.get("path")
    expected_sha256 = _required_sha256(entry.get("content_sha256"), label="Stage-1 parity train_eval hash")
    expected_size = entry.get("byte_count")
    if (
        not isinstance(declared_path, str)
        or Path(declared_path).name != STAGE1_PARITY_TRAIN_EVAL_FILENAME
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or entry.get("row_count") != STAGE1_PARITY_TRAIN_EVAL_ROWS
    ):
        raise EvaluationError("Stage-1 parity manifest has an invalid train_eval identity")
    data = _resolve_unlinked(parity_data, label="staged Stage-1 parity data")
    if data.name != STAGE1_PARITY_TRAIN_EVAL_FILENAME:
        raise EvaluationError("staged parity data has an unexpected filename")
    identity = _identity(data, label="staged Stage-1 parity data")
    if identity["sha256"] != expected_sha256 or identity["size_bytes"] != expected_size:
        raise EvaluationError("staged parity data differs from the frozen Stage-1 manifest entry")
    return {
        **identity,
        "canonical_manifest": manifest_identity,
        "manifest_train_eval": {
            "declared_path": declared_path,
            "filename": STAGE1_PARITY_TRAIN_EVAL_FILENAME,
            "content_sha256": expected_sha256,
            "byte_count": expected_size,
            "row_count": STAGE1_PARITY_TRAIN_EVAL_ROWS,
        },
    }


def _parse_four_gpu_tokens(value: str) -> tuple[str, str, str, str]:
    """Parse the four *local* tokens visible only to the parity srun step."""

    tokens = tuple(value.split(","))
    if (
        len(tokens) != 4
        or any(
            not token
            or token != token.strip()
            or any(character.isspace() for character in token)
            or "," in token
            or token in {"-1", "NoDevFiles"}
            for token in tokens
        )
        or len(set(tokens)) != 4
    ):
        raise EvaluationError("parity requires exactly four distinct Slurm-visible CUDA tokens")
    return tokens  # type: ignore[return-value]


def _sampler_runtime(tokens: Sequence[str]) -> dict[str, Any]:
    values = tuple(tokens)
    if len(values) != 4 or len(set(values)) != 4:
        raise EvaluationError("vLLM parity runtime requires exactly four local device tokens")
    return {
        "vllm_version": VLLM_VERSION,
        "environment": dict(VLLM_SAMPLER_ENVIRONMENT),
        "implementation": "pytorch_native",
        "parity_gdn_prefill_backend": "triton",
        "parity_top_token_count": PARITY_TOP_TOKEN_COUNT,
        "parity_max_logprobs": PARITY_MAX_LOGPROBS,
        "parity_logprobs_mode": PARITY_LOGPROBS_MODE,
        "parity_score_transport": PARITY_SCORE_TRANSPORT,
        "parity_sampling": dict(PARITY_SAMPLING),
        "persistent_gdn_prefill_backend": "triton",
        "isolate_vllm_variants": True,
        "parallel_isolated_vllm_variants": True,
        "enforce_eager": True,
        # These are deliberately local to the one four-GPU parity step.  They
        # are never used as campaign-wide GPU identifiers; a later worker on
        # another node may quite legitimately see CUDA token "0" as well.
        "vllm_device_tokens": list(values),
        "vllm_device_count": 4,
        "one_adapter_per_server": True,
        "max_loras_per_server": 1,
        "parity_result_variants": list(PARITY_VARIANTS),
    }


def _launch_runtime_contract(sampler: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete immutable runtime policy for one parity token plan."""

    return {
        "profile": "vllm",
        "sampler": dict(sampler),
        "model_args": dict(VLLM_MODEL_ARGS),
        "generation_config": dict(GENERATION_CONFIG),
        "hf_fallback_permitted": False,
        "requires_peft_translation_and_strict_parity": True,
        "persistent_vllm_server": True,
    }


def _launch_matrix_contract() -> dict[str, Any]:
    """Return the exact 21-cell matrix policy, independent of checkpoint."""

    return {
        "task_factory": TASK_FACTORY,
        "task_count": TASK_COUNT,
        "sampling": _sampling_contract(),
        "clean_task_indices": list(CLEAN_TASK_INDICES),
        "biased_task_indices": list(BIASED_TASK_INDICES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "clean_and_biased_concurrent": True,
        "switch_scorer_shared_canonical_raw_root": True,
        "switch_scorer_waits_for_matching_clean_log": True,
        "switch_scorer_wait_timeout_seconds": 3600,
    }


def _launch_outputs_contract(paths: Paths) -> dict[str, str]:
    return {
        "root": str(paths.root),
        "raw": str(paths.raw),
        "attempts": str(paths.attempts),
        "receipts": str(paths.receipts),
        "runtime": str(paths.runtime),
        "completion": str(paths.completion),
    }


def _launch_policy_contract() -> dict[str, bool]:
    return {
        "self_submits": False,
        "chains_successors": False,
        "cancels_jobs": False,
        "r005_output_reused": False,
        "overwrite_differing_logs": False,
        "partial_attempts_preserved": True,
    }


def build_launch_contract(
    *,
    repository: str | Path,
    step: int,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    parity_data: str | Path,
    parity_manifest: str | Path,
    output_root: str | Path,
    parity_gpu_tokens: Sequence[str],
) -> dict[str, Any]:
    """Build one immutable, target-specific launch contract before GPU work."""

    target = _target(step)
    paths = _paths(output_root, target=target)
    source_manifest = _resolve_unlinked(source_stage2_manifest, label="source Stage-2 manifest")
    staged_artifacts = _resolve_unlinked(stage2_artifact_root, label="Stage-2 artifact root")
    if not source_manifest.is_file() or not staged_artifacts.is_dir():
        raise EvaluationError("source Stage-2 manifest/artifact root must be regular deployment inputs")
    checkpoint = validate_approved_target(repository, step=step)
    snapshot = _validate_model_snapshot()
    deployment = _materialize_deployment_manifest(
        source_manifest,
        artifact_root=staged_artifacts,
        output=paths.deployment_manifest,
    )
    substrate = _validate_two_bias_substrate(paths.deployment_manifest)
    sampler = _sampler_runtime(parity_gpu_tokens)
    return {
        "schema": LAUNCH_SCHEMA,
        "condition": target.condition,
        "target": checkpoint["target"],
        "repository": str(Path(repository).expanduser().resolve()),
        "checkpoint_custody": checkpoint,
        "model_snapshot": snapshot,
        "source_stage2_manifest": _identity(source_manifest, label="source Stage-2 manifest"),
        "stage2_artifact_root": str(staged_artifacts),
        "deployment_manifest": {"path": str(paths.deployment_manifest), "provenance": deployment, "substrate": substrate},
        "parity_data": _parity_data(parity_data, parity_manifest),
        "critical_sources": _critical_source_identities(),
        "runtime": _launch_runtime_contract(sampler),
        "matrix": _launch_matrix_contract(),
        "outputs": _launch_outputs_contract(paths),
        "policy": _launch_policy_contract(),
    }


def prepare_launch_contract(**kwargs: Any) -> tuple[dict[str, Any], Paths, str]:
    target = _target(int(kwargs["step"]))
    paths = _paths(kwargs["output_root"], target=target)
    if paths.root.exists() and not paths.contract.exists() and any(paths.root.iterdir()):
        raise FileExistsError(f"refusing to seed a launch contract in a non-empty target root: {paths.root}")
    contract = build_launch_contract(**kwargs)
    _write_immutable_json(paths.contract, contract, label="target evaluation launch contract")
    return contract, paths, _sha256_file(paths.contract)


def _raw_checkpoint_identity(custody: Mapping[str, Any]) -> dict[str, str]:
    checkpoint = custody.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise EvaluationError("launch contract has no strict checkpoint custody")
    path = checkpoint.get("path")
    model_sha = checkpoint.get("adapter_model_sha256")
    config_sha = checkpoint.get("adapter_config_sha256")
    if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(model_sha, str) or not isinstance(config_sha, str):
        raise EvaluationError("launch contract checkpoint has incomplete adapter identity")
    return {"path": str(Path(path).resolve()), "adapter_model_sha256": model_sha, "adapter_config_sha256": config_sha}


def _next_attempt(root: Path, *, label: str) -> Path:
    """Reserve a new preserved attempt directory without deleting an old one."""

    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"attempt root must be a regular directory: {root}")
    for index in range(1, 10_000):
        candidate = root / f"{label}-{index:04d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            if candidate.is_symlink() or not candidate.is_dir():
                raise EvaluationError(f"attempt name is occupied by a non-directory: {candidate}")
            continue
        return candidate
    raise EvaluationError(f"exhausted preserved attempt slots under {root}")


def _reserve_uncreated_output_destination(root: Path, *, label: str) -> Path:
    """Reserve an *absent* converter/parity destination without precreating it.

    The PEFT compatibility converter and the strict parity utility both
    create their own output directories and deliberately reject an existing
    destination.  A normal ``mkdir``-based attempt helper therefore cannot be
    used for them.  This companion helper atomically creates a durable marker
    file instead, leaving the output path absent for the tool to create.  A
    stale marker is evidence of an interrupted reservation and is never
    removed or reused; a later invocation takes the next suffix.
    """

    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"output reservation root must be a regular directory: {root}")
    for index in range(1, 10_000):
        candidate = root / f"{label}-{index:04d}"
        marker = root / f".{label}-{index:04d}.reservation.json"
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or not candidate.is_dir():
                raise EvaluationError(f"output destination name is occupied by a non-directory: {candidate}")
            continue
        if marker.is_symlink():
            raise EvaluationError(f"output reservation marker must not be a symlink: {marker}")
        try:
            descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        try:
            payload = _canonical_json(
                {
                    "schema": "rmct-checkpoint-two-bias-16gpu-output-reservation-v1",
                    "label": label,
                    "destination": str(candidate),
                }
            )
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            # The marker remains durable evidence of the failed reservation;
            # never erase it in a recovery path.
            raise
        # A non-cooperating writer could have populated the leaf after the
        # marker was won.  Preserve both artefacts and use a fresh suffix.
        if candidate.exists() or candidate.is_symlink():
            continue
        return candidate
    raise EvaluationError(f"exhausted uncreated output destinations under {root}")


def _validate_translation_only(raw: Mapping[str, str], adapter: Path) -> None:
    """Check translation evidence without treating it as parity-attested yet."""

    manifest_path = adapter / "compatibility-manifest.json"
    model_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    for path, label in ((manifest_path, "compatibility manifest"), (model_path, "compatibility weights"), (config_path, "compatibility config")):
        _identity(path, label=label)
    document = _read_json(manifest_path, label="compatibility manifest")
    source = document.get("source")
    destination = document.get("destination")
    translation = document.get("translation")
    if (
        document.get("schema") != "qwen35-vllm-compat-adapter-v1"
        or not isinstance(source, Mapping)
        or not isinstance(destination, Mapping)
        or not isinstance(translation, Mapping)
        or source.get("path") != raw["path"]
        or source.get("adapter_model_sha256") != raw["adapter_model_sha256"]
        or destination.get("path") != str(adapter)
        or destination.get("adapter_model_sha256") != _sha256_file(model_path)
        or translation.get("source_prefix") != "base_model.model.model.layers."
        or translation.get("destination_prefix") != "base_model.model.model.language_model.layers."
    ):
        raise EvaluationError("compatibility adapter does not prove the exact raw PEFT-to-vLLM translation")


def _validate_parity_environment(sampler: Mapping[str, Any]) -> tuple[str, str, str, str]:
    if sampler.get("environment") != VLLM_SAMPLER_ENVIRONMENT:
        raise EvaluationError("launch contract has an unexpected vLLM sampler environment")
    for name, expected in VLLM_SAMPLER_ENVIRONMENT.items():
        if os.environ.get(name) != expected:
            raise EvaluationError(f"{name} must be exactly {expected!r} for the frozen vLLM sampler")
    try:
        import vllm
    except ImportError as exc:  # pragma: no cover - configured remote runtime
        raise EvaluationError("vLLM is unavailable for strict parity") from exc
    if getattr(vllm, "__version__", None) != VLLM_VERSION:
        raise EvaluationError(f"strict parity requires vLLM {VLLM_VERSION}, got {getattr(vllm, '__version__', None)!r}")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        raise EvaluationError("strict parity requires CUDA_VISIBLE_DEVICES from its dedicated four-GPU Slurm step")
    actual = _parse_four_gpu_tokens(inherited)
    expected = tuple(sampler.get("vllm_device_tokens", ()))
    if actual != expected:
        raise EvaluationError("strict parity refuses to remap its Slurm-visible four-GPU token order")
    return actual


def _parity_command(*, python: str, adapter: Path, raw: Mapping[str, str], data: str, attempt: Path, sampler: Mapping[str, Any]) -> list[str]:
    tokens = tuple(sampler.get("vllm_device_tokens", ()))
    if len(tokens) != 4 or len(set(tokens)) != 4:
        raise EvaluationError("strict parity has no four-token server plan")
    return [
        python,
        "-m",
        "experiments.act_repair_gate.runtime_parity",
        "--model",
        str(MODEL_SNAPSHOT),
        "--adapter",
        str(adapter),
        "--hf-adapter",
        raw["path"],
        "--data",
        data,
        "--output-dir",
        str(attempt),
        "--enforce-eager",
        "--isolate-vllm-variants",
        "--parallel-isolated-vllm-variants",
        "--vllm-device-tokens",
        *tokens,
        "--top-token-count",
        str(PARITY_TOP_TOKEN_COUNT),
        "--max-logprobs",
        str(PARITY_MAX_LOGPROBS),
        "--gdn-prefill-backend",
        "triton",
    ]


def _validate_strict_parity_report(report: Path, *, adapter: Path, sampler: Mapping[str, Any]) -> dict[str, Any]:
    """Require the same four-server processed-logprob protocol as r005."""

    document = _read_json(report, label="runtime parity report")
    backends = document.get("backends")
    vllm = backends.get("vllm") if isinstance(backends, Mapping) else None
    if (
        not isinstance(vllm, Mapping)
        or vllm.get("gdn_prefill_backend") != sampler.get("parity_gdn_prefill_backend")
        or vllm.get("max_logprobs") != PARITY_MAX_LOGPROBS
        or vllm.get("logprobs_mode") != PARITY_LOGPROBS_MODE
        or vllm.get("score_transport") != PARITY_SCORE_TRANSPORT
        or vllm.get("parity_sampling") != PARITY_SAMPLING
        or vllm.get("isolate_vllm_variants") is not True
        or vllm.get("parallel_isolated_vllm_variants") is not True
        or vllm.get("enforce_eager") is not True
    ):
        raise EvaluationError("parity report does not bind the frozen eager four-server vLLM protocol")
    protocol = document.get("token_protocol")
    plan = vllm.get("parallel_isolated_server_plan")
    expected_tokens = sampler.get("vllm_device_tokens")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("requested_result_variants") != list(PARITY_VARIANTS)
        or protocol.get("top_token_count") != PARITY_TOP_TOKEN_COUNT
        or protocol.get("vllm_score_transport") != PARITY_SCORE_TRANSPORT
        or protocol.get("vllm_allowed_token_ids") != "requested_token_ids"
        or protocol.get("vllm_response_token_ids") != "exactly_requested_token_ids"
        or not isinstance(plan, Mapping)
        or set(plan) != set(PARITY_VARIANTS)
        or not isinstance(expected_tokens, list)
    ):
        raise EvaluationError("parity report does not bind the exact allowed-token four-server topology")
    actual_tokens: list[str] = []
    for variant in PARITY_VARIANTS:
        server = plan.get(variant)
        if (
            not isinstance(server, Mapping)
            or server.get("max_loras") != 1
            or server.get("max_logprobs") != PARITY_MAX_LOGPROBS
            or server.get("logprobs_mode") != PARITY_LOGPROBS_MODE
            or not isinstance(server.get("device_token"), str)
        ):
            raise EvaluationError(f"parity report server plan is malformed for {variant!r}")
        actual_tokens.append(server["device_token"])
    if actual_tokens != expected_tokens:
        raise EvaluationError("parity report changed the Slurm-local four-GPU token order")
    try:
        from experiments.rmct_two_bias_eval import contract

        checked = contract.validate_r005_parallel_parity_report(report, compatibility_adapter=adapter)
    except Exception as exc:
        raise EvaluationError(f"parity report fails the strict shared Qwen3.5 validator: {exc}") from exc
    return dict(checked)


def _compatibility_identity(adapter: Path, *, raw: Mapping[str, str], sampler: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from experiments.rmct_two_bias_eval import contract

        identity = contract.vllm_compatibility_identity(adapter, raw_checkpoint=raw)
    except Exception as exc:
        raise EvaluationError(f"compatibility adapter lacks strict translation/parity custody: {exc}") from exc
    reports = identity.get("parity_reports") if isinstance(identity, Mapping) else None
    if not isinstance(reports, list) or not reports or not isinstance(reports[0], Mapping):
        raise EvaluationError("compatibility adapter has no authenticated primary parity report")
    path = reports[0].get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise EvaluationError("compatibility adapter has an invalid primary parity report path")
    _validate_strict_parity_report(Path(path), adapter=adapter, sampler=sampler)
    return dict(identity)


def _validated_launch_runtime_policy(launch: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Return the sole frozen serving policy admitted by a launch contract.

    The parity-device names are deliberately allocation-local, but every other
    field is immutable.  Reconstructing the sampler object here prevents a
    later worker from accepting a hand-edited launch/runtime receipt with a
    superficially plausible vLLM configuration.
    """

    runtime = launch.get("runtime")
    sampler = runtime.get("sampler") if isinstance(runtime, Mapping) else None
    if not isinstance(sampler, Mapping):
        raise EvaluationError("launch contract lacks the frozen vLLM sampler policy")
    token_values = sampler.get("vllm_device_tokens")
    if not isinstance(token_values, list) or any(not isinstance(value, str) for value in token_values):
        raise EvaluationError("launch contract has invalid local parity-device tokens")
    tokens = _parse_four_gpu_tokens(",".join(token_values))
    expected_sampler = _sampler_runtime(tokens)
    expected_runtime = _launch_runtime_contract(expected_sampler)
    if dict(runtime) != expected_runtime:
        raise EvaluationError("launch contract has a changed vLLM serving/parity policy")
    custody = launch.get("checkpoint_custody")
    if not isinstance(custody, Mapping):
        raise EvaluationError("launch contract lacks checkpoint custody")
    return _raw_checkpoint_identity(custody), expected_sampler


def _runtime_receipt_document(
    *,
    raw: Mapping[str, str],
    sampler: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    parity_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact write-once runtime receipt from replayed evidence."""

    return {
        "schema": RUNTIME_SCHEMA,
        "profile": "vllm",
        "raw_checkpoint": dict(raw),
        "compatibility_adapter": dict(compatibility),
        "model_snapshot": str(MODEL_SNAPSHOT),
        "sampler": dict(sampler),
        "model_args": dict(VLLM_MODEL_ARGS),
        "generation_config": dict(GENERATION_CONFIG),
        "parity_attestation": dict(parity_attestation),
    }


def ensure_attested_vllm_runtime(*, launch: Mapping[str, Any], paths: Paths, python: str) -> dict[str, Any]:
    """Create/replay one isolated translation and strict parity attestation."""

    raw, sampler = _validated_launch_runtime_policy(launch)
    _validate_parity_environment(sampler)
    compatibility_root = paths.runtime / "compatibility"
    parity_root = paths.runtime / "parity-attempts"
    compatibility_root.mkdir(parents=True, exist_ok=True)
    if compatibility_root.is_symlink() or not compatibility_root.is_dir():
        raise EvaluationError("compatibility root must be a regular directory")

    selected: Path | None = None
    unfinished: Path | None = None
    for candidate in sorted(item for item in compatibility_root.iterdir() if item.is_dir() and not item.is_symlink()):
        translation_files = (
            candidate / "adapter_config.json",
            candidate / "adapter_model.safetensors",
            candidate / "compatibility-manifest.json",
        )
        if any(path.is_symlink() or not path.is_file() for path in translation_files):
            continue  # an interrupted conversion remains evidence, but is not reusable
        _validate_translation_only(raw, candidate)
        attestation = candidate / "vllm-parity-attestation.json"
        if attestation.exists():
            try:
                _compatibility_identity(candidate, raw=raw, sampler=sampler)
            except EvaluationError:
                continue  # preserved weaker/invalid proof cannot be upgraded in place
            selected = candidate
            break
        if unfinished is None:
            unfinished = candidate
    if selected is None:
        selected = unfinished
        if selected is None:
            # ``make_compat_adapter`` owns destination creation and refuses
            # an existing directory.  Reserve only a marker, not this leaf.
            selected = _reserve_uncreated_output_destination(compatibility_root, label="adapter")
            try:
                from experiments.act_repair_gate.vllm_compat_adapter import make_compat_adapter

                make_compat_adapter(raw["path"], selected)
            except Exception as exc:
                raise EvaluationError(f"could not create the immutable PEFT-to-vLLM compatibility adapter: {exc}") from exc
            _validate_translation_only(raw, selected)

    attestation = selected / "vllm-parity-attestation.json"
    if not attestation.exists():
        report: Path | None = None
        if parity_root.exists():
            if parity_root.is_symlink() or not parity_root.is_dir():
                raise EvaluationError("parity attempt root must be a regular directory")
            for candidate in sorted(parity_root.iterdir()):
                possible = candidate / "report.json"
                if candidate.is_symlink() or not candidate.is_dir() or not possible.is_file() or possible.is_symlink():
                    continue
                try:
                    document = _read_json(possible, label="prior parity report")
                    report_adapter = document.get("adapter")
                    if isinstance(report_adapter, Mapping) and report_adapter.get("path") == str(selected):
                        _validate_strict_parity_report(possible, adapter=selected, sampler=sampler)
                        report = possible
                        break
                except EvaluationError:
                    continue
        if report is None:
            # ``runtime_parity`` has the same absent-output contract as the
            # converter, so it must not receive a precreated attempt dir.
            attempt = _reserve_uncreated_output_destination(parity_root, label="parity")
            command = _parity_command(
                python=python,
                adapter=selected,
                raw=raw,
                data=str(launch["parity_data"]["path"]),
                attempt=attempt,
                sampler=sampler,
            )
            environment = os.environ.copy()
            environment.update(VLLM_SAMPLER_ENVIRONMENT)
            for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "CTM_PERSISTENT_VLLM_SERVER_METADATA"):
                environment.pop(name, None)
            result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
            if result.returncode:
                raise EvaluationError(f"HF/vLLM parity failed with exit {result.returncode}; preserved attempt: {attempt}")
            report = attempt / "report.json"
            _identity(report, label="runtime parity report")
            _validate_strict_parity_report(report, adapter=selected, sampler=sampler)
        try:
            from experiments.act_repair_gate.vllm_compat_adapter import attest_compat_adapter

            attest_compat_adapter(selected, report)
        except Exception as exc:
            raise EvaluationError(f"could not atomically attest the strict parity proof: {exc}") from exc

    compatibility = _compatibility_identity(selected, raw=raw, sampler=sampler)
    attestation_record = _identity(selected / "vllm-parity-attestation.json", label="parity attestation")
    record = _runtime_receipt_document(
        raw=raw,
        sampler=sampler,
        compatibility=compatibility,
        parity_attestation=attestation_record,
    )
    _write_immutable_json(paths.runtime_receipt, record, label="target vLLM runtime receipt")
    return record


def _build_evaluation_receipt(*, launch: Mapping[str, Any], runtime: Mapping[str, Any], paths: Paths) -> dict[str, Any]:
    """Build a generic early-checkpoint receipt; it does not masquerade as r005."""

    condition = launch.get("condition")
    target = launch.get("target")
    deployment = launch.get("deployment_manifest")
    if not isinstance(condition, str) or not isinstance(target, Mapping) or not isinstance(deployment, Mapping):
        raise EvaluationError("launch contract has incomplete target/substrate identity")
    compatibility = runtime.get("compatibility_adapter")
    if not isinstance(compatibility, Mapping):
        raise EvaluationError("runtime receipt lacks a strict compatibility adapter")
    return {
        "schema": EVALUATION_RECEIPT_SCHEMA,
        "condition": condition,
        "target": dict(target),
        "launch_contract": _identity(paths.contract, label="launch contract"),
        "checkpoint_custody": launch["checkpoint_custody"],
        "stage2_substrate": deployment,
        "raw_generation": {
            "raw_log_root": str(paths.raw),
            **_sampling_contract(),
        },
        "runtime": {
            "profile": "vllm",
            "base_model": str(MODEL_SNAPSHOT),
            "raw_checkpoint": runtime["raw_checkpoint"],
            "checkpoint": compatibility["path"],
            "compatibility_adapter": compatibility,
            "sampler": runtime["sampler"],
            "model_args": runtime["model_args"],
            "generation_config": runtime["generation_config"],
        },
        "science": {
            "all_biases": list(ALL_BIASES),
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "include_bias_acknowledged": False,
            "grader_model": None,
        },
    }


def _load_runtime_receipt(*, paths: Paths, launch: Mapping[str, Any]) -> dict[str, Any]:
    """Replay the translated-adapter and four-server parity evidence.

    This performs no new parity work and therefore needs no four-GPU Slurm
    visibility.  It reopens the compatibility adapter, its attestation, and
    its strict primary parity report, then accepts the persisted runtime
    receipt only if it is byte-for-byte represented by those replayed facts.
    """

    raw, sampler = _validated_launch_runtime_policy(launch)
    document = _read_json(paths.runtime_receipt, label="target vLLM runtime receipt")
    declared_compatibility = document.get("compatibility_adapter")
    if not isinstance(declared_compatibility, Mapping):
        raise EvaluationError("target vLLM runtime receipt lacks a compatibility-adapter identity")
    adapter_value = declared_compatibility.get("path")
    if not isinstance(adapter_value, str) or not Path(adapter_value).is_absolute():
        raise EvaluationError("target vLLM runtime receipt has an invalid compatibility-adapter path")
    adapter_declared = Path(adapter_value)
    if adapter_declared.is_symlink():
        raise EvaluationError("target vLLM runtime receipt names a linked compatibility adapter")
    adapter = adapter_declared.resolve()
    compatibility_root = (paths.runtime / "compatibility").resolve()
    _under_root(adapter, compatibility_root, label="runtime compatibility adapter")
    if adapter.is_symlink() or not adapter.is_dir():
        raise EvaluationError("target vLLM runtime receipt names a non-directory compatibility adapter")
    _validate_translation_only(raw, adapter)
    compatibility = _compatibility_identity(adapter, raw=raw, sampler=sampler)
    attestation = _identity(adapter / "vllm-parity-attestation.json", label="runtime parity attestation")
    expected = _runtime_receipt_document(
        raw=raw,
        sampler=sampler,
        compatibility=compatibility,
        parity_attestation=attestation,
    )
    if document != expected:
        raise EvaluationError(
            "target vLLM runtime receipt is not the exact replayed translated-adapter/parity evidence"
        )
    return expected


def _replay_launch_contract(*, paths: Paths, target: ApprovedTarget, launch: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct every write-once launch fact without creating any files.

    This is deliberately more than a schema check.  A worker must not trust a
    coordinated edit of launch, runtime, and evaluation receipts: it reopens
    the sealed checkpoint/decision evidence, deployed Stage-2 inputs, frozen
    parity corpus, and every source implementation boundary before serving an
    adapter or accepting a pre-existing raw log.
    """

    expected_target = {
        "step": target.step,
        "condition": target.condition,
        "run_prefix": target.run_prefix,
        "run_name": target.run_name,
        "segment_index": target.segment_index,
    }
    repository_value = launch.get("repository")
    if not isinstance(repository_value, str) or not Path(repository_value).is_absolute():
        raise EvaluationError("launch contract has an invalid training-repository path")
    repository = _resolve_unlinked(repository_value, label="launch training repository")
    if not repository.is_dir():
        raise EvaluationError("launch training repository is not a regular directory")
    checkpoint_custody = validate_approved_target(repository, step=target.step)
    if checkpoint_custody.get("target") != expected_target:
        raise EvaluationError("replayed checkpoint custody does not bind the approved target")

    source_record = launch.get("source_stage2_manifest")
    if not isinstance(source_record, Mapping):
        raise EvaluationError("launch contract lacks a source Stage-2 manifest identity")
    source_value = source_record.get("path")
    if not isinstance(source_value, str) or not Path(source_value).is_absolute():
        raise EvaluationError("launch contract has an invalid source Stage-2 manifest path")
    source_manifest = _resolve_unlinked(source_value, label="launch source Stage-2 manifest")
    if not source_manifest.is_file():
        raise EvaluationError("launch source Stage-2 manifest is not a regular file")
    source_identity = _identity(source_manifest, label="launch source Stage-2 manifest")

    artifact_value = launch.get("stage2_artifact_root")
    if not isinstance(artifact_value, str) or not Path(artifact_value).is_absolute():
        raise EvaluationError("launch contract has an invalid Stage-2 artifact-root path")
    artifact_root = _resolve_unlinked(artifact_value, label="launch Stage-2 artifact root")
    if not artifact_root.is_dir():
        raise EvaluationError("launch Stage-2 artifact root is not a regular directory")
    deployment = _replay_deployment_manifest(
        paths=paths,
        source_manifest=source_manifest,
        source_identity=source_identity,
        artifact_root=artifact_root,
    )
    substrate = _validate_two_bias_substrate(paths.deployment_manifest)

    parity_record = launch.get("parity_data")
    if not isinstance(parity_record, Mapping):
        raise EvaluationError("launch contract lacks Stage-1 parity data custody")
    parity_data_value = parity_record.get("path")
    manifest_record = parity_record.get("canonical_manifest")
    parity_manifest_value = manifest_record.get("path") if isinstance(manifest_record, Mapping) else None
    if (
        not isinstance(parity_data_value, str)
        or not Path(parity_data_value).is_absolute()
        or not isinstance(parity_manifest_value, str)
        or not Path(parity_manifest_value).is_absolute()
    ):
        raise EvaluationError("launch contract has invalid Stage-1 parity corpus/manifest paths")
    parity_data = _resolve_unlinked(parity_data_value, label="launch Stage-1 parity data")
    parity_manifest = _resolve_unlinked(parity_manifest_value, label="launch Stage-1 parity manifest")
    expected_parity = _parity_data(parity_data, parity_manifest)

    runtime_raw, sampler = _validated_launch_runtime_policy(launch)
    expected_raw = _raw_checkpoint_identity(checkpoint_custody)
    if runtime_raw != expected_raw:
        raise EvaluationError("launch runtime raw-checkpoint identity differs from replayed sealed checkpoint custody")

    expected = {
        "schema": LAUNCH_SCHEMA,
        "condition": target.condition,
        "target": expected_target,
        "repository": str(repository),
        "checkpoint_custody": checkpoint_custody,
        "model_snapshot": _validate_model_snapshot(),
        "source_stage2_manifest": source_identity,
        "stage2_artifact_root": str(artifact_root),
        "deployment_manifest": {
            "path": str(paths.deployment_manifest),
            "provenance": deployment,
            "substrate": substrate,
        },
        "parity_data": expected_parity,
        "critical_sources": _critical_source_identities(),
        "runtime": _launch_runtime_contract(sampler),
        "matrix": _launch_matrix_contract(),
        "outputs": _launch_outputs_contract(paths),
        "policy": _launch_policy_contract(),
    }
    if dict(launch) != expected:
        raise EvaluationError(
            "launch contract is not the exact replayed checkpoint/input/parity/source/runtime/matrix/output/policy custody"
        )
    return expected


def _load_launch(paths: Paths, *, target: ApprovedTarget) -> tuple[dict[str, Any], str]:
    launch = _read_json(paths.contract, label="target launch contract")
    replayed = _replay_launch_contract(paths=paths, target=target, launch=launch)
    return replayed, _sha256_file(paths.contract)


def _load_evaluation_receipt(
    paths: Paths,
    *,
    target: ApprovedTarget,
    launch: Mapping[str, Any],
    launch_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Replay every serving/custody binding before a worker may use it."""

    if _sha256_file(paths.contract) != launch_sha256:
        raise EvaluationError("launch-contract digest changed before evaluation receipt replay")
    receipt = _read_json(paths.evaluation_receipt, label="target evaluation receipt")
    runtime = _load_runtime_receipt(paths=paths, launch=launch)
    expected = _build_evaluation_receipt(launch=launch, runtime=runtime, paths=paths)
    if receipt != expected:
        raise EvaluationError(
            "evaluation receipt is not the exact replayed launch/runtime/substrate/checkpoint-custody binding"
        )
    # The explicit target parameter keeps this replay boundary impossible to
    # invoke for any arbitrary condition, even if a caller supplied a forged
    # in-memory launch dictionary.
    if receipt.get("target", {}).get("step") != target.step:
        raise EvaluationError("evaluation receipt target changed during strict replay")
    return expected, _sha256_file(paths.evaluation_receipt)


def prepare(
    *,
    repository: str | Path,
    step: int,
    source_stage2_manifest: str | Path,
    stage2_artifact_root: str | Path,
    parity_data: str | Path,
    parity_manifest: str | Path,
    output_root: str | Path,
    parity_gpus: str,
    python: str,
) -> dict[str, Any]:
    """Prepare one target using the four GPUs allocated solely for parity."""

    target = _target(step)
    tokens = _parse_four_gpu_tokens(parity_gpus)
    contract, paths, _ = prepare_launch_contract(
        repository=repository,
        step=step,
        source_stage2_manifest=source_stage2_manifest,
        stage2_artifact_root=stage2_artifact_root,
        parity_data=parity_data,
        parity_manifest=parity_manifest,
        output_root=output_root,
        parity_gpu_tokens=tokens,
    )
    runtime = ensure_attested_vllm_runtime(launch=contract, paths=paths, python=python)
    receipt = _build_evaluation_receipt(launch=contract, runtime=runtime, paths=paths)
    status = _write_immutable_json(paths.evaluation_receipt, receipt, label="target evaluation receipt")
    return {
        "step": target.step,
        "condition": target.condition,
        "contract": _identity(paths.contract, label="launch contract"),
        "runtime": _identity(paths.runtime_receipt, label="runtime receipt"),
        "evaluation_receipt": _identity(paths.evaluation_receipt, label="evaluation receipt"),
        "evaluation_receipt_status": status,
    }


def _task_receipt_path(paths: Paths, task_index: int) -> Path:
    return paths.receipts / f"task-{task_index:03d}.json"


def _inspect_success(path: Path, *, task_index: int) -> int:
    """Validate a successful task and return its exact r002 sample count."""

    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured remote runtime
        raise EvaluationError("Inspect AI is required to validate task receipts") from exc
    try:
        log = read_eval_log(str(path), header_only=True)
    except Exception as exc:
        raise EvaluationError(f"could not read Inspect EvalLog: {path}") from exc
    evaluation = getattr(log, "eval", None)
    metadata = getattr(evaluation, "metadata", {}) if evaluation is not None else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    if (
        getattr(log, "status", None) != "success"
        or metadata.get("task_indices") != [task_index]
        or metadata.get("task_count") != TASK_COUNT
    ):
        raise EvaluationError(f"EvalLog is not the exact successful Stage-2 task {task_index}: {path}")
    try:
        full_log = read_eval_log(str(path), header_only=False)
    except Exception as exc:
        raise EvaluationError(f"could not read full Inspect EvalLog: {path}") from exc
    samples = getattr(full_log, "samples", None)
    if not isinstance(samples, (list, tuple)):
        raise EvaluationError(f"EvalLog has no materialized sample sequence: {path}")
    sample_count = len(samples)
    expected = _sample_count_for_task(task_index)
    if sample_count != expected:
        raise EvaluationError(
            f"EvalLog has the wrong r002 sample count for task-{task_index}: "
            f"got={sample_count}, expected={expected}: {path}"
        )
    return sample_count


def _value_field(value: Any, field: str, default: Any = None) -> Any:
    return value.get(field, default) if isinstance(value, Mapping) else getattr(value, field, default)


def _full_question_ids_for_task(*, launch: Mapping[str, Any], task_index: int) -> tuple[str, ...]:
    """Rebuild the untruncated frozen task IDs before promotion writes custody."""

    deployment = launch.get("deployment_manifest")
    deployment_path = deployment.get("path") if isinstance(deployment, Mapping) else None
    if not isinstance(deployment_path, str):
        raise EvaluationError("launch contract has no deployed Stage-2 manifest for promotion validation")
    try:
        from experiments.stage2_ood_hle.tasks import ood_task_specs

        specs = list(ood_task_specs(deployment_path))
        _validate_r002_sampling_matrix(specs)
    except Exception as exc:
        raise EvaluationError(f"could not replay frozen Stage-2 IDs before task promotion: {exc}") from exc
    if task_index < 1 or task_index > len(specs):
        raise EvaluationError("task promotion index is not present in the deployed Stage-2 matrix")
    return tuple(specs[task_index - 1].question_ids)


def _validate_promotable_eval_log(*, path: Path, launch: Mapping[str, Any], task_index: int) -> int:
    """Require the output's limit and source IDs before publishing a receipt.

    Header ``question_ids_from`` remains the immutable full 100-ID source
    pool.  Inspect's `--limit` must record the r002 prefix both in EvalConfig
    and DatasetInfo, so a valid 50-sample count alone cannot silently select a
    different IID subset.
    """

    sample_count = _inspect_success(path, task_index=task_index)
    full_ids = _full_question_ids_for_task(launch=launch, task_index=task_index)
    expected_ids = full_ids[:sample_count]
    try:
        from experiments.stage2_ood_hle import raw_preflight as mechanical
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(path), header_only=False)
    except Exception as exc:
        raise EvaluationError(f"could not reopen EvalLog for r002 promotion validation: {path}") from exc
    evaluation = _value_field(log, "eval")
    config = _value_field(evaluation, "config")
    dataset = _value_field(evaluation, "dataset")
    try:
        # This reads task_args and metadata and rejects a version-dependent
        # disagreement rather than trusting only one Inspect header field.
        header_ids = mechanical._header_value(evaluation, "question_ids_from")
    except Exception as exc:
        raise EvaluationError(f"task-{task_index} EvalLog has no exact full source-ID header: {path}") from exc
    dataset_ids = _value_field(dataset, "sample_ids")
    if (
        not isinstance(header_ids, list)
        or tuple(header_ids) != full_ids
        or _value_field(config, "limit") != sample_count
        or _value_field(dataset, "samples") != 100
        or _value_field(dataset, "shuffled") is not False
        or not isinstance(dataset_ids, list)
        or tuple(dataset_ids) != expected_ids
    ):
        raise EvaluationError(
            f"task-{task_index} EvalLog does not prove the exact r002 limit/full-ID/prefix-ID contract: {path}"
        )
    return sample_count


def _load_task_receipt(
    paths: Paths,
    *,
    task_index: int,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
    allow_pending_clean_publication: bool = False,
) -> dict[str, Any] | None:
    receipt_path = _task_receipt_path(paths, task_index)
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    document = _read_json(receipt_path, label=f"task-{task_index} receipt")
    expected_keys = {
        "schema",
        "task_index",
        "sample_count",
        "launch_contract_sha256",
        "evaluation_receipt_sha256",
        "attempt_log",
        "canonical_log",
    }
    if set(document) != expected_keys or document.get("schema") != TASK_RECEIPT_SCHEMA:
        raise EvaluationError(f"task-{task_index} receipt has an unsupported schema")
    if (
        document.get("task_index") != task_index
        or document.get("sample_count") != _sample_count_for_task(task_index)
        or document.get("launch_contract_sha256") != launch_contract_sha256
        or document.get("evaluation_receipt_sha256") != evaluation_receipt_sha256
    ):
        raise EvaluationError(f"task-{task_index} receipt binds different launch/evaluation evidence")
    attempt = document.get("attempt_log")
    canonical = document.get("canonical_log")
    if not isinstance(attempt, Mapping) or not isinstance(canonical, Mapping):
        raise EvaluationError(f"task-{task_index} receipt lacks attempt/canonical identities")
    attempt_declared = Path(str(attempt.get("path", "")))
    canonical_declared = Path(str(canonical.get("path", "")))
    if attempt_declared.is_symlink() or canonical_declared.is_symlink():
        raise EvaluationError(f"task-{task_index} receipt names a linked attempt/canonical log")
    attempt_path = attempt_declared.resolve()
    canonical_path = canonical_declared.resolve()
    _under_root(attempt_path, paths.attempts, label=f"task-{task_index} attempt log")
    _under_root(canonical_path, paths.raw, label=f"task-{task_index} canonical log")
    if attempt != _identity(attempt_path, label=f"task-{task_index} attempt log"):
        raise EvaluationError(f"task-{task_index} attempt log changed after receipt publication")
    # Clean receipt publication deliberately precedes clean `.eval` gate
    # publication.  This closes the switch-scorer race: a concurrent biased
    # worker can never observe a clean EvalLog that has no matching receipt.
    # An interruption in this narrow interval is resumable from the immutable
    # attempt log; callers may opt in to seeing that pending state.
    if not canonical_path.exists() and task_index in CLEAN_TASK_INDICES:
        if allow_pending_clean_publication:
            return document
        return None
    if canonical != _identity(canonical_path, label=f"task-{task_index} canonical log"):
        raise EvaluationError(f"task-{task_index} canonical log changed after receipt publication")
    if _inspect_success(canonical_path, task_index=task_index) != document["sample_count"]:
        raise EvaluationError(f"task-{task_index} canonical EvalLog count differs from its receipt")
    return document


def _copy_immutable_file(source: Path, destination: Path, *, label: str) -> None:
    """Copy with write-once link publication; never overwrite a different log."""

    _identity(source, label=label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
            raise FileExistsError(f"refusing to overwrite differing {label}: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != _sha256_file(source):
                raise FileExistsError(f"{label} appeared with different bytes: {destination}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _promote_attempt(
    *,
    attempt: Path,
    task_index: int,
    launch: Mapping[str, Any],
    paths: Paths,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> bool:
    """Promote precisely one successful task log, leaving every other attempt intact."""

    if attempt.is_symlink() or not attempt.is_dir():
        raise EvaluationError(f"task attempt must be a regular directory: {attempt}")
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # pragma: no cover - configured remote runtime
        raise EvaluationError("Inspect AI is required to promote task logs") from exc
    selected: Path | None = None
    for candidate in sorted(attempt.rglob("*.eval")):
        if candidate.is_symlink() or not candidate.is_file():
            raise EvaluationError(f"task attempt contains a linked/non-file EvalLog: {candidate}")
        try:
            log = read_eval_log(str(candidate), header_only=True)
        except Exception:
            continue  # corrupt/incomplete output remains preserved in its attempt
        evaluation = getattr(log, "eval", None)
        metadata = getattr(evaluation, "metadata", {}) if evaluation is not None else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        if (
            getattr(log, "status", None) != "success"
            or metadata.get("task_indices") != [task_index]
            or metadata.get("task_count") != TASK_COUNT
        ):
            continue
        if selected is not None:
            raise EvaluationError(f"one task attempt produced ambiguous successful EvalLogs: {selected} and {candidate}")
        selected = candidate
    if selected is None:
        return False
    existing = _load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
        allow_pending_clean_publication=True,
    )
    if existing is not None and task_index not in CLEAN_TASK_INDICES:
        return False
    digest = _sha256_file(selected)
    canonical = paths.raw / f"task-{task_index:03d}" / f"{digest}.eval"
    sample_count = _validate_promotable_eval_log(path=selected, launch=launch, task_index=task_index)
    receipt = {
        "schema": TASK_RECEIPT_SCHEMA,
        "task_index": task_index,
        "sample_count": sample_count,
        "launch_contract_sha256": launch_contract_sha256,
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "attempt_log": _identity(selected, label=f"task-{task_index} attempt EvalLog"),
        # The source bytes are the proposed canonical identity.  For clean
        # cells this declaration is written *before* the gate file itself.
        "canonical_log": {
            "path": str(canonical),
            "sha256": digest,
            "size_bytes": selected.stat().st_size,
        },
    }
    if task_index in CLEAN_TASK_INDICES:
        _write_immutable_json(_task_receipt_path(paths, task_index), receipt, label=f"task-{task_index} receipt")
        _copy_immutable_file(selected, canonical, label=f"task-{task_index} clean gate EvalLog")
        _validate_promotable_eval_log(path=canonical, launch=launch, task_index=task_index)
    else:
        _copy_immutable_file(selected, canonical, label=f"task-{task_index} canonical EvalLog")
        _validate_promotable_eval_log(path=canonical, launch=launch, task_index=task_index)
        _write_immutable_json(_task_receipt_path(paths, task_index), receipt, label=f"task-{task_index} receipt")
    # Re-open with normal strict publication semantics.  This catches any
    # source/canonical mismatch before a worker reports successful promotion.
    if _load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_contract_sha256,
        evaluation_receipt_sha256=evaluation_receipt_sha256,
    ) is None:
        raise EvaluationError(f"task-{task_index} receipt was published without its canonical gate EvalLog")
    return True


def _promote_prior_attempts(
    *,
    paths: Paths,
    task_index: int,
    launch: Mapping[str, Any],
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> bool:
    root = paths.attempts / f"task-{task_index:03d}"
    if not root.exists():
        return False
    if root.is_symlink() or not root.is_dir():
        raise EvaluationError(f"task attempt root must be a regular directory: {root}")
    promoted = False
    for attempt in sorted(item for item in root.iterdir() if item.is_dir() and not item.is_symlink()):
        promoted = _promote_attempt(
            attempt=attempt,
            task_index=task_index,
            launch=launch,
            paths=paths,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        ) or promoted
    return promoted


def _one_worker_gpu_token() -> str:
    """Require one local Slurm allocation but do not assume global uniqueness."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if (
        len(values) != 1
        or values[0] in {"-1", "NoDevFiles"}
        or any(character.isspace() for character in values[0])
    ):
        raise EvaluationError("each evaluator worker requires exactly one Slurm-visible CUDA device")
    # The string is intentionally not returned as a global identifier or used
    # in an output path.  Every node may expose its local GPU as "0".
    return values[0]


def _task_command(*, launch: Mapping[str, Any], receipt: Mapping[str, Any], paths: Paths, attempt: Path, task_index: int, python: str) -> list[str]:
    if task_index < 1 or task_index > TASK_COUNT:
        raise EvaluationError("worker task index is outside the frozen 21-cell matrix")
    deployment = launch.get("deployment_manifest")
    runtime = receipt.get("runtime")
    if not isinstance(deployment, Mapping) or not isinstance(runtime, Mapping):
        raise EvaluationError("launch/evaluation receipt lacks deployment/runtime configuration")
    task_args = {
        "manifest": str(deployment["path"]),
        # This shared canonical root is essential: it lets the installed
        # switch_scorer poll for matching clean logs while biased workers are
        # already running.  Do not replace it with a phase-local directory.
        "unbiased_log": str(paths.raw),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "run_evals.py"),
        "--task-factory",
        TASK_FACTORY,
        "--local-checkpoint",
        str(runtime["checkpoint"]),
        "--base-model",
        str(MODEL_SNAPSHOT),
        "--task-args",
        json.dumps(task_args, sort_keys=True, separators=(",", ":")),
        "--model-args",
        json.dumps(VLLM_MODEL_ARGS, sort_keys=True, separators=(",", ":")),
        "--generation-config",
        json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":")),
        "--log-dir",
        str(attempt),
        "--limit",
        str(_sample_count_for_task(task_index)),
        "--max-tasks",
        "1",
        "--isolate-tasks",
        "--persistent-vllm-server",
        "--task-index",
        str(task_index),
        "--yes",
    ]


def worker(
    *,
    step: int,
    output_root: str | Path,
    task_index: int,
    python: str,
    phase: int | None = None,
    campaign_root: str | Path | None = None,
    step16_root: str | Path | None = None,
    step64_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run/resume one independent cell; never wait for a clean barrier."""

    if phase is not None:
        if campaign_root is None or step16_root is None or step64_root is None:
            raise EvaluationError("phase-bound workers require campaign and both target roots")
        validate_worker_phase_admission(
            phase=phase,
            step=step,
            task_index=task_index,
            campaign_root=campaign_root,
            step16_root=step16_root,
            step64_root=step64_root,
        )
    target = _target(step)
    paths = _paths(output_root, target=target)
    launch, launch_sha256 = _load_launch(paths, target=target)
    receipt, evaluation_sha256 = _load_evaluation_receipt(
        paths,
        target=target,
        launch=launch,
        launch_sha256=launch_sha256,
    )
    if _load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    ) is not None:
        return {"step": step, "task_index": task_index, "status": "resumed", "promoted": False}
    promoted_prior = _promote_prior_attempts(
        paths=paths,
        task_index=task_index,
        launch=launch,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    if _load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    ) is not None:
        return {"step": step, "task_index": task_index, "status": "promoted-prior", "promoted": promoted_prior}
    _one_worker_gpu_token()
    if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") != "0":
        raise EvaluationError("worker requires VLLM_USE_FLASHINFER_SAMPLER=0")
    try:
        import vllm
    except ImportError as exc:  # pragma: no cover - configured remote runtime
        raise EvaluationError("worker vLLM runtime is unavailable") from exc
    if getattr(vllm, "__version__", None) != VLLM_VERSION:
        raise EvaluationError(f"worker requires vLLM {VLLM_VERSION}")
    attempt = _next_attempt(paths.attempts / f"task-{task_index:03d}", label="attempt")
    command = _task_command(
        launch=launch,
        receipt=receipt,
        paths=paths,
        attempt=attempt,
        task_index=task_index,
        python=python,
    )
    environment = os.environ.copy()
    environment.update(VLLM_SAMPLER_ENVIRONMENT)
    for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "CTM_PERSISTENT_VLLM_SERVER_METADATA"):
        environment.pop(name, None)
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
    promoted = _promote_attempt(
        attempt=attempt,
        task_index=task_index,
        launch=launch,
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    if result.returncode:
        raise EvaluationError(
            f"task-{task_index} exited {result.returncode}; successful output promoted={promoted}; preserved attempt={attempt}"
        )
    if _load_task_receipt(
        paths,
        task_index=task_index,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    ) is None:
        raise EvaluationError(f"task-{task_index} returned success without one exact successful EvalLog: {attempt}")
    return {"step": step, "task_index": task_index, "status": "completed", "promoted": promoted, "attempt": str(attempt)}


def _validate_canonical_raw_custody(
    *,
    paths: Paths,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> list[dict[str, Any]]:
    """Require exactly the 21 receipt-selected canonical EvalLogs, no retries."""

    selected: list[dict[str, Any]] = []
    expected: set[Path] = set()
    for task_index in range(1, TASK_COUNT + 1):
        receipt = _load_task_receipt(
            paths,
            task_index=task_index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        )
        if receipt is None:
            raise EvaluationError(f"cannot finalize: task-{task_index} has no valid immutable receipt")
        canonical = receipt["canonical_log"]
        assert isinstance(canonical, Mapping)
        expected.add(Path(str(canonical["path"])).resolve())
        selected.append(
            {
                "task_index": task_index,
                "sample_count": _sample_count_for_task(task_index),
                "receipt": _identity(_task_receipt_path(paths, task_index), label=f"task-{task_index} receipt"),
            }
        )
    if not paths.raw.exists() or paths.raw.is_symlink() or not paths.raw.is_dir():
        raise EvaluationError("canonical raw root is absent or not a regular directory")
    found: set[Path] = set()
    for path in paths.raw.rglob("*.eval"):
        if path.is_symlink() or not path.is_file():
            raise EvaluationError(f"canonical raw tree contains a linked/non-file EvalLog: {path}")
        found.add(path.resolve())
    if found != expected:
        raise EvaluationError(
            "canonical raw tree contains unreceipted/missing EvalLogs; "
            f"unexpected={sorted(str(path) for path in found - expected)}, "
            f"missing={sorted(str(path) for path in expected - found)}"
        )
    if sum(record["sample_count"] for record in selected) != TOTAL_SAMPLES_PER_CHECKPOINT:
        raise EvaluationError("canonical raw custody does not bind the exact r002 total sample count")
    return selected


def _seal_clean_gate(
    *,
    paths: Paths,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> dict[str, Any]:
    """Hash-bind the three receipt-before-publication clean gate entries."""

    rows: list[dict[str, Any]] = []
    for task_index in CLEAN_TASK_INDICES:
        receipt = _load_task_receipt(
            paths,
            task_index=task_index,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        )
        if receipt is None:
            raise EvaluationError(f"paired-clean gate cannot seal without task-{task_index} receipt/publication")
        canonical = receipt["canonical_log"]
        assert isinstance(canonical, Mapping)
        rows.append(
            {
                "task_index": task_index,
                "sample_count": _sample_count_for_task(task_index),
                "task_receipt": _identity(_task_receipt_path(paths, task_index), label=f"task-{task_index} receipt"),
                "clean_log": dict(canonical),
            }
        )
    document = {
        "schema": CLEAN_GATE_RECEIPT_SCHEMA,
        "raw_root": str(paths.raw),
        "launch_contract_sha256": launch_contract_sha256,
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "clean": rows,
        "publication_order": "task_receipt_before_clean_eval_log",
    }
    _write_immutable_json(paths.clean_gate_receipt, document, label="paired-clean gate receipt")
    return document


def _expected_r002_task_identities() -> tuple[tuple[str, str, str, str, str | None], ...]:
    """Return the one frozen Stage-2 factory order accepted by r002.

    HLE's dataset spelling is imported from the owned Stage-2 materializer;
    the stable ordered cell topology itself is deliberately local and cannot
    be changed by a task-factory implementation swap.
    """

    try:
        # Import the lightweight frozen HLE spelling directly rather than the
        # materializer, whose optional preparation dependencies are not
        # needed for this static identity table.
        from experiments.rmct_tbsr.constants import HLE_DATASET
        from experiments.stage2_ood_hle.prepare import TRAINING_BIAS
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("frozen Stage-2 constants are unavailable for r002 identity validation") from exc
    IID = "iid"
    HELDOUT_DATASET = "heldout_dataset"
    HELDOUT_BIAS = "heldout_bias"
    HELDOUT_DATASET_AND_BIAS = "heldout_dataset_and_bias"
    HELDOUT_BIASES = ("suggested_answer", *HELD_OUT_BIASES)
    in_domain = ("logiqa", "hellaswag")
    identities: list[tuple[str, str, str, str, str | None]] = [
        *[("unbiased", IID, "in_domain", dataset, None) for dataset in in_domain],
        ("unbiased", HELDOUT_DATASET, "hle", HLE_DATASET, None),
        *[("biased", IID, "in_domain", dataset, TRAINING_BIAS) for dataset in in_domain],
        ("biased", HELDOUT_DATASET, "hle", HLE_DATASET, TRAINING_BIAS),
    ]
    for bias_type in HELDOUT_BIASES:
        identities.extend(("biased", HELDOUT_BIAS, "in_domain", dataset, bias_type) for dataset in in_domain)
    for bias_type in HELDOUT_BIASES:
        identities.append(("biased", HELDOUT_DATASET_AND_BIAS, "hle", HLE_DATASET, bias_type))
    if len(identities) != TASK_COUNT:  # pragma: no cover - frozen topology guard
        raise EvaluationError("r002 static Stage-2 identity topology is not 21 cells")
    return tuple(identities)


def _validate_r002_sampling_matrix(specs: Sequence[Any]) -> None:
    """Reject a task-factory/deployment order that no longer matches r002.

    Counts are keyed by the stable task-factory positions so the command line,
    receipts, and plots cannot accidentally reinterpret a 50-sample IID cell
    as a 100-sample HLE cell.  The population assertions make that position
    binding explicit and protect the intended shared clean question pools.
    """

    if len(specs) != TASK_COUNT:
        raise EvaluationError("Stage-2 task factory no longer yields the frozen 21-cell matrix")
    expected_identities = _expected_r002_task_identities()
    question_pools: dict[tuple[str, str], list[tuple[str, ...]]] = {}
    for task_index, spec in enumerate(specs, start=1):
        expected_count = _sample_count_for_task(task_index)
        expected_population = "in_domain" if task_index in IID_TASK_INDICES else "hle"
        expected_kind = "unbiased" if task_index in CLEAN_TASK_INDICES else "biased"
        question_ids = getattr(spec, "question_ids", None)
        identity = (
            getattr(spec, "kind", None),
            getattr(spec, "regime", None),
            getattr(spec, "population", None),
            getattr(spec, "dataset", None),
            getattr(spec, "bias_type", None),
        )
        if (
            identity != expected_identities[task_index - 1]
            or getattr(spec, "population", None) != expected_population
            or getattr(spec, "kind", None) != expected_kind
            or not isinstance(question_ids, tuple)
            or len(question_ids) != 100
            or len(question_ids) != len(set(question_ids))
            or any(not isinstance(question_id, str) or not question_id for question_id in question_ids)
            or len(question_ids) < expected_count
        ):
            raise EvaluationError(
                f"Stage-2 task-{task_index} no longer matches its frozen r002 population/kind/question-pool contract"
            )
        key = (str(spec.population), str(spec.dataset))
        question_pools.setdefault(key, []).append(question_ids)
    if len(question_pools) != len(CLEAN_TASK_INDICES) or any(len(pools) != 7 for pools in question_pools.values()):
        raise EvaluationError("Stage-2 task factory no longer has seven matched variants for every clean question pool")
    for population_dataset, pools in question_pools.items():
        if any(pool != pools[0] for pool in pools[1:]):
            raise EvaluationError(
                "Stage-2 task factory no longer shares one ordered full question-id pool across variants: "
                f"{population_dataset}"
            )


def _subset_spec_for_task(spec: Any, *, task_index: int) -> Any:
    """Make the exact ordered source-id subset expected after ``--limit``."""

    sample_count = _sample_count_for_task(task_index)
    question_ids = tuple(getattr(spec, "question_ids", ()))
    if len(question_ids) < sample_count:
        raise EvaluationError(f"task-{task_index} has fewer frozen question IDs than its r002 sample cap")
    return replace(spec, question_ids=question_ids[:sample_count])


def _early_snapshot_vllm_runtime(evaluation_runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact snapshot-backed runtime expected in raw EvalLog headers.

    The early-step receipt is different from r005's terminal receipt, but it
    serves the identical absolute Qwen snapshot through a parity-attested
    compatibility adapter.  Do not route this through the legacy Stage-2
    alias verifier: native vLLM records the absolute snapshot in ``eval.model``.
    """

    expected = {
        "profile": "vllm",
        "base_model": str(MODEL_SNAPSHOT),
        "model_args": dict(VLLM_MODEL_ARGS),
        "generation_config": dict(GENERATION_CONFIG),
    }
    checkpoint = evaluation_runtime.get("checkpoint")
    if not isinstance(checkpoint, str) or not Path(checkpoint).is_absolute():
        raise EvaluationError("early evaluation receipt has no absolute compatibility checkpoint")
    for field, value in expected.items():
        if evaluation_runtime.get(field) != value:
            raise EvaluationError(f"early evaluation receipt has a changed snapshot vLLM {field} contract")
    return {**expected, "checkpoint": checkpoint}


def _assert_early_snapshot_vllm_runtime(
    header_log: Any,
    *,
    path: Path,
    runtime: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Use r005's exact snapshot/header verifier for this early receipt.

    Its checks cover the absolute ``vllm/<snapshot>:<adapter>`` model string,
    local checkpoint metadata, command model args, and generation config.  We
    first prove r005's frozen constants equal the early evaluator's controls,
    so this reuse cannot silently reinterpret a different runtime profile.
    """

    try:
        from experiments.rmct_two_bias_eval import raw_preflight as r005_raw_preflight
    except ImportError as exc:  # pragma: no cover - configured evaluation environment
        raise EvaluationError("snapshot-aware r005 raw-preflight verifier is unavailable") from exc
    if (
        r005_raw_preflight.BASE_MODEL != str(MODEL_SNAPSHOT)
        or r005_raw_preflight.VLLM_MODEL_ARGS != VLLM_MODEL_ARGS
        or r005_raw_preflight.VLLM_GENERATION_CONFIG != GENERATION_CONFIG
    ):
        raise EvaluationError("r005 snapshot verifier constants do not match the frozen early evaluation runtime")
    try:
        model, observed = r005_raw_preflight._assert_snapshot_vllm_runtime(
            header_log,
            path=path,
            runtime=runtime,
        )
    except Exception as exc:
        raise EvaluationError(f"raw EvalLog fails the exact snapshot-backed vLLM header contract: {exc}") from exc
    if not isinstance(model, str) or not isinstance(observed, Mapping) or dict(observed) != runtime:
        raise EvaluationError("snapshot-aware raw-preflight verifier returned incomplete runtime evidence")
    return model, dict(observed)


def _native_two_bias_preflight(*, target: ApprovedTarget, paths: Paths, launch: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate r002's mixed 50/100 sample matrix without changing r005 code.

    The shared Stage-2 mechanical preflight deliberately retains its legacy
    full-100-cell contract for r005.  This isolated evaluator instead reuses
    its strict header/runtime/switch primitives with a task-local subset spec.
    It additionally requires every biased cell's *ordered* question IDs to
    equal the matching published clean cell's IDs, closing the usual `--limit`
    ambiguity for the IID 50-question comparison.
    """

    try:
        from experiments.stage2_ood_hle import raw_preflight as mechanical
        from experiments.stage2_ood_hle.tasks import ood_task_specs
    except ImportError as exc:  # pragma: no cover - configured remote runtime
        raise EvaluationError("Stage-2 native raw-preflight primitives are unavailable") from exc
    deployment = launch.get("deployment_manifest")
    runtime = evaluation.get("runtime")
    if not isinstance(deployment, Mapping) or not isinstance(runtime, Mapping):
        raise EvaluationError("cannot preflight incomplete launch/evaluation receipts")
    deployment_path = deployment.get("path")
    checkpoint = runtime.get("checkpoint")
    if not isinstance(deployment_path, str) or not isinstance(checkpoint, str):
        raise EvaluationError("cannot preflight launch/evaluation with no deployment/checkpoint path")
    launch_sha256 = _sha256_file(paths.contract)
    evaluation_sha256 = _sha256_file(paths.evaluation_receipt)
    # Re-open all canonical task receipts first; the native report is never
    # allowed to select a retry outside the receipt-selected raw tree.
    _validate_canonical_raw_custody(
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    try:
        mechanical.validate_manifest(deployment_path)
        specs = list(ood_task_specs(deployment_path))
        _validate_r002_sampling_matrix(specs)
        # r002's worker writes the absolute pinned snapshot in ``eval.model``.
        # Validate the early evaluation receipt before examining a raw log,
        # then use r005's snapshot-aware header verifier below rather than the
        # legacy Stage-2 alias-only runtime contract.
        snapshot_runtime = _early_snapshot_vllm_runtime(runtime)
    except Exception as exc:
        raise EvaluationError(f"native Stage-2 deployment/runtime preflight failed: {exc}") from exc

    loaded: dict[int, Any] = {}
    for task_index, source_spec in enumerate(specs, start=1):
        task_receipt = _load_task_receipt(
            paths,
            task_index=task_index,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        )
        if task_receipt is None:  # covered above, retained for a local clear error
            raise EvaluationError(f"native preflight has no sealed task-{task_index} receipt")
        canonical = task_receipt.get("canonical_log")
        if not isinstance(canonical, Mapping):
            raise EvaluationError(f"native preflight task-{task_index} receipt has no canonical log")
        path = Path(str(canonical.get("path", ""))).resolve()
        try:
            header_log = mechanical._read_eval_log(path, header_only=True)
            created = mechanical._validate_header(header_log, path=path, spec=source_spec, raw_root=paths.raw)
            model, observed_runtime = _assert_early_snapshot_vllm_runtime(
                header_log,
                path=path,
                runtime=snapshot_runtime,
            )
        except Exception as exc:
            raise EvaluationError(f"native preflight could not verify task-{task_index} header/runtime: {exc}") from exc
        loaded[task_index] = mechanical.LoadedTaskLog(
            _subset_spec_for_task(source_spec, task_index=task_index),
            path,
            created,
            header_log,
            model,
            observed_runtime,
        )

    clean_paths = {
        (item.spec.population, item.spec.dataset): item.path
        for task_index, item in loaded.items()
        if task_index in CLEAN_TASK_INDICES
    }
    if len(clean_paths) != len(CLEAN_TASK_INDICES):
        raise EvaluationError("native r002 preflight has no unique clean reference for every population/dataset")

    sample_ids: dict[int, tuple[str, ...]] = {}
    sample_counts: dict[int, int] = {}
    for task_index in range(1, TASK_COUNT + 1):
        item = loaded[task_index]
        try:
            full_log = mechanical._read_eval_log(item.path, header_only=False)
            count = mechanical._validate_samples(full_log, loaded=item, clean_paths=clean_paths)
        except Exception as exc:
            raise EvaluationError(f"native r002 sample/switch preflight failed for task-{task_index}: {exc}") from exc
        samples = list(mechanical._attribute(full_log, "samples", []) or [])
        ids = tuple(mechanical._attribute(sample, "id", "") for sample in samples)
        expected_ids = tuple(item.spec.question_ids)
        if (
            count != _sample_count_for_task(task_index)
            or len(ids) != count
            or any(not isinstance(sample_id, str) or not sample_id for sample_id in ids)
            or ids != expected_ids
        ):
            raise EvaluationError(
                f"task-{task_index} does not retain its exact ordered r002 question-id subset "
                f"(count={count}, expected={_sample_count_for_task(task_index)})"
            )
        sample_counts[task_index] = count
        sample_ids[task_index] = ids
        if item.spec.kind == "biased":
            for sample in samples:
                _switch, switch_metadata = mechanical._switch_score(sample, path=item.path)
                for key, note in switch_metadata.items():
                    if "missing" in str(key).lower() or "unmatched" in str(key).lower():
                        # Empty lists/dicts are safe diagnostics too; use
                        # equality rather than truthiness so any stated
                        # missing ID is fail-closed.
                        safe_note = (
                            note is None
                            or note is False
                            or note == ""
                            or note == 0
                            or note == []
                            or note == {}
                        )
                        if not safe_note:
                            raise EvaluationError(
                                f"task-{task_index} switch scorer reported missing/unmatched IDs in {key!r}"
                            )

    clean_ids = {
        (loaded[task_index].spec.population, loaded[task_index].spec.dataset): sample_ids[task_index]
        for task_index in CLEAN_TASK_INDICES
    }
    for task_index in BIASED_TASK_INDICES:
        item = loaded[task_index]
        matching_clean = clean_ids.get((item.spec.population, item.spec.dataset))
        if matching_clean is None or sample_ids[task_index] != matching_clean:
            raise EvaluationError(
                f"task-{task_index} does not use the matching clean task's exact ordered r002 question-id pool"
            )

    clean_records: dict[tuple[str, str], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for task_index in range(1, TASK_COUNT + 1):
        item = loaded[task_index]
        if item.spec.kind != "unbiased":
            continue
        row = mechanical._source_record(item, sample_count=sample_counts[task_index], clean_records={})
        row.update(
            {
                "task_index": task_index,
                "expected_sample_count": _sample_count_for_task(task_index),
                "task_receipt": _identity(_task_receipt_path(paths, task_index), label=f"task-{task_index} receipt"),
                "evaluation_bias_status": None,
            }
        )
        clean_records[(item.spec.population, item.spec.dataset)] = row
        sources.append(row)
    for task_index in range(1, TASK_COUNT + 1):
        item = loaded[task_index]
        if item.spec.kind != "biased":
            continue
        row = mechanical._source_record(item, sample_count=sample_counts[task_index], clean_records=clean_records)
        row["unbiased_log"] = str(paths.raw)
        row.update(
            {
                "task_index": task_index,
                "expected_sample_count": _sample_count_for_task(task_index),
                "task_receipt": _identity(_task_receipt_path(paths, task_index), label=f"task-{task_index} receipt"),
                "evaluation_bias_status": "seen" if item.spec.bias_type in SEEN_BIASES else "held_out",
            }
        )
        sources.append(row)

    _seal_clean_gate(
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    report = {
        "schema": NATIVE_PREFLIGHT_SCHEMA,
        "condition": target.condition,
        "target": {"step": target.step, "run_name": target.run_name, "segment_index": target.segment_index},
        "evaluation_receipt": _identity(paths.evaluation_receipt, label="evaluation receipt"),
        "raw_root": str(paths.raw),
        "clean_gate": _identity(paths.clean_gate_receipt, label="paired-clean gate receipt"),
        "sampling": _sampling_contract(),
        "science": {
            "all_biases": list(ALL_BIASES),
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "include_bias_acknowledged": False,
            "grader_model": None,
        },
        "sources": sources,
    }
    validate_native_preflight(report)
    _write_immutable_json(paths.native_preflight, report, label="native r002 two-bias raw preflight")
    return report


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _valid_identity_record(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("path"), str)
        and Path(value["path"]).is_absolute()
        and _is_sha256(value.get("sha256"))
        and not isinstance(value.get("size_bytes"), bool)
        and isinstance(value.get("size_bytes"), int)
        and value["size_bytes"] > 0
    )


def validate_native_preflight(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Validate the self-contained r002 mixed-count preflight summary."""

    if isinstance(value, Mapping):
        report = dict(value)
    else:
        report = _read_json(_resolve_unlinked(value, label="native two-bias preflight"), label="native two-bias preflight")
    if set(report) != {
        "schema",
        "condition",
        "target",
        "evaluation_receipt",
        "raw_root",
        "clean_gate",
        "sampling",
        "science",
        "sources",
    } or report.get("schema") != NATIVE_PREFLIGHT_SCHEMA:
        raise EvaluationError("native two-bias preflight has an unsupported schema")
    condition = report.get("condition")
    matching = [target for target in APPROVED_TARGETS.values() if target.condition == condition]
    if len(matching) != 1:
        raise EvaluationError("native preflight condition is not an approved early-checkpoint target")
    target = matching[0]
    target_record = report.get("target")
    if not isinstance(target_record, Mapping) or target_record != {
        "step": target.step,
        "run_name": target.run_name,
        "segment_index": target.segment_index,
    }:
        raise EvaluationError("native preflight target identity is inconsistent")
    raw_root = report.get("raw_root")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise EvaluationError("native preflight raw root is invalid")
    for field in ("evaluation_receipt", "clean_gate"):
        if not _valid_identity_record(report.get(field)):
            raise EvaluationError(f"native preflight {field} identity is invalid")
    science = report.get("science")
    if science != {
        "all_biases": list(ALL_BIASES),
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "include_bias_acknowledged": False,
        "grader_model": None,
    }:
        raise EvaluationError("native preflight science contract drifted")
    if report.get("sampling") != _sampling_contract():
        raise EvaluationError("native preflight mixed r002 sampling contract drifted")
    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != TASK_COUNT:
        raise EvaluationError("native preflight must contain exactly 21 Stage-2 cells")
    common_fields = {
        "task_index",
        "kind",
        "regime",
        "population",
        "dataset",
        "bias_type",
        "raw_log",
        "raw_log_sha256",
        "created",
        "sample_count",
        "expected_sample_count",
        "question_ids_sha256",
        "frozen_file",
        "frozen_file_sha256",
        "source_identity_digest",
        "prompt_style",
        "model",
        "runtime",
        "task_receipt",
        "evaluation_bias_status",
    }
    clean = 0
    by_bias = {bias: 0 for bias in ALL_BIASES}
    paths_seen: set[str] = set()
    by_task: dict[int, Mapping[str, Any]] = {}
    clean_records: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_identities = _expected_r002_task_identities()
    for expected_task_index, source in enumerate(sources, start=1):
        if not isinstance(source, Mapping):
            raise EvaluationError("native preflight source must be an object")
        task_index = source.get("task_index")
        expected_sample_count = _sample_count_for_task(expected_task_index)
        if task_index != expected_task_index or source.get("sample_count") != expected_sample_count or source.get("expected_sample_count") != expected_sample_count:
            raise EvaluationError("native preflight source has an invalid task-index/sample-count binding")
        expected_fields = set(common_fields)
        if expected_task_index in BIASED_TASK_INDICES:
            expected_fields.update({"unbiased_log", "paired_clean"})
        if set(source) != expected_fields:
            raise EvaluationError(f"native preflight source has an unsupported schema at task-{expected_task_index}")
        identity = (
            source.get("kind"),
            source.get("regime"),
            source.get("population"),
            source.get("dataset"),
            source.get("bias_type"),
        )
        if identity != expected_identities[expected_task_index - 1]:
            raise EvaluationError("native preflight source identity differs from the frozen r002 task-index matrix")
        raw_log = source.get("raw_log")
        if not isinstance(raw_log, str) or not Path(raw_log).is_absolute() or raw_log in paths_seen:
            raise EvaluationError("native preflight source has invalid/duplicate raw log provenance")
        try:
            Path(raw_log).resolve().relative_to(Path(raw_root).resolve())
        except ValueError as exc:
            raise EvaluationError("native preflight source lies outside its canonical raw root") from exc
        paths_seen.add(raw_log)
        if (
            not _is_sha256(source.get("raw_log_sha256"))
            or not _is_sha256(source.get("question_ids_sha256"))
            or not _is_sha256(source.get("frozen_file_sha256"))
            or not isinstance(source.get("frozen_file"), str)
            or not Path(source["frozen_file"]).is_absolute()
            or not isinstance(source.get("created"), str)
            or not source["created"]
            or not isinstance(source.get("source_identity_digest"), str)
            or not source["source_identity_digest"]
            or source.get("prompt_style") != "none"
            or not isinstance(source.get("model"), str)
            or not source["model"]
            or not isinstance(source.get("runtime"), Mapping)
            or source["runtime"].get("profile") != "vllm"
            or not _valid_identity_record(source.get("task_receipt"))
        ):
            raise EvaluationError("native preflight source has malformed immutable provenance")
        by_task[expected_task_index] = source
        if source.get("kind") == "unbiased":
            clean += 1
            if (
                expected_task_index not in CLEAN_TASK_INDICES
                or source.get("bias_type") is not None
                or source.get("evaluation_bias_status") is not None
            ):
                raise EvaluationError("native preflight clean cell has a bias label")
            key = (source.get("population"), source.get("dataset"))
            if not all(isinstance(item, str) and item for item in key) or key in clean_records:
                raise EvaluationError("native preflight clean cell has no unique population/dataset identity")
            clean_records[key] = source
        elif source.get("kind") == "biased":
            bias = source.get("bias_type")
            expected = "seen" if bias in SEEN_BIASES else "held_out"
            if (
                expected_task_index not in BIASED_TASK_INDICES
                or bias not in by_bias
                or source.get("evaluation_bias_status") != expected
                or source.get("unbiased_log") != raw_root
                or not isinstance(source.get("paired_clean"), Mapping)
            ):
                raise EvaluationError("native preflight biased cell has a misclassified bias status")
            by_bias[str(bias)] += 1
        else:
            raise EvaluationError("native preflight source has an unsupported task kind")
    if clean != len(CLEAN_TASK_INDICES) or by_bias != {bias: 3 for bias in ALL_BIASES}:
        raise EvaluationError("native preflight does not retain the exact 3-clean/18-biased matrix")
    if sum(int(source["sample_count"]) for source in sources) != TOTAL_SAMPLES_PER_CHECKPOINT:
        raise EvaluationError("native preflight does not retain the exact r002 1,400-sample total")
    if len(clean_records) != len(CLEAN_TASK_INDICES):
        raise EvaluationError("native preflight lacks the exact clean reference matrix")
    for task_index in BIASED_TASK_INDICES:
        source = by_task[task_index]
        clean_source = clean_records.get((source.get("population"), source.get("dataset")))
        if clean_source is None:
            raise EvaluationError("native preflight biased cell has no matching clean population/dataset")
        expected_pair = {
            "raw_log": clean_source["raw_log"],
            "raw_log_sha256": clean_source["raw_log_sha256"],
            "question_ids_sha256": clean_source["question_ids_sha256"],
            "source_identity_digest": clean_source["source_identity_digest"],
        }
        if source.get("paired_clean") != expected_pair or source.get("question_ids_sha256") != clean_source.get("question_ids_sha256"):
            raise EvaluationError("native preflight biased cell does not bind the exact matching clean question-id subset")
    return report


def finalize(*, step: int, output_root: str | Path) -> dict[str, Any]:
    """Seal one fully complete 21-cell target only after native preflight."""

    target = _target(step)
    paths = _paths(output_root, target=target)
    launch, launch_sha256 = _load_launch(paths, target=target)
    evaluation, evaluation_sha256 = _load_evaluation_receipt(
        paths,
        target=target,
        launch=launch,
        launch_sha256=launch_sha256,
    )
    task_receipts = _validate_canonical_raw_custody(
        paths=paths,
        launch_contract_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )
    report = _native_two_bias_preflight(target=target, paths=paths, launch=launch, evaluation=evaluation)
    # Re-open after write so a pre-existing conflicting report cannot be
    # mistaken for the report constructed above.
    validated = validate_native_preflight(paths.native_preflight)
    if validated != report:
        raise EvaluationError("persisted native preflight differs from the validated report")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "condition": target.condition,
        "target": {"step": target.step, "run_name": target.run_name, "segment_index": target.segment_index},
        "launch_contract": _identity(paths.contract, label="launch contract"),
        "evaluation_receipt": _identity(paths.evaluation_receipt, label="evaluation receipt"),
        "generation": _sampling_contract(),
        "task_receipts": task_receipts,
        "native_preflight": _identity(paths.native_preflight, label="native two-bias preflight"),
    }
    status = _write_immutable_json(paths.completion, completion, label="target evaluation completion receipt")
    return {"step": step, "completion": _identity(paths.completion, label="completion receipt"), "status": status}


def _campaign_path(campaign_root: str | Path) -> Path:
    requested = Path(campaign_root).expanduser()
    if requested.is_symlink() or (requested.exists() and not requested.is_dir()):
        raise EvaluationError(f"campaign root must be a regular directory: {requested}")
    root = requested.resolve()
    if root.exists() and not root.is_dir():
        raise EvaluationError(f"campaign root must be a regular directory: {root}")
    return root


def _phase_receipt_path(campaign_root: Path, phase: int) -> Path:
    if phase not in PHASES:
        raise EvaluationError("campaign phase must be one of 1, 2, or 3")
    return campaign_root / "phases" / f"phase-{phase:03d}.json"


def _phase_target_record(
    *,
    step: int,
    indices: Sequence[int],
    output_root: str | Path,
) -> dict[str, Any]:
    target = _target(step)
    paths = _paths(output_root, target=target)
    launch, launch_sha256 = _load_launch(paths, target=target)
    _evaluation, evaluation_sha256 = _load_evaluation_receipt(
        paths,
        target=target,
        launch=launch,
        launch_sha256=launch_sha256,
    )
    receipt_records: list[dict[str, Any]] = []
    for index in indices:
        if _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        ) is None:
            raise EvaluationError(f"phase cannot seal: step-{step} task-{index} has no valid receipt/publication")
        receipt_records.append(
            {
                "task_index": index,
                "sample_count": _sample_count_for_task(index),
                "receipt": _identity(_task_receipt_path(paths, index), label=f"step-{step} task-{index} receipt"),
            }
        )
    # A target's three clean tasks are always 1..3.  This function is invoked
    # only after the phase's cells have been revalidated; when all three are
    # available, write/replay the gate receipt immediately.
    gate: dict[str, Any] | None = None
    if all(
        _load_task_receipt(
            paths,
            task_index=index,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        ) is not None
        for index in CLEAN_TASK_INDICES
    ):
        _seal_clean_gate(
            paths=paths,
            launch_contract_sha256=launch_sha256,
            evaluation_receipt_sha256=evaluation_sha256,
        )
        gate = _identity(paths.clean_gate_receipt, label=f"step-{step} paired-clean gate receipt")
    return {
        "step": step,
        "condition": target.condition,
        "output_root": str(paths.root),
        "launch_contract": _identity(paths.contract, label=f"step-{step} launch contract"),
        "evaluation_receipt": _identity(paths.evaluation_receipt, label=f"step-{step} evaluation receipt"),
        "task_indices": list(indices),
        "task_receipts": receipt_records,
        "paired_clean_gate": gate,
        # Retain a direct reference to the target receipt's runtime binding;
        # it is helpful when auditing the interleaved phase-two workers.
        "checkpoint_custody": launch["checkpoint_custody"],
    }


def _expected_phase_target_roots(step16_root: str | Path, step64_root: str | Path) -> dict[int, str]:
    return {
        16: str(_paths(step16_root, target=_target(16)).root),
        64: str(_paths(step64_root, target=_target(64)).root),
    }


def validate_phase_receipt(
    receipt: str | Path,
    *,
    phase: int,
    campaign_root: str | Path,
    step16_root: str | Path,
    step64_root: str | Path,
) -> dict[str, Any]:
    """Replay a phase seal and every worker receipt named by its fixed plan."""

    campaign = _campaign_path(campaign_root)
    document = _read_json(_resolve_unlinked(receipt, label=f"phase-{phase} receipt"), label=f"phase-{phase} receipt")
    expected_plan = PHASES.get(phase)
    if expected_plan is None:
        raise EvaluationError("unknown campaign phase")
    required = {"schema", "campaign_root", "phase", "previous_phase", "targets"}
    if set(document) != required or document.get("schema") != PHASE_RECEIPT_SCHEMA:
        raise EvaluationError("phase receipt has an unsupported schema")
    if document.get("campaign_root") != str(campaign) or document.get("phase") != phase:
        raise EvaluationError("phase receipt binds a different campaign/phase")
    previous = document.get("previous_phase")
    if phase == 1:
        if previous is not None:
            raise EvaluationError("phase one receipt must not declare a predecessor")
    else:
        expected_previous_path = _phase_receipt_path(campaign, phase - 1)
        if not isinstance(previous, Mapping) or previous != _identity(expected_previous_path, label=f"phase-{phase - 1} receipt"):
            raise EvaluationError("phase receipt does not bind its exact predecessor phase receipt")
        validate_phase_receipt(
            expected_previous_path,
            phase=phase - 1,
            campaign_root=campaign,
            step16_root=step16_root,
            step64_root=step64_root,
        )
    targets = document.get("targets")
    if not isinstance(targets, list) or len(targets) != len(expected_plan):
        raise EvaluationError("phase receipt target set is incomplete")
    roots = _expected_phase_target_roots(step16_root, step64_root)
    expected_steps = list(sorted(expected_plan))
    if [record.get("step") if isinstance(record, Mapping) else None for record in targets] != expected_steps:
        raise EvaluationError("phase receipt target order differs from fixed campaign plan")
    for record in targets:
        assert isinstance(record, Mapping)
        step = record["step"]
        target = _target(step)
        indices = tuple(expected_plan[step])
        if (
            record.get("condition") != target.condition
            or record.get("output_root") != roots[step]
            or record.get("task_indices") != list(indices)
            or not isinstance(record.get("launch_contract"), Mapping)
            or not isinstance(record.get("evaluation_receipt"), Mapping)
            or not isinstance(record.get("task_receipts"), list)
            or not isinstance(record.get("checkpoint_custody"), Mapping)
        ):
            raise EvaluationError("phase receipt contains malformed target custody")
        paths = _paths(roots[step], target=target)
        launch, launch_sha256 = _load_launch(paths, target=target)
        _evaluation, evaluation_sha256 = _load_evaluation_receipt(
            paths,
            target=target,
            launch=launch,
            launch_sha256=launch_sha256,
        )
        if record["launch_contract"] != _identity(paths.contract, label=f"step-{step} launch contract"):
            raise EvaluationError("phase receipt launch-contract identity changed")
        if record["evaluation_receipt"] != _identity(paths.evaluation_receipt, label=f"step-{step} evaluation receipt"):
            raise EvaluationError("phase receipt evaluation-receipt identity changed")
        if record["checkpoint_custody"] != launch["checkpoint_custody"]:
            raise EvaluationError("phase receipt checkpoint custody differs from target launch contract")
        expected_receipts: list[dict[str, Any]] = []
        for index in indices:
            if _load_task_receipt(
                paths,
                task_index=index,
                launch_contract_sha256=launch_sha256,
                evaluation_receipt_sha256=evaluation_sha256,
            ) is None:
                raise EvaluationError(f"phase receipt names an unsealed step-{step} task-{index}")
            expected_receipts.append(
                {
                    "task_index": index,
                    "sample_count": _sample_count_for_task(index),
                    "receipt": _identity(_task_receipt_path(paths, index), label=f"step-{step} task-{index} receipt"),
                }
            )
        if record["task_receipts"] != expected_receipts:
            raise EvaluationError("phase receipt task receipt identities changed")
        gate = record.get("paired_clean_gate")
        clean_ready = all(
            _load_task_receipt(
                paths,
                task_index=index,
                launch_contract_sha256=launch_sha256,
                evaluation_receipt_sha256=evaluation_sha256,
            ) is not None
            for index in CLEAN_TASK_INDICES
        )
        if clean_ready:
            _seal_clean_gate(
                paths=paths,
                launch_contract_sha256=launch_sha256,
                evaluation_receipt_sha256=evaluation_sha256,
            )
            if gate != _identity(paths.clean_gate_receipt, label=f"step-{step} paired-clean gate receipt"):
                raise EvaluationError("phase receipt clean-gate identity changed")
        elif gate is not None:
            raise EvaluationError("phase receipt claims a clean gate before all three clean receipts are sealed")
    return document


def seal_phase(
    *,
    phase: int,
    campaign_root: str | Path,
    step16_root: str | Path,
    step64_root: str | Path,
) -> dict[str, Any]:
    """Write/replay a deterministic phase receipt after its exact 14 cells seal."""

    campaign = _campaign_path(campaign_root)
    plan = PHASES.get(phase)
    if plan is None:
        raise EvaluationError("campaign phase must be one of 1, 2, or 3")
    if phase > 1:
        predecessor = _phase_receipt_path(campaign, phase - 1)
        validate_phase_receipt(
            predecessor,
            phase=phase - 1,
            campaign_root=campaign,
            step16_root=step16_root,
            step64_root=step64_root,
        )
        previous: dict[str, Any] | None = _identity(predecessor, label=f"phase-{phase - 1} receipt")
    else:
        previous = None
    root_by_step = {16: step16_root, 64: step64_root}
    records = [
        _phase_target_record(step=step, indices=plan[step], output_root=root_by_step[step])
        for step in sorted(plan)
    ]
    if sum(len(record["task_indices"]) for record in records) != 14:
        raise EvaluationError("every campaign phase must seal exactly fourteen one-GPU cells")
    document = {
        "schema": PHASE_RECEIPT_SCHEMA,
        "campaign_root": str(campaign),
        "phase": phase,
        "previous_phase": previous,
        "targets": records,
    }
    path = _phase_receipt_path(campaign, phase)
    status = _write_immutable_json(path, document, label=f"phase-{phase} receipt")
    validate_phase_receipt(
        path,
        phase=phase,
        campaign_root=campaign,
        step16_root=step16_root,
        step64_root=step64_root,
    )
    return {"phase": phase, "receipt": _identity(path, label=f"phase-{phase} receipt"), "status": status}


def validate_worker_phase_admission(
    *,
    phase: int,
    step: int,
    task_index: int,
    campaign_root: str | Path,
    step16_root: str | Path,
    step64_root: str | Path,
) -> None:
    """Prevent manually launched later-phase workers from bypassing phase seals."""

    plan = PHASES.get(phase)
    if plan is None or task_index not in plan.get(step, ()):
        raise EvaluationError("worker step/task is not admitted by the requested fixed campaign phase")
    if phase > 1:
        campaign = _campaign_path(campaign_root)
        validate_phase_receipt(
            _phase_receipt_path(campaign, phase - 1),
            phase=phase - 1,
            campaign_root=campaign,
            step16_root=step16_root,
            step64_root=step64_root,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="validate one target, translate PEFT, and prove strict vLLM parity")
    prepare_parser.add_argument("--repository", required=True, type=Path)
    prepare_parser.add_argument("--step", required=True, type=int, choices=sorted(APPROVED_TARGETS))
    prepare_parser.add_argument("--source-stage2-manifest", required=True, type=Path)
    prepare_parser.add_argument("--stage2-artifact-root", required=True, type=Path)
    prepare_parser.add_argument("--parity-data", required=True, type=Path)
    prepare_parser.add_argument("--parity-manifest", required=True, type=Path)
    prepare_parser.add_argument("--output-root", required=True, type=Path)
    prepare_parser.add_argument("--parity-gpus", required=True)
    prepare_parser.add_argument("--python", required=True)
    # Step 64 must not even begin parity work until phase 1's exact receipt
    # exists.  That makes the interleaved phase-two mapping explicit rather
    # than relying on a scheduler dependency hidden outside the evidence.
    prepare_parser.add_argument("--campaign-root", type=Path)
    prepare_parser.add_argument("--step16-root", type=Path)
    prepare_parser.add_argument("--step64-root", type=Path)
    prepare_parser.add_argument("--step16-phase-receipt", type=Path)
    prepare_parser.add_argument("--yes", action="store_true")

    worker_parser = commands.add_parser("worker", help="evaluate one task index on one Slurm-assigned GPU")
    worker_parser.add_argument("--step", required=True, type=int, choices=sorted(APPROVED_TARGETS))
    worker_parser.add_argument("--output-root", required=True, type=Path)
    worker_parser.add_argument("--task-index", required=True, type=int)
    worker_parser.add_argument("--python", required=True)
    worker_parser.add_argument("--phase", required=True, type=int, choices=sorted(PHASES))
    worker_parser.add_argument("--campaign-root", required=True, type=Path)
    worker_parser.add_argument("--step16-root", required=True, type=Path)
    worker_parser.add_argument("--step64-root", required=True, type=Path)

    seal_parser = commands.add_parser("seal-phase", help="seal one exact 14-cell campaign phase")
    seal_parser.add_argument("--phase", required=True, type=int, choices=sorted(PHASES))
    seal_parser.add_argument("--campaign-root", required=True, type=Path)
    seal_parser.add_argument("--step16-root", required=True, type=Path)
    seal_parser.add_argument("--step64-root", required=True, type=Path)

    final_parser = commands.add_parser("finalize", help="require 21 receipts and write native two-bias preflight")
    final_parser.add_argument("--step", required=True, type=int, choices=sorted(APPROVED_TARGETS))
    final_parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if not args.yes:
                parser.error("prepare requires --yes after reviewing the exact step-16/64 target")
            if args.step == 64:
                if not all((args.campaign_root, args.step16_root, args.step64_root, args.step16_phase_receipt)):
                    parser.error("step-64 prepare requires phase-one campaign/root/receipt evidence")
                validate_phase_receipt(
                    args.step16_phase_receipt,
                    phase=1,
                    campaign_root=args.campaign_root,
                    step16_root=args.step16_root,
                    step64_root=args.step64_root,
                )
            result = prepare(
                repository=args.repository,
                step=args.step,
                source_stage2_manifest=args.source_stage2_manifest,
                stage2_artifact_root=args.stage2_artifact_root,
                parity_data=args.parity_data,
                parity_manifest=args.parity_manifest,
                output_root=args.output_root,
                parity_gpus=args.parity_gpus,
                python=args.python,
            )
        elif args.command == "worker":
            result = worker(
                step=args.step,
                output_root=args.output_root,
                task_index=args.task_index,
                python=args.python,
                phase=args.phase,
                campaign_root=args.campaign_root,
                step16_root=args.step16_root,
                step64_root=args.step64_root,
            )
        elif args.command == "seal-phase":
            result = seal_phase(
                phase=args.phase,
                campaign_root=args.campaign_root,
                step16_root=args.step16_root,
                step64_root=args.step64_root,
            )
        elif args.command == "finalize":
            result = finalize(step=args.step, output_root=args.output_root)
        else:  # pragma: no cover - argparse makes this unreachable
            parser.error("unsupported command")
            return 2
    except (EvaluationError, FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "APPROVED_TARGETS",
    "CLEAN_TASK_INDICES",
    "EVALUATION_RECEIPT_SCHEMA",
    "EvaluationError",
    "GENERATION_CONFIG",
    "NATIVE_PREFLIGHT_SCHEMA",
    "PHASES",
    "TASK_COUNT",
    "VLLM_MODEL_ARGS",
    "_parse_four_gpu_tokens",
    "_paths",
    "_task_command",
    "build_launch_contract",
    "finalize",
    "prepare",
    "seal_phase",
    "validate_approved_target",
    "validate_native_preflight",
    "validate_phase_receipt",
    "worker",
]
