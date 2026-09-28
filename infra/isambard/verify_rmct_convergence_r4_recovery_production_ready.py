#!/usr/bin/env python3
"""Capture and verify the fail-closed r4 RMCT recovery readiness chain.

The r4 namespace begins at logical segment four from the sealed r3 s004
checkpoint (optimizer step 64).  It binds failed Slurm job 6031786 as an
immutable provenance record, but never accepts a checkpoint or receipt from
that failed attempt.  The only runtime change from r3 is the physical padded
forward-microbatch cap, reduced from 49,152 to 40,960 tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.rmct_convergence_accelerated import plan as r3
from experiments.rmct_convergence_r4_recovery import plan as r4
from infra.isambard import verify_rmct_convergence_accelerated_production_ready as r3_ready


SCHEMA = "rmct-convergence-gcall-r2-mb40960-r4-production-ready-v1"
SOURCE_READY_SCHEMA = "rmct-convergence-gcall-r2-mb40960-r4-production-source-ready-v1"
CONDITION = r4.CONDITION_NAME
RUN_PREFIX = r4.RUN_PREFIX
PARENT_RUN_PREFIX = r4.PARENT_RUN_PREFIX
PARENT_RUN_NAME = r4.parent_run_name()
PARENT_SEGMENT_INDEX = r4.PARENT_SEGMENT_INDEX
START_SEGMENT_INDEX = r4.START_SEGMENT_INDEX
SOURCE_READY_FILENAME = ".rmct-convergence-gcall-r2-mb40960-r4-production-source-ready"
PLAN_RELATIVE = (
    "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct_convergence_gcall_r2_mb40960_r4_isambard_20260818.yaml"
)
FAILURE_PARENT_ARTIFACT_RELATIVE = r4.FAILURE_PARENT_ARTIFACT
FAILURE_PARENT_ARTIFACT_SHA256 = r4.FAILURE_PARENT_ARTIFACT_SHA256
PREFLIGHT_RESULT_RELATIVE = r4.PREFLIGHT_RESULT_ARTIFACT
PREFLIGHT_RESULT_SHA256 = r4.PREFLIGHT_RESULT_ARTIFACT_SHA256
PREFLIGHT_CONTRACT_RELATIVE = r4.PREFLIGHT_CONTRACT_ARTIFACT
PREFLIGHT_CONTRACT_SHA256 = r4.PREFLIGHT_CONTRACT_ARTIFACT_SHA256
PARENT_RUN_RELATIVE = f"logs/{CONDITION}/{PARENT_RUN_NAME}"
PARENT_CHECKPOINT_RELATIVE = f"{PARENT_RUN_RELATIVE}/checkpoints/{CONDITION}_{PARENT_RUN_NAME}"
PARENT_CHECKPOINT_RECEIPT_RELATIVE = f"{PARENT_RUN_RELATIVE}/segment/checkpoint-receipt.json"
PARENT_COMPLETION_RECEIPT_RELATIVE = f"{PARENT_RUN_RELATIVE}/segment/completion-receipt.json"
PARENT_R3_READY_RECEIPT_RELATIVE = "artifacts/rmct-convergence-gcall-r2-mb49152-r3-production-ready.json"
DATA_RELATIVE = r4.DATA_PATH
MANIFEST_RELATIVE = r4.MANIFEST_PATH
DATA_SHA256 = r4.DATA_SHA256
MANIFEST_SHA256 = r4.MANIFEST_SHA256
MODEL = r4.MODEL
BASE_SNAPSHOT = r4.BASE_SNAPSHOT
TOPOLOGY_PROFILE = r4.TOPOLOGY_PROFILE
WORKER_PARITY_DIRECTORY_RELATIVE = r3_ready.WORKER_PARITY_DIRECTORY_RELATIVE
RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS = r4.RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS
PARENT_FORWARD_MICROBATCH_MAX_TOKENS = r3.ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS
FAILED_ATTEMPT_EVIDENCE = r4.FAILED_ATTEMPT_EVIDENCE
REPLICATED_PROTOCOL_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

# Bind the r3 implementation as well as every r4-specific file.  A source
# readiness capture must become stale if either the inherited resume path or
# the fresh recovery path changes.
CRITICAL_SOURCES = tuple(
    dict.fromkeys(
        (*r3_ready.CRITICAL_SOURCES,
         "experiments/rmct_convergence_r4_recovery/__init__.py",
         "experiments/rmct_convergence_r4_recovery/plan.py",
         "infra/isambard/verify_rmct_convergence_r4_recovery_production_ready.py",
         "infra/isambard/run_qwen35_rmct_convergence_r4_recovery_deadline.sh",
         "infra/isambard/run_qwen35_rmct_convergence_r4_recovery_segment.sbatch")
    )
)


class ReadyError(ValueError):
    """An r4 recovery receipt is incomplete, mutable, or unsafe."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ReadyError(f"repository must be a regular directory: {root}")
    return root


def _under_root(root: Path, relative: str, *, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"{label} escapes repository root: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ReadyError(f"{label} must be a regular file: {path}")
    return path


def _identity(root: Path, relative: str, *, label: str) -> dict[str, Any]:
    path = _under_root(root, relative, label=label)
    return {"path": relative, "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _identity_path(root: Path, path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(root))
    except ValueError as exc:
        raise ReadyError(f"{label} escapes repository root: {resolved}") from exc
    return _identity(root, relative, label=label)


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReadyError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadyError(f"{label} is not JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReadyError(f"{label} must be an object")
    return value


def _failed_attempt_evidence_identity(root: Path) -> dict[str, dict[str, Any]]:
    """Require the exact remote evidence for the failed r3 s005 attempt.

    The failure/parent artifact alone is a policy record.  This check makes
    the record auditable by hash-binding the actual Slurm output and every
    recorded r3 s005 command/start/metric/configuration artifact before a
    source-ready or final-ready receipt can exist.
    """

    if not isinstance(FAILED_ATTEMPT_EVIDENCE, Mapping) or not FAILED_ATTEMPT_EVIDENCE:
        raise ReadyError("r4 failed-attempt evidence contract is malformed")
    identities: dict[str, dict[str, Any]] = {}
    for name, expected in FAILED_ATTEMPT_EVIDENCE.items():
        if not isinstance(name, str) or not isinstance(expected, Mapping) or set(expected) != {"path", "sha256"}:
            raise ReadyError("r4 failed-attempt evidence contract has an invalid entry")
        relative = expected.get("path")
        expected_sha = expected.get("sha256")
        if not isinstance(relative, str) or not relative or not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None:
            raise ReadyError("r4 failed-attempt evidence contract has an invalid path or SHA-256")
        identity = _identity(root, relative, label=f"failed job 6031786 {name} evidence")
        if identity["sha256"] != expected_sha:
            raise ReadyError(f"failed job 6031786 {name} evidence differs from the fixed custody hash")
        identities[name] = identity
    return identities


def _failure_parent_artifact_identity(root: Path) -> dict[str, Any]:
    """Bind the immutable failed-attempt record and chosen sealed parent."""

    identity = _identity(root, FAILURE_PARENT_ARTIFACT_RELATIVE, label="r4 failure/parent artifact")
    if identity["sha256"] != FAILURE_PARENT_ARTIFACT_SHA256:
        raise ReadyError("r4 failure/parent artifact differs from its fixed custody record")
    document = _json(root / FAILURE_PARENT_ARTIFACT_RELATIVE, label="r4 failure/parent artifact")
    parent = document.get("parent")
    failed = document.get("failed_attempt_not_used_as_state")
    failed_evidence = document.get("failed_attempt_evidence")
    evidence = document.get("preflight_evidence")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(failed, Mapping)
        or not isinstance(failed_evidence, Mapping)
        or not isinstance(evidence, Mapping)
    ):
        raise ReadyError("r4 failure/parent artifact has malformed parent, failed-attempt, failure-evidence, or preflight records")
    expected_parent = {
        "checkpoint_relative_path": PARENT_CHECKPOINT_RELATIVE,
        "completion_receipt_relative_path": PARENT_COMPLETION_RECEIPT_RELATIVE,
        "condition": CONDITION,
        "decision_directory_relative_path": f"{PARENT_RUN_RELATIVE}/decisions",
        "expected_final": True,
        "expected_kind": "both",
        "optimizer_step": START_SEGMENT_INDEX * r4.UPDATES_PER_SEGMENT,
        "run_name": PARENT_RUN_NAME,
        "run_prefix": PARENT_RUN_PREFIX,
        "segment_index": PARENT_SEGMENT_INDEX,
    }
    expected_failed = r4.CONTINUATION_METADATA["failed_attempt_not_used_as_state"]
    expected_evidence = {
        "hardware_scope": "same-GH200",
        "selected_validated_local_forward_microbatch_max_tokens": RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS,
        "highest_validated_local_forward_microbatch_max_tokens": PARENT_FORWARD_MICROBATCH_MAX_TOKENS,
        "all_gradient_checkpointing_layers": "all",
        "result": {"path": PREFLIGHT_RESULT_RELATIVE, "sha256": PREFLIGHT_RESULT_SHA256},
        "contract": {"path": PREFLIGHT_CONTRACT_RELATIVE, "sha256": PREFLIGHT_CONTRACT_SHA256},
        "result_sha256": PREFLIGHT_RESULT_SHA256,
        "contract_sha256": PREFLIGHT_CONTRACT_SHA256,
    }
    if (
        document.get("schema") != "rmct-convergence-r4-recovery-failure-parent-v2"
        or document.get("condition") != CONDITION
        or document.get("run_prefix") != RUN_PREFIX
        or document.get("logical_start_segment_index") != START_SEGMENT_INDEX
        or dict(parent) != expected_parent
        or dict(failed) != expected_failed
        or dict(failed_evidence) != dict(FAILED_ATTEMPT_EVIDENCE)
        or dict(evidence) != expected_evidence
        or document.get("execution_delta")
        != {
            "local_forward_microbatch_max_tokens": {
                "from": PARENT_FORWARD_MICROBATCH_MAX_TOKENS,
                "to": RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS,
            },
            "scientific_contract_unchanged": True,
        }
    ):
        raise ReadyError("r4 failure/parent artifact does not bind job 6031786 evidence, r3 s004, and the sole cap reduction")
    return identity


def _preflight_evidence_identity(root: Path) -> dict[str, Any]:
    """Replay the r3 same-GH200 evidence specifically at the 40,960 cap."""

    result_identity = _identity(root, PREFLIGHT_RESULT_RELATIVE, label="r4 inherited same-GH200 preflight result")
    contract_identity = _identity(root, PREFLIGHT_CONTRACT_RELATIVE, label="r4 inherited same-GH200 preflight contract")
    if result_identity["sha256"] != PREFLIGHT_RESULT_SHA256 or contract_identity["sha256"] != PREFLIGHT_CONTRACT_SHA256:
        raise ReadyError("r4 inherited same-GH200 preflight evidence differs from its fixed hashes")
    result = _json(root / PREFLIGHT_RESULT_RELATIVE, label="r4 inherited same-GH200 preflight result")
    contract = _json(root / PREFLIGHT_CONTRACT_RELATIVE, label="r4 inherited same-GH200 preflight contract")
    config = contract.get("config")
    resolved = contract.get("resolved")
    if not isinstance(config, Mapping) or not isinstance(resolved, Mapping):
        raise ReadyError("r4 inherited same-GH200 preflight contract lacks config/resolved objects")
    if (
        contract.get("schema") != "qwen35-phase-shared-preflight-v1"
        or contract.get("kind") != "non_production_phase_shared_preflight_contract"
        or contract.get("non_production") is not True
        or contract.get("production_output_touched") is not False
        or config.get("model") != MODEL
        or config.get("forward_microbatch_max_datums") != 8
        or config.get("target_logprob_chunk_size") != 2048
        or config.get("packing_budgets") != [20480, 40960, 49152]
        or resolved.get("gradient_checkpoint_layers") != "all"
        or resolved.get("world_size") != 4
        or resolved.get("visible_devices") != ["0", "1", "2", "3"]
    ):
        raise ReadyError("same-GH200 preflight contract does not attest the r4 all-GC 40,960-token topology")
    for label in ("training_gpus", "rollout_gpus"):
        values = resolved.get(label)
        if not isinstance(values, list) or [item.get("logical_index") if isinstance(item, Mapping) else None for item in values] != [0, 1, 2, 3]:
            raise ReadyError(f"same-GH200 preflight contract has unsafe {label} topology")
    phase_shared = result.get("phase_shared")
    if result.get("schema") != "qwen35-phase-shared-preflight-v1" or result.get("passed") is not True or not isinstance(phase_shared, Mapping):
        raise ReadyError("same-GH200 preflight result is not a passed phase-shared receipt")
    sweep = phase_shared.get("rank_zero_packing_sweep")
    if not isinstance(sweep, list):
        raise ReadyError("same-GH200 preflight result lacks the rank-zero packing sweep")
    matching = [
        item for item in sweep
        if isinstance(item, Mapping)
        and item.get("packing_budget") == RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS
        and item.get("padded_token_slots") == RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS
        and item.get("passed") is True
        and item.get("scope") == "rank_zero_capacity_only_with_all_vllm_workers_asleep"
    ]
    if len(matching) != 1:
        raise ReadyError("same-GH200 preflight result does not contain exactly one passed 40,960-token all-workers-asleep probe")
    memory = phase_shared.get("packing_workers_asleep_memory")
    rows = memory.get("nvidia_smi_rows") if isinstance(memory, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 4 or any(not isinstance(row, str) or "NVIDIA GH200" not in row for row in rows):
        raise ReadyError("same-GH200 preflight result does not attest four GH200 devices")
    for parity_key in ("fixed_update_loss_parity", "fixed_update_parameter_parity", "fixed_update_pre_optimizer_gradient_parity"):
        record = phase_shared.get(parity_key)
        passed = record.get("passed") if isinstance(record, Mapping) else None
        if parity_key == "fixed_update_pre_optimizer_gradient_parity" and isinstance(record, Mapping):
            gate = record.get("gate")
            passed = gate.get("passed") if isinstance(gate, Mapping) else None
        if passed is not True:
            raise ReadyError(f"same-GH200 preflight result failed required {parity_key}")
    return {
        "result": result_identity,
        "contract": contract_identity,
        "semantics": {
            "hardware_scope": "same-GH200",
            "gpu_count": 4,
            "gradient_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS,
            "local_target_logprob_chunk_size": 2048,
            "passed": True,
        },
    }


def _snapshot_identity() -> dict[str, Any]:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise ReadyError("HF_HOME must be set to capture or verify the pinned offline snapshot")
    snapshot = Path(hf_home).expanduser().resolve() / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / BASE_SNAPSHOT
    if snapshot.is_symlink() or not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise ReadyError(f"pinned Qwen snapshot is absent or incomplete: {snapshot}")
    return {
        "repo_id": MODEL,
        "revision": BASE_SNAPSHOT,
        "snapshot_path": str(snapshot),
        "config_sha256": _sha256(snapshot / "config.json"),
        "offline_only": True,
    }


def _strict_checkpoint_identity(
    root: Path, checkpoint: Path, *, expected_segment_index: int, label: str
) -> dict[str, Any]:
    """Validate one complete strict four-rank final local-RL checkpoint.

    ``boundary seal`` validates the local loop state, but a protected r4
    parent/recovery needs the full replicated optimizer and RNG bundle as
    well.  This helper makes that bundle explicit and returns file identities
    for inclusion in readiness/custody receipts.
    """

    from ctm.training.resume_state import load_strict_local_rl_resume_state

    if isinstance(expected_segment_index, bool) or not isinstance(expected_segment_index, int) or expected_segment_index < 0:
        raise ReadyError(f"{label} expected segment index is invalid: {expected_segment_index!r}")
    raw_checkpoint = Path(checkpoint)
    if raw_checkpoint.is_symlink() or not raw_checkpoint.is_dir():
        raise ReadyError(f"{label} checkpoint must be a regular directory: {raw_checkpoint}")
    checkpoint = raw_checkpoint.resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"{label} checkpoint escapes repository: {checkpoint}") from exc
    try:
        state = load_strict_local_rl_resume_state(checkpoint)
    except Exception as exc:
        raise ReadyError(f"{label} checkpoint does not satisfy strict local-RL resume validation: {exc}") from exc
    expected_step = (expected_segment_index + 1) * r4.UPDATES_PER_SEGMENT
    if state.global_step != expected_step or state.optimizer_step != expected_step:
        raise ReadyError(
            f"{label} checkpoint has inconsistent final steps: global={state.global_step}, "
            f"optimizer={state.optimizer_step}, expected={expected_step}"
        )
    required_files = {
        "adapter_config": "adapter_config.json",
        "adapter_model": "adapter_model.safetensors",
        "optimizer": "optimizer.pt",
        "manifest": "manifest.json",
        "replicated_training_manifest": "replicated_training_manifest.json",
        "replicated_training_rng": "replicated_training_rng.pt",
    }
    files: dict[str, dict[str, Any]] = {}
    for name, filename in required_files.items():
        path = checkpoint / filename
        if path.is_symlink() or not path.is_file():
            raise ReadyError(f"{label} checkpoint requires regular {name} file: {path}")
        identity = _identity_path(root, path, label=f"{label} checkpoint {name}")
        if identity["size_bytes"] <= 0:
            raise ReadyError(f"{label} checkpoint {name} file is empty")
        files[name] = identity
    manifest = _json(checkpoint / "manifest.json", label=f"{label} checkpoint manifest")
    loop = manifest.get("loop_state")
    if (
        manifest.get("backend") != "local"
        or manifest.get("kind") != "both"
        or not isinstance(loop, Mapping)
        or loop.get("global_step") != expected_step
        or loop.get("optimizer_step") != expected_step
        or loop.get("step") != expected_step
        or loop.get("completed_epochs") != state.completed_epochs
        or loop.get("accumulated_grads") != 0
        or loop.get("final") is not True
    ):
        raise ReadyError(f"{label} checkpoint manifest is not a final local kind='both' step-{expected_step} boundary")
    replicated = _json(
        checkpoint / "replicated_training_manifest.json", label=f"{label} replicated-training manifest"
    )
    state_hash = replicated.get("state_hash")
    timeout = replicated.get("optimizer_timeout_seconds")
    if (
        replicated.get("schema") != REPLICATED_PROTOCOL_VERSION
        or replicated.get("checkpoint_kind") != "both"
        or replicated.get("world_size") != 4
        or replicated.get("train_logical_indices") != [0, 1, 2, 3]
        or replicated.get("process_group_backend") != "nccl"
        or replicated.get("device_type") != "cuda"
        or replicated.get("rng_state_file") != "replicated_training_rng.pt"
        or not isinstance(state_hash, str)
        or _SHA256.fullmatch(state_hash) is None
        or isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
    ):
        raise ReadyError(f"{label} replicated-training manifest is not the strict four-GPU all-rank optimizer boundary")
    return {
        "files": files,
        "global_step": state.global_step,
        "optimizer_step": state.optimizer_step,
        "completed_epochs": state.completed_epochs,
        "kind": "both",
        "final": True,
        "manifest": {
            "backend": manifest["backend"],
            "kind": manifest["kind"],
            "loop_state": {
                "global_step": loop["global_step"],
                "optimizer_step": loop["optimizer_step"],
                "step": loop["step"],
                "completed_epochs": loop["completed_epochs"],
                "accumulated_grads": loop["accumulated_grads"],
                "final": loop["final"],
            },
        },
        "replicated_training": {
            "schema": replicated["schema"],
            "checkpoint_kind": replicated["checkpoint_kind"],
            "world_size": replicated["world_size"],
            "train_logical_indices": replicated["train_logical_indices"],
            "process_group_backend": replicated["process_group_backend"],
            "device_type": replicated["device_type"],
            "state_hash": state_hash,
            "rng_state_file": replicated["rng_state_file"],
        },
    }


def _sealed_parent_identity(root: Path) -> dict[str, Any]:
    """Require the exact sealed r3 s004 boundary, state, and continuation."""

    sys.path.insert(0, str(root))
    from experiments.rmct_convergence import controller

    r3_ready_path = _under_root(root, PARENT_R3_READY_RECEIPT_RELATIVE, label="r3 parent readiness receipt")
    try:
        r3_custody = r3_ready.verify_sealed_segment_custody(
            root, segment_index=PARENT_SEGMENT_INDEX, ready_receipt=r3_ready_path
        )
    except Exception as exc:
        raise ReadyError(f"r4 parent does not satisfy r3 sealed-custody verification: {exc}") from exc
    checkpoint = (root / PARENT_CHECKPOINT_RELATIVE).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"r4 parent checkpoint escapes repository: {checkpoint}") from exc
    checkpoint_identity = _strict_checkpoint_identity(
        root,
        checkpoint,
        expected_segment_index=PARENT_SEGMENT_INDEX,
        label="r4 parent r3 s004",
    )
    checkpoint_receipt = _identity(root, PARENT_CHECKPOINT_RECEIPT_RELATIVE, label="r4 parent checkpoint receipt")
    completion_receipt = _identity(root, PARENT_COMPLETION_RECEIPT_RELATIVE, label="r4 parent completion receipt")
    decisions = (root / PARENT_RUN_RELATIVE / "decisions").resolve()
    try:
        decisions.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"r4 parent decisions escape repository: {decisions}") from exc
    if decisions.is_symlink() or not decisions.is_dir():
        raise ReadyError("r4 parent decisions directory is absent or linked")
    candidates = sorted(decisions.glob(f"checkpoint-window-decision-s{PARENT_SEGMENT_INDEX:03d}-*.json"))
    if len(candidates) != 1 or candidates[0].is_symlink() or not candidates[0].is_file():
        raise ReadyError(f"r4 parent requires exactly one r3 s004 decision receipt; found {len(candidates)}")
    try:
        controller.require_continue(candidates[0])
    except Exception as exc:
        raise ReadyError(f"r4 parent r3 s004 decision is not a verified continue receipt: {exc}") from exc
    return {
        "condition": CONDITION,
        "run_prefix": PARENT_RUN_PREFIX,
        "run_name": PARENT_RUN_NAME,
        "logical_segment_index": PARENT_SEGMENT_INDEX,
        "checkpoint_relative_path": PARENT_CHECKPOINT_RELATIVE,
        "global_step": checkpoint_identity["global_step"],
        "optimizer_step": checkpoint_identity["optimizer_step"],
        "checkpoint": checkpoint_identity,
        "r3_readiness_receipt": _identity(root, PARENT_R3_READY_RECEIPT_RELATIVE, label="r3 parent readiness receipt"),
        "r3_sealed_custody": r3_custody,
        "checkpoint_receipt": checkpoint_receipt,
        "completion_receipt": completion_receipt,
        "continue_decision": _identity_path(root, candidates[0], label="r4 parent continue decision"),
    }


def _version(module: str) -> str | None:
    try:
        loaded = importlib.import_module(module)
    except Exception:
        return None
    value = getattr(loaded, "__version__", None)
    return str(value) if value is not None else None


def _compiled_contract(root: Path) -> dict[str, Any]:
    from scripts import run_experiment

    compiled = run_experiment.load_experiment(root / PLAN_RELATIVE, topology_profile=TOPOLOGY_PROFILE)
    convergence = compiled.get("rmct_convergence")
    if not isinstance(convergence, Mapping):
        raise ReadyError("compiled r4 recovery lacks rmct_convergence custody block")
    if convergence.get("schema") != r4.CONTINUATION_SCHEMA or convergence.get("condition") != CONDITION:
        raise ReadyError("compiled r4 recovery has an unexpected schema or condition")
    if convergence.get("run_prefix") != RUN_PREFIX or convergence.get("logical_start_segment_index") != START_SEGMENT_INDEX:
        raise ReadyError("compiled r4 recovery has an unsafe run namespace/start index")
    if convergence.get("continuation") != r4.CONTINUATION_METADATA:
        raise ReadyError("compiled r4 recovery does not bind reviewed failure/parent/preflight metadata")
    if convergence.get("data", {}).get("content_sha256") != DATA_SHA256 or convergence.get("data", {}).get("manifest_sha256") != MANIFEST_SHA256:
        raise ReadyError("compiled r4 recovery does not bind frozen shared-QID data")
    if convergence.get("convergence", {}).get("hard_cap_optimizer_steps") != 512:
        raise ReadyError("compiled r4 recovery hard cap must be 512 optimizer steps")
    entries = compiled.get("training")
    if not isinstance(entries, list) or len(entries) != r4.TOTAL_SEGMENTS - START_SEGMENT_INDEX:
        raise ReadyError("compiled r4 recovery must expose only logical segments 4..31")
    first = entries[0].get("args")
    if not isinstance(first, Mapping):
        raise ReadyError("compiled r4 recovery segment four lacks an argument map")
    if first.get("load_config") != {"n_datapoints": 32, "segment_index": START_SEGMENT_INDEX}:
        raise ReadyError("compiled r4 recovery must start from logical segment four")
    if first.get("run_name") != f"{RUN_PREFIX}-s005" or first.get("resume_from") != f"file://{root / PARENT_CHECKPOINT_RELATIVE}":
        raise ReadyError("compiled r4 recovery does not bind r3 s004 as its first parent")
    if (
        first.get("local_forward_microbatch_max_datums") != 8
        or first.get("local_forward_microbatch_max_tokens") != RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS
        or first.get("local_target_logprob_chunk_size") != 2048
        or first.get("local_gradient_checkpointing") is not True
        or "local_gradient_checkpointing_layers" in first
    ):
        raise ReadyError("compiled r4 recovery physical microbatch contract drifted")
    r3_args = r3.segment_args(root, START_SEGMENT_INDEX)
    if set(first) != set(r3_args):
        raise ReadyError("compiled r4 recovery command keys differ from the inherited r3 command")
    delta = {key: (r3_args[key], first[key]) for key in r3_args if r3_args[key] != first[key]}
    if delta != {
        "run_name": (r3.run_name(START_SEGMENT_INDEX), r4.run_name(START_SEGMENT_INDEX)),
        "local_forward_microbatch_max_tokens": (PARENT_FORWARD_MICROBATCH_MAX_TOKENS, RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS),
    }:
        raise ReadyError("compiled r4 recovery changes more than namespace and the 40,960-token physical cap")
    return {
        "compiler_module": "experiments.rmct_convergence_r4_recovery.plan",
        "compiler_source_sha256": _sha256(root / "experiments/rmct_convergence_r4_recovery/plan.py"),
        "compiled_plan_sha256": hashlib.sha256(_canonical_json(compiled)).hexdigest(),
        "frozen_hyperparameters": {
            "run_prefix": convergence["run_prefix"],
            "optimizer": convergence["optimizer"],
            "method": convergence["method"],
            "sampling": convergence["sampling"],
            "loop": convergence["loop"],
            "convergence": convergence["convergence"],
            "topology": convergence["topology"],
            "continuation": convergence["continuation"],
        },
        "logical_segment_four_args_sha256": hashlib.sha256(_canonical_json(first)).hexdigest(),
        "logical_segment_five_args_sha256": hashlib.sha256(_canonical_json(entries[1]["args"])).hexdigest(),
        "model_constants": {"model": MODEL, "base_snapshot": BASE_SNAPSHOT},
    }


def _runtime() -> dict[str, Any]:
    return {
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "platform": {"machine": platform.machine(), "system": platform.system()},
        "modules": {"torch": _version("torch"), "transformers": _version("transformers"), "vllm": _version("vllm")},
        "environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "CUDA_VISIBLE_DEVICES_count": len([part for part in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if part]),
        },
    }


def build_source_ready_receipt(repository: str | Path) -> dict[str, Any]:
    root = _root(repository)
    plan = _identity(root, PLAN_RELATIVE, label="r4 RMCT recovery YAML")
    data = _identity(root, DATA_RELATIVE, label="shared-QID data")
    manifest = _identity(root, MANIFEST_RELATIVE, label="shared-QID manifest")
    parent_artifact = _failure_parent_artifact_identity(root)
    failed_attempt_evidence = _failed_attempt_evidence_identity(root)
    preflight_evidence = _preflight_evidence_identity(root)
    if data["sha256"] != DATA_SHA256 or manifest["sha256"] != MANIFEST_SHA256:
        raise ReadyError("frozen shared-QID data or manifest differs from the r4 recovery contract")
    source = {relative: _identity(root, relative, label="critical r4 recovery source") for relative in CRITICAL_SOURCES}
    return {
        "schema": SOURCE_READY_SCHEMA,
        "condition": CONDITION,
        "run_prefix": RUN_PREFIX,
        "source_ready": True,
        "plan": plan,
        "data": {"data": data, "manifest": manifest},
        "failure_parent_artifact": parent_artifact,
        "failed_attempt_evidence": failed_attempt_evidence,
        "same_hardware_preflight": preflight_evidence,
        "critical_sources": source,
        "execution_disclosure": {
            "topology_profile": TOPOLOGY_PROFILE,
            "run_prefix": RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS,
            "local_target_logprob_chunk_size": 2048,
            "same_hardware_all_gc_preflight_validated": True,
            "failed_job_6031786_not_used_as_state": True,
        },
    }


def verify_source_ready_receipt(repository: str | Path, receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    recorded = _json(Path(receipt).resolve(), label="r4 recovery source-ready receipt")
    if (
        recorded.get("schema") != SOURCE_READY_SCHEMA
        or recorded.get("condition") != CONDITION
        or recorded.get("run_prefix") != RUN_PREFIX
        or recorded.get("source_ready") is not True
    ):
        raise ReadyError("r4 recovery source-ready receipt has an unexpected schema/condition/namespace")
    expected = build_source_ready_receipt(root)
    if recorded != expected:
        raise ReadyError("r4 recovery source-ready receipt does not match this synchronized checkout")
    return recorded


def build_receipt(repository: str | Path) -> dict[str, Any]:
    root = _root(repository)
    source_ready = build_source_ready_receipt(root)
    snapshot = _snapshot_identity()
    return {
        "schema": SCHEMA,
        "condition": CONDITION,
        "run_prefix": RUN_PREFIX,
        "ready": True,
        "source_ready": source_ready,
        "plan": source_ready["plan"],
        "data": source_ready["data"],
        "failure_parent_artifact": source_ready["failure_parent_artifact"],
        "failed_attempt_evidence": source_ready["failed_attempt_evidence"],
        "same_hardware_preflight": source_ready["same_hardware_preflight"],
        "sealed_continuation_parent": _sealed_parent_identity(root),
        "base_snapshot": snapshot,
        "worker_parity": r3_ready._worker_parity_identity(root, snapshot=snapshot),
        "critical_sources": source_ready["critical_sources"],
        "compiled_contract": _compiled_contract(root),
        "runtime": _runtime(),
        "execution_disclosure": {
            "topology_profile": TOPOLOGY_PROFILE,
            "run_prefix": RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS,
            "local_target_logprob_chunk_size": 2048,
            "gpu_count": 4,
            "training_gpus": "all",
            "rollout_gpus": "all",
            "phase_shared": True,
            "same_hardware_all_gc_preflight_validated": True,
            "failed_job_6031786_not_used_as_state": True,
        },
    }


def _write_immutable(path: Path, document: Mapping[str, Any]) -> str:
    payload = _canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ReadyError(f"receipt parent must be a regular directory: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ReadyError(f"refusing to overwrite different immutable r4 recovery receipt: {path}")
        return "resumed"
    return "written"


def verify_receipt(repository: str | Path, receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    recorded = _json(Path(receipt).resolve(), label="r4 recovery readiness receipt")
    if (
        recorded.get("schema") != SCHEMA
        or recorded.get("condition") != CONDITION
        or recorded.get("run_prefix") != RUN_PREFIX
        or recorded.get("ready") is not True
    ):
        raise ReadyError("r4 recovery readiness receipt has an unexpected schema/condition/namespace")
    expected = build_receipt(root)
    if recorded != expected:
        raise ReadyError("r4 recovery readiness receipt does not match this checkout, r3 parent, or runtime")
    return recorded


def _receipt_under_root(root: Path, value: str | Path, *, label: str) -> Path:
    raw = Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise ReadyError(f"{label} must be an absolute regular file: {raw}")
    path = raw.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"{label} must remain inside the synchronized repository: {path}") from exc
    return path


def _marker_binding(value: Any, *, path: Path, sha256: str, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ReadyError(f"{label} must be one exact path/SHA-256 binding")
    if value.get("path") != str(path.resolve()) or value.get("sha256") != sha256:
        raise ReadyError(f"{label} does not bind the expected immutable evidence")


def verify_sealed_segment_custody(
    repository: str | Path, *, segment_index: int, ready_receipt: str | Path
) -> dict[str, Any]:
    """Fail closed before sealing an existing r4 checkpoint after a crash."""

    root = _root(repository)
    if isinstance(segment_index, bool) or not isinstance(segment_index, int) or not START_SEGMENT_INDEX <= segment_index < r4.TOTAL_SEGMENTS:
        raise ReadyError(f"sealed r4 recovery segment index must be in [{START_SEGMENT_INDEX}, {r4.TOTAL_SEGMENTS - 1}]")
    ready_path = _receipt_under_root(root, ready_receipt, label="r4 recovery readiness receipt")
    ready = verify_receipt(root, ready_path)
    base_snapshot = ready.get("base_snapshot")
    if not isinstance(base_snapshot, Mapping):
        raise ReadyError("verified r4 recovery readiness lacks its base snapshot")
    snapshot_raw = base_snapshot.get("snapshot_path")
    if not isinstance(snapshot_raw, str) or not snapshot_raw:
        raise ReadyError("verified r4 recovery readiness lacks a pinned snapshot path")
    snapshot = Path(snapshot_raw).resolve()
    if (
        not Path(snapshot_raw).is_absolute()
        or Path(snapshot_raw).is_symlink()
        or not snapshot.is_dir()
        or snapshot.name != BASE_SNAPSHOT
        or not (snapshot / "config.json").is_file()
        or base_snapshot.get("repo_id") != MODEL
        or base_snapshot.get("revision") != BASE_SNAPSHOT
    ):
        raise ReadyError("verified r4 recovery readiness has an invalid pinned Qwen snapshot")
    run = r4.run_name(segment_index)
    segment_directory = root / "logs" / CONDITION / run / "segment"
    command_path = segment_directory / "training-command.json"
    marker_path = segment_directory / "training-started.json"
    command_identity = _identity_path(root, command_path, label="r4 recovery training command")
    marker_identity = _identity_path(root, marker_path, label="r4 recovery training-started marker")
    command = _json(command_path, label="r4 recovery training command")
    required_command = {"schema", "condition", "run_prefix", "logical_segment_index", "plan", "model", "segment", "argv", "environment_contract"}
    if set(command) != required_command:
        raise ReadyError("r4 recovery training command has an unexpected envelope")
    from scripts.run_experiment import _argument_tokens

    plan_path = _under_root(root, PLAN_RELATIVE, label="r4 recovery YAML")
    expected_plan = {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)}
    expected_model = {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "snapshot_path": str(snapshot)}
    expected_segment = r4.segment_record(root, segment_index)
    expected_args = r4.segment_args(root, segment_index, model_path=snapshot)
    argv = command.get("argv")
    if (
        command.get("schema") != r4.COMMAND_SCHEMA
        or command.get("condition") != CONDITION
        or command.get("run_prefix") != RUN_PREFIX
        or command.get("logical_segment_index") != segment_index
        or command.get("plan") != expected_plan
        or command.get("model") != expected_model
        or command.get("segment") != expected_segment
        or not isinstance(argv, list)
        or len(argv) < 2
        or not all(isinstance(token, str) and token for token in argv)
        or not Path(argv[0]).name.startswith("python")
        or argv[1] != str((root / "scripts" / "train_rlct.py").resolve())
        or argv[2:] != _argument_tokens(expected_args)
    ):
        raise ReadyError("r4 recovery training command differs from the full expected r4 segment argv")
    expected_environment = {
        "cuda_visible_devices_preserved": True,
        "topology_profile": TOPOLOGY_PROFILE,
        "phase_shared": True,
        "gradient_checkpointing_layers": "all",
        "local_forward_microbatch_max_datums": 8,
        "local_forward_microbatch_max_tokens": RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS,
        "local_target_logprob_chunk_size": 2048,
    }
    if command.get("environment_contract") != expected_environment:
        raise ReadyError("r4 recovery training command has an unsafe execution contract")
    marker = _json(marker_path, label="r4 recovery training-started marker")
    if (
        set(marker) != {"schema", "segment_index", "command_attestation", "ready_receipt"}
        or marker.get("schema") != "rmct-convergence-training-started-v1"
        or marker.get("segment_index") != segment_index
    ):
        raise ReadyError("r4 recovery training-started marker is not the expected logical segment")
    _marker_binding(marker.get("command_attestation"), path=command_path, sha256=command_identity["sha256"], label="r4 marker command")
    _marker_binding(marker.get("ready_receipt"), path=ready_path, sha256=_sha256(ready_path), label="r4 marker readiness")
    checkpoint = r4.final_checkpoint_path(root, segment_index)
    checkpoint_identity = _strict_checkpoint_identity(
        root,
        checkpoint,
        expected_segment_index=segment_index,
        label=f"r4 recovery s{segment_index + 1:03d}",
    )
    return {
        "segment_index": segment_index,
        "run_name": run,
        "command": command_identity,
        "training_started": marker_identity,
        "ready_receipt": _identity_path(root, ready_path, label="r4 recovery readiness receipt"),
        "checkpoint_relative_path": str(checkpoint.relative_to(root)),
        "checkpoint": checkpoint_identity,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("capture-source-ready", "write immutable source/data r4 recovery readiness"),
        ("verify-source-ready", "verify immutable source/data r4 recovery readiness"),
        ("capture", "write immutable final r4 recovery readiness"),
        ("verify", "verify immutable final r4 recovery readiness"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repository", required=True, type=Path)
        command.add_argument("--output" if name.startswith("capture") else "--receipt", required=True, type=Path)
    custody = commands.add_parser("verify-sealed-segment-custody", help="validate r4 marker/command/readiness custody before sealing an existing checkpoint")
    custody.add_argument("--repository", required=True, type=Path)
    custody.add_argument("--ready-receipt", required=True, type=Path)
    custody.add_argument("--segment-index", required=True, type=int)
    parser.add_argument("--repository", dest="default_repository", type=Path)
    parser.add_argument("--receipt", dest="default_receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "capture-source-ready":
            document = build_source_ready_receipt(args.repository)
            status = _write_immutable(args.output.resolve(), document)
            print(json.dumps({"path": str(args.output.resolve()), "status": status}, sort_keys=True))
            return 0
        if args.command == "verify-source-ready":
            verify_source_ready_receipt(args.repository, args.receipt)
            print(json.dumps({"path": str(args.receipt.resolve()), "status": "verified"}, sort_keys=True))
            return 0
        if args.command == "capture":
            document = build_receipt(args.repository)
            status = _write_immutable(args.output.resolve(), document)
            print(json.dumps({"path": str(args.output.resolve()), "status": status}, sort_keys=True))
            return 0
        if args.command == "verify":
            verify_receipt(args.repository, args.receipt)
            print(json.dumps({"path": str(args.receipt.resolve()), "status": "verified"}, sort_keys=True))
            return 0
        if args.command == "verify-sealed-segment-custody":
            identity = verify_sealed_segment_custody(args.repository, segment_index=args.segment_index, ready_receipt=args.ready_receipt)
            print(json.dumps({"status": "verified", **identity}, sort_keys=True))
            return 0
        if args.default_repository is not None and args.default_receipt is not None:
            verify_receipt(args.default_repository, args.default_receipt)
            print(json.dumps({"path": str(args.default_receipt.resolve()), "status": "verified"}, sort_keys=True))
            return 0
        _parser().error("use capture/verify, or supply --repository and --receipt")
    except (OSError, ReadyError, ValueError) as exc:
        _parser().error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
