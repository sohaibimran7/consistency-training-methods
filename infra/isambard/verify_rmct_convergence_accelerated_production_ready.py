#!/usr/bin/env python3
"""Capture/verify readiness for the r3 49,152-token RMCT continuation.

The source receipt binds the frozen continuation plan and parent-custody
artifact before an allocation.  The final receipt additionally proves that
the sealed gcall-r2 s001 checkpoint and its ``continue`` controller decision
still exist, so r3 cannot replay or mutate the current segment zero.
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


SCHEMA = "rmct-convergence-gcall-r2-mb49152-r3-production-ready-v1"
SOURCE_READY_SCHEMA = "rmct-convergence-gcall-r2-mb49152-r3-production-source-ready-v1"
CONDITION = "rmct-convergence"
RUN_PREFIX = "rmct-convergence-gcall-r2-mb49152-r3"
PARENT_RUN_PREFIX = "rmct-convergence-gcall-r2"
PARENT_RUN_NAME = "rmct-convergence-gcall-r2-s001"
SOURCE_READY_FILENAME = ".rmct-convergence-gcall-r2-mb49152-r3-production-source-ready"
PLAN_RELATIVE = (
    "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct_convergence_gcall_r2_mb49152_r3_isambard_20260814.yaml"
)
CONTINUATION_PARENT_ARTIFACT_RELATIVE = (
    "artifacts/rmct-convergence-gcall-r2-mb49152-r3-20260814/continuation-parent.json"
)
CONTINUATION_PARENT_ARTIFACT_SHA256 = "b4ad494f7b20fc20af4610489a2ccff315d651a0e2636957a353a89065a1f906"
PREFLIGHT_RESULT_RELATIVE = (
    "artifacts/rmct-convergence-gcall-r2-mb49152-r3-20260814/"
    "same-gh200-all-gc-preflight-result.json"
)
PREFLIGHT_RESULT_SHA256 = "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9"
PREFLIGHT_CONTRACT_RELATIVE = (
    "artifacts/rmct-convergence-gcall-r2-mb49152-r3-20260814/"
    "same-gh200-all-gc-preflight-contract.json"
)
PREFLIGHT_CONTRACT_SHA256 = "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394"
PARENT_RUN_RELATIVE = f"logs/{CONDITION}/{PARENT_RUN_NAME}"
PARENT_CHECKPOINT_RELATIVE = (
    f"{PARENT_RUN_RELATIVE}/checkpoints/{CONDITION}_{PARENT_RUN_NAME}"
)
PARENT_CHECKPOINT_RECEIPT_RELATIVE = f"{PARENT_RUN_RELATIVE}/segment/checkpoint-receipt.json"
PARENT_COMPLETION_RECEIPT_RELATIVE = f"{PARENT_RUN_RELATIVE}/segment/completion-receipt.json"
PARENT_TRAINING_COMMAND_RELATIVE = f"{PARENT_RUN_RELATIVE}/segment/training-command.json"
PARENT_TRAINING_STARTED_RELATIVE = f"{PARENT_RUN_RELATIVE}/segment/training-started.json"
PARENT_READY_RECEIPT_RELATIVE = "artifacts/rmct-convergence-gcall-r2-production-ready.json"
PARENT_READY_SCHEMA = "rmct-convergence-gcall-r2-production-ready-v1"
PARENT_SOURCE_READY_SCHEMA = "rmct-convergence-gcall-r2-production-source-ready-v1"
PARENT_COMPILER_MODULE = "experiments.rmct_convergence.plan"
PARENT_PLAN_RELATIVE = (
    "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct_convergence_gcall_r2_isambard_20260814.yaml"
)
DATA_RELATIVE = (
    "artifacts/rmct-shared-qid-two-bias-20260813/"
    "shared-qid-two-bias-n1000-cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075.jsonl"
)
MANIFEST_RELATIVE = (
    "artifacts/rmct-shared-qid-two-bias-20260813/"
    "shared-qid-two-bias-n1000-cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075"
    ".manifest-eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc.json"
)
DATA_SHA256 = "cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075"
MANIFEST_SHA256 = "eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc"
MODEL = "Qwen/Qwen3.5-9B"
BASE_SNAPSHOT = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TOPOLOGY_PROFILE = "phase-shared-four-lane"
WORKER_PARITY_DIRECTORY_RELATIVE = "artifacts/rmct-convergence-worker-parity-20260813"
WORKER_PARITY_ATTESTATION = "qwen35-rollout-worker-parity-attestation.json"
WORKER_PARITY_RESULT = "result.json"
WORKER_PARITY_SUCCESS = "SUCCESS"
# This is deliberately duplicated from ``ctm.backends.local.replicated`` rather
# than imported: source-ready capture is intentionally usable before the GPU
# environment (and hence torch) is installed.  The implementation source is
# hash-bound in CRITICAL_SOURCES below, while the stored strict checkpoint
# manifest must carry this exact protocol value.
REPLICATED_PROTOCOL_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

CRITICAL_SOURCES = (
    "experiments/rmct_convergence/__init__.py",
    "experiments/rmct_convergence/plan.py",
    "experiments/rmct_convergence/controller.py",
    "experiments/rmct_convergence_accelerated/__init__.py",
    "experiments/rmct_convergence_accelerated/plan.py",
    "infra/isambard/verify_rmct_convergence_accelerated_production_ready.py",
    "infra/isambard/run_qwen35_rmct_convergence_accelerated_deadline.sh",
    "infra/isambard/run_qwen35_rmct_convergence_accelerated_segment.sbatch",
    "infra/isambard/rmct_convergence_segment_boundary.py",
    "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py",
    "ctm/__init__.py",
    "ctm/artifacts.py",
    "ctm/cli_safety.py",
    "ctm/backends/__init__.py",
    "ctm/backends/base.py",
    "ctm/backends/cli.py",
    "ctm/backends/run_metadata.py",
    "ctm/identity.py",
    "ctm/provenance.py",
    "ctm/experiments/__init__.py",
    "ctm/experiments/records.py",
    "ctm/backends/renderers.py",
    "ctm/backends/local/engine.py",
    "ctm/backends/local/__init__.py",
    "ctm/backends/local/losses.py",
    "ctm/backends/local/mlp_hooks.py",
    "ctm/backends/local/phase_shared.py",
    "ctm/backends/local/replicated.py",
    "ctm/backends/local/rollout_workers.py",
    "ctm/backends/local/qwen35_vllm_compat.py",
    "ctm/backends/local/vllm_sampler.py",
    "ctm/core/advantages.py",
    "ctm/core/__init__.py",
    "ctm/core/config.py",
    "ctm/core/rewards.py",
    "ctm/core/types.py",
    "ctm/importing.py",
    "ctm/settings/base.py",
    "ctm/settings/__init__.py",
    "ctm/settings/runtime.py",
    "ctm/training/checkpoints.py",
    "ctm/training/__init__.py",
    "ctm/training/consistency_losses.py",
    "ctm/training/manifest.py",
    "ctm/training/rl.py",
    "ctm/training/resume_state.py",
    "ctm/training/rollout_log.py",
    "ctm/training/run_utils.py",
    "ctm_data/adapters/mcq_bias/shared_qid_two_bias.py",
    "ctm_data/__init__.py",
    "ctm_data/adapters/__init__.py",
    "ctm_data/adapters/mcq_bias/__init__.py",
    "ctm_data/adapters/mcq_bias/data.py",
    "infra/vastai/preflight_qwen35_phase_shared.py",
    "scripts/run_experiment.py",
    "scripts/train_rlct.py",
    "requirements.txt",
)


class ReadyError(ValueError):
    """A continuation receipt is incomplete, mutable, or unsafe."""


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


def _continuation_parent_artifact_identity(root: Path) -> dict[str, Any]:
    identity = _identity(root, CONTINUATION_PARENT_ARTIFACT_RELATIVE, label="accelerated continuation parent artifact")
    if identity["sha256"] != CONTINUATION_PARENT_ARTIFACT_SHA256:
        raise ReadyError("accelerated continuation parent artifact differs from the fixed custody record")
    document = _json(root / CONTINUATION_PARENT_ARTIFACT_RELATIVE, label="accelerated continuation parent artifact")
    parent = document.get("parent")
    evidence = document.get("preflight_evidence")
    if not isinstance(parent, Mapping) or not isinstance(evidence, Mapping):
        raise ReadyError("accelerated continuation parent artifact has malformed parent/evidence records")
    if (
        document.get("schema") != "rmct-convergence-accelerated-continuation-parent-v1"
        or document.get("condition") != CONDITION
        or document.get("run_prefix") != RUN_PREFIX
        or document.get("logical_start_segment_index") != 1
        or parent.get("run_prefix") != PARENT_RUN_PREFIX
        or parent.get("run_name") != PARENT_RUN_NAME
        or parent.get("segment_index") != 0
        or parent.get("optimizer_step") != 16
        or parent.get("expected_kind") != "both"
        or parent.get("expected_final") is not True
        or evidence.get("all_gradient_checkpointing_layers") != "all"
        or evidence.get("hardware_scope") != "same-GH200"
        or evidence.get("highest_validated_local_forward_microbatch_max_tokens") != 49152
        or evidence.get("result_sha256") != "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9"
        or evidence.get("contract_sha256") != "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394"
        or evidence.get("result") != {"path": PREFLIGHT_RESULT_RELATIVE, "sha256": PREFLIGHT_RESULT_SHA256}
        or evidence.get("contract") != {"path": PREFLIGHT_CONTRACT_RELATIVE, "sha256": PREFLIGHT_CONTRACT_SHA256}
    ):
        raise ReadyError("accelerated continuation parent artifact does not bind the reviewed r2 s001 parent")
    return identity


def _preflight_evidence_identity(root: Path) -> dict[str, Any]:
    """Bind and semantically replay the same-GH200 all-GC capacity evidence."""

    result_identity = _identity(root, PREFLIGHT_RESULT_RELATIVE, label="same-GH200 preflight result")
    contract_identity = _identity(root, PREFLIGHT_CONTRACT_RELATIVE, label="same-GH200 preflight contract")
    if result_identity["sha256"] != PREFLIGHT_RESULT_SHA256 or contract_identity["sha256"] != PREFLIGHT_CONTRACT_SHA256:
        raise ReadyError("staged same-GH200 all-GC preflight evidence differs from its fixed hashes")

    result = _json(root / PREFLIGHT_RESULT_RELATIVE, label="same-GH200 preflight result")
    contract = _json(root / PREFLIGHT_CONTRACT_RELATIVE, label="same-GH200 preflight contract")
    config = contract.get("config")
    resolved = contract.get("resolved")
    if not isinstance(config, Mapping) or not isinstance(resolved, Mapping):
        raise ReadyError("same-GH200 preflight contract lacks config/resolved objects")
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
        raise ReadyError("same-GH200 preflight contract does not attest the r3 all-GC 49,152-token topology")
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
        item
        for item in sweep
        if isinstance(item, Mapping)
        and item.get("packing_budget") == 49152
        and item.get("padded_token_slots") == 49152
        and item.get("passed") is True
        and item.get("scope") == "rank_zero_capacity_only_with_all_vllm_workers_asleep"
    ]
    if len(matching) != 1:
        raise ReadyError("same-GH200 preflight result does not contain exactly one passed 49,152-token all-workers-asleep probe")
    memory = phase_shared.get("packing_workers_asleep_memory")
    rows = memory.get("nvidia_smi_rows") if isinstance(memory, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 4 or any(not isinstance(row, str) or "NVIDIA GH200" not in row for row in rows):
        raise ReadyError("same-GH200 preflight result does not attest four GH200 devices")
    for parity_key in (
        "fixed_update_loss_parity",
        "fixed_update_parameter_parity",
        "fixed_update_pre_optimizer_gradient_parity",
    ):
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
            "local_forward_microbatch_max_tokens": 49152,
            "local_target_logprob_chunk_size": 2048,
            "passed": True,
        },
    }


def _version(module: str) -> str | None:
    try:
        loaded = importlib.import_module(module)
    except Exception:
        return None
    value = getattr(loaded, "__version__", None)
    return str(value) if value is not None else None


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


def _worker_parity_identity(root: Path, *, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    directory = (root / WORKER_PARITY_DIRECTORY_RELATIVE).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"worker-parity directory escapes repository root: {directory}") from exc
    if directory.is_symlink() or not directory.is_dir():
        raise ReadyError(f"worker-parity directory must be a regular directory: {directory}")
    helper = _under_root(root, "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py", label="worker-parity helper")
    snapshot_path = snapshot.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise ReadyError("pinned snapshot identity lacks a regular snapshot path")
    completed = subprocess.run(
        [sys.executable, str(helper), "--output-dir", str(directory), "--model-snapshot", snapshot_path, "--resume"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise ReadyError(f"worker-parity sidecar did not revalidate: {detail[-1] if detail else completed.returncode}")
    relative_files = {
        "attestation": f"{WORKER_PARITY_DIRECTORY_RELATIVE}/{WORKER_PARITY_ATTESTATION}",
        "result": f"{WORKER_PARITY_DIRECTORY_RELATIVE}/{WORKER_PARITY_RESULT}",
        "success": f"{WORKER_PARITY_DIRECTORY_RELATIVE}/{WORKER_PARITY_SUCCESS}",
    }
    files = {name: _identity(root, relative, label=f"worker-parity {name}") for name, relative in relative_files.items()}
    result = _json(root / relative_files["result"], label="worker-parity result")
    attestation = _json(root / relative_files["attestation"], label="worker-parity attestation")
    if result.get("schema") != "rmct-convergence-worker-parity-fastpath-v1" or result.get("status") != "passed" or result.get("passed") is not True:
        raise ReadyError("worker-parity result is not a passed receipt")
    if result.get("model_snapshot") != snapshot_path or result.get("attestation_sha256") != files["attestation"]["sha256"]:
        raise ReadyError("worker-parity receipt is bound to different snapshot/attestation evidence")
    if attestation.get("schema") != "qwen35-rollout-worker-parity-attestation-v1":
        raise ReadyError("worker-parity attestation has an unexpected schema")
    return {
        "directory": WORKER_PARITY_DIRECTORY_RELATIVE,
        "helper_resume_validated": True,
        "files": files,
        "result_schema": result["schema"],
        "attestation_schema": attestation["schema"],
    }


def _marker_binding(value: Any, *, path: Path, sha256: str, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ReadyError(f"{label} must be one exact path/SHA-256 binding")
    if value.get("path") != str(path.resolve()) or value.get("sha256") != sha256:
        raise ReadyError(f"{label} does not bind the expected immutable parent evidence")


def _parent_training_provenance_identity(root: Path) -> dict[str, Any]:
    """Validate the exact r2 s001 command and start-marker custody chain."""

    sys.path.insert(0, str(root))
    from experiments.rmct_convergence import plan as gcall
    from scripts.run_experiment import _argument_tokens

    command_path = _under_root(root, PARENT_TRAINING_COMMAND_RELATIVE, label="continuation parent training command")
    command_identity = _identity(root, PARENT_TRAINING_COMMAND_RELATIVE, label="continuation parent training command")
    command = _json(command_path, label="continuation parent training command")
    required = {"schema", "condition", "run_prefix", "plan", "model", "segment", "argv", "environment_contract"}
    if set(command) != required:
        raise ReadyError("continuation parent training command has an unexpected envelope")
    if (
        command.get("schema") != gcall.COMMAND_SCHEMA
        or command.get("condition") != CONDITION
        or command.get("run_prefix") != PARENT_RUN_PREFIX
    ):
        raise ReadyError("continuation parent training command is not the gcall-r2 command schema/namespace")

    plan_path = _under_root(root, PARENT_PLAN_RELATIVE, label="gcall-r2 authored plan")
    plan = command.get("plan")
    expected_plan = {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)}
    if plan != expected_plan:
        raise ReadyError("continuation parent training command is not bound to the current immutable gcall-r2 plan")

    model = command.get("model")
    if not isinstance(model, Mapping) or set(model) != {"repo_id", "revision", "snapshot_path"}:
        raise ReadyError("continuation parent training command has an invalid model identity")
    snapshot_raw = model.get("snapshot_path")
    if not isinstance(snapshot_raw, str) or not snapshot_raw:
        raise ReadyError("continuation parent training command lacks its resolved pinned snapshot path")
    snapshot = Path(snapshot_raw).resolve()
    if (
        not Path(snapshot_raw).is_absolute()
        or Path(snapshot_raw).is_symlink()
        or not snapshot.is_dir()
        or snapshot.name != BASE_SNAPSHOT
        or not (snapshot / "config.json").is_file()
        or dict(model) != {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "snapshot_path": str(snapshot)}
    ):
        raise ReadyError("continuation parent training command does not use the regular pinned Qwen snapshot")

    expected_segment = gcall.segment_record(root, 0, run_prefix=PARENT_RUN_PREFIX)
    if command.get("segment") != expected_segment:
        raise ReadyError("continuation parent training command does not attest logical gcall-r2 segment zero")
    expected_args = gcall.segment_args(root, 0, model_path=snapshot, run_prefix=PARENT_RUN_PREFIX)
    if any(key in expected_args for key in ("resume_from", "resume_with_optimizer", "resume_state_required")):
        raise ReadyError("expected original gcall-r2 s001 command unexpectedly contains a resume contract")
    argv = command.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or not all(isinstance(token, str) and token for token in argv)
        or not Path(argv[0]).name.startswith("python")
        or argv[1] != str((root / "scripts/train_rlct.py").resolve())
        or argv[2:] != _argument_tokens(expected_args)
        or any(token in {"--resume-from", "--resume-with-optimizer", "--resume-state-required"} for token in argv)
    ):
        raise ReadyError("continuation parent training command argv differs from the full expected non-resume gcall-r2 s001 argv")
    expected_environment = {
        "cuda_visible_devices_preserved": True,
        "topology_profile": TOPOLOGY_PROFILE,
        "phase_shared": True,
        "gradient_checkpointing_layers": "all",
    }
    if command.get("environment_contract") != expected_environment:
        raise ReadyError("continuation parent training command has an unsafe execution contract")

    ready_path = _under_root(root, PARENT_READY_RECEIPT_RELATIVE, label="gcall-r2 parent readiness receipt")
    ready_identity = _identity(root, PARENT_READY_RECEIPT_RELATIVE, label="gcall-r2 parent readiness receipt")
    ready = _json(ready_path, label="gcall-r2 parent readiness receipt")
    # The real gcall-r2 readiness schema intentionally has no top-level
    # ``run_prefix``.  Its compiler custody block is the authoritative place
    # where the namespace is recorded; accepting a made-up top-level field
    # would incorrectly reject the sealed parent (or accept a different
    # receipt layout on a future run).
    compiled_contract = ready.get("compiled_contract")
    source_ready = ready.get("source_ready")
    frozen_hyperparameters = (
        compiled_contract.get("frozen_hyperparameters") if isinstance(compiled_contract, Mapping) else None
    )
    if (
        ready.get("schema") != PARENT_READY_SCHEMA
        or ready.get("condition") != CONDITION
        or ready.get("ready") is not True
        or "run_prefix" in ready
        or not isinstance(compiled_contract, Mapping)
        or compiled_contract.get("compiler_module") != PARENT_COMPILER_MODULE
        or compiled_contract.get("model_constants") != {"model": MODEL, "base_snapshot": BASE_SNAPSHOT}
        or not isinstance(frozen_hyperparameters, Mapping)
        or frozen_hyperparameters.get("run_prefix") != PARENT_RUN_PREFIX
        or not isinstance(source_ready, Mapping)
        or source_ready.get("schema") != PARENT_SOURCE_READY_SCHEMA
        or source_ready.get("condition") != CONDITION
        or source_ready.get("source_ready") is not True
        or "run_prefix" in source_ready
    ):
        raise ReadyError("continuation parent training marker is bound to an unexpected gcall-r2 readiness schema")
    marker_path = _under_root(root, PARENT_TRAINING_STARTED_RELATIVE, label="continuation parent training-started marker")
    marker_identity = _identity(root, PARENT_TRAINING_STARTED_RELATIVE, label="continuation parent training-started marker")
    marker = _json(marker_path, label="continuation parent training-started marker")
    if set(marker) != {"schema", "segment_index", "command_attestation", "ready_receipt"} or marker.get("schema") != "rmct-convergence-training-started-v1" or marker.get("segment_index") != 0:
        raise ReadyError("continuation parent training-started marker is not logical segment zero")
    _marker_binding(marker.get("command_attestation"), path=command_path, sha256=command_identity["sha256"], label="parent marker command")
    _marker_binding(marker.get("ready_receipt"), path=ready_path, sha256=ready_identity["sha256"], label="parent marker readiness")
    return {
        "command": command_identity,
        "training_started": marker_identity,
        "production_ready": ready_identity,
        "semantics": {
            "condition": CONDITION,
            "run_prefix": PARENT_RUN_PREFIX,
            "run_name": PARENT_RUN_NAME,
            "logical_segment_index": 0,
            "parent_kind": "pinned_base_snapshot",
            "resume": False,
            "model": {"repo_id": MODEL, "revision": BASE_SNAPSHOT},
            "gradient_checkpointing_layers": "all",
        },
    }


def _parent_checkpoint_identity(root: Path, checkpoint: Path, *, state: Any) -> dict[str, Any]:
    """Require the complete strict four-rank r2 checkpoint boundary."""

    files = {
        "adapter_config": _identity_path(root, checkpoint / "adapter_config.json", label="continuation parent adapter config"),
        "adapter_model": _identity_path(root, checkpoint / "adapter_model.safetensors", label="continuation parent adapter model"),
        "optimizer": _identity_path(root, checkpoint / "optimizer.pt", label="continuation parent optimizer state"),
        "manifest": _identity_path(root, checkpoint / "manifest.json", label="continuation parent checkpoint manifest"),
        "replicated_training_manifest": _identity_path(
            root, checkpoint / "replicated_training_manifest.json", label="continuation parent replicated-training manifest"
        ),
        "replicated_training_rng": _identity_path(
            root, checkpoint / "replicated_training_rng.pt", label="continuation parent replicated-training RNG state"
        ),
    }
    manifest = _json(checkpoint / "manifest.json", label="continuation parent checkpoint manifest")
    if manifest.get("backend") != "local" or manifest.get("kind") != "both":
        raise ReadyError("continuation parent checkpoint manifest is not a local kind='both' boundary")
    replicated = _json(
        checkpoint / "replicated_training_manifest.json", label="continuation parent replicated-training manifest"
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
        raise ReadyError("continuation parent replicated-training manifest is not the strict four-GPU all-rank optimizer boundary")
    return {
        "files": files,
        "global_step": state.global_step,
        "optimizer_step": state.optimizer_step,
        "completed_epochs": state.completed_epochs,
        "kind": "both",
        "final": True,
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
    """Read-only proof that r2 s001 is the exact external continuation parent."""

    sys.path.insert(0, str(root))
    from ctm.training.resume_state import load_strict_local_rl_resume_state
    from experiments.rmct_convergence import controller

    checkpoint = (root / PARENT_CHECKPOINT_RELATIVE).resolve()
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"continuation parent checkpoint escapes repository: {checkpoint}") from exc
    state = load_strict_local_rl_resume_state(checkpoint)
    if state.global_step != 16 or state.optimizer_step != 16:
        raise ReadyError(
            f"continuation parent must seal gcall-r2 segment zero at step 16; got global={state.global_step}, optimizer={state.optimizer_step}"
        )
    checkpoint_identity = _parent_checkpoint_identity(root, checkpoint, state=state)
    training_provenance = _parent_training_provenance_identity(root)
    checkpoint_receipt = _identity(root, PARENT_CHECKPOINT_RECEIPT_RELATIVE, label="continuation parent checkpoint receipt")
    completion_receipt = _identity(root, PARENT_COMPLETION_RECEIPT_RELATIVE, label="continuation parent completion receipt")
    decisions = (root / PARENT_RUN_RELATIVE / "decisions").resolve()
    if decisions.is_symlink() or not decisions.is_dir():
        raise ReadyError("continuation parent decisions directory is absent or linked")
    candidates = sorted(decisions.glob("checkpoint-window-decision-s000-*.json"))
    if len(candidates) != 1:
        raise ReadyError(f"continuation parent requires exactly one r2 s001 decision receipt; found {len(candidates)}")
    try:
        controller.require_continue(candidates[0])
    except Exception as exc:
        raise ReadyError(f"continuation parent r2 s001 decision is not a verified continue receipt: {exc}") from exc
    return {
        "condition": CONDITION,
        "run_prefix": PARENT_RUN_PREFIX,
        "run_name": PARENT_RUN_NAME,
        "logical_segment_index": 0,
        "checkpoint_relative_path": PARENT_CHECKPOINT_RELATIVE,
        "checkpoint": checkpoint_identity,
        "training_provenance": training_provenance,
        "checkpoint_receipt": checkpoint_receipt,
        "completion_receipt": completion_receipt,
        "continue_decision": _identity_path(root, candidates[0], label="continuation parent continue decision"),
    }


def _receipt_under_root(root: Path, value: str | Path, *, label: str) -> Path:
    """Resolve an immutable receipt without accepting a linked/outside path."""

    raw = Path(value)
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise ReadyError(f"{label} must be an absolute regular file: {raw}")
    path = raw.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"{label} must remain inside the synchronized repository: {path}") from exc
    return path


def _sealed_r3_segment_custody_identity(
    root: Path, *, segment_index: int, ready_receipt: str | Path
) -> dict[str, Any]:
    """Validate the marker/command/receipt chain before sealing an r3 recovery.

    A checkpoint directory alone is not provenance: a process might have
    created it before emitting the immutable command attestation or against a
    different readiness receipt.  The launcher calls this only in its
    checkpoint-recovery branch, immediately before ``boundary seal``.  It
    therefore refuses to turn such a partial or unrelated output into a
    sealed continuation segment.
    """

    sys.path.insert(0, str(root))
    from experiments.rmct_convergence_accelerated import plan as accelerated
    from scripts.run_experiment import _argument_tokens

    if isinstance(segment_index, bool) or not isinstance(segment_index, int) or not accelerated.START_SEGMENT_INDEX <= segment_index < accelerated.TOTAL_SEGMENTS:
        raise ReadyError(
            f"sealed accelerated continuation segment index must be in [{accelerated.START_SEGMENT_INDEX}, {accelerated.TOTAL_SEGMENTS - 1}]"
        )
    ready_path = _receipt_under_root(root, ready_receipt, label="accelerated continuation readiness receipt")
    # Rebuild the receipt at the point the existing checkpoint is accepted;
    # checking only its marker SHA would otherwise permit a stale or modified
    # readiness receipt to become a new sealed segment.
    ready = verify_receipt(root, ready_path)
    base_snapshot = ready.get("base_snapshot")
    if not isinstance(base_snapshot, Mapping):
        raise ReadyError("verified accelerated continuation readiness lacks its base snapshot")
    snapshot_raw = base_snapshot.get("snapshot_path")
    if not isinstance(snapshot_raw, str) or not snapshot_raw:
        raise ReadyError("verified accelerated continuation readiness lacks a pinned snapshot path")
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
        raise ReadyError("verified accelerated continuation readiness has an invalid pinned Qwen snapshot")

    run = accelerated.run_name(segment_index)
    segment_directory = root / "logs" / CONDITION / run / "segment"
    command_path = segment_directory / "training-command.json"
    marker_path = segment_directory / "training-started.json"
    command_identity = _identity_path(root, command_path, label="accelerated recovery training command")
    marker_identity = _identity_path(root, marker_path, label="accelerated recovery training-started marker")
    command = _json(command_path, label="accelerated recovery training command")
    required_command = {
        "schema",
        "condition",
        "run_prefix",
        "logical_segment_index",
        "plan",
        "model",
        "segment",
        "argv",
        "environment_contract",
    }
    if set(command) != required_command:
        raise ReadyError("accelerated recovery training command has an unexpected envelope")
    plan_path = _under_root(root, PLAN_RELATIVE, label="accelerated continuation YAML")
    expected_plan = {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)}
    expected_model = {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "snapshot_path": str(snapshot)}
    expected_segment = accelerated.segment_record(root, segment_index)
    expected_args = accelerated.segment_args(root, segment_index, model_path=snapshot)
    argv = command.get("argv")
    if (
        command.get("schema") != accelerated.COMMAND_SCHEMA
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
        raise ReadyError("accelerated recovery training command differs from the full expected r3 segment argv")
    expected_environment = {
        "cuda_visible_devices_preserved": True,
        "topology_profile": TOPOLOGY_PROFILE,
        "phase_shared": True,
        "gradient_checkpointing_layers": "all",
        "local_forward_microbatch_max_datums": 8,
        "local_forward_microbatch_max_tokens": 49152,
        "local_target_logprob_chunk_size": 2048,
    }
    if command.get("environment_contract") != expected_environment:
        raise ReadyError("accelerated recovery training command has an unsafe execution contract")

    marker = _json(marker_path, label="accelerated recovery training-started marker")
    if (
        set(marker) != {"schema", "segment_index", "command_attestation", "ready_receipt"}
        or marker.get("schema") != "rmct-convergence-training-started-v1"
        or marker.get("segment_index") != segment_index
    ):
        raise ReadyError("accelerated recovery training-started marker is not the expected logical segment")
    _marker_binding(
        marker.get("command_attestation"),
        path=command_path,
        sha256=command_identity["sha256"],
        label="accelerated recovery marker command",
    )
    _marker_binding(
        marker.get("ready_receipt"),
        path=ready_path,
        sha256=_sha256(ready_path),
        label="accelerated recovery marker readiness",
    )

    checkpoint = accelerated.final_checkpoint_path(root, segment_index)
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise ReadyError("accelerated recovery checkpoint must be the expected regular r3 checkpoint directory")
    return {
        "segment_index": segment_index,
        "run_name": run,
        "command": command_identity,
        "training_started": marker_identity,
        "ready_receipt": _identity_path(root, ready_path, label="accelerated continuation readiness receipt"),
        "checkpoint_relative_path": str(checkpoint.relative_to(root)),
    }


def verify_sealed_segment_custody(
    repository: str | Path, *, segment_index: int, ready_receipt: str | Path
) -> dict[str, Any]:
    """Public fail-closed entrypoint for the launcher recovery branch."""

    return _sealed_r3_segment_custody_identity(
        _root(repository), segment_index=segment_index, ready_receipt=ready_receipt
    )


def build_source_ready_receipt(repository: str | Path) -> dict[str, Any]:
    root = _root(repository)
    plan = _identity(root, PLAN_RELATIVE, label="accelerated RMCT continuation YAML")
    data = _identity(root, DATA_RELATIVE, label="shared-QID data")
    manifest = _identity(root, MANIFEST_RELATIVE, label="shared-QID manifest")
    parent_artifact = _continuation_parent_artifact_identity(root)
    preflight_evidence = _preflight_evidence_identity(root)
    if data["sha256"] != DATA_SHA256 or manifest["sha256"] != MANIFEST_SHA256:
        raise ReadyError("frozen shared-QID data or manifest differs from the continuation contract")
    source = {relative: _identity(root, relative, label="critical continuation source") for relative in CRITICAL_SOURCES}
    return {
        "schema": SOURCE_READY_SCHEMA,
        "condition": CONDITION,
        "run_prefix": RUN_PREFIX,
        "source_ready": True,
        "plan": plan,
        "data": {"data": data, "manifest": manifest},
        "continuation_parent_artifact": parent_artifact,
        "same_hardware_preflight": preflight_evidence,
        "critical_sources": source,
        "execution_disclosure": {
            "topology_profile": TOPOLOGY_PROFILE,
            "run_prefix": RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": 49152,
            "local_target_logprob_chunk_size": 2048,
            "same_hardware_all_gc_preflight_validated": True,
        },
    }


def verify_source_ready_receipt(repository: str | Path, receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    recorded = _json(Path(receipt).resolve(), label="accelerated continuation source-ready receipt")
    if (
        recorded.get("schema") != SOURCE_READY_SCHEMA
        or recorded.get("condition") != CONDITION
        or recorded.get("run_prefix") != RUN_PREFIX
        or recorded.get("source_ready") is not True
    ):
        raise ReadyError("accelerated continuation source-ready receipt has an unexpected schema/condition/namespace")
    expected = build_source_ready_receipt(root)
    if recorded != expected:
        raise ReadyError("accelerated continuation source-ready receipt does not match this synchronized checkout")
    return recorded


def _compiled_contract(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    from experiments.rmct_convergence_accelerated import plan as accelerated
    from scripts import run_experiment

    compiled = run_experiment.load_experiment(root / PLAN_RELATIVE, topology_profile=TOPOLOGY_PROFILE)
    convergence = compiled.get("rmct_convergence")
    if not isinstance(convergence, Mapping):
        raise ReadyError("compiled accelerated continuation lacks rmct_convergence custody block")
    if convergence.get("schema") != accelerated.CONTINUATION_SCHEMA or convergence.get("condition") != CONDITION:
        raise ReadyError("compiled accelerated continuation has an unexpected schema or condition")
    if convergence.get("run_prefix") != RUN_PREFIX or convergence.get("logical_start_segment_index") != 1:
        raise ReadyError("compiled accelerated continuation has an unsafe run namespace/start index")
    if convergence.get("continuation") != accelerated.CONTINUATION_METADATA:
        raise ReadyError("compiled accelerated continuation does not bind reviewed parent/preflight metadata")
    if convergence.get("data", {}).get("content_sha256") != DATA_SHA256 or convergence.get("data", {}).get("manifest_sha256") != MANIFEST_SHA256:
        raise ReadyError("compiled accelerated continuation does not bind frozen shared-QID data")
    if convergence.get("convergence", {}).get("hard_cap_optimizer_steps") != 512:
        raise ReadyError("compiled accelerated continuation hard cap must be 512 optimizer steps")
    entries = compiled.get("training")
    if not isinstance(entries, list) or len(entries) != 31:
        raise ReadyError("compiled accelerated continuation must expose only logical segments 1..31")
    first = entries[0].get("args")
    if not isinstance(first, Mapping):
        raise ReadyError("compiled accelerated continuation segment one lacks an argument map")
    if first.get("load_config") != {"n_datapoints": 32, "segment_index": 1}:
        raise ReadyError("compiled accelerated continuation must start from logical segment one")
    if first.get("run_name") != f"{RUN_PREFIX}-s002" or first.get("resume_from") != f"file://{root / PARENT_CHECKPOINT_RELATIVE}":
        raise ReadyError("compiled accelerated continuation does not bind gcall-r2 s001 as its first parent")
    if first.get("local_forward_microbatch_max_datums") != 8 or first.get("local_forward_microbatch_max_tokens") != 49152 or first.get("local_target_logprob_chunk_size") != 2048:
        raise ReadyError("compiled accelerated continuation physical microbatch contract drifted")
    if first.get("local_gradient_checkpointing") is not True or "local_gradient_checkpointing_layers" in first:
        raise ReadyError("compiled accelerated continuation must retain all-layer activation checkpointing")
    return {
        "compiler_module": "experiments.rmct_convergence_accelerated.plan",
        "compiler_source_sha256": _sha256(root / "experiments/rmct_convergence_accelerated/plan.py"),
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
        "logical_segment_one_args_sha256": hashlib.sha256(_canonical_json(first)).hexdigest(),
        "logical_segment_two_args_sha256": hashlib.sha256(_canonical_json(entries[1]["args"])).hexdigest(),
        "model_constants": {"model": accelerated.MODEL, "base_snapshot": accelerated.BASE_SNAPSHOT},
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
        "continuation_parent_artifact": source_ready["continuation_parent_artifact"],
        "same_hardware_preflight": source_ready["same_hardware_preflight"],
        "sealed_continuation_parent": _sealed_parent_identity(root),
        "base_snapshot": snapshot,
        "worker_parity": _worker_parity_identity(root, snapshot=snapshot),
        "critical_sources": source_ready["critical_sources"],
        "compiled_contract": _compiled_contract(root),
        "runtime": _runtime(),
        "execution_disclosure": {
            "topology_profile": TOPOLOGY_PROFILE,
            "run_prefix": RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": 49152,
            "local_target_logprob_chunk_size": 2048,
            "gpu_count": 4,
            "training_gpus": "all",
            "rollout_gpus": "all",
            "phase_shared": True,
            "same_hardware_all_gc_preflight_validated": True,
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
            raise ReadyError(f"refusing to overwrite different immutable continuation receipt: {path}")
        return "resumed"
    return "written"


def verify_receipt(repository: str | Path, receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    recorded = _json(Path(receipt).resolve(), label="accelerated continuation readiness receipt")
    if (
        recorded.get("schema") != SCHEMA
        or recorded.get("condition") != CONDITION
        or recorded.get("run_prefix") != RUN_PREFIX
        or recorded.get("ready") is not True
    ):
        raise ReadyError("accelerated continuation readiness receipt has an unexpected schema/condition/namespace")
    expected = build_receipt(root)
    if recorded != expected:
        raise ReadyError("accelerated continuation readiness receipt does not match this checkout, parent, or runtime")
    return recorded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for name, help_text in (
        ("capture-source-ready", "write immutable source/data continuation readiness"),
        ("verify-source-ready", "verify immutable source/data continuation readiness"),
        ("capture", "write immutable final continuation readiness"),
        ("verify", "verify immutable final continuation readiness"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repository", required=True, type=Path)
        option = "--output" if name.startswith("capture") else "--receipt"
        command.add_argument(option, required=True, type=Path)
    custody = commands.add_parser(
        "verify-sealed-segment-custody",
        help="validate r3 marker/command/readiness custody before sealing an existing checkpoint",
    )
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
            identity = verify_sealed_segment_custody(
                args.repository,
                segment_index=args.segment_index,
                ready_receipt=args.ready_receipt,
            )
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
