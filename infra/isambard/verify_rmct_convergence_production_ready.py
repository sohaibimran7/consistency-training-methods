#!/usr/bin/env python3
"""Capture and verify the immutable production-readiness receipt for RMCT.

The receipt is deliberately a small custody document.  It does not claim that
the deadline-selected four-lane topology is globally optimal; it proves only
that the exact source, immutable data, compiler, controller and environment
that were reviewed are the ones the launcher will use.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "rmct-convergence-gcall-r2-production-ready-v1"
SOURCE_READY_SCHEMA = "rmct-convergence-gcall-r2-production-source-ready-v1"
CONDITION = "rmct-convergence"
RUN_PREFIX = "rmct-convergence-gcall-r2"
SOURCE_READY_FILENAME = ".rmct-convergence-gcall-r2-production-source-ready"
PLAN_RELATIVE = "experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_convergence_gcall_r2_isambard_20260814.yaml"
RECOVERY_PARENT_RELATIVE = "artifacts/rmct-convergence-gcall-r2-20260814/recovery-parent.json"
RECOVERY_PARENT_SHA256 = "b2a5c3d63323f43563a0281a4efb402a7778cc1e9a8d4a363ada90a9eee89e15"
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
_SHA256 = re.compile(r"[0-9a-f]{64}\\Z")

CRITICAL_SOURCES = (
    "experiments/rmct_convergence/__init__.py",
    "experiments/rmct_convergence/plan.py",
    "experiments/rmct_convergence/controller.py",
    "infra/isambard/verify_rmct_convergence_production_ready.py",
    "infra/isambard/rmct_convergence_segment_boundary.py",
    "infra/isambard/run_qwen35_rmct_convergence_deadline.sh",
    "infra/isambard/run_qwen35_rmct_convergence_segment.sbatch",
    "infra/isambard/submit_qwen35_rmct_convergence_chain.sh",
    "infra/isambard/run_rmct_convergence_deadline_broker.sbatch",
    "infra/isambard/run_qwen35_rmct_convergence_segment0_interactive.sbatch",
    "infra/isambard/submit_qwen35_rmct_convergence_segment0_when_ready.sh",
    "infra/isambard/run_qwen35_rmct_convergence_gcall_r2_workq_bootstrap.sbatch",
    "infra/isambard/run_qwen35_rmct_convergence_gcall_r2_segment.sbatch",
    "ctm/backends/cli.py",
    "ctm/backends/run_metadata.py",
    "ctm/identity.py",
    "ctm/provenance.py",
    "ctm/experiments/__init__.py",
    "ctm/experiments/records.py",
    "ctm/backends/local/engine.py",
    "ctm/backends/local/phase_shared.py",
    "ctm/backends/local/replicated.py",
    "ctm/backends/local/rollout_workers.py",
    "ctm/backends/local/qwen35_vllm_compat.py",
    "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py",
    "infra/vastai/preflight_qwen35_phase_shared.py",
    "ctm/training/rl.py",
    "scripts/train_rlct.py",
    "scripts/run_experiment.py",
    "ctm/training/resume_state.py",
    "ctm_data/adapters/mcq_bias/shared_qid_two_bias.py",
    "requirements.txt",
)


class ReadyError(ValueError):
    """A receipt is incomplete, mutable, or does not bind this checkout."""


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


def _recovery_parent_identity(root: Path) -> dict[str, Any]:
    """Bind the immutable pre-optimizer recovery parent evidence.

    The literal SHA makes the failed attempt's custody record distinct from
    both the original 16-layer source receipt and the recovery's new readiness
    receipt.  The plan separately freezes the scientific/execution delta.
    """

    identity = _identity(root, RECOVERY_PARENT_RELATIVE, label="RMCT recovery-parent artifact")
    if identity["sha256"] != RECOVERY_PARENT_SHA256:
        raise ReadyError("RMCT recovery-parent artifact differs from the fixed pre-optimizer custody record")
    document = _json(root / RECOVERY_PARENT_RELATIVE, label="RMCT recovery-parent artifact")
    prior_attempt = document.get("prior_attempt")
    recovery_parent = document.get("recovery_parent")
    if not isinstance(prior_attempt, Mapping) or not isinstance(recovery_parent, Mapping):
        raise ReadyError("RMCT recovery-parent artifact has malformed prior-attempt or parent records")
    if (
        document.get("schema") != "rmct-convergence-gcall-r2-recovery-parent-v1"
        or document.get("condition") != CONDITION
        or document.get("run_prefix") != RUN_PREFIX
        or prior_attempt.get("job_id") != "6011179"
        or prior_attempt.get("successful_optimizer_steps") != 0
        or prior_attempt.get("checkpoint_written") is not False
        or recovery_parent.get("kind") != "pinned_base_snapshot"
        or recovery_parent.get("resume") is not False
    ):
        raise ReadyError("RMCT recovery-parent artifact does not prove a clean pinned-base restart")
    return identity


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


def _version(module: str) -> str | None:
    try:
        loaded = importlib.import_module(module)
    except Exception:  # Deployment verifier records absence but never invents a version.
        return None
    value = getattr(loaded, "__version__", None)
    return str(value) if value is not None else None


def _snapshot_identity() -> dict[str, Any]:
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise ReadyError("HF_HOME must be set to capture or verify the pinned offline snapshot")
    root = Path(hf_home).expanduser().resolve()
    snapshot = root / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / BASE_SNAPSHOT
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
    """Bind and revalidate the global four-lane parity sidecar.

    The dedicated helper's ``--resume`` mode is intentionally read-only.  It
    checks the exact all-four topology, constant seed, worker ABI, model path,
    immutable attestation and terminal SUCCESS marker before this verifier
    records the three file identities.  This makes the production readiness
    receipt custody-complete rather than merely source-complete.
    """

    directory = (root / WORKER_PARITY_DIRECTORY_RELATIVE).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ReadyError(f"worker-parity directory escapes repository root: {directory}") from exc
    if directory.is_symlink() or not directory.is_dir():
        raise ReadyError(f"worker-parity directory must be a regular directory: {directory}")

    helper_relative = "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
    helper = _under_root(root, helper_relative, label="worker-parity helper")
    snapshot_path = snapshot.get("snapshot_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        raise ReadyError("pinned snapshot identity lacks a regular snapshot path")
    completed = subprocess.run(
        [
            sys.executable,
            str(helper),
            "--output-dir",
            str(directory),
            "--model-snapshot",
            snapshot_path,
            "--resume",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        summary = detail[-1] if detail else f"exit status {completed.returncode}"
        raise ReadyError(f"worker-parity sidecar did not revalidate: {summary}")

    relative_files = {
        "attestation": f"{WORKER_PARITY_DIRECTORY_RELATIVE}/{WORKER_PARITY_ATTESTATION}",
        "result": f"{WORKER_PARITY_DIRECTORY_RELATIVE}/{WORKER_PARITY_RESULT}",
        "success": f"{WORKER_PARITY_DIRECTORY_RELATIVE}/{WORKER_PARITY_SUCCESS}",
    }
    files = {
        name: _identity(root, relative, label=f"worker-parity {name}")
        for name, relative in relative_files.items()
    }
    result = _json(root / relative_files["result"], label="worker-parity result")
    attestation = _json(root / relative_files["attestation"], label="worker-parity attestation")
    if result.get("schema") != "rmct-convergence-worker-parity-fastpath-v1":
        raise ReadyError("worker-parity result has an unexpected schema")
    if result.get("status") != "passed" or result.get("passed") is not True:
        raise ReadyError("worker-parity result is not a passed receipt")
    if result.get("model_snapshot") != snapshot_path:
        raise ReadyError("worker-parity result is bound to a different model snapshot")
    if result.get("attestation_sha256") != files["attestation"]["sha256"]:
        raise ReadyError("worker-parity result does not bind its attestation identity")
    if attestation.get("schema") != "qwen35-rollout-worker-parity-attestation-v1":
        raise ReadyError("worker-parity attestation has an unexpected schema")
    return {
        "directory": WORKER_PARITY_DIRECTORY_RELATIVE,
        "helper_resume_validated": True,
        "files": files,
        "result_schema": result["schema"],
        "attestation_schema": attestation["schema"],
    }


def build_source_ready_receipt(repository: str | Path) -> dict[str, Any]:
    """Build the source/data-only prerequisite that can be captured pre-GPU.

    The source-ready receipt intentionally excludes the CUDA environment,
    local snapshot and worker sidecar because those are only meaningful inside
    the allocated four-GPU step.  It is an immutable, recomputable manifest of
    everything that must be synchronized before Slurm receives a job request.
    """

    root = _root(repository)
    plan = _identity(root, PLAN_RELATIVE, label="canonical RMCT convergence YAML")
    data = _identity(root, DATA_RELATIVE, label="shared-QID data")
    manifest = _identity(root, MANIFEST_RELATIVE, label="shared-QID manifest")
    recovery_parent = _recovery_parent_identity(root)
    if data["sha256"] != DATA_SHA256 or manifest["sha256"] != MANIFEST_SHA256:
        raise ReadyError("frozen shared-QID data or manifest identity differs from the production contract")
    source = {relative: _identity(root, relative, label="critical production source") for relative in CRITICAL_SOURCES}
    return {
        "schema": SOURCE_READY_SCHEMA,
        "condition": CONDITION,
        "source_ready": True,
        "plan": plan,
        "data": {"data": data, "manifest": manifest},
        "recovery_parent": recovery_parent,
        "critical_sources": source,
        "execution_disclosure": {
            "topology_profile": TOPOLOGY_PROFILE,
            "run_prefix": RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "deadline_execution_choice": True,
            "comparative_optimality_validated": False,
            "benchmark_gate_required": False,
        },
    }


def verify_source_ready_receipt(repository: str | Path, receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    path = Path(receipt).resolve()
    recorded = _json(path, label="production source-ready receipt")
    if (
        recorded.get("schema") != SOURCE_READY_SCHEMA
        or recorded.get("condition") != CONDITION
        or recorded.get("source_ready") is not True
    ):
        raise ReadyError("source-ready receipt has an unexpected schema, condition, or ready status")
    expected = build_source_ready_receipt(root)
    if recorded != expected:
        raise ReadyError("source-ready receipt does not match this synchronized checkout or immutable data")
    return recorded


def _compiled_contract(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    from experiments.rmct_convergence import plan as rmct_plan
    from scripts import run_experiment

    compiled = run_experiment.load_experiment(root / PLAN_RELATIVE, topology_profile=TOPOLOGY_PROFILE)
    convergence = compiled.get("rmct_convergence")
    if not isinstance(convergence, Mapping):
        raise ReadyError("compiled plan lacks rmct_convergence custody block")
    if convergence.get("condition") != CONDITION or convergence.get("data", {}).get("content_sha256") != DATA_SHA256:
        raise ReadyError("compiled RMCT convergence plan does not bind the frozen condition/data")
    if convergence.get("run_prefix") != RUN_PREFIX:
        raise ReadyError("compiled RMCT recovery plan does not use the distinct gcall-r2 run prefix")
    if convergence.get("recovery") != rmct_plan.RECOVERY_METADATA:
        raise ReadyError("compiled RMCT recovery plan does not bind the reviewed pre-optimizer recovery metadata")
    if convergence.get("data", {}).get("manifest_sha256") != MANIFEST_SHA256:
        raise ReadyError("compiled RMCT convergence plan does not bind the frozen manifest")
    if convergence.get("convergence", {}).get("hard_cap_optimizer_steps") != 512:
        raise ReadyError("compiled RMCT convergence hard cap must be 512 optimizer steps")
    if len(compiled.get("training", [])) != 32:
        raise ReadyError("compiled RMCT convergence plan must have 32 sealed segments")
    first_args = compiled["training"][0].get("args")
    if not isinstance(first_args, Mapping):
        raise ReadyError("compiled RMCT recovery segment zero lacks an argument map")
    if first_args.get("local_gradient_checkpointing") is not True:
        raise ReadyError("compiled RMCT recovery must enable activation checkpointing")
    if "local_gradient_checkpointing_layers" in first_args:
        raise ReadyError("compiled RMCT recovery must omit a limiting checkpoint-layer CLI argument")
    if first_args.get("run_name") != f"{RUN_PREFIX}-s001":
        raise ReadyError("compiled RMCT recovery segment zero run name is not distinct from the failed attempt")
    return {
        "compiler_module": "experiments.rmct_convergence.plan",
        "compiler_source_sha256": _sha256(root / "experiments/rmct_convergence/plan.py"),
        "compiled_plan_sha256": hashlib.sha256(_canonical_json(compiled)).hexdigest(),
        "frozen_hyperparameters": {
            "run_prefix": convergence["run_prefix"],
            "optimizer": convergence["optimizer"],
            "method": convergence["method"],
            "sampling": convergence["sampling"],
            "loop": convergence["loop"],
            "convergence": convergence["convergence"],
            "topology": convergence["topology"],
            "recovery": convergence["recovery"],
        },
        "segment_zero_args_sha256": hashlib.sha256(_canonical_json(compiled["training"][0]["args"])).hexdigest(),
        "segment_one_args_sha256": hashlib.sha256(_canonical_json(compiled["training"][1]["args"])).hexdigest(),
        "model_constants": {"model": rmct_plan.MODEL, "base_snapshot": rmct_plan.BASE_SNAPSHOT},
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
        "ready": True,
        "source_ready": source_ready,
        "plan": source_ready["plan"],
        "data": source_ready["data"],
        "recovery_parent": source_ready["recovery_parent"],
        "base_snapshot": snapshot,
        "worker_parity": _worker_parity_identity(root, snapshot=snapshot),
        "critical_sources": source_ready["critical_sources"],
        "compiled_contract": _compiled_contract(root),
        "runtime": _runtime(),
        "execution_disclosure": {
            "topology_profile": TOPOLOGY_PROFILE,
            "run_prefix": RUN_PREFIX,
            "activation_checkpointing_layers": "all",
            "gpu_count": 4,
            "training_gpus": "all",
            "rollout_gpus": "all",
            "phase_shared": True,
            "deadline_execution_choice": True,
            "comparative_optimality_validated": False,
            "benchmark_gate_required": False,
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
            raise ReadyError(f"refusing to overwrite a different immutable readiness receipt: {path}")
        return "resumed"
    return "written"


def verify_receipt(repository: str | Path, receipt: str | Path) -> dict[str, Any]:
    root = _root(repository)
    path = Path(receipt).resolve()
    recorded = _json(path, label="production readiness receipt")
    if recorded.get("schema") != SCHEMA or recorded.get("condition") != CONDITION or recorded.get("ready") is not True:
        raise ReadyError("receipt has an unexpected schema, condition, or ready status")
    expected = build_receipt(root)
    if recorded != expected:
        raise ReadyError("production readiness receipt does not match this checkout, data, controller, or runtime")
    return recorded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    source_capture = commands.add_parser(
        "capture-source-ready",
        help="write the immutable source/data receipt before requesting GPUs",
    )
    source_capture.add_argument("--repository", required=True, type=Path)
    source_capture.add_argument("--output", required=True, type=Path)
    source_verify = commands.add_parser(
        "verify-source-ready",
        help="verify the immutable source/data receipt before requesting GPUs",
    )
    source_verify.add_argument("--repository", required=True, type=Path)
    source_verify.add_argument("--receipt", required=True, type=Path)
    capture = commands.add_parser("capture", help="write the immutable production readiness receipt")
    capture.add_argument("--repository", required=True, type=Path)
    capture.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="verify the immutable production readiness receipt")
    verify.add_argument("--repository", required=True, type=Path)
    verify.add_argument("--receipt", required=True, type=Path)
    # The broker intentionally uses this no-subcommand form.
    parser.add_argument("--repository", dest="default_repository", type=Path)
    parser.add_argument("--receipt", dest="default_receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
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
        if args.default_repository is not None and args.default_receipt is not None:
            verify_receipt(args.default_repository, args.default_receipt)
            print(json.dumps({"path": str(args.default_receipt.resolve()), "status": "verified"}, sort_keys=True))
            return 0
        parser.error("use capture/verify, or supply --repository and --receipt for broker verification")
    except (OSError, ReadyError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
