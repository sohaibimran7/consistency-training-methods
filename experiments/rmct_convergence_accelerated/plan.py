"""Compile the immutable accelerated RMCT continuation after gcall-r2 s001.

This is not a restart and must never execute logical segment zero.  It resumes
only the sealed gcall-r2 segment-0 checkpoint at logical segment index one,
then maintains a distinct r3 checkpoint chain.  Its only training-runtime
delta is the preflight-validated padded-token microbatch cap (20,480 ->
49,152); every scientific/data/optimizer/sampling/controller field remains
the exact gcall-r2 value.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.rmct_convergence import plan as gcall


MODEL = gcall.MODEL
BASE_SNAPSHOT = gcall.BASE_SNAPSHOT
CONDITION_NAME = gcall.CONDITION_NAME
TOPOLOGY_PROFILE = gcall.TOPOLOGY_PROFILE
GPU_COUNT = gcall.GPU_COUNT
UPDATES_PER_SEGMENT = gcall.UPDATES_PER_SEGMENT
TOTAL_SEGMENTS = gcall.TOTAL_SEGMENTS
HARD_CAP_OPTIMIZER_STEPS = gcall.HARD_CAP_OPTIMIZER_STEPS
DATAPOINTS_PER_SEGMENT = gcall.DATAPOINTS_PER_SEGMENT
BATCH_SIZE = gcall.BATCH_SIZE
BASE_ROLLOUT_SEED = gcall.BASE_ROLLOUT_SEED
DATA_PATH = gcall.DATA_PATH
DATA_SHA256 = gcall.DATA_SHA256
MANIFEST_PATH = gcall.MANIFEST_PATH
MANIFEST_SHA256 = gcall.MANIFEST_SHA256
SETTING_FACTORY = gcall.SETTING_FACTORY
WORKER_PARITY_ATTESTATION = gcall.WORKER_PARITY_ATTESTATION

RUN_PREFIX = "rmct-convergence-gcall-r2-mb49152-r3"
PARENT_RUN_PREFIX = gcall.RUN_PREFIX
START_SEGMENT_INDEX = 1
ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS = 49152
CONTINUATION_SCHEMA = "rmct-convergence-accelerated-continuation-plan-v1"
SEGMENT_SCHEMA = "rmct-convergence-accelerated-segment-v1"
COMMAND_SCHEMA = "rmct-convergence-accelerated-segment-command-v1"
CONTINUATION_PARENT_ARTIFACT = (
    "artifacts/rmct-convergence-gcall-r2-mb49152-r3-20260814/continuation-parent.json"
)
CONTINUATION_PARENT_ARTIFACT_SHA256 = "b4ad494f7b20fc20af4610489a2ccff315d651a0e2636957a353a89065a1f906"
PREFLIGHT_RESULT_ARTIFACT = (
    "artifacts/rmct-convergence-gcall-r2-mb49152-r3-20260814/"
    "same-gh200-all-gc-preflight-result.json"
)
PREFLIGHT_RESULT_ARTIFACT_SHA256 = "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9"
PREFLIGHT_CONTRACT_ARTIFACT = (
    "artifacts/rmct-convergence-gcall-r2-mb49152-r3-20260814/"
    "same-gh200-all-gc-preflight-contract.json"
)
PREFLIGHT_CONTRACT_ARTIFACT_SHA256 = "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PlanError(ValueError):
    """The accelerated continuation cannot be executed safely."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not START_SEGMENT_INDEX <= value < TOTAL_SEGMENTS:
        raise PlanError(f"accelerated segment_index must be an integer in [{START_SEGMENT_INDEX}, {TOTAL_SEGMENTS - 1}]")
    return value


def run_name(segment_index: int) -> str:
    return f"{RUN_PREFIX}-s{_index(segment_index) + 1:03d}"


def target_name(segment_index: int) -> str:
    return run_name(segment_index)


def parent_run_name() -> str:
    return gcall.run_name(0, run_prefix=PARENT_RUN_PREFIX)


def parent_checkpoint_path(repository: str | Path) -> Path:
    root = Path(repository).resolve()
    run = parent_run_name()
    return root / "logs" / CONDITION_NAME / run / "checkpoints" / f"{CONDITION_NAME}_{run}"


def final_checkpoint_path(repository: str | Path, segment_index: int) -> Path:
    root = Path(repository).resolve()
    run = run_name(segment_index)
    return root / "logs" / CONDITION_NAME / run / "checkpoints" / f"{CONDITION_NAME}_{run}"


def _parent(repository: str | Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    if index == START_SEGMENT_INDEX:
        checkpoint = parent_checkpoint_path(repository)
        return {
            "kind": "sealed_external_final_checkpoint",
            "resume": True,
            "condition": CONDITION_NAME,
            "run_prefix": PARENT_RUN_PREFIX,
            "segment_index": 0,
            "target": parent_run_name(),
            "run_name": parent_run_name(),
            "uri": f"file://{checkpoint}",
            "optimizer_step": UPDATES_PER_SEGMENT,
            "expected_kind": "both",
            "expected_final": True,
            "resume_with_optimizer": True,
            "resume_state_required": True,
        }
    previous = index - 1
    checkpoint = final_checkpoint_path(repository, previous)
    return {
        "kind": "sealed_strict_final_checkpoint",
        "resume": True,
        "condition": CONDITION_NAME,
        "run_prefix": RUN_PREFIX,
        "segment_index": previous,
        "target": target_name(previous),
        "run_name": run_name(previous),
        "uri": f"file://{checkpoint}",
        "optimizer_step": (previous + 1) * UPDATES_PER_SEGMENT,
        "expected_kind": "both",
        "expected_final": True,
        "resume_with_optimizer": True,
        "resume_state_required": True,
    }


CONTINUATION_METADATA: dict[str, Any] = {
    "schema": "rmct-convergence-accelerated-continuation-v1",
    "logical_start_segment_index": START_SEGMENT_INDEX,
    "continuation_of": {
        "condition": CONDITION_NAME,
        "parent_run_prefix": PARENT_RUN_PREFIX,
        "parent_run_name": "rmct-convergence-gcall-r2-s001",
        "parent_segment_index": 0,
        "parent_optimizer_step": UPDATES_PER_SEGMENT,
        "parent_checkpoint_relative_path": (
            "logs/rmct-convergence/rmct-convergence-gcall-r2-s001/checkpoints/"
            "rmct-convergence_rmct-convergence-gcall-r2-s001"
        ),
        "parent_checkpoint_kind": "both",
        "parent_checkpoint_final": True,
        "parent_continue_decision_required": True,
    },
    "execution_delta": {
        "local_forward_microbatch_max_tokens": {"from": 20480, "to": ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS},
        "local_forward_microbatch_max_datums": 8,
        "local_target_logprob_chunk_size": 2048,
        "gradient_checkpointing_layers": "all",
    },
    "scientific_contract_unchanged": True,
    "same_hardware_all_gradient_checkpointing_preflight_evidence": {
        "hardware_scope": "same-GH200",
        "highest_validated_local_forward_microbatch_max_tokens": ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS,
        "all_gradient_checkpointing_layers": "all",
        "result": {"path": PREFLIGHT_RESULT_ARTIFACT, "sha256": PREFLIGHT_RESULT_ARTIFACT_SHA256},
        "contract": {"path": PREFLIGHT_CONTRACT_ARTIFACT, "sha256": PREFLIGHT_CONTRACT_ARTIFACT_SHA256},
        "result_sha256": PREFLIGHT_RESULT_ARTIFACT_SHA256,
        "contract_sha256": PREFLIGHT_CONTRACT_ARTIFACT_SHA256,
    },
    "continuation_parent_artifact": {
        "path": CONTINUATION_PARENT_ARTIFACT,
        "sha256": CONTINUATION_PARENT_ARTIFACT_SHA256,
    },
}


def _frozen_spec() -> dict[str, Any]:
    """Produce the one full exact continuation spec from gcall-r2 values."""

    spec = copy.deepcopy(gcall._frozen_spec(recovery=True))
    del spec["recovery"]
    spec["topology_profiles"][TOPOLOGY_PROFILE]["gradient_checkpointing_layers"] = "all"
    spec["local"]["gradient_checkpointing_layers"] = "all"
    spec["local"]["forward_microbatch_max_tokens"] = ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS
    spec["continuation"] = CONTINUATION_METADATA
    return spec


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise PlanError("accelerated RMCT continuation spec must be an object")
    actual = dict(spec)
    expected = _frozen_spec()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        if missing or unknown:
            raise PlanError(f"accelerated RMCT continuation spec keys differ: missing={missing}, unknown={unknown}")
        raise PlanError("accelerated RMCT continuation spec differs from the frozen production contract")
    return expected


def _validate_profile(spec: Mapping[str, Any], requested: str | None) -> dict[str, Any]:
    if requested != TOPOLOGY_PROFILE:
        raise PlanError(f"accelerated RMCT continuation requires explicit topology profile {TOPOLOGY_PROFILE!r}")
    profile = spec["topology_profiles"][TOPOLOGY_PROFILE]
    if not isinstance(profile, Mapping) or profile.get("gradient_checkpointing_layers") != "all":
        raise PlanError("accelerated RMCT continuation requires all-layer gradient checkpointing metadata")
    return dict(profile)


def segment_record(repository: str | Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    return {
        "schema": SEGMENT_SCHEMA,
        "condition": CONDITION_NAME,
        "run_prefix": RUN_PREFIX,
        "segment_index": index,
        "target": target_name(index),
        "run_name": run_name(index),
        "optimizer_step_start": index * UPDATES_PER_SEGMENT + 1,
        "optimizer_step_end": (index + 1) * UPDATES_PER_SEGMENT,
        "optimizer_steps": UPDATES_PER_SEGMENT,
        "base_qids": DATAPOINTS_PER_SEGMENT,
        "batch_size": BATCH_SIZE,
        "expected_batches": UPDATES_PER_SEGMENT,
        "dataset_balance": "one_logiqa_plus_one_hellaswag_per_optimizer_update",
        "parent": _parent(repository, index),
        "rollout_seed_base": BASE_ROLLOUT_SEED,
    }


def segment_args(repository: str | Path, segment_index: int, *, model_path: str | Path | None = None) -> dict[str, Any]:
    """Emit a gcall-r2-identical command except for the reviewed runtime delta."""

    index = _index(segment_index)
    root = Path(repository).resolve()
    record = segment_record(root, index)
    # gcall-r2 already emits all-layer checkpointing by leaving the selective
    # layer argument absent.  Reuse its complete scientific argv map, then
    # replace only the r3 namespace, strict parent, and physical token cap.
    args = gcall.segment_args(root, index, model_path=model_path, run_prefix=PARENT_RUN_PREFIX)
    args["run_name"] = record["run_name"]
    args["local_forward_microbatch_max_tokens"] = ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS
    if "local_gradient_checkpointing_layers" in args:
        raise PlanError("accelerated continuation must omit a limiting checkpoint-layer CLI argument")
    parent = record["parent"]
    args.update(
        {
            "resume_from": parent["uri"],
            "resume_with_optimizer": True,
            "resume_state_required": True,
        }
    )
    return args


def _training_entry(repository: str | Path, segment_index: int) -> dict[str, Any]:
    record = segment_record(repository, segment_index)
    return {
        "name": f"{RUN_PREFIX.replace('-', '_')}_s{segment_index + 1:03d}",
        "target": record["target"],
        "gpu_count": GPU_COUNT,
        "command": ["${python}", "scripts/train_rlct.py"],
        "args": segment_args(repository, segment_index),
    }


def compile_experiment(*, name: str, spec: Mapping[str, Any], topology_profile: str | None = None) -> dict[str, Any]:
    if name != CONDITION_NAME:
        raise PlanError(f"condition name must be {CONDITION_NAME!r}")
    frozen = _validate_spec(spec)
    profile = _validate_profile(frozen, topology_profile)
    indices = list(range(START_SEGMENT_INDEX, TOTAL_SEGMENTS))
    segments = [segment_record(_PROJECT_ROOT, index) for index in indices]
    return {
        "name": name,
        "training_only": True,
        "onpolicy_topology_profile": TOPOLOGY_PROFILE,
        "onpolicy_topology": {
            "gpu_count": GPU_COUNT,
            "training_gpus": "all",
            "rollout_gpus": "all",
            "phase_shared": True,
            "execution_choice": "same_hardware_all_gc_preflight_validated_physical_microbatch_acceleration",
        },
        "rmct_convergence": {
            "schema": CONTINUATION_SCHEMA,
            "condition": CONDITION_NAME,
            "run_prefix": RUN_PREFIX,
            "logical_start_segment_index": START_SEGMENT_INDEX,
            "base_snapshot": frozen["base_snapshot"],
            "data": {
                "path": DATA_PATH,
                "content_sha256": DATA_SHA256,
                "manifest_path": MANIFEST_PATH,
                "manifest_sha256": MANIFEST_SHA256,
            },
            "topology": profile,
            "optimizer": frozen["optimizer"],
            "method": frozen["method"],
            "sampling": frozen["sampling"],
            "loop": frozen["loop"],
            "convergence": frozen["convergence"],
            "continuation": frozen["continuation"],
            "segments": segments,
        },
        "training": [_training_entry(_PROJECT_ROOT, index) for index in indices],
    }


def _regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PlanError(f"{label} must be a regular file: {path}")


def _load_compiled_plan(plan: Path) -> dict[str, Any]:
    from scripts.run_experiment import load_experiment

    _regular_file(plan, label="authored accelerated RMCT continuation YAML")
    return load_experiment(plan, topology_profile=TOPOLOGY_PROFILE)


def _immutable_write(path: Path, document: Mapping[str, Any]) -> str:
    payload = _canonical_json(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PlanError(f"command-attestation parent must be a regular directory: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        _regular_file(path, label="command attestation")
        if path.read_bytes() != payload:
            raise PlanError(f"refusing to overwrite different command attestation: {path}")
        return "resumed"
    return "written"


def command_attestation(
    *, repository: str | Path, plan: str | Path, segment_index: int, model_snapshot: str | Path
) -> dict[str, Any]:
    root = Path(repository).resolve()
    plan_path = Path(plan).resolve()
    compiled = _load_compiled_plan(plan_path)
    index = _index(segment_index)
    entries = compiled.get("training")
    if not isinstance(entries, list):
        raise PlanError("compiled accelerated continuation lacks training entries")
    compiled_entry = next((entry for entry in entries if entry.get("target") == target_name(index)), None)
    if not isinstance(compiled_entry, Mapping):
        raise PlanError(f"compiled accelerated continuation lacks logical segment {index}")
    entry = _training_entry(root, index)
    if compiled_entry.get("target") != entry["target"] or compiled_entry.get("gpu_count") != entry["gpu_count"]:
        raise PlanError("compiled target identity diverges from the frozen accelerated continuation contract")
    args = segment_args(root, index, model_path=model_snapshot)
    expected = dict(entry["args"])
    expected["model"] = args["model"]
    if expected != args:
        raise PlanError("dynamic accelerated continuation command diverges from the compiled frozen plan")
    from scripts.run_experiment import _argument_tokens

    argv = [sys.executable, str(root / "scripts" / "train_rlct.py"), *_argument_tokens(args)]
    return {
        "schema": COMMAND_SCHEMA,
        "condition": CONDITION_NAME,
        "run_prefix": RUN_PREFIX,
        "logical_segment_index": index,
        "plan": {"path": str(plan_path), "sha256": _sha256_bytes(plan_path.read_bytes())},
        "model": {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "snapshot_path": args["model"]},
        "segment": segment_record(root, index),
        "argv": argv,
        "environment_contract": {
            "cuda_visible_devices_preserved": True,
            "topology_profile": TOPOLOGY_PROFILE,
            "phase_shared": True,
            "gradient_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS,
            "local_target_logprob_chunk_size": 2048,
        },
    }


def _render_or_execute(args: argparse.Namespace, *, execute: bool) -> int:
    document = command_attestation(
        repository=args.repository,
        plan=args.plan,
        segment_index=args.segment_index,
        model_snapshot=args.model_snapshot,
    )
    status = _immutable_write(args.output, document)
    if not execute:
        print(_canonical_json({"path": str(Path(args.output).resolve()), "status": status}).decode(), end="")
        return 0
    if not args.yes:
        raise PlanError("execute requires --yes")
    process = subprocess.run(document["argv"], cwd=Path(args.repository).resolve(), env=dict(os.environ), check=False)
    return int(process.returncode)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("render", "write one exact continuation command"), ("execute", "write and run one continuation command")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repository", required=True, type=Path)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--segment-index", required=True, type=int)
        command.add_argument("--model-snapshot", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "execute":
            command.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "render":
            return _render_or_execute(args, execute=False)
        if args.command == "execute":
            return _render_or_execute(args, execute=True)
    except (OSError, PlanError, ValueError) as exc:
        _parser().error(str(exc))
    raise AssertionError(f"unexpected command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ACCELERATED_FORWARD_MICROBATCH_MAX_TOKENS",
    "BASE_SNAPSHOT",
    "COMMAND_SCHEMA",
    "CONDITION_NAME",
    "CONTINUATION_METADATA",
    "CONTINUATION_PARENT_ARTIFACT",
    "CONTINUATION_PARENT_ARTIFACT_SHA256",
    "CONTINUATION_SCHEMA",
    "PREFLIGHT_CONTRACT_ARTIFACT",
    "PREFLIGHT_CONTRACT_ARTIFACT_SHA256",
    "PREFLIGHT_RESULT_ARTIFACT",
    "PREFLIGHT_RESULT_ARTIFACT_SHA256",
    "PARENT_RUN_PREFIX",
    "RUN_PREFIX",
    "SEGMENT_SCHEMA",
    "START_SEGMENT_INDEX",
    "command_attestation",
    "compile_experiment",
    "final_checkpoint_path",
    "parent_checkpoint_path",
    "parent_run_name",
    "run_name",
    "segment_args",
    "segment_record",
    "target_name",
]
