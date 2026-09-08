#!/usr/bin/env python3
"""Fail-closed custody checks for the Isambard RMCT-256 convergence chain.

The chain deliberately uses one fresh run namespace per 64-question segment.
This module contains the CPU-only checks which make that policy enforceable:

* a submitted segment must match its static target, run name, data slice,
  parent checkpoint and worker seed in the authored plan;
* a completed segment is represented by an immutable receipt which hashes the
  target sidecar, runner output state, final checkpoint, and its direct parent;
* a namespace containing any production residue but no valid receipt is never
  retried; and
* model resolution is offline and pinned to the exact Qwen3.5 snapshot used by
  both the coordinator and the rollout-worker preflight.

This is intentionally not a generic resume helper.  It only accepts the
authored 16-segment Isambard condition, so an accidental ad-hoc continuation
cannot be mistaken for a member of the scientific chain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONDITION = "rmct256-convergence-isambard-4x64-20260811"
PLAN_RELATIVE = (
    "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct256_convergence_isambard_4x64_20260811.yaml"
)
TOPOLOGY_PROFILE = "four-gpu"
MODEL_REPO = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SEGMENTS_PER_PASS = 4
MAX_SEGMENTS = 16
ROWS_PER_SEGMENT = 64
UPDATES_PER_SEGMENT = 16
BASE_WORKER_SEED = 42
RECEIPT_SCHEMA = "rmct256-convergence-isambard-segment-receipt-v1"
BASE_SNAPSHOT_SCHEMA = "rmct256-convergence-isambard-base-snapshot-v1"
LORA_FINGERPRINT_SCHEMA = "rmct256-convergence-isambard-lora-fingerprint-v1"
RUNTIME_POLICY_SCHEMA = "rmct256-convergence-isambard-runtime-policy-receipt-v1"
RUNTIME_SOURCE_MANIFEST_SCHEMA = "rmct256-convergence-isambard-runtime-source-manifest-v1"
PREFLIGHT_RECEIPT_SCHEMA = "rmct256-convergence-isambard-preflight-success-receipt-v1"
FROZEN_LORA_CONFIG = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.0,
    "train_mlp": True,
    "train_attn": True,
    "train_unembed": False,
    "seed": 42,
}
PREFLIGHT_SCHEMA = "rmct256-convergence-isambard-preflight-plan-v1"
PREFLIGHT_EXPERIMENT = f"{CONDITION}-preflight"
PREFLIGHT_RUN_NAME = f"{PREFLIGHT_EXPERIMENT}-p01-s01"


class ContractError(ValueError):
    """The submitted chain state is not safe to use."""


@dataclass(frozen=True)
class Segment:
    """The non-configurable identity of one 16-update segment."""

    global_index: int
    pass_index: int
    segment_index: int
    target: str
    run_name: str
    row_offset: int
    checkpoint_step: int
    worker_seed_base: int

    @property
    def previous(self) -> "Segment | None":
        return segment_for_index(self.global_index - 1) if self.global_index else None

    @property
    def final_checkpoint_name(self) -> str:
        return f"{CONDITION}_{self.run_name}"


def segment_for_index(index: int) -> Segment:
    """Return the fixed static namespace and slice for one global segment."""

    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < MAX_SEGMENTS:
        raise ContractError(f"segment index must be an integer in [0, {MAX_SEGMENTS - 1}]")
    pass_index = index // SEGMENTS_PER_PASS + 1
    segment_index = index % SEGMENTS_PER_PASS
    label = f"p{pass_index:02d}-s{segment_index + 1:02d}"
    return Segment(
        global_index=index,
        pass_index=pass_index,
        segment_index=segment_index,
        target=f"rmct256-convergence-{label}",
        run_name=f"{CONDITION}-{label}",
        row_offset=segment_index * ROWS_PER_SEGMENT,
        checkpoint_step=(index + 1) * UPDATES_PER_SEGMENT,
        worker_seed_base=BASE_WORKER_SEED + 3 * index,
    )


def _absolute_root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"repo root must be a regular directory: {root}")
    return root


def _under_root(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"path escapes repository root: {path}") from exc
    return path


def _assert_regular_file(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular file: {path}")


def _assert_regular_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label} must be a regular directory: {path}")


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    _assert_regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path, *, label: str) -> dict[str, Any]:
    """Return a deliberately small, secret-free identity for a regular file."""

    _assert_regular_file(path, label=label)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _same_identity(current: dict[str, Any], recorded: Any, *, label: str) -> None:
    if current != recorded:
        raise ContractError(f"{label} changed after its receipt was written")


def _write_immutable_json(path: Path, document: Mapping[str, Any], *, label: str) -> str:
    """Write canonical JSON exactly once, or resume an identical sidecar."""

    payload = (json.dumps(dict(document), indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ContractError(f"{label} parent must not be a symlink: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        _assert_regular_file(path, label=label)
        if path.read_bytes() != payload:
            raise ContractError(f"refusing to overwrite different {label}: {path}")
        return "resumed"
    return "written"


def plan_path(root: Path, override: str | Path | None) -> Path:
    path = _under_root(root, override or PLAN_RELATIVE)
    _assert_regular_file(path, label="authored RMCT256 convergence plan")
    return path


def run_root(root: Path, segment: Segment) -> Path:
    return root / "logs" / CONDITION / segment.run_name


def receipt_path(root: Path, segment: Segment) -> Path:
    return run_root(root, segment) / "segment" / "rmct256-convergence-segment-receipt.json"


def base_snapshot_attestation_path(root: Path, segment: Segment) -> Path:
    return run_root(root, segment) / "preflight" / "qwen35-base-snapshot-attestation.json"


def target_attestation_path(root: Path, segment: Segment) -> Path:
    return run_root(root, segment) / "preflight" / "qwen35-onpolicy-target-attestation.json"


def lora_fingerprint_path(root: Path, segment: Segment) -> Path:
    return run_root(root, segment) / "preflight" / "qwen35-lora-fingerprint-attestation.json"


def runtime_policy_receipt_path(root: Path, segment: Segment) -> Path:
    return run_root(root, segment) / "preflight" / "qwen35-runtime-policy-receipt.json"


def output_state_path(root: Path, segment: Segment) -> Path:
    return root / "logs" / "experiments" / CONDITION / "targets" / segment.target / "outputs.json"


def checkpoint_directory(root: Path, segment: Segment) -> Path:
    return run_root(root, segment) / "checkpoints" / segment.final_checkpoint_name


def checkpoint_uri(root: Path, segment: Segment) -> str:
    return f"file://{checkpoint_directory(root, segment)}"


def preflight_run_root(root: Path) -> Path:
    """The isolated namespace used only for a real segment-0 transport probe."""

    return root / "logs" / PREFLIGHT_EXPERIMENT / PREFLIGHT_RUN_NAME


def preflight_plan_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "rmct256-convergence-segment0-preflight.yaml"


def preflight_runtime_policy_receipt_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "qwen35-runtime-policy-receipt.json"


def preflight_receipt_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "rmct256-convergence-preflight-success-receipt.json"


def preflight_base_snapshot_attestation_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "qwen35-base-snapshot-attestation.json"


def preflight_lora_fingerprint_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "qwen35-lora-fingerprint-attestation.json"


def preflight_source_attestation_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "qwen35-recovered-none-source-attestation.json"


def preflight_worker_parity_attestation_path(root: Path) -> Path:
    return preflight_run_root(root) / "rollout_workers" / "qwen35-rollout-worker-parity-attestation.json"


def preflight_target_attestation_path(root: Path) -> Path:
    return preflight_run_root(root) / "preflight" / "qwen35-onpolicy-target-attestation.json"


def _as_exact_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    return value


def _expect(value: Any, expected: Any, *, label: str) -> None:
    if value != expected:
        raise ContractError(f"{label}: expected {expected!r}, got {value!r}")


def _resolved_project_value(root: Path, value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value.replace("${project_root}", str(root)).replace("${experiment}", CONDITION)


def _only_target_entry(compiled: Mapping[str, Any], segment: Segment) -> dict[str, Any]:
    training = compiled.get("training")
    if not isinstance(training, list):
        raise ContractError("compiled plan has no training entries")
    entries = [entry for entry in training if isinstance(entry, dict) and entry.get("target") == segment.target]
    if len(entries) != 1:
        raise ContractError(f"compiled plan must have exactly one training entry for {segment.target}")
    return entries[0]


def validate_segment_plan(root: Path, plan: Path, segment: Segment) -> dict[str, Any]:
    """Compile and verify the exact entry before any model or worker starts."""

    # Import only after the caller supplied the repository root.  This keeps
    # this utility runnable from a durable checkout rather than the shell's
    # incidental current directory.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts import run_experiment as experiment_runner
    except ImportError as exc:  # pragma: no cover - deployment error
        raise ContractError(f"cannot import the experiment runner from {root}") from exc

    try:
        compiled = experiment_runner.load_experiment(plan, topology_profile=TOPOLOGY_PROFILE)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot compile RMCT256 convergence plan: {exc}") from exc
    _expect(compiled.get("name"), CONDITION, label="compiled experiment name")
    _expect(compiled.get("onpolicy_topology_profile"), TOPOLOGY_PROFILE, label="compiled topology profile")
    _expect(
        compiled.get("onpolicy_topology"),
        {
            "gpu_count": 4,
            "coordinator_device": "cuda:0",
            "rollout_gpus": [1, 2, 3],
        },
        label="compiled Isambard topology",
    )
    convergence = compiled.get("rmct256_convergence")
    if not isinstance(convergence, Mapping):
        raise ContractError("compiled plan has no rmct256_convergence contract")
    _expect(convergence.get("condition"), CONDITION, label="compiled convergence condition")
    _expect(convergence.get("schema"), "rmct256_convergence_plan_v1", label="compiled convergence schema")
    hard_cap = convergence.get("hard_cap")
    if not isinstance(hard_cap, Mapping):
        raise ContractError("compiled plan has no hard_cap contract")
    _expect(hard_cap.get("passes"), 4, label="compiled hard-cap passes")
    _expect(hard_cap.get("segments"), MAX_SEGMENTS, label="compiled hard-cap segments")
    _expect(hard_cap.get("updates"), MAX_SEGMENTS * UPDATES_PER_SEGMENT, label="compiled hard-cap updates")

    entry = _only_target_entry(compiled, segment)
    _expect(entry.get("gpu_count"), 4, label="compiled segment gpu count")
    args = entry.get("args")
    if not isinstance(args, Mapping):
        raise ContractError("compiled segment has no argument object")
    _expect(args.get("run_name"), segment.run_name, label="compiled segment run name")
    _expect(args.get("experiment_name"), "${experiment}", label="compiled segment experiment placeholder")
    _expect(
        args.get("setting_config"),
        {
            "data_paths": [
                "artifacts/rmct-256-training-20260804/"
                "rmct-256-training-7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a.jsonl"
            ],
            "control": False,
        },
        label="compiled biased-prompt setting configuration",
    )
    _expect(args.get("seed"), 42, label="trainer/LoRA seed")
    _expect(args.get("local_dtype"), "bfloat16", label="coordinator and worker dtype")
    _expect(args.get("local_device"), "cuda:0", label="coordinator device")
    _expect(args.get("local_rollout_gpus"), "1,2,3", label="rollout worker GPUs")
    _expect(args.get("local_rollout_seed_base"), segment.worker_seed_base, label="rollout worker seed base")
    _expect(args.get("local_vllm_gdn_prefill_backend"), "triton", label="rollout GDN backend")
    _expect(args.get("local_gradient_checkpointing"), True, label="all-layer gradient checkpointing")
    if args.get("local_gradient_checkpointing_layers") is not None:
        raise ContractError("all-layer gradient checkpointing must not impose a layer limit")
    _expect(args.get("batch_size"), 4, label="segment batch size")
    _expect(args.get("n_epochs"), 1, label="segment epochs")
    _expect(args.get("checkpoint_every"), UPDATES_PER_SEGMENT, label="segment checkpoint interval")
    _expect(args.get("save_state"), True, label="segment optimizer-state saving")
    _expect(args.get("require_onpolicy_target_attestation"), True, label="segment child attestation requirement")
    _expect(args.get("kl_coef"), 0.05, label="frozen PPO KL coefficient")
    _expect(args.get("kl_discount_factor"), 0.0, label="frozen PPO KL discount factor")
    _expect(args.get("local_ppo_clip_epsilon"), 0.2, label="frozen local PPO clip epsilon")
    _expect(
        args.get("lora_config"),
        FROZEN_LORA_CONFIG,
        label="frozen LoRA configuration",
    )
    for key, expected in {
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }.items():
        _expect(args.get(key), expected, label=f"frozen optimizer argument {key}")

    load_config = args.get("load_config")
    if not isinstance(load_config, Mapping):
        raise ContractError("compiled segment has no load_config")
    _expect(load_config.get("n_datapoints"), ROWS_PER_SEGMENT, label="segment row count")
    _expect(load_config.get("row_offset"), segment.row_offset, label="segment row offset")
    _expect(load_config.get("rmct256_segment_index"), segment.segment_index, label="segment block index")
    metadata = load_config.get("rmct256_convergence_metadata")
    if not isinstance(metadata, Mapping):
        raise ContractError("compiled segment has no rmct256_convergence_metadata")
    for key, expected in {
        "schema": "rmct256_convergence_segment_v1",
        "condition": CONDITION,
        "base_snapshot_commit": MODEL_REVISION,
        "global_segment_index": segment.global_index,
        "pass_index": segment.pass_index,
        "segment_index": segment.segment_index,
        "checkpoint_step": segment.checkpoint_step,
        "updates": UPDATES_PER_SEGMENT,
        "questions": ROWS_PER_SEGMENT,
        "row_offset": segment.row_offset,
    }.items():
        _expect(metadata.get(key), expected, label=f"segment metadata {key}")
    base_snapshot = metadata.get("base_snapshot")
    if not isinstance(base_snapshot, Mapping):
        raise ContractError("segment metadata has no base_snapshot")
    _expect(base_snapshot.get("repo_id"), MODEL_REPO, label="base snapshot repository")
    _expect(base_snapshot.get("revision"), MODEL_REVISION, label="base snapshot revision")
    _expect(base_snapshot.get("offline_only"), True, label="base snapshot offline policy")
    runtime = metadata.get("isambard_runtime")
    if not isinstance(runtime, Mapping):
        raise ContractError("segment metadata has no isambard_runtime")
    _expect(runtime.get("platform"), "isambard-ai-gh200", label="Isambard platform")
    _expect(runtime.get("architecture"), "aarch64", label="Isambard architecture")
    _expect(runtime.get("vllm_version"), "0.21.0", label="Isambard vLLM version")
    _expect(runtime.get("vllm_cuda"), "12.9", label="Isambard vLLM CUDA build")
    _expect(runtime.get("transformers_version"), "5.5.4", label="Isambard Transformers version")
    _expect(runtime.get("gdn_prefill_backend"), "triton", label="Isambard GDN backend")
    expected_optimizer = {
        "learning_rate": 1e-4,
        "learning_rate_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    _expect(metadata.get("optimizer"), expected_optimizer, label="segment optimizer metadata")
    _expect(metadata.get("setting_config"), {"control": False}, label="segment biased-prompt metadata")
    _expect(convergence.get("setting_config"), {"control": False}, label="compiled biased-prompt convergence metadata")
    expected_method = {
        "loss_fn": "ppo",
        "kl_coefficient": 0.05,
        "kl_discount_factor": 0.0,
        "ppo_clip_epsilon": 0.2,
    }
    _expect(metadata.get("method"), expected_method, label="segment PPO method metadata")
    _expect(convergence.get("method"), expected_method, label="compiled PPO method metadata")

    parent = metadata.get("parent")
    if not isinstance(parent, Mapping):
        raise ContractError("segment metadata has no parent descriptor")
    if segment.previous is None:
        _expect(parent.get("kind"), "pinned_base_snapshot", label="first-segment parent kind")
        _expect(parent.get("resume"), False, label="first-segment resume policy")
        if "resume_from" in args or args.get("resume_with_optimizer") or args.get("resume_state_required"):
            raise ContractError("first segment must not have a resume checkpoint or optimizer-state resume")
    else:
        previous = segment.previous
        assert previous is not None
        expected_uri = checkpoint_uri(root, previous)
        _expect(parent.get("kind"), "strict_final_checkpoint", label="continuation parent kind")
        _expect(parent.get("resume"), True, label="continuation parent resume policy")
        _expect(parent.get("global_segment_index"), previous.global_index, label="continuation parent segment index")
        _expect(parent.get("target"), previous.target, label="continuation parent target")
        _expect(parent.get("run_name"), previous.run_name, label="continuation parent run name")
        _expect(
            _resolved_project_value(root, parent.get("uri"), label="continuation parent URI"),
            expected_uri,
            label="continuation parent URI",
        )
        _expect(
            parent.get("checkpoint_step"),
            previous.checkpoint_step,
            label="continuation parent final checkpoint step",
        )
        _expect(parent.get("expected_kind"), "both", label="continuation parent checkpoint kind")
        _expect(parent.get("expected_final"), True, label="continuation parent final flag")
        _expect(parent.get("resume_with_optimizer"), True, label="continuation optimizer resume policy")
        _expect(parent.get("resume_state_required"), True, label="continuation strict state policy")
        _expect(
            _resolved_project_value(root, args.get("resume_from"), label="compiled resume URI"),
            expected_uri,
            label="compiled resume URI",
        )
        _expect(args.get("resume_with_optimizer"), True, label="compiled optimizer resume")
        _expect(args.get("resume_state_required"), True, label="compiled strict resume state")
    return {
        "compiled": compiled,
        "entry": entry,
        "args": dict(args),
        "metadata": dict(metadata),
        "convergence": dict(convergence),
    }


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _write_immutable_bytes(path: Path, payload: bytes, *, label: str) -> str:
    """Create immutable bytes exactly once, permitting only byte-identical reuse."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ContractError(f"{label} parent must not be a symlink: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        _assert_regular_file(path, label=label)
        if path.read_bytes() != payload:
            raise ContractError(f"refusing to overwrite different {label}: {path}")
        return "resumed"
    return "written"


def _preflight_plan_document(root: Path, plan: Path, segment: Segment) -> dict[str, Any]:
    """Build an isolated factory plan equivalent to production segment zero."""

    if segment.global_index != 0:
        raise ContractError("the real RMCT256 preflight is defined only for static segment 0")
    validated = validate_segment_plan(root, plan, segment)
    entry = validated["entry"]
    return {
        "name": PREFLIGHT_EXPERIMENT,
        "experiment_factory": "infra.isambard.rmct256_convergence_preflight:compile_preflight_experiment",
        "spec": {
            "schema": PREFLIGHT_SCHEMA,
            "production_plan": str(plan),
            "production_plan_identity": file_identity(plan, label="RMCT256 production plan for preflight"),
            "target": segment.target,
            "preflight_run_name": PREFLIGHT_RUN_NAME,
            # Keep this immutable source-entry hash visible in the authored
            # preflight plan as a human-auditable equivalence anchor.  The
            # factory independently recompiles the production YAML before it
            # returns the command.
            "production_entry_sha256": _canonical_json_sha256(entry),
        },
    }


def validate_preflight_plan(root: Path, plan: Path, path: Path) -> dict[str, Any]:
    """Recompile the preflight factory and prove it only changes the namespace."""

    _assert_regular_file(path, label="isolated RMCT256 preflight plan")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - deployment error
        raise ContractError("PyYAML is required to validate the RMCT256 preflight plan") from exc
    try:
        source = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ContractError(f"isolated RMCT256 preflight plan is invalid YAML: {path}") from exc
    if not isinstance(source, Mapping):
        raise ContractError("isolated RMCT256 preflight plan must be an object")
    _expect(source.get("name"), PREFLIGHT_EXPERIMENT, label="preflight experiment namespace")
    _expect(
        source.get("experiment_factory"),
        "infra.isambard.rmct256_convergence_preflight:compile_preflight_experiment",
        label="preflight experiment factory",
    )
    spec = source.get("spec")
    if not isinstance(spec, Mapping):
        raise ContractError("isolated RMCT256 preflight plan has no spec")
    expected_document = _preflight_plan_document(root, plan, segment_for_index(0))
    _expect(dict(source), expected_document, label="immutable preflight plan source")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts import run_experiment as experiment_runner
    except ImportError as exc:  # pragma: no cover - deployment error
        raise ContractError("cannot import the experiment runner for the preflight plan") from exc
    try:
        compiled = experiment_runner.load_experiment(path, topology_profile=TOPOLOGY_PROFILE)
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"cannot compile the isolated RMCT256 preflight plan: {exc}") from exc
    _expect(compiled.get("name"), PREFLIGHT_EXPERIMENT, label="compiled preflight experiment namespace")
    _expect(compiled.get("onpolicy_topology_profile"), TOPOLOGY_PROFILE, label="compiled preflight topology profile")
    _expect(
        compiled.get("onpolicy_topology"),
        {"gpu_count": 4, "coordinator_device": "cuda:0", "rollout_gpus": [1, 2, 3]},
        label="compiled preflight topology",
    )
    training = compiled.get("training")
    if not isinstance(training, list) or len(training) != 1 or not isinstance(training[0], Mapping):
        raise ContractError("compiled preflight plan must contain exactly one segment-0 target")
    production = validate_segment_plan(root, plan, segment_for_index(0))["entry"]
    expected_entry = json.loads(json.dumps(production))
    expected_args = expected_entry.get("args")
    if not isinstance(expected_args, dict):  # pragma: no cover - production validator already checks this
        raise ContractError("validated segment-0 target has no argument object")
    expected_args["run_name"] = PREFLIGHT_RUN_NAME
    _expect(training[0], expected_entry, label="preflight semantic equivalence to production segment 0")
    _expect(
        compiled.get("rmct256_preflight", {}).get("production_entry_sha256")
        if isinstance(compiled.get("rmct256_preflight"), Mapping)
        else None,
        _canonical_json_sha256(production),
        label="preflight production entry hash",
    )
    return {
        "plan": file_identity(path, label="isolated RMCT256 preflight plan"),
        "target": segment_for_index(0).target,
        "run_name": PREFLIGHT_RUN_NAME,
        "experiment": PREFLIGHT_EXPERIMENT,
    }


def write_preflight_plan(root: Path, plan: Path, segment: Segment, *, output: Path | None = None) -> dict[str, Any]:
    """Mint or resume the immutable, isolated real-GPU preflight plan."""

    document = _preflight_plan_document(root, plan, segment)
    path = preflight_plan_path(root) if output is None else _under_root(root, output)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - deployment error
        raise ContractError("PyYAML is required to write the RMCT256 preflight plan") from exc
    payload = yaml.safe_dump(document, sort_keys=False).encode("utf-8")
    status = _write_immutable_bytes(path, payload, label="isolated RMCT256 preflight plan")
    validated = validate_preflight_plan(root, plan, path)
    return {"status": status, "path": str(path), **validated}


def _checkpoint_manifest(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_regular_directory(path, label="final checkpoint directory")
    manifest_path = path / "manifest.json"
    manifest = _json_object(manifest_path, label="final checkpoint manifest")
    _expect(manifest.get("backend"), "local", label="final checkpoint backend")
    _expect(manifest.get("kind"), "both", label="final checkpoint kind")
    loop = manifest.get("loop_state")
    if not isinstance(loop, Mapping):
        raise ContractError("final checkpoint has no loop_state")
    return manifest, dict(loop)


def _validate_final_checkpoint(root: Path, segment: Segment) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = checkpoint_directory(root, segment)
    manifest, loop = _checkpoint_manifest(directory)
    _expect(loop.get("schema"), "ctm.rl_loop_state.v1", label="final checkpoint loop-state schema")
    _expect(loop.get("global_step"), segment.checkpoint_step, label="final checkpoint global step")
    _expect(loop.get("step"), segment.checkpoint_step, label="final checkpoint legacy step")
    _expect(loop.get("segment_start_global_step"), segment.global_index * UPDATES_PER_SEGMENT, label="segment start step")
    _expect(loop.get("segment_step"), UPDATES_PER_SEGMENT, label="segment update count")
    _expect(loop.get("completed_epochs"), segment.global_index + 1, label="completed segment epochs")
    _expect(loop.get("accumulated_grads"), 0, label="final checkpoint accumulated gradients")
    _expect(loop.get("final"), True, label="final checkpoint flag")
    optimizer_step = _as_exact_int(loop.get("optimizer_step"), label="final checkpoint optimizer step")
    if not 0 <= optimizer_step <= segment.checkpoint_step:
        raise ContractError("final checkpoint optimizer step is outside [0, global_step]")
    runtime_rng = loop.get("runtime_rng")
    if not isinstance(runtime_rng, Mapping) or runtime_rng.get("schema") != "ctm.rl_runtime_rng.v1":
        raise ContractError("final checkpoint lacks strict coordinator runtime RNG state")
    for key in ("python_random_state", "torch_cpu_rng_state_base64", "torch_cuda_rng_state_base64"):
        if key not in runtime_rng:
            raise ContractError(f"final checkpoint runtime RNG has no {key}")
    _expect(runtime_rng.get("torch_cuda_coordinator_device"), 0, label="checkpoint coordinator RNG device")
    optimizer_path = directory / "optimizer.pt"
    _assert_regular_file(optimizer_path, label="final checkpoint optimizer state")

    # Reuse the runner's local artifact definition so a receipt and the
    # target-scoped output publication hash the same concrete checkpoint set.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scripts.run_experiment import _checkpoint_artifact_manifest
    except ImportError as exc:  # pragma: no cover - deployment error
        raise ContractError("cannot import checkpoint artifact validator") from exc
    try:
        artifacts = _checkpoint_artifact_manifest(checkpoint_uri(root, segment))
    except (OSError, TypeError, ValueError) as exc:
        raise ContractError(f"final checkpoint artifact validation failed: {exc}") from exc
    return artifacts, loop


def _validate_target_attestation(root: Path, segment: Segment) -> dict[str, Any]:
    path = target_attestation_path(root, segment)
    document = _json_object(path, label="on-policy target attestation")
    _expect(document.get("schema"), "qwen35-onpolicy-target-attestation-v1", label="target attestation schema")
    _expect(document.get("target"), segment.target, label="target attestation target")
    _expect(document.get("run_name"), segment.run_name, label="target attestation run name")
    _expect(document.get("experiment_name"), CONDITION, label="target attestation experiment")
    _expect(document.get("topology_profile"), TOPOLOGY_PROFILE, label="target attestation topology")
    return file_identity(path, label="on-policy target attestation")


def _base_snapshot_record(root: Path, segment: Segment) -> dict[str, Any]:
    """Return the resolved offline cache identity carried into the receipt."""

    path = base_snapshot_attestation_path(root, segment)
    document = _json_object(path, label="base snapshot attestation")
    _expect(document.get("schema"), BASE_SNAPSHOT_SCHEMA, label="base snapshot schema")
    _expect(document.get("repo_id"), MODEL_REPO, label="base snapshot repository")
    _expect(document.get("revision"), MODEL_REVISION, label="base snapshot revision")
    _expect(document.get("segment"), segment.global_index, label="base snapshot segment")
    _expect(document.get("hf_hub_offline"), True, label="base snapshot HF offline flag")
    _expect(document.get("transformers_offline"), True, label="base snapshot Transformers offline flag")
    snapshot = document.get("snapshot")
    main_ref = document.get("main_ref")
    if not isinstance(snapshot, Mapping) or not isinstance(main_ref, Mapping):
        raise ContractError("base snapshot attestation has no resolved snapshot/ref record")
    resolved_path = snapshot.get("path")
    ref_commit = main_ref.get("commit")
    if not isinstance(resolved_path, str) or not resolved_path:
        raise ContractError("base snapshot attestation has no resolved snapshot path")
    _expect(ref_commit, MODEL_REVISION, label="base snapshot main ref commit")
    return {
        "attestation": file_identity(path, label="base snapshot attestation"),
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "resolved_snapshot_path": resolved_path,
        "main_ref_commit": ref_commit,
    }


def _validate_base_snapshot_attestation(root: Path, segment: Segment) -> dict[str, Any]:
    """Return the immutable sidecar identity used by older receipt fields."""

    return _base_snapshot_record(root, segment)["attestation"]


def _validate_lora_fingerprint_attestation(
    root: Path,
    plan: Path,
    segment: Segment,
    *,
    path: Path | None = None,
    base_snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Check the sealed PEFT inventory before retaining it as provenance.

    The launcher invokes the fingerprint helper's expensive meta-model
    recomputation immediately before it writes the training-start marker.  A
    receipt validator intentionally performs only this cheap, deterministic
    structural/hash validation: it proves that the sidecar still binds the
    current static plan and pinned-cache sidecar without constructing another
    nine-billion-parameter module graph while walking a parent chain.
    """

    attestation_path = lora_fingerprint_path(root, segment) if path is None else path.resolve()
    base_path = base_snapshot_attestation_path(root, segment) if base_snapshot_path is None else base_snapshot_path.resolve()
    document = _json_object(attestation_path, label="LoRA fingerprint attestation")
    _validate_no_secret_keys(document, path="LoRA fingerprint attestation")
    _expect(document.get("schema"), LORA_FINGERPRINT_SCHEMA, label="LoRA fingerprint schema")
    _expect(document.get("condition"), CONDITION, label="LoRA fingerprint condition")
    _expect(
        document.get("segment"),
        {
            "global_segment_index": segment.global_index,
            "target": segment.target,
            "run_name": segment.run_name,
        },
        label="LoRA fingerprint segment identity",
    )
    _same_identity(
        file_identity(plan, label="LoRA fingerprint plan"),
        document.get("plan"),
        label="LoRA fingerprint plan",
    )
    _same_identity(
        file_identity(base_path, label="LoRA fingerprint base-snapshot attestation"),
        document.get("base_snapshot_attestation"),
        label="LoRA fingerprint base-snapshot attestation",
    )
    base_document = _json_object(base_path, label="LoRA fingerprint base-snapshot attestation")
    _expect(base_document.get("schema"), BASE_SNAPSHOT_SCHEMA, label="LoRA fingerprint base snapshot schema")
    _expect(base_document.get("repo_id"), MODEL_REPO, label="LoRA fingerprint base repository")
    _expect(base_document.get("revision"), MODEL_REVISION, label="LoRA fingerprint base revision")
    _expect(base_document.get("segment"), segment.global_index, label="LoRA fingerprint base segment")
    base_snapshot = document.get("base_snapshot")
    if not isinstance(base_snapshot, Mapping):
        raise ContractError("LoRA fingerprint has no base snapshot record")
    _expect(base_snapshot.get("repo_id"), MODEL_REPO, label="LoRA fingerprint recorded repository")
    _expect(base_snapshot.get("revision"), MODEL_REVISION, label="LoRA fingerprint recorded revision")
    source_snapshot = base_document.get("snapshot")
    if not isinstance(source_snapshot, Mapping) or not isinstance(source_snapshot.get("path"), str):
        raise ContractError("LoRA fingerprint base attestation has no snapshot path")
    _expect(
        base_snapshot.get("resolved_snapshot_path"),
        source_snapshot.get("path"),
        label="LoRA fingerprint resolved snapshot path",
    )
    _expect(document.get("lora_config"), FROZEN_LORA_CONFIG, label="LoRA fingerprint frozen configuration")
    inventory = document.get("inventory")
    if not isinstance(inventory, Mapping):
        raise ContractError("LoRA fingerprint has no inventory")
    _expect(
        inventory.get("derivation_mode"),
        "meta_model_from_pinned_snapshot_config",
        label="LoRA fingerprint derivation mode",
    )
    target_modules = inventory.get("resolved_target_modules")
    target_parameters = inventory.get("resolved_target_parameters")
    if not isinstance(target_modules, list) or not all(isinstance(item, str) and item for item in target_modules):
        raise ContractError("LoRA fingerprint resolved target modules are invalid")
    if not isinstance(target_parameters, list) or not all(isinstance(item, str) and item for item in target_parameters):
        raise ContractError("LoRA fingerprint resolved target parameters are invalid")
    if target_modules != sorted(set(target_modules)) or target_parameters != sorted(set(target_parameters)):
        raise ContractError("LoRA fingerprint target lists must be sorted and unique")
    if not target_modules and not target_parameters:
        raise ContractError("LoRA fingerprint selected no PEFT target modules or parameters")
    peft = inventory.get("peft")
    if not isinstance(peft, Mapping):
        raise ContractError("LoRA fingerprint has no effective PEFT configuration")
    _expect(peft.get("r"), FROZEN_LORA_CONFIG["rank"], label="LoRA fingerprint effective rank")
    _expect(peft.get("lora_alpha"), FROZEN_LORA_CONFIG["alpha"], label="LoRA fingerprint effective alpha")
    _expect(peft.get("lora_dropout"), FROZEN_LORA_CONFIG["dropout"], label="LoRA fingerprint effective dropout")
    _expect(peft.get("bias"), "none", label="LoRA fingerprint effective bias policy")
    trainable = inventory.get("trainable_parameters")
    if not isinstance(trainable, list) or not trainable:
        raise ContractError("LoRA fingerprint has no trainable PEFT parameters")
    names: list[str] = []
    total_numel = 0
    for index, record in enumerate(trainable):
        if not isinstance(record, Mapping):
            raise ContractError(f"LoRA fingerprint trainable parameter {index} is not an object")
        name = record.get("name")
        shape = record.get("shape")
        numel = record.get("numel")
        dtype = record.get("dtype")
        if not isinstance(name, str) or not name:
            raise ContractError(f"LoRA fingerprint trainable parameter {index} has no name")
        if not isinstance(shape, list) or not shape or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
            raise ContractError(f"LoRA fingerprint trainable parameter {name} has an invalid shape")
        if isinstance(numel, bool) or not isinstance(numel, int) or numel <= 0:
            raise ContractError(f"LoRA fingerprint trainable parameter {name} has an invalid numel")
        expected_numel = 1
        for dimension in shape:
            expected_numel *= dimension
        _expect(numel, expected_numel, label=f"LoRA fingerprint trainable parameter {name} numel")
        if not isinstance(dtype, str) or not dtype:
            raise ContractError(f"LoRA fingerprint trainable parameter {name} has no dtype")
        names.append(name)
        total_numel += numel
    if names != sorted(names) or len(set(names)) != len(names):
        raise ContractError("LoRA fingerprint trainable parameter names must be sorted and unique")
    _expect(inventory.get("trainable_parameter_count"), len(trainable), label="LoRA fingerprint trainable count")
    _expect(inventory.get("trainable_parameter_numel"), total_numel, label="LoRA fingerprint trainable numel")
    _expect(
        document.get("fingerprint_sha256"),
        _canonical_json_sha256({"lora_config": FROZEN_LORA_CONFIG, "inventory": dict(inventory)}),
        label="LoRA fingerprint semantic hash",
    )
    return file_identity(attestation_path, label="LoRA fingerprint attestation")


def _validate_runtime_policy_receipt(
    root: Path,
    plan: Path,
    segment: Segment,
    *,
    path: Path | None = None,
    require_preflight_reference: bool,
) -> dict[str, Any]:
    """Validate the CPU-safe custody portion of a live runtime-policy receipt.

    The dedicated helper recomputes vLLM, CUDA and numerical state on the
    four-GH200 allocation immediately before training.  Completion and
    submitter checks cannot assume that allocation still exists, so this
    verifier rechecks the immutable receipt's static scope and its imported
    method-source manifest without trying to initialize CUDA on a login node.
    """

    try:
        from infra.isambard import rmct256_runtime_policy_receipt as runtime_policy
    except ImportError as exc:  # pragma: no cover - deployment/bootstrap error
        raise ContractError("RMCT256 runtime-policy receipt helper is unavailable") from exc

    receipt = runtime_policy_receipt_path(root, segment) if path is None else _under_root(root, path)
    document = _json_object(receipt, label="runtime-policy receipt")
    _validate_no_secret_keys(document, path="runtime-policy receipt")
    _expect(document.get("schema"), RUNTIME_POLICY_SCHEMA, label="runtime-policy receipt schema")
    scope = document.get("scope")
    if not isinstance(scope, Mapping):
        raise ContractError("runtime-policy receipt has no scope")
    validated = validate_segment_plan(root, plan, segment)
    args = validated["args"]
    expected_scope = {
        "condition": CONDITION,
        "segment": {
            "global_segment_index": segment.global_index,
            "target": segment.target,
            "run_name": segment.run_name,
            "worker_seed_base": segment.worker_seed_base,
        },
        "plan": file_identity(plan, label="runtime-policy receipt plan"),
        "compiled_target_args_sha256": runtime_policy._sha256(
            runtime_policy._canonical_value(dict(args), label="compiled target args")
        ),
    }
    _expect(dict(scope), expected_scope, label="runtime-policy receipt scope")
    payload = document.get("runtime_policy")
    if not isinstance(payload, Mapping):
        raise ContractError("runtime-policy receipt has no policy payload")
    _expect(
        document.get("runtime_policy_sha256"),
        runtime_policy._sha256(dict(payload)),
        label="runtime-policy receipt payload hash",
    )
    worker_dtype = payload.get("worker_dtype")
    if not isinstance(worker_dtype, Mapping):
        raise ContractError("runtime-policy receipt has no worker dtype record")
    _expect(worker_dtype.get("vllm_dtype_argument"), "bfloat16", label="explicit vLLM worker dtype")
    _expect(worker_dtype.get("resolved_worker_dtype"), "bfloat16", label="resolved vLLM worker dtype")
    _expect(
        worker_dtype.get("assertion"),
        "explicit_local_dtype_bfloat16",
        label="explicit vLLM worker dtype assertion",
    )
    source_manifest = payload.get("method_source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise ContractError("runtime-policy receipt has no method source manifest")
    _expect(source_manifest.get("schema"), RUNTIME_SOURCE_MANIFEST_SCHEMA, label="runtime source manifest schema")
    # Reuse the runtime helper's own source-manifest implementation rather
    # than maintaining a second, subtly divergent import/path allowlist here.
    # It performs no model or CUDA initialization.
    _expect(
        dict(source_manifest),
        runtime_policy._method_source_manifest(root),
        label="runtime-policy imported method source manifest",
    )
    reference = document.get("reference_preflight_receipt")
    if not require_preflight_reference:
        _expect(reference, None, label="preflight runtime-policy reference policy")
    else:
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise ContractError("production runtime-policy receipt has no preflight reference identity")
        expected_reference = preflight_runtime_policy_receipt_path(root)
        _same_identity(
            file_identity(expected_reference, label="preflight runtime-policy receipt"),
            dict(reference),
            label="production runtime-policy preflight reference",
        )
        reference_document = _validate_runtime_policy_receipt(
            root,
            plan,
            segment_for_index(0),
            path=expected_reference,
            require_preflight_reference=False,
        )
        reference_payload = _json_object(expected_reference, label="preflight runtime-policy receipt").get("runtime_policy")
        _expect(reference_payload, dict(payload), label="production runtime policy versus preflight baseline")
        # Keep the recursive call visibly used: its return value is the
        # immutable reference identity carried by this receipt.
        if not reference_document.get("sha256"):
            raise ContractError("preflight runtime-policy receipt has no identity hash")
    return file_identity(receipt, label="runtime-policy receipt")


def _plain_file_identity(path: Path, *, label: str) -> dict[str, Any]:
    """Match the existing on-policy sidecars' path/hash/row-count identity."""

    _assert_regular_file(path, label=label)
    payload = path.read_bytes()
    return {
        "path": str(path),
        "content_sha256": _sha256_bytes(payload),
        "row_count": sum(1 for line in payload.splitlines() if line.strip()),
    }


def _validate_preflight_base_snapshot_attestation(root: Path) -> dict[str, Any]:
    segment = segment_for_index(0)
    path = preflight_base_snapshot_attestation_path(root)
    document = _json_object(path, label="preflight base snapshot attestation")
    _validate_no_secret_keys(document, path="preflight base snapshot attestation")
    _expect(document.get("schema"), BASE_SNAPSHOT_SCHEMA, label="preflight base snapshot schema")
    _expect(document.get("condition"), CONDITION, label="preflight base snapshot condition")
    _expect(document.get("segment"), segment.global_index, label="preflight base snapshot segment")
    _expect(document.get("repo_id"), MODEL_REPO, label="preflight base snapshot repository")
    _expect(document.get("revision"), MODEL_REVISION, label="preflight base snapshot revision")
    _expect(document.get("hf_hub_offline"), True, label="preflight base snapshot offline policy")
    _expect(document.get("transformers_offline"), True, label="preflight Transformers offline policy")
    snapshot = document.get("snapshot")
    main_ref = document.get("main_ref")
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("path"), str):
        raise ContractError("preflight base snapshot attestation has no resolved snapshot path")
    if not isinstance(main_ref, Mapping):
        raise ContractError("preflight base snapshot attestation has no main-ref record")
    _expect(main_ref.get("commit"), MODEL_REVISION, label="preflight base snapshot main-ref commit")
    return file_identity(path, label="preflight base snapshot attestation")


def _validate_preflight_source_attestation(root: Path) -> dict[str, Any]:
    path = preflight_source_attestation_path(root)
    document = _json_object(path, label="preflight recovered-source attestation")
    _validate_no_secret_keys(document, path="preflight recovered-source attestation")
    _expect(
        document.get("schema"),
        "qwen35-no-cot-recovered-source-attestation-v1",
        label="preflight recovered-source schema",
    )
    _expect(document.get("model"), MODEL_REPO, label="preflight recovered-source model")
    source = document.get("source")
    manifest = document.get("source_manifest")
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
        raise ContractError("preflight recovered-source attestation has no source identity")
    if not isinstance(manifest, Mapping) or not isinstance(manifest.get("path"), str):
        raise ContractError("preflight recovered-source attestation has no manifest identity")
    source_path = _under_root(root, source["path"])
    manifest_path = _under_root(root, manifest["path"])
    _expect(
        _plain_file_identity(source_path, label="preflight recovered source"),
        dict(source),
        label="preflight recovered source identity",
    )
    _expect(
        _plain_file_identity(manifest_path, label="preflight recovered-source manifest"),
        dict(manifest),
        label="preflight recovered-source manifest identity",
    )
    return file_identity(path, label="preflight recovered-source attestation")


def _validate_preflight_worker_parity_attestation(root: Path) -> dict[str, Any]:
    path = preflight_worker_parity_attestation_path(root)
    document = _json_object(path, label="preflight worker-parity attestation")
    _validate_no_secret_keys(document, path="preflight worker-parity attestation")
    _expect(
        document.get("schema"),
        "qwen35-rollout-worker-parity-attestation-v1",
        label="preflight worker-parity schema",
    )
    _expect(document.get("model"), MODEL_REPO, label="preflight worker-parity model")
    worker_gpus = document.get("worker_gpus")
    if not isinstance(worker_gpus, list) or len(worker_gpus) != 3:
        raise ContractError("preflight worker-parity attestation must bind exactly three rollout workers")
    if not isinstance(document.get("worker_engine_kwargs"), Mapping):
        raise ContractError("preflight worker-parity attestation has no worker engine contract")
    if not isinstance(document.get("aggregate_effect_parity"), Mapping):
        raise ContractError("preflight worker-parity attestation has no nonzero aggregate effect proof")
    if not isinstance(document.get("per_worker_effect_parity"), list) or len(document["per_worker_effect_parity"]) != 3:
        raise ContractError("preflight worker-parity attestation has incomplete per-worker evidence")
    return file_identity(path, label="preflight worker-parity attestation")


def _validate_preflight_target_attestation(root: Path, preflight_plan: Path) -> dict[str, Any]:
    segment = segment_for_index(0)
    path = preflight_target_attestation_path(root)
    document = _json_object(path, label="preflight on-policy target attestation")
    _validate_no_secret_keys(document, path="preflight on-policy target attestation")
    _expect(document.get("schema"), "qwen35-onpolicy-target-attestation-v1", label="preflight target schema")
    _expect(document.get("target"), segment.target, label="preflight target")
    _expect(document.get("experiment_name"), PREFLIGHT_EXPERIMENT, label="preflight target experiment")
    _expect(document.get("run_name"), PREFLIGHT_RUN_NAME, label="preflight target run name")
    _expect(document.get("topology_profile"), TOPOLOGY_PROFILE, label="preflight target topology profile")
    _expect(
        document.get("authored_plan"),
        _plain_file_identity(preflight_plan, label="preflight target plan"),
        label="preflight target plan identity",
    )
    _expect(
        document.get("source_attestation"),
        _plain_file_identity(preflight_source_attestation_path(root), label="preflight target source attestation"),
        label="preflight target source-attestation identity",
    )
    _expect(
        document.get("worker_parity_attestation"),
        _plain_file_identity(preflight_worker_parity_attestation_path(root), label="preflight target worker attestation"),
        label="preflight target worker-attestation identity",
    )
    if not isinstance(document.get("compiled_entry"), Mapping) or not isinstance(document.get("child_argv"), list):
        raise ContractError("preflight target attestation has no compiled child command")
    return file_identity(path, label="preflight on-policy target attestation")


def _preflight_execution_code_identities(root: Path) -> dict[str, dict[str, Any]]:
    """Hash the launcher surface that turns a successful probe into a job chain."""

    paths = (
        "infra/isambard/rmct256_convergence_segment_contract.py",
        "infra/isambard/rmct256_convergence_lora_fingerprint.py",
        "infra/isambard/rmct256_runtime_policy_receipt.py",
        "infra/isambard/rmct256_convergence_preflight.py",
        "infra/isambard/extract_rmct256_convergence_metrics.py",
        "infra/isambard/run_qwen35_rmct256_convergence_segment.sh",
        "infra/isambard/run_qwen35_rmct256_convergence_segment.sbatch",
        "infra/isambard/preflight_qwen35_rmct256_convergence.sbatch",
        "infra/isambard/submit_qwen35_rmct256_convergence_chain.sh",
    )
    return {
        relative: file_identity(_under_root(root, relative), label=f"preflight execution code {relative}")
        for relative in paths
    }


def _assert_production_chain_pristine(root: Path) -> None:
    residue: list[str] = []
    for index in range(MAX_SEGMENTS):
        segment = segment_for_index(index)
        for path in _residue_paths(root, segment):
            residue.append(str(path))
    if residue:
        raise ContractError(
            "a dedicated preflight can authorize only a clean convergence chain; found production residue: "
            + ", ".join(residue)
        )


def _preflight_success_document(root: Path, plan: Path) -> dict[str, Any]:
    """Collect the immutable proof that authorizes first-chain submission."""

    segment = segment_for_index(0)
    _assert_production_chain_pristine(root)
    production = validate_segment_plan(root, plan, segment)
    preflight_plan = preflight_plan_path(root)
    preflight = validate_preflight_plan(root, plan, preflight_plan)
    base = _validate_preflight_base_snapshot_attestation(root)
    lora = _validate_lora_fingerprint_attestation(
        root,
        plan,
        segment,
        path=preflight_lora_fingerprint_path(root),
        base_snapshot_path=preflight_base_snapshot_attestation_path(root),
    )
    runtime = _validate_runtime_policy_receipt(
        root,
        plan,
        segment,
        path=preflight_runtime_policy_receipt_path(root),
        require_preflight_reference=False,
    )
    source = _validate_preflight_source_attestation(root)
    workers = _validate_preflight_worker_parity_attestation(root)
    target = _validate_preflight_target_attestation(root, preflight_plan)
    return {
        "schema": PREFLIGHT_RECEIPT_SCHEMA,
        "condition": CONDITION,
        "preflight": {
            "experiment": PREFLIGHT_EXPERIMENT,
            "run_name": PREFLIGHT_RUN_NAME,
            "target": segment.target,
            "topology_profile": TOPOLOGY_PROFILE,
            "gpu_count": 4,
        },
        "production_plan": file_identity(plan, label="preflight production plan"),
        "production_entry_sha256": _canonical_json_sha256(production["entry"]),
        "isolated_preflight_plan": preflight["plan"],
        "base_snapshot_attestation": base,
        "lora_fingerprint_attestation": lora,
        "runtime_policy_receipt": runtime,
        "source_attestation": source,
        "worker_parity_attestation": workers,
        "target_attestation": target,
        "execution_code": _preflight_execution_code_identities(root),
    }


def seal_preflight(root: Path, plan: Path) -> dict[str, Any]:
    """Write the single immutable success receipt after the real 4-GPU probe."""

    path = preflight_receipt_path(root)
    if path.exists() or path.is_symlink():
        document = validate_preflight_receipt(root, plan)
        return {"status": "resumed", "receipt": str(path), "condition": document["condition"]}
    document = _preflight_success_document(root, plan)
    status = _write_immutable_json(path, document, label="RMCT256 dedicated preflight success receipt")
    return {"status": status, "receipt": str(path), "condition": CONDITION}


def validate_preflight_receipt(root: Path, plan: Path) -> dict[str, Any]:
    """Fail closed before any ``sbatch`` call if the probe is missing or stale."""

    path = preflight_receipt_path(root)
    recorded = _json_object(path, label="RMCT256 dedicated preflight success receipt")
    _validate_no_secret_keys(recorded, path="RMCT256 dedicated preflight success receipt")
    expected = _preflight_success_document(root, plan)
    _expect(recorded, expected, label="RMCT256 dedicated preflight success receipt")
    return recorded


def _validate_output_state(root: Path, segment: Segment) -> dict[str, Any]:
    path = output_state_path(root, segment)
    document = _json_object(path, label="target output state")
    _expect(document.get("schema_version"), 1, label="target output schema version")
    _expect(document.get("experiment"), CONDITION, label="target output experiment")
    _expect(document.get("execution_target"), segment.target, label="target output target")
    checkpoints = document.get("training_checkpoints")
    if not isinstance(checkpoints, Mapping) or len(checkpoints) != 1:
        raise ContractError("target output must publish exactly one training checkpoint")
    actual = next(iter(checkpoints.values()))
    _expect(actual, checkpoint_uri(root, segment), label="target output final checkpoint")
    return file_identity(path, label="target output state")


def _receipt_document(
    *,
    root: Path,
    plan: Path,
    segment: Segment,
    plan_contract: Mapping[str, Any],
    parent: dict[str, Any] | None,
    target_attestation: dict[str, Any],
    base_snapshot_attestation: dict[str, Any],
    lora_fingerprint_attestation: dict[str, Any],
    runtime_policy_receipt: dict[str, Any],
    base_snapshot: dict[str, Any],
    output_state: dict[str, Any],
    checkpoint_artifacts: dict[str, Any],
    loop_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "condition": CONDITION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segment": {
            "global_segment_index": segment.global_index,
            "pass_index": segment.pass_index,
            "segment_index": segment.segment_index,
            "target": segment.target,
            "run_name": segment.run_name,
            "row_offset": segment.row_offset,
            "questions": ROWS_PER_SEGMENT,
            "updates": UPDATES_PER_SEGMENT,
            "checkpoint_step": segment.checkpoint_step,
            "worker_seed_base": segment.worker_seed_base,
        },
        "plan": file_identity(plan, label="authored convergence plan"),
        "segment_manifest": plan_contract["metadata"].get("segment_manifest"),
        "parent": parent,
        "base_snapshot": base_snapshot,
        "base_snapshot_attestation": base_snapshot_attestation,
        "lora_fingerprint_attestation": lora_fingerprint_attestation,
        "runtime_policy_receipt": runtime_policy_receipt,
        "target_attestation": target_attestation,
        "target_output_state": output_state,
        "checkpoint": checkpoint_artifacts,
        "loop_state": loop_state,
        "continuation": {
            "mode": "pinned_base_snapshot" if segment.previous is None else "optimizer_data_segment",
            "coordinator_rng_restored": bool(segment.previous is not None),
            # The strict local state captures coordinator RNG, but vLLM's
            # private worker streams have no supported snapshot/restore API.
            "vllm_worker_rng_restored": False,
        },
    }


def _validate_no_secret_keys(value: Any, *, path: str = "receipt") -> None:
    """Reject accidental environment/credential dumps in a custody record."""

    forbidden = re.compile(r"(?:api[_-]?key|secret|token|password|credential)", re.IGNORECASE)
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str) and forbidden.search(key):
                raise ContractError(f"{path} must not contain a credential-like key: {key}")
            _validate_no_secret_keys(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_no_secret_keys(nested, path=f"{path}[{index}]")


def validate_receipt(
    root: Path,
    path: Path,
    *,
    expected_segment: Segment | None = None,
    _seen: set[Path] | None = None,
) -> dict[str, Any]:
    """Validate an immutable receipt and its complete direct-parent chain."""

    resolved = path.resolve()
    _assert_regular_file(resolved, label="segment receipt")
    seen = set() if _seen is None else _seen
    if resolved in seen:
        raise ContractError(f"segment receipt parent cycle: {resolved}")
    seen.add(resolved)
    document = _json_object(resolved, label="segment receipt")
    _validate_no_secret_keys(document)
    _expect(document.get("schema"), RECEIPT_SCHEMA, label="segment receipt schema")
    _expect(document.get("condition"), CONDITION, label="segment receipt condition")
    segment_record = document.get("segment")
    if not isinstance(segment_record, Mapping):
        raise ContractError("segment receipt has no segment record")
    index = _as_exact_int(segment_record.get("global_segment_index"), label="receipt global segment index")
    segment = segment_for_index(index)
    if expected_segment is not None and segment != expected_segment:
        raise ContractError("segment receipt belongs to a different static segment")
    expected_segment_record = {
        "global_segment_index": segment.global_index,
        "pass_index": segment.pass_index,
        "segment_index": segment.segment_index,
        "target": segment.target,
        "run_name": segment.run_name,
        "row_offset": segment.row_offset,
        "questions": ROWS_PER_SEGMENT,
        "updates": UPDATES_PER_SEGMENT,
        "checkpoint_step": segment.checkpoint_step,
        "worker_seed_base": segment.worker_seed_base,
    }
    _expect(dict(segment_record), expected_segment_record, label="segment receipt identity")
    plan_record = document.get("plan")
    if not isinstance(plan_record, Mapping) or not isinstance(plan_record.get("path"), str):
        raise ContractError("segment receipt has no plan identity")
    plan = _under_root(root, plan_record["path"])
    _same_identity(file_identity(plan, label="receipt plan"), dict(plan_record), label="receipt plan")
    validate_segment_plan(root, plan, segment)

    _same_identity(
        _validate_lora_fingerprint_attestation(root, plan, segment),
        document.get("lora_fingerprint_attestation"),
        label="LoRA fingerprint attestation",
    )
    _same_identity(
        _validate_runtime_policy_receipt(root, plan, segment, require_preflight_reference=True),
        document.get("runtime_policy_receipt"),
        label="runtime-policy receipt",
    )
    _same_identity(
        _validate_base_snapshot_attestation(root, segment),
        document.get("base_snapshot_attestation"),
        label="base snapshot attestation",
    )
    _expect(
        document.get("base_snapshot"),
        _base_snapshot_record(root, segment),
        label="base snapshot resolved cache record",
    )
    _same_identity(
        _validate_target_attestation(root, segment),
        document.get("target_attestation"),
        label="target attestation",
    )
    _same_identity(
        _validate_output_state(root, segment),
        document.get("target_output_state"),
        label="target output state",
    )
    artifacts, loop = _validate_final_checkpoint(root, segment)
    _expect(document.get("checkpoint"), artifacts, label="receipt checkpoint artifacts")
    _expect(document.get("loop_state"), loop, label="receipt loop state")

    continuation = document.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ContractError("segment receipt has no continuation record")
    expected_mode = "pinned_base_snapshot" if segment.previous is None else "optimizer_data_segment"
    _expect(continuation.get("mode"), expected_mode, label="receipt continuation mode")
    _expect(continuation.get("vllm_worker_rng_restored"), False, label="receipt worker RNG restoration claim")
    _expect(
        continuation.get("coordinator_rng_restored"), bool(segment.previous is not None), label="receipt coordinator RNG claim"
    )

    parent_record = document.get("parent")
    if segment.previous is None:
        _expect(parent_record, None, label="first-segment receipt parent")
    else:
        previous = segment.previous
        assert previous is not None
        if not isinstance(parent_record, Mapping):
            raise ContractError("continuation receipt has no parent record")
        parent_identity = parent_record.get("receipt")
        if not isinstance(parent_identity, Mapping) or not isinstance(parent_identity.get("path"), str):
            raise ContractError("continuation receipt has no parent receipt identity")
        parent_path = _under_root(root, parent_identity["path"])
        _same_identity(file_identity(parent_path, label="parent receipt"), dict(parent_identity), label="parent receipt")
        parent_document = validate_receipt(root, parent_path, expected_segment=previous, _seen=seen)
        parent_checkpoint = parent_document.get("checkpoint")
        if not isinstance(parent_checkpoint, Mapping):
            raise ContractError("parent receipt has no checkpoint record")
        _expect(parent_checkpoint.get("checkpoint"), checkpoint_uri(root, previous), label="parent receipt checkpoint URI")
        _expect(parent_record.get("checkpoint"), checkpoint_uri(root, previous), label="receipt parent checkpoint URI")
    return document


def _residue_paths(root: Path, segment: Segment) -> list[Path]:
    """Return all state that makes a missing receipt an unsafe retry."""

    result: list[Path] = []
    directory = run_root(root, segment)
    if directory.exists():
        _assert_regular_directory(directory, label="segment run directory")
        # Every write below this unique namespace is evidence of a previous
        # preflight/training attempt.  The chain never resumes an incomplete
        # namespace, including a job killed after the child but before sealing.
        result.append(directory)
    state = output_state_path(root, segment)
    if state.exists() or state.is_symlink():
        result.append(state)
    return result


def guard_segment(root: Path, plan: Path, segment: Segment) -> dict[str, Any]:
    """Return ``proceed`` or an idempotent completed no-op; otherwise fail."""

    validate_segment_plan(root, plan, segment)
    receipt = receipt_path(root, segment)
    if receipt.exists() or receipt.is_symlink():
        validate_receipt(root, receipt, expected_segment=segment)
        return {"action": "completed", "receipt": str(receipt), "segment": segment.global_index}
    residue = _residue_paths(root, segment)
    if residue:
        raise ContractError(
            "refusing to reuse a segment namespace with production/preflight residue but no valid receipt: "
            + ", ".join(str(path) for path in residue)
        )
    if segment.previous is not None:
        previous_receipt = receipt_path(root, segment.previous)
        validate_receipt(root, previous_receipt, expected_segment=segment.previous)
    return {"action": "proceed", "receipt": str(receipt), "segment": segment.global_index}


def validate_parent(root: Path, plan: Path, segment: Segment) -> dict[str, Any]:
    """Revalidate the direct continuation boundary immediately before launch.

    ``guard_segment`` performs this same validation before any expensive
    preflight.  It is deliberately repeated after target attestation because
    an operator or a faulty process could otherwise alter the parent's
    receipt/checkpoint while a child is spending time on its worker probe.
    ``validate_receipt`` recursively re-hashes the complete parent chain and
    validates the parent final checkpoint, optimizer state, and loop state.
    """

    validate_segment_plan(root, plan, segment)
    if segment.previous is None:
        return {"status": "pinned-base", "segment": segment.global_index}
    previous = segment.previous
    assert previous is not None
    receipt = receipt_path(root, previous)
    document = validate_receipt(root, receipt, expected_segment=previous)
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ContractError("parent receipt has no checkpoint record")
    _expect(checkpoint.get("checkpoint"), checkpoint_uri(root, previous), label="validated parent checkpoint URI")
    return {
        "status": "validated",
        "segment": segment.global_index,
        "parent_segment": previous.global_index,
        "parent_receipt": str(receipt),
        "parent_checkpoint": checkpoint_uri(root, previous),
    }


def seal_segment(root: Path, plan: Path, segment: Segment) -> dict[str, Any]:
    """Seal the CPU-only completion receipt after the runner has returned 0."""

    receipt = receipt_path(root, segment)
    if receipt.exists() or receipt.is_symlink():
        validate_receipt(root, receipt, expected_segment=segment)
        return {"status": "resumed", "receipt": str(receipt), "segment": segment.global_index}

    plan_contract = validate_segment_plan(root, plan, segment)
    parent_record: dict[str, Any] | None = None
    if segment.previous is not None:
        previous = segment.previous
        assert previous is not None
        previous_receipt = receipt_path(root, previous)
        parent_document = validate_receipt(root, previous_receipt, expected_segment=previous)
        parent_checkpoint = parent_document.get("checkpoint")
        if not isinstance(parent_checkpoint, Mapping):
            raise ContractError("parent receipt has no checkpoint record")
        _expect(parent_checkpoint.get("checkpoint"), checkpoint_uri(root, previous), label="sealed parent checkpoint URI")
        parent_record = {
            "receipt": file_identity(previous_receipt, label="parent receipt"),
            "checkpoint": checkpoint_uri(root, previous),
        }
    base_snapshot = _base_snapshot_record(root, segment)
    base_attestation = base_snapshot["attestation"]
    lora_fingerprint_attestation = _validate_lora_fingerprint_attestation(root, plan, segment)
    runtime_policy_receipt = _validate_runtime_policy_receipt(root, plan, segment, require_preflight_reference=True)
    target_attestation = _validate_target_attestation(root, segment)
    output_state = _validate_output_state(root, segment)
    artifacts, loop = _validate_final_checkpoint(root, segment)
    document = _receipt_document(
        root=root,
        plan=plan,
        segment=segment,
        plan_contract=plan_contract,
        parent=parent_record,
        target_attestation=target_attestation,
        base_snapshot_attestation=base_attestation,
        lora_fingerprint_attestation=lora_fingerprint_attestation,
        runtime_policy_receipt=runtime_policy_receipt,
        base_snapshot=base_snapshot,
        output_state=output_state,
        checkpoint_artifacts=artifacts,
        loop_state=loop,
    )
    _validate_no_secret_keys(document)
    # Receipts carry a timestamp, so a race must be validated rather than
    # byte-compared: only one writer can create it and a later job validates it.
    receipt.parent.mkdir(parents=True, exist_ok=True)
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        status = "written"
    except FileExistsError:
        validate_receipt(root, receipt, expected_segment=segment)
        status = "resumed"
    return {"status": status, "receipt": str(receipt), "segment": segment.global_index}


def _cache_ref_path() -> Path:
    try:
        from huggingface_hub import constants
    except ImportError as exc:  # pragma: no cover - production bootstrap error
        raise ContractError("huggingface_hub is required for the offline base-model gate") from exc
    cache = Path(os.environ.get("HF_HUB_CACHE") or constants.HF_HUB_CACHE)
    return cache / "models--Qwen--Qwen3.5-9B" / "refs" / "main"


def _hf_cache_blob_identity(path: Path, *, blobs: Path, label: str) -> dict[str, Any]:
    """Identify one HF snapshot entry without trusting arbitrary symlinks.

    Hugging Face snapshots normally contain symlinks into the repository-local
    ``blobs/`` directory.  Those links are legitimate cache layout, but a
    general ``Path.resolve`` would also follow an attacker-controlled escape
    elsewhere on the filesystem.  This narrowly accepts *one direct symlink*
    whose target is a regular, non-symlink file strictly beneath the exact
    Qwen cache's ``blobs/`` directory, then records both the link spelling and
    the resolved content identity.
    """

    if not path.is_symlink():
        raise ContractError(f"{label} must be a Hugging Face snapshot symlink: {path}")
    _assert_regular_directory(blobs, label="Hugging Face Qwen blobs directory")
    try:
        link_target = os.readlink(path)
    except OSError as exc:  # pragma: no cover - race/protected filesystem
        raise ContractError(f"cannot read {label} symlink: {path}") from exc
    target_path = Path(link_target)
    direct_target = target_path if target_path.is_absolute() else path.parent / target_path
    # A snapshot entry must point directly at a blob, rather than using a
    # second link whose eventual resolution could change behind this proof.
    if direct_target.is_symlink():
        raise ContractError(f"{label} symlink target must not itself be a symlink: {path}")
    try:
        resolved = direct_target.resolve(strict=True)
        blobs_root = blobs.resolve(strict=True)
        resolved.relative_to(blobs_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContractError(f"{label} symlink target escapes the exact Qwen blobs directory: {path}") from exc
    if resolved == blobs_root or resolved.is_symlink() or not resolved.is_file():
        raise ContractError(f"{label} symlink target must be a regular Qwen blob file: {path}")
    payload = resolved.read_bytes()
    return {
        "link_path": str(path),
        "link_target": link_target,
        "resolved_path": str(resolved),
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _snapshot_attestation_document(root: Path, segment: Segment) -> dict[str, Any]:
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ContractError("HF_HUB_OFFLINE=1 is required before model resolution")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ContractError("TRANSFORMERS_OFFLINE=1 is required before model resolution")
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - production bootstrap error
        raise ContractError("huggingface_hub and transformers are required for the base-model gate") from exc

    ref = _cache_ref_path()
    _assert_regular_file(ref, label="Hugging Face Qwen main ref")
    ref_commit = ref.read_text(encoding="utf-8").strip()
    _expect(ref_commit, MODEL_REVISION, label="Hugging Face Qwen main ref commit")
    try:
        pinned = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)).resolve()
        default = Path(snapshot_download(MODEL_REPO, revision="main", local_files_only=True)).resolve()
    except (OSError, ValueError) as exc:
        raise ContractError(f"offline Qwen snapshot is unavailable: {exc}") from exc
    _assert_regular_directory(pinned, label="pinned Qwen snapshot")
    _expect(pinned.name, MODEL_REVISION, label="pinned Qwen snapshot directory name")
    _expect(default, pinned, label="default Qwen snapshot resolution")
    expected_snapshot = ref.parent.parent / "snapshots" / MODEL_REVISION
    _expect(pinned, expected_snapshot.resolve(), label="pinned Qwen cache snapshot path")
    blobs = ref.parent.parent / "blobs"
    config_path = pinned / "config.json"
    config_identity = _hf_cache_blob_identity(config_path, blobs=blobs, label="pinned Qwen config")
    tokenizer_candidates = [pinned / name for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")]
    tokenizer_files = [
        _hf_cache_blob_identity(path, blobs=blobs, label="pinned tokenizer file")
        for path in tokenizer_candidates
        if path.exists() or path.is_symlink()
    ]
    if not tokenizer_files:
        raise ContractError("pinned Qwen snapshot has no tokenizer artifact")
    weights = sorted(pinned.glob("model*.safetensors")) + sorted(pinned.glob("pytorch_model*.bin"))
    if not weights:
        raise ContractError("pinned Qwen snapshot has no model weight artifact")
    weight_files = [_hf_cache_blob_identity(path, blobs=blobs, label="pinned Qwen model weight") for path in weights]
    try:
        config = AutoConfig.from_pretrained(str(pinned), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(pinned), local_files_only=True)
    except (OSError, ValueError) as exc:
        raise ContractError(f"coordinator/tokenizer cannot resolve the pinned offline Qwen snapshot: {exc}") from exc
    return {
        "schema": BASE_SNAPSHOT_SCHEMA,
        "condition": CONDITION,
        "segment": segment.global_index,
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "hf_hub_offline": True,
        "transformers_offline": True,
        "main_ref": {"path": str(ref), "commit": ref_commit, "sha256": _sha256_bytes(ref.read_bytes())},
        "snapshot": {
            "path": str(pinned),
            "config": config_identity,
            "tokenizer_files": tokenizer_files,
            "weight_files": weight_files,
        },
        "consumers": {
            "coordinator": {"model": MODEL_REPO, "revision": "main", "resolved_snapshot": str(pinned)},
            "tokenizer": {
                "model": MODEL_REPO,
                "revision": "main",
                "resolved_snapshot": str(pinned),
                "class": tokenizer.__class__.__name__,
            },
            "rollout_workers": [
                {"logical_gpu": gpu, "model": MODEL_REPO, "revision": "main", "resolved_snapshot": str(pinned)}
                for gpu in (1, 2, 3)
            ],
            "config_class": config.__class__.__name__,
        },
    }


def attest_base_snapshot(
    root: Path,
    plan: Path,
    segment: Segment,
    *,
    check_only: bool,
    output: Path | None = None,
) -> dict[str, Any]:
    validate_segment_plan(root, plan, segment)
    document = _snapshot_attestation_document(root, segment)
    path = base_snapshot_attestation_path(root, segment) if output is None else _under_root(root, output)
    status = "checked" if check_only else _write_immutable_json(path, document, label="base snapshot attestation")
    return {"status": status, "path": str(path), "attestation": document}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "validate-plan",
        "guard",
        "validate-parent",
        "seal",
        "base-snapshot",
        "preflight-plan",
        "seal-preflight",
        "validate-preflight",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--repo-root", type=Path, required=True)
        command.add_argument("--plan", type=Path, help="authored static convergence YAML")
        command.add_argument("--segment-index", type=int, required=True)
        if name == "base-snapshot":
            command.add_argument("--check-only", action="store_true", help="verify cache without writing its attestation")
            command.add_argument("--output", type=Path, help="dedicated immutable attestation path under the repository")
        if name == "preflight-plan":
            command.add_argument("--output", type=Path, help="dedicated immutable preflight-plan path under the repository")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = _absolute_root(args.repo_root)
        plan = plan_path(root, args.plan)
        segment = segment_for_index(args.segment_index)
        if args.command == "validate-plan":
            result = {"status": "validated", "segment": segment.global_index, "target": segment.target, "run_name": segment.run_name}
            validate_segment_plan(root, plan, segment)
        elif args.command == "guard":
            result = guard_segment(root, plan, segment)
        elif args.command == "validate-parent":
            result = validate_parent(root, plan, segment)
        elif args.command == "seal":
            result = seal_segment(root, plan, segment)
        elif args.command == "base-snapshot":
            result = attest_base_snapshot(root, plan, segment, check_only=args.check_only, output=args.output)
        elif args.command == "preflight-plan":
            result = write_preflight_plan(root, plan, segment, output=args.output)
        elif args.command == "seal-preflight":
            if segment.global_index != 0:
                raise ContractError("the dedicated RMCT256 preflight receipt is defined only for segment index 0")
            result = seal_preflight(root, plan)
        elif args.command == "validate-preflight":
            if segment.global_index != 0:
                raise ContractError("the dedicated RMCT256 preflight receipt is defined only for segment index 0")
            document = validate_preflight_receipt(root, plan)
            result = {
                "status": "validated",
                "receipt": str(preflight_receipt_path(root)),
                "condition": document["condition"],
            }
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(args.command)
    except (ContractError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
