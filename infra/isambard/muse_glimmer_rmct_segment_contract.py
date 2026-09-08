"""Fail-closed custody and convergence checks for Muse Glimmer RMCT segments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ctm.backends.local.muse_glimmer import (
    MODEL_ID,
    MODEL_REVISION,
    PARITY_ATTESTATION_SCHEMA,
    TRANSFORMERS_VERSION,
    VLLM_COMMIT,
    file_sha256,
)
from experiments.muse_glimmer_rmct_replication import plan


RECEIPT_SCHEMA = "muse-glimmer-rmct-segment-receipt-v1"
CONVERGENCE_SCHEMA = "muse-glimmer-rmct-convergence-window-v1"
LAUNCH_SCHEMA = "muse-glimmer-rmct-segment-launch-v1"
EXPECTED_QUESTIONS_PER_UPDATE = 2
EXPECTED_TRAINING_PERTURBATIONS = (1, 2)
EXPECTED_OBSERVATIONS_PER_WINDOW = (
    plan.UPDATES_PER_SEGMENT * EXPECTED_QUESTIONS_PER_UPDATE * len(EXPECTED_TRAINING_PERTURBATIONS)
)
MINIMUM_COVERAGE = 0.85
MAXIMUM_WEIGHTED_ABS_GAP = 0.10
MAXIMUM_ABS_CHANGE = 0.01
FIRST_ELIGIBLE_OPTIMIZER_STEP = 32


class ContractError(RuntimeError):
    """A production Muse segment cannot prove its authored contract."""


def _root(value: str | Path) -> Path:
    root = Path(value).resolve()
    if root.is_symlink() or not root.is_dir():
        raise ContractError(f"repository root must be a regular directory: {root}")
    return root


def _index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < plan.TOTAL_SEGMENTS:
        raise ContractError(f"segment index must be in [0, {plan.TOTAL_SEGMENTS - 1}]")
    return value


def _regular_file(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{label} must be a regular file: {path}")
    return path


def _regular_directory(path: Path, *, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ContractError(f"{label} must be a regular directory: {path}")
    return path


def _json(path: Path, *, label: str) -> dict[str, Any]:
    _regular_file(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise ContractError(f"{label} must not be a symlink: {path}")
    resolved = _regular_file(path.resolve(), label=label)
    return {"path": str(resolved), "sha256": file_sha256(resolved), "size_bytes": resolved.stat().st_size}


def _same_identity(expected: Mapping[str, Any], actual: Any, *, label: str) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise ContractError(f"{label} identity changed")


def _expect(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} is {actual!r}, expected {expected!r}")


def control_root(root: Path, segment_index: int) -> Path:
    return root / "artifacts" / "muse-glimmer-rmct-segments-20260824" / f"segment-{_index(segment_index):03d}"


def launch_path(root: Path, segment_index: int) -> Path:
    return control_root(root, segment_index) / "launch.json"


def receipt_path(root: Path, segment_index: int) -> Path:
    return control_root(root, segment_index) / "receipt.json"


def convergence_path(root: Path, segment_index: int) -> Path:
    return control_root(root, segment_index) / "convergence.json"


def run_root(root: Path, segment_index: int) -> Path:
    return root / "logs" / plan.CONDITION_NAME / plan.run_name(_index(segment_index))


def checkpoint_path(root: Path, segment_index: int) -> Path:
    return plan.final_checkpoint_path(root, _index(segment_index))


def preflight_path(root: Path) -> Path:
    return root / plan.PARITY_ATTESTATION


def _validate_source_manifest(root: Path, manifest: Any) -> None:
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "muse-glimmer-source-manifest-v1":
        raise ContractError("preflight has no Muse source manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ContractError("preflight source manifest has no files")
    compact = []
    seen: set[str] = set()
    for record in files:
        if not isinstance(record, Mapping):
            raise ContractError("preflight source manifest contains a non-object")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ContractError("preflight source manifest paths must be non-empty and unique")
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"preflight source file escapes repository: {relative}") from exc
        expected = _identity(path, label=f"preflight source {relative}")
        _expect(record.get("path"), expected["path"], label=f"preflight source path {relative}")
        _expect(record.get("sha256"), expected["sha256"], label=f"preflight source hash {relative}")
        _expect(record.get("size_bytes"), expected["size_bytes"], label=f"preflight source size {relative}")
        compact.append(
            {"relative_path": relative, "sha256": expected["sha256"], "size_bytes": expected["size_bytes"]}
        )
    digest = hashlib.sha256(_canonical(compact)).hexdigest()
    _expect(manifest.get("sha256"), digest, label="preflight source-manifest digest")


def validate_preflight(root_value: str | Path) -> dict[str, Any]:
    root = _root(root_value)
    path = preflight_path(root)
    document = _json(path, label="Muse parity preflight")
    _expect(document.get("schema"), PARITY_ATTESTATION_SCHEMA, label="preflight schema")
    model = document.get("model")
    if not isinstance(model, Mapping):
        raise ContractError("preflight has no model record")
    _expect(model.get("repo_id"), MODEL_ID, label="preflight model repo")
    _expect(model.get("revision"), MODEL_REVISION, label="preflight model revision")
    snapshot = Path(str(model.get("snapshot_path", ""))).resolve()
    _regular_directory(snapshot, label="pinned Muse snapshot")
    _expect(snapshot.name, MODEL_REVISION, label="pinned Muse snapshot directory")
    _expect(model.get("config_sha256"), file_sha256(snapshot / "config.json"), label="snapshot config hash")

    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ContractError("preflight has no runtime record")
    _expect(runtime.get("transformers_version"), TRANSFORMERS_VERSION, label="preflight Transformers")
    _expect(runtime.get("vllm_commit"), VLLM_COMMIT, label="preflight vLLM commit")
    receipts = runtime.get("receipts")
    if not isinstance(receipts, Mapping):
        raise ContractError("preflight has no runtime receipts")
    for key, filename in (
        ("runtime", "runtime.json"),
        ("pip_freeze", "pip-freeze.txt"),
        ("model_snapshot", "model-snapshot.json"),
    ):
        expected = _identity(
            root / plan.RUNTIME_RECEIPT_DIR / filename,
            label=f"Muse runtime {key}",
        )
        _same_identity(expected, receipts.get(key), label=f"Muse runtime {key}")

    scientific = document.get("scientific_contract")
    if not isinstance(scientific, Mapping):
        raise ContractError("preflight has no scientific contract")
    _expect(scientific.get("frozen_spec_sha256"), plan.FROZEN_SPEC_SHA256, label="frozen spec hash")
    _same_identity(_identity(root / plan.DATA_PATH, label="frozen training data"), scientific.get("data"), label="data")
    _same_identity(
        _identity(root / plan.MANIFEST_PATH, label="frozen training manifest"),
        scientific.get("manifest"),
        label="data manifest",
    )
    _expect(
        scientific.get("generation"),
        {"max_tokens": None, "termination": "eos_only", "non_eos_termination_policy": "fail_run"},
        label="preflight no-cap generation contract",
    )
    _validate_source_manifest(root, document.get("source_manifest"))

    probe = document.get("probe")
    if not isinstance(probe, Mapping):
        raise ContractError("preflight has no parity probe")
    _expect(probe.get("max_tokens"), None, label="preflight generation cap")
    _expect(probe.get("termination"), "eos_only", label="preflight termination")
    _expect(probe.get("non_eos_termination_count"), 0, label="preflight non-EOS terminations")
    _expect(probe.get("source_prompt_count"), 6, label="preflight source prompt count")
    _expect(probe.get("score_row_count"), 9, label="preflight score row count")
    _expect(probe.get("source_indices")[:3], [probe.get("anchor_source_index")] * 3, label="preflight worker anchors")
    _expect(probe.get("source_indices")[3:], list(range(6)), label="preflight diverse source prompts")
    _expect(probe.get("request_count"), 18, label="preflight request count")
    for label in ("aggregate_base_parity", "aggregate_policy_parity", "aggregate_effect_parity"):
        result = document.get(label)
        if not isinstance(result, Mapping) or result.get("passed") is not True:
            raise ContractError(f"preflight {label} did not pass")
    workers = document.get("per_worker_effect_parity")
    if not isinstance(workers, list) or len(workers) != 3 or any(
        not isinstance(item, Mapping)
        or not isinstance(item.get("effect_parity"), Mapping)
        or item["effect_parity"].get("passed") is not True
        for item in workers
    ):
        raise ContractError("preflight did not pass effect parity on all three workers")
    return document


def _checkpoint_artifacts(root: Path, segment_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = _index(segment_index)
    directory = _regular_directory(checkpoint_path(root, index), label="final Muse checkpoint")
    required = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "manifest.json")
    for name in required:
        _regular_file(directory / name, label=f"checkpoint {name}")
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"checkpoint contains a non-regular entry: {path}")

    adapter = _json(directory / "adapter_config.json", label="PEFT adapter config")
    _expect(adapter.get("peft_type"), "LORA", label="adapter type")
    _expect(adapter.get("r"), 8, label="adapter rank")
    _expect(adapter.get("lora_alpha"), 16, label="adapter alpha")
    _expect(float(adapter.get("lora_dropout", -1)), 0.0, label="adapter dropout")
    _expect(adapter.get("bias"), "none", label="adapter bias")
    targets = adapter.get("target_modules")
    if not isinstance(targets, list) or not targets or any(not isinstance(value, str) for value in targets):
        raise ContractError("adapter target_modules must be a non-empty string list")
    if any("vision" in value.lower() or "lm_head" in value.lower() for value in targets):
        raise ContractError("adapter unexpectedly targets vision or unembedding modules")
    if not any("mlp" in value.lower() for value in targets) or not any(
        "attn" in value.lower() or "attention" in value.lower() for value in targets
    ):
        raise ContractError("adapter does not contain both MLP and attention targets")

    manifest = _json(directory / "manifest.json", label="checkpoint manifest")
    _expect(manifest.get("backend"), "local", label="checkpoint backend")
    _expect(manifest.get("kind"), "both", label="checkpoint kind")
    _expect(manifest.get("lora"), True, label="checkpoint LoRA mode")
    _expect(manifest.get("keep_frozen_base"), False, label="checkpoint frozen-base copy")
    names = manifest.get("trainable_parameter_names")
    if not isinstance(names, list) or not names or any(not isinstance(value, str) for value in names):
        raise ContractError("checkpoint has no trainable parameter names")
    if any("vision" in value.lower() or "lm_head" in value.lower() for value in names):
        raise ContractError("checkpoint trainable parameters include vision or unembedding tensors")

    loop = manifest.get("loop_state")
    if not isinstance(loop, Mapping):
        raise ContractError("checkpoint has no loop state")
    expected_end = (index + 1) * plan.UPDATES_PER_SEGMENT
    _expect(loop.get("schema"), "ctm.rl_loop_state.v1", label="checkpoint loop schema")
    _expect(loop.get("global_step"), expected_end, label="checkpoint global step")
    _expect(loop.get("optimizer_step"), expected_end, label="checkpoint optimizer step")
    _expect(loop.get("segment_start_global_step"), index * plan.UPDATES_PER_SEGMENT, label="segment start")
    _expect(loop.get("segment_step"), plan.UPDATES_PER_SEGMENT, label="segment update count")
    _expect(loop.get("completed_epochs"), index + 1, label="completed segment epochs")
    _expect(loop.get("accumulated_grads"), 0, label="checkpoint accumulated gradients")
    _expect(loop.get("final"), True, label="checkpoint final boundary")
    runtime_rng = loop.get("runtime_rng")
    if not isinstance(runtime_rng, Mapping) or not all(
        key in runtime_rng
        for key in (
            "python_random_state",
            "torch_cpu_rng_state_base64",
            "torch_cuda_rng_state_base64",
            "torch_cuda_coordinator_device",
        )
    ):
        raise ContractError("checkpoint does not contain strict coordinator RNG state")
    artifacts = [_identity(path, label=f"checkpoint artifact {path.name}") for path in sorted(directory.iterdir())]
    return artifacts, dict(loop)


def write_launch(root_value: str | Path, segment_index: int, *, model_snapshot: str | Path, argv: Sequence[str]) -> dict[str, Any]:
    root = _root(root_value)
    index = _index(segment_index)
    validate_preflight(root)
    snapshot = _regular_directory(Path(model_snapshot).resolve(), label="pinned Muse snapshot")
    _expect(snapshot.name, MODEL_REVISION, label="launch model revision directory")
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ContractError("launch argv must be a non-empty string list")
    forbidden = ("--max-new-tokens", "--max-tokens", "--max-output-tokens", "--max-completion-tokens")
    if any(value in forbidden for value in argv) or "--no-max-new-tokens" not in argv:
        raise ContractError("Muse training launch must be uncapped and select --no-max-new-tokens")
    record = plan.segment_record(root, index)
    document = {
        "schema": LAUNCH_SCHEMA,
        "segment": record,
        "model_snapshot": str(snapshot),
        "argv": list(argv),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "preflight": _identity(preflight_path(root), label="Muse parity preflight"),
        "generation": {"max_tokens": None, "termination": "eos_only"},
    }
    path = launch_path(root, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(document)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ContractError(f"refusing to overwrite a different segment launch: {path}")
    return document


def validate_receipt(root_value: str | Path, segment_index: int, *, _seen: set[int] | None = None) -> dict[str, Any]:
    root = _root(root_value)
    index = _index(segment_index)
    seen = set() if _seen is None else _seen
    if index in seen:
        raise ContractError("segment receipt parent cycle")
    seen.add(index)
    document = _json(receipt_path(root, index), label="Muse segment receipt")
    _expect(document.get("schema"), RECEIPT_SCHEMA, label="segment receipt schema")
    _expect(document.get("segment"), plan.segment_record(root, index), label="segment receipt record")
    _same_identity(_identity(preflight_path(root), label="Muse parity preflight"), document.get("preflight"), label="preflight")
    validate_preflight(root)
    _same_identity(_identity(launch_path(root, index), label="Muse segment launch"), document.get("launch"), label="launch")
    artifacts, loop = _checkpoint_artifacts(root, index)
    _expect(document.get("checkpoint_artifacts"), artifacts, label="checkpoint artifacts")
    _expect(document.get("loop_state"), loop, label="checkpoint loop state")
    parent = document.get("parent")
    if index == 0:
        _expect(parent, None, label="base segment parent")
    else:
        previous = validate_receipt(root, index - 1, _seen=seen)
        expected = {
            "receipt": _identity(receipt_path(root, index - 1), label="parent segment receipt"),
            "checkpoint": str(checkpoint_path(root, index - 1)),
        }
        _expect(parent, expected, label="segment parent")
        _expect(previous.get("segment"), plan.segment_record(root, index - 1), label="parent segment record")
    return document


def guard(root_value: str | Path, segment_index: int) -> dict[str, Any]:
    root = _root(root_value)
    index = _index(segment_index)
    validate_preflight(root)
    receipt = receipt_path(root, index)
    if receipt.exists() or receipt.is_symlink():
        validate_receipt(root, index)
        return {"action": "completed", "segment": index, "receipt": str(receipt)}
    residue = [path for path in (control_root(root, index), run_root(root, index)) if path.exists() or path.is_symlink()]
    if residue:
        raise ContractError(
            "refusing a Muse segment namespace with residue but no valid receipt: "
            + ", ".join(str(path) for path in residue)
        )
    if index:
        validate_receipt(root, index - 1)
        first_passing = None
        for prior_index in range(index):
            prior = validate_convergence(root, prior_index)
            if first_passing is None and prior.get("passed") is True:
                first_passing = prior_index
        completed_step = index * plan.UPDATES_PER_SEGMENT
        if first_passing is not None and completed_step >= plan.MINIMUM_COMPARISON_OPTIMIZER_STEP:
            return {
                "action": "converged",
                "segment": index,
                "final_segment": first_passing,
                "comparison_terminal_segment": index - 1,
                "comparison_terminal_optimizer_step": completed_step,
            }
    return {"action": "proceed", "segment": index}


def seal(root_value: str | Path, segment_index: int) -> dict[str, Any]:
    root = _root(root_value)
    index = _index(segment_index)
    path = receipt_path(root, index)
    if path.exists() or path.is_symlink():
        validate_receipt(root, index)
        return {"status": "resumed", "segment": index, "receipt": str(path)}
    validate_preflight(root)
    launch = _identity(launch_path(root, index), label="Muse segment launch")
    artifacts, loop = _checkpoint_artifacts(root, index)
    parent = None
    if index:
        validate_receipt(root, index - 1)
        parent = {
            "receipt": _identity(receipt_path(root, index - 1), label="parent segment receipt"),
            "checkpoint": str(checkpoint_path(root, index - 1)),
        }
    document = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "segment": plan.segment_record(root, index),
        "preflight": _identity(preflight_path(root), label="Muse parity preflight"),
        "launch": launch,
        "parent": parent,
        "checkpoint": str(checkpoint_path(root, index)),
        "checkpoint_artifacts": artifacts,
        "loop_state": loop,
        "continuation": {
            "mode": "pinned_base_snapshot" if index == 0 else "optimizer_data_segment",
            "coordinator_rng_restored": bool(index),
            "vllm_worker_rng_restored": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_canonical(document))
    validate_receipt(root, index)
    return {"status": "written", "segment": index, "receipt": str(path)}


def _window_metrics(root: Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    path = _regular_file(run_root(root, index) / "metrics.jsonl", label="Muse trainer metrics")
    start = index * plan.UPDATES_PER_SEGMENT + 1
    end = (index + 1) * plan.UPDATES_PER_SEGMENT
    selected: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            raise ContractError(f"trainer metrics contain a blank line at {line_number}")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(f"trainer metrics line {line_number} is invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise ContractError(f"trainer metrics line {line_number} is not an object")
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or not start <= step <= end:
            continue
        core_keys = [
            f"train/consistency_gap_abs_{kind}_{perturbation}"
            for perturbation in EXPECTED_TRAINING_PERTURBATIONS
            for kind in ("sum", "count")
        ]
        present = [key for key in core_keys if key in record]
        if not present:
            continue
        if len(present) != len(core_keys) or step in selected:
            raise ContractError(f"trainer metrics have partial or duplicate consistency aggregates at step {step}")
        selected[step] = dict(record)
    expected_steps = list(range(start, end + 1))
    if sorted(selected) != expected_steps:
        raise ContractError(f"trainer metrics do not contain exactly the segment steps: {expected_steps}")

    absolute_sum = 0.0
    count = 0
    per_perturbation: dict[str, dict[str, Any]] = {}
    for perturbation in EXPECTED_TRAINING_PERTURBATIONS:
        pert_sum = 0.0
        pert_count = 0
        for step in expected_steps:
            record = selected[step]
            value_sum = record[f"train/consistency_gap_abs_sum_{perturbation}"]
            value_count = record[f"train/consistency_gap_abs_count_{perturbation}"]
            mean_key = f"train/consistency_gap_abs_mean_{perturbation}"
            value_mean = record.get(mean_key)
            if (
                isinstance(value_sum, bool)
                or not isinstance(value_sum, (int, float))
                or not math.isfinite(float(value_sum))
                or isinstance(value_count, bool)
                or not isinstance(value_count, int)
                or not 0 <= value_count <= EXPECTED_QUESTIONS_PER_UPDATE
            ):
                raise ContractError(f"invalid convergence aggregate at step {step}, perturbation {perturbation}")
            if value_count == 0:
                if not math.isclose(float(value_sum), 0.0, rel_tol=0.0, abs_tol=1e-12) or value_mean is not None:
                    raise ContractError(
                        f"zero-count convergence aggregate must have zero sum and no mean at step {step}, "
                        f"perturbation {perturbation}"
                    )
            elif (
                isinstance(value_mean, bool)
                or not isinstance(value_mean, (int, float))
                or not math.isfinite(float(value_mean))
                or not math.isclose(float(value_mean), float(value_sum) / value_count, rel_tol=1e-9, abs_tol=1e-9)
            ):
                raise ContractError(f"inconsistent convergence sum/count/mean at step {step}, perturbation {perturbation}")
            pert_sum += float(value_sum)
            pert_count += value_count
        per_perturbation[str(perturbation)] = {"absolute_sum": pert_sum, "count": pert_count}
        absolute_sum += pert_sum
        count += pert_count
    if count <= 0:
        raise ContractError("Muse convergence window contains no resolved observations")
    return {
        "metrics": _identity(path, label="Muse trainer metrics"),
        "optimizer_step": end,
        "expected_observations": EXPECTED_OBSERVATIONS_PER_WINDOW,
        "resolved_observations": count,
        "coverage": count / EXPECTED_OBSERVATIONS_PER_WINDOW,
        "absolute_sum": absolute_sum,
        "weighted_abs_gap": absolute_sum / count,
        "per_perturbation": per_perturbation,
    }


def validate_convergence(root_value: str | Path, segment_index: int) -> dict[str, Any]:
    root = _root(root_value)
    index = _index(segment_index)
    document = _json(convergence_path(root, index), label="Muse convergence receipt")
    _expect(document.get("schema"), CONVERGENCE_SCHEMA, label="convergence schema")
    _expect(document.get("segment_index"), index, label="convergence segment")
    _same_identity(_identity(receipt_path(root, index), label="Muse segment receipt"), document.get("segment_receipt"), label="segment receipt")
    current = _window_metrics(root, index)
    _expect(document.get("window"), current, label="convergence window")
    return document


def evaluate_convergence(root_value: str | Path, segment_index: int) -> dict[str, Any]:
    root = _root(root_value)
    index = _index(segment_index)
    validate_receipt(root, index)
    path = convergence_path(root, index)
    if path.exists() or path.is_symlink():
        return validate_convergence(root, index)
    window = _window_metrics(root, index)
    previous_gap = None
    absolute_change = None
    if index:
        previous = validate_convergence(root, index - 1)
        previous_window = previous.get("window")
        if not isinstance(previous_window, Mapping):
            raise ContractError("prior convergence receipt has no window")
        previous_gap = float(previous_window["weighted_abs_gap"])
        absolute_change = abs(float(window["weighted_abs_gap"]) - previous_gap)
    eligible = int(window["optimizer_step"]) >= FIRST_ELIGIBLE_OPTIMIZER_STEP and absolute_change is not None
    passed = bool(
        eligible
        and float(window["coverage"]) >= MINIMUM_COVERAGE
        and float(window["weighted_abs_gap"]) <= MAXIMUM_WEIGHTED_ABS_GAP
        and float(absolute_change) <= MAXIMUM_ABS_CHANGE
    )
    document = {
        "schema": CONVERGENCE_SCHEMA,
        "segment_index": index,
        "segment_receipt": _identity(receipt_path(root, index), label="Muse segment receipt"),
        "window": window,
        "previous_weighted_abs_gap": previous_gap,
        "absolute_change": absolute_change,
        "thresholds": {
            "first_eligible_optimizer_step": FIRST_ELIGIBLE_OPTIMIZER_STEP,
            "minimum_coverage": MINIMUM_COVERAGE,
            "maximum_weighted_abs_gap": MAXIMUM_WEIGHTED_ABS_GAP,
            "maximum_abs_change": MAXIMUM_ABS_CHANGE,
        },
        "eligible": eligible,
        "passed": passed,
        "hard_cap_reached": index == plan.TOTAL_SEGMENTS - 1,
    }
    with path.open("xb") as handle:
        handle.write(_canonical(document))
    return validate_convergence(root, index)


__all__ = [
    "ContractError",
    "checkpoint_path",
    "control_root",
    "convergence_path",
    "evaluate_convergence",
    "guard",
    "launch_path",
    "receipt_path",
    "seal",
    "validate_convergence",
    "validate_preflight",
    "validate_receipt",
    "write_launch",
]
