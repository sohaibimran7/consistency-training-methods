#!/usr/bin/env python3
"""Audited validation amendment for Muse RMCT segment boundaries.

The original training source is immutable and remains the only code which
performs model generation or optimization.  This module corrects two
post-training custody assumptions without changing that source:

* PEFT stores suffix-only ``target_modules`` (for example ``q_proj``), so
  attention/MLP coverage must be proven from the complete trainable tensor
  names rather than by searching those suffixes for ``attn``/``mlp``.
* the RL loop deliberately distinguishes attempted/global batches from
  realized optimizer updates.  An empty consistency-reward batch is recorded
  as ``skipped_empty_batch=1`` and advances the global/data axis without
  advancing Adam.  Boundary validation must reproduce that counter exactly.

The amendment monkey-patches only the private checkpoint/window validators in
the frozen segment contract.  It is self-attested in a write-once receipt and
does not alter model loading, data selection, sampling, loss, optimizer, RNG,
or the no-output-token-cap contract.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from safetensors import safe_open

from experiments.muse_glimmer_rmct_replication import plan
from infra.isambard import muse_glimmer_rmct_segment_contract as frozen


AMENDMENT_SCHEMA = "muse-glimmer-rmct-boundary-validator-amendment-v1"
AMENDMENT_DIR = Path("artifacts/muse-glimmer-rmct-boundary-amendment-20260825")
AMENDMENT_NAME = "attestation.json"
EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
EXPECTED_FAMILY_PROJECTIONS = {
    "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
}
_TRAINABLE_RE = re.compile(
    r"^base_model\.model\.model\.language_model\.layers\.(?P<layer>[0-9]+)\."
    r"(?P<family>self_attn|mlp)\.(?P<projection>[a-z_]+)\."
    r"lora_(?P<kind>A|B)\.default\.weight$"
)
_ORIGINAL_CHECKPOINT_ARTIFACTS = frozen._checkpoint_artifacts
_ORIGINAL_WINDOW_METRICS = frozen._window_metrics
_INSTALLED = False


def amendment_path(root: str | Path) -> Path:
    return Path(root).resolve() / AMENDMENT_DIR / AMENDMENT_NAME


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _checkpoint_manifest(root: Path, segment_index: int) -> dict[str, Any]:
    path = frozen.checkpoint_path(root, segment_index) / "manifest.json"
    return frozen._json(path, label=f"Muse checkpoint {segment_index} manifest")


def _metric_progress(root: Path, segment_index: int) -> dict[str, Any]:
    index = frozen._index(segment_index)
    path = frozen._regular_file(
        frozen.run_root(root, index) / "metrics.jsonl",
        label="Muse trainer metrics",
    )
    start = index * plan.UPDATES_PER_SEGMENT + 1
    end = (index + 1) * plan.UPDATES_PER_SEGMENT
    selected: dict[int, dict[str, Any]] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            raise frozen.ContractError(f"trainer metrics contain a blank line at {line_number}")
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise frozen.ContractError(f"trainer metrics line {line_number} is invalid JSON") from exc
        if not isinstance(record, Mapping):
            raise frozen.ContractError(f"trainer metrics line {line_number} is not an object")
        step = record.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or not start <= step <= end:
            continue
        if "train/skipped_empty_batch" not in record:
            # The final-checkpoint logger writes a second boundary row without
            # training metrics.  It is provenance, not a second training step.
            continue
        if step in selected:
            raise frozen.ContractError(f"trainer metrics duplicate training step {step}")
        selected[step] = dict(record)
    expected_steps = list(range(start, end + 1))
    if sorted(selected) != expected_steps:
        raise frozen.ContractError(f"trainer metrics do not contain exactly the segment steps: {expected_steps}")

    if index == 0:
        prior_optimizer_step = 0
    else:
        previous_loop = _checkpoint_manifest(root, index - 1).get("loop_state")
        if not isinstance(previous_loop, Mapping):
            raise frozen.ContractError("parent checkpoint has no loop state")
        frozen._expect(
            previous_loop.get("global_step"),
            start - 1,
            label="parent checkpoint global step",
        )
        prior_optimizer_step = previous_loop.get("optimizer_step")
        if (
            isinstance(prior_optimizer_step, bool)
            or not isinstance(prior_optimizer_step, int)
            or prior_optimizer_step < 0
        ):
            raise frozen.ContractError("parent checkpoint has an invalid optimizer step")

    realized = prior_optimizer_step
    skipped_steps: list[int] = []
    for step in expected_steps:
        record = selected[step]
        skipped = record.get("train/skipped_empty_batch")
        if isinstance(skipped, bool) or skipped not in (0, 1):
            raise frozen.ContractError(f"training step {step} has invalid skipped_empty_batch={skipped!r}")
        if skipped:
            skipped_steps.append(step)
        else:
            realized += 1
        frozen._expect(
            record.get("train/optimizer_step"),
            realized,
            label=f"training step {step} realized optimizer counter",
        )
        failure_count = record.get("rollout/grader_failure_count")
        if isinstance(failure_count, bool) or not isinstance(failure_count, int) or failure_count < 0:
            raise frozen.ContractError(f"training step {step} has an invalid grader failure count")

    return {
        "metrics": frozen._identity(path, label="Muse trainer metrics"),
        "global_step_start": start,
        "global_step_end": end,
        "prior_optimizer_step": prior_optimizer_step,
        "optimizer_step": realized,
        "realized_optimizer_updates": realized - prior_optimizer_step,
        "skipped_empty_batches": len(skipped_steps),
        "skipped_global_steps": skipped_steps,
    }


def _normalized_safetensor_name(name: str) -> str:
    for kind in ("A", "B"):
        suffix = f".lora_{kind}.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".lora_{kind}.default.weight"
    return name


def _validate_lora_coverage(directory: Path, adapter: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    targets = adapter.get("target_modules")
    if (
        not isinstance(targets, list)
        or not targets
        or any(not isinstance(value, str) or not value for value in targets)
        or len(targets) != len(set(targets))
    ):
        raise frozen.ContractError("adapter target_modules must be a unique non-empty string list")
    if set(targets) != EXPECTED_TARGET_MODULES:
        raise frozen.ContractError(
            f"Muse adapter target suffixes differ: got={sorted(targets)!r}, "
            f"expected={sorted(EXPECTED_TARGET_MODULES)!r}"
        )

    names = manifest.get("trainable_parameter_names")
    if not isinstance(names, list) or not names or any(not isinstance(value, str) for value in names):
        raise frozen.ContractError("checkpoint has no trainable parameter names")
    if len(names) != len(set(names)):
        raise frozen.ContractError("checkpoint trainable parameter names are not unique")
    parsed: set[tuple[int, str, str, str]] = set()
    for name in names:
        match = _TRAINABLE_RE.fullmatch(name)
        if match is None:
            raise frozen.ContractError(f"unexpected Muse trainable parameter: {name}")
        parsed.add(
            (
                int(match.group("layer")),
                match.group("family"),
                match.group("projection"),
                match.group("kind"),
            )
        )
    layers = sorted({item[0] for item in parsed})
    if not layers or layers != list(range(layers[-1] + 1)):
        raise frozen.ContractError("Muse adapter layers are absent or non-contiguous")
    expected = {
        (layer, family, projection, kind)
        for layer in layers
        for family, projections in EXPECTED_FAMILY_PROJECTIONS.items()
        for projection in projections
        for kind in ("A", "B")
    }
    if parsed != expected:
        missing = sorted(expected - parsed)[:10]
        extra = sorted(parsed - expected)[:10]
        raise frozen.ContractError(f"Muse LoRA coverage differs: missing={missing!r}, extra={extra!r}")

    weights = directory / "adapter_model.safetensors"
    try:
        with safe_open(weights, framework="pt", device="cpu") as handle:
            tensor_names = {_normalized_safetensor_name(name) for name in handle.keys()}
    except Exception as exc:  # pragma: no cover - safetensors owns the concrete errors
        raise frozen.ContractError("Muse adapter safetensors cannot be indexed") from exc
    if tensor_names != set(names):
        raise frozen.ContractError("Muse adapter safetensor keys differ from the trainable manifest")


def _checkpoint_artifacts(root: Path, segment_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index = frozen._index(segment_index)
    directory = frozen._regular_directory(frozen.checkpoint_path(root, index), label="final Muse checkpoint")
    required = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "manifest.json")
    for name in required:
        frozen._regular_file(directory / name, label=f"checkpoint {name}")
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise frozen.ContractError(f"checkpoint contains a non-regular entry: {path}")

    adapter = frozen._json(directory / "adapter_config.json", label="PEFT adapter config")
    frozen._expect(adapter.get("peft_type"), "LORA", label="adapter type")
    frozen._expect(adapter.get("r"), 8, label="adapter rank")
    frozen._expect(adapter.get("lora_alpha"), 16, label="adapter alpha")
    frozen._expect(float(adapter.get("lora_dropout", -1)), 0.0, label="adapter dropout")
    frozen._expect(adapter.get("bias"), "none", label="adapter bias")

    manifest = frozen._json(directory / "manifest.json", label="checkpoint manifest")
    frozen._expect(manifest.get("backend"), "local", label="checkpoint backend")
    frozen._expect(manifest.get("kind"), "both", label="checkpoint kind")
    frozen._expect(manifest.get("lora"), True, label="checkpoint LoRA mode")
    frozen._expect(manifest.get("keep_frozen_base"), False, label="checkpoint frozen-base copy")
    _validate_lora_coverage(directory, adapter, manifest)

    loop = manifest.get("loop_state")
    if not isinstance(loop, Mapping):
        raise frozen.ContractError("checkpoint has no loop state")
    expected_end = (index + 1) * plan.UPDATES_PER_SEGMENT
    progress = _metric_progress(root, index)
    frozen._expect(loop.get("schema"), "ctm.rl_loop_state.v1", label="checkpoint loop schema")
    frozen._expect(loop.get("global_step"), expected_end, label="checkpoint global step")
    frozen._expect(loop.get("optimizer_step"), progress["optimizer_step"], label="checkpoint optimizer step")
    frozen._expect(
        loop.get("segment_start_global_step"),
        index * plan.UPDATES_PER_SEGMENT,
        label="segment start",
    )
    frozen._expect(loop.get("segment_step"), plan.UPDATES_PER_SEGMENT, label="segment batch count")
    frozen._expect(loop.get("completed_epochs"), index + 1, label="completed segment epochs")
    frozen._expect(loop.get("accumulated_grads"), 0, label="checkpoint accumulated gradients")
    frozen._expect(loop.get("final"), True, label="checkpoint final boundary")
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
        raise frozen.ContractError("checkpoint does not contain strict coordinator RNG state")
    artifacts = [frozen._identity(path, label=f"checkpoint artifact {path.name}") for path in sorted(directory.iterdir())]
    return artifacts, dict(loop)


def _window_metrics(root: Path, segment_index: int) -> dict[str, Any]:
    window = dict(_ORIGINAL_WINDOW_METRICS(root, segment_index))
    progress = _metric_progress(root, segment_index)
    labelled_global_step = window.pop("optimizer_step")
    frozen._expect(labelled_global_step, progress["global_step_end"], label="convergence global step")
    return {
        **window,
        "global_step": progress["global_step_end"],
        "optimizer_step": progress["optimizer_step"],
        "prior_optimizer_step": progress["prior_optimizer_step"],
        "realized_optimizer_updates": progress["realized_optimizer_updates"],
        "skipped_empty_batches": progress["skipped_empty_batches"],
        "skipped_global_steps": progress["skipped_global_steps"],
    }


def install() -> None:
    """Install the validation-only amendment in this interpreter."""

    global _INSTALLED
    if _INSTALLED:
        return
    if frozen._checkpoint_artifacts is not _ORIGINAL_CHECKPOINT_ARTIFACTS:
        raise frozen.ContractError("a different Muse checkpoint validator is already installed")
    if frozen._window_metrics is not _ORIGINAL_WINDOW_METRICS:
        raise frozen.ContractError("a different Muse convergence validator is already installed")
    frozen._checkpoint_artifacts = _checkpoint_artifacts
    frozen._window_metrics = _window_metrics
    _INSTALLED = True


def _semantics() -> dict[str, Any]:
    return {
        "checkpoint_label_axis": "global_training_batch_and_frozen_data_window",
        "global_step": "advances_once_per_batch_including_explicit_empty-reward_skips",
        "optimizer_step": "advances_only_after_a_realized_nonempty_gradient_update",
        "skip_validation": "replayed_exactly_from_train/skipped_empty_batch_and_train/optimizer_step",
        "comparison_checkpoints": [16, 64, "final"],
        "comparison_axis": "global_step",
        "convergence_eligibility_axis": "realized_optimizer_step",
        "output_token_cap": None,
        "generation_termination": "model_eos_only",
    }


def write_attestation(
    root_value: str | Path,
    *,
    job_id: str,
    slurm_state: str,
    slurm_exit_code: str,
    slurm_output: str | Path,
) -> dict[str, Any]:
    root = Path(root_value).resolve()
    frozen.validate_preflight(root)
    if not job_id.isdigit() or slurm_state != "FAILED" or slurm_exit_code != "1:0":
        raise frozen.ContractError("boundary amendment requires the exact failed Slurm incident identity")
    output = frozen._regular_file(Path(slurm_output).resolve(), label="failed segment Slurm output")
    text = output.read_text(encoding="utf-8", errors="replace")
    required_markers = (
        "Training complete. Final checkpoint:",
        "adapter does not contain both MLP and attention targets",
        "infra.isambard.muse_glimmer_rmct_segment_contract.ContractError",
    )
    if any(marker not in text for marker in required_markers):
        raise frozen.ContractError("failed Slurm output does not prove the post-training validator incident")
    document = {
        "schema": AMENDMENT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "post-training-boundary-validation-and-convergence-provenance-only",
        "original_training_source_unchanged": True,
        "original_contract": frozen._identity(Path(frozen.__file__).resolve(), label="frozen segment contract"),
        "amendment_code": frozen._identity(Path(__file__).resolve(), label="boundary validator amendment"),
        "preflight": frozen._identity(frozen.preflight_path(root), label="Muse parity preflight"),
        "incident": {
            "slurm_job_id": job_id,
            "state": slurm_state,
            "exit_code": slurm_exit_code,
            "output": frozen._identity(output, label="failed segment Slurm output"),
            "training_completed_before_validator_failure": True,
            "original_validator_failure": "suffix-only-PEFT-target-module-family-inference",
        },
        "semantics": _semantics(),
        "scientific_surface": {
            "model_changed": False,
            "data_changed": False,
            "sampling_changed": False,
            "loss_changed": False,
            "optimizer_changed": False,
            "rng_state_changed": False,
            "generation_or_output_token_cap_added": False,
        },
    }
    path = amendment_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(document)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise frozen.ContractError(f"refusing to overwrite a different amendment attestation: {path}")
    return validate_attestation(root)


def validate_attestation(root_value: str | Path) -> dict[str, Any]:
    root = Path(root_value).resolve()
    document = frozen._json(amendment_path(root), label="Muse boundary amendment attestation")
    frozen._expect(document.get("schema"), AMENDMENT_SCHEMA, label="boundary amendment schema")
    frozen._expect(document.get("scope"), "post-training-boundary-validation-and-convergence-provenance-only", label="boundary amendment scope")
    frozen._expect(document.get("original_training_source_unchanged"), True, label="training-source preservation")
    frozen._expect(document.get("semantics"), _semantics(), label="boundary amendment semantics")
    frozen._same_identity(
        frozen._identity(Path(frozen.__file__).resolve(), label="frozen segment contract"),
        document.get("original_contract"),
        label="frozen segment contract",
    )
    frozen._same_identity(
        frozen._identity(Path(__file__).resolve(), label="boundary validator amendment"),
        document.get("amendment_code"),
        label="boundary validator amendment",
    )
    frozen._same_identity(
        frozen._identity(frozen.preflight_path(root), label="Muse parity preflight"),
        document.get("preflight"),
        label="Muse parity preflight",
    )
    scientific = document.get("scientific_surface")
    if not isinstance(scientific, Mapping) or scientific != {
        "model_changed": False,
        "data_changed": False,
        "sampling_changed": False,
        "loss_changed": False,
        "optimizer_changed": False,
        "rng_state_changed": False,
        "generation_or_output_token_cap_added": False,
    }:
        raise frozen.ContractError("boundary amendment scientific surface differs")
    return document


def recover_seal(
    root_value: str | Path,
    segment_index: int,
    *,
    job_id: str,
    slurm_state: str,
    slurm_exit_code: str,
    slurm_output: str | Path,
) -> dict[str, Any]:
    root = Path(root_value).resolve()
    write_attestation(
        root,
        job_id=job_id,
        slurm_state=slurm_state,
        slurm_exit_code=slurm_exit_code,
        slurm_output=slurm_output,
    )
    install()
    sealed = frozen.seal(root, segment_index)
    convergence = frozen.evaluate_convergence(root, segment_index)
    return {"sealed": sealed, "convergence": convergence, "progress": _metric_progress(root, segment_index)}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recover = subparsers.add_parser("recover-seal")
    recover.add_argument("--repo-root", type=Path, required=True)
    recover.add_argument("--segment-index", type=int, required=True)
    recover.add_argument("--job-id", required=True)
    recover.add_argument("--slurm-state", required=True)
    recover.add_argument("--slurm-exit-code", required=True)
    recover.add_argument("--slurm-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--segment-index", type=int)
    return parser.parse_args()


def main() -> int:
    args = _args()
    root = args.repo_root.resolve()
    if args.command == "recover-seal":
        result = recover_seal(
            root,
            args.segment_index,
            job_id=args.job_id,
            slurm_state=args.slurm_state,
            slurm_exit_code=args.slurm_exit_code,
            slurm_output=args.slurm_output,
        )
    else:
        validate_attestation(root)
        install()
        result = {"attestation": str(amendment_path(root))}
        if args.segment_index is not None:
            result["receipt"] = frozen.validate_receipt(root, args.segment_index)
            result["convergence"] = frozen.validate_convergence(root, args.segment_index)
            result["progress"] = _metric_progress(root, args.segment_index)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AMENDMENT_SCHEMA",
    "amendment_path",
    "install",
    "recover_seal",
    "validate_attestation",
    "write_attestation",
]
