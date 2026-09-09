"""Fail-closed launch evidence for the fresh repaired-ACT evaluation.

This is deliberately narrower than the generic Stage-1 raw-log preflight.  A
fresh repaired-ACT adapter is useful evidence only when all three links below
refer to the same immutable raw PEFT weights:

1. the target-scoped output state emitted by the fresh 4,000-step training
   command;
2. a *passed* tiny native Transformers/PEFT behavioural gate; and
3. for a vLLM evaluation, the translated compatibility adapter plus its live
   HF/vLLM parity evidence.

The historical Qwen3.5 incident broke the third link silently: vLLM accepted
the raw adapter directory but evaluated the base model.  A path alone is not
enough evidence, so this module writes a small deterministic, write-once chain
attestation before a long evaluation may be launched.  The no-CoT raw-log
preflight can revalidate the same chain after generation, before grading.

The module is CPU-only.  It neither trains nor loads a model.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from ctm.evals.qwen35_vllm_attestation import (
    COMPATIBILITY_MANIFEST_SCHEMA,
    file_sha256,
    is_verified_qwen35_vllm_compat_adapter,
)
from experiments.act_repair_gate.behavioral_gate import (
    SCHEMA as TINY_GATE_SCHEMA,
    build_attestation as build_tiny_gate_attestation,
)


SCHEMA = "qwen35-repaired-act-evaluation-chain-attestation-v1"
MODEL = "Qwen/Qwen3.5-9B"
FRESH_EXPERIMENT = "rmct_paper_vast_dense_qwen3_5_9b_stage1_supervised_recovery_none_calibration_20260803"
FRESH_TARGET = "repaired-act"
FRESH_TRAINING_COMMAND = "repaired_act_none"
RUNTIME_PROFILES = ("hf-peft", "vllm")
_SOURCE_PREFIX = "base_model.model.model.layers."
_DESTINATION_PREFIX = "base_model.model.model.language_model.layers."


def _local_path(value: str | Path) -> Path:
    """Resolve a local path or LocalBackend ``file://`` URI exactly once."""

    raw = str(value)
    if raw.startswith("file://"):
        parsed = urlparse(raw)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"file URI must name this host, got {raw!r}")
        raw = unquote(parsed.path)
    return Path(raw).expanduser().resolve()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _same_path(value: Any, expected: Path, *, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} has no path")
    if _local_path(value) != expected:
        raise ValueError(f"{label} does not match the supplied path")


def _checkpoint_identity(checkpoint: str | Path) -> dict[str, str]:
    """Validate the raw LocalBackend adapter emitted by fresh ACT training."""

    root = _local_path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f"fresh repaired-ACT checkpoint directory is missing: {root}")
    paths = {
        "adapter_model_sha256": root / "adapter_model.safetensors",
        "adapter_config_sha256": root / "adapter_config.json",
        "checkpoint_manifest_sha256": root / "manifest.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"fresh repaired-ACT checkpoint artifact is missing: {path}")
    manifest = _read_json_object(paths["checkpoint_manifest_sha256"], label="fresh repaired-ACT checkpoint manifest")
    if manifest.get("backend") != "local" or manifest.get("model") != MODEL or manifest.get("lora") is not True:
        raise ValueError("fresh repaired-ACT checkpoint manifest is not a local Qwen3.5 LoRA checkpoint")
    return {
        "path": str(root),
        **{name: file_sha256(path) for name, path in paths.items()},
    }


def _training_output_identity(training_output_state: str | Path, *, checkpoint: Path) -> dict[str, str]:
    """Bind the raw adapter to the one target-scoped fresh ACT publication."""

    state_path = _local_path(training_output_state)
    if not state_path.is_file():
        raise FileNotFoundError(f"fresh repaired-ACT training output state is missing: {state_path}")
    state = _read_json_object(state_path, label="fresh repaired-ACT training output state")
    if state.get("schema_version") != 1:
        raise ValueError("fresh repaired-ACT training output state has an unsupported schema_version")
    if state.get("experiment") != FRESH_EXPERIMENT:
        raise ValueError("training output state is not from the fresh repaired-ACT experiment")
    if state.get("execution_target") != FRESH_TARGET:
        raise ValueError("training output state is not the repaired-act target publication")
    checkpoints = _require_mapping(state.get("training_checkpoints"), label="training output state.training_checkpoints")
    if set(checkpoints) != {FRESH_TRAINING_COMMAND}:
        raise ValueError("training output state must publish exactly the fresh repaired-ACT command")
    published = checkpoints[FRESH_TRAINING_COMMAND]
    if not isinstance(published, str) or not published:
        raise ValueError("training output state has no repaired-ACT checkpoint URI")
    if _local_path(published) != checkpoint:
        raise ValueError("training output state checkpoint does not match the supplied fresh repaired-ACT checkpoint")
    return {
        "path": str(state_path),
        "sha256": file_sha256(state_path),
        "training_command": FRESH_TRAINING_COMMAND,
        "checkpoint": str(checkpoint),
    }


def _tiny_gate_identity(tiny_gate_attestation: str | Path, *, checkpoint: Path) -> dict[str, Any]:
    """Recompute and require the passed native-HF tiny behavioural gate."""

    attestation_path = _local_path(tiny_gate_attestation)
    if not attestation_path.is_file():
        raise FileNotFoundError(f"tiny native-HF behavioural attestation is missing: {attestation_path}")
    attestation = _read_json_object(attestation_path, label="tiny native-HF behavioural attestation")
    if attestation.get("schema") != TINY_GATE_SCHEMA:
        raise ValueError("tiny native-HF behavioural attestation has an unsupported schema")
    if attestation.get("model") != MODEL:
        raise ValueError("tiny native-HF behavioural attestation does not name Qwen/Qwen3.5-9B")
    adapter = _require_mapping(attestation.get("adapter"), label="tiny native-HF behavioural attestation.adapter")
    _same_path(adapter.get("path"), checkpoint, label="tiny native-HF behavioural attestation adapter")
    raw_adapter_sha = file_sha256(checkpoint / "adapter_model.safetensors")
    if adapter.get("adapter_model_sha256") != raw_adapter_sha:
        raise ValueError("tiny native-HF behavioural attestation adapter hash does not match fresh ACT checkpoint")
    if adapter.get("condition_name") != "repaired-act":
        raise ValueError("tiny native-HF behavioural attestation does not name the repaired-act condition")

    report = _require_mapping(attestation.get("report"), label="tiny native-HF behavioural attestation.report")
    report_path_value = report.get("path")
    if not isinstance(report_path_value, str) or not report_path_value:
        raise ValueError("tiny native-HF behavioural attestation has no report path")
    report_path = _local_path(report_path_value)
    if not report_path.is_file() or report.get("sha256") != file_sha256(report_path):
        raise ValueError("tiny native-HF behavioural attestation report hash does not match report bytes")

    sources = _require_mapping(attestation.get("sources"), label="tiny native-HF behavioural attestation.sources")
    train = _require_mapping(sources.get("train_eval"), label="tiny native-HF behavioural attestation.sources.train_eval")
    heldout = _require_mapping(
        sources.get("heldout_in_domain"), label="tiny native-HF behavioural attestation.sources.heldout_in_domain"
    )
    train_path = train.get("path")
    heldout_path = heldout.get("path")
    if not isinstance(train_path, str) or not isinstance(heldout_path, str):
        raise ValueError("tiny native-HF behavioural attestation has no frozen split paths")

    # Rebuilding the document catches a tampered report, stale adapter, changed
    # split, or an attester called before the report completed.  These are the
    # exact fixed parameters in the fresh tiny gate YAML.
    expected = build_tiny_gate_attestation(
        report_path=report_path,
        adapter=checkpoint,
        train_data=train_path,
        heldout_data=heldout_path,
        gate_split="train_eval",
        min_base_switches=1,
        expected_limit_per_dataset=4,
    )
    if attestation != expected:
        raise ValueError("tiny native-HF behavioural attestation does not exactly revalidate its report and inputs")
    gate = _require_mapping(attestation.get("gate"), label="tiny native-HF behavioural attestation.gate")
    if gate.get("passed") is not True:
        raise ValueError("tiny native-HF behavioural gate did not pass")
    return {
        "path": str(attestation_path),
        "sha256": file_sha256(attestation_path),
        "report_sha256": str(report["sha256"]),
        "adapter_model_sha256": raw_adapter_sha,
        "train_data_sha256": str(train.get("sha256")),
        "heldout_data_sha256": str(heldout.get("sha256")),
    }


def _vllm_identity(vllm_compat_adapter: str | Path, *, raw_checkpoint: Path) -> dict[str, str]:
    """Validate vLLM parity and bind its translated copy back to raw ACT."""

    compat = _local_path(vllm_compat_adapter)
    if not compat.is_dir():
        raise FileNotFoundError(f"repaired-ACT vLLM compatibility adapter directory is missing: {compat}")
    paths = {
        "adapter_model_sha256": compat / "adapter_model.safetensors",
        "adapter_config_sha256": compat / "adapter_config.json",
        "compatibility_manifest_sha256": compat / "compatibility-manifest.json",
        "parity_attestation_sha256": compat / "vllm-parity-attestation.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"repaired-ACT vLLM compatibility artifact is missing: {path}")
    if not is_verified_qwen35_vllm_compat_adapter(compat):
        raise ValueError("repaired-ACT vLLM compatibility adapter lacks a valid immutable HF/vLLM parity attestation")

    manifest = _read_json_object(paths["compatibility_manifest_sha256"], label="repaired-ACT vLLM compatibility manifest")
    if manifest.get("schema") != COMPATIBILITY_MANIFEST_SCHEMA:
        raise ValueError("repaired-ACT vLLM compatibility manifest has an unsupported schema")
    source = _require_mapping(manifest.get("source"), label="repaired-ACT vLLM compatibility manifest.source")
    destination = _require_mapping(
        manifest.get("destination"), label="repaired-ACT vLLM compatibility manifest.destination"
    )
    translation = _require_mapping(
        manifest.get("translation"), label="repaired-ACT vLLM compatibility manifest.translation"
    )
    _same_path(source.get("path"), raw_checkpoint, label="repaired-ACT vLLM compatibility manifest source")
    _same_path(destination.get("path"), compat, label="repaired-ACT vLLM compatibility manifest destination")
    raw_adapter_sha = file_sha256(raw_checkpoint / "adapter_model.safetensors")
    if source.get("adapter_model_sha256") != raw_adapter_sha:
        raise ValueError("repaired-ACT vLLM compatibility manifest source adapter hash does not match fresh ACT checkpoint")
    compat_adapter_sha = file_sha256(paths["adapter_model_sha256"])
    if destination.get("adapter_model_sha256") != compat_adapter_sha:
        raise ValueError("repaired-ACT vLLM compatibility manifest destination adapter hash does not match adapter bytes")
    if translation.get("source_prefix") != _SOURCE_PREFIX or translation.get("destination_prefix") != _DESTINATION_PREFIX:
        raise ValueError("repaired-ACT vLLM compatibility manifest has an unexpected key translation")
    tensor_count = translation.get("tensor_count")
    translated_count = translation.get("translated_tensor_count")
    if isinstance(tensor_count, bool) or not isinstance(tensor_count, int) or tensor_count < 1 or tensor_count != translated_count:
        raise ValueError("repaired-ACT vLLM compatibility manifest has invalid translated tensor counts")
    if not _is_sha256(translation.get("translated_tensor_names_sha256")):
        raise ValueError("repaired-ACT vLLM compatibility manifest has no translated tensor-name hash")
    return {
        "path": str(compat),
        **{name: file_sha256(path) for name, path in paths.items()},
        "source_adapter_model_sha256": raw_adapter_sha,
    }


def build_repaired_act_evaluation_chain_attestation(
    *,
    training_output_state: str | Path,
    checkpoint: str | Path,
    tiny_gate_attestation: str | Path,
    runtime_profile: str,
    vllm_compat_adapter: str | Path | None = None,
) -> dict[str, Any]:
    """Construct a fully revalidated pre-launch ACT evaluation chain."""

    if runtime_profile not in RUNTIME_PROFILES:
        raise ValueError(f"runtime_profile must be one of {RUNTIME_PROFILES}")
    raw_checkpoint = _local_path(checkpoint)
    checkpoint_identity = _checkpoint_identity(raw_checkpoint)
    training_identity = _training_output_identity(training_output_state, checkpoint=raw_checkpoint)
    tiny_identity = _tiny_gate_identity(tiny_gate_attestation, checkpoint=raw_checkpoint)
    runtime: dict[str, Any] = {"profile": runtime_profile}
    if runtime_profile == "vllm":
        if vllm_compat_adapter is None:
            raise ValueError("vLLM repaired-ACT evaluation requires --vllm-compat-adapter")
        runtime["vllm_compatibility_adapter"] = _vllm_identity(vllm_compat_adapter, raw_checkpoint=raw_checkpoint)
    elif vllm_compat_adapter is not None:
        raise ValueError("--vllm-compat-adapter applies only to runtime_profile='vllm'")
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "fresh_training": training_identity,
        "raw_checkpoint": checkpoint_identity,
        "tiny_native_hf_gate": tiny_identity,
        "runtime": runtime,
    }


def write_repaired_act_evaluation_chain_attestation(path: str | Path, attestation: Mapping[str, Any]) -> tuple[Path, str]:
    """Write once, or accept only the byte-identical result of a prior run."""

    destination = _local_path(path)
    payload = json.dumps(dict(attestation), indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite differing repaired-ACT evaluation chain attestation: {destination}")
        return destination, "resumed"
    with destination.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
    return destination, "written"


def validate_repaired_act_evaluation_chain_attestation(
    path: str | Path,
    *,
    expected_runtime_profile: str | None = None,
    expected_checkpoint: str | Path | None = None,
    expected_vllm_compat_adapter: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild a stored chain and optionally bind it to an evaluator path."""

    attestation_path = _local_path(path)
    if not attestation_path.is_file():
        raise FileNotFoundError(f"repaired-ACT evaluation chain attestation is missing: {attestation_path}")
    stored = _read_json_object(attestation_path, label="repaired-ACT evaluation chain attestation")
    if stored.get("schema") != SCHEMA:
        raise ValueError("repaired-ACT evaluation chain attestation has an unsupported schema")
    if stored.get("model") != MODEL:
        raise ValueError("repaired-ACT evaluation chain attestation does not name Qwen/Qwen3.5-9B")
    fresh_training = _require_mapping(stored.get("fresh_training"), label="repaired-ACT evaluation chain fresh_training")
    raw = _require_mapping(stored.get("raw_checkpoint"), label="repaired-ACT evaluation chain raw_checkpoint")
    tiny = _require_mapping(stored.get("tiny_native_hf_gate"), label="repaired-ACT evaluation chain tiny_native_hf_gate")
    runtime = _require_mapping(stored.get("runtime"), label="repaired-ACT evaluation chain runtime")
    raw_path = raw.get("path")
    training_state = fresh_training.get("path")
    tiny_path = tiny.get("path")
    runtime_profile = runtime.get("profile")
    if not all(isinstance(value, str) and value for value in (raw_path, training_state, tiny_path, runtime_profile)):
        raise ValueError("repaired-ACT evaluation chain has incomplete path/profile fields")
    compat: str | None = None
    if runtime_profile == "vllm":
        compat_record = _require_mapping(
            runtime.get("vllm_compatibility_adapter"), label="repaired-ACT evaluation chain runtime.vllm_compatibility_adapter"
        )
        compat_value = compat_record.get("path")
        if not isinstance(compat_value, str) or not compat_value:
            raise ValueError("repaired-ACT evaluation chain has no vLLM compatibility adapter path")
        compat = compat_value
    rebuilt = build_repaired_act_evaluation_chain_attestation(
        training_output_state=training_state,
        checkpoint=raw_path,
        tiny_gate_attestation=tiny_path,
        runtime_profile=runtime_profile,
        vllm_compat_adapter=compat,
    )
    if stored != rebuilt:
        raise ValueError("repaired-ACT evaluation chain attestation does not exactly revalidate its evidence")
    if expected_runtime_profile is not None and runtime_profile != expected_runtime_profile:
        raise ValueError("repaired-ACT evaluation chain runtime profile does not match the requested evaluator")
    raw_root = _local_path(raw_path)
    if expected_checkpoint is not None and _local_path(expected_checkpoint) != raw_root:
        raise ValueError("repaired-ACT evaluation chain raw checkpoint does not match the requested evaluator checkpoint")
    if expected_vllm_compat_adapter is not None:
        if runtime_profile != "vllm" or compat is None:
            raise ValueError("requested vLLM compatibility adapter but chain is not a vLLM chain")
        if _local_path(expected_vllm_compat_adapter) != _local_path(compat):
            raise ValueError("repaired-ACT evaluation chain compatibility adapter does not match the requested evaluator checkpoint")
    return {
        **stored,
        "attestation": {"path": str(attestation_path), "sha256": file_sha256(attestation_path)},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-state", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, help="Raw fresh repaired-ACT LocalBackend adapter (path or file:// URI).")
    parser.add_argument("--tiny-gate-attestation", required=True, type=Path)
    parser.add_argument("--runtime-profile", required=True, choices=RUNTIME_PROFILES)
    parser.add_argument("--vllm-compat-adapter", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        attestation = build_repaired_act_evaluation_chain_attestation(
            training_output_state=args.training_output_state,
            checkpoint=args.checkpoint,
            tiny_gate_attestation=args.tiny_gate_attestation,
            runtime_profile=args.runtime_profile,
            vllm_compat_adapter=args.vllm_compat_adapter,
        )
        output, status = write_repaired_act_evaluation_chain_attestation(args.output, attestation)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _parser().error(str(exc))
    print(f"repaired-ACT evaluation chain {status}: {output}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()


__all__ = [
    "FRESH_EXPERIMENT",
    "FRESH_TARGET",
    "FRESH_TRAINING_COMMAND",
    "MODEL",
    "RUNTIME_PROFILES",
    "SCHEMA",
    "build_repaired_act_evaluation_chain_attestation",
    "validate_repaired_act_evaluation_chain_attestation",
    "write_repaired_act_evaluation_chain_attestation",
]
