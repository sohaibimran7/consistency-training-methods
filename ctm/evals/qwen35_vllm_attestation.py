"""Fail-closed evidence checks for Qwen3.5 vLLM LoRA compatibility copies.

vLLM 0.26 can acknowledge a Transformers/PEFT Qwen3.5 adapter while applying
none of it.  These checks intentionally accept only two immutable attestation
formats:

* v1: every production parity variant passes directly; and
* the narrowly scoped composite format below: the production adapter passes
  full/linear/evaluator-path checks, while an otherwise weak self-attention
  signal is independently amplified fourfold and passes on separately bound
  raw and vLLM-compatible adapters.

The composite format is not a generic exception mechanism.  Its exact
reports, adapters, manifests, variant names, and scale are all verified here.
"""

from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


PARITY_ATTESTATION_SCHEMA_V1 = "qwen35-vllm-parity-attestation-v1"
COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1 = "qwen35-vllm-composite-parity-attestation-v1"
RUNTIME_PARITY_REPORT_SCHEMA = "qwen35-lora-runtime-parity-v1"
AMPLIFICATION_MANIFEST_SCHEMA = "qwen35-self-attn-amplified-parity-adapter-v1"
COMPATIBILITY_MANIFEST_SCHEMA = "qwen35-vllm-compat-adapter-v1"

V1_REQUIRED_PARITY_VARIANTS = ("full", "linear_only", "self_attn_only", "evaluator_path")
COMPOSITE_PRIMARY_PASSING_VARIANTS = ("full", "linear_only", "evaluator_path")
COMPOSITE_WEAK_SIGNAL_VARIANT = "self_attn_only"
COMPOSITE_WEAK_SIGNAL_VERDICT = "nonzero_but_delta_mismatch"
COMPOSITE_AMPLIFICATION_SCALE = 4.0
COMPOSITE_AMPLIFIED_REQUIRED_VARIANT = "self_attn_only"
PASSING_VERDICT = "hf_vllm_effects_agree"


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file without accepting directories."""

    source = Path(path)
    digest = sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _absolute_existing_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() or not candidate.exists():
        return None
    return candidate.resolve()


def _absolute_existing_file(value: Any) -> Path | None:
    candidate = _absolute_existing_path(value)
    return candidate if candidate is not None and candidate.is_file() else None


def _absolute_existing_adapter_directory(value: Any) -> Path | None:
    """Resolve an adapter directory only when its weights are present."""

    candidate = _absolute_existing_path(value)
    if candidate is None or not candidate.is_dir():
        return None
    return candidate if (candidate / "adapter_model.safetensors").is_file() else None


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def _same_path(value: Any, expected: Path) -> bool:
    candidate = _absolute_existing_path(value)
    return candidate == expected


def _same_hashed_file(value: Any, path: Path) -> bool:
    return _is_sha256(value) and value == file_sha256(path)


def _adapter_entry_matches(entry: Any, *, path: Path, sha_key: str) -> bool:
    return (
        isinstance(entry, Mapping)
        and _same_path(entry.get("path"), path)
        and _same_hashed_file(entry.get(sha_key), path / "adapter_model.safetensors")
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _primary_weak_signal_result_is_documented(result: Any) -> bool:
    """Require a real but below-threshold self-attention diagnostic signal."""

    if not isinstance(result, Mapping) or result.get("verdict") != COMPOSITE_WEAK_SIGNAL_VERDICT:
        return False
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return False
    hf_max_abs = _finite_number(summary.get("hf_max_abs"))
    vllm_max_abs = _finite_number(summary.get("vllm_max_abs"))
    cosine = _finite_number(summary.get("cosine_similarity"))
    return (
        hf_max_abs is not None
        and hf_max_abs >= 1e-5
        and vllm_max_abs is not None
        and vllm_max_abs >= 1e-5
        and cosine is not None
        and -1.0 <= cosine < 0.90
    )


def _amplified_passing_result_is_documented(result: Any) -> bool:
    if not isinstance(result, Mapping) or result.get("verdict") != PASSING_VERDICT:
        return False
    summary = result.get("summary")
    if not isinstance(summary, Mapping):
        return False
    hf_max_abs = _finite_number(summary.get("hf_max_abs"))
    vllm_max_abs = _finite_number(summary.get("vllm_max_abs"))
    cosine = _finite_number(summary.get("cosine_similarity"))
    return (
        hf_max_abs is not None
        and hf_max_abs >= 1e-5
        and vllm_max_abs is not None
        and vllm_max_abs >= 1e-5
        and cosine is not None
        and cosine >= 0.90
    )


def _verified_v1_attestation(directory: Path, attestation: Mapping[str, Any]) -> bool:
    """Retain the original all-variants v1 contract exactly."""

    adapter = directory / "adapter_model.safetensors"
    if attestation.get("schema") != PARITY_ATTESTATION_SCHEMA_V1:
        return False
    if attestation.get("adapter_path") != str(directory.resolve()):
        return False
    if attestation.get("adapter_model_sha256") != file_sha256(adapter):
        return False
    variants = attestation.get("required_variants")
    verdicts = attestation.get("verdicts")
    if variants != list(V1_REQUIRED_PARITY_VARIANTS) or not isinstance(verdicts, Mapping):
        return False
    if any(verdicts.get(variant) != PASSING_VERDICT for variant in variants):
        return False
    report_path = _absolute_existing_file(attestation.get("report_path"))
    if report_path is None or attestation.get("report_sha256") != file_sha256(report_path):
        return False
    report = _read_json_object(report_path)
    if report is None:
        return False
    report_adapter = report.get("adapter")
    results = report.get("results")
    if not isinstance(report_adapter, Mapping) or not isinstance(results, Mapping):
        return False
    if report_adapter.get("path") != str(directory.resolve()):
        return False
    if report_adapter.get("adapter_model_sha256") != attestation["adapter_model_sha256"]:
        return False
    return all(
        isinstance(results.get(variant), Mapping) and results[variant].get("verdict") == PASSING_VERDICT
        for variant in V1_REQUIRED_PARITY_VARIANTS
    )


def _manifest_entry_matches(entry: Any, *, path: Path, sha_key: str) -> bool:
    return _adapter_entry_matches(entry, path=path, sha_key=sha_key)


def _all_named_lora_b(values: Any, *, family: str | None) -> bool:
    if not isinstance(values, list) or not values:
        return False
    return all(
        isinstance(value, str)
        and ".lora_B" in value
        and ((family in value) if family is not None else ("self_attn" not in value))
        for value in values
    )


def _verified_composite_attestation(directory: Path, attestation: Mapping[str, Any]) -> bool:
    """Verify the one documented amplified-self-attention exception.

    The production adapter must still pass full, DeltaNet/linear-only, and the
    exact evaluator-path probes.  The only permitted primary failure is a
    nonzero self-attention signal below the parity cosine threshold.  A
    separately materialized self-attention-only, 4x adapter must then prove
    that both HF and vLLM apply that family consistently.
    """

    adapter = directory / "adapter_model.safetensors"
    if attestation.get("schema") != COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1:
        return False
    if attestation.get("adapter_path") != str(directory.resolve()):
        return False
    if attestation.get("adapter_model_sha256") != file_sha256(adapter):
        return False
    model = attestation.get("model")
    if not isinstance(model, str) or not model:
        return False

    primary = attestation.get("primary")
    amplified = attestation.get("amplified_self_attn")
    if not isinstance(primary, Mapping) or not isinstance(amplified, Mapping):
        return False

    primary_report_path = _absolute_existing_file(
        primary.get("report", {}).get("path") if isinstance(primary.get("report"), Mapping) else None
    )
    if primary_report_path is None:
        return False
    primary_report_record = primary.get("report")
    if not isinstance(primary_report_record, Mapping) or not _same_hashed_file(
        primary_report_record.get("sha256"), primary_report_path
    ):
        return False
    if not _adapter_entry_matches(primary.get("compat_adapter"), path=directory, sha_key="adapter_model_sha256"):
        return False
    primary_hf = primary.get("hf_adapter")
    if not isinstance(primary_hf, Mapping):
        return False
    primary_hf_path = _absolute_existing_adapter_directory(primary_hf.get("path"))
    if primary_hf_path is None or not _same_hashed_file(
        primary_hf.get("adapter_model_sha256"), primary_hf_path / "adapter_model.safetensors"
    ):
        return False
    if primary.get("passing_variants") != list(COMPOSITE_PRIMARY_PASSING_VARIANTS):
        return False
    verdicts = primary.get("verdicts")
    if not isinstance(verdicts, Mapping) or any(
        verdicts.get(variant) != PASSING_VERDICT for variant in COMPOSITE_PRIMARY_PASSING_VARIANTS
    ):
        return False
    weak_signal = primary.get("weak_signal")
    if weak_signal != {"variant": COMPOSITE_WEAK_SIGNAL_VARIANT, "verdict": COMPOSITE_WEAK_SIGNAL_VERDICT}:
        return False

    primary_report = _read_json_object(primary_report_path)
    if primary_report is None or primary_report.get("schema") != RUNTIME_PARITY_REPORT_SCHEMA:
        return False
    if primary_report.get("model") != model:
        return False
    report_adapter = primary_report.get("adapter")
    results = primary_report.get("results")
    if not isinstance(report_adapter, Mapping) or not isinstance(results, Mapping):
        return False
    if not _adapter_entry_matches(report_adapter, path=directory, sha_key="adapter_model_sha256"):
        return False
    if not _same_path(report_adapter.get("hf_path"), primary_hf_path):
        return False
    if report_adapter.get("hf_adapter_model_sha256") != primary_hf.get("adapter_model_sha256"):
        return False
    if any(
        not isinstance(results.get(variant), Mapping) or results[variant].get("verdict") != PASSING_VERDICT
        for variant in COMPOSITE_PRIMARY_PASSING_VARIANTS
    ):
        return False
    if not _primary_weak_signal_result_is_documented(results.get(COMPOSITE_WEAK_SIGNAL_VARIANT)):
        return False

    amplified_report_record = amplified.get("report")
    amplified_report_path = _absolute_existing_file(
        amplified_report_record.get("path") if isinstance(amplified_report_record, Mapping) else None
    )
    if amplified_report_path is None or not isinstance(amplified_report_record, Mapping):
        return False
    if not _same_hashed_file(amplified_report_record.get("sha256"), amplified_report_path):
        return False
    amplified_hf = amplified.get("hf_adapter")
    amplified_vllm = amplified.get("vllm_adapter")
    if not isinstance(amplified_hf, Mapping) or not isinstance(amplified_vllm, Mapping):
        return False
    amplified_hf_path = _absolute_existing_adapter_directory(amplified_hf.get("path"))
    amplified_vllm_path = _absolute_existing_adapter_directory(amplified_vllm.get("path"))
    if amplified_hf_path is None or amplified_vllm_path is None:
        return False
    if not _same_hashed_file(
        amplified_hf.get("adapter_model_sha256"), amplified_hf_path / "adapter_model.safetensors"
    ) or not _same_hashed_file(
        amplified_vllm.get("adapter_model_sha256"), amplified_vllm_path / "adapter_model.safetensors"
    ):
        return False

    amplification = amplified.get("amplification")
    compatibility = amplified.get("compatibility")
    if not isinstance(amplification, Mapping) or not isinstance(compatibility, Mapping):
        return False
    if amplification.get("family") != "self_attn" or amplification.get("scale") != COMPOSITE_AMPLIFICATION_SCALE:
        return False
    amplification_manifest_record = amplification.get("manifest")
    compatibility_manifest_record = compatibility.get("manifest")
    amplification_manifest_path = _absolute_existing_file(
        amplification_manifest_record.get("path") if isinstance(amplification_manifest_record, Mapping) else None
    )
    compatibility_manifest_path = _absolute_existing_file(
        compatibility_manifest_record.get("path") if isinstance(compatibility_manifest_record, Mapping) else None
    )
    if amplification_manifest_path is None or compatibility_manifest_path is None:
        return False
    if not isinstance(amplification_manifest_record, Mapping) or not isinstance(compatibility_manifest_record, Mapping):
        return False
    if not _same_hashed_file(amplification_manifest_record.get("sha256"), amplification_manifest_path):
        return False
    if not _same_hashed_file(compatibility_manifest_record.get("sha256"), compatibility_manifest_path):
        return False
    amplification_manifest = _read_json_object(amplification_manifest_path)
    compatibility_manifest = _read_json_object(compatibility_manifest_path)
    if amplification_manifest is None or compatibility_manifest is None:
        return False
    if amplification_manifest.get("schema") != AMPLIFICATION_MANIFEST_SCHEMA:
        return False
    if amplification_manifest.get("scale") != COMPOSITE_AMPLIFICATION_SCALE:
        return False
    if not _manifest_entry_matches(amplification_manifest.get("source"), path=primary_hf_path, sha_key="sha256"):
        return False
    if not _manifest_entry_matches(amplification_manifest.get("destination"), path=amplified_hf_path, sha_key="sha256"):
        return False
    if not _all_named_lora_b(amplification_manifest.get("kept_self_attn_lora_b"), family="self_attn"):
        return False
    if not _all_named_lora_b(amplification_manifest.get("zeroed_non_self_attn_lora_b"), family=None):
        return False
    if compatibility_manifest.get("schema") != COMPATIBILITY_MANIFEST_SCHEMA:
        return False
    if not _manifest_entry_matches(
        compatibility_manifest.get("source"), path=amplified_hf_path, sha_key="adapter_model_sha256"
    ):
        return False
    if not _manifest_entry_matches(
        compatibility_manifest.get("destination"), path=amplified_vllm_path, sha_key="adapter_model_sha256"
    ):
        return False
    translation = compatibility_manifest.get("translation")
    if not isinstance(translation, Mapping):
        return False
    if translation.get("source_prefix") != "base_model.model.model.layers.":
        return False
    if translation.get("destination_prefix") != "base_model.model.model.language_model.layers.":
        return False
    tensor_count = translation.get("tensor_count")
    translated_tensor_count = translation.get("translated_tensor_count")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count < 1
        or tensor_count != translated_tensor_count
    ):
        return False

    if amplified.get("required_variant") != COMPOSITE_AMPLIFIED_REQUIRED_VARIANT:
        return False
    if amplified.get("verdict") != PASSING_VERDICT:
        return False
    amplified_report = _read_json_object(amplified_report_path)
    if amplified_report is None or amplified_report.get("schema") != RUNTIME_PARITY_REPORT_SCHEMA:
        return False
    if amplified_report.get("model") != model:
        return False
    amplified_adapter = amplified_report.get("adapter")
    amplified_results = amplified_report.get("results")
    if not isinstance(amplified_adapter, Mapping) or not isinstance(amplified_results, Mapping):
        return False
    if not _adapter_entry_matches(amplified_adapter, path=amplified_vllm_path, sha_key="adapter_model_sha256"):
        return False
    if not _same_path(amplified_adapter.get("hf_path"), amplified_hf_path):
        return False
    if amplified_adapter.get("hf_adapter_model_sha256") != amplified_hf.get("adapter_model_sha256"):
        return False
    token_protocol = amplified_report.get("token_protocol")
    backend = amplified_report.get("backends")
    vllm_backend = backend.get("vllm") if isinstance(backend, Mapping) else None
    if not isinstance(token_protocol, Mapping) or token_protocol.get("requested_result_variants") != [
        COMPOSITE_AMPLIFIED_REQUIRED_VARIANT
    ]:
        return False
    if not isinstance(vllm_backend, Mapping) or vllm_backend.get("enforce_eager") is not True:
        return False
    if vllm_backend.get("isolate_vllm_variants") is not True:
        return False
    return _amplified_passing_result_is_documented(amplified_results.get(COMPOSITE_AMPLIFIED_REQUIRED_VARIANT))


def is_verified_qwen35_vllm_compat_adapter(directory: str | Path) -> bool:
    """Return whether an adapter has immutable, sufficient parity evidence."""

    root = Path(directory).resolve()
    adapter = root / "adapter_model.safetensors"
    attestation_path = root / "vllm-parity-attestation.json"
    if not adapter.is_file() or not attestation_path.is_file():
        return False
    attestation = _read_json_object(attestation_path)
    if attestation is None:
        return False
    try:
        schema = attestation.get("schema")
        if schema == PARITY_ATTESTATION_SCHEMA_V1:
            return _verified_v1_attestation(root, attestation)
        if schema == COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1:
            return _verified_composite_attestation(root, attestation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return False


def validate_qwen35_vllm_composite_attestation(
    directory: str | Path,
    attestation: Mapping[str, Any],
) -> bool:
    """Validate an unwritten composite document before an immutable write."""

    root = Path(directory).resolve()
    adapter = root / "adapter_model.safetensors"
    if not adapter.is_file():
        return False
    try:
        return _verified_composite_attestation(root, attestation)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


__all__ = [
    "AMPLIFICATION_MANIFEST_SCHEMA",
    "COMPATIBILITY_MANIFEST_SCHEMA",
    "COMPOSITE_AMPLIFICATION_SCALE",
    "COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1",
    "COMPOSITE_PRIMARY_PASSING_VARIANTS",
    "COMPOSITE_WEAK_SIGNAL_VARIANT",
    "COMPOSITE_WEAK_SIGNAL_VERDICT",
    "PARITY_ATTESTATION_SCHEMA_V1",
    "PASSING_VERDICT",
    "file_sha256",
    "is_verified_qwen35_vllm_compat_adapter",
    "validate_qwen35_vllm_composite_attestation",
]
