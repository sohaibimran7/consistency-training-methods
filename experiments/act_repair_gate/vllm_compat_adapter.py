"""Make a read-only Qwen3.5 PEFT LoRA copy loadable by vLLM's VL wrapper.

Transformers loads Qwen3.5-9B as a causal language model, whose PEFT adapter
keys name text layers as ``base_model.model.model.layers.*``.  vLLM 0.26
selects its conditional-generation wrapper for the same checkpoint.  That
wrapper expects the text tower to be named ``model.language_model`` before its
normal weight mapper is applied.  Without that segment vLLM accepts and lists
the adapter, but applies no LoRA tensors.

This utility does *not* edit a trained adapter.  It writes a new adapter
directory, changing only that path segment in Safetensors keys.  The result
must still pass the fixed-token HF/vLLM parity probe before it is used for an
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from ctm.evals.qwen35_vllm_attestation import (
    COMPOSITE_AMPLIFICATION_SCALE,
    COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1,
    COMPOSITE_PRIMARY_PASSING_VARIANTS,
    COMPOSITE_WEAK_SIGNAL_VARIANT,
    COMPOSITE_WEAK_SIGNAL_VERDICT,
    PASSING_VERDICT,
    file_sha256,
    validate_qwen35_vllm_composite_attestation,
)


SCHEMA = "qwen35-vllm-compat-adapter-v1"
PARITY_ATTESTATION_SCHEMA = "qwen35-vllm-parity-attestation-v1"
SOURCE_PREFIX = "base_model.model.model.layers."
DESTINATION_PREFIX = "base_model.model.model.language_model.layers."
REQUIRED_PARITY_VARIANTS = ("full", "linear_only", "self_attn_only", "evaluator_path")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def translate_lora_key(key: str) -> str:
    """Return the vLLM-wrapper spelling of one PEFT adapter tensor key."""

    if key.startswith(SOURCE_PREFIX):
        return DESTINATION_PREFIX + key.removeprefix(SOURCE_PREFIX)
    return key


def make_compat_adapter(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy one adapter once, refusing to overwrite either source or output."""

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - requires GPU/runtime deps
        raise RuntimeError("qwen3.5 compatibility adapter needs safetensors") from exc

    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    config = source_path / "adapter_config.json"
    weights = source_path / "adapter_model.safetensors"
    if not config.is_file() or not weights.is_file():
        raise FileNotFoundError(
            "adapter must contain adapter_config.json and adapter_model.safetensors: "
            f"{source_path}"
        )
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite compatibility adapter: {destination_path}")

    tensors = load_file(str(weights), device="cpu")
    translated: dict[str, Any] = {}
    changed: list[str] = []
    for key, tensor in tensors.items():
        new_key = translate_lora_key(key)
        if new_key in translated:
            raise ValueError(f"key collision while translating {key!r} to {new_key!r}")
        translated[new_key] = tensor
        if new_key != key:
            changed.append(key)
    if not changed:
        raise ValueError(f"{source_path}: no Qwen3.5 text-layer LoRA keys matched {SOURCE_PREFIX!r}")

    destination_path.mkdir(parents=True)
    shutil.copy2(config, destination_path / config.name)
    for optional_name in ("README.md", "manifest.json"):
        optional = source_path / optional_name
        if optional.is_file():
            shutil.copy2(optional, destination_path / optional_name)
    save_file(translated, str(destination_path / weights.name), metadata={"format": "pt"})
    report = {
        "schema": SCHEMA,
        "source": {
            "path": str(source_path),
            "adapter_model_sha256": _sha256(weights),
        },
        "destination": {
            "path": str(destination_path),
            "adapter_model_sha256": _sha256(destination_path / weights.name),
        },
        "translation": {
            "source_prefix": SOURCE_PREFIX,
            "destination_prefix": DESTINATION_PREFIX,
            "tensor_count": len(tensors),
            "translated_tensor_count": len(changed),
            "translated_tensor_names_sha256": hashlib.sha256(
                json.dumps(changed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        },
    }
    report_path = destination_path / "compatibility-manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def attest_compat_adapter(adapter: str | Path, report_path: str | Path) -> dict[str, Any]:
    """Bind a compatibility copy to one successful immutable parity report.

    The evaluator bridge accepts a Qwen3.5 vLLM compatibility copy only when
    this small attestation and its referenced report both still match the
    adapter bytes.  This makes the exception auditable and prevents a generic
    ``allow Qwen3.5`` switch from reintroducing the silent-base failure.
    """

    adapter_path = Path(adapter).resolve()
    weights = adapter_path / "adapter_model.safetensors"
    target = adapter_path / "vllm-parity-attestation.json"
    report = Path(report_path).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"compatibility adapter has no adapter_model.safetensors: {adapter_path}")
    if not report.is_file():
        raise FileNotFoundError(f"parity report does not exist: {report}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite parity attestation: {target}")
    try:
        parsed = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid parity report: {report}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"parity report is not an object: {report}")
    report_adapter = parsed.get("adapter")
    results = parsed.get("results")
    actual_hash = _sha256(weights)
    if not isinstance(report_adapter, Mapping) or report_adapter.get("path") != str(adapter_path):
        raise ValueError("parity report was not run against this exact vLLM compatibility adapter")
    if report_adapter.get("adapter_model_sha256") != actual_hash:
        raise ValueError("parity report adapter hash does not match the compatibility adapter")
    if not isinstance(results, Mapping):
        raise ValueError("parity report has no result mapping")
    verdicts: dict[str, str] = {}
    for variant in REQUIRED_PARITY_VARIANTS:
        result = results.get(variant)
        verdict = result.get("verdict") if isinstance(result, Mapping) else None
        if verdict != "hf_vllm_effects_agree":
            raise ValueError(f"parity report does not pass {variant!r}: {verdict!r}")
        verdicts[variant] = verdict
    attestation = {
        "schema": PARITY_ATTESTATION_SCHEMA,
        "adapter_path": str(adapter_path),
        "adapter_model_sha256": actual_hash,
        "model": parsed.get("model"),
        "report_path": str(report),
        "report_sha256": _sha256(report),
        "required_variants": list(REQUIRED_PARITY_VARIANTS),
        "verdicts": verdicts,
    }
    target.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return attestation


def _required_report_adapter_path(report: Mapping[str, Any], *, field: str) -> Path:
    adapter = report.get("adapter")
    value = adapter.get(field) if isinstance(adapter, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"parity report has no adapter.{field}")
    path = Path(value).expanduser().resolve()
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"parity report adapter.{field} is not an adapter directory: {path}")
    return path


def _read_runtime_parity_report(path: str | Path) -> tuple[Path, Mapping[str, Any]]:
    report_path = Path(path).resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"parity report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid parity report: {report_path}") from exc
    if not isinstance(report, Mapping):
        raise ValueError(f"parity report is not an object: {report_path}")
    return report_path, report


def attest_compat_adapter_with_amplified_self_attention(
    adapter: str | Path,
    *,
    primary_report_path: str | Path,
    amplified_self_attention_report_path: str | Path,
) -> dict[str, Any]:
    """Bind the narrowly validated weak-self-attention parity evidence.

    This is intentionally not a generic fallback for a failed parity report.
    The production adapter must pass full, linear-only, and evaluator-path
    probes directly. Its only permitted failure is a nonzero weak
    self-attention signal. The separately materialized 4x self-attention-only
    diagnostic must pass from a fresh eager one-slot vLLM server. The shared
    validator checks every report, adapter, manifest, path, and byte hash
    before this immutable attestation is written.
    """

    adapter_path = Path(adapter).resolve()
    target = adapter_path / "vllm-parity-attestation.json"
    weights = adapter_path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"compatibility adapter has no adapter_model.safetensors: {adapter_path}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite parity attestation: {target}")

    primary_path, primary = _read_runtime_parity_report(primary_report_path)
    amplified_path, amplified = _read_runtime_parity_report(amplified_self_attention_report_path)
    if primary.get("model") != amplified.get("model") or not isinstance(primary.get("model"), str):
        raise ValueError("primary and amplified parity reports must name the same non-empty model")

    primary_compat = _required_report_adapter_path(primary, field="path")
    primary_hf = _required_report_adapter_path(primary, field="hf_path")
    amplified_compat = _required_report_adapter_path(amplified, field="path")
    amplified_hf = _required_report_adapter_path(amplified, field="hf_path")
    if primary_compat != adapter_path:
        raise ValueError("primary parity report was not run against this exact compatibility adapter")

    primary_results = primary.get("results")
    amplified_results = amplified.get("results")
    if not isinstance(primary_results, Mapping) or not isinstance(amplified_results, Mapping):
        raise ValueError("parity report has no result mapping")
    if any(
        not isinstance(primary_results.get(variant), Mapping)
        or primary_results[variant].get("verdict") != PASSING_VERDICT
        for variant in COMPOSITE_PRIMARY_PASSING_VARIANTS
    ):
        raise ValueError("primary report does not pass full, linear-only, and evaluator-path variants")
    if (
        not isinstance(primary_results.get(COMPOSITE_WEAK_SIGNAL_VARIANT), Mapping)
        or primary_results[COMPOSITE_WEAK_SIGNAL_VARIANT].get("verdict") != COMPOSITE_WEAK_SIGNAL_VERDICT
    ):
        raise ValueError("primary report does not contain the documented weak self-attention result")
    if (
        not isinstance(amplified_results.get(COMPOSITE_WEAK_SIGNAL_VARIANT), Mapping)
        or amplified_results[COMPOSITE_WEAK_SIGNAL_VARIANT].get("verdict") != PASSING_VERDICT
    ):
        raise ValueError("amplified report does not pass its self-attention-only variant")

    amplification_manifest = amplified_hf / "parity-amplification-manifest.json"
    compatibility_manifest = amplified_compat / "compatibility-manifest.json"
    if not amplification_manifest.is_file() or not compatibility_manifest.is_file():
        raise FileNotFoundError("amplified adapters are missing their required immutable manifests")

    primary_hash = _sha256(weights)
    attestation = {
        "schema": COMPOSITE_PARITY_ATTESTATION_SCHEMA_V1,
        "adapter_path": str(adapter_path),
        "adapter_model_sha256": primary_hash,
        "model": primary["model"],
        "primary": {
            "report": {"path": str(primary_path), "sha256": file_sha256(primary_path)},
            "compat_adapter": {"path": str(adapter_path), "adapter_model_sha256": primary_hash},
            "hf_adapter": {
                "path": str(primary_hf),
                "adapter_model_sha256": _sha256(primary_hf / "adapter_model.safetensors"),
            },
            "passing_variants": list(COMPOSITE_PRIMARY_PASSING_VARIANTS),
            "verdicts": {variant: PASSING_VERDICT for variant in COMPOSITE_PRIMARY_PASSING_VARIANTS},
            "weak_signal": {
                "variant": COMPOSITE_WEAK_SIGNAL_VARIANT,
                "verdict": COMPOSITE_WEAK_SIGNAL_VERDICT,
            },
        },
        "amplified_self_attn": {
            "report": {"path": str(amplified_path), "sha256": file_sha256(amplified_path)},
            "hf_adapter": {
                "path": str(amplified_hf),
                "adapter_model_sha256": _sha256(amplified_hf / "adapter_model.safetensors"),
            },
            "vllm_adapter": {
                "path": str(amplified_compat),
                "adapter_model_sha256": _sha256(amplified_compat / "adapter_model.safetensors"),
            },
            "amplification": {
                "family": "self_attn",
                "scale": COMPOSITE_AMPLIFICATION_SCALE,
                "manifest": {"path": str(amplification_manifest), "sha256": file_sha256(amplification_manifest)},
            },
            "compatibility": {
                "manifest": {"path": str(compatibility_manifest), "sha256": file_sha256(compatibility_manifest)},
            },
            "required_variant": COMPOSITE_WEAK_SIGNAL_VARIANT,
            "verdict": PASSING_VERDICT,
        },
    }
    if not validate_qwen35_vllm_composite_attestation(adapter_path, attestation):
        raise ValueError("composite Qwen3.5 parity evidence failed its fail-closed validator")
    target.write_text(json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return attestation


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original PEFT adapter to copy.")
    parser.add_argument("--destination", type=Path, help="New vLLM compatibility adapter directory.")
    parser.add_argument("--adapter", type=Path, help="Existing compatibility adapter to attest.")
    parser.add_argument("--parity-report", type=Path, help="Successful runtime_parity.py report for --adapter.")
    parser.add_argument(
        "--primary-parity-report",
        type=Path,
        help="Primary weak-self-attention runtime parity report for composite attestation.",
    )
    parser.add_argument(
        "--amplified-self-attn-report",
        type=Path,
        help="Passing amplified self-attention-only parity report for composite attestation.",
    )
    args = parser.parse_args(argv)
    try:
        copying = args.source is not None or args.destination is not None
        # ``--adapter`` is shared by both attestation modes.  The report flag,
        # rather than adapter presence, selects the direct all-variants mode.
        attesting = args.parity_report is not None
        composite_attesting = (
            args.adapter is not None
            or args.primary_parity_report is not None
            or args.amplified_self_attn_report is not None
        ) and args.parity_report is None
        if sum((copying, attesting, composite_attesting)) != 1:
            raise ValueError(
                "choose exactly one mode: --source/--destination, --adapter/--parity-report, "
                "or --adapter/--primary-parity-report/--amplified-self-attn-report"
            )
        if copying:
            if args.source is None or args.destination is None:
                raise ValueError("copy mode requires both --source and --destination")
            report = make_compat_adapter(args.source, args.destination)
        elif attesting:
            if args.adapter is None or args.parity_report is None:
                raise ValueError("attestation mode requires both --adapter and --parity-report")
            report = attest_compat_adapter(args.adapter, args.parity_report)
        else:
            if (
                args.adapter is None
                or args.primary_parity_report is None
                or args.amplified_self_attn_report is None
            ):
                raise ValueError(
                    "composite attestation requires --adapter, --primary-parity-report, "
                    "and --amplified-self-attn-report"
                )
            report = attest_compat_adapter_with_amplified_self_attention(
                args.adapter,
                primary_report_path=args.primary_parity_report,
                amplified_self_attention_report_path=args.amplified_self_attn_report,
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    main()
