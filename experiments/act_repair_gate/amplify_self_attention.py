"""Materialize the auditable 4x Qwen3.5 self-attention parity probe.

Qwen3.5's ordinary self-attention LoRA tensors can be much smaller than its
DeltaNet / linear-attention tensors.  The production adapter is never altered:
this utility makes a fresh diagnostic PEFT adapter where self-attention
LoRA-B tensors are multiplied by exactly four and every other LoRA-B tensor is
zeroed.  A later HF/vLLM parity run can therefore establish that the weak
self-attention transport signal is genuinely applied by both runtimes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from ctm.evals.qwen35_vllm_attestation import (
    AMPLIFICATION_MANIFEST_SCHEMA,
    COMPOSITE_AMPLIFICATION_SCALE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lora_b_family(key: str) -> str | None:
    """Return the Qwen3.5 projection family of one LoRA-B tensor."""

    if ".lora_B." not in key and not key.endswith(".lora_B.weight"):
        return None
    if "self_attn" in key:
        return "self_attn"
    if "linear_attn" in key:
        return "linear_attn"
    # Some historical adapters include ordinary SwiGLU MLP projections. This
    # diagnostic is *self-attention only*, so known MLP LoRA-B tensors must be
    # zeroed rather than rejected. Keep the check narrow: an unknown
    # projection family is still a hard error rather than silently becoming
    # part of the probe.
    if ".mlp." in key and any(f".mlp.{projection}." in key for projection in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    return "unsupported"


def make_amplified_self_attention_adapter(
    source: str | Path,
    destination: str | Path,
    *,
    scale: float = COMPOSITE_AMPLIFICATION_SCALE,
) -> Mapping[str, Any]:
    """Create one fresh, self-attention-only fourfold PEFT diagnostic adapter."""

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - requires runtime dependency
        raise RuntimeError("self-attention parity amplification needs safetensors") from exc

    if scale != COMPOSITE_AMPLIFICATION_SCALE:
        raise ValueError(f"the composite Qwen3.5 attestation requires the fixed scale {COMPOSITE_AMPLIFICATION_SCALE}")
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    config = source_path / "adapter_config.json"
    weights = source_path / "adapter_model.safetensors"
    source_manifest = source_path / "manifest.json"
    if not config.is_file() or not weights.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(f"adapter must contain adapter_config.json, adapter_model.safetensors, and manifest.json: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite amplified adapter: {destination_path}")

    tensors = load_file(str(weights), device="cpu")
    amplified: dict[str, Any] = {}
    kept_self_attn_lora_b: list[str] = []
    zeroed_non_self_attn_lora_b: list[str] = []
    for key, tensor in tensors.items():
        family = _lora_b_family(key)
        if family == "self_attn":
            amplified[key] = tensor * scale
            kept_self_attn_lora_b.append(key)
        elif family in {"linear_attn", "mlp"}:
            amplified[key] = tensor.new_zeros(tensor.shape)
            zeroed_non_self_attn_lora_b.append(key)
        elif family == "unsupported":
            raise ValueError(f"{source_path}: unsupported LoRA-B projection family in {key!r}; expected self_attn, linear_attn, or a Qwen3.5 MLP gate/up/down projection")
        else:
            amplified[key] = tensor
    if not kept_self_attn_lora_b:
        raise ValueError(f"{source_path}: no self-attention LoRA-B tensors were found")
    if not zeroed_non_self_attn_lora_b:
        raise ValueError(f"{source_path}: no non-self-attention LoRA-B tensors were found")

    destination_path.mkdir(parents=True)
    shutil.copy2(config, destination_path / config.name)
    # The source checkpoint provenance is part of the diagnostic's audit
    # chain, so manifest.json is required above and copied byte-for-byte.
    shutil.copy2(source_manifest, destination_path / source_manifest.name)
    for optional_name in ("README.md",):
        optional = source_path / optional_name
        if optional.is_file():
            shutil.copy2(optional, destination_path / optional_name)
    destination_weights = destination_path / weights.name
    save_file(amplified, str(destination_weights), metadata={"format": "pt"})
    manifest = {
        "schema": AMPLIFICATION_MANIFEST_SCHEMA,
        "scale": COMPOSITE_AMPLIFICATION_SCALE,
        "source": {"path": str(source_path), "sha256": _sha256(weights)},
        "destination": {"path": str(destination_path), "sha256": _sha256(destination_weights)},
        "kept_self_attn_lora_b": sorted(kept_self_attn_lora_b),
        "zeroed_non_self_attn_lora_b": sorted(zeroed_non_self_attn_lora_b),
    }
    manifest_path = destination_path / "parity-amplification-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original raw PEFT adapter.")
    parser.add_argument("--destination", required=True, type=Path, help="Fresh amplified PEFT adapter directory.")
    args = parser.parse_args(argv)
    try:
        result = make_amplified_self_attention_adapter(args.source, args.destination)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    main()
