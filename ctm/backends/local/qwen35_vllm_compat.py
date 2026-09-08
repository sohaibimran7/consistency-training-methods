"""Fail-closed Qwen3.5 PEFT-to-vLLM snapshots for on-policy rollouts.

Transformers/PEFT writes Qwen3.5 text-tower LoRA tensors under
``base_model.model.model.layers.*``.  The validated vLLM Qwen3.5 stack selects
a conditional-generation wrapper for the same checkpoint and expects the corresponding
runtime tensors under ``base_model.model.model.language_model.layers.*``.
It otherwise accepts the adapter but silently applies none of it.

Training checkpoints must remain in the Transformers/PEFT spelling: the
coordinator uses them for differentiable forward/backward passes and they are
the source of record.  This module therefore creates a sibling, immutable
vLLM-only copy for *each published policy version*.  The copy is bound to its
raw source by hashes and a deterministic key/tensor-content invariant.  The
vLLM sampler accepts a Qwen3.5 adapter only after that sidecar validates.

This is deliberately distinct from the evaluation attestation in
``ctm.evals.qwen35_vllm_attestation``.  A training run creates a new adapter
after every optimizer update, so a full external HF/vLLM parity report cannot
be byte-bound to every snapshot.  Instead, the translation is verified when
it is materialized, and the cheap hash-bound sidecar is rechecked before every
worker loads the version.  A real fixed-token parity preflight on a non-zero
adapter is still required for a new vLLM/Qwen stack before paid training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "qwen35-vllm-rollout-compat-v1"
MANIFEST_NAME = "ctm_qwen35_vllm_rollout_compat.json"
WORKER_PARITY_ATTESTATION_SCHEMA = "qwen35-rollout-worker-parity-attestation-v1"
WORKER_PARITY_ATTESTATION_NAME = "qwen35-rollout-worker-parity-attestation.json"
WORKER_PARITY_MIN_EFFECT = 1e-5
WORKER_PARITY_MIN_COSINE = 0.90
SOURCE_PREFIX = "base_model.model.model.layers."
DESTINATION_PREFIX = "base_model.model.model.language_model.layers."
_QWEN35_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_5_moe"})


def is_qwen35_model_name(model_name: str | None) -> bool:
    """Return whether a model identifier/path uses the affected Qwen3.5 vLLM wrapper.

    Public model IDs retain the historical name-substring check.  A local
    snapshot may be relocated to an opaque path (for example,
    ``/models/pinned``), so inspect only that directory's local
    ``config.json`` as a fallback.  This deliberately does not use
    ``AutoConfig`` or any Hub client: detecting the safety-critical wrapper
    must never trigger a network request.
    """

    if not isinstance(model_name, str):
        return False
    if "qwen3.5" in model_name.lower().replace("_", "."):
        return True

    try:
        config = json.loads((Path(model_name) / "config.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    return isinstance(config, Mapping) and config.get("model_type") in _QWEN35_MODEL_TYPES


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def translate_lora_key(key: str) -> str:
    """Map one Transformers/PEFT Qwen3.5 text-tower key to vLLM spelling."""

    if key.startswith(SOURCE_PREFIX):
        return DESTINATION_PREFIX + key.removeprefix(SOURCE_PREFIX)
    return key


def _relative_adapter_path(*, root: Path, target: Path) -> str:
    """Return a portable relative path while rejecting a self-inconsistent one."""

    relative = os.path.relpath(target, start=root)
    if (root / relative).resolve() != target:
        raise ValueError(f"could not bind compatibility adapter path {target} relative to {root}")
    return relative


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _tensor_content_sha256(tensors: Mapping[str, Any], *, key_transform) -> str:
    """Hash tensor names, dtypes, shapes, and bytes in one canonical order."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - LocalBackend already needs torch
        raise RuntimeError("Qwen3.5 rollout compatibility needs torch") from exc

    digest = hashlib.sha256()
    normalized: list[tuple[str, Any]] = []
    for key, tensor in tensors.items():
        normalized.append((key_transform(key), tensor))
    for key, tensor in sorted(normalized, key=lambda item: item[0]):
        if not isinstance(key, str):
            raise TypeError("adapter tensor key is not a string")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"adapter tensor {key!r} is not a torch.Tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        # ``Tensor.numpy`` does not support bfloat16 on all supported torch
        # versions. Viewing the contiguous storage as bytes does.
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _lora_b_stats(tensors: Mapping[str, Any]) -> dict[str, float | int]:
    """Summarize the actual LoRA update factor without changing any tensor."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - LocalBackend already needs torch
        raise RuntimeError("Qwen3.5 rollout compatibility needs torch") from exc

    lora_b = [tensor for key, tensor in tensors.items() if ".lora_B." in key or key.endswith(".lora_B.weight")]
    if not lora_b:
        raise ValueError("Qwen3.5 LoRA snapshot has no lora_B tensors")
    nonzero = 0
    max_abs = 0.0
    for tensor in lora_b:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("LoRA-B entry is not a torch.Tensor")
        value = tensor.detach().cpu()
        nonzero += int(torch.count_nonzero(value).item())
        if value.numel():
            max_abs = max(max_abs, float(value.float().abs().max().item()))
    return {
        "lora_b_tensor_count": len(lora_b),
        "lora_b_nonzero_elements": nonzero,
        "lora_b_max_abs": max_abs,
    }


def _require_adapter_directory(path: Path, *, label: str) -> tuple[Path, Path]:
    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not path.is_dir() or not config.is_file() or not weights.is_file():
        raise FileNotFoundError(f"{label} must contain adapter_config.json and adapter_model.safetensors: {path}")
    return config, weights


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Qwen3.5 rollout compatibility manifest: {path}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"Qwen3.5 rollout compatibility manifest is not an object: {path}")
    return parsed


def _resolve_manifest_adapter(
    manifest: Mapping[str, Any],
    *,
    manifest_root: Path,
    field: str,
) -> tuple[Path, Mapping[str, Any]]:
    entry = manifest.get(field)
    if not isinstance(entry, Mapping):
        raise ValueError(f"Qwen3.5 rollout compatibility manifest has no {field} object")
    relative = entry.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Qwen3.5 rollout compatibility manifest has no {field}.relative_path")
    candidate = (manifest_root / relative).resolve()
    # Published layout is deliberately narrow: raw is an immutable sibling,
    # compatibility weights are the manifest's own directory.  Do not permit
    # a general pointer to an unrelated adapter from a worker process.
    if field == "source_adapter":
        expected = manifest_root.parent / "raw"
    elif field == "destination_adapter":
        expected = manifest_root
    else:  # pragma: no cover - internal caller only
        raise AssertionError(field)
    if candidate != expected.resolve():
        raise ValueError(f"Qwen3.5 rollout compatibility manifest {field} path is not the expected immutable sibling")
    return candidate, entry


def _validated_adapter_hashes(path: Path, entry: Mapping[str, Any], *, label: str) -> tuple[Path, Path]:
    config, weights = _require_adapter_directory(path, label=label)
    config_hash = entry.get("adapter_config_sha256")
    weight_hash = entry.get("adapter_model_sha256")
    if not isinstance(config_hash, str) or config_hash != file_sha256(config):
        raise ValueError(f"Qwen3.5 rollout compatibility {label} config hash mismatch")
    if not isinstance(weight_hash, str) or weight_hash != file_sha256(weights):
        raise ValueError(f"Qwen3.5 rollout compatibility {label} weight hash mismatch")
    return config, weights


def validate_qwen35_vllm_rollout_compat_adapter(
    adapter_dir: str | Path,
    *,
    expected_version: int | None = None,
    strict_tensor_content: bool = False,
) -> dict[str, Any]:
    """Validate one immutable dynamic Qwen3.5 compatibility adapter.

    ``strict_tensor_content`` loads both safetensors files and proves the
    key-renaming bijection plus identical tensor bytes.  Materialization uses
    it once before publication.  Workers use the cheaper default path: file
    hashes and the already-bound manifest are sufficient to catch a stale,
    partial, or replaced snapshot before vLLM can consume it.
    """

    root = Path(adapter_dir).resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(
            "vLLM LoRA policy sampling for Qwen3.5 requires a per-version, hash-bound "
            f"compatibility snapshot ({MANIFEST_NAME} missing at {root})"
        )
    manifest = _read_manifest(manifest_path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unexpected Qwen3.5 rollout compatibility manifest schema")
    model = manifest.get("model")
    if not is_qwen35_model_name(model):
        raise ValueError(f"Qwen3.5 rollout compatibility manifest has non-Qwen3.5 model {model!r}")
    version = manifest.get("adapter_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Qwen3.5 rollout compatibility manifest has invalid adapter_version")
    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"Qwen3.5 rollout compatibility snapshot version mismatch: manifest={version}, expected={expected_version}"
        )

    source, source_entry = _resolve_manifest_adapter(manifest, manifest_root=root, field="source_adapter")
    destination, destination_entry = _resolve_manifest_adapter(
        manifest, manifest_root=root, field="destination_adapter"
    )
    source_config, source_weights = _validated_adapter_hashes(source, source_entry, label="source adapter")
    destination_config, destination_weights = _validated_adapter_hashes(
        destination, destination_entry, label="destination adapter"
    )
    if source_config.read_bytes() != destination_config.read_bytes():
        raise ValueError("Qwen3.5 rollout compatibility adapter_config.json differs from the raw source")

    translation = manifest.get("translation")
    if not isinstance(translation, Mapping):
        raise ValueError("Qwen3.5 rollout compatibility manifest has no translation object")
    if translation.get("source_prefix") != SOURCE_PREFIX or translation.get("destination_prefix") != DESTINATION_PREFIX:
        raise ValueError("Qwen3.5 rollout compatibility manifest names an unexpected key translation")
    tensor_count = translation.get("tensor_count")
    translated_count = translation.get("translated_tensor_count")
    if (
        isinstance(tensor_count, bool)
        or not isinstance(tensor_count, int)
        or tensor_count < 1
        or isinstance(translated_count, bool)
        or not isinstance(translated_count, int)
        or not 1 <= translated_count <= tensor_count
    ):
        raise ValueError("Qwen3.5 rollout compatibility manifest has invalid tensor counts")
    for key in (
        "source_tensor_keyset_sha256",
        "destination_tensor_keyset_sha256",
        "tensor_content_sha256",
    ):
        value = translation.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Qwen3.5 rollout compatibility manifest has invalid translation.{key}")
    lora_b = translation.get("lora_b")
    if not isinstance(lora_b, Mapping):
        raise ValueError("Qwen3.5 rollout compatibility manifest has no translation.lora_b")
    for key in ("lora_b_tensor_count", "lora_b_nonzero_elements"):
        value = lora_b.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Qwen3.5 rollout compatibility manifest has invalid translation.lora_b.{key}")
    max_abs = lora_b.get("lora_b_max_abs")
    if isinstance(max_abs, bool) or not isinstance(max_abs, (int, float)) or max_abs < 0:
        raise ValueError("Qwen3.5 rollout compatibility manifest has invalid translation.lora_b.lora_b_max_abs")

    if strict_tensor_content:
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover - exercised on GPU/runtime environments
            raise RuntimeError("strict Qwen3.5 rollout compatibility validation needs safetensors and torch") from exc
        source_tensors = load_file(str(source_weights), device="cpu")
        destination_tensors = load_file(str(destination_weights), device="cpu")
        expected_destination = {translate_lora_key(key): tensor for key, tensor in source_tensors.items()}
        if len(expected_destination) != len(source_tensors):
            raise ValueError("Qwen3.5 rollout compatibility translation produced a tensor-key collision")
        if set(destination_tensors) != set(expected_destination):
            raise ValueError("Qwen3.5 rollout compatibility destination tensor keys are not the translated source keys")
        changed = sum(key != translate_lora_key(key) for key in source_tensors)
        if len(source_tensors) != tensor_count or changed != translated_count:
            raise ValueError("Qwen3.5 rollout compatibility manifest tensor counts do not match its files")
        for key, source_tensor in source_tensors.items():
            destination_tensor = destination_tensors[translate_lora_key(key)]
            if source_tensor.dtype != destination_tensor.dtype or tuple(source_tensor.shape) != tuple(
                destination_tensor.shape
            ):
                raise ValueError(f"Qwen3.5 rollout compatibility tensor metadata changed for {key!r}")
            if not torch.equal(source_tensor, destination_tensor):
                raise ValueError(f"Qwen3.5 rollout compatibility tensor bytes changed for {key!r}")
        if translation["source_tensor_keyset_sha256"] != _json_sha256(sorted(source_tensors)):
            raise ValueError("Qwen3.5 rollout compatibility source key-set hash mismatch")
        if translation["destination_tensor_keyset_sha256"] != _json_sha256(sorted(destination_tensors)):
            raise ValueError("Qwen3.5 rollout compatibility destination key-set hash mismatch")
        content_hash = _tensor_content_sha256(source_tensors, key_transform=lambda key: key)
        if content_hash != _tensor_content_sha256(
            destination_tensors,
            key_transform=lambda key: (
                SOURCE_PREFIX + key.removeprefix(DESTINATION_PREFIX) if key.startswith(DESTINATION_PREFIX) else key
            ),
        ):
            raise ValueError("Qwen3.5 rollout compatibility destination tensor content hash mismatch")
        if translation["tensor_content_sha256"] != content_hash:
            raise ValueError("Qwen3.5 rollout compatibility manifest tensor-content hash mismatch")
        if dict(lora_b) != _lora_b_stats(source_tensors):
            raise ValueError("Qwen3.5 rollout compatibility manifest LoRA-B summary mismatch")

    # Return an ordinary dict so callers cannot accidentally rely on a mutable
    # JSON Mapping implementation.
    return dict(manifest)


def materialize_qwen35_vllm_rollout_compat_adapter(
    source_adapter: str | Path,
    destination_adapter: str | Path,
    *,
    model: str,
    adapter_version: int,
) -> dict[str, Any]:
    """Create one immutable translated vLLM snapshot without editing ``source``.

    Callers create the raw adapter under ``<version>/raw`` and this function
    creates ``<version>/vllm_compat``.  Both directories are completed before
    the enclosing version directory is atomically published to workers.
    """

    if not is_qwen35_model_name(model):
        raise ValueError(f"Qwen3.5 compatibility requested for non-Qwen3.5 model {model!r}")
    if isinstance(adapter_version, bool) or not isinstance(adapter_version, int) or adapter_version < 1:
        raise ValueError("adapter_version must be a positive integer")
    source = Path(source_adapter).resolve()
    destination = Path(destination_adapter).resolve()
    source_config, source_weights = _require_adapter_directory(source, label="raw source adapter")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Qwen3.5 vLLM compatibility adapter: {destination}")
    if destination.name != "vllm_compat" or source.name != "raw" or source.parent != destination.parent:
        raise ValueError("Qwen3.5 rollout snapshots must use immutable sibling raw/ and vllm_compat/ directories")

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - requires the local GPU runtime
        raise RuntimeError("Qwen3.5 rollout compatibility needs safetensors") from exc

    source_tensors = load_file(str(source_weights), device="cpu")
    destination_tensors: dict[str, Any] = {}
    changed = 0
    for key, tensor in source_tensors.items():
        translated = translate_lora_key(key)
        if translated in destination_tensors:
            raise ValueError(f"Qwen3.5 compatibility key collision translating {key!r} to {translated!r}")
        destination_tensors[translated] = tensor
        changed += translated != key
    if not changed:
        raise ValueError(
            f"{source}: no Qwen3.5 PEFT text-layer tensor matched {SOURCE_PREFIX!r}; refusing a no-op adapter"
        )

    destination.mkdir(parents=True)
    destination_config = destination / source_config.name
    destination_weights = destination / source_weights.name
    destination_config.write_bytes(source_config.read_bytes())
    save_file(destination_tensors, str(destination_weights), metadata={"format": "pt"})

    manifest = {
        "schema": SCHEMA,
        "model": model,
        "adapter_version": adapter_version,
        "source_adapter": {
            "relative_path": _relative_adapter_path(root=destination, target=source),
            "adapter_config_sha256": file_sha256(source_config),
            "adapter_model_sha256": file_sha256(source_weights),
        },
        "destination_adapter": {
            "relative_path": ".",
            "adapter_config_sha256": file_sha256(destination_config),
            "adapter_model_sha256": file_sha256(destination_weights),
        },
        "translation": {
            "source_prefix": SOURCE_PREFIX,
            "destination_prefix": DESTINATION_PREFIX,
            "tensor_count": len(source_tensors),
            "translated_tensor_count": changed,
            "source_tensor_keyset_sha256": _json_sha256(sorted(source_tensors)),
            "destination_tensor_keyset_sha256": _json_sha256(sorted(destination_tensors)),
            "tensor_content_sha256": _tensor_content_sha256(source_tensors, key_transform=lambda key: key),
            "lora_b": _lora_b_stats(source_tensors),
        },
    }
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Prove the byte-preserving key bijection before the caller publishes this
    # version to any worker.  This is deliberately a full strict check once;
    # workers later use the hash-bound fast path.
    validate_qwen35_vllm_rollout_compat_adapter(
        destination, expected_version=adapter_version, strict_tensor_content=True
    )
    return manifest


def qwen35_rollout_adapter_has_nonzero_lora_effect(adapter_dir: str | Path) -> bool:
    """Return whether a verified snapshot's LoRA-B factor is nonzero.

    The fresh PEFT adapter has a zero B factor and is exactly the base policy;
    a fixed-token effect probe is not meaningful until at least one optimizer
    update has made this value nonzero.
    """

    manifest = validate_qwen35_vllm_rollout_compat_adapter(adapter_dir)
    summary = manifest["translation"]["lora_b"]
    return bool(summary["lora_b_nonzero_elements"] and summary["lora_b_max_abs"] > 0)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json_value(value: Any, *, label: str) -> Any:
    """Return a JSON-only canonical value, rejecting NaN and exotic objects."""

    try:
        encoded = json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Qwen3.5 rollout worker parity {label} is not canonical JSON") from exc


def _normalized_worker_gpus(workers: Sequence[Any], *, label: str) -> list[dict[str, Any]]:
    """Normalize the narrow GPU identity contract without importing workers."""

    normalized: list[dict[str, Any]] = []
    for index, worker in enumerate(workers):
        value = worker.as_dict() if callable(getattr(worker, "as_dict", None)) else worker
        if not isinstance(value, Mapping) or set(value) != {"logical_index", "device_token"}:
            raise ValueError(f"Qwen3.5 rollout worker parity {label}[{index}] must name logical_index and device_token")
        logical_index, device_token = value["logical_index"], value["device_token"]
        if isinstance(logical_index, bool) or not isinstance(logical_index, int) or logical_index < 0:
            raise ValueError(f"Qwen3.5 rollout worker parity {label}[{index}].logical_index is invalid")
        if not isinstance(device_token, str) or not device_token:
            raise ValueError(f"Qwen3.5 rollout worker parity {label}[{index}].device_token is invalid")
        normalized.append({"logical_index": logical_index, "device_token": device_token})
    if not normalized:
        raise ValueError(f"Qwen3.5 rollout worker parity {label} must not be empty")
    if len({row["logical_index"] for row in normalized}) != len(normalized):
        raise ValueError(f"Qwen3.5 rollout worker parity {label} has duplicate logical GPU indices")
    if len({row["device_token"] for row in normalized}) != len(normalized):
        raise ValueError(f"Qwen3.5 rollout worker parity {label} has duplicate device tokens")
    return normalized


def _adapter_identity(path: str | Path, *, label: str) -> dict[str, str]:
    root = Path(path).resolve()
    config, weights = _require_adapter_directory(root, label=label)
    return {
        "path": str(root),
        "adapter_config_sha256": file_sha256(config),
        "adapter_model_sha256": file_sha256(weights),
    }


def _require_passing_effect_summary(summary: Any, *, label: str) -> None:
    """Reject a no-op or a worker/HF effect disagreement deterministically."""

    if not isinstance(summary, Mapping):
        raise ValueError(f"Qwen3.5 rollout worker parity {label} has no effect summary")
    cosine = summary.get("cosine_similarity")
    if isinstance(cosine, bool) or not isinstance(cosine, (int, float)) or not math.isfinite(float(cosine)):
        raise ValueError(f"Qwen3.5 rollout worker parity {label} has an invalid cosine similarity")
    for side in ("worker_v2_minus_base", "coordinator_updated_minus_base"):
        metrics = summary.get(side)
        value = metrics.get("max_abs_difference") if isinstance(metrics, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Qwen3.5 rollout worker parity {label}.{side} has an invalid maximum effect")
        if float(value) < WORKER_PARITY_MIN_EFFECT:
            raise ValueError(
                f"Qwen3.5 rollout worker parity {label}.{side} has no meaningful nonzero LoRA effect: "
                f"max_abs={float(value):.3e}, required>={WORKER_PARITY_MIN_EFFECT:.3e}"
            )
    if float(cosine) < WORKER_PARITY_MIN_COSINE:
        raise ValueError(
            f"Qwen3.5 rollout worker parity {label} disagrees with HF/PEFT: "
            f"cosine_similarity={float(cosine):.6f}, required>={WORKER_PARITY_MIN_COSINE:.2f}"
        )


def _fixed_token_probe(value: Any) -> dict[str, Any]:
    """Validate the compact provenance record for the teacher-forced probe."""

    if not isinstance(value, Mapping):
        raise ValueError("Qwen3.5 rollout worker parity has no fixed_token_probe object")
    required = {"kind", "score_rows", "completion_token_count", "score_inputs_sha256"}
    if set(value) != required:
        raise ValueError("Qwen3.5 rollout worker parity fixed_token_probe has unexpected fields")
    kind, score_rows, completion_token_count, digest = (
        value["kind"],
        value["score_rows"],
        value["completion_token_count"],
        value["score_inputs_sha256"],
    )
    if kind != "post-update-worker-score-completions-v1":
        raise ValueError(f"Qwen3.5 rollout worker parity has unsupported fixed-token probe {kind!r}")
    for field, item in (("score_rows", score_rows), ("completion_token_count", completion_token_count)):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"Qwen3.5 rollout worker parity fixed_token_probe.{field} is invalid")
    if completion_token_count < score_rows:
        raise ValueError("Qwen3.5 rollout worker parity has fewer fixed probe tokens than rows")
    if not _is_sha256(digest):
        raise ValueError("Qwen3.5 rollout worker parity fixed_token_probe.score_inputs_sha256 is invalid")
    return {
        "kind": kind,
        "score_rows": score_rows,
        "completion_token_count": completion_token_count,
        "score_inputs_sha256": digest,
    }


def build_qwen35_rollout_worker_parity_attestation(
    *,
    model: str,
    raw_adapter: str | Path,
    vllm_adapter: str | Path,
    adapter_version: int,
    worker_gpus: Sequence[Any],
    worker_engine_kwargs: Mapping[str, Any],
    fixed_token_probe: Mapping[str, Any],
    aggregate_effect_parity: Mapping[str, Any],
    per_worker_effect_parity: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a portable, fail-closed attestation for the worker transport.

    The source and translated LoRA bytes are revalidated here.  The production
    backend later revalidates this document before it starts a new Qwen3.5
    rollout pool, then records this document's digest in every dynamic policy
    snapshot.  The actual per-snapshot tensors remain bound by the existing
    raw/translated compatibility manifest.
    """

    if not is_qwen35_model_name(model):
        raise ValueError(f"Qwen3.5 rollout worker parity requested for non-Qwen3.5 model {model!r}")
    if isinstance(adapter_version, bool) or not isinstance(adapter_version, int) or adapter_version < 1:
        raise ValueError("Qwen3.5 rollout worker parity adapter_version must be positive")
    raw = Path(raw_adapter).resolve()
    compat = Path(vllm_adapter).resolve()
    manifest = validate_qwen35_vllm_rollout_compat_adapter(
        compat,
        expected_version=adapter_version,
        strict_tensor_content=True,
    )
    if manifest["model"] != model:
        raise ValueError(
            "Qwen3.5 rollout worker parity model does not match its translated compatibility snapshot: "
            f"attestation={model!r}, snapshot={manifest['model']!r}"
        )
    if raw != compat.parent / "raw":
        raise ValueError("Qwen3.5 rollout worker parity raw adapter is not the translated snapshot's raw sibling")
    raw_identity = _adapter_identity(raw, label="raw parity adapter")
    compat_identity = _adapter_identity(compat, label="vLLM parity adapter")
    if raw_identity["adapter_model_sha256"] != manifest["source_adapter"]["adapter_model_sha256"]:
        raise ValueError("Qwen3.5 rollout worker parity raw adapter hash differs from compatibility manifest")
    if compat_identity["adapter_model_sha256"] != manifest["destination_adapter"]["adapter_model_sha256"]:
        raise ValueError("Qwen3.5 rollout worker parity vLLM adapter hash differs from compatibility manifest")
    if not qwen35_rollout_adapter_has_nonzero_lora_effect(compat):
        raise ValueError("Qwen3.5 rollout worker parity requires a meaningful nonzero LoRA adapter, not v1/base")

    normalized_gpus = _normalized_worker_gpus(worker_gpus, label="worker_gpus")
    engine_kwargs = _canonical_json_value(dict(worker_engine_kwargs), label="worker_engine_kwargs")
    if not isinstance(engine_kwargs, dict):  # defensive; dict() above should make this unreachable
        raise AssertionError("canonical worker engine kwargs must remain an object")
    probe = _fixed_token_probe(fixed_token_probe)
    _require_passing_effect_summary(aggregate_effect_parity, label="aggregate_effect_parity")

    normalized_workers: list[dict[str, Any]] = []
    if len(per_worker_effect_parity) != len(normalized_gpus):
        raise ValueError(
            "Qwen3.5 rollout worker parity must cover every worker exactly once: "
            f"attestation={len(per_worker_effect_parity)}, workers={len(normalized_gpus)}"
        )
    expected_by_index = {index: gpu for index, gpu in enumerate(normalized_gpus)}
    for entry in per_worker_effect_parity:
        if not isinstance(entry, Mapping):
            raise ValueError("Qwen3.5 rollout worker parity per-worker entry is not an object")
        worker_index = entry.get("worker_index")
        worker_gpu = entry.get("worker_gpu")
        if isinstance(worker_index, bool) or not isinstance(worker_index, int) or worker_index not in expected_by_index:
            raise ValueError("Qwen3.5 rollout worker parity per-worker entry has an invalid worker_index")
        if worker_gpu != expected_by_index[worker_index]:
            raise ValueError("Qwen3.5 rollout worker parity per-worker GPU identity does not match worker_index")
        _require_passing_effect_summary(entry.get("effect_parity"), label=f"worker[{worker_index}].effect_parity")
        normalized_workers.append(
            {
                "worker_index": worker_index,
                "worker_gpu": dict(worker_gpu),
                "effect_parity": _canonical_json_value(entry["effect_parity"], label="per_worker_effect_parity"),
            }
        )
    normalized_workers.sort(key=lambda entry: entry["worker_index"])
    if [entry["worker_index"] for entry in normalized_workers] != list(range(len(normalized_gpus))):
        raise ValueError("Qwen3.5 rollout worker parity has duplicate or missing per-worker evidence")

    compatibility_manifest = compat / MANIFEST_NAME
    return {
        "schema": WORKER_PARITY_ATTESTATION_SCHEMA,
        "model": model,
        "adapter_version": adapter_version,
        "raw_adapter": raw_identity,
        "vllm_adapter": {
            **compat_identity,
            "compatibility_manifest_sha256": file_sha256(compatibility_manifest),
        },
        "worker_gpus": normalized_gpus,
        "worker_engine_kwargs": engine_kwargs,
        "fixed_token_probe": probe,
        "aggregate_effect_parity": _canonical_json_value(aggregate_effect_parity, label="aggregate_effect_parity"),
        "per_worker_effect_parity": normalized_workers,
    }


def write_qwen35_rollout_worker_parity_attestation(
    destination: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Write one immutable worker-parity attestation without overwriting evidence."""

    attestation = build_qwen35_rollout_worker_parity_attestation(**kwargs)
    path = Path(destination)
    if path.name != WORKER_PARITY_ATTESTATION_NAME:
        raise ValueError(
            "Qwen3.5 rollout worker parity attestation must use " f"{WORKER_PARITY_ATTESTATION_NAME!r}: {path}"
        )
    serialized = json.dumps(attestation, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"refusing to overwrite different Qwen3.5 rollout worker parity evidence: {path}")
    return attestation


def validate_qwen35_rollout_worker_parity_attestation(
    path: str | Path,
    *,
    expected_model: str | None = None,
    expected_worker_gpus: Sequence[Any] | None = None,
    expected_worker_engine_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the immutable nonzero fixed-token worker parity evidence.

    This intentionally rebuilds the canonical document from current files. A
    stale report, an altered raw/translated adapter, a missing worker, or a
    different vLLM worker configuration is therefore rejected before policy
    sampling can begin.
    """

    source = Path(path).resolve()
    if source.name != WORKER_PARITY_ATTESTATION_NAME or not source.is_file():
        raise FileNotFoundError(f"Qwen3.5 rollout worker parity attestation is missing: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Qwen3.5 rollout worker parity attestation: {source}") from exc
    if not isinstance(document, Mapping) or document.get("schema") != WORKER_PARITY_ATTESTATION_SCHEMA:
        raise ValueError("unsupported Qwen3.5 rollout worker parity attestation schema")
    if expected_model is not None and document.get("model") != expected_model:
        raise ValueError(
            "Qwen3.5 rollout worker parity attestation model mismatch: "
            f"attestation={document.get('model')!r}, expected={expected_model!r}"
        )
    raw = document.get("raw_adapter")
    compat = document.get("vllm_adapter")
    if not isinstance(raw, Mapping) or not isinstance(compat, Mapping):
        raise ValueError("Qwen3.5 rollout worker parity attestation has no adapter identities")
    rebuilt = build_qwen35_rollout_worker_parity_attestation(
        model=document.get("model"),
        raw_adapter=raw.get("path"),
        vllm_adapter=compat.get("path"),
        adapter_version=document.get("adapter_version"),
        worker_gpus=document.get("worker_gpus"),
        worker_engine_kwargs=document.get("worker_engine_kwargs"),
        fixed_token_probe=document.get("fixed_token_probe"),
        aggregate_effect_parity=document.get("aggregate_effect_parity"),
        per_worker_effect_parity=document.get("per_worker_effect_parity"),
    )
    if dict(document) != rebuilt:
        raise ValueError("Qwen3.5 rollout worker parity attestation does not match its hash-bound source files")
    if expected_worker_gpus is not None and rebuilt["worker_gpus"] != _normalized_worker_gpus(
        expected_worker_gpus,
        label="expected_worker_gpus",
    ):
        raise ValueError("Qwen3.5 rollout worker parity attestation was produced for different rollout GPUs")
    if expected_worker_engine_kwargs is not None and rebuilt["worker_engine_kwargs"] != _canonical_json_value(
        dict(expected_worker_engine_kwargs),
        label="expected_worker_engine_kwargs",
    ):
        raise ValueError("Qwen3.5 rollout worker parity attestation was produced for different vLLM worker options")
    return rebuilt


__all__ = [
    "DESTINATION_PREFIX",
    "MANIFEST_NAME",
    "SCHEMA",
    "SOURCE_PREFIX",
    "WORKER_PARITY_ATTESTATION_NAME",
    "WORKER_PARITY_ATTESTATION_SCHEMA",
    "WORKER_PARITY_MIN_COSINE",
    "WORKER_PARITY_MIN_EFFECT",
    "build_qwen35_rollout_worker_parity_attestation",
    "file_sha256",
    "is_qwen35_model_name",
    "materialize_qwen35_vllm_rollout_compat_adapter",
    "qwen35_rollout_adapter_has_nonzero_lora_effect",
    "translate_lora_key",
    "validate_qwen35_rollout_worker_parity_attestation",
    "validate_qwen35_vllm_rollout_compat_adapter",
    "write_qwen35_rollout_worker_parity_attestation",
]
