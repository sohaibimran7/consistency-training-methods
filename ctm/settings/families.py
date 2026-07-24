"""Shared frozen prompt-family artifacts for item-specific perturbations.

Some settings do not have one global prompt template. Each source item carries its own reference prompt and a
family of matched variants.  This module is the small, setting-agnostic seam
that lets the existing RL trainer consume those families without learning any
dataset-specific schemas.

Frozen JSONL schema (version 1)::

    {
      "schema_version": 1,
      "source_id": "stable item id",
      "source": "dataset name",
      "reference_messages": [{"role": "user", "content": "..."}],
      "variants": [
        {
          "variant_id": "stable variant id",
          "messages": [{"role": "user", "content": "..."}],
          "axes": {"tactics": ["..."]}
        }
      ],
      "metadata": {"source_split": "train"}
    }

The materializers deliberately choose a fixed number of variants per family.
That is a compatibility layer for ``RLTrainer``'s global perturbation list; it
also makes sampling cost and manifests stable across reruns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ctm.artifacts import artifact_manifest_path, read_verified_artifact_manifest

FAMILY_SCHEMA_VERSION = 1


class FamilyValidationError(ValueError):
    """A frozen prompt-family row does not satisfy the versioned contract."""


def stable_digest(value: str, *, seed: str = "42") -> str:
    """A process-independent digest used for partitions and deterministic order."""

    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _validate_messages(messages: Any, *, field: str) -> None:
    if not isinstance(messages, list) or not messages:
        raise FamilyValidationError(f"{field} must be a non-empty message list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise FamilyValidationError(f"{field}[{index}] must be an object")
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            raise FamilyValidationError(f"{field}[{index}].role must be a non-empty string")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise FamilyValidationError(f"{field}[{index}].content must be a non-empty string")


def validate_family(row: Mapping[str, Any], *, min_variants: int = 1) -> dict[str, Any]:
    """Validate and shallow-copy one version-1 family row.

    Validation happens at the setting boundary so malformed or mixed-schema
    artifacts fail before a costly training run starts.
    """

    if row.get("schema_version") != FAMILY_SCHEMA_VERSION:
        raise FamilyValidationError(
            f"unsupported schema_version {row.get('schema_version')!r}; expected {FAMILY_SCHEMA_VERSION}"
        )
    source_id = row.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise FamilyValidationError("source_id must be a non-empty string")
    if not isinstance(row.get("source"), str) or not row["source"]:
        raise FamilyValidationError(f"family {source_id!r}: source must be a non-empty string")
    _validate_messages(row.get("reference_messages"), field=f"family {source_id!r}.reference_messages")

    variants = row.get("variants")
    if not isinstance(variants, list) or len(variants) < min_variants:
        raise FamilyValidationError(
            f"family {source_id!r} has {len(variants) if isinstance(variants, list) else 0} variants; "
            f"needs at least {min_variants}"
        )
    seen: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise FamilyValidationError(f"family {source_id!r}.variants[{index}] must be an object")
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id:
            raise FamilyValidationError(f"family {source_id!r}.variants[{index}].variant_id is required")
        if variant_id in seen:
            raise FamilyValidationError(f"family {source_id!r} repeats variant_id {variant_id!r}")
        seen.add(variant_id)
        _validate_messages(variant.get("messages"), field=f"family {source_id!r}.variants[{index}].messages")
        if "axes" in variant and not isinstance(variant["axes"], dict):
            raise FamilyValidationError(f"family {source_id!r}.variants[{index}].axes must be an object")
    if "metadata" in row and not isinstance(row["metadata"], dict):
        raise FamilyValidationError(f"family {source_id!r}.metadata must be an object")
    return dict(row)


def load_family_artifact(
    path: str | Path,
    *,
    n_datapoints: int | None = None,
    n_variants: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a family artifact once and return its rows and manifest."""

    if n_datapoints is not None and n_datapoints < 0:
        raise ValueError("n_datapoints must be >= 0")
    manifest = read_verified_artifact_manifest(
        path,
        expected_schema="ctm.prompt_families",
        expected_schema_version=FAMILY_SCHEMA_VERSION,
    )
    available = manifest["row_count"]
    if n_datapoints is not None and available < n_datapoints:
        raise ValueError(f"{path} contains only {available}/{n_datapoints} requested families")
    if n_datapoints == 0:
        return [], manifest
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                row = validate_family(raw, min_variants=n_variants)
            except (json.JSONDecodeError, FamilyValidationError) as exc:
                raise FamilyValidationError(f"{path}:{line_number}: {exc}") from exc
            # The fixed-K trainer contract intentionally ignores any surplus
            # variants while preserving their source order in the artifact.
            row["variants"] = row["variants"][:n_variants]
            rows.append(row)
            if n_datapoints is not None and len(rows) >= n_datapoints:
                break
    if n_datapoints is not None and len(rows) != n_datapoints:
        raise ValueError(f"{path} yielded only {len(rows)}/{n_datapoints} requested families")
    return rows, manifest


def load_families(
    path: str | Path,
    *,
    n_datapoints: int | None = None,
    n_variants: int = 1,
) -> list[dict[str, Any]]:
    """Load strict family JSONL, optionally taking a deterministic prefix."""

    rows, _ = load_family_artifact(path, n_datapoints=n_datapoints, n_variants=n_variants)
    return rows


def select_fixed_variants(
    variants: Sequence[Mapping[str, Any]],
    *,
    source_id: str,
    n_variants: int,
    seed: str = "42",
) -> list[dict[str, Any]]:
    """Choose a stable fixed-K subset using variant identity, never input order."""

    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    unique: dict[str, Mapping[str, Any]] = {}
    for variant in variants:
        variant_id = variant.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id:
            raise FamilyValidationError(f"family {source_id!r}: every candidate needs variant_id")
        if variant_id in unique and dict(unique[variant_id]) != dict(variant):
            raise FamilyValidationError(f"family {source_id!r}: conflicting duplicate variant {variant_id!r}")
        unique[variant_id] = variant
    if len(unique) < n_variants:
        raise FamilyValidationError(f"family {source_id!r} has {len(unique)} unique variants; needs {n_variants}")
    ordered = sorted(unique.values(), key=lambda item: stable_digest(f"{source_id}\0{item['variant_id']}", seed=seed))
    return [dict(item) for item in ordered[:n_variants]]


def make_family_perturbations(
    n_variants: int,
    *,
    control: bool = False,
) -> list[Callable[[dict[str, Any]], dict[str, Any]]]:
    """Build ``[reference, variant 0, ...]`` functions for ``RLTrainer``."""

    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")

    def reference(datapoint: dict[str, Any]) -> dict[str, Any]:
        return {"messages": datapoint["reference_messages"]}

    def make_variant(index: int) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def variant(datapoint: dict[str, Any]) -> dict[str, Any]:
            if control:
                return {"messages": datapoint["reference_messages"]}
            return {"messages": datapoint["variants"][index]["messages"]}

        return variant

    return [reference] + [make_variant(index) for index in range(n_variants)]


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def write_frozen_artifact(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an immutable JSONL artifact plus an identity-bearing manifest.

    Existing paths are never overwritten.  Re-running a materializer with
    incompatible source revisions or split specs therefore fails loudly rather
    than silently changing the meaning of a training path.
    """

    target = Path(path)
    manifest_path = artifact_manifest_path(target)
    if target.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact or manifest: {target}")

    validated = [validate_family(row) for row in rows]
    payload = _jsonl_bytes(validated)
    source_ids = sorted(row["source_id"] for row in validated)
    manifest = {
        "artifact_schema": "ctm.prompt_families",
        "schema_version": FAMILY_SCHEMA_VERSION,
        "row_count": len(validated),
        "source_ids_sha256": hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "provenance": dict(provenance),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
