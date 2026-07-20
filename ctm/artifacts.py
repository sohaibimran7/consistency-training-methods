"""Shared verification for immutable JSONL artifact/manifest pairs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from collections.abc import Sequence
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ArtifactManifestError(ValueError):
    """An artifact sidecar is missing, malformed, or does not match its JSONL."""


def artifact_manifest_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".manifest.json")


def read_verified_artifact_manifest(
    path: str | Path,
    *,
    expected_schema: str | None = None,
    expected_schema_version: int | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read a sidecar and verify its schema, row count, and content digest."""

    target = Path(path)
    sidecar = Path(manifest_path) if manifest_path is not None else artifact_manifest_path(target)
    if not target.is_file():
        raise ArtifactManifestError(f"missing frozen artifact: {target}")
    if not sidecar.is_file():
        raise ArtifactManifestError(f"missing immutable artifact manifest: {sidecar}")
    try:
        manifest = json.loads(sidecar.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactManifestError(f"invalid artifact manifest {sidecar}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactManifestError(f"artifact manifest {sidecar} must contain a JSON object")

    schema = manifest.get("artifact_schema")
    version = manifest.get("schema_version")
    row_count = manifest.get("row_count")
    digest = manifest.get("content_sha256")
    provenance = manifest.get("provenance")
    if not isinstance(schema, str) or not schema:
        raise ArtifactManifestError(f"artifact manifest {sidecar} has no artifact_schema")
    if not isinstance(version, int) or isinstance(version, bool):
        raise ArtifactManifestError(f"artifact manifest {sidecar} has an invalid schema_version")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        raise ArtifactManifestError(f"artifact manifest {sidecar} has an invalid row_count")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ArtifactManifestError(f"artifact manifest {sidecar} has an invalid content_sha256")
    if not isinstance(provenance, dict):
        raise ArtifactManifestError(f"artifact manifest {sidecar} has no provenance object")
    if expected_schema is not None and schema != expected_schema:
        raise ArtifactManifestError(f"artifact manifest {sidecar} has schema {schema!r}, expected {expected_schema!r}")
    if expected_schema_version is not None and version != expected_schema_version:
        raise ArtifactManifestError(
            f"artifact manifest {sidecar} has schema_version {version!r}, expected {expected_schema_version}"
        )

    payload = target.read_bytes()
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != digest:
        raise ArtifactManifestError(
            f"frozen artifact digest mismatch for {target}: manifest has {digest}, bytes hash to {actual_digest}"
        )
    actual_rows = sum(1 for line in payload.splitlines() if line.strip())
    if actual_rows != row_count:
        raise ArtifactManifestError(
            f"frozen artifact row-count mismatch for {target}: manifest has {row_count}, file has {actual_rows}"
        )
    return manifest


def artifact_identity(path: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe identity suitable for training/eval run metadata."""

    return {
        "path": str(Path(path)),
        "artifact_schema": manifest["artifact_schema"],
        "schema_version": manifest["schema_version"],
        "row_count": manifest["row_count"],
        "content_sha256": manifest["content_sha256"],
        "provenance": manifest["provenance"],
    }


def plain_file_identity(path: str | Path) -> dict[str, Any]:
    """Record the path, digest, and non-empty line count of a plain file."""

    target = Path(path)
    payload = target.read_bytes()
    return {
        "path": str(target.resolve()),
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": sum(1 for line in payload.splitlines() if line.strip()),
    }


def write_atomic_bytes(path: str | Path, payload: bytes) -> None:
    """Publish bytes by replacing the destination with a completed temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


def artifact_selection_identity(source_ids: Sequence[str], *, n_variants: int | None = None) -> dict[str, Any]:
    """Identity of the exact ordered family prefix consumed by one training run."""

    selected = list(source_ids)
    if any(not isinstance(source_id, str) or not source_id for source_id in selected):
        raise ArtifactManifestError("selected source IDs must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ArtifactManifestError("selected source IDs must be unique")
    payload = "\n".join(selected).encode()
    identity: dict[str, Any] = {
        "selected_row_count": len(selected),
        "selected_source_ids": selected,
        "selected_source_ids_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if n_variants is not None:
        if n_variants < 1:
            raise ArtifactManifestError("selection n_variants must be >= 1")
        identity["n_variants"] = n_variants
    return identity


__all__ = [
    "ArtifactManifestError",
    "artifact_identity",
    "artifact_manifest_path",
    "artifact_selection_identity",
    "plain_file_identity",
    "read_verified_artifact_manifest",
    "write_atomic_bytes",
]
