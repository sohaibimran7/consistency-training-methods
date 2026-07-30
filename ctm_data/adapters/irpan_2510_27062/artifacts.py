"""Immutable, verified JSONL artifacts for the paper reproduction pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctm.artifacts import (
    ArtifactManifestError,
    artifact_manifest_path,
    read_verified_artifact_manifest,
    write_atomic_bytes,
)
from ctm_data.adapters.irpan_2510_27062.partitions import PartitionError, require_artifact_role
from ctm_data.adapters.irpan_2510_27062.schema import (
    ARTIFACT_SCHEMA,
    PAPER_ID,
    SCHEMA_VERSION,
    canonical_json,
    require_sha256,
    sha256_bytes,
    sha256_json,
    validate_record,
)

MANIFEST_SHA256_FIELD = "manifest_sha256"


def producer_identity(name: str, *paths: str | Path) -> dict[str, Any]:
    """Hash the exact local implementation files used to build an artifact."""

    if not isinstance(name, str) or not name.strip():
        raise ArtifactManifestError("producer name must be non-empty")
    if not paths:
        raise ArtifactManifestError("producer_identity requires at least one implementation file")
    files: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise ArtifactManifestError(f"producer implementation file does not exist: {path}")
        files.append({"name": path.name, "content_sha256": sha256_bytes(path.read_bytes())})
    return {"name": name.strip(), "files": files, "code_sha256": sha256_json(files)}


def write_artifact(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    artifact_kind: str,
    role: str,
    producer: Mapping[str, Any],
    config: Mapping[str, Any],
    parent_artifacts: Sequence[str | Path] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable JSONL/manifest pair and return the verified manifest."""

    target = Path(path)
    sidecar = artifact_manifest_path(target)
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact pair: {target} / {sidecar}")
    if not isinstance(artifact_kind, str) or not artifact_kind:
        raise ArtifactManifestError("artifact_kind must be non-empty")
    validated_role = _role(role, field="role")
    producer_copy = _validate_producer(producer)
    if not isinstance(config, Mapping):
        raise ArtifactManifestError("artifact config must be a mapping")

    validated_rows = [validate_record(row) for row in rows]
    payload = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in validated_rows)
    parent_refs = [_parent_identity(parent_path) for parent_path in parent_artifacts]
    extra_provenance = dict(provenance or {})
    reserved = {
        "paper_id",
        "artifact_kind",
        "role",
        "created_at_utc",
        "producer",
        "config",
        "config_sha256",
        "parent_artifacts",
    }
    overlap = sorted(reserved & set(extra_provenance))
    if overlap:
        raise ArtifactManifestError(f"custom provenance cannot replace reserved fields: {overlap}")
    manifest_without_digest = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "row_count": len(validated_rows),
        "content_sha256": sha256_bytes(payload),
        "provenance": {
            "paper_id": PAPER_ID,
            "artifact_kind": artifact_kind,
            "role": validated_role,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": producer_copy,
            "config": dict(config),
            "config_sha256": sha256_json(config),
            "parent_artifacts": parent_refs,
            **extra_provenance,
        },
    }
    manifest = {
        **manifest_without_digest,
        MANIFEST_SHA256_FIELD: sha256_json(manifest_without_digest),
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_atomic_bytes(target, payload)
    # If this fails, keep the completed artifact visible so callers can recover
    # or archive it; later reads fail closed because the sidecar is absent.
    write_atomic_bytes(sidecar, manifest_payload)
    return _read_adapter_manifest(target, expected_role=validated_role)


def read_artifact(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    expected_role: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify an artifact pair, then validate every versioned row."""

    target = Path(path)
    manifest = _read_adapter_manifest(target, expected_role=expected_role)
    kind = manifest["provenance"].get("artifact_kind")
    if not isinstance(kind, str) or not kind:
        raise ArtifactManifestError(f"artifact manifest for {target} has no artifact_kind")
    if expected_kind is not None and kind != expected_kind:
        raise ArtifactManifestError(f"artifact kind is {kind!r}, expected {expected_kind!r}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactManifestError(f"invalid JSON in {target} line {line_number}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ArtifactManifestError(f"{target} line {line_number} must be a JSON object")
        rows.append(validate_record(decoded))
    return rows, manifest


def require_local_artifact(path: str | Path, *, source_key: str, acquisition_url: str) -> Path:
    """Fail with an actionable, network-free acquisition message."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"missing local export for {source_key!r}: {target}. Acquire it under the upstream terms at "
            f"{acquisition_url}, pin a revision or record its SHA-256, then pass the local path. "
            "Adapters never download datasets during import, task construction, or dry-run."
        )
    return target


def _validate_producer(producer: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(producer, Mapping):
        raise ArtifactManifestError("producer must be a mapping")
    copy = dict(producer)
    if not isinstance(copy.get("name"), str) or not copy["name"]:
        raise ArtifactManifestError("producer has no name")
    require_sha256(copy.get("code_sha256"), field="producer.code_sha256")
    return copy


def _parent_identity(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    manifest = _read_adapter_manifest(target)
    return {
        "path": str(target),
        "artifact_kind": manifest["provenance"].get("artifact_kind"),
        "role": manifest["provenance"]["role"],
        "artifact_schema": manifest["artifact_schema"],
        "schema_version": manifest["schema_version"],
        "row_count": manifest["row_count"],
        "content_sha256": manifest["content_sha256"],
        MANIFEST_SHA256_FIELD: manifest[MANIFEST_SHA256_FIELD],
    }


def _read_adapter_manifest(target: Path, *, expected_role: str | None = None) -> dict[str, Any]:
    manifest = read_verified_artifact_manifest(
        target,
        expected_schema=ARTIFACT_SCHEMA,
        expected_schema_version=SCHEMA_VERSION,
    )
    provenance = manifest["provenance"]
    if "role" not in provenance:
        raise ArtifactManifestError(f"artifact manifest for {target} has no provenance.role")
    actual_role = _role(provenance["role"], field=f"manifest provenance.role for {target}")
    if expected_role is not None:
        required_role = _role(expected_role, field="expected_role")
        if actual_role != required_role:
            raise ArtifactManifestError(f"artifact role is {actual_role!r}, expected {required_role!r}")

    try:
        recorded_digest = require_sha256(
            manifest.get(MANIFEST_SHA256_FIELD),
            field=f"artifact manifest {MANIFEST_SHA256_FIELD}",
        )
    except ValueError as exc:
        raise ArtifactManifestError(str(exc)) from exc
    unsigned_manifest = {key: value for key, value in manifest.items() if key != MANIFEST_SHA256_FIELD}
    actual_digest = sha256_json(unsigned_manifest)
    if recorded_digest != actual_digest:
        raise ArtifactManifestError(
            f"artifact manifest integrity digest mismatch for {target}: "
            f"manifest has {recorded_digest}, fields hash to {actual_digest}"
        )
    return manifest


def _role(value: Any, *, field: str) -> str:
    try:
        return require_artifact_role(value)
    except PartitionError as exc:
        raise ArtifactManifestError(f"{field}: {exc}") from exc


__all__ = [
    "MANIFEST_SHA256_FIELD",
    "producer_identity",
    "read_artifact",
    "require_local_artifact",
    "write_artifact",
]
