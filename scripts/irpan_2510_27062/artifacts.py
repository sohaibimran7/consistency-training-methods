"""Paper-policy wrapper over CTM's shared immutable artifact protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ctm.artifacts import (
    MANIFEST_SHA256_FIELD,
    ArtifactManifestError,
    parent_artifact_identity,
    producer_identity,
    read_verified_artifact_manifest,
    read_verified_jsonl_artifact,
    write_verified_jsonl_artifact,
)
from scripts.irpan_2510_27062.partitions import PartitionError, require_artifact_role
from scripts.irpan_2510_27062.schema import (
    ARTIFACT_SCHEMA,
    PAPER_ID,
    SCHEMA_VERSION,
    require_sha256,
    sha256_json,
    validate_record,
)


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
    """Publish paper records using the repository-wide JSONL protocol."""

    if not isinstance(artifact_kind, str) or not artifact_kind:
        raise ArtifactManifestError("artifact_kind must be non-empty")
    validated_role = _role(role, field="role")
    producer_copy = _validate_producer(producer)
    if not isinstance(config, Mapping):
        raise ArtifactManifestError("artifact config must be a mapping")
    extra = dict(provenance or {})
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
    overlap = sorted(reserved & set(extra))
    if overlap:
        raise ArtifactManifestError(f"custom provenance cannot replace reserved fields: {overlap}")
    return write_verified_jsonl_artifact(
        path,
        rows,
        artifact_schema=ARTIFACT_SCHEMA,
        schema_version=SCHEMA_VERSION,
        row_validator=validate_record,
        provenance={
            "paper_id": PAPER_ID,
            "artifact_kind": artifact_kind,
            "role": validated_role,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "producer": producer_copy,
            "config": dict(config),
            "config_sha256": sha256_json(config),
            "parent_artifacts": [_parent_identity(parent) for parent in parent_artifacts],
            **extra,
        },
    )


def read_artifact(
    path: str | Path,
    *,
    expected_kind: str | None = None,
    expected_role: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify a shared artifact, then enforce the paper's kind and role policy."""

    target = Path(path)
    # Check the requested role before self-hash verification so a mismatched
    # partition gets the most actionable error; the subsequent shared reader
    # still verifies the entire manifest before returning any rows.
    preliminary = read_verified_artifact_manifest(
        target,
        expected_schema=ARTIFACT_SCHEMA,
        expected_schema_version=SCHEMA_VERSION,
    )
    _validate_paper_provenance(
        preliminary,
        target=target,
        expected_kind=expected_kind,
        expected_role=expected_role,
    )
    rows, manifest = read_verified_jsonl_artifact(
        target,
        expected_schema=ARTIFACT_SCHEMA,
        expected_schema_version=SCHEMA_VERSION,
        row_validator=validate_record,
    )
    _validate_paper_provenance(
        manifest,
        target=target,
        expected_kind=expected_kind,
        expected_role=expected_role,
    )
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


def _validate_paper_provenance(
    manifest: Mapping[str, Any],
    *,
    target: Path,
    expected_kind: str | None,
    expected_role: str | None,
) -> None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ArtifactManifestError(f"artifact manifest for {target} has no provenance object")
    if provenance.get("paper_id") != PAPER_ID:
        raise ArtifactManifestError(f"artifact paper_id is {provenance.get('paper_id')!r}, expected {PAPER_ID!r}")
    kind = provenance.get("artifact_kind")
    if not isinstance(kind, str) or not kind:
        raise ArtifactManifestError(f"artifact manifest for {target} has no artifact_kind")
    if expected_kind is not None and kind != expected_kind:
        raise ArtifactManifestError(f"artifact kind is {kind!r}, expected {expected_kind!r}")
    if "role" not in provenance:
        raise ArtifactManifestError(f"artifact manifest for {target} has no provenance.role")
    actual_role = _role(provenance["role"], field=f"manifest provenance.role for {target}")
    if expected_role is not None:
        required_role = _role(expected_role, field="expected_role")
        if actual_role != required_role:
            raise ArtifactManifestError(f"artifact role is {actual_role!r}, expected {required_role!r}")


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
    _rows, manifest = read_artifact(target)
    return {
        **parent_artifact_identity(target),
        "artifact_kind": manifest["provenance"]["artifact_kind"],
        "role": manifest["provenance"]["role"],
    }


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
