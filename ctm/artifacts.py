"""Shared verification for immutable JSONL artifact/manifest pairs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.identity import canonical_json, require_sha256, sha256_bytes, sha256_json

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_SHA256_FIELD = "manifest_sha256"

RowValidator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class ArtifactManifestError(ValueError):
    """An artifact sidecar is missing, malformed, or does not match its JSONL."""


def artifact_manifest_path(path: str | Path) -> Path:
    target = Path(path)
    return target.with_suffix(target.suffix + ".manifest.json")


def producer_identity(name: str, *paths: str | Path) -> dict[str, Any]:
    """Hash the exact implementation files used to produce an artifact."""

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
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
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


def verify_data_manifest_binding(
    data_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify that one explicit manifest commits to one exact data file.

    Shared JSONL artifacts use the canonical self-hashed sidecar contract.
    BCT targets retain their matched main/control manifest, so this verifier
    also resolves the requested file against exactly one declared output.
    Unknown manifest protocols fail closed.
    """

    target = Path(data_path)
    sidecar = Path(manifest_path)
    if not target.is_file():
        raise ArtifactManifestError(f"missing data file: {target}")
    if not sidecar.is_file():
        raise ArtifactManifestError(f"missing data manifest: {sidecar}")
    try:
        raw_manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactManifestError(f"invalid data manifest {sidecar}: {exc}") from exc
    if not isinstance(raw_manifest, dict):
        raise ArtifactManifestError(f"data manifest {sidecar} must contain a JSON object")

    if "artifact_schema" in raw_manifest:
        manifest = read_verified_artifact_manifest(target, manifest_path=sidecar)
        _verify_manifest_self_hash(manifest, target=target)
        return manifest
    if raw_manifest.get("kind") == "ctm_bct_targets":
        return _verify_bct_target_binding(target, sidecar, raw_manifest)
    raise ArtifactManifestError(f"data manifest {sidecar} uses no supported artifact protocol")


def verify_data_manifest_bindings(
    data_paths: Sequence[str | Path],
    manifest_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    """Verify one ordered, unambiguous manifest binding per data file."""

    if len(data_paths) != len(manifest_paths):
        raise ArtifactManifestError(
            "data files and data manifests must have the same count " f"({len(data_paths)} != {len(manifest_paths)})"
        )
    return [
        verify_data_manifest_binding(data_path, manifest_path)
        for data_path, manifest_path in zip(data_paths, manifest_paths, strict=True)
    ]


def parent_artifact_identity(path: str | Path) -> dict[str, Any]:
    """Return a compact identity that commits to an artifact's full manifest."""

    target = Path(path)
    manifest = _read_self_verified_manifest(target)
    return {
        "path": str(target),
        "artifact_schema": manifest["artifact_schema"],
        "schema_version": manifest["schema_version"],
        "row_count": manifest["row_count"],
        "content_sha256": manifest["content_sha256"],
        MANIFEST_SHA256_FIELD: manifest[MANIFEST_SHA256_FIELD],
    }


def write_verified_jsonl_artifact(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    artifact_schema: str,
    schema_version: int,
    provenance: Mapping[str, Any],
    row_validator: RowValidator | None = None,
    nonempty: bool = False,
) -> dict[str, Any]:
    """Publish an immutable JSONL/sidecar pair with a self-hashed manifest.

    Each destination is atomically replaced from a complete temporary file and
    the manifest is published last as the completion marker. Existing artifact
    or sidecar paths are never overwritten.
    """

    target = Path(path)
    sidecar = artifact_manifest_path(target)
    if target.exists() or sidecar.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact pair: {target} / {sidecar}")
    if not isinstance(artifact_schema, str) or not artifact_schema:
        raise ArtifactManifestError("artifact_schema must be non-empty")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
        raise ArtifactManifestError("schema_version must be a positive integer")
    if not isinstance(provenance, Mapping):
        raise ArtifactManifestError("provenance must be a mapping")
    if not isinstance(nonempty, bool):
        raise ArtifactManifestError("nonempty must be a boolean")
    if row_validator is not None and not callable(row_validator):
        raise ArtifactManifestError("row_validator must be callable")

    validated_rows = [
        _validate_jsonl_row(row, row_number=index, row_validator=row_validator)
        for index, row in enumerate(rows, start=1)
    ]
    if nonempty and not validated_rows:
        raise ArtifactManifestError("artifact must contain at least one row")

    encoded_rows: list[bytes] = []
    for index, row in enumerate(validated_rows, start=1):
        try:
            encoded_rows.append((canonical_json(row) + "\n").encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ArtifactManifestError(f"artifact row {index} is not canonical JSON: {exc}") from exc
    payload = b"".join(encoded_rows)
    provenance_copy = _json_copy(provenance, field="provenance")
    manifest_without_digest = {
        "artifact_schema": artifact_schema,
        "schema_version": schema_version,
        "row_count": len(validated_rows),
        "content_sha256": sha256_bytes(payload),
        "provenance": provenance_copy,
    }
    manifest = {
        **manifest_without_digest,
        MANIFEST_SHA256_FIELD: sha256_json(manifest_without_digest),
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    write_atomic_bytes(target, payload)
    # The sidecar is the completion marker. If this write fails, the payload is
    # deliberately left visible for explicit recovery instead of being erased.
    write_atomic_bytes(sidecar, manifest_payload)
    return _read_self_verified_manifest(
        target,
        expected_schema=artifact_schema,
        expected_schema_version=schema_version,
    )


def read_verified_jsonl_artifact(
    path: str | Path,
    *,
    expected_schema: str | None = None,
    expected_schema_version: int | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
    row_validator: RowValidator | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and decode a self-hashed immutable JSONL artifact.

    ``expected_provenance`` is matched as a recursive mapping subset: omitted
    keys are unconstrained, nested mappings recurse, and non-mapping values
    must match exactly.
    """

    target = Path(path)
    if row_validator is not None and not callable(row_validator):
        raise ArtifactManifestError("row_validator must be callable")
    manifest = _read_self_verified_manifest(
        target,
        expected_schema=expected_schema,
        expected_schema_version=expected_schema_version,
        manifest_path=manifest_path,
    )
    if expected_provenance is not None:
        if not isinstance(expected_provenance, Mapping):
            raise ArtifactManifestError("expected_provenance must be a mapping")
        expected_copy = _json_copy(expected_provenance, field="expected_provenance")
        _require_recursive_subset(
            manifest["provenance"],
            expected_copy,
            path="provenance",
        )

    payload = target.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.split(b"\n"), start=1):
        if not raw_line.strip():
            continue
        try:
            line = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactManifestError(f"invalid UTF-8 in {target} line {line_number}: {exc}") from exc
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ArtifactManifestError(f"invalid JSON in {target} line {line_number}: {exc}") from exc
        rows.append(_validate_jsonl_row(decoded, row_number=line_number, row_validator=row_validator))
    return rows, manifest


def _read_self_verified_manifest(
    path: str | Path,
    *,
    expected_schema: str | None = None,
    expected_schema_version: int | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    target = Path(path)
    manifest = read_verified_artifact_manifest(
        target,
        expected_schema=expected_schema,
        expected_schema_version=expected_schema_version,
        manifest_path=manifest_path,
    )
    _verify_manifest_self_hash(manifest, target=target)
    return manifest


def _verify_manifest_self_hash(manifest: Mapping[str, Any], *, target: Path) -> None:
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


def _verify_bct_target_binding(
    target: Path,
    sidecar: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ArtifactManifestError(f"BCT target manifest {sidecar} has unsupported schema_version")
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 1:
        raise ArtifactManifestError(f"BCT target manifest {sidecar} has an invalid row_count")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ArtifactManifestError(f"BCT target manifest {sidecar} has no outputs object")
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for name, identity in outputs.items():
        if not isinstance(name, str) or not isinstance(identity, Mapping):
            raise ArtifactManifestError(f"BCT target manifest {sidecar} has an invalid output identity")
        recorded_path = identity.get("path")
        if isinstance(recorded_path, str) and Path(recorded_path).resolve() == target.resolve():
            matches.append((name, identity))
    if len(matches) != 1:
        raise ArtifactManifestError(
            f"BCT target manifest {sidecar} binds {len(matches)} outputs to {target}; expected exactly one"
        )
    _name, identity = matches[0]
    try:
        recorded_digest = require_sha256(
            identity.get("content_sha256"),
            field=f"BCT target manifest {sidecar} output content_sha256",
        )
    except ValueError as exc:
        raise ArtifactManifestError(str(exc)) from exc
    payload = target.read_bytes()
    actual_digest = sha256_bytes(payload)
    if recorded_digest != actual_digest:
        raise ArtifactManifestError(
            f"BCT target digest mismatch for {target}: manifest has {recorded_digest}, "
            f"bytes hash to {actual_digest}"
        )
    actual_rows = sum(1 for line in payload.splitlines() if line.strip())
    if actual_rows != row_count:
        raise ArtifactManifestError(
            f"BCT target row-count mismatch for {target}: manifest has {row_count}, " f"file has {actual_rows}"
        )
    return dict(manifest)


def _validate_jsonl_row(
    value: Any,
    *,
    row_number: int,
    row_validator: RowValidator | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactManifestError(f"artifact row {row_number} must be a mapping")
    row: Mapping[str, Any] = dict(value)
    if row_validator is not None:
        try:
            row = row_validator(row)
        except Exception as exc:
            raise ArtifactManifestError(f"artifact row {row_number} failed validation: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ArtifactManifestError(f"artifact row {row_number} validator must return a mapping")
    return dict(row)


def _json_copy(value: Any, *, field: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ArtifactManifestError(f"{field} must contain canonical JSON values: {exc}") from exc


def _require_recursive_subset(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            raise ArtifactManifestError(f"{path} is {actual!r}, expected a mapping containing {dict(expected)!r}")
        for key, expected_value in expected.items():
            child_path = f"{path}.{key}"
            if key not in actual:
                raise ArtifactManifestError(f"{child_path} is missing from artifact provenance")
            _require_recursive_subset(actual[key], expected_value, path=child_path)
        return
    if canonical_json(actual) != canonical_json(expected):
        raise ArtifactManifestError(f"{path} is {actual!r}, expected {expected!r}")


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
    "MANIFEST_SHA256_FIELD",
    "ArtifactManifestError",
    "RowValidator",
    "artifact_identity",
    "artifact_manifest_path",
    "artifact_selection_identity",
    "parent_artifact_identity",
    "plain_file_identity",
    "producer_identity",
    "read_verified_artifact_manifest",
    "read_verified_jsonl_artifact",
    "verify_data_manifest_binding",
    "verify_data_manifest_bindings",
    "write_atomic_bytes",
    "write_verified_jsonl_artifact",
]
