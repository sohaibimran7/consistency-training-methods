"""Immutable training-attempt manifests with portable source/runtime evidence.

manifest.json remains a convenient current-run discovery path for existing
tools. Its bytes are also written once to manifest-attempts/<attempt>.json;
that immutable record is the provenance authority for a particular attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ctm.cli_safety import redact_secrets
from ctm.identity import canonical_json, require_sha256, sha256_bytes, sha256_json
from ctm.provenance import (
    RuntimeProvenanceError,
    SourceSnapshotError,
    capture_runtime_identity,
    capture_source_snapshot,
    default_source_root,
    verify_runtime_identity,
    verify_source_snapshot,
)

MANIFEST_NAME = "manifest.json"
ATTEMPT_MANIFEST_DIRECTORY = "manifest-attempts"
MANIFEST_SCHEMA = "ctm_training_run_manifest"
MANIFEST_SCHEMA_VERSION = 2

_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_RESERVED_EXTRA_KEYS = {
    "manifest_schema",
    "schema_version",
    "attempt_id",
    "attempt_manifest",
    "kind",
    "model",
    "backend",
    "config_hash",
    "config",
    "input",
    "input_identity",
    "git",
    "source_snapshot",
    "source_identity",
    "environment",
    "environment_identity",
    "written_at",
    "manifest_sha256",
}


class RunManifestError(ValueError):
    """A training manifest is malformed, tampered with, or cannot be published."""


def config_hash(config_dump: dict) -> str:
    """Return the historical configuration identity byte-for-byte.

    New full identities use ctm.identity canonical JSON. This legacy field
    remains compatible with existing manifests and must keep its original JSON
    whitespace, Unicode escaping, and default string conversion.
    """

    canonical = json.dumps(config_dump, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def write_run_manifest(
    log_dir: str | Path,
    *,
    kind: str,
    model: str,
    backend: Any,
    config_dump: dict,
    extra: Optional[dict] = None,
    source_root: str | Path | None = None,
    attempt_id: str | None = None,
) -> Path:
    """Write one immutable attempt manifest and refresh manifest.json.

    Source capture and runtime enumeration occur before any manifest is
    published. If either fails, callers receive an exception and no attempt is
    represented as a valid record.
    """

    if not isinstance(kind, str) or not kind.strip():
        raise RunManifestError("kind must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise RunManifestError("model must be a non-empty string")
    if not isinstance(config_dump, Mapping):
        raise RunManifestError("config_dump must be a mapping")
    if extra is not None and not isinstance(extra, Mapping):
        raise RunManifestError("extra must be a mapping when provided")

    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    current_path = directory / MANIFEST_NAME
    _require_regular_file(current_path, label="current run manifest", missing_ok=True)
    resolved_attempt_id = _normalise_attempt_id(attempt_id)
    redacted_config = _json_copy(redact_secrets(config_dump), field="config_dump")
    redacted_extra = _json_copy(redact_secrets(dict(extra or {})), field="extra")
    overlap = sorted(set(redacted_extra) & _RESERVED_EXTRA_KEYS)
    if overlap:
        raise RunManifestError(f"extra cannot replace manifest-owned fields: {overlap}")

    backend_name = type(backend).__name__
    source_root_path = Path(source_root) if source_root is not None else default_source_root()
    source_snapshot = capture_source_snapshot(
        source_root_path,
        bundle_dir=directory / "provenance" / "source",
        bundle_root=directory,
        exclude_paths=_run_output_exclusions(source_root_path, directory),
    )
    environment = capture_runtime_identity()
    inputs = {
        "kind": kind,
        "model": model,
        "backend": backend_name,
        "config": redacted_config,
        "extra": redacted_extra,
    }
    input_identity = sha256_json(inputs)
    attempt_relative = (Path(ATTEMPT_MANIFEST_DIRECTORY) / f"{resolved_attempt_id}.json").as_posix()
    manifest: dict[str, Any] = {
        "manifest_schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "attempt_id": resolved_attempt_id,
        "attempt_manifest": attempt_relative,
        "kind": kind,
        "model": model,
        "backend": backend_name,
        "config_hash": config_hash(redacted_config),
        "config": redacted_config,
        "input": inputs,
        "input_identity": input_identity,
        "git": _legacy_git_hint(source_snapshot["git"]),
        "source_snapshot": source_snapshot,
        "source_identity": source_snapshot["source_sha256"],
        "environment": environment,
        "environment_identity": environment["runtime_sha256"],
        "written_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **redacted_extra,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    payload = _manifest_payload(manifest)

    immutable_path = directory / attempt_relative
    _write_immutable(immutable_path, payload)
    _preserve_legacy_current_manifest(current_path)
    _write_current(current_path, payload)
    return current_path


def read_run_manifest(log_dir: str | Path) -> Optional[dict]:
    """Read the current manifest without requiring a provenance verification pass."""

    path = Path(log_dir) / MANIFEST_NAME
    if not _require_regular_file(path, label="current run manifest", missing_ok=True):
        return None
    return _read_json_object(path, label="run manifest")


def read_attempt_manifest(log_dir: str | Path, attempt_id: str) -> Optional[dict]:
    """Read one immutable attempt record by its validated attempt identifier."""

    path = Path(log_dir) / ATTEMPT_MANIFEST_DIRECTORY / f"{_normalise_attempt_id(attempt_id)}.json"
    if not _require_regular_file(path, label="attempt manifest", missing_ok=True):
        return None
    return _read_json_object(path, label="attempt manifest")


def verify_run_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    source_root: str | Path | None = None,
    bundle_root: str | Path | None = None,
    verify_environment: bool = False,
) -> dict[str, Any]:
    """Verify a new-format manifest, retained source bytes, and optional runtime.

    The source bundle is always checked when a path is supplied. Runtime
    comparison is opt-in because a historical run is often inspected from a
    different machine or virtual environment.
    """

    manifest_path: Path | None = None
    if isinstance(manifest_or_path, Mapping):
        manifest = _validate_manifest(manifest_or_path)
    else:
        candidate = Path(manifest_or_path)
        manifest_path = candidate / MANIFEST_NAME if candidate.is_dir() else candidate
        manifest = _validate_manifest(_read_json_object(manifest_path, label="run manifest"))

    if bundle_root is None:
        if manifest_path is None:
            raise RunManifestError("bundle_root is required when verifying an in-memory manifest")
        resolved_bundle_root = _run_directory_for_manifest_path(manifest_path)
    else:
        resolved_bundle_root = Path(bundle_root)
    try:
        verify_source_snapshot(
            manifest["source_snapshot"],
            bundle_root=resolved_bundle_root,
            source_root=source_root,
        )
    except SourceSnapshotError as exc:
        raise RunManifestError(f"source snapshot verification failed: {exc}") from exc
    if verify_environment:
        try:
            verify_runtime_identity(manifest["environment"])
        except RuntimeProvenanceError as exc:
            raise RunManifestError(f"runtime identity verification failed: {exc}") from exc
    return manifest


def _normalise_attempt_id(value: str | None) -> str:
    attempt_id = uuid.uuid4().hex if value is None else value
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        raise RunManifestError("attempt_id must use only letters, digits, '.', '_', or '-' and cannot contain a path")
    return attempt_id


def _run_output_exclusions(source_root: Path, log_dir: Path) -> list[Path]:
    """Exclude a run directory when a caller deliberately places it under source."""

    try:
        root = source_root.expanduser().resolve(strict=False)
        output = log_dir.expanduser().resolve(strict=False)
        relative = output.relative_to(root)
    except (OSError, ValueError):
        return []
    if relative == Path("."):
        raise RunManifestError("log_dir cannot be the same directory as source_root")
    return [output]


def _legacy_git_hint(git: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the legacy field useful without reintroducing an incomplete diff."""

    if git.get("available") is True:
        return {
            "git_sha": git["commit"],
            "git_branch": git["branch"],
            "git_dirty": git["dirty"],
            "git_status_sha256": git["status_sha256"],
        }
    reason = git.get("reason")
    return {"git_error": reason if isinstance(reason, str) and reason else "git metadata unavailable"}


def _json_copy(value: object, *, field: str) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RunManifestError(f"{field} must contain JSON-compatible values: {exc}") from exc


def _manifest_payload(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable training attempt manifest: {path}") from exc
    except OSError as exc:
        raise RunManifestError(f"cannot publish immutable attempt manifest {path}: {exc}") from exc


def _write_current(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_regular_file(path, label="current run manifest", missing_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if descriptor != -1:
            os.close(descriptor)
        retained = _archive_failed_manifest_partial(temporary)
        retained_path = retained if retained is not None else temporary
        raise RunManifestError(f"cannot refresh current run manifest {path}; partial retained at {retained_path}: {exc}") from exc


def _read_json_object(path: Path, *, label: str) -> dict:
    _require_regular_file(path, label=label, missing_ok=False)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunManifestError(f"{label} {path} must contain a JSON object")
    return value


def _require_regular_file(path: Path, *, label: str, missing_ok: bool) -> bool:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise RunManifestError(f"cannot read {label} {path}: file does not exist") from None
    except OSError as exc:
        raise RunManifestError(f"cannot inspect {label} {path}: {exc}") from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise RunManifestError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(file_stat.st_mode):
        raise RunManifestError(f"{label} must be a regular file: {path}")
    return True


def _preserve_legacy_current_manifest(current_path: Path) -> None:
    if not _require_regular_file(current_path, label="current run manifest", missing_ok=True):
        return
    try:
        legacy_bytes = current_path.read_bytes()
    except OSError as exc:
        raise RunManifestError(f"cannot read current run manifest {current_path}: {exc}") from exc
    if _is_retained_v2_current_manifest(current_path, legacy_bytes):
        return
    legacy_sha256 = sha256_bytes(legacy_bytes)
    legacy_path = current_path.parent / ATTEMPT_MANIFEST_DIRECTORY / f"legacy-{legacy_sha256}.json"
    _write_legacy_manifest_bytes(legacy_path, legacy_bytes)


def _is_retained_v2_current_manifest(current_path: Path, payload: bytes) -> bool:
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA or manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return False
    attempt_id = manifest.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_ID_RE.fullmatch(attempt_id) is None:
        return False
    expected_path = (Path(ATTEMPT_MANIFEST_DIRECTORY) / f"{attempt_id}.json").as_posix()
    if manifest.get("attempt_manifest") != expected_path:
        return False
    immutable_path = current_path.parent / expected_path
    if not _require_regular_file(immutable_path, label="immutable attempt manifest", missing_ok=True):
        return False
    try:
        return immutable_path.read_bytes() == payload
    except OSError as exc:
        raise RunManifestError(f"cannot read immutable attempt manifest {immutable_path}: {exc}") from exc


def _write_legacy_manifest_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return
    except FileExistsError:
        _require_regular_file(path, label="preserved legacy manifest", missing_ok=False)
    except OSError as exc:
        raise RunManifestError(f"cannot preserve legacy manifest {path}: {exc}") from exc
    try:
        existing = path.read_bytes()
    except OSError as exc:
        raise RunManifestError(f"cannot read preserved legacy manifest {path}: {exc}") from exc
    if existing != payload:
        raise RunManifestError(f"preserved legacy manifest has conflicting bytes: {path}")


def _archive_failed_manifest_partial(temporary: Path) -> Path | None:
    try:
        temporary_stat = temporary.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(temporary_stat.st_mode):
        return None
    archive_dir = temporary.parent / "_archive" / "failed-provenance-writes"
    destination = archive_dir / f"{temporary.name}.{uuid.uuid4().hex}"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except OSError:
        return None
    return destination


def _run_directory_for_manifest_path(path: Path) -> Path:
    if path.name == MANIFEST_NAME:
        return path.parent
    if path.parent.name == ATTEMPT_MANIFEST_DIRECTORY:
        return path.parent.parent
    return path.parent


def _validate_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RunManifestError("run manifest must be a mapping")
    try:
        copied = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise RunManifestError(f"run manifest must contain canonical JSON values: {exc}") from exc
    if copied.get("manifest_schema") != MANIFEST_SCHEMA or copied.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise RunManifestError("unsupported run manifest schema")
    for field in ("kind", "model", "backend", "attempt_id", "attempt_manifest", "written_at"):
        if not isinstance(copied.get(field), str) or not copied[field]:
            raise RunManifestError(f"run manifest has invalid {field}")
    attempt_id = _normalise_attempt_id(copied["attempt_id"])
    expected_attempt_path = (Path(ATTEMPT_MANIFEST_DIRECTORY) / f"{attempt_id}.json").as_posix()
    if copied["attempt_manifest"] != expected_attempt_path:
        raise RunManifestError("run manifest attempt_manifest does not match attempt_id")
    source_snapshot = copied.get("source_snapshot")
    environment = copied.get("environment")
    if not isinstance(source_snapshot, dict) or not isinstance(environment, dict):
        raise RunManifestError("run manifest has malformed source_snapshot or environment")
    if not isinstance(copied.get("config"), dict) or not isinstance(copied.get("input"), dict):
        raise RunManifestError("run manifest has malformed config or input")
    inputs = copied["input"]
    expected_input_keys = {"kind", "model", "backend", "config", "extra"}
    if set(inputs) != expected_input_keys:
        raise RunManifestError("run manifest input has unexpected fields")
    if inputs["kind"] != copied["kind"] or inputs["model"] != copied["model"] or inputs["backend"] != copied["backend"]:
        raise RunManifestError("run manifest input disagrees with top-level identity fields")
    if inputs["config"] != copied["config"] or not isinstance(inputs["extra"], dict):
        raise RunManifestError("run manifest input disagrees with config or has malformed extra")
    expected_config_hash = config_hash(copied["config"])
    if copied.get("config_hash") != expected_config_hash:
        raise RunManifestError("run manifest config_hash does not match config")
    expected_input_identity = sha256_json(inputs)
    if copied.get("input_identity") != expected_input_identity:
        raise RunManifestError("run manifest input_identity does not match inputs")
    if copied.get("source_identity") != source_snapshot.get("source_sha256"):
        raise RunManifestError("run manifest source_identity does not match source_snapshot")
    if copied.get("environment_identity") != environment.get("runtime_sha256"):
        raise RunManifestError("run manifest environment_identity does not match environment")
    try:
        require_sha256(copied.get("input_identity"), field="run manifest input_identity")
        require_sha256(copied.get("source_identity"), field="run manifest source_identity")
        require_sha256(copied.get("environment_identity"), field="run manifest environment_identity")
        require_sha256(copied.get("manifest_sha256"), field="run manifest manifest_sha256")
    except ValueError as exc:
        raise RunManifestError(str(exc)) from exc
    try:
        verify_source_snapshot(source_snapshot)
    except SourceSnapshotError as exc:
        raise RunManifestError(f"run manifest source_snapshot is invalid: {exc}") from exc
    try:
        # Structural validation only; a historical runtime is not compared to the
        # current environment unless the caller requests it in verify_run_manifest.
        verify_runtime_identity(environment, actual=environment)
    except RuntimeProvenanceError as exc:
        raise RunManifestError(f"run manifest environment is invalid: {exc}") from exc
    unsigned = {key: item for key, item in copied.items() if key != "manifest_sha256"}
    if copied["manifest_sha256"] != sha256_json(unsigned):
        raise RunManifestError("run manifest integrity digest does not match its fields")
    return copied


__all__ = [
    "ATTEMPT_MANIFEST_DIRECTORY",
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA",
    "MANIFEST_SCHEMA_VERSION",
    "RunManifestError",
    "config_hash",
    "read_attempt_manifest",
    "read_run_manifest",
    "verify_run_manifest",
    "write_run_manifest",
]
