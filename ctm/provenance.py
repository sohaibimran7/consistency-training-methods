"""Portable source and runtime provenance for reproducible local runs.

The source record deliberately does not depend on Git.  Git metadata is a
useful descriptive hint when it is safely available, but the source bundle is
the authority: it contains the bytes, modes, and safe symlinks that were
actually captured.  A snapshot is complete only for its explicit active-source
scope; excluded machine-local and output paths are recorded rather than hidden.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO
from urllib.parse import urlsplit, urlunsplit

from ctm.identity import canonical_json, require_sha256, sha256_bytes, sha256_json

SOURCE_SNAPSHOT_SCHEMA = "ctm_source_snapshot"
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
SOURCE_SCOPE_NAME = "ctm_active_source_tree_v1"
SOURCE_EXCLUSION_POLICY = "ctm_source_exclusions_v2"
RUNTIME_IDENTITY_SCHEMA = "ctm_runtime_identity"
RUNTIME_IDENTITY_SCHEMA_VERSION = 2
_SUPPORTED_RUNTIME_IDENTITY_SCHEMA_VERSIONS = frozenset({1, RUNTIME_IDENTITY_SCHEMA_VERSION})


class ProvenanceError(ValueError):
    """A provenance record is incomplete, unsafe, or does not verify."""


class SourceSnapshotError(ProvenanceError):
    """A source tree cannot be captured or restored as a complete declared scope."""


class RuntimeProvenanceError(ProvenanceError):
    """A runtime identity is malformed or differs from the recorded environment."""


# These are path policies, not a secret scanner.  We never inspect the contents
# of a path that policy has excluded, which avoids putting credential values in
# either a snapshot or a provenance error.
_EXCLUDED_DIRECTORY_REASONS = {
    ".git": "version_control_metadata",
    ".hg": "version_control_metadata",
    ".svn": "version_control_metadata",
    ".venv": "environment",
    "venv": "environment",
    "env": "environment",
    ".tox": "environment",
    ".nox": "environment",
    "node_modules": "environment",
    "__pypackages__": "environment",
    "site-packages": "environment",
    "dist-packages": "environment",
    "sitepackages": "environment",
    "__pycache__": "cache",
    ".cache": "cache",
    ".pytest_cache": "cache",
    ".mypy_cache": "cache",
    ".ruff_cache": "cache",
    ".ipynb_checkpoints": "cache",
    "cache": "cache",
    "caches": "cache",
    "logs": "run_output",
    "log": "run_output",
    "artifacts": "run_output",
    "outputs": "run_output",
    "output": "run_output",
    "runs": "run_output",
    "wandb": "run_output",
    "checkpoints": "model_or_checkpoint_output",
    "weights": "model_or_checkpoint_output",
    "model_weights": "model_or_checkpoint_output",
    "models": "model_or_checkpoint_output",
    "_archive": "archived_material",
}
_EXCLUDED_FILE_NAMES = {
    ".ds_store": "machine_metadata",
    ".env": "environment_or_credentials",
    "credentials": "credentials",
    "credentials.json": "credentials",
    "credential.json": "credentials",
    "secrets": "credentials",
    "secrets.json": "credentials",
    "secrets.toml": "credentials",
    "id_rsa": "credentials",
    "id_ed25519": "credentials",
}
_EXCLUDED_SUFFIX_REASONS = {
    ".env": "environment_or_credentials",
    ".key": "credentials",
    ".pem": "credentials",
    ".p12": "credentials",
    ".pfx": "credentials",
    ".kdbx": "credentials",
    ".log": "run_output",
    ".ckpt": "model_or_checkpoint_output",
    ".bin": "model_or_checkpoint_output",
    ".pt": "model_or_checkpoint_output",
    ".pth": "model_or_checkpoint_output",
    ".safetensors": "model_or_checkpoint_output",
    ".onnx": "model_or_checkpoint_output",
    ".gguf": "model_or_checkpoint_output",
    ".h5": "model_or_checkpoint_output",
    ".hdf5": "model_or_checkpoint_output",
}
_DIST_NAME_RE = re.compile(r"[-_.]+")
_VCS_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_VCS_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PORTABLE_DIRECT_URL_SCHEMES = frozenset({"git", "hg", "http", "https", "ssh", "svn"})


def default_source_root() -> Path:
    """Return the checked-in package root without consulting the ambient cwd."""

    return Path(__file__).resolve().parents[1]


def capture_source_snapshot(
    source_root: str | Path | None = None,
    *,
    bundle_dir: str | Path,
    bundle_root: str | Path | None = None,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Capture a self-contained active-source snapshot into a deterministic tar.

    ``bundle_dir`` must be under ``bundle_root`` so the returned reference is
    portable.  Every regular file, directory, and safe relative symlink in the
    declared scope is represented.  A capture failure raises instead of
    returning a partial record.
    """

    root = _resolve_source_root(source_root)
    resolved_bundle_dir = _resolve_path(bundle_dir)
    resolved_bundle_root = _resolve_path(bundle_root if bundle_root is not None else resolved_bundle_dir)
    try:
        resolved_bundle_dir.relative_to(resolved_bundle_root)
    except ValueError as exc:
        raise SourceSnapshotError("bundle_dir must be inside bundle_root") from exc

    extra_exclusions = _normalise_exclusion_paths(root, exclude_paths)
    bundle_relative_to_source = _relative_if_inside(resolved_bundle_dir, root)
    if bundle_relative_to_source is not None and not _policy_exclusion_reason(bundle_relative_to_source, True):
        extra_exclusions.add(bundle_relative_to_source)

    scope, entries, exclusions = _scan_source_tree(root, extra_exclusions)
    source_sha256 = _source_identity(scope, entries)
    resolved_bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = resolved_bundle_dir / f"{source_sha256}.tar"
    bundle_sha256 = _publish_source_bundle(bundle_path, root, entries)
    bundle_relative = _bundle_reference(bundle_path, resolved_bundle_root)
    git = _capture_git_hint(root)

    snapshot: dict[str, Any] = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "scope": scope,
        "entries": entries,
        "exclusions": exclusions,
        "source_sha256": source_sha256,
        "bundle": {
            "format": "tar",
            "path": bundle_relative,
            "sha256": bundle_sha256,
        },
        "git": git,
    }
    snapshot["snapshot_sha256"] = sha256_json(snapshot)
    _validate_source_snapshot(snapshot)
    return snapshot


def verify_source_snapshot(
    snapshot: Mapping[str, Any],
    *,
    bundle_root: str | Path | None = None,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a source record, its retained bundle, and optionally a source tree.

    Passing ``bundle_root`` validates the referenced tar without extracting it.
    Passing ``source_root`` compares the declared scope against files currently
    present there.  Both checks fail closed on a mismatch or unsafe member.
    """

    validated = _validate_source_snapshot(snapshot)
    if bundle_root is not None:
        bundle_path = _resolve_bundle_path(validated, bundle_root)
        actual_bundle_sha256 = _sha256_regular_file(bundle_path)
        expected_bundle_sha256 = validated["bundle"]["sha256"]
        if actual_bundle_sha256 != expected_bundle_sha256:
            raise SourceSnapshotError(
                f"source bundle digest mismatch: record has {expected_bundle_sha256}, "
                f"bundle hashes to {actual_bundle_sha256}"
            )
        tar_entries = _inventory_from_tar(bundle_path)
        if tar_entries != validated["entries"]:
            raise SourceSnapshotError("source bundle inventory does not match the recorded source entries")

    if source_root is not None:
        root = _resolve_source_root(source_root)
        extra_exclusions = set(validated["scope"]["additional_exclusions"])
        actual_scope, actual_entries, _actual_exclusions = _scan_source_tree(root, extra_exclusions)
        actual_sha256 = _source_identity(actual_scope, actual_entries)
        if actual_sha256 != validated["source_sha256"]:
            raise SourceSnapshotError(
                f"source tree digest mismatch: record has {validated['source_sha256']}, " f"tree hashes to {actual_sha256}"
            )
    return validated


def restore_source_snapshot(
    snapshot: Mapping[str, Any],
    destination: str | Path,
    *,
    bundle_root: str | Path,
) -> Path:
    """Restore a verified source bundle into an empty destination directory.

    Tar members are validated before writing.  The restorer never follows a
    bundle symlink, accepts absolute/traversing member names, or writes through
    a symlinked ancestor.
    """

    validated = verify_source_snapshot(snapshot, bundle_root=bundle_root)
    target_root = Path(destination)
    if target_root.exists():
        if target_root.is_symlink() or not target_root.is_dir():
            raise SourceSnapshotError(f"restore destination must be a directory, not {target_root}")
        if any(target_root.iterdir()):
            raise SourceSnapshotError(f"restore destination must be empty: {target_root}")
    else:
        target_root.mkdir(parents=True, mode=0o700)

    bundle_path = _resolve_bundle_path(validated, bundle_root)
    entries = _inventory_from_tar(bundle_path)
    _validate_restore_layout(entries)
    by_path = {entry["path"]: entry for entry in entries}
    directory_entries = [entry for entry in entries if entry["type"] == "directory"]
    file_entries = [entry for entry in entries if entry["type"] == "file"]
    symlink_entries = [entry for entry in entries if entry["type"] == "symlink"]

    for entry in directory_entries:
        path = _restore_target(target_root, entry["path"])
        path.mkdir(mode=0o700)

    with tarfile.open(bundle_path, mode="r") as archive:
        members = { _normalise_tar_name(member.name): member for member in archive.getmembers() }
        if set(members) != set(by_path):
            raise SourceSnapshotError("source bundle changed while being restored")
        for entry in file_entries:
            target = _restore_target(target_root, entry["path"])
            _ensure_directory_ancestors(target_root, entry["path"], by_path)
            member = members[entry["path"]]
            stream = archive.extractfile(member)
            if stream is None:
                raise SourceSnapshotError(f"cannot read source bundle member {entry['path']}")
            _write_restored_regular_file(target, stream, entry)
        for entry in symlink_entries:
            target = _restore_target(target_root, entry["path"])
            _ensure_directory_ancestors(target_root, entry["path"], by_path)
            os.symlink(entry["target"], target)

    # Apply directory permissions after descendants are written, because an
    # intentionally read-only source directory must still be restorable.
    for entry in reversed(directory_entries):
        os.chmod(_restore_target(target_root, entry["path"]), entry["mode"])
    os.chmod(target_root, validated["scope"]["root_mode"])
    verify_source_snapshot(validated, bundle_root=bundle_root, source_root=target_root)
    return target_root


def capture_runtime_identity() -> dict[str, Any]:
    """Record Python, platform, and installed-distribution metadata in-process.

    This is an inventory of installed distribution metadata, not a claim about
    which duplicate distribution an import resolver selected at runtime.
    """

    try:
        distributions, distribution_name_ambiguities = _capture_distribution_identities(importlib.metadata.distributions())
    except RuntimeProvenanceError:
        raise
    except Exception as exc:
        raise RuntimeProvenanceError(f"cannot enumerate installed distributions: {type(exc).__name__}") from exc

    runtime: dict[str, Any] = {
        "schema": RUNTIME_IDENTITY_SCHEMA,
        "schema_version": RUNTIME_IDENTITY_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "distributions": distributions,
        "distribution_name_ambiguities": distribution_name_ambiguities,
    }
    runtime["runtime_sha256"] = sha256_json(runtime)
    _validate_runtime_identity(runtime)
    return runtime


def verify_runtime_identity(
    recorded: Mapping[str, Any],
    *,
    actual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail if a runtime record is malformed or differs from the current runtime."""

    expected = _validate_runtime_identity(recorded)
    observed = _validate_runtime_identity(actual if actual is not None else capture_runtime_identity())
    if expected["runtime_sha256"] != observed["runtime_sha256"]:
        raise RuntimeProvenanceError(
            f"runtime identity mismatch: record has {expected['runtime_sha256']}, "
            f"current runtime has {observed['runtime_sha256']}"
        )
    return observed


def _capture_distribution_identities(installed: Iterable[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    identities: dict[str, dict[str, Any]] = {}
    for distribution in installed:
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise RuntimeProvenanceError("installed distribution has no metadata Name")
        if not isinstance(raw_version, str) or not raw_version.strip():
            raise RuntimeProvenanceError(f"installed distribution {raw_name!r} has no version")
        name = _canonical_distribution_name(raw_name)
        record: dict[str, Any] = {"name": name, "version": raw_version.strip()}
        direct_url = _capture_direct_url_provenance(distribution, distribution_name=name)
        if direct_url is not None:
            record["direct_url"] = direct_url
        identities[canonical_json(record)] = record
    inventory = sorted(identities.values(), key=_distribution_identity_sort_key)
    return inventory, _distribution_name_ambiguities(inventory)


def _distribution_identity_sort_key(identity: Mapping[str, Any]) -> tuple[str, str]:
    return identity["name"], canonical_json(identity)


def _distribution_name_ambiguities(inventory: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    identity_counts: dict[str, int] = {}
    for identity in inventory:
        name = identity["name"]
        identity_counts[name] = identity_counts.get(name, 0) + 1
    return [
        {"name": name, "distinct_identity_count": identity_counts[name]}
        for name in sorted(identity_counts)
        if identity_counts[name] > 1
    ]


def _capture_direct_url_provenance(distribution: Any, *, distribution_name: str) -> dict[str, Any] | None:
    """Return safe PEP 610 provenance without retaining local paths or credentials."""

    try:
        direct_url_text = distribution.read_text("direct_url.json")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise RuntimeProvenanceError(
            f"cannot read direct_url metadata for installed distribution {distribution_name}: {type(exc).__name__}"
        ) from exc
    if direct_url_text is None:
        return None
    if not isinstance(direct_url_text, str):
        raise RuntimeProvenanceError(f"direct_url metadata for installed distribution {distribution_name} is not text")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError as exc:
        raise RuntimeProvenanceError(f"direct_url metadata for installed distribution {distribution_name} is invalid JSON") from exc
    if not isinstance(direct_url, dict):
        raise RuntimeProvenanceError(f"direct_url metadata for installed distribution {distribution_name} is not an object")
    raw_url = direct_url.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise RuntimeProvenanceError(f"direct_url metadata for installed distribution {distribution_name} has no URL")

    editable = _direct_url_editable(direct_url, distribution_name=distribution_name)
    if _direct_url_transport(raw_url, distribution_name=distribution_name) == "file" or editable:
        # A local editable install only identifies a machine-local reference.
        # The source snapshot captures the root application's executable bytes.
        return {"kind": "local", "editable": editable, "portable": False}

    vcs_info = direct_url.get("vcs_info")
    if vcs_info is None:
        return {
            "kind": "archive",
            "url": _sanitise_portable_direct_url(raw_url, distribution_name=distribution_name),
        }
    if not isinstance(vcs_info, Mapping):
        raise RuntimeProvenanceError(f"VCS direct_url metadata for installed distribution {distribution_name} is malformed")
    raw_vcs = vcs_info.get("vcs")
    raw_commit = vcs_info.get("commit_id")
    if not isinstance(raw_vcs, str) or _VCS_KIND_RE.fullmatch(raw_vcs.strip()) is None:
        raise RuntimeProvenanceError(f"VCS direct_url metadata for installed distribution {distribution_name} has an invalid VCS kind")
    if not isinstance(raw_commit, str) or _VCS_VALUE_RE.fullmatch(raw_commit.strip()) is None:
        raise RuntimeProvenanceError(f"VCS direct_url metadata for installed distribution {distribution_name} has an invalid commit id")
    return {
        "kind": "vcs",
        "vcs": raw_vcs.strip().lower(),
        "commit_id": raw_commit.strip(),
        "url": _sanitise_portable_direct_url(raw_url, distribution_name=distribution_name),
    }


def _direct_url_editable(direct_url: Mapping[str, Any], *, distribution_name: str) -> bool:
    directory_info = direct_url.get("dir_info")
    if directory_info is None:
        return False
    if not isinstance(directory_info, Mapping):
        raise RuntimeProvenanceError(f"direct_url directory metadata for installed distribution {distribution_name} is malformed")
    editable = directory_info.get("editable", False)
    if not isinstance(editable, bool):
        raise RuntimeProvenanceError(f"direct_url editable flag for installed distribution {distribution_name} is invalid")
    return editable


def _direct_url_transport(raw_url: str, *, distribution_name: str) -> str:
    try:
        scheme = urlsplit(raw_url).scheme.lower()
    except ValueError as exc:
        raise RuntimeProvenanceError(f"direct_url metadata for installed distribution {distribution_name} has an invalid URL") from exc
    if "+" in scheme:
        _vcs_prefix, scheme = scheme.split("+", 1)
    return scheme


def _sanitise_portable_direct_url(raw_url: str, *, distribution_name: str) -> str:
    """Remove userinfo, query parameters, and fragments from a repository URL."""

    try:
        parsed = urlsplit(raw_url)
        scheme = parsed.scheme.lower()
        if "+" in scheme:
            _vcs_prefix, scheme = scheme.split("+", 1)
        hostname = parsed.hostname
        # Accessing port validates malformed netloc values such as ``host:bad``.
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeProvenanceError(f"direct_url metadata for installed distribution {distribution_name} has an invalid URL") from exc
    if scheme not in _PORTABLE_DIRECT_URL_SCHEMES or not hostname:
        raise RuntimeProvenanceError(
            f"direct_url metadata for installed distribution {distribution_name} is not a portable repository URL"
        )
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    if not netloc:
        raise RuntimeProvenanceError(
            f"direct_url metadata for installed distribution {distribution_name} is not a portable repository URL"
        )
    return urlunsplit((scheme, netloc, parsed.path, "", ""))


def _resolve_source_root(source_root: str | Path | None) -> Path:
    root = _resolve_path(default_source_root() if source_root is None else source_root)
    try:
        metadata = root.stat()
    except OSError as exc:
        raise SourceSnapshotError(f"cannot inspect source root {root}: {exc}") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise SourceSnapshotError(f"source root must be a directory: {root}")
    if _is_pyvenv_root(root):
        raise SourceSnapshotError("source root is a Python virtual environment, not an active source tree")
    return root


def _resolve_path(value: str | Path) -> Path:
    try:
        return Path(value).expanduser().resolve(strict=False)
    except OSError as exc:
        raise SourceSnapshotError(f"cannot resolve path {value!r}: {exc}") from exc


def _normalise_exclusion_paths(root: Path, paths: Iterable[str | Path]) -> set[str]:
    normalised: set[str] = set()
    for raw_path in paths:
        raw = Path(raw_path)
        candidate = _resolve_path(raw if raw.is_absolute() else root / raw)
        relative = _relative_if_inside(candidate, root)
        if relative is None or relative == ".":
            raise SourceSnapshotError(f"additional exclusion must name a child of the source root: {raw_path}")
        normalised.add(relative)
    return normalised


def _relative_if_inside(path: Path, root: Path) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    if relative == Path("."):
        return "."
    return _validate_relative_path(relative.as_posix(), field="path")


def _scan_source_tree(root: Path, extra_exclusions: set[str]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]]]:
    try:
        root_mode = stat.S_IMODE(root.stat().st_mode)
    except OSError as exc:
        raise SourceSnapshotError(f"cannot stat source root {root}: {exc}") from exc
    scope: dict[str, Any] = {
        "name": SOURCE_SCOPE_NAME,
        "coverage": "all regular files, directories, and safe relative symlinks within source_root",
        "exclusion_policy": SOURCE_EXCLUSION_POLICY,
        "additional_exclusions": sorted(extra_exclusions),
        "root_mode": root_mode,
        "status": "complete_for_declared_scope",
    }
    entries: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            with os.scandir(directory) as iterator:
                children = sorted(iterator, key=lambda child: child.name)
        except OSError as exc:
            raise SourceSnapshotError(f"cannot list source directory {directory}: {exc}") from exc
        for child in children:
            relative = _validate_relative_path(PurePosixPath(*prefix, child.name).as_posix(), field="source entry")
            try:
                child_stat = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise SourceSnapshotError(f"cannot stat source entry {relative}: {exc}") from exc
            is_directory = stat.S_ISDIR(child_stat.st_mode)
            path = directory / child.name
            reason = _exclusion_reason(relative, is_directory, extra_exclusions)
            if reason is None and is_directory and _is_pyvenv_root(path):
                reason = "environment"
            if reason is not None:
                exclusions.append(
                    {
                        "path": relative,
                        "reason": reason,
                        "type": "directory" if is_directory else "file",
                    }
                )
                continue
            mode = stat.S_IMODE(child_stat.st_mode)
            if stat.S_ISDIR(child_stat.st_mode):
                entries.append({"path": relative, "type": "directory", "mode": mode})
                visit(path, (*prefix, child.name))
            elif stat.S_ISREG(child_stat.st_mode):
                digest, size = _hash_source_regular_file(path, expected_mode=mode, expected_size=child_stat.st_size)
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "mode": mode,
                        "size": size,
                        "sha256": digest,
                    }
                )
            elif stat.S_ISLNK(child_stat.st_mode):
                try:
                    target = os.readlink(path)
                except OSError as exc:
                    raise SourceSnapshotError(f"cannot read source symlink {relative}: {exc}") from exc
                _validate_symlink_target(target, field=f"source symlink {relative}", link_path=relative)
                entries.append({"path": relative, "type": "symlink", "mode": mode, "target": target})
            else:
                raise SourceSnapshotError(
                    f"source entry {relative} is not a regular file, directory, or safe symlink; "
                    "refusing an incomplete source snapshot"
                )

    visit(root, ())
    entries.sort(key=lambda entry: entry["path"])
    exclusions.sort(key=lambda entry: (entry["path"], entry["reason"], entry["type"]))
    return scope, entries, exclusions


def _exclusion_reason(relative: str, is_directory: bool, extra_exclusions: set[str]) -> str | None:
    for extra in extra_exclusions:
        if relative == extra or relative.startswith(extra + "/"):
            return "run_output_or_snapshot_storage"
    policy_reason = _policy_exclusion_reason(relative, is_directory)
    return policy_reason


def _policy_exclusion_reason(relative: str, is_directory: bool) -> str | None:
    parts = relative.split("/")
    for part in parts:
        name = part.lower()
        if name.startswith(".venv"):
            return "environment"
        reason = _EXCLUDED_DIRECTORY_REASONS.get(name)
        if reason is not None:
            return reason
        if name.startswith(".env."):
            return "environment_or_credentials"
        sensitive_reason = _EXCLUDED_FILE_NAMES.get(name)
        if sensitive_reason is not None:
            return sensitive_reason
    if is_directory:
        return None
    name = parts[-1].lower()
    for suffix, reason in _EXCLUDED_SUFFIX_REASONS.items():
        if name.endswith(suffix):
            return reason
    return None


def _is_pyvenv_root(directory: Path) -> bool:
    """Detect a virtualenv root without reading its metadata or following links."""

    try:
        marker = (directory / "pyvenv.cfg").lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SourceSnapshotError(f"cannot inspect potential virtual environment {directory}: {exc}") from exc
    return stat.S_ISREG(marker.st_mode)


def _hash_source_regular_file(path: Path, *, expected_mode: int, expected_size: int) -> tuple[str, int]:
    file_handle = _open_checked_regular_file(path, expected_mode=expected_mode, expected_size=expected_size)
    try:
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = file_handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        _check_open_file_state(file_handle, path, expected_mode=expected_mode, expected_size=expected_size)
    finally:
        file_handle.close()
    if size != expected_size:
        raise SourceSnapshotError(f"source file changed while being read: {path}")
    return digest.hexdigest(), size


def _open_checked_regular_file(path: Path, *, expected_mode: int, expected_size: int) -> BinaryIO:
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceSnapshotError(f"cannot safely open source file {path}: {exc}") from exc
    handle: BinaryIO | None = None
    try:
        handle = os.fdopen(descriptor, "rb")
        _check_open_file_state(handle, path, expected_mode=expected_mode, expected_size=expected_size)
        return handle
    except Exception:
        if handle is not None:
            handle.close()
        else:
            os.close(descriptor)
        raise


def _check_open_file_state(handle: BinaryIO, path: Path, *, expected_mode: int, expected_size: int) -> None:
    try:
        file_stat = os.fstat(handle.fileno())
    except OSError as exc:
        raise SourceSnapshotError(f"cannot stat open source file {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise SourceSnapshotError(f"source file changed type while being read: {path}")
    if stat.S_IMODE(file_stat.st_mode) != expected_mode or file_stat.st_size != expected_size:
        raise SourceSnapshotError(f"source file changed while being read: {path}")


class _HashingReader:
    def __init__(self, handle: BinaryIO):
        self._handle = handle
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._handle.read(size)
        if chunk:
            self.digest.update(chunk)
            self.size += len(chunk)
        return chunk


def _publish_source_bundle(bundle_path: Path, source_root: Path, entries: list[dict[str, Any]]) -> str:
    existing_sha256 = _existing_source_bundle_sha256(bundle_path, entries)
    if existing_sha256 is not None:
        return existing_sha256

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{bundle_path.stem}.", suffix=".partial", dir=bundle_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.close(descriptor)
        _write_source_tar(temporary_path, source_root, entries)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary_sha256 = _sha256_regular_file(temporary_path)
        try:
            os.link(temporary_path, bundle_path)
        except FileExistsError:
            existing_sha256 = _existing_source_bundle_sha256(bundle_path, entries)
            if existing_sha256 is None:
                raise SourceSnapshotError(f"source bundle disappeared while being published: {bundle_path}")
            _archive_failed_bundle_partial(temporary_path)
            return existing_sha256
        # The hard link reserves this content-addressed destination without
        # replacing a concurrent publisher. Renaming our other link away leaves
        # only the completed final bundle and no temporary path to clean up.
        os.replace(temporary_path, bundle_path)
        return temporary_sha256
    except SourceSnapshotError:
        _archive_failed_bundle_partial(temporary_path)
        raise
    except OSError as exc:
        _archive_failed_bundle_partial(temporary_path)
        raise SourceSnapshotError(f"cannot publish source bundle {bundle_path}: {exc}") from exc
    except Exception:
        _archive_failed_bundle_partial(temporary_path)
        raise


def _existing_source_bundle_sha256(bundle_path: Path, entries: list[dict[str, Any]]) -> str | None:
    try:
        bundle_stat = bundle_path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise SourceSnapshotError(f"cannot stat source bundle {bundle_path}: {exc}") from exc
    if not stat.S_ISREG(bundle_stat.st_mode):
        raise SourceSnapshotError(f"source bundle must be a regular file: {bundle_path}")
    if _inventory_from_tar(bundle_path) != entries:
        raise SourceSnapshotError(f"existing source bundle does not match the captured source: {bundle_path}")
    return _sha256_regular_file(bundle_path)


def _archive_failed_bundle_partial(temporary_path: Path) -> Path | None:
    """Move a failed bundle write aside without deleting its forensic bytes."""

    try:
        partial_stat = temporary_path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(partial_stat.st_mode):
        return None
    archive_dir = temporary_path.parent / "_archive" / "failed-provenance-writes"
    destination = archive_dir / f"{temporary_path.name}.{uuid.uuid4().hex}"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_path, destination)
    except OSError:
        return None
    return destination


def _write_source_tar(bundle_path: Path, source_root: Path, entries: list[dict[str, Any]]) -> None:
    try:
        with tarfile.open(bundle_path, mode="w", format=tarfile.GNU_FORMAT) as archive:
            for entry in entries:
                info = tarfile.TarInfo(entry["path"])
                info.mode = entry["mode"]
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if entry["type"] == "directory":
                    info.type = tarfile.DIRTYPE
                    info.size = 0
                    archive.addfile(info)
                elif entry["type"] == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = entry["target"]
                    info.size = 0
                    archive.addfile(info)
                else:
                    info.type = tarfile.REGTYPE
                    info.size = entry["size"]
                    source_path = source_root.joinpath(*PurePosixPath(entry["path"]).parts)
                    handle = _open_checked_regular_file(
                        source_path,
                        expected_mode=entry["mode"],
                        expected_size=entry["size"],
                    )
                    try:
                        reader = _HashingReader(handle)
                        archive.addfile(info, reader)
                        _check_open_file_state(
                            handle,
                            source_path,
                            expected_mode=entry["mode"],
                            expected_size=entry["size"],
                        )
                    finally:
                        handle.close()
                    if reader.size != entry["size"] or reader.digest.hexdigest() != entry["sha256"]:
                        raise SourceSnapshotError(f"source file changed while writing source bundle: {entry['path']}")
    except SourceSnapshotError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise SourceSnapshotError(f"cannot write deterministic source bundle {bundle_path}: {exc}") from exc


def _source_identity(scope: Mapping[str, Any], entries: list[dict[str, Any]]) -> str:
    return sha256_json(
        {
            "schema": SOURCE_SNAPSHOT_SCHEMA,
            "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "scope": scope,
            "entries": entries,
        }
    )


def _bundle_reference(bundle_path: Path, bundle_root: Path) -> str:
    relative = _relative_if_inside(bundle_path, bundle_root)
    if relative is None or relative == ".":
        raise SourceSnapshotError("source bundle is not below the declared bundle root")
    return relative


def _resolve_bundle_path(snapshot: Mapping[str, Any], bundle_root: str | Path) -> Path:
    root = _resolve_path(bundle_root)
    reference = _validate_relative_path(snapshot["bundle"]["path"], field="source bundle path")
    candidate = _resolve_path(root / Path(*PurePosixPath(reference).parts))
    if _relative_if_inside(candidate, root) is None:
        raise SourceSnapshotError("source bundle reference escapes bundle_root")
    try:
        file_stat = candidate.lstat()
    except OSError as exc:
        raise SourceSnapshotError(f"cannot stat source bundle {candidate}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise SourceSnapshotError(f"source bundle must be a regular file: {candidate}")
    return candidate


def _sha256_regular_file(path: Path) -> str:
    try:
        file_stat = path.lstat()
    except OSError as exc:
        raise SourceSnapshotError(f"cannot stat file {path}: {exc}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise SourceSnapshotError(f"expected regular file: {path}")
    handle = _open_checked_regular_file(
        path,
        expected_mode=stat.S_IMODE(file_stat.st_mode),
        expected_size=file_stat.st_size,
    )
    try:
        digest = hashlib.sha256()
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        _check_open_file_state(
            handle,
            path,
            expected_mode=stat.S_IMODE(file_stat.st_mode),
            expected_size=file_stat.st_size,
        )
        return digest.hexdigest()
    finally:
        handle.close()


def _inventory_from_tar(bundle_path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        with tarfile.open(bundle_path, mode="r") as archive:
            for member in archive.getmembers():
                path = _normalise_tar_name(member.name)
                mode = member.mode & 0o7777
                if member.isdir():
                    entries.append({"path": path, "type": "directory", "mode": mode})
                elif member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise SourceSnapshotError(f"cannot read source bundle member {path}")
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        size += len(chunk)
                    if size != member.size:
                        raise SourceSnapshotError(f"truncated source bundle member {path}")
                    entries.append(
                        {"path": path, "type": "file", "mode": mode, "size": size, "sha256": digest.hexdigest()}
                    )
                elif member.issym():
                    _validate_symlink_target(member.linkname, field=f"source bundle symlink {path}", link_path=path)
                    entries.append({"path": path, "type": "symlink", "mode": mode, "target": member.linkname})
                else:
                    raise SourceSnapshotError(f"unsupported source bundle member type for {path}")
    except SourceSnapshotError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise SourceSnapshotError(f"cannot read source bundle {bundle_path}: {exc}") from exc
    if entries != sorted(entries, key=lambda entry: entry["path"]):
        raise SourceSnapshotError("source bundle members are not in deterministic path order")
    if len({entry["path"] for entry in entries}) != len(entries):
        raise SourceSnapshotError("source bundle has duplicate member paths")
    return entries


def _normalise_tar_name(name: str) -> str:
    return _validate_relative_path(name.rstrip("/"), field="source bundle member")


def _validate_restore_layout(entries: list[dict[str, Any]]) -> None:
    by_path = {entry["path"]: entry for entry in entries}
    for entry in entries:
        parts = PurePosixPath(entry["path"]).parts
        for index in range(1, len(parts)):
            parent = PurePosixPath(*parts[:index]).as_posix()
            parent_entry = by_path.get(parent)
            if parent_entry is None or parent_entry["type"] != "directory":
                raise SourceSnapshotError(f"source bundle member has a non-directory or missing parent: {entry['path']}")


def _restore_target(destination: Path, relative: str) -> Path:
    target = destination.joinpath(*PurePosixPath(relative).parts)
    try:
        target.relative_to(destination)
    except ValueError as exc:
        raise SourceSnapshotError(f"restore path escapes destination: {relative}") from exc
    return target


def _ensure_directory_ancestors(destination: Path, relative: str, by_path: Mapping[str, Mapping[str, Any]]) -> None:
    parts = PurePosixPath(relative).parts
    for index in range(1, len(parts)):
        parent_relative = PurePosixPath(*parts[:index]).as_posix()
        parent = _restore_target(destination, parent_relative)
        parent_entry = by_path[parent_relative]
        if parent_entry["type"] != "directory" or parent.is_symlink() or not parent.is_dir():
            raise SourceSnapshotError(f"restore path has an unsafe directory ancestor: {parent_relative}")


def _write_restored_regular_file(target: Path, stream: BinaryIO, entry: Mapping[str, Any]) -> None:
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise SourceSnapshotError(f"cannot create restored source file {target}: {exc}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as handle:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != entry["size"] or digest.hexdigest() != entry["sha256"]:
            raise SourceSnapshotError(f"restored source file does not match recorded bytes: {entry['path']}")
        os.chmod(target, entry["mode"])
    except Exception:
        raise


def _validate_relative_path(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SourceSnapshotError(f"{field} must be a non-empty relative path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise SourceSnapshotError(f"{field} must not be absolute or traverse its root: {value!r}")
    return candidate.as_posix()


def _validate_symlink_target(target: str, *, field: str, link_path: str | None = None) -> None:
    if not isinstance(target, str) or not target or "\x00" in target:
        raise SourceSnapshotError(f"{field} must be a non-empty relative target")
    candidate = PurePosixPath(target)
    if candidate.is_absolute():
        raise SourceSnapshotError(f"{field} escapes the source root")
    if link_path is None:
        if any(part == ".." for part in candidate.parts):
            raise SourceSnapshotError(f"{field} escapes the source root")
        return
    parent_parts = list(PurePosixPath(link_path).parts[:-1])
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parent_parts:
                raise SourceSnapshotError(f"{field} escapes the source root")
            parent_parts.pop()
        else:
            parent_parts.append(part)


def _capture_git_hint(root: Path) -> dict[str, Any]:
    git_directory = root / ".git"
    try:
        git_stat = git_directory.lstat()
    except FileNotFoundError:
        return {"available": False, "reason": "no_git_metadata_in_source_root"}
    except OSError as exc:
        return {"available": False, "reason": f"cannot_stat_git_metadata:{type(exc).__name__}"}
    if not stat.S_ISDIR(git_stat.st_mode) or stat.S_ISLNK(git_stat.st_mode):
        return {"available": False, "reason": "git_metadata_outside_source_root"}

    def run(*args: str) -> str | None:
        executable = "/usr/bin/git" if Path("/usr/bin/git").is_file() else "git"
        try:
            result = subprocess.run(
                [executable, f"--git-dir={git_directory}", f"--work-tree={root}", *args],
                cwd=root,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=10,
                check=False,
                env={
                    "PATH": os.defpath,
                    "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": os.devnull,
                    "GIT_TERMINAL_PROMPT": "0",
                    "LC_ALL": "C",
                },
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status_output = run("status", "--porcelain=v1", "--untracked-files=all")
    if not commit or branch is None or status_output is None:
        return {"available": False, "reason": "git_commands_unavailable"}
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_output),
        "status_sha256": sha256_bytes(status_output.encode("utf-8")),
    }


def _validate_source_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        raise SourceSnapshotError("source snapshot must be a mapping")
    try:
        copied = json.loads(canonical_json(snapshot))
    except (TypeError, ValueError) as exc:
        raise SourceSnapshotError(f"source snapshot must contain canonical JSON values: {exc}") from exc
    if copied.get("schema") != SOURCE_SNAPSHOT_SCHEMA or copied.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise SourceSnapshotError("unsupported source snapshot schema")
    scope = copied.get("scope")
    entries = copied.get("entries")
    exclusions = copied.get("exclusions")
    bundle = copied.get("bundle")
    if not isinstance(scope, dict) or not isinstance(entries, list) or not isinstance(exclusions, list) or not isinstance(bundle, dict):
        raise SourceSnapshotError("source snapshot has malformed scope, entries, exclusions, or bundle")
    _validate_scope(scope)
    _validate_entries(entries)
    _validate_exclusions(exclusions)
    if bundle.get("format") != "tar":
        raise SourceSnapshotError("source snapshot must reference a tar bundle")
    _validate_relative_path(bundle.get("path"), field="source bundle path")
    try:
        require_sha256(bundle.get("sha256"), field="source bundle sha256")
        require_sha256(copied.get("source_sha256"), field="source snapshot source_sha256")
        require_sha256(copied.get("snapshot_sha256"), field="source snapshot snapshot_sha256")
    except ValueError as exc:
        raise SourceSnapshotError(str(exc)) from exc
    if copied["source_sha256"] != _source_identity(scope, entries):
        raise SourceSnapshotError("source snapshot source_sha256 does not match its scope and entries")
    unsigned = {key: value for key, value in copied.items() if key != "snapshot_sha256"}
    if copied["snapshot_sha256"] != sha256_json(unsigned):
        raise SourceSnapshotError("source snapshot integrity digest does not match its fields")
    return copied


def _validate_scope(scope: Mapping[str, Any]) -> None:
    if scope.get("name") != SOURCE_SCOPE_NAME:
        raise SourceSnapshotError("source snapshot has an unknown scope")
    if scope.get("coverage") != "all regular files, directories, and safe relative symlinks within source_root":
        raise SourceSnapshotError("source snapshot has an unsupported coverage statement")
    if scope.get("exclusion_policy") != SOURCE_EXCLUSION_POLICY:
        raise SourceSnapshotError("source snapshot has an unknown exclusion policy")
    if scope.get("status") != "complete_for_declared_scope":
        raise SourceSnapshotError("source snapshot does not claim a complete declared scope")
    mode = scope.get("root_mode")
    if not _is_mode(mode):
        raise SourceSnapshotError("source snapshot root_mode is invalid")
    extras = scope.get("additional_exclusions")
    if not isinstance(extras, list) or extras != sorted(extras) or len(set(extras)) != len(extras):
        raise SourceSnapshotError("source snapshot additional_exclusions must be sorted and unique")
    for value in extras:
        _validate_relative_path(value, field="source snapshot additional exclusion")


def _validate_entries(entries: list[Any]) -> None:
    paths: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise SourceSnapshotError("source snapshot entry must be an object")
        path = _validate_relative_path(entry.get("path"), field="source snapshot entry path")
        paths.append(path)
        if not _is_mode(entry.get("mode")):
            raise SourceSnapshotError(f"source snapshot entry has invalid mode: {path}")
        kind = entry.get("type")
        if kind == "directory":
            if set(entry) != {"path", "type", "mode"}:
                raise SourceSnapshotError(f"directory source entry has unexpected fields: {path}")
        elif kind == "file":
            if set(entry) != {"path", "type", "mode", "size", "sha256"}:
                raise SourceSnapshotError(f"file source entry has unexpected fields: {path}")
            if not isinstance(entry.get("size"), int) or isinstance(entry["size"], bool) or entry["size"] < 0:
                raise SourceSnapshotError(f"source file entry has invalid size: {path}")
            try:
                require_sha256(entry.get("sha256"), field=f"source file {path} sha256")
            except ValueError as exc:
                raise SourceSnapshotError(str(exc)) from exc
        elif kind == "symlink":
            if set(entry) != {"path", "type", "mode", "target"}:
                raise SourceSnapshotError(f"symlink source entry has unexpected fields: {path}")
            _validate_symlink_target(entry.get("target"), field=f"source symlink {path}", link_path=path)
        else:
            raise SourceSnapshotError(f"source snapshot entry has unknown type: {path}")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise SourceSnapshotError("source snapshot entries must be sorted and unique")
    _validate_restore_layout(entries)


def _validate_exclusions(exclusions: list[Any]) -> None:
    for entry in exclusions:
        if not isinstance(entry, dict) or set(entry) != {"path", "reason", "type"}:
            raise SourceSnapshotError("source snapshot exclusion has malformed fields")
        _validate_relative_path(entry.get("path"), field="source snapshot exclusion path")
        if not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise SourceSnapshotError("source snapshot exclusion has no reason")
        if entry.get("type") not in {"file", "directory"}:
            raise SourceSnapshotError("source snapshot exclusion has invalid type")


def _is_mode(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0o7777


def _canonical_distribution_name(value: str) -> str:
    name = _DIST_NAME_RE.sub("-", value.strip().lower())
    if not name or any(character.isspace() for character in name):
        raise RuntimeProvenanceError(f"invalid installed distribution name: {value!r}")
    return name


def _validate_runtime_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeProvenanceError("runtime identity must be a mapping")
    try:
        copied = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise RuntimeProvenanceError(f"runtime identity must contain canonical JSON values: {exc}") from exc
    schema_version = copied.get("schema_version")
    if (
        copied.get("schema") != RUNTIME_IDENTITY_SCHEMA
        or not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in _SUPPORTED_RUNTIME_IDENTITY_SCHEMA_VERSIONS
    ):
        raise RuntimeProvenanceError("unsupported runtime identity schema")
    expected_fields = {
        "schema",
        "schema_version",
        "python",
        "platform",
        "distributions",
        "runtime_sha256",
    }
    if schema_version == RUNTIME_IDENTITY_SCHEMA_VERSION:
        expected_fields.add("distribution_name_ambiguities")
    if set(copied) != expected_fields:
        raise RuntimeProvenanceError("runtime identity has unexpected fields")
    python = copied.get("python")
    platform_info = copied.get("platform")
    distributions = copied.get("distributions")
    if not isinstance(python, dict) or set(python) != {"implementation", "version", "cache_tag"}:
        raise RuntimeProvenanceError("runtime identity has malformed Python details")
    if not isinstance(platform_info, dict) or set(platform_info) != {"system", "release", "machine"}:
        raise RuntimeProvenanceError("runtime identity has malformed platform details")
    if not all(isinstance(item, str) or item is None for item in python.values()):
        raise RuntimeProvenanceError("runtime identity Python details must be strings or null")
    if not all(isinstance(item, str) for item in platform_info.values()):
        raise RuntimeProvenanceError("runtime identity platform details must be strings")
    if not isinstance(distributions, list):
        raise RuntimeProvenanceError("runtime identity distributions must be a list")
    names: list[str] = []
    identity_keys: list[str] = []
    for distribution in distributions:
        if not isinstance(distribution, dict) or set(distribution) not in ({"name", "version"}, {"name", "version", "direct_url"}):
            raise RuntimeProvenanceError("runtime identity distribution entry is malformed")
        name = distribution.get("name")
        version = distribution.get("version")
        if not isinstance(name, str) or _canonical_distribution_name(name) != name:
            raise RuntimeProvenanceError("runtime identity distribution name is not canonical")
        if not isinstance(version, str) or not version:
            raise RuntimeProvenanceError("runtime identity distribution version is invalid")
        if "direct_url" in distribution:
            _validate_direct_url_provenance(distribution["direct_url"])
        names.append(name)
        identity_keys.append(canonical_json(distribution))
    if distributions != sorted(distributions, key=_distribution_identity_sort_key):
        raise RuntimeProvenanceError("runtime identity distributions must be in deterministic identity order")
    if len(identity_keys) != len(set(identity_keys)):
        raise RuntimeProvenanceError("runtime identity distributions must contain distinct identities")
    if schema_version == 1:
        if len(names) != len(set(names)):
            raise RuntimeProvenanceError("runtime identity v1 distributions must have unique names")
    else:
        _validate_distribution_name_ambiguities(copied["distribution_name_ambiguities"], distributions)
    try:
        require_sha256(copied.get("runtime_sha256"), field="runtime identity runtime_sha256")
    except ValueError as exc:
        raise RuntimeProvenanceError(str(exc)) from exc
    unsigned = {key: item for key, item in copied.items() if key != "runtime_sha256"}
    if copied["runtime_sha256"] != sha256_json(unsigned):
        raise RuntimeProvenanceError("runtime identity integrity digest does not match its fields")
    return copied


def _validate_distribution_name_ambiguities(value: Any, distributions: list[dict[str, Any]]) -> None:
    if not isinstance(value, list):
        raise RuntimeProvenanceError("runtime identity distribution_name_ambiguities must be a list")
    names: list[str] = []
    for ambiguity in value:
        if not isinstance(ambiguity, dict) or set(ambiguity) != {"name", "distinct_identity_count"}:
            raise RuntimeProvenanceError("runtime identity distribution_name_ambiguity is malformed")
        name = ambiguity.get("name")
        count = ambiguity.get("distinct_identity_count")
        if not isinstance(name, str) or _canonical_distribution_name(name) != name:
            raise RuntimeProvenanceError("runtime identity distribution_name_ambiguity name is not canonical")
        if not isinstance(count, int) or isinstance(count, bool) or count < 2:
            raise RuntimeProvenanceError("runtime identity distribution_name_ambiguity count is invalid")
        names.append(name)
    if names != sorted(names) or len(names) != len(set(names)):
        raise RuntimeProvenanceError("runtime identity distribution_name_ambiguities must be sorted and unique")
    expected = _distribution_name_ambiguities(distributions)
    if value != expected:
        raise RuntimeProvenanceError("runtime identity distribution_name_ambiguities do not match the inventory")


def _validate_direct_url_provenance(value: Any) -> None:
    if not isinstance(value, dict):
        raise RuntimeProvenanceError("runtime identity distribution direct_url is malformed")
    kind = value.get("kind")
    if kind == "local":
        if set(value) != {"kind", "editable", "portable"}:
            raise RuntimeProvenanceError("runtime identity local direct_url has unexpected fields")
        if not isinstance(value["editable"], bool) or value["portable"] is not False:
            raise RuntimeProvenanceError("runtime identity local direct_url is invalid")
        return
    if kind == "archive":
        if set(value) != {"kind", "url"}:
            raise RuntimeProvenanceError("runtime identity archive direct_url has unexpected fields")
        _validate_sanitised_direct_url(value.get("url"))
        return
    if kind == "vcs":
        if set(value) != {"kind", "vcs", "commit_id", "url"}:
            raise RuntimeProvenanceError("runtime identity VCS direct_url has unexpected fields")
        vcs = value.get("vcs")
        commit_id = value.get("commit_id")
        if not isinstance(vcs, str) or _VCS_KIND_RE.fullmatch(vcs) is None or vcs != vcs.lower():
            raise RuntimeProvenanceError("runtime identity VCS direct_url has an invalid VCS kind")
        if not isinstance(commit_id, str) or _VCS_VALUE_RE.fullmatch(commit_id) is None:
            raise RuntimeProvenanceError("runtime identity VCS direct_url has an invalid commit id")
        _validate_sanitised_direct_url(value.get("url"))
        return
    raise RuntimeProvenanceError("runtime identity distribution direct_url has an unknown kind")


def _validate_sanitised_direct_url(value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise RuntimeProvenanceError("runtime identity direct_url has an invalid URL")
    sanitised = _sanitise_portable_direct_url(value, distribution_name="runtime identity")
    if sanitised != value:
        raise RuntimeProvenanceError("runtime identity direct_url retains userinfo, query parameters, or a fragment")


__all__ = [
    "ProvenanceError",
    "RUNTIME_IDENTITY_SCHEMA",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "RuntimeProvenanceError",
    "SOURCE_SNAPSHOT_SCHEMA",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "SourceSnapshotError",
    "capture_runtime_identity",
    "capture_source_snapshot",
    "default_source_root",
    "restore_source_snapshot",
    "verify_runtime_identity",
    "verify_source_snapshot",
]
