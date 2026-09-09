"""Immutable local lifecycle records for generic experiment attempts.

The experiment runner owns orchestration; this module owns the durable record
of what it actually asked a child process to run.  Each attempt and command
gets a UUID-addressed directory, so concurrent commands never append to a
shared mutable journal.  Source and parent-runtime provenance are captured
once while an attempt is created.  A missing terminal event deliberately
means ``incomplete`` (for example after a SIGKILL), never success.

Records retain content identities for explicitly declared references and for a
small, clearly heuristic set of argv references.  External URIs are retained
only as a sanitized digest and scheme: they remain explicitly unverified.
"""

from __future__ import annotations

import datetime as datetime
import hashlib
import json
import os
import stat
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from ctm.identity import canonical_json, require_sha256, sha256_bytes, sha256_json, sha256_text
from ctm.provenance import (
    ProvenanceError,
    capture_runtime_identity,
    capture_source_snapshot,
    default_source_root,
    verify_runtime_identity,
    verify_source_snapshot,
)


EXPERIMENT_RECORD_SCHEMA = "ctm_experiment_lifecycle_record"
EXPERIMENT_RECORD_SCHEMA_VERSION = 1
ATTEMPT_RECORD_FILE = "attempt.json"
COMMANDS_DIRECTORY = "commands"
_RETAINED_ATTEMPT_INPUTS_DIRECTORY = "inputs"
_RETAINED_SOURCE_YAML_FILE = f"{_RETAINED_ATTEMPT_INPUTS_DIRECTORY}/source.yaml"
_RETAINED_RESOLVED_PLAN_FILE = f"{_RETAINED_ATTEMPT_INPUTS_DIRECTORY}/resolved-plan.yaml"
_TERMINAL_EVENTS = {
    "succeeded": ("end.json", "end"),
    "failed": ("error.json", "error"),
    "error": ("error.json", "error"),
    "interrupted": ("interrupted.json", "interrupted"),
}
_TERMINAL_FILE_NAMES = tuple(dict.fromkeys(name for name, _event in _TERMINAL_EVENTS.values()))
_INPUT_FLAG_PARTS = frozenset(
    {
        "artifact",
        "attestation",
        "checkpoint",
        "data",
        "dataset",
        "file",
        "input",
        "manifest",
        "path",
        "resume",
        "source",
    }
)
_OUTPUT_FLAG_PARTS = frozenset({"destination", "export", "out", "output", "report", "results", "save", "write"})
_NON_REFERENCE_FLAG_SUFFIXES = frozenset({"every", "interval", "steps", "count", "limit", "size", "workers"})
_PATHLIKE_SUFFIXES = frozenset(
    {
        ".csv",
        ".json",
        ".jsonl",
        ".npy",
        ".npz",
        ".parquet",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)


class ExperimentRecordError(ValueError):
    """An immutable attempt or command record could not be created or verified."""


@dataclass(frozen=True)
class ExperimentAttempt:
    """Location and identity of one immutable runner attempt."""

    attempt_id: str
    directory: Path
    bundle_root: Path


@dataclass(frozen=True)
class ExperimentCommand:
    """Location and identity of one command started within an attempt."""

    attempt: ExperimentAttempt
    command_id: str
    directory: Path
    declared_outputs: tuple[str, ...]
    working_directory: Path
    inferred_outputs: tuple[str, ...] = ()


def create_attempt_record(
    attempts_directory: str | Path,
    *,
    experiment_name: str,
    runner_argv: Sequence[str],
    source_yaml: str | Path,
    resolved_plan_text: str,
    resolved_plan_path: str | Path | None = None,
    source_root: str | Path | None = None,
    bundle_root: str | Path | None = None,
    bundle_directory: str | Path | None = None,
    working_directory: str | Path | None = None,
    execution_target: str | None = None,
    selected_stages: Sequence[str] = (),
) -> ExperimentAttempt:
    """Capture one immutable, locally retained runner attempt.

    ``source_root`` defaults to :func:`ctm.provenance.default_source_root`, so
    source capture never depends on the shell's current directory.  Source
    snapshots are retained in ``bundle_directory`` (a local shared content
    store by default) and referenced from the immutable attempt descriptor.
    A source snapshot failure creates no valid attempt record and raises before
    a caller can launch a child process.
    """

    if not isinstance(experiment_name, str) or not experiment_name:
        raise ExperimentRecordError("experiment_name must be a non-empty string")
    argv = _string_list(runner_argv, field="runner_argv")
    if not isinstance(resolved_plan_text, str):
        raise ExperimentRecordError("resolved_plan_text must be a string")
    if execution_target is not None and (not isinstance(execution_target, str) or not execution_target):
        raise ExperimentRecordError("execution_target must be a non-empty string when provided")
    stages = _string_list(selected_stages, field="selected_stages")

    attempts_path = _ensure_safe_directory(_absolute_path(attempts_directory))
    root = _absolute_path(source_root if source_root is not None else default_source_root())
    working_path = _absolute_path(working_directory if working_directory is not None else Path.cwd())
    storage_root = _ensure_safe_directory(_absolute_path(bundle_root if bundle_root is not None else attempts_path.parent))
    storage_directory = _ensure_safe_directory(
        _absolute_path(bundle_directory if bundle_directory is not None else storage_root / "source-bundles")
    )
    if not _is_lexically_within(storage_directory, storage_root):
        raise ExperimentRecordError("bundle_directory must be inside bundle_root")

    attempt_id, attempt_directory = _create_unique_directory(attempts_path)
    try:
        snapshot = capture_source_snapshot(root, bundle_dir=storage_directory, bundle_root=storage_root)
        runtime_identity = capture_runtime_identity()
        source_identity = _capture_and_retain_source_yaml(
            source_yaml,
            working_directory=working_path,
            attempt_directory=attempt_directory,
        )
        plan_bytes = resolved_plan_text.encode("utf-8")
        plan_identity: dict[str, Any] = {
            "encoding": "utf-8",
            "size_bytes": len(plan_bytes),
            "content_sha256": sha256_bytes(plan_bytes),
            "retained_copy": _retain_attempt_bytes(
                attempt_directory,
                relative_path=_RETAINED_RESOLVED_PLAN_FILE,
                payload=plan_bytes,
            ),
        }
        if resolved_plan_path is not None:
            plan_identity["path"] = _stored_path(_absolute_path(resolved_plan_path, base=working_path), working_path)

        document: dict[str, Any] = {
            "schema": EXPERIMENT_RECORD_SCHEMA,
            "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
            "record_type": "attempt",
            "attempt_id": attempt_id,
            "created_at": _utc_now(),
            "experiment": {
                "name": experiment_name,
                "execution_target": execution_target,
                "selected_stages": stages,
            },
            "runner": {
                "argv": argv,
                "working_directory": str(working_path),
                "runtime_identity_scope": (
                    "parent_runner_process_only; command child environments are not inferred or claimed by this record"
                ),
            },
            "plan": {
                "source_yaml": source_identity,
                "resolved_selected_plan": plan_identity,
            },
            "provenance": {
                "source_snapshot": snapshot,
                "runtime_identity": runtime_identity,
                "source_bundle_storage": "local_shared_content_addressed_store",
            },
            "lifecycle": {
                "terminal_event_required_for_completion": True,
                "status_without_terminal_event": "incomplete",
            },
        }
        _publish_immutable_json(attempt_directory / ATTEMPT_RECORD_FILE, _with_record_digest(document))
    except Exception as exc:
        _write_creation_failure(attempt_directory, attempt_id, exc)
        if isinstance(exc, ExperimentRecordError):
            raise
        if isinstance(exc, ProvenanceError):
            raise ExperimentRecordError(f"cannot capture attempt provenance: {exc}") from exc
        raise ExperimentRecordError(f"cannot create experiment attempt record: {type(exc).__name__}") from exc
    return ExperimentAttempt(attempt_id=attempt_id, directory=attempt_directory, bundle_root=storage_root)


def start_command_record(
    attempt: ExperimentAttempt,
    *,
    stage: str,
    name: str,
    argv: Sequence[str],
    cuda_visible_devices: Sequence[str] | str | None = None,
    cuda_placement_mode: str | None = None,
    declared_inputs: str | Path | Sequence[str | Path] | None = None,
    declared_outputs: str | Path | Sequence[str | Path] | None = None,
    working_directory: str | Path | None = None,
) -> ExperimentCommand:
    """Create an immutable command-start record before launching a child.

    A caller must call this before ``subprocess.Popen``.  If it raises, no
    complete start record exists and the caller can fail closed without
    launching the command.
    """

    _validate_attempt_handle(attempt)
    if not isinstance(stage, str) or not stage:
        raise ExperimentRecordError("stage must be a non-empty string")
    if not isinstance(name, str) or not name:
        raise ExperimentRecordError("name must be a non-empty string")
    command_argv = _string_list(argv, field="command argv")
    if not command_argv:
        raise ExperimentRecordError("command argv must be non-empty")
    working_path = _absolute_path(working_directory if working_directory is not None else attempt.directory.parent)
    inputs = _declared_references(declared_inputs, field="declared_inputs")
    outputs = _declared_references(declared_outputs, field="declared_outputs")
    command_id, command_directory = _create_unique_directory(_ensure_safe_directory(attempt.directory / COMMANDS_DIRECTORY))

    declared_input_records = [capture_path_identity(reference, working_directory=working_path) for reference in inputs]
    declared_output_baseline = [capture_path_identity(reference, working_directory=working_path) for reference in outputs]
    inferred_input_records = _infer_argv_input_references(command_argv, working_directory=working_path)
    inferred_outputs = _infer_argv_output_values(command_argv)
    inferred_output_baseline = [capture_path_identity(reference, working_directory=working_path) for reference in inferred_outputs]
    document: dict[str, Any] = {
        "schema": EXPERIMENT_RECORD_SCHEMA,
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "record_type": "command_start",
        "attempt_id": attempt.attempt_id,
        "command_id": command_id,
        "started_at": _utc_now(),
        "stage": stage,
        "name": name,
        "argv": command_argv,
        "working_directory": str(working_path),
        "cuda_placement": _cuda_placement(cuda_visible_devices, mode=cuda_placement_mode),
        "input_lineage": {
            "declared_inputs": declared_input_records,
            "inferred_inputs": inferred_input_records,
            "declared_output_baseline": declared_output_baseline,
            "inferred_output_baseline": inferred_output_baseline,
            "coverage": {
                "declared_inputs": _declared_coverage(declared_input_records),
                "declared_outputs": _declared_coverage(declared_output_baseline),
                "inferred_inputs": {
                    "scope": "heuristic references inferred from argv flags and path-like positional values",
                    "status": "not_complete_argument_coverage",
                },
                "inferred_outputs": {
                    "scope": "heuristic output references inferred from argv flags",
                    "status": "not_complete_argument_coverage",
                },
            },
        },
    }
    try:
        _publish_immutable_json(command_directory / "start.json", _with_record_digest(document))
    except Exception as exc:
        _write_command_creation_failure(command_directory, attempt.attempt_id, command_id, exc)
        if isinstance(exc, ExperimentRecordError):
            raise
        raise ExperimentRecordError(f"cannot create command start record: {type(exc).__name__}") from exc
    return ExperimentCommand(
        attempt=attempt,
        command_id=command_id,
        directory=command_directory,
        declared_outputs=tuple(outputs),
        inferred_outputs=tuple(inferred_outputs),
        working_directory=working_path,
    )


def complete_command_record(
    command: ExperimentCommand,
    *,
    outcome: str,
    return_code: int | None = None,
    checkpoint: str | Path | None = None,
    error: BaseException | None = None,
) -> Path:
    """Write exactly one immutable command terminal event.

    ``succeeded`` is valid only when ``return_code`` is zero.  A process that
    dies with its runner never reaches this function, leaving only ``start``
    and therefore an explicit incomplete lifecycle.
    """

    _validate_command_handle(command)
    if outcome not in _TERMINAL_EVENTS:
        raise ExperimentRecordError(f"unsupported command outcome: {outcome!r}")
    if outcome == "succeeded" and return_code not in {None, 0}:
        raise ExperimentRecordError("a successful command terminal record requires return_code 0")
    if return_code is not None and (isinstance(return_code, bool) or not isinstance(return_code, int)):
        raise ExperimentRecordError("return_code must be an integer when provided")
    _load_and_validate_command_handle(command)
    _reject_existing_terminal_event(command.directory)
    filename, event = _TERMINAL_EVENTS[outcome]
    output_records = [capture_path_identity(reference, working_directory=command.working_directory) for reference in command.declared_outputs]
    inferred_output_records = [
        capture_path_identity(reference, working_directory=command.working_directory) for reference in command.inferred_outputs
    ]
    checkpoint_record = (
        capture_path_identity(checkpoint, working_directory=command.working_directory) if checkpoint is not None else None
    )
    document: dict[str, Any] = {
        "schema": EXPERIMENT_RECORD_SCHEMA,
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "record_type": "command_terminal",
        "event": event,
        "attempt_id": command.attempt.attempt_id,
        "command_id": command.command_id,
        "recorded_at": _utc_now(),
        "outcome": outcome,
        "return_code": 0 if outcome == "succeeded" and return_code is None else return_code,
        "output_lineage": {
            "declared_outputs": output_records,
            "inferred_outputs": inferred_output_records,
            "announced_checkpoint": checkpoint_record,
            "coverage": {
                "declared_outputs": _declared_coverage(output_records),
                "inferred_outputs": {
                    "scope": "heuristic output references inferred from argv flags",
                    "status": "not_complete_argument_coverage",
                },
                "announced_checkpoint": _declared_coverage([checkpoint_record] if checkpoint_record is not None else []),
            },
        },
    }
    error_record = _error_record(error, return_code=return_code)
    if error_record is not None:
        document["error"] = error_record
    target = command.directory / filename
    _publish_immutable_json(target, _with_record_digest(document))
    return target


def complete_attempt_record(
    attempt: ExperimentAttempt,
    *,
    outcome: str,
    error: BaseException | None = None,
) -> Path:
    """Write the immutable terminal event for a runner attempt."""

    _validate_attempt_handle(attempt)
    if outcome not in _TERMINAL_EVENTS:
        raise ExperimentRecordError(f"unsupported attempt outcome: {outcome!r}")
    _load_and_validate_attempt(attempt.directory)
    _reject_existing_terminal_event(attempt.directory)
    filename, event = _TERMINAL_EVENTS[outcome]
    command_count = len(_command_directories(attempt.directory))
    document: dict[str, Any] = {
        "schema": EXPERIMENT_RECORD_SCHEMA,
        "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
        "record_type": "attempt_terminal",
        "event": event,
        "attempt_id": attempt.attempt_id,
        "recorded_at": _utc_now(),
        "outcome": outcome,
        "command_count": command_count,
    }
    error_record = _error_record(error)
    if error_record is not None:
        document["error"] = error_record
    target = attempt.directory / filename
    _publish_immutable_json(target, _with_record_digest(document))
    return target


def capture_path_identity(
    reference: str | Path,
    *,
    working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Capture a safe content identity for one local path or external URI.

    Regular files are hashed in chunks with a before/after state check.
    Directories receive a deterministic recursive tree identity without
    following symlinks.  Missing paths, symlinks, special files, unstable
    paths, and external URIs are retained as explicit unverified states.
    """

    if isinstance(reference, Path):
        raw = str(reference)
    elif isinstance(reference, str):
        raw = reference
    else:
        raise ExperimentRecordError("path reference must be a string or Path")
    if not raw or "\x00" in raw:
        raise ExperimentRecordError("path reference must be a non-empty string without NUL")
    base = _absolute_path(working_directory if working_directory is not None else Path.cwd())
    local = _file_uri_path(raw)
    if local is None:
        return _external_uri_record(raw)
    path = _absolute_path(local, base=base)
    record: dict[str, Any] = {
        "kind": "local_path",
        "path": _stored_path(path, base),
    }
    if _first_symlink_ancestor(path) is not None:
        return {**record, "status": "unverified_symlink_ancestor"}
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {**record, "status": "missing"}
    except OSError as exc:
        return {**record, "status": "unverified_os_error", "error_type": type(exc).__name__}
    if stat.S_ISLNK(metadata.st_mode):
        try:
            target = os.readlink(path)
        except OSError as exc:
            return {**record, "status": "unverified_symlink", "error_type": type(exc).__name__}
        return {
            **record,
            "status": "unverified_symlink",
            "target_sha256": sha256_text(target),
        }
    if stat.S_ISREG(metadata.st_mode):
        try:
            digest, size = _hash_regular_file(path, metadata)
        except _ReferenceChanged:
            return {**record, "status": "unverified_changed_during_capture", "file_type": "regular"}
        except OSError as exc:
            return {**record, "status": "unverified_os_error", "file_type": "regular", "error_type": type(exc).__name__}
        return {
            **record,
            "status": "captured",
            "file_type": "regular",
            "mode": stat.S_IMODE(metadata.st_mode),
            "size_bytes": size,
            "content_sha256": digest,
        }
    if stat.S_ISDIR(metadata.st_mode):
        try:
            tree_sha256, file_count, total_size = _hash_directory(path, metadata)
        except _ReferenceChanged:
            return {**record, "status": "unverified_changed_during_capture", "file_type": "directory"}
        except _DirectoryContainsSymlink:
            return {**record, "status": "unverified_symlink", "file_type": "directory"}
        except OSError as exc:
            return {**record, "status": "unverified_os_error", "file_type": "directory", "error_type": type(exc).__name__}
        return {
            **record,
            "status": "captured",
            "file_type": "directory",
            "mode": stat.S_IMODE(metadata.st_mode),
            "tree_sha256": tree_sha256,
            "file_count": file_count,
            "total_size_bytes": total_size,
        }
    return {**record, "status": "unverified_special_file", "file_type": "special"}


def _capture_and_retain_source_yaml(
    source_yaml: str | Path,
    *,
    working_directory: Path,
    attempt_directory: Path,
) -> dict[str, Any]:
    """Stable-read an explicit YAML input and retain those exact bytes locally."""

    if isinstance(source_yaml, Path):
        raw = str(source_yaml)
    elif isinstance(source_yaml, str):
        raw = source_yaml
    else:
        raise ExperimentRecordError("source YAML must be a string or Path")
    if not raw or "\x00" in raw:
        raise ExperimentRecordError("source YAML must be a non-empty path without NUL")
    local = _file_uri_path(raw)
    if local is None:
        raise ExperimentRecordError("source YAML must be a local regular file")
    path = _absolute_path(local, base=working_directory)
    payload, metadata, digest = _read_stable_regular_bytes(path, label="source YAML")
    retained_copy = _retain_attempt_bytes(
        attempt_directory,
        relative_path=_RETAINED_SOURCE_YAML_FILE,
        payload=payload,
    )
    if retained_copy["content_sha256"] != digest:
        raise ExperimentRecordError("retained source YAML bytes do not match their stable-read digest")
    return {
        "kind": "local_path",
        "path": _stored_path(path, working_directory),
        "status": "captured",
        "file_type": "regular",
        "mode": stat.S_IMODE(metadata.st_mode),
        "size_bytes": len(payload),
        "content_sha256": digest,
        "retained_copy": retained_copy,
    }


def _retain_attempt_bytes(
    attempt_directory: Path,
    *,
    relative_path: str,
    payload: bytes,
) -> dict[str, Any]:
    """Publish one immutable per-attempt input copy and return its identity."""

    target = _attempt_relative_path(attempt_directory, relative_path)
    _publish_immutable_bytes(target, payload)
    return {
        "path": relative_path,
        "file_type": "regular",
        "size_bytes": len(payload),
        "content_sha256": sha256_bytes(payload),
    }


def verify_attempt_record(
    attempt: ExperimentAttempt | str | Path,
    *,
    bundle_root: str | Path | None = None,
    source_root: str | Path | None = None,
    verify_runtime: bool = True,
    allowed_path_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Verify an attempt descriptor, retained source bundle, and lifecycle.

    The returned lifecycle status is ``incomplete`` when no terminal event is
    present; this function never treats absence of an end record as success.
    """

    directory, default_bundle_root = _attempt_location(attempt)
    document = _load_and_validate_attempt(directory)
    if isinstance(attempt, ExperimentAttempt) and document["attempt_id"] != attempt.attempt_id:
        raise ExperimentRecordError("attempt handle does not match its immutable attempt record")
    selected_bundle_root = _absolute_path(bundle_root) if bundle_root is not None else default_bundle_root
    if selected_bundle_root is None:
        raise ExperimentRecordError("bundle_root is required when verifying an attempt path")
    plan = document["plan"]
    _verify_retained_attempt_copy(
        directory,
        plan["source_yaml"],
        label="source YAML",
        expected_relative_path=_RETAINED_SOURCE_YAML_FILE,
    )
    _verify_retained_attempt_copy(
        directory,
        plan["resolved_selected_plan"],
        label="resolved selected plan",
        expected_relative_path=_RETAINED_RESOLVED_PLAN_FILE,
    )
    try:
        verify_source_snapshot(
            document["provenance"]["source_snapshot"],
            bundle_root=selected_bundle_root,
            source_root=source_root,
        )
        if verify_runtime:
            verify_runtime_identity(document["provenance"]["runtime_identity"])
    except ProvenanceError as exc:
        raise ExperimentRecordError(f"attempt provenance verification failed: {exc}") from exc
    if allowed_path_roots is not None:
        roots = tuple(_absolute_path(path) for path in allowed_path_roots)
        if not roots:
            raise ExperimentRecordError("allowed_path_roots must be non-empty when path verification is requested")
        working_directory = _absolute_path(document["runner"]["working_directory"])
        _verify_captured_reference(
            document["plan"]["source_yaml"],
            working_directory=working_directory,
            allowed_path_roots=roots,
        )
        _verify_persisted_plan(document["plan"]["resolved_selected_plan"], working_directory=working_directory, allowed_path_roots=roots)
    terminal = _load_terminal_event(directory, attempt_id=document["attempt_id"], command_id=None)
    return {
        "attempt": document,
        "lifecycle": terminal,
    }


def verify_command_record(
    command: ExperimentCommand | str | Path,
    *,
    allowed_path_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Verify one command lifecycle and optionally its captured local paths.

    Verification never reads a recorded local path unless the caller provides
    ``allowed_path_roots`` containing it.  This prevents a tampered record from
    turning verification into an arbitrary filesystem read.
    """

    if isinstance(command, ExperimentCommand):
        _validate_command_handle(command)
        directory = command.directory
        start = _load_and_validate_command_handle(command)
    else:
        directory = _absolute_path(command)
        start = _load_and_validate_command_start(directory)
    terminal = _load_terminal_event(directory, attempt_id=start["attempt_id"], command_id=start["command_id"])
    if allowed_path_roots is not None:
        roots = tuple(_absolute_path(path) for path in allowed_path_roots)
        if not roots:
            raise ExperimentRecordError("allowed_path_roots must be non-empty when path verification is requested")
        references = [
            *start["input_lineage"]["declared_inputs"],
            *start["input_lineage"]["inferred_inputs"],
        ]
        if terminal["status"] != "incomplete":
            event = terminal["event"]
            references.extend(event["output_lineage"]["declared_outputs"])
            references.extend(event["output_lineage"]["inferred_outputs"])
            checkpoint = event["output_lineage"]["announced_checkpoint"]
            if checkpoint is not None:
                references.append(checkpoint)
        working_directory = _absolute_path(start["working_directory"])
        for reference in references:
            _verify_captured_reference(reference, working_directory=working_directory, allowed_path_roots=roots)
    return {
        "command": start,
        "lifecycle": terminal,
    }


def list_attempt_records(attempts_directory: str | Path) -> list[Path]:
    """Return immutable attempt directories in stable UUID-path order."""

    root = _absolute_path(attempts_directory)
    try:
        entries = list(root.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ExperimentRecordError(f"cannot list attempt records: {type(exc).__name__}") from exc
    records: list[Path] = []
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise ExperimentRecordError(f"cannot stat attempt record entry: {type(exc).__name__}") from exc
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            descriptor = entry / ATTEMPT_RECORD_FILE
            try:
                descriptor_metadata = descriptor.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ExperimentRecordError(f"cannot stat attempt descriptor: {type(exc).__name__}") from exc
            if stat.S_ISREG(descriptor_metadata.st_mode):
                records.append(entry)
    return sorted(records, key=lambda path: path.name)


def _infer_argv_input_references(argv: Sequence[str], *, working_directory: Path) -> list[dict[str, Any]]:
    inferred: list[dict[str, Any]] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and token != "--":
            flag, value, consumed = _flag_value(argv, index)
            role = _reference_role_for_flag(flag)
            if role == "input" and value is not None and _looks_like_reference(value):
                record = capture_path_identity(value, working_directory=working_directory)
                record["inference"] = {"kind": "flag", "flag": flag}
                inferred.append(record)
            index += consumed
            continue
        if _looks_like_reference(token):
            record = capture_path_identity(token, working_directory=working_directory)
            record["inference"] = {"kind": "pathlike_positional"}
            inferred.append(record)
        index += 1
    return _deduplicate_references(inferred)


def _infer_argv_output_values(argv: Sequence[str]) -> list[str]:
    values: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and token != "--":
            flag, value, consumed = _flag_value(argv, index)
            if _reference_role_for_flag(flag) == "output" and value is not None and _looks_like_reference(value):
                values.append(value)
            index += consumed
            continue
        index += 1
    return list(dict.fromkeys(values))


def _flag_value(argv: Sequence[str], index: int) -> tuple[str, str | None, int]:
    token = argv[index]
    if "=" in token:
        flag, value = token.split("=", 1)
        return flag, value, 1
    if index + 1 < len(argv) and not argv[index + 1].startswith("-"):
        return token, argv[index + 1], 2
    return token, None, 1


def _reference_role_for_flag(flag: str) -> str | None:
    normalized = flag.lstrip("-").replace("_", "-").lower()
    parts = tuple(part for part in normalized.split("-") if part)
    if not parts or any(part in _NON_REFERENCE_FLAG_SUFFIXES for part in parts):
        return None
    if any(part in _OUTPUT_FLAG_PARTS for part in parts):
        return "output"
    if any(part in _INPUT_FLAG_PARTS for part in parts):
        return "input"
    return None


def _looks_like_reference(value: str) -> bool:
    if not value or value in {"true", "false", "none", "null"}:
        return False
    if _file_uri_path(value) is None:
        return True
    if value.startswith((".", "~", "/")) or "/" in value or "\\" in value:
        return True
    lowered = value.lower()
    return any(lowered.endswith(suffix) for suffix in _PATHLIKE_SUFFIXES)


def _file_uri_path(value: str) -> str | None:
    """Return a local path, or ``None`` when a value is an external URI."""

    if "://" not in value and not value.startswith("file:"):
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    if parsed.scheme.lower() != "file":
        return None
    if parsed.netloc not in {"", "localhost"} or parsed.query or parsed.fragment:
        return None
    if not parsed.path:
        return None
    return unquote(parsed.path)


def _external_uri_record(value: str) -> dict[str, Any]:
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower() or "unknown"
        hostname = parsed.hostname
        _ = parsed.port
        netloc = parsed.netloc.rsplit("@", 1)[-1]
        safe = urlunsplit((scheme, netloc, parsed.path, "", "")) if hostname or netloc else f"{scheme}:"
    except ValueError:
        scheme = "invalid"
        safe = "invalid:"
    return {
        "kind": "external_uri",
        "scheme": scheme,
        "status": "unverified_external_uri",
        "sanitized_reference_sha256": sha256_text(safe),
    }


def _hash_regular_file(path: Path, expected: os.stat_result) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _stat_key(before) != _stat_key(expected):
            raise _ReferenceChanged
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        if size != expected.st_size or _stat_key(after) != _stat_key(expected) or _stat_key(final) != _stat_key(expected):
            raise _ReferenceChanged
        return digest.hexdigest(), size
    finally:
        os.close(descriptor)


def _hash_directory(path: Path, expected: os.stat_result) -> tuple[str, int, int]:
    entries: list[dict[str, Any]] = []
    files = 0
    total_size = 0

    def visit(directory: Path, relative: str, directory_metadata: os.stat_result) -> None:
        nonlocal files, total_size
        entries.append({"path": relative, "type": "directory", "mode": stat.S_IMODE(directory_metadata.st_mode)})
        before = _stat_key(directory_metadata)
        with os.scandir(directory) as scanner:
            children = sorted(scanner, key=lambda entry: entry.name)
        for child in children:
            child_path = directory / child.name
            child_relative = child.name if relative == "." else f"{relative}/{child.name}"
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise _DirectoryContainsSymlink
            if stat.S_ISDIR(metadata.st_mode):
                visit(child_path, child_relative, metadata)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise _ReferenceChanged
            digest, size = _hash_regular_file(child_path, metadata)
            entries.append(
                {
                    "path": child_relative,
                    "type": "file",
                    "mode": stat.S_IMODE(metadata.st_mode),
                    "size_bytes": size,
                    "content_sha256": digest,
                }
            )
            files += 1
            total_size += size
        after = directory.lstat()
        if _stat_key(before) != _stat_key(after):
            raise _ReferenceChanged

    visit(path, ".", expected)
    final = path.lstat()
    if _stat_key(final) != _stat_key(expected):
        raise _ReferenceChanged
    return sha256_json({"entries": entries}), files, total_size


def _stat_key(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _cuda_placement(value: Sequence[str] | str | None, *, mode: str | None) -> dict[str, Any]:
    if mode is None:
        mode = "runner_assigned" if value is not None else "ambient"
    if mode not in {"ambient", "runner_assigned"}:
        raise ExperimentRecordError("cuda_placement_mode must be 'ambient' or 'runner_assigned'")
    if value is None:
        return {"mode": mode, "visible_devices": None}
    if isinstance(value, str):
        devices = [part.strip() for part in value.split(",") if part.strip()]
    else:
        devices = _string_list(value, field="cuda_visible_devices")
    return {"mode": mode, "visible_devices": devices}


def _declared_references(value: str | Path | Sequence[str | Path] | None, *, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value)]
    if not isinstance(value, Sequence):
        raise ExperimentRecordError(f"{field} must be a path or a sequence of paths")
    result: list[str] = []
    for item in value:
        if not isinstance(item, (str, Path)):
            raise ExperimentRecordError(f"{field} must contain only paths")
        result.append(str(item))
    return result


def _declared_coverage(references: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if all(reference.get("status") == "captured" for reference in references):
        return {
            "scope": "all caller-declared references",
            "status": "complete_for_declared_reference_list",
        }
    return {
        "scope": "all caller-declared references",
        "status": "contains_explicit_unverified_references",
    }


def _deduplicate_references(references: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for reference in references:
        result[canonical_json(reference)] = reference
    return [result[key] for key in sorted(result)]


def _verify_captured_reference(
    recorded: Mapping[str, Any],
    *,
    working_directory: Path,
    allowed_path_roots: Sequence[Path],
) -> None:
    if recorded.get("kind") != "local_path" or recorded.get("status") != "captured":
        return
    path = _path_from_stored_path(recorded.get("path"), working_directory=working_directory)
    if not any(_is_lexically_within(path, root) for root in allowed_path_roots):
        raise ExperimentRecordError(f"recorded local reference is outside allowed_path_roots: {path}")
    observed = capture_path_identity(path, working_directory=working_directory)
    expected = {key: value for key, value in recorded.items() if key not in {"inference", "retained_copy"}}
    if canonical_json(observed) != canonical_json(expected):
        raise ExperimentRecordError(f"captured local reference changed or is no longer safely readable: {path}")


def _verify_retained_attempt_copy(
    attempt_directory: Path,
    recorded: Mapping[str, Any],
    *,
    label: str,
    expected_relative_path: str,
) -> None:
    descriptor = _retained_copy_descriptor(
        recorded,
        label=label,
        expected_relative_path=expected_relative_path,
    )
    path = _attempt_relative_path(attempt_directory, expected_relative_path)
    payload, _metadata, digest = _read_stable_regular_bytes(path, label=f"retained {label} copy")
    if len(payload) != descriptor["size_bytes"] or digest != descriptor["content_sha256"]:
        raise ExperimentRecordError(f"retained {label} copy bytes do not match its recorded content identity: {path}")


def _retained_copy_descriptor(
    recorded: Mapping[str, Any],
    *,
    label: str,
    expected_relative_path: str,
) -> dict[str, Any]:
    descriptor = recorded.get("retained_copy")
    if not isinstance(descriptor, Mapping):
        raise ExperimentRecordError(f"attempt record has no retained {label} copy")
    if descriptor.get("path") != expected_relative_path or descriptor.get("file_type") != "regular":
        raise ExperimentRecordError(f"attempt record has malformed retained {label} copy")
    size = descriptor.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ExperimentRecordError(f"attempt record has invalid retained {label} copy size")
    try:
        digest = require_sha256(descriptor.get("content_sha256"), field=f"retained {label} copy content_sha256")
    except ValueError as exc:
        raise ExperimentRecordError(str(exc)) from exc
    return {
        "path": expected_relative_path,
        "file_type": "regular",
        "size_bytes": size,
        "content_sha256": digest,
    }


def _verify_persisted_plan(
    recorded: Mapping[str, Any],
    *,
    working_directory: Path,
    allowed_path_roots: Sequence[Path],
) -> None:
    path_record = recorded.get("path")
    if path_record is None:
        return
    if not isinstance(path_record, Mapping):
        raise ExperimentRecordError("resolved selected plan has malformed persisted path")
    path = _path_from_stored_path(path_record, working_directory=working_directory)
    if not any(_is_lexically_within(path, root) for root in allowed_path_roots):
        raise ExperimentRecordError(f"resolved selected plan is outside allowed_path_roots: {path}")
    observed = capture_path_identity(path, working_directory=working_directory)
    if observed.get("status") != "captured" or observed.get("file_type") != "regular":
        raise ExperimentRecordError(f"resolved selected plan is no longer a stable regular file: {path}")
    if observed.get("size_bytes") != recorded.get("size_bytes") or observed.get("content_sha256") != recorded.get("content_sha256"):
        raise ExperimentRecordError(f"resolved selected plan bytes changed: {path}")


def _path_from_stored_path(value: Any, *, working_directory: Path) -> Path:
    if not isinstance(value, Mapping):
        raise ExperimentRecordError("recorded local reference has malformed path")
    base = value.get("base")
    path_value = value.get("value")
    if not isinstance(path_value, str) or not path_value or "\x00" in path_value:
        raise ExperimentRecordError("recorded local reference has invalid path value")
    if base == "working_directory":
        return _absolute_path(path_value, base=working_directory)
    if base == "absolute":
        return _absolute_path(path_value)
    raise ExperimentRecordError("recorded local reference has invalid path base")


def _stored_path(path: Path, working_directory: Path) -> dict[str, str]:
    if _is_lexically_within(path, working_directory):
        relative = os.path.relpath(path, working_directory)
        return {"base": "working_directory", "value": relative}
    return {"base": "absolute", "value": str(path)}


def _attempt_relative_path(attempt_directory: Path, relative_path: str) -> Path:
    """Resolve one fixed retained-input location without escaping an attempt."""

    if not isinstance(relative_path, str) or not relative_path or "\x00" in relative_path:
        raise ExperimentRecordError("retained attempt path must be a non-empty relative path without NUL")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ExperimentRecordError("retained attempt path must stay inside its attempt directory")
    directory = _absolute_path(attempt_directory)
    target = _absolute_path(directory / relative)
    if not _is_lexically_within(target, directory):
        raise ExperimentRecordError("retained attempt path must stay inside its attempt directory")
    return target


def _attempt_location(attempt: ExperimentAttempt | str | Path) -> tuple[Path, Path | None]:
    if isinstance(attempt, ExperimentAttempt):
        return attempt.directory, attempt.bundle_root
    return _absolute_path(attempt), None


def _validate_attempt_handle(attempt: ExperimentAttempt) -> None:
    if not isinstance(attempt, ExperimentAttempt):
        raise ExperimentRecordError("attempt must be an ExperimentAttempt")
    _validate_uuid(attempt.attempt_id, field="attempt_id")
    _require_safe_directory(attempt.directory)


def _validate_command_handle(command: ExperimentCommand) -> None:
    if not isinstance(command, ExperimentCommand):
        raise ExperimentRecordError("command must be an ExperimentCommand")
    _validate_attempt_handle(command.attempt)
    _validate_uuid(command.command_id, field="command_id")
    _require_safe_directory(command.directory)


def _load_and_validate_attempt(directory: Path) -> dict[str, Any]:
    document = _load_immutable_json(directory / ATTEMPT_RECORD_FILE)
    _validate_record(document, record_type="attempt")
    _validate_uuid(document.get("attempt_id"), field="attempt_id")
    provenance = document.get("provenance")
    if not isinstance(provenance, Mapping) or "source_snapshot" not in provenance or "runtime_identity" not in provenance:
        raise ExperimentRecordError("attempt record has incomplete provenance")
    plan = document.get("plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("source_yaml"), Mapping):
        raise ExperimentRecordError("attempt record has no source YAML identity")
    if not isinstance(plan.get("resolved_selected_plan"), Mapping):
        raise ExperimentRecordError("attempt record has no resolved selected plan identity")
    _retained_copy_descriptor(
        plan["source_yaml"],
        label="source YAML",
        expected_relative_path=_RETAINED_SOURCE_YAML_FILE,
    )
    _retained_copy_descriptor(
        plan["resolved_selected_plan"],
        label="resolved selected plan",
        expected_relative_path=_RETAINED_RESOLVED_PLAN_FILE,
    )
    return document


def _load_and_validate_command_start(directory: Path) -> dict[str, Any]:
    document = _load_immutable_json(directory / "start.json")
    _validate_record(document, record_type="command_start")
    _validate_uuid(document.get("attempt_id"), field="attempt_id")
    _validate_uuid(document.get("command_id"), field="command_id")
    if not isinstance(document.get("argv"), list) or not all(isinstance(value, str) for value in document["argv"]):
        raise ExperimentRecordError("command start record has invalid argv")
    lineage = document.get("input_lineage")
    if not isinstance(lineage, Mapping):
        raise ExperimentRecordError("command start record has no input lineage")
    for key in ("declared_inputs", "inferred_inputs", "declared_output_baseline", "inferred_output_baseline"):
        if not isinstance(lineage.get(key), list):
            raise ExperimentRecordError(f"command start record has invalid {key}")
    return document


def _load_and_validate_command_handle(command: ExperimentCommand) -> dict[str, Any]:
    document = _load_and_validate_command_start(command.directory)
    if document["attempt_id"] != command.attempt.attempt_id or document["command_id"] != command.command_id:
        raise ExperimentRecordError("command handle does not match its immutable start record")
    return document


def _load_terminal_event(directory: Path, *, attempt_id: str, command_id: str | None) -> dict[str, Any]:
    found: list[Path] = []
    for filename in _TERMINAL_FILE_NAMES:
        path = directory / filename
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ExperimentRecordError(f"cannot stat terminal event: {type(exc).__name__}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ExperimentRecordError(f"terminal event must be a regular file: {path}")
        found.append(path)
    if not found:
        return {"status": "incomplete", "terminal_event": None}
    if len(found) != 1:
        raise ExperimentRecordError(f"record directory has multiple terminal events: {[path.name for path in found]}")
    event = _load_immutable_json(found[0])
    expected_type = "attempt_terminal" if command_id is None else "command_terminal"
    _validate_record(event, record_type=expected_type)
    if event.get("attempt_id") != attempt_id:
        raise ExperimentRecordError("terminal event attempt_id does not match its start record")
    if command_id is not None and event.get("command_id") != command_id:
        raise ExperimentRecordError("terminal event command_id does not match its start record")
    outcome = event.get("outcome")
    if outcome not in _TERMINAL_EVENTS:
        raise ExperimentRecordError("terminal event has an unknown outcome")
    filename, expected_event = _TERMINAL_EVENTS[outcome]
    if found[0].name != filename or event.get("event") != expected_event:
        raise ExperimentRecordError("terminal event filename and outcome do not agree")
    if outcome == "succeeded" and event.get("return_code") not in {None, 0}:
        raise ExperimentRecordError("successful terminal event has a nonzero return code")
    if command_id is not None:
        lineage = event.get("output_lineage")
        if (
            not isinstance(lineage, Mapping)
            or not isinstance(lineage.get("declared_outputs"), list)
            or not isinstance(lineage.get("inferred_outputs"), list)
        ):
            raise ExperimentRecordError("command terminal event has invalid output lineage")
    return {"status": outcome, "terminal_event": found[0].name, "event": event}


def _reject_existing_terminal_event(directory: Path) -> None:
    found: list[str] = []
    for filename in _TERMINAL_FILE_NAMES:
        path = directory / filename
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ExperimentRecordError(f"cannot stat terminal event: {type(exc).__name__}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ExperimentRecordError(f"terminal event must be a regular file: {path}")
        found.append(filename)
    if found:
        raise ExperimentRecordError(f"record already has an immutable terminal event: {found}")


def _command_directories(attempt_directory: Path) -> list[Path]:
    root = attempt_directory / COMMANDS_DIRECTORY
    try:
        entries = list(root.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ExperimentRecordError(f"cannot list command records: {type(exc).__name__}") from exc
    directories: list[Path] = []
    for entry in entries:
        try:
            metadata = entry.lstat()
        except OSError as exc:
            raise ExperimentRecordError(f"cannot stat command record: {type(exc).__name__}") from exc
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            directories.append(entry)
    return directories


def _validate_record(document: Any, *, record_type: str) -> None:
    if not isinstance(document, Mapping):
        raise ExperimentRecordError("record must be a JSON object")
    try:
        copied = json.loads(canonical_json(document))
    except (TypeError, ValueError) as exc:
        raise ExperimentRecordError(f"record is not canonical JSON data: {type(exc).__name__}") from exc
    if copied.get("schema") != EXPERIMENT_RECORD_SCHEMA or copied.get("schema_version") != EXPERIMENT_RECORD_SCHEMA_VERSION:
        raise ExperimentRecordError("record has an unsupported schema")
    if copied.get("record_type") != record_type:
        raise ExperimentRecordError(f"record type is not {record_type!r}")
    digest = copied.get("record_sha256")
    try:
        require_sha256(digest, field="record_sha256")
    except ValueError as exc:
        raise ExperimentRecordError(str(exc)) from exc
    unsigned = {key: value for key, value in copied.items() if key != "record_sha256"}
    if digest != sha256_json(unsigned):
        raise ExperimentRecordError("record_sha256 does not match immutable record contents")


def _with_record_digest(document: Mapping[str, Any]) -> dict[str, Any]:
    if "record_sha256" in document:
        raise ExperimentRecordError("record document already has record_sha256")
    copied = json.loads(canonical_json(document))
    copied["record_sha256"] = sha256_json(copied)
    return copied


def _load_immutable_json(path: Path) -> dict[str, Any]:
    payload = _read_regular_bytes(path)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentRecordError(f"immutable record is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExperimentRecordError(f"immutable record must be a JSON object: {path}")
    return value


def _read_regular_bytes(path: Path) -> bytes:
    payload, _metadata, _digest = _read_stable_regular_bytes(path, label="immutable record")
    return payload


def _read_stable_regular_bytes(path: Path, *, label: str) -> tuple[bytes, os.stat_result, str]:
    """Read one regular file without following links and prove its stable bytes."""

    if _first_symlink_ancestor(path) is not None:
        raise ExperimentRecordError(f"{label} must not have a symlink ancestor: {path}")
    try:
        before = path.lstat()
    except OSError as exc:
        raise ExperimentRecordError(f"cannot stat {label} {path}: {type(exc).__name__}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ExperimentRecordError(f"{label} must be a regular file without symlinks: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ExperimentRecordError(f"cannot safely open {label} {path}: {type(exc).__name__}") from exc
    try:
        current = os.fstat(descriptor)
        if not stat.S_ISREG(current.st_mode) or _stat_key(current) != _stat_key(before):
            raise ExperimentRecordError(f"{label} changed while opening: {path}")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                chunks.append(chunk)
        after = os.fstat(descriptor)
        final = path.lstat()
        if (
            _first_symlink_ancestor(path) is not None
            or _stat_key(after) != _stat_key(before)
            or _stat_key(final) != _stat_key(before)
        ):
            raise ExperimentRecordError(f"{label} changed while reading: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ExperimentRecordError(f"{label} changed while reading: {path}")
        return payload, before, digest.hexdigest()
    except OSError as exc:
        raise ExperimentRecordError(f"cannot safely read {label} {path}: {type(exc).__name__}") from exc
    finally:
        os.close(descriptor)


def _publish_immutable_json(path: Path, document: Mapping[str, Any]) -> None:
    payload = (canonical_json(document) + "\n").encode("utf-8")
    _publish_immutable_bytes(path, payload)


def _publish_immutable_bytes(path: Path, payload: bytes) -> None:
    parent = _ensure_safe_directory(path.parent)
    target = parent / path.name
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise ExperimentRecordError(f"immutable record already exists: {target}") from exc
        # The hard link reserves a never-before-used destination.  Replacing
        # our other name removes the temporary path without deleting bytes.
        os.replace(temporary, target)
    except Exception:
        _archive_partial(temporary)
        raise


def _archive_partial(path: Path) -> Path | None:
    """Archive a failed write; never delete an on-disk ``.partial`` file."""

    try:
        metadata = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(metadata.st_mode):
        return None
    archive = path.parent / "_archive" / "failed-experiment-record-writes"
    try:
        _ensure_safe_directory(archive)
        destination = archive / f"{path.name}.{uuid.uuid4().hex}"
        os.replace(path, destination)
    except OSError:
        return None
    return destination


def _write_creation_failure(directory: Path, attempt_id: str, error: BaseException) -> None:
    document = _with_record_digest(
        {
            "schema": EXPERIMENT_RECORD_SCHEMA,
            "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
            "record_type": "attempt_creation_failure",
            "attempt_id": attempt_id,
            "recorded_at": _utc_now(),
            "error": _error_record(error) or {"type": type(error).__name__},
        }
    )
    try:
        _publish_immutable_json(directory / "creation-error.json", document)
    except Exception:
        # The original failure is authoritative.  Any publisher partial has
        # already been archived by _publish_immutable_bytes.
        return


def _write_command_creation_failure(directory: Path, attempt_id: str, command_id: str, error: BaseException) -> None:
    document = _with_record_digest(
        {
            "schema": EXPERIMENT_RECORD_SCHEMA,
            "schema_version": EXPERIMENT_RECORD_SCHEMA_VERSION,
            "record_type": "command_creation_failure",
            "attempt_id": attempt_id,
            "command_id": command_id,
            "recorded_at": _utc_now(),
            "error": _error_record(error) or {"type": type(error).__name__},
        }
    )
    try:
        _publish_immutable_json(directory / "creation-error.json", document)
    except Exception:
        return


def _error_record(error: BaseException | None, *, return_code: int | None = None) -> dict[str, Any] | None:
    if error is None and return_code is None:
        return None
    value: dict[str, Any] = {}
    if error is not None:
        value["type"] = type(error).__name__
    if return_code is not None:
        value["return_code"] = return_code
    return value


def _create_unique_directory(parent: Path) -> tuple[str, Path]:
    for _attempt in range(32):
        identifier = str(uuid.uuid4())
        directory = parent / identifier
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            continue
        except OSError as exc:
            raise ExperimentRecordError(f"cannot create immutable record directory: {type(exc).__name__}") from exc
        _require_safe_directory(directory)
        return identifier, directory
    raise ExperimentRecordError("cannot allocate a unique immutable record UUID")


def _ensure_safe_directory(path: Path) -> Path:
    absolute = _absolute_path(path)
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                metadata = current.lstat()
            except OSError as exc:
                raise ExperimentRecordError(f"cannot create record directory {current}: {type(exc).__name__}") from exc
            else:
                metadata = current.lstat()
        except OSError as exc:
            raise ExperimentRecordError(f"cannot stat record directory {current}: {type(exc).__name__}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ExperimentRecordError(f"record directory must not be a symlink or non-directory: {current}")
    return absolute


def _require_safe_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ExperimentRecordError(f"cannot stat record directory {path}: {type(exc).__name__}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ExperimentRecordError(f"record directory must be a non-symlink directory: {path}")


def _absolute_path(value: str | Path, *, base: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        root = Path(base).expanduser() if base is not None else Path.cwd()
        path = root / path
    return Path(os.path.abspath(os.fspath(path)))


def _is_lexically_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _first_symlink_ancestor(path: Path) -> Path | None:
    """Return an existing symlink ancestor without resolving through it."""

    absolute = _absolute_path(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            return None
        except OSError:
            return current
        if stat.S_ISLNK(metadata.st_mode):
            return current
        if not stat.S_ISDIR(metadata.st_mode):
            return current
    return None


def _string_list(value: Sequence[str], *, field: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExperimentRecordError(f"{field} must be a sequence of strings")
    result = list(value)
    if any(not isinstance(item, str) or "\x00" in item for item in result):
        raise ExperimentRecordError(f"{field} must contain strings without NUL")
    return result


def _validate_uuid(value: Any, *, field: str) -> None:
    if not isinstance(value, str):
        raise ExperimentRecordError(f"{field} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ExperimentRecordError(f"{field} must be a UUID string") from exc
    if str(parsed) != value:
        raise ExperimentRecordError(f"{field} must use canonical lowercase UUID form")


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class _ReferenceChanged(Exception):
    pass


class _DirectoryContainsSymlink(Exception):
    pass


__all__ = [
    "ATTEMPT_RECORD_FILE",
    "COMMANDS_DIRECTORY",
    "EXPERIMENT_RECORD_SCHEMA",
    "EXPERIMENT_RECORD_SCHEMA_VERSION",
    "ExperimentAttempt",
    "ExperimentCommand",
    "ExperimentRecordError",
    "capture_path_identity",
    "complete_attempt_record",
    "complete_command_record",
    "create_attempt_record",
    "list_attempt_records",
    "start_command_record",
    "verify_attempt_record",
    "verify_command_record",
]
