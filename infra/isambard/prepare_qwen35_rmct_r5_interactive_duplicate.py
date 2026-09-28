#!/usr/bin/env python3
"""Prepare and run an isolated interactive duplicate of the uncapped r5 RMCT continuation.

This helper deliberately never submits, cancels, modifies, or reads the Slurm
queue.  It is for the *second* copy of the continuation only: the production
afterok chain remains in its original repository while this helper constructs a
new regular sibling repository with its own r5 output paths.

The duplicate is intentionally not a symlinked view of the production tree.
It copies the frozen remote Python sources, the two immutable data files, the
completed worker-parity sidecar, and the sealed r4 step-176 checkpoint.  It
then writes a fresh clone-local patience amendment that still references the
original immutable legacy receipt chain.  A shared source ``.venv`` interpreter
and the shared offline HF snapshot are used directly; neither is linked into
the clone.

Use ``prepare`` and ``validate`` on the login node.  Within an already-created
four-GPU ``interactive`` reservation, use ``smoke`` for one isolated optimiser
update, then ``run``.  ``run`` invokes the clone's unchanged r5 runner directly
for exactly one full window.  The external broker must inspect ``status`` and
obtain a fresh (at most eight-hour) interactive allocation before every later
window, so no continuation window begins near an allocation deadline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.rmct_convergence import patience
from experiments.rmct_convergence import plan as base_plan
from experiments.rmct_convergence_r5_patience import plan as r5
from infra.isambard import verify_rmct_convergence_r4_recovery_production_ready as r4_ready
from scripts.run_experiment import _argument_tokens
from ctm_data.adapters.mcq_bias.shared_qid_two_bias import (
    SEGMENT_DATAPOINTS,
    create_shared_qid_two_bias_setting,
)


PREPARATION_SCHEMA = "rmct-r5-interactive-duplicate-preparation-v1"
SMOKE_COMMAND_SCHEMA = "rmct-r5-interactive-duplicate-smoke-command-v1"
SMOKE_MARKER_SCHEMA = "rmct-r5-interactive-duplicate-smoke-started-v1"
SMOKE_SUCCESS_SCHEMA = "rmct-r5-interactive-duplicate-smoke-success-v1"
EXECUTION_RECEIPT_SCHEMA = "rmct-r5-interactive-duplicate-execution-v1"

CLONE_ARTIFACT_RELATIVE = Path("artifacts/rmct-r5-interactive-duplicate-20260908")
PREPARATION_RECEIPT_RELATIVE = CLONE_ARTIFACT_RELATIVE / "preparation.json"
CLONE_AMENDMENT_RELATIVE = CLONE_ARTIFACT_RELATIVE / "amendment.json"
SMOKE_DIRECTORY_RELATIVE = CLONE_ARTIFACT_RELATIVE / "smoke-s012"

SOURCE_DIRECTORIES = ("ctm", "ctm_data", "experiments", "infra", "scripts")
SOURCE_FILES = ("requirements.txt", "pyproject.toml", "uv.lock")
VOLATILE_DIRECTORY_NAMES = frozenset({"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"})
FORBIDDEN_GENERATION_FLAGS = frozenset(
    {"--max-new-tokens", "--max-tokens", "--max-output-tokens", "--max-completion-tokens"}
)
SMOKE_SETTING_FACTORY = "infra.isambard.prepare_qwen35_rmct_r5_interactive_duplicate:create_one_batch_smoke_setting"


class DuplicateError(ValueError):
    """The isolated duplicate is incomplete, mutable, or unsafe to run."""


class _OneBatchSmokeSetting:
    """Clone-only adapter that makes a real two-datum, one-update smoke legal.

    The canonical shared-QID setting deliberately permits only its fixed
    32-datum segment.  This adapter loads and verifies that exact segment
    first, then returns deep copies of its first deterministic two rows.  It
    is used only by the smoke command; production r5 continues to name the
    canonical factory and never reads the smoke output as a parent.
    """

    def __init__(self, base: Any) -> None:
        self._base = base
        self.name = f"{base.name}-r5-interactive-one-batch-smoke"
        self._selection: dict[str, Any] | None = None

    def load_datapoints(self, n_datapoints: int = 2, *, segment_index: int = 0, **kwargs: Any) -> list[dict[str, Any]]:
        if isinstance(n_datapoints, bool) or not isinstance(n_datapoints, int) or n_datapoints != 2:
            raise ValueError("r5 interactive smoke setting requires n_datapoints=2")
        rows = self._base.load_datapoints(
            n_datapoints=SEGMENT_DATAPOINTS,
            segment_index=segment_index,
            **kwargs,
        )
        if len(rows) != SEGMENT_DATAPOINTS:
            raise ValueError(
                f"canonical shared-QID smoke parent returned {len(rows)} rows, expected {SEGMENT_DATAPOINTS}"
            )
        selected = copy.deepcopy(rows[:2])
        self._selection = {
            "non_production": True,
            "parent_segment_datapoints": SEGMENT_DATAPOINTS,
            "selected_datapoints": 2,
            "segment_index": segment_index,
            "selection": "first_two_rows_of_verified_canonical_interleaved_segment",
            "question_ids": [row.get("question_id") for row in selected],
        }
        return selected

    def perturbations(self):
        return self._base.perturbations()

    def training_perturbation_indices(self):
        return self._base.training_perturbation_indices()

    def trait_classifier(self):
        return self._base.trait_classifier()

    def answer_parser(self):
        return self._base.answer_parser()

    def run_metadata(self) -> dict[str, Any]:
        metadata = copy.deepcopy(self._base.run_metadata())
        metadata["interactive_one_batch_smoke"] = copy.deepcopy(self._selection)
        return metadata

    def training_artifact_identity(self):
        identity = copy.deepcopy(self._base.training_artifact_identity())
        return identity


def create_one_batch_smoke_setting(**kwargs: Any) -> _OneBatchSmokeSetting:
    """Factory named only in the isolated smoke command, never production r5."""

    return _OneBatchSmokeSetting(create_shared_qid_two_bias_setting(**kwargs))


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root(value: str | Path, *, label: str) -> Path:
    supplied = Path(value)
    if supplied.is_symlink() or not supplied.is_dir():
        raise DuplicateError(f"{label} must be a regular directory: {supplied}")
    resolved = supplied.resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise DuplicateError(f"{label} must resolve to a regular directory: {supplied}")
    return resolved


def _regular_file(value: str | Path, *, label: str) -> Path:
    supplied = Path(value)
    if supplied.is_symlink() or not supplied.is_file():
        raise DuplicateError(f"{label} must be a regular file: {supplied}")
    resolved = supplied.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise DuplicateError(f"{label} must resolve to a regular file: {supplied}")
    return resolved


def _under(root: Path, relative: str | Path, *, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DuplicateError(f"{label} escapes repository root: {path}") from exc
    return path


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label=label)
    return {"path": str(regular), "sha256": _sha256(regular), "size_bytes": regular.stat().st_size}


def _relative_file_identity(root: Path, path: Path, *, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label=label)
    try:
        relative = regular.relative_to(root)
    except ValueError as exc:
        raise DuplicateError(f"{label} escapes repository root: {regular}") from exc
    return {"path": str(relative), "sha256": _sha256(regular), "size_bytes": regular.stat().st_size}


def _tree_inventory(root: Path, *, label: str) -> dict[str, dict[str, Any]]:
    """Return a stable regular-file inventory and reject every symlink."""

    directory = _root(root, label=label)
    result: dict[str, dict[str, Any]] = {}
    for current, directory_names, file_names in os.walk(directory, followlinks=False):
        current_path = Path(current)
        retained: list[str] = []
        for name in sorted(directory_names):
            child = current_path / name
            if name in VOLATILE_DIRECTORY_NAMES:
                continue
            if child.is_symlink() or not child.is_dir():
                raise DuplicateError(f"{label} contains a linked or non-directory child: {child}")
            retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            child = current_path / name
            if child.is_symlink() or not child.is_file():
                raise DuplicateError(f"{label} contains a linked or non-regular file: {child}")
            result[str(child.relative_to(directory))] = {
                "sha256": _sha256(child),
                "size_bytes": child.stat().st_size,
            }
    if not result:
        raise DuplicateError(f"{label} must not be empty: {directory}")
    return result


def _copy_regular_tree(source: Path, destination: Path, *, label: str) -> dict[str, dict[str, Any]]:
    """Copy one tree without following links, then prove the bytes match."""

    source = _root(source, label=f"source {label}")
    if destination.exists() or destination.is_symlink():
        raise DuplicateError(f"destination {label} already exists: {destination}")
    before = _tree_inventory(source, label=f"source {label}")
    destination.mkdir(parents=True, mode=0o700)
    for relative in sorted(before):
        source_file = source / relative
        target_file = destination / relative
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file, follow_symlinks=False)
        if target_file.is_symlink() or not target_file.is_file():
            raise DuplicateError(f"copy produced a non-regular {label} file: {target_file}")
    after = _tree_inventory(source, label=f"source {label}")
    copied = _tree_inventory(destination, label=f"copied {label}")
    if before != after:
        raise DuplicateError(f"source {label} changed while it was being copied")
    if copied != before:
        raise DuplicateError(f"copied {label} differs from its source")
    return copied


def _copy_regular_file(source: Path, destination: Path, *, label: str) -> dict[str, Any]:
    source = _regular_file(source, label=f"source {label}")
    if destination.exists() or destination.is_symlink():
        raise DuplicateError(f"destination {label} already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=False)
    if destination.is_symlink() or not destination.is_file():
        raise DuplicateError(f"copy produced a non-regular {label} file: {destination}")
    source_identity = _file_identity(source, label=f"source {label}")
    target_identity = _file_identity(destination, label=f"copied {label}")
    if source_identity["sha256"] != target_identity["sha256"] or source_identity["size_bytes"] != target_identity["size_bytes"]:
        raise DuplicateError(f"copied {label} differs from its source")
    return target_identity


def _source_inventory(root: Path) -> dict[str, dict[str, Any]]:
    root = _root(root, label="source repository")
    result: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_DIRECTORIES:
        directory = _under(root, relative, label=f"source directory {relative}")
        inventory = _tree_inventory(directory, label=f"source directory {relative}")
        result.update({f"{relative}/{name}": value for name, value in inventory.items()})
    for relative in SOURCE_FILES:
        path = _under(root, relative, label=f"source file {relative}")
        identity = _relative_file_identity(root, path, label=f"source file {relative}")
        result[relative] = {"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}
    return dict(sorted(result.items()))


def _copy_sources(source: Path, destination: Path) -> dict[str, dict[str, Any]]:
    source = _root(source, label="source repository")
    destination = _root(destination, label="duplicate repository")
    for relative in SOURCE_DIRECTORIES:
        _copy_regular_tree(source / relative, destination / relative, label=f"source directory {relative}")
    for relative in SOURCE_FILES:
        _copy_regular_file(source / relative, destination / relative, label=f"source file {relative}")
    expected = _source_inventory(source)
    actual = _source_inventory(destination)
    if actual != expected:
        raise DuplicateError("duplicate runnable source inventory differs from frozen source")
    return actual


def _parent_checkpoint_relative(root: Path) -> Path:
    checkpoint = r5.parent_checkpoint_path(root)
    try:
        return checkpoint.resolve().relative_to(root)
    except ValueError as exc:  # pragma: no cover - plan contract makes this impossible
        raise DuplicateError(f"r5 parent checkpoint escapes source repository: {checkpoint}") from exc


def _dependency_paths(root: Path) -> dict[str, tuple[str, Path, bool]]:
    """Return exactly the non-source inputs used by the clone's production argv."""

    parent_relative = _parent_checkpoint_relative(root)
    return {
        "data": (base_plan.DATA_PATH, _under(root, base_plan.DATA_PATH, label="source RMCT data"), False),
        "manifest": (base_plan.MANIFEST_PATH, _under(root, base_plan.MANIFEST_PATH, label="source RMCT manifest"), False),
        "worker_parity": (
            str(Path(base_plan.WORKER_PARITY_ATTESTATION).parent),
            _under(root, Path(base_plan.WORKER_PARITY_ATTESTATION).parent, label="source worker-parity sidecar"),
            True,
        ),
        "parent_checkpoint": (
            str(parent_relative),
            _under(root, parent_relative, label="sealed r4 step-176 parent checkpoint"),
            True,
        ),
    }


def _dependency_inventory(root: Path) -> dict[str, Any]:
    root = _root(root, label="repository")
    result: dict[str, Any] = {}
    for name, (relative, path, is_tree) in _dependency_paths(root).items():
        if is_tree:
            result[name] = {"relative_path": relative, "files": _tree_inventory(path, label=f"{name} dependency")}
        else:
            identity = _relative_file_identity(root, path, label=f"{name} dependency")
            result[name] = {"relative_path": relative, "file": identity}
    return result


def _copy_dependencies(source: Path, duplicate: Path) -> None:
    for name, (relative, source_path, is_tree) in _dependency_paths(source).items():
        target = _under(duplicate, relative, label=f"clone {name} dependency")
        if is_tree:
            _copy_regular_tree(source_path, target, label=f"{name} dependency")
        else:
            _copy_regular_file(source_path, target, label=f"{name} dependency")
    if _dependency_inventory(source) != _dependency_inventory(duplicate):
        raise DuplicateError("duplicate immutable dependency inventory differs from frozen source")


def _runtime_identity(value: str | Path, *, source: Path) -> dict[str, Any]:
    """Identify a source runtime launcher without requiring it to be regular.

    A standard virtualenv launcher is normally a symlink.  It is deliberately
    *not* copied into the clone: direct invocation is safe, whereas putting a
    symlink at clone/.venv/bin/python would violate the frozen sbatch guard.
    """

    raw = Path(value).expanduser()
    absolute = raw if raw.is_absolute() else (Path.cwd() / raw)
    absolute = absolute.absolute()
    try:
        absolute.relative_to(source)
    except ValueError as exc:
        raise DuplicateError("runtime Python launcher must be supplied from the frozen source repository") from exc
    if not absolute.exists() or not os.access(absolute, os.X_OK):
        raise DuplicateError(f"runtime Python launcher is not executable: {absolute}")
    resolved = absolute.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise DuplicateError(f"runtime Python launcher does not resolve to an executable file: {absolute}")
    return {
        "launcher_path": str(absolute),
        "launcher_is_symlink": absolute.is_symlink(),
        "resolved_path": str(resolved),
        "resolved_sha256": _sha256(resolved),
    }


def _json(path: Path, *, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label=label)
    try:
        value = json.loads(regular.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DuplicateError(f"{label} is not valid JSON: {regular}") from exc
    if not isinstance(value, dict):
        raise DuplicateError(f"{label} must contain a JSON object: {regular}")
    return value


def _immutable(path: Path, document: Mapping[str, Any], *, label: str) -> str:
    payload = _canonical(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise DuplicateError(f"{label} parent must be a regular directory: {path.parent}")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise DuplicateError(f"refusing to overwrite a different immutable {label}: {path}")
        return "resumed"
    return "written"


def _content_addressed(directory: Path, *, stem: str, document: Mapping[str, Any], label: str) -> Path:
    payload = _canonical(document)
    digest = hashlib.sha256(payload).hexdigest()
    path = directory / f"{stem}-{digest}.json"
    _immutable(path, document, label=label)
    return path


def _source_amendment(source: Path, supplied: str | Path | None) -> tuple[Path, dict[str, Any]]:
    path = (
        _regular_file(supplied, label="source r5 amendment")
        if supplied is not None
        else _regular_file(source / patience.AMENDMENT_RELATIVE, label="source r5 amendment")
    )
    try:
        document = patience.verify_amendment(source, path)
    except Exception as exc:
        raise DuplicateError(f"frozen source r5 amendment does not replay: {exc}") from exc
    return path, document


def _validate_parent_checkpoint(root: Path) -> dict[str, Any]:
    try:
        return r4_ready._strict_checkpoint_identity(
            root,
            r5.parent_checkpoint_path(root),
            expected_segment_index=r5.PARENT_SEGMENT_INDEX,
            label="sealed r4 step-176 parent",
        )
    except Exception as exc:
        raise DuplicateError(f"sealed r4 step-176 parent is not strictly resumable: {exc}") from exc


def _preparation_document(
    *,
    source: Path,
    duplicate: Path,
    runtime_python: str | Path,
    source_amendment: Path,
    source_amendment_document: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = source_amendment_document.get("legacy_terminal_receipt")
    if not isinstance(terminal, Mapping) or not isinstance(terminal.get("path"), str):
        raise DuplicateError("source r5 amendment lacks its legacy terminal receipt")
    legacy_terminal = _regular_file(terminal["path"], label="original legacy terminal receipt")
    clone_amendment = duplicate / CLONE_AMENDMENT_RELATIVE
    return {
        "schema": PREPARATION_SCHEMA,
        "source_repository": str(source),
        "duplicate_repository": str(duplicate),
        "isolation": {
            "source_is_symlinked": False,
            "duplicate_is_symlinked": False,
            "duplicate_is_sibling": duplicate.parent == source.parent,
            "clone_venv_present": (duplicate / ".venv").exists() or (duplicate / ".venv").is_symlink(),
            "queued_source_modified": False,
        },
        "runtime_python": _runtime_identity(runtime_python, source=source),
        "source_r5_amendment": _file_identity(source_amendment, label="source r5 amendment"),
        "original_legacy_terminal_receipt": _file_identity(legacy_terminal, label="original legacy terminal receipt"),
        "runnable_source_files": _source_inventory(source),
        "duplicate_runnable_source_files": _source_inventory(duplicate),
        "source_dependencies": _dependency_inventory(source),
        "duplicate_dependencies": _dependency_inventory(duplicate),
        "duplicate_parent_checkpoint": _validate_parent_checkpoint(duplicate),
        "duplicate_amendment": _file_identity(clone_amendment, label="clone-local r5 amendment"),
        "generation": {"output_token_cap": None, "termination": "model_eos_only"},
        "interactive_contract": {
            "reservation": "interactive",
            "nodes": 1,
            "gpus": 4,
            "maximum_allocation_walltime": "08:00:00",
            "queue_chain_cancelled": False,
            "clone_submitter_forbidden": True,
            "fresh_worker_parity_probe_forbidden": True,
            "worker_parity_resume_only": True,
        },
    }


def _require_sibling(source: Path, duplicate: Path) -> None:
    if duplicate == source:
        raise DuplicateError("duplicate repository must not be the source repository")
    if duplicate.parent != source.parent:
        raise DuplicateError(
            f"duplicate repository must be a fresh sibling of source; source={source}, duplicate={duplicate}"
        )


def prepare_duplicate(
    *,
    source_repository: str | Path,
    duplicate_repository: str | Path,
    runtime_python: str | Path,
    source_amendment: str | Path | None = None,
) -> dict[str, Any]:
    """Create one new duplicate root; never overwrite or alter an existing root."""

    source = _root(source_repository, label="source repository")
    duplicate = Path(duplicate_repository).expanduser()
    if not duplicate.is_absolute():
        duplicate = (Path.cwd() / duplicate).absolute()
    if duplicate.exists() or duplicate.is_symlink():
        raise DuplicateError(f"duplicate repository already exists; refusing to reuse or overwrite it: {duplicate}")
    if duplicate.parent.is_symlink() or not duplicate.parent.is_dir():
        raise DuplicateError(f"duplicate repository parent must be a regular directory: {duplicate.parent}")
    _require_sibling(source, duplicate)
    _runtime_identity(runtime_python, source=source)
    original_amendment, original_document = _source_amendment(source, source_amendment)

    # Stage all byte copies before publishing the final root.  The fresh
    # amendment cannot be made in staging because it intentionally binds
    # absolute clone source paths, so an interruption after publication leaves
    # an inspectable partial root rather than an overwritten one.
    staging = Path(tempfile.mkdtemp(prefix=f".{duplicate.name}.partial-", dir=duplicate.parent))
    try:
        _copy_sources(source, staging)
        _copy_dependencies(source, staging)
        if (staging / ".venv").exists() or (staging / ".venv").is_symlink():
            raise DuplicateError("duplicate source copy unexpectedly contains .venv")
        try:
            os.rename(staging, duplicate)
        except FileExistsError as exc:
            raise DuplicateError(f"duplicate repository appeared while preparing; refusing to overwrite it: {duplicate}") from exc
    except Exception:
        # Staging is deliberately retained for audit/recovery; do not delete it.
        raise

    duplicate = _root(duplicate, label="duplicate repository")
    _require_sibling(source, duplicate)
    terminal = original_document.get("legacy_terminal_receipt")
    if not isinstance(terminal, Mapping) or not isinstance(terminal.get("path"), str):
        raise DuplicateError("source r5 amendment lacks original legacy terminal receipt path")
    try:
        amendment_result = patience.write_amendment(
            duplicate,
            terminal_receipt=Path(terminal["path"]),
            output=duplicate / CLONE_AMENDMENT_RELATIVE,
        )
    except Exception as exc:
        raise DuplicateError(f"could not create the clone-local patience amendment: {exc}") from exc
    try:
        patience.verify_amendment(duplicate, duplicate / CLONE_AMENDMENT_RELATIVE)
    except Exception as exc:
        raise DuplicateError(f"clone-local patience amendment does not replay: {exc}") from exc

    receipt = _preparation_document(
        source=source,
        duplicate=duplicate,
        runtime_python=runtime_python,
        source_amendment=original_amendment,
        source_amendment_document=original_document,
    )
    receipt_path = duplicate / PREPARATION_RECEIPT_RELATIVE
    receipt_status = _immutable(receipt_path, receipt, label="interactive duplicate preparation receipt")
    return {
        "status": "prepared",
        "duplicate_repository": str(duplicate),
        "preparation_receipt": str(receipt_path),
        "preparation_receipt_status": receipt_status,
        "clone_amendment": amendment_result,
    }


def validate_duplicate(
    *,
    source_repository: str | Path,
    duplicate_repository: str | Path,
    runtime_python: str | Path,
    source_amendment: str | Path | None = None,
    model_snapshot: str | Path | None = None,
) -> dict[str, Any]:
    """Replay all clone custody checks without generating or writing model output."""

    source = _root(source_repository, label="source repository")
    duplicate = _root(duplicate_repository, label="duplicate repository")
    _require_sibling(source, duplicate)
    original_amendment, original_document = _source_amendment(source, source_amendment)
    clone_amendment = duplicate / CLONE_AMENDMENT_RELATIVE
    try:
        patience.verify_amendment(duplicate, clone_amendment)
    except Exception as exc:
        raise DuplicateError(f"clone-local patience amendment does not replay: {exc}") from exc
    expected = _preparation_document(
        source=source,
        duplicate=duplicate,
        runtime_python=runtime_python,
        source_amendment=original_amendment,
        source_amendment_document=original_document,
    )
    receipt_path = duplicate / PREPARATION_RECEIPT_RELATIVE
    recorded = _json(receipt_path, label="interactive duplicate preparation receipt")
    if recorded != expected:
        raise DuplicateError("interactive duplicate preparation receipt no longer matches frozen source or clone custody")
    output: dict[str, Any] = {
        "status": "validated",
        "duplicate_repository": str(duplicate),
        "preparation_receipt": str(receipt_path),
        "clone_amendment": str(clone_amendment),
        "generation": {"output_token_cap": None, "termination": "model_eos_only"},
    }
    if model_snapshot is not None:
        snapshot = _snapshot(model_snapshot)
        try:
            command = r5.command_attestation(duplicate, r5.START_SEGMENT_INDEX, model_snapshot=snapshot)
        except Exception as exc:
            raise DuplicateError(f"clone r5 production command cannot be rendered: {exc}") from exc
        _assert_uncapped_argv(command["argv"], label="clone r5 production command")
        output["r5_segment_12_command"] = {
            "run_name": command["segment"]["run_name"],
            "optimizer_step_start": command["segment"]["optimizer_step_start"],
            "optimizer_step_end": command["segment"]["optimizer_step_end"],
            "output_token_cap": command["environment_contract"]["output_token_cap"],
        }
    return output


def duplicate_status(
    *,
    source_repository: str | Path,
    duplicate_repository: str | Path,
    source_amendment: str | Path | None = None,
) -> dict[str, Any]:
    """Return the one clone-local next action without allocating GPUs or writing."""

    source = _root(source_repository, label="source repository")
    duplicate = _root(duplicate_repository, label="duplicate repository")
    _require_sibling(source, duplicate)
    original_amendment, _ = _source_amendment(source, source_amendment)
    clone_amendment = duplicate / CLONE_AMENDMENT_RELATIVE
    try:
        patience.verify_amendment(duplicate, clone_amendment)
    except Exception as exc:
        raise DuplicateError(f"clone-local patience amendment does not replay: {exc}") from exc
    preparation = _json(duplicate / PREPARATION_RECEIPT_RELATIVE, label="interactive duplicate preparation receipt")
    if (
        preparation.get("schema") != PREPARATION_SCHEMA
        or preparation.get("source_repository") != str(source)
        or preparation.get("duplicate_repository") != str(duplicate)
        or preparation.get("source_r5_amendment") != _file_identity(original_amendment, label="source r5 amendment")
        or preparation.get("duplicate_amendment") != _file_identity(clone_amendment, label="clone-local r5 amendment")
    ):
        raise DuplicateError("interactive duplicate preparation receipt has inconsistent source or amendment identity")
    status, index, detail = _next_pending_segment(duplicate)
    return {
        "status": status,
        "next_segment_index": index if status == "pending" else None,
        "terminal_segment_index": index if status == "terminal" else None,
        "detail": detail,
        "clone_amendment": str(clone_amendment),
        "queued_source_modified": False,
    }


def _snapshot(value: str | Path) -> Path:
    snapshot = Path(value).expanduser().resolve()
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != r5.BASE_SNAPSHOT:
        raise DuplicateError(f"model snapshot must be the regular pinned Qwen directory: {snapshot}")
    config = snapshot / "config.json"
    if not config.is_file():
        raise DuplicateError(f"model snapshot lacks config.json: {snapshot}")
    # Hugging Face snapshots normally hard-link or symlink small files into the
    # model cache's content-addressed blobs directory.  The production r5 plan
    # permits that canonical layout, so accept it while rejecting an arbitrary
    # external config target.
    resolved_config = config.resolve()
    blobs = snapshot.parent.parent / "blobs"
    try:
        resolved_config.relative_to(snapshot)
    except ValueError:
        try:
            resolved_config.relative_to(blobs)
        except ValueError as exc:
            raise DuplicateError(f"model snapshot config.json resolves outside its canonical HF cache: {config}") from exc
    if not resolved_config.is_file() or resolved_config.is_symlink():
        raise DuplicateError(f"model snapshot config.json does not resolve to a regular file: {config}")
    return snapshot


def _assert_uncapped_argv(argv: Iterable[Any], *, label: str) -> None:
    tokens = [str(value) for value in argv]
    forbidden = sorted(FORBIDDEN_GENERATION_FLAGS.intersection(tokens))
    if forbidden:
        raise DuplicateError(f"{label} contains forbidden generation-cap flags: {forbidden}")
    if "--no-max-new-tokens" not in tokens:
        raise DuplicateError(f"{label} must explicitly request EOS-only generation")


def _require_four_gpus() -> None:
    tokens = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")]
    if len(tokens) != 4 or any(not item or item in {"-1", "NoDevFiles"} for item in tokens) or len(set(tokens)) != 4:
        raise DuplicateError("interactive r5 duplicate requires exactly four distinct Slurm-visible GPUs")


def _require_offline_snapshot(snapshot: Path) -> None:
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise DuplicateError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise DuplicateError("HF_HOME is required for the pinned offline snapshot")
    expected = Path(hf_home).resolve() / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / r5.BASE_SNAPSHOT
    if expected != snapshot:
        raise DuplicateError(f"model snapshot must be the HF_HOME pinned offline snapshot: expected {expected}, got {snapshot}")


def _require_interactive_allocation(*, minimum_remaining_seconds: int) -> int:
    reservation = os.environ.get("SLURM_JOB_RESERVATION") or os.environ.get("SLURM_RESERVATION")
    if reservation != "interactive":
        raise DuplicateError("run requires a Slurm allocation created with --reservation=interactive")
    if not os.environ.get("SLURM_JOB_ID"):
        raise DuplicateError("run requires an active Slurm interactive allocation")
    raw_end = os.environ.get("SLURM_JOB_END_TIME")
    if raw_end is None:
        raise DuplicateError("SLURM_JOB_END_TIME is required to avoid beginning a window near allocation expiry")
    try:
        end = int(raw_end)
    except ValueError as exc:
        raise DuplicateError(f"SLURM_JOB_END_TIME is invalid: {raw_end!r}") from exc
    remaining = end - int(time.time())
    if remaining <= 0:
        raise DuplicateError("interactive allocation has already expired")
    if minimum_remaining_seconds < 0:
        raise DuplicateError("minimum remaining seconds must be non-negative")
    return remaining


def _runtime_launcher(value: str | Path, *, source: Path, recorded: Mapping[str, Any]) -> str:
    actual = _runtime_identity(value, source=source)
    if dict(recorded) != actual:
        raise DuplicateError("source runtime Python differs from the immutable duplicate preparation receipt")
    return str(actual["launcher_path"])


def _run_environment(duplicate: Path) -> dict[str, str]:
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(duplicate) if not existing else f"{duplicate}{os.pathsep}{existing}"
    return environment


def _worker_parity_resume(*, python: str, duplicate: Path, snapshot: Path) -> None:
    """Read-only sidecar validation only; never invoke the capped fresh probe."""

    helper = duplicate / "infra/isambard/preflight_qwen35_rmct_convergence_worker_parity.py"
    output = duplicate / Path(base_plan.WORKER_PARITY_ATTESTATION).parent
    _regular_file(helper, label="clone worker-parity helper")
    _root(output, label="clone worker-parity sidecar")
    command = [python, str(helper), "--output-dir", str(output), "--model-snapshot", str(snapshot), "--resume"]
    completed = subprocess.run(command, cwd=duplicate, env=_run_environment(duplicate), check=False)
    if completed.returncode != 0:
        raise DuplicateError(f"read-only worker-parity sidecar validation exited with status {completed.returncode}")


def _smoke_run_name() -> str:
    return f"{r5.RUN_PREFIX}-interactive-smoke-s{r5.START_SEGMENT_INDEX + 1:03d}"


def smoke_args(duplicate: Path, *, snapshot: Path) -> dict[str, Any]:
    """One real update that preserves r5 sampling/topology and has no token cap."""

    args = r5.segment_args(duplicate, r5.START_SEGMENT_INDEX, model_snapshot=snapshot)
    args["run_name"] = _smoke_run_name()
    args["n_datapoints"] = 2
    args["load_config"] = {"n_datapoints": 2, "segment_index": r5.START_SEGMENT_INDEX}
    args["checkpoint_every"] = 1
    args["setting_factory"] = SMOKE_SETTING_FACTORY
    if args.get("n_ref_rollouts") != 96 or args.get("n_train_rollouts") != 96:
        raise DuplicateError("interactive smoke must preserve the 96/96 production rollout counts")
    if args.get("local_phase_shared") is not True or args.get("local_training_gpus") != "all" or args.get("local_rollout_gpus") != "all":
        raise DuplicateError("interactive smoke must preserve the production four-lane phase-shared topology")
    if args.get("no_max_new_tokens") is not True:
        raise DuplicateError("interactive smoke must explicitly use EOS-only generation")
    for key in ("max_new_tokens", "max_tokens", "max_output_tokens", "max_completion_tokens"):
        if key in args:
            raise DuplicateError(f"interactive smoke contains forbidden generation cap key: {key}")
    return args


def _smoke_paths(duplicate: Path) -> dict[str, Path]:
    directory = duplicate / SMOKE_DIRECTORY_RELATIVE
    run_name = _smoke_run_name()
    checkpoint = duplicate / "logs" / r5.CONDITION_NAME / run_name / "checkpoints" / f"{r5.CONDITION_NAME}_{run_name}"
    return {
        "directory": directory,
        "command": directory / "command.json",
        "marker": directory / "training-started.json",
        "success": directory / "success.json",
        "checkpoint": checkpoint,
    }


def _smoke_command(*, duplicate: Path, python: str, snapshot: Path, amendment: Path) -> dict[str, Any]:
    args = smoke_args(duplicate, snapshot=snapshot)
    argv = [python, str(duplicate / "scripts/train_rlct.py"), *_argument_tokens(args)]
    _assert_uncapped_argv(argv, label="interactive r5 smoke command")
    paths = _smoke_paths(duplicate)
    return {
        "schema": SMOKE_COMMAND_SCHEMA,
        "mode": "one_optimizer_update_before_r5_production",
        "source_parent": {
            "checkpoint": _relative_file_identity(
                duplicate,
                duplicate / _parent_checkpoint_relative(duplicate) / "manifest.json",
                label="clone smoke parent checkpoint manifest",
            ),
            "optimizer_step": (r5.PARENT_SEGMENT_INDEX + 1) * r5.UPDATES_PER_SEGMENT,
        },
        "clone_amendment": _file_identity(amendment, label="clone-local r5 amendment"),
        "model_snapshot": {"path": str(snapshot), "config_sha256": _sha256(snapshot / "config.json")},
        "run_name": _smoke_run_name(),
        "setting_factory": SMOKE_SETTING_FACTORY,
        "smoke_factory_source": _relative_file_identity(
            duplicate,
            duplicate / "infra/isambard/prepare_qwen35_rmct_r5_interactive_duplicate.py",
            label="clone interactive smoke factory source",
        ),
        "checkpoint": str(paths["checkpoint"]),
        "argv": argv,
        "generation": {"output_token_cap": None, "termination": "model_eos_only"},
        "scope": {
            "optimizer_updates": 1,
            "n_datapoints": 2,
            "checkpoint_every": 1,
            "production_parent_not_replaced": True,
            "smoke_output_not_a_production_parent": True,
        },
    }


def _checkpoint_inventory(root: Path, checkpoint: Path, *, label: str) -> dict[str, dict[str, Any]]:
    checkpoint = _root(checkpoint, label=label)
    try:
        checkpoint.relative_to(root)
    except ValueError as exc:
        raise DuplicateError(f"{label} escapes clone repository: {checkpoint}") from exc
    return _tree_inventory(checkpoint, label=label)


def run_smoke(
    *,
    source_repository: str | Path,
    duplicate_repository: str | Path,
    runtime_python: str | Path,
    model_snapshot: str | Path,
    source_amendment: str | Path | None = None,
    yes: bool,
) -> dict[str, Any]:
    if not yes:
        raise DuplicateError("interactive smoke requires --yes")
    validation = validate_duplicate(
        source_repository=source_repository,
        duplicate_repository=duplicate_repository,
        runtime_python=runtime_python,
        source_amendment=source_amendment,
        model_snapshot=model_snapshot,
    )
    source = _root(source_repository, label="source repository")
    duplicate = _root(duplicate_repository, label="duplicate repository")
    snapshot = _snapshot(model_snapshot)
    _require_four_gpus()
    _require_offline_snapshot(snapshot)
    _require_interactive_allocation(minimum_remaining_seconds=0)
    receipt = _json(duplicate / PREPARATION_RECEIPT_RELATIVE, label="interactive duplicate preparation receipt")
    python = _runtime_launcher(runtime_python, source=source, recorded=receipt["runtime_python"])
    _worker_parity_resume(python=python, duplicate=duplicate, snapshot=snapshot)
    amendment = duplicate / CLONE_AMENDMENT_RELATIVE
    paths = _smoke_paths(duplicate)
    command = _smoke_command(duplicate=duplicate, python=python, snapshot=snapshot, amendment=amendment)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    _immutable(paths["command"], command, label="interactive smoke command")
    if paths["success"].exists() or paths["success"].is_symlink():
        success = _json(paths["success"], label="interactive smoke success receipt")
        expected = {
            "schema": SMOKE_SUCCESS_SCHEMA,
            "command": _file_identity(paths["command"], label="interactive smoke command"),
            "checkpoint_files": _checkpoint_inventory(duplicate, paths["checkpoint"], label="interactive smoke checkpoint"),
            "optimizer_step": (r5.PARENT_SEGMENT_INDEX + 1) * r5.UPDATES_PER_SEGMENT + 1,
            "production_parent_not_replaced": True,
            "smoke_output_not_a_production_parent": True,
        }
        if success != expected:
            raise DuplicateError("existing interactive smoke success receipt differs from its actual output")
        return {"status": "reused", "validation": validation, "smoke_success": str(paths["success"])}
    if paths["marker"].exists() or paths["marker"].is_symlink():
        raise DuplicateError("interactive smoke has a training-started marker but no success receipt; refusing ambiguous replay")
    marker = {
        "schema": SMOKE_MARKER_SCHEMA,
        "command": _file_identity(paths["command"], label="interactive smoke command"),
        "clone_amendment": _file_identity(amendment, label="clone-local r5 amendment"),
        "production_parent_not_replaced": True,
    }
    _immutable(paths["marker"], marker, label="interactive smoke training-started marker")
    completed = subprocess.run(command["argv"], cwd=duplicate, env=_run_environment(duplicate), check=False)
    if completed.returncode != 0:
        raise DuplicateError(f"interactive smoke training exited with status {completed.returncode}")
    expected_step = (r5.PARENT_SEGMENT_INDEX + 1) * r5.UPDATES_PER_SEGMENT + 1
    try:
        from ctm.training.resume_state import load_strict_local_rl_resume_state

        state = load_strict_local_rl_resume_state(paths["checkpoint"])
    except Exception as exc:
        raise DuplicateError(f"interactive smoke did not write a strict resumable checkpoint: {exc}") from exc
    if state.global_step != expected_step or state.optimizer_step != expected_step:
        raise DuplicateError(
            "interactive smoke checkpoint has an unexpected step: "
            f"global={state.global_step}, optimizer={state.optimizer_step}, expected={expected_step}"
        )
    success = {
        "schema": SMOKE_SUCCESS_SCHEMA,
        "command": _file_identity(paths["command"], label="interactive smoke command"),
        "checkpoint_files": _checkpoint_inventory(duplicate, paths["checkpoint"], label="interactive smoke checkpoint"),
        "optimizer_step": expected_step,
        "production_parent_not_replaced": True,
        "smoke_output_not_a_production_parent": True,
    }
    status = _immutable(paths["success"], success, label="interactive smoke success receipt")
    return {"status": status, "validation": validation, "smoke_success": str(paths["success"])}


def _require_smoke_success(duplicate: Path) -> dict[str, Any]:
    paths = _smoke_paths(duplicate)
    success = _json(paths["success"], label="interactive smoke success receipt")
    expected = {
        "schema": SMOKE_SUCCESS_SCHEMA,
        "command": _file_identity(paths["command"], label="interactive smoke command"),
        "checkpoint_files": _checkpoint_inventory(duplicate, paths["checkpoint"], label="interactive smoke checkpoint"),
        "optimizer_step": (r5.PARENT_SEGMENT_INDEX + 1) * r5.UPDATES_PER_SEGMENT + 1,
        "production_parent_not_replaced": True,
        "smoke_output_not_a_production_parent": True,
    }
    if success != expected:
        raise DuplicateError("interactive smoke success receipt no longer matches its isolated output")
    return success


def _next_pending_segment(duplicate: Path) -> tuple[str, int | None, dict[str, Any] | None]:
    for index in range(r5.START_SEGMENT_INDEX, r5.TOTAL_SEGMENTS):
        run = r5.run_name(index)
        run_root = duplicate / "logs" / r5.CONDITION_NAME / run
        decisions = run_root / "patience-decisions"
        if decisions.exists() or decisions.is_symlink():
            try:
                result = patience.decision_in_directory(duplicate, decisions, index)
            except Exception as exc:
                raise DuplicateError(f"clone r5 patience receipt for segment {index} does not replay: {exc}") from exc
            if result["decision"] != "continue":
                return "terminal", index, result
            continue
        marker = run_root / "segment" / "training-started.json"
        if marker.exists() or marker.is_symlink():
            raise DuplicateError(
                f"clone r5 segment {index} has a training-started marker but no patience receipt; refusing ambiguous replay"
            )
        return "pending", index, None
    return "complete", None, None


def _execution_receipt(
    *,
    duplicate: Path,
    kind: str,
    segment_index: int | None,
    detail: Mapping[str, Any],
) -> Path:
    amendment = duplicate / CLONE_AMENDMENT_RELATIVE
    document: dict[str, Any] = {
        "schema": EXECUTION_RECEIPT_SCHEMA,
        "kind": kind,
        "clone_repository": str(duplicate),
        "clone_amendment": _file_identity(amendment, label="clone-local r5 amendment"),
        "segment_index": segment_index,
        "detail": dict(detail),
        "generation": {"output_token_cap": None, "termination": "model_eos_only"},
        "queued_source_modified": False,
    }
    directory = duplicate / CLONE_ARTIFACT_RELATIVE / "execution"
    stem = f"{kind}-s{segment_index:03d}" if segment_index is not None else kind
    return _content_addressed(directory, stem=stem, document=document, label="interactive duplicate execution receipt")


def run_windows(
    *,
    source_repository: str | Path,
    duplicate_repository: str | Path,
    runtime_python: str | Path,
    model_snapshot: str | Path,
    source_amendment: str | Path | None = None,
    max_segments: int,
    minimum_remaining_seconds: int,
    yes: bool,
) -> dict[str, Any]:
    """Run complete clone-local windows only; never touch the queued source root."""

    if not yes:
        raise DuplicateError("interactive r5 execution requires --yes")
    if isinstance(max_segments, bool) or not isinstance(max_segments, int) or max_segments != 1:
        raise DuplicateError(
            "interactive allocations are limited to one full r5 window; use the external broker to obtain a fresh allocation for the next segment"
        )
    validation = validate_duplicate(
        source_repository=source_repository,
        duplicate_repository=duplicate_repository,
        runtime_python=runtime_python,
        source_amendment=source_amendment,
        model_snapshot=model_snapshot,
    )
    source = _root(source_repository, label="source repository")
    duplicate = _root(duplicate_repository, label="duplicate repository")
    snapshot = _snapshot(model_snapshot)
    _require_four_gpus()
    _require_offline_snapshot(snapshot)
    initial_remaining = _require_interactive_allocation(minimum_remaining_seconds=minimum_remaining_seconds)
    _require_smoke_success(duplicate)
    receipt = _json(duplicate / PREPARATION_RECEIPT_RELATIVE, label="interactive duplicate preparation receipt")
    python = _runtime_launcher(runtime_python, source=source, recorded=receipt["runtime_python"])
    _worker_parity_resume(python=python, duplicate=duplicate, snapshot=snapshot)
    amendment = duplicate / CLONE_AMENDMENT_RELATIVE
    runner = _regular_file(
        duplicate / "infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.py",
        label="clone r5 segment runner",
    )

    executed: list[dict[str, Any]] = []
    for _ in range(max_segments):
        status, index, detail = _next_pending_segment(duplicate)
        if status in {"terminal", "complete"}:
            execution = _execution_receipt(
                duplicate=duplicate,
                kind=status,
                segment_index=index,
                detail={} if detail is None else detail,
            )
            return {
                "status": status,
                "validation": validation,
                "initial_remaining_seconds": initial_remaining,
                "executed": executed,
                "execution_receipt": str(execution),
            }
        assert status == "pending" and index is not None
        remaining = _require_interactive_allocation(minimum_remaining_seconds=minimum_remaining_seconds)
        if remaining < minimum_remaining_seconds:
            execution = _execution_receipt(
                duplicate=duplicate,
                kind="allocation_reserve_stop",
                segment_index=index,
                detail={"remaining_seconds": remaining, "minimum_remaining_seconds": minimum_remaining_seconds},
            )
            return {
                "status": "allocation_reserve_stop",
                "validation": validation,
                "initial_remaining_seconds": initial_remaining,
                "executed": executed,
                "execution_receipt": str(execution),
            }
        command = [
            python,
            str(runner),
            "--repository",
            str(duplicate),
            "--segment-index",
            str(index),
            "--amendment",
            str(amendment),
            "--yes",
        ]
        completed = subprocess.run(command, cwd=duplicate, env=_run_environment(duplicate), check=False)
        if completed.returncode != 0:
            raise DuplicateError(f"clone r5 segment {index} exited with status {completed.returncode}")
        decision_directory = duplicate / "logs" / r5.CONDITION_NAME / r5.run_name(index) / "patience-decisions"
        try:
            decision = patience.decision_in_directory(duplicate, decision_directory, index)
        except Exception as exc:
            raise DuplicateError(f"clone r5 segment {index} completed without a valid patience receipt: {exc}") from exc
        raw_end = os.environ.get("SLURM_JOB_END_TIME")
        try:
            remaining_after = max(0, int(raw_end) - int(time.time())) if raw_end is not None else None
        except ValueError:
            remaining_after = None
        execution = _execution_receipt(
            duplicate=duplicate,
            kind="segment_complete",
            segment_index=index,
            detail={"decision": decision, "remaining_seconds_after_segment": remaining_after},
        )
        executed.append({"segment_index": index, "decision": decision, "execution_receipt": str(execution)})
        if decision["decision"] != "continue":
            return {
                "status": "terminal",
                "validation": validation,
                "initial_remaining_seconds": initial_remaining,
                "executed": executed,
                "execution_receipt": str(execution),
            }
    execution = _execution_receipt(
        duplicate=duplicate,
        kind="one_window_stop",
        segment_index=executed[-1]["segment_index"] if executed else None,
        detail={"max_segments": max_segments, "fresh_interactive_allocation_required_for_next_window": True},
    )
    return {
        "status": "one_window_stop",
        "validation": validation,
        "initial_remaining_seconds": initial_remaining,
        "executed": executed,
        "execution_receipt": str(execution),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, *, snapshot: bool = False) -> None:
        command.add_argument("--source-repository", required=True, type=Path)
        command.add_argument("--duplicate-repository", required=True, type=Path)
        command.add_argument("--runtime-python", required=True, type=Path)
        command.add_argument("--source-amendment", type=Path)
        if snapshot:
            command.add_argument("--model-snapshot", required=True, type=Path)

    prepare = commands.add_parser("prepare", help="copy frozen source/dependencies into one fresh sibling root")
    common(prepare)
    validate = commands.add_parser("validate", help="replay clone custody checks without model generation")
    common(validate)
    validate.add_argument("--model-snapshot", type=Path)
    status = commands.add_parser("status", help="read the clone-local next segment or terminal decision")
    status.add_argument("--source-repository", required=True, type=Path)
    status.add_argument("--duplicate-repository", required=True, type=Path)
    status.add_argument("--source-amendment", type=Path)
    smoke = commands.add_parser("smoke", help="run one isolated uncapped optimiser-update smoke")
    common(smoke, snapshot=True)
    smoke.add_argument("--yes", action="store_true")
    run = commands.add_parser("run", help="run clone-local r5 windows inside an existing interactive allocation")
    common(run, snapshot=True)
    run.add_argument("--max-segments", type=int, default=1)
    run.add_argument("--minimum-remaining-seconds", type=int, default=0)
    run.add_argument("--yes", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o077)
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_duplicate(
                source_repository=args.source_repository,
                duplicate_repository=args.duplicate_repository,
                runtime_python=args.runtime_python,
                source_amendment=args.source_amendment,
            )
        elif args.command == "validate":
            result = validate_duplicate(
                source_repository=args.source_repository,
                duplicate_repository=args.duplicate_repository,
                runtime_python=args.runtime_python,
                source_amendment=args.source_amendment,
                model_snapshot=args.model_snapshot,
            )
        elif args.command == "status":
            result = duplicate_status(
                source_repository=args.source_repository,
                duplicate_repository=args.duplicate_repository,
                source_amendment=args.source_amendment,
            )
        elif args.command == "smoke":
            result = run_smoke(
                source_repository=args.source_repository,
                duplicate_repository=args.duplicate_repository,
                runtime_python=args.runtime_python,
                source_amendment=args.source_amendment,
                model_snapshot=args.model_snapshot,
                yes=args.yes,
            )
        elif args.command == "run":
            result = run_windows(
                source_repository=args.source_repository,
                duplicate_repository=args.duplicate_repository,
                runtime_python=args.runtime_python,
                source_amendment=args.source_amendment,
                model_snapshot=args.model_snapshot,
                max_segments=args.max_segments,
                minimum_remaining_seconds=args.minimum_remaining_seconds,
                yes=args.yes,
            )
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(args.command)
    except (DuplicateError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(_canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
