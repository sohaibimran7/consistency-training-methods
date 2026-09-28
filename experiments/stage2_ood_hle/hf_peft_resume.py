"""Fail-closed resume support for native-HF/PEFT Stage 2 OOD matrices.

This module deliberately does no model loading, generation, grading, or
network access.  It is the CPU-side companion to
``ctm_stage2_ood_hf_peft_resume_condition_20260803.sh``:

* retain only header-validated successful EvalLogs already present in a
  condition-local raw directory;
* move unreadable, partial, or non-success ``.eval`` files to a recoverable,
  condition-specific archive before any new worker starts;
* write a hash-bound resume contract before generation, so a later invocation
  cannot silently switch the raw adapter, frozen matrix, or inherited success
  cells; and
* make each completed biased cell available to an *incremental* local handoff
  only after its three shared clean cells have passed full native-HF checks.

The authoritative public handoff remains the existing 21-cell
``raw_preflight`` followed by ``stage_luna``/``grade_luna``.  Incremental
handoff receipts are intentionally separate: they allow an approved grader
worker to begin from an individual, hash-bound biased cell without weakening
that final all-cell gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.stage2_ood_hle import raw_preflight
from experiments.stage2_ood_hle.grade_luna import _staged_path
from experiments.stage2_ood_hle.hf_peft_runner import (
    BASE_MODEL,
    build_launch_contract,
)
from experiments.stage2_ood_hle.materialize import PROMPT_STYLE
from experiments.stage2_ood_hle.stage_luna import _copy_new
from experiments.stage2_ood_hle.tasks import OODTaskSpec, ood_task_specs

RESUME_SCHEMA = "stage2-ood-hf-peft-resume-v1"
ACTIVE_HANDOFF_SCHEMA = "stage2-ood-hf-peft-active-handoff-contract-v1"
ARCHIVE_SCHEMA = "stage2-ood-hf-peft-resume-archive-v1"
HANDOFF_SCHEMA = "stage2-ood-hf-peft-incremental-handoff-v1"
EXPECTED_MAX_CONNECTIONS = 8

# An active generation directory may contain an Inspect file that is still
# being written by another worker.  This policy records the deliberately
# narrow exception for an incremental handoff: it may *read and ignore* those
# candidates, but it must never archive, rename, delete, or otherwise mutate
# them.  Keep it in the external contract so a later invocation cannot turn
# this into the ordinary resume-and-archive path by accident.
ACTIVE_HANDOFF_POLICY = {
    "raw_eval_logs_read_only": True,
    "archive_or_delete_raw_eval_logs": False,
    "allow_unrelated_inflight_partials": True,
}

Identity = tuple[str, str, str, str, str | None]


@dataclass(frozen=True, slots=True)
class ArchiveCandidate:
    """A non-success or unreadable raw EvalLog to move out of the matrix."""

    path: Path
    reason: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class Audit:
    """Selected exact success cells and recoverable partial artifacts."""

    selected: Mapping[Identity, raw_preflight.LoadedTaskLog]
    archive_candidates: tuple[ArchiveCandidate, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_or_none(path: Path) -> str | None:
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically write JSON once, accepting only a byte-identical resume."""

    encoded = _json_bytes(payload)
    if path.exists():
        if path.is_file() and path.read_bytes() == encoded:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing resume artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != encoded:
                raise FileExistsError(f"resume artifact appeared and differs: {path}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "written"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _identity_record(identity: Identity) -> dict[str, str | None]:
    kind, regime, population, dataset, bias_type = identity
    return {
        "kind": kind,
        "regime": regime,
        "population": population,
        "dataset": dataset,
        "bias_type": bias_type,
    }


def _identity_from_record(value: Any) -> Identity:
    if not isinstance(value, Mapping):
        raise ValueError("resume contract has an invalid task identity")
    kind = value.get("kind")
    regime = value.get("regime")
    population = value.get("population")
    dataset = value.get("dataset")
    bias_type = value.get("bias_type")
    if not all(isinstance(item, str) and item for item in (kind, regime, population, dataset)):
        raise ValueError("resume contract task identity is incomplete")
    if bias_type is not None and (not isinstance(bias_type, str) or not bias_type):
        raise ValueError("resume contract task identity has an invalid bias_type")
    return str(kind), str(regime), str(population), str(dataset), bias_type


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the condition raw-log root: {resolved}") from exc
    return resolved


def _all_eval_paths(raw_root: Path) -> list[Path]:
    """Discover every physical ``.eval`` file, including malformed partials."""

    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw Stage 2 OOD directory does not exist: {raw_root}")
    paths: list[Path] = []
    for candidate in raw_root.rglob("*.eval"):
        if candidate.is_symlink():
            raise ValueError(f"raw Stage 2 OOD EvalLog may not be a symlink: {candidate}")
        if not candidate.is_file():
            continue
        paths.append(_under_root(candidate, raw_root, label="raw Stage 2 OOD EvalLog"))
    return sorted(set(paths))


def _runtime(checkpoint: Path) -> dict[str, Any]:
    return raw_preflight._validate_runtime_contract(
        runtime_profile="hf-peft",
        expected_base_model=BASE_MODEL,
        expected_checkpoint=str(checkpoint),
        expected_max_connections=EXPECTED_MAX_CONNECTIONS,
        require_vllm_adapter_attestation=False,
    )


def _expected_by_index(manifest: Path) -> tuple[dict[Identity, OODTaskSpec], dict[Identity, int]]:
    specs = ood_task_specs(manifest)
    expected = raw_preflight._expected_cell_specs(specs)
    indices = {raw_preflight._task_identity(spec): index for index, spec in enumerate(specs, start=1)}
    if set(indices) != set(expected) or len(indices) != raw_preflight.EXPECTED_TASKS:
        raise ValueError("Stage 2 OOD task indices do not cover the exact frozen 21-cell matrix")
    return expected, indices


def audit_raw_logs(
    raw_log_dir: str | Path,
    *,
    manifest: str | Path,
    checkpoint: str | Path,
) -> Audit:
    """Select valid successful cells and identify only safe-to-archive failures.

    A successful log with an unexpected header, task identity, or runtime is a
    scientific conflict, not disposable noise; it aborts the resume.  In
    contrast, unreadable, partial, and explicit non-success EvalLogs are
    recoverable retries and are returned for archival.
    """

    raw_root = Path(raw_log_dir).resolve()
    manifest_path = Path(manifest).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    raw_preflight.validate_manifest(manifest_path)
    expected, _ = _expected_by_index(manifest_path)
    runtime = _runtime(checkpoint_path)
    candidates: dict[Identity, list[raw_preflight.LoadedTaskLog]] = {}
    archive_candidates: list[ArchiveCandidate] = []
    fatal: list[str] = []

    for path in _all_eval_paths(raw_root):
        try:
            header = raw_preflight._read_eval_log(path, header_only=True)
        except Exception as exc:
            archive_candidates.append(
                ArchiveCandidate(path, f"unreadable_header:{type(exc).__name__}", _sha256_or_none(path))
            )
            continue
        if raw_preflight._attribute(header, "status") != "success":
            archive_candidates.append(ArchiveCandidate(path, "non_success_status", _sha256_or_none(path)))
            continue
        try:
            evaluation = raw_preflight._attribute(header, "eval")
            task_name = raw_preflight._task_basename(raw_preflight._attribute(evaluation, "task"))
            if task_name not in {raw_preflight.TASK_UNBIASED, raw_preflight.TASK_BIASED}:
                raise ValueError(f"unexpected successful Inspect task {task_name!r}")
            identity = raw_preflight._parse_candidate_identity(evaluation, task_name=task_name, path=path)
            spec = expected.get(identity)
            if spec is None:
                raise ValueError(f"successful EvalLog has a task outside the frozen matrix: {identity!r}")
            created = raw_preflight._validate_header(header, path=path, spec=spec, raw_root=raw_root)
            model, observed_runtime = raw_preflight._assert_runtime(path, runtime=runtime)
        except Exception as exc:
            # Header/runtime mismatches are never automatically hidden.
            fatal.append(f"{path}: {exc}")
            continue
        try:
            # A complete header can precede a truncated payload.  Read the
            # entire EvalLog now, but defer semantic paired-score validation
            # until clean references have been selected below.
            raw_preflight._read_eval_log(path, header_only=False)
        except Exception as exc:
            archive_candidates.append(
                ArchiveCandidate(path, f"partial_payload:{type(exc).__name__}", _sha256_or_none(path))
            )
            continue
        candidates.setdefault(identity, []).append(
            raw_preflight.LoadedTaskLog(spec, path, created, header, model, observed_runtime)
        )

    if fatal:
        rendered = "\n".join(fatal)
        raise ValueError(f"refusing to resume around a successful but invalid Stage 2 EvalLog:\n{rendered}")

    selected: dict[Identity, raw_preflight.LoadedTaskLog] = {}
    for identity, entries in candidates.items():
        entries.sort(key=lambda item: (item.created, str(item.path)))
        if len(entries) > 1 and entries[-1].created == entries[-2].created:
            raise ValueError(
                "ambiguous successful Stage 2 retries with the same timestamp for "
                f"{raw_preflight._display_identity(identity)}: {entries[-2].path} and {entries[-1].path}"
            )
        selected[identity] = entries[-1]
    return Audit(selected=selected, archive_candidates=tuple(archive_candidates))


def _archive_candidates(
    candidates: Sequence[ArchiveCandidate],
    *,
    raw_root: Path,
    archive_dir: str | Path | None,
    condition: str,
) -> list[dict[str, Any]]:
    """Move only retryable EvalLogs to a fresh, condition-specific archive."""

    if not candidates:
        return []
    if archive_dir is None:
        raise ValueError("non-success/partial EvalLogs need --archive-dir before this condition can resume")
    destination_root = Path(archive_dir).resolve()
    if destination_root.exists():
        raise FileExistsError(f"refusing to reuse an existing resume archive directory: {destination_root}")
    if condition not in destination_root.parts:
        raise ValueError("resume archive directory must include the exact condition name")
    try:
        destination_root.relative_to(raw_root)
    except ValueError:
        pass
    else:
        raise ValueError("resume archive directory may not be nested under the raw-log root")

    planned: list[tuple[ArchiveCandidate, Path]] = []
    for candidate in candidates:
        source = _under_root(candidate.path, raw_root, label="archive source")
        relative = source.relative_to(raw_root)
        destination = destination_root / "partial-or-non-success" / relative
        if destination.exists():
            raise FileExistsError(f"resume archive destination already exists: {destination}")
        planned.append((candidate, destination))

    destination_root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for candidate, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate.path), str(destination))
        observed = _sha256_or_none(destination)
        if candidate.sha256 is not None and observed != candidate.sha256:
            raise RuntimeError(f"archived EvalLog hash changed while moving {candidate.path}")
        records.append(
            {
                "source": str(candidate.path),
                "destination": str(destination),
                "reason": candidate.reason,
                "sha256": observed,
            }
        )
    _write_immutable_json(
        destination_root / "archive-receipt.json",
        {
            "schema": ARCHIVE_SCHEMA,
            "condition": condition,
            "raw_log_dir": str(raw_root),
            "entries": records,
        },
    )
    return records


def _success_records(
    selected: Mapping[Identity, raw_preflight.LoadedTaskLog], *, index_by_identity: Mapping[Identity, int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for identity, loaded in selected.items():
        records.append(
            {
                "task_index": index_by_identity[identity],
                "identity": _identity_record(identity),
                "raw_log": str(loaded.path),
                "raw_log_sha256": _sha256_file(loaded.path),
                "created": loaded.created,
                "model": loaded.model,
                "runtime": dict(loaded.runtime),
            }
        )
    return sorted(records, key=lambda item: int(item["task_index"]))


def _resume_execution_contract() -> dict[str, Any]:
    return {
        "runtime_profile": "hf-peft",
        "model_provider": "hf",
        "model_device": "cuda:0",
        "model_dtype": "bfloat16",
        "max_connections": EXPECTED_MAX_CONNECTIONS,
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_k": 20,
        "top_p": 0.95,
        "max_tasks": 1,
        "isolate_tasks": True,
        "one_model_process_per_physical_gpu": True,
        "prompt_style": PROMPT_STYLE,
        "include_bias_acknowledged": False,
    }


def build_resume_contract(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    selected: Mapping[Identity, raw_preflight.LoadedTaskLog],
) -> dict[str, Any]:
    """Build the immutable contract before workers fill missing cells."""

    raw_root = Path(raw_log_dir).resolve()
    manifest_path = Path(manifest).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    launch = build_launch_contract(
        condition=condition,
        checkpoint=checkpoint_path,
        manifest=manifest_path,
        raw_log_dir=raw_root,
        max_connections=EXPECTED_MAX_CONNECTIONS,
    )
    _, index_by_identity = _expected_by_index(manifest_path)
    return {
        "schema": RESUME_SCHEMA,
        "condition": condition,
        "raw_log_dir": str(raw_root),
        "launch_contract": launch,
        "resume_execution": _resume_execution_contract(),
        "initial_successes": _success_records(selected, index_by_identity=index_by_identity),
    }


def build_active_handoff_contract(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    selected: Mapping[Identity, raw_preflight.LoadedTaskLog],
) -> dict[str, Any]:
    """Bind currently successful cells without changing a live raw directory.

    This is intentionally separate from :func:`build_resume_contract`: the
    regular resume protocol may recoverably archive retryable files before it
    launches new workers, whereas an active handoff is only allowed to observe
    a still-running matrix.  The contract itself must live outside the raw
    EvalLog directory (enforced by its writer and validator below).
    """

    raw_root = Path(raw_log_dir).resolve()
    manifest_path = Path(manifest).resolve()
    checkpoint_path = Path(checkpoint).resolve()
    launch = build_launch_contract(
        condition=condition,
        checkpoint=checkpoint_path,
        manifest=manifest_path,
        raw_log_dir=raw_root,
        max_connections=EXPECTED_MAX_CONNECTIONS,
    )
    _, index_by_identity = _expected_by_index(manifest_path)
    return {
        "schema": ACTIVE_HANDOFF_SCHEMA,
        "condition": condition,
        "raw_log_dir": str(raw_root),
        "launch_contract": launch,
        "resume_execution": _resume_execution_contract(),
        "active_handoff_policy": dict(ACTIVE_HANDOFF_POLICY),
        "initial_successes": _success_records(selected, index_by_identity=index_by_identity),
    }


def _validate_initial_successes(contract: Mapping[str, Any], *, raw_root: Path) -> None:
    """Prove every success inherited by a contract still has its original bytes."""

    successes = contract.get("initial_successes")
    if not isinstance(successes, list):
        raise ValueError("handoff contract has no initial success list")
    seen_indices: set[int] = set()
    seen_identities: set[Identity] = set()
    for record in successes:
        if not isinstance(record, Mapping):
            raise ValueError("handoff contract has an invalid initial success record")
        index = record.get("task_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= raw_preflight.EXPECTED_TASKS:
            raise ValueError("handoff contract initial success has an invalid task index")
        identity = _identity_from_record(record.get("identity"))
        if index in seen_indices or identity in seen_identities:
            raise ValueError("handoff contract duplicates an initial success cell")
        seen_indices.add(index)
        seen_identities.add(identity)
        source = record.get("raw_log")
        digest = record.get("raw_log_sha256")
        if not isinstance(source, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("handoff contract initial success has an invalid raw-log identity")
        path = _under_root(Path(source), raw_root, label="contract-bound successful EvalLog")
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"contract-bound successful EvalLog changed or disappeared: {path}")


def _validate_resume_contract(
    contract: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    raw_root: Path,
) -> None:
    if contract.get("schema") != RESUME_SCHEMA:
        raise ValueError("resume contract has an unexpected schema")
    if contract.get("condition") != expected.get("condition"):
        raise ValueError("resume contract condition differs from this launch")
    if contract.get("raw_log_dir") != str(raw_root):
        raise ValueError("resume contract raw-log directory differs from this launch")
    if contract.get("launch_contract") != expected.get("launch_contract"):
        raise ValueError("resume contract checkpoint, manifest, or decode contract differs from this launch")
    if contract.get("resume_execution") != _resume_execution_contract():
        raise ValueError("resume contract execution policy differs from this launcher")
    _validate_initial_successes(contract, raw_root=raw_root)


def _validate_active_handoff_contract(
    contract: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    raw_root: Path,
    contract_path: Path,
) -> None:
    """Validate a read-only active-run contract without inspecting partials."""

    if contract.get("schema") != ACTIVE_HANDOFF_SCHEMA:
        raise ValueError("active handoff contract has an unexpected schema")
    if contract_path == raw_root or contract_path.is_relative_to(raw_root):
        raise ValueError("active handoff contract must live outside the raw EvalLog directory")
    if set(contract) != set(expected):
        raise ValueError("active handoff contract has unexpected or missing fields")
    for field, message in (
        ("condition", "condition"),
        ("raw_log_dir", "raw-log directory"),
        ("launch_contract", "checkpoint, manifest, or decode contract"),
        ("resume_execution", "execution policy"),
    ):
        if contract.get(field) != expected.get(field):
            raise ValueError(f"active handoff contract {message} differs from this invocation")
    if contract.get("active_handoff_policy") != ACTIVE_HANDOFF_POLICY:
        raise ValueError("active handoff contract does not preserve the read-only active-run policy")
    _validate_initial_successes(contract, raw_root=raw_root)


def _read_or_write_contract(path: Path, document: Mapping[str, Any]) -> str:
    if path.exists():
        existing = _read_json_object(path, label="resume contract")
        _validate_resume_contract(existing, expected=document, raw_root=Path(str(document["raw_log_dir"])).resolve())
        return "resumed"
    return _write_immutable_json(path, document)


def establish_active_handoff_contract(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Create or revalidate an external, read-only live-run handoff contract.

    Unlike :func:`prepare_resume`, this function does not create the raw
    directory, move a retryable log, or write anything below it.  A partial or
    non-success ``.eval`` file is merely reported as an in-flight candidate.
    That lets independent workers finish normally while already-complete
    biased cells are hash-bound for incremental staging and grading.
    """

    raw_root = Path(raw_log_dir).resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"active handoff raw Stage 2 OOD directory does not exist: {raw_root}")
    contract = Path(contract_path).resolve()
    if contract == raw_root or contract.is_relative_to(raw_root):
        raise ValueError("active handoff contract must live outside the raw EvalLog directory")
    audit = audit_raw_logs(raw_root, manifest=manifest, checkpoint=checkpoint)
    document = build_active_handoff_contract(
        condition=condition,
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        selected=audit.selected,
    )
    if contract.exists():
        existing = _read_json_object(contract, label="active handoff contract")
        _validate_active_handoff_contract(
            existing,
            expected=document,
            raw_root=raw_root,
            contract_path=contract,
        )
        status = "resumed"
    else:
        status = _write_immutable_json(contract, document)
    _, index_by_identity = _expected_by_index(Path(manifest).resolve())
    return {
        "schema": ACTIVE_HANDOFF_SCHEMA,
        "contract": str(contract),
        "contract_status": status,
        "raw_eval_logs_mutated": False,
        "unmodified_inflight_candidates": [
            {"path": str(candidate.path), "reason": candidate.reason, "sha256": candidate.sha256}
            for candidate in audit.archive_candidates
        ],
        **_state(audit.selected, index_by_identity=index_by_identity),
    }


def _validated_contract_and_audit(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    contract_path: str | Path,
) -> Audit:
    """Validate immutable launch identity, then inspect the live raw directory."""

    raw_root = Path(raw_log_dir).resolve()
    path = Path(contract_path).resolve()
    existing = _read_json_object(path, label="resume or active handoff contract")
    if existing.get("schema") == RESUME_SCHEMA:
        expected = build_resume_contract(
            condition=condition,
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=raw_root,
            selected={},
        )
        _validate_resume_contract(existing, expected=expected, raw_root=raw_root)
    elif existing.get("schema") == ACTIVE_HANDOFF_SCHEMA:
        expected = build_active_handoff_contract(
            condition=condition,
            checkpoint=checkpoint,
            manifest=manifest,
            raw_log_dir=raw_root,
            selected={},
        )
        _validate_active_handoff_contract(
            existing,
            expected=expected,
            raw_root=raw_root,
            contract_path=path,
        )
    else:
        raise ValueError("handoff contract has an unsupported schema")
    return audit_raw_logs(raw_root, manifest=manifest, checkpoint=checkpoint)


def _state(
    selected: Mapping[Identity, raw_preflight.LoadedTaskLog], *, index_by_identity: Mapping[Identity, int]
) -> dict[str, Any]:
    present = {index_by_identity[identity] for identity in selected}
    clean = [index for index in range(1, 4) if index not in present]
    biased = [index for index in range(4, 22) if index not in present]
    return {
        "selected_task_indices": sorted(present),
        "missing_clean_task_indices": clean,
        "missing_biased_task_indices": biased,
        "complete": not clean and not biased,
    }


def prepare_resume(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    contract_path: str | Path,
    archive_dir: str | Path | None,
) -> dict[str, Any]:
    """Archive retryable failures, bind inherited successes, and list gaps."""

    raw_root = Path(raw_log_dir).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    contract = Path(contract_path).resolve()
    if contract.parent != raw_root:
        raise ValueError("resume contract must live directly inside the condition raw-log directory")
    audit = audit_raw_logs(raw_root, manifest=manifest, checkpoint=checkpoint)
    archived = _archive_candidates(
        audit.archive_candidates,
        raw_root=raw_root,
        archive_dir=archive_dir,
        condition=condition,
    )
    document = build_resume_contract(
        condition=condition,
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_root,
        selected=audit.selected,
    )
    status = _read_or_write_contract(contract, document)
    _, index_by_identity = _expected_by_index(Path(manifest).resolve())
    return {
        "schema": RESUME_SCHEMA,
        "contract": str(contract),
        "contract_status": status,
        "archived": archived,
        **_state(audit.selected, index_by_identity=index_by_identity),
    }


def resume_status(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Revalidate the immutable contract and report only currently missing cells."""

    audit = _validated_contract_and_audit(
        condition=condition,
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_log_dir,
        contract_path=contract_path,
    )
    if audit.archive_candidates:
        paths = ", ".join(str(candidate.path) for candidate in audit.archive_candidates)
        raise ValueError(f"new non-success/partial EvalLogs must be archived before continuing: {paths}")
    _, index_by_identity = _expected_by_index(Path(manifest).resolve())
    return {
        "schema": RESUME_SCHEMA,
        "contract": str(Path(contract_path).resolve()),
        **_state(audit.selected, index_by_identity=index_by_identity),
    }


def verify_clean_barrier(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    contract_path: str | Path,
) -> dict[str, Any]:
    """Require all three canonical clean cells before any biased launch."""

    status = resume_status(
        condition=condition,
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_log_dir,
        contract_path=contract_path,
    )
    missing = status["missing_clean_task_indices"]
    if missing:
        raise ValueError(f"clean Stage 2 OOD barrier is incomplete; missing task indices {missing}")
    # Do not treat a success header as enough to unlock expensive biased work:
    # each of the shared clean logs must also have a complete frozen sample
    # payload before it becomes a pairing dependency.
    audit = _validated_contract_and_audit(
        condition=condition,
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_log_dir,
        contract_path=contract_path,
    )
    _validated_selected_full(audit.selected, requested=set())
    return status


def _validated_selected_full(
    selected: Mapping[Identity, raw_preflight.LoadedTaskLog],
    *,
    requested: set[Identity],
) -> dict[Identity, Any]:
    clean_paths = {
        (loaded.spec.population, loaded.spec.dataset): loaded.path
        for loaded in selected.values()
        if loaded.spec.kind == "unbiased"
    }
    if len(clean_paths) != raw_preflight.EXPECTED_CLEAN_TASKS:
        raise ValueError("incremental handoff requires all three exact clean EvalLogs")
    result: dict[Identity, Any] = {}
    for identity, loaded in selected.items():
        if loaded.spec.kind == "unbiased" or identity in requested:
            try:
                complete = raw_preflight._read_eval_log(loaded.path, header_only=False)
            except Exception as exc:
                raise ValueError(f"could not read full Stage 2 EvalLog for incremental handoff: {loaded.path}") from exc
            raw_preflight._validate_samples(complete, loaded=loaded, clean_paths=clean_paths)
            result[identity] = complete
    missing = requested - set(result)
    if missing:
        raise ValueError(f"incremental handoff selected missing biased cells: {sorted(missing)}")
    return result


def _handoff_receipt(
    *,
    condition: str,
    loaded: raw_preflight.LoadedTaskLog,
    raw_root: Path,
    staged: Path,
    manifest: Path,
    checkpoint_record: Mapping[str, Any],
    clean_paths: Mapping[tuple[str, str], Path],
    task_index: int,
) -> dict[str, Any]:
    clean = clean_paths[(loaded.spec.population, loaded.spec.dataset)]
    return {
        "schema": HANDOFF_SCHEMA,
        "condition": condition,
        "task_index": task_index,
        "identity": _identity_record(raw_preflight._task_identity(loaded.spec)),
        "raw_log": {"path": str(loaded.path), "sha256": _sha256_file(loaded.path)},
        "staged_log": {"path": str(staged), "sha256": _sha256_file(staged)},
        "paired_clean": {"path": str(clean), "sha256": _sha256_file(clean)},
        "raw_log_dir": str(raw_root),
        "manifest": {"path": str(manifest), "sha256": _sha256_file(manifest)},
        "checkpoint": dict(checkpoint_record),
        "protocol": {
            "runtime_profile": "hf-peft",
            "prompt_style": PROMPT_STYLE,
            "include_bias_acknowledged": False,
            "max_tokens": 20480,
            "max_connections": EXPECTED_MAX_CONNECTIONS,
            "validated_with_full_paired_switch_scores": True,
        },
    }


def stage_incremental_handoff(
    *,
    condition: str,
    checkpoint: str | Path,
    manifest: str | Path,
    raw_log_dir: str | Path,
    contract_path: str | Path,
    output_root: str | Path,
    task_indices: Sequence[int] | None,
    all_biased: bool,
    allow_inflight_partials: bool,
) -> list[dict[str, Any]]:
    """Stage full-validated biased successes without calling an external grader."""

    if all_biased and task_indices:
        raise ValueError("choose --all-biased or --task-index, not both")
    if not all_biased and not task_indices:
        raise ValueError("incremental handoff needs --all-biased or at least one --task-index")
    audit = _validated_contract_and_audit(
        condition=condition,
        checkpoint=checkpoint,
        manifest=manifest,
        raw_log_dir=raw_log_dir,
        contract_path=contract_path,
    )
    _, index_by_identity = _expected_by_index(Path(manifest).resolve())
    state = _state(audit.selected, index_by_identity=index_by_identity)
    if state["missing_clean_task_indices"]:
        raise ValueError(
            "incremental handoff requires all exact clean cells; "
            f"missing task indices {state['missing_clean_task_indices']}"
        )
    raw_root = Path(raw_log_dir).resolve()
    manifest_path = Path(manifest).resolve()
    output = Path(output_root).resolve()
    if output == raw_root or output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError("incremental handoff root must be separate from and non-nested with raw EvalLogs")
    if audit.archive_candidates and not allow_inflight_partials:
        raise ValueError("incremental handoff refuses non-success/partial EvalLogs until they are archived")
    by_index = {index: identity for identity, index in index_by_identity.items()}
    if all_biased:
        requested = {identity for identity in audit.selected if identity[0] == "biased"}
    else:
        requested = set()
        for index in task_indices or ():
            if isinstance(index, bool) or not isinstance(index, int) or index not in by_index:
                raise ValueError(f"incremental handoff has an invalid task index: {index!r}")
            identity = by_index[index]
            if identity[0] != "biased":
                raise ValueError(f"incremental handoff accepts biased task indices only, got {index}")
            if identity not in audit.selected:
                raise ValueError(f"incremental handoff task {index} has no successful EvalLog yet")
            requested.add(identity)
    if not requested:
        return []
    _validated_selected_full(audit.selected, requested=requested)
    launch = build_launch_contract(
        condition=condition,
        checkpoint=Path(checkpoint).resolve(),
        manifest=manifest_path,
        raw_log_dir=raw_root,
        max_connections=EXPECTED_MAX_CONNECTIONS,
    )
    checkpoint_record = launch["checkpoint"]
    assert isinstance(checkpoint_record, Mapping)
    clean_paths = {
        (loaded.spec.population, loaded.spec.dataset): loaded.path
        for loaded in audit.selected.values()
        if loaded.spec.kind == "unbiased"
    }
    records: list[dict[str, Any]] = []
    for identity in sorted(requested, key=index_by_identity.__getitem__):
        loaded = audit.selected[identity]
        source = {
            "population": loaded.spec.population,
            "bias_type": loaded.spec.bias_type,
            "dataset": loaded.spec.dataset,
            "raw_log": str(loaded.path),
        }
        staged = _staged_path(output, condition=condition, source=source)
        source_sha = _sha256_file(loaded.path)
        copy_status = _copy_new(loaded.path, staged, expected_sha256=source_sha)
        # A live worker must never be able to turn a just-completed handoff
        # into a silent mixed-byte receipt.  `_copy_new` checks its temporary
        # copy; repeat the source/staged comparison before writing the
        # immutable receipt, so a concurrent rewrite fails closed.
        if _sha256_file(loaded.path) != source_sha:
            raise RuntimeError(f"successful raw EvalLog changed while staging incremental handoff: {loaded.path}")
        if _sha256_file(staged) != source_sha:
            raise RuntimeError(f"staged raw EvalLog changed while staging incremental handoff: {staged}")
        receipt = _handoff_receipt(
            condition=condition,
            loaded=loaded,
            raw_root=raw_root,
            staged=staged,
            manifest=manifest_path,
            checkpoint_record=checkpoint_record,
            clean_paths=clean_paths,
            task_index=index_by_identity[identity],
        )
        receipt_status = _write_immutable_json(staged.with_suffix(".resume-handoff.json"), receipt)
        records.append(
            {
                "task_index": index_by_identity[identity],
                "raw_log": str(loaded.path),
                "staged_log": str(staged),
                "copy_status": copy_status,
                "receipt_status": receipt_status,
            }
        )
    return records


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--condition", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--raw-log-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="archive retryable failures and write/resume the contract")
    _add_common_arguments(prepare)
    prepare.add_argument("--archive-dir", type=Path)
    active = commands.add_parser(
        "active-contract",
        help="write or revalidate an external read-only contract for an active matrix; never archives raw EvalLogs",
    )
    _add_common_arguments(active)
    status = commands.add_parser("status", help="report remaining cells after revalidating the contract")
    _add_common_arguments(status)
    clean = commands.add_parser("verify-clean", help="require all three exact successful clean cells")
    _add_common_arguments(clean)
    handoff = commands.add_parser("handoff", help="stage full-validated biased successes for an incremental grader")
    _add_common_arguments(handoff)
    handoff.add_argument("--output-root", required=True, type=Path)
    handoff.add_argument("--task-index", action="append", type=int)
    handoff.add_argument("--all-biased", action="store_true")
    handoff.add_argument(
        "--allow-inflight-partials",
        action="store_true",
        help="Permit unrelated workers' non-success/partial logs while staging already-completed biased task(s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result: Any = prepare_resume(
                condition=args.condition,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                raw_log_dir=args.raw_log_dir,
                contract_path=args.contract,
                archive_dir=args.archive_dir,
            )
        elif args.command == "active-contract":
            result = establish_active_handoff_contract(
                condition=args.condition,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                raw_log_dir=args.raw_log_dir,
                contract_path=args.contract,
            )
        elif args.command == "status":
            result = resume_status(
                condition=args.condition,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                raw_log_dir=args.raw_log_dir,
                contract_path=args.contract,
            )
        elif args.command == "verify-clean":
            result = verify_clean_barrier(
                condition=args.condition,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                raw_log_dir=args.raw_log_dir,
                contract_path=args.contract,
            )
        else:
            result = stage_incremental_handoff(
                condition=args.condition,
                checkpoint=args.checkpoint,
                manifest=args.manifest,
                raw_log_dir=args.raw_log_dir,
                contract_path=args.contract,
                output_root=args.output_root,
                task_indices=args.task_index,
                all_biased=args.all_biased,
                allow_inflight_partials=args.allow_inflight_partials,
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


__all__ = [
    "ACTIVE_HANDOFF_POLICY",
    "ACTIVE_HANDOFF_SCHEMA",
    "ARCHIVE_SCHEMA",
    "HANDOFF_SCHEMA",
    "RESUME_SCHEMA",
    "ArchiveCandidate",
    "Audit",
    "audit_raw_logs",
    "build_active_handoff_contract",
    "build_resume_contract",
    "establish_active_handoff_contract",
    "prepare_resume",
    "resume_status",
    "stage_incremental_handoff",
    "verify_clean_barrier",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
