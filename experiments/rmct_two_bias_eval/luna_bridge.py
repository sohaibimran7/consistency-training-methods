"""Custody-safe post-hoc Luna grading for the sealed r005 two-bias run.

The terminal r005 evaluation intentionally generated its 21 raw EvalLogs
without an LLM judge.  This module is the *separate* post-hoc consumer for
the eighteen biased cells only.  It never changes a raw log, its task receipt,
the native raw-preflight report, or the evaluation receipt.

Before any grade request it proves all of the following again:

* the exact r005 evaluation receipt still verifies;
* the persisted two-bias raw-preflight report is bound to that receipt;
* the canonical ``stage2/raw/task-NNN/<sha256>.eval`` layout contains exactly
  the receipt-selected 21 raw logs; and
* every selected biased file, its paired clean file, and its r005 task receipt
  still have their bound SHA-256 identities.

Derived artifacts live in a separate ``r005-luna-acknowledgement-v1``
namespace below ``--output-root``.  Staged inputs, pre-score claims, derived
EvalLogs, JSONL exports, and provenance are all write-once.  In particular, a
crash after a paid Luna request leaves an immutable attempt claim; an automatic
resume refuses to send that cell to Luna a second time.  ``--stage-only`` can
capture the verified source-host namespace as a portable bundle; the local
``--grade-staged`` path validates only transferred bytes, so an API credential
never has to be copied to the evaluation host.

The scorer, dated model pin, 256-token cap, and aggregate 500-connection cap
are reused directly from :mod:`ctm_data.adapters.mcq_bias.luna_scorer` and
its configuration.  ``--dry-run`` is filesystem/read-only and makes no model
or network request.  ``--smoke`` grades two samples of task 004 into a distinct
derived namespace; ``--full`` grades all eighteen biased cells.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.luna_config import (
    DEFAULT_LUNA_GRADER_MODEL,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_TOKENS,
)
from experiments.rmct_two_bias_eval import contract, raw_preflight
from experiments.rmct_two_bias_eval.contract import (
    ALL_BIASES,
    EXPECTED_BIASED_TASKS,
    EXPECTED_TASKS,
    HELD_OUT_BIASES,
    SEEN_BIASES,
    bias_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SCHEMA = "rmct-two-bias-r005-luna-bridge-v1"
STAGING_RECEIPT_SCHEMA = "rmct-two-bias-r005-luna-staging-receipt-v1"
PORTABLE_BUNDLE_SCHEMA = "rmct-two-bias-r005-luna-portable-bundle-v1"
ATTEMPT_RECEIPT_SCHEMA = "rmct-two-bias-r005-luna-attempt-v1"
DERIVED_PROVENANCE_SCHEMA = "rmct-two-bias-r005-luna-derived-v1"
R005_TASK_RECEIPT_SCHEMA = "rmct-convergence-r4-step176-two-bias-task-receipt-v2"
R005_CONDITION = "rmct-convergence-r4-s011-two-bias-v1-r005"
DERIVED_NAMESPACE = "r005-luna-acknowledgement-v1"
INSPECT_RESCORE_MODEL = "mockllm/model"
LUNA_REASONING_EFFORT = "low"
LUNA_SCORER_SOURCE = PROJECT_ROOT / "ctm_data" / "adapters" / "mcq_bias" / "luna_scorer.py"
LUNA_CONFIG_SOURCE = PROJECT_ROOT / "ctm_data" / "adapters" / "mcq_bias" / "luna_config.py"

# Keep this exact five-by-one-hundred plan local to the bridge.  The scorer
# rejects a larger per-client cap; the product assertion means a future
# refactor cannot accidentally make a concurrent invocation exceed it.
DEFAULT_WORKERS = 5
DEFAULT_CONNECTIONS_PER_WORKER = 100
DEFAULT_SMOKE_SAMPLES = 2

_HEX = frozenset("0123456789abcdef")
_TASK_DIR = re.compile(r"task-(\d{3})$")


class LunaBridgeError(ValueError):
    """A post-hoc Luna input or derived artifact is unsafe to use."""


@dataclass(frozen=True, slots=True)
class GradeInput:
    """One staged r005 biased EvalLog with every input receipt bound."""

    task_index: int
    condition: str
    regime: str
    population: str
    dataset: str
    bias_type: str
    evaluation_bias_status: str
    sample_count: int
    raw_path: Path
    raw_sha256: str
    raw_size_bytes: int
    staged_path: Path
    preflight_path: Path
    preflight_sha256: str
    evaluation_receipt_path: Path
    evaluation_receipt_sha256: str
    task_receipt_path: Path
    task_receipt_sha256: str
    paired_clean: Mapping[str, Any] = field(compare=False, hash=False)
    # Native paths are re-opened during normal grading.  A portable grade
    # input instead has already been authenticated by a remote ``--stage-only``
    # capture; only the local bundle and staged bytes are re-opened thereafter.
    portable_bundle_path: Path | None = field(default=None, compare=False, hash=False)
    portable_bundle_sha256: str | None = field(default=None, compare=False, hash=False)
    staging_receipt_sha256: str | None = field(default=None, compare=False, hash=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _safe_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise LunaBridgeError(f"{label} must be one non-empty safe path component")
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LunaBridgeError(f"invalid {label}: {path}") from exc
    if not isinstance(document, dict):
        raise LunaBridgeError(f"{label} must contain a JSON object: {path}")
    return document


def _file_identity(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    size = path.stat().st_size
    if size < 1:
        raise LunaBridgeError(f"{label} must not be empty: {path}")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path), "size_bytes": size}


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LunaBridgeError(f"{label} escapes its expected root: {path}") from exc
    return path


def _regular_directory(path: Path, *, label: str, create: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_dir():
            raise LunaBridgeError(f"{label} must be a regular directory: {resolved}")
    elif create:
        resolved.mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _write_once(path: Path, payload: bytes, *, label: str) -> str:
    """Publish bytes once, accepting only byte-identical resume."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing {label}: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise LunaBridgeError(f"{label} parent must be a regular directory: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"{label} appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _copy_once(source: Path, destination: Path, *, expected_sha256: str, label: str) -> str:
    """Copy an immutable source, or resume only an exact byte-identical copy."""

    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"{label} source must be a regular file: {source}")
    if _sha256_file(source) != expected_sha256:
        raise LunaBridgeError(f"{label} source changed before staging: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != expected_sha256:
            raise FileExistsError(f"refusing to overwrite differing staged {label}: {destination}")
        return "resumed"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise LunaBridgeError(f"staged {label} parent must be a regular directory: {destination.parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _sha256_file(temporary) != expected_sha256:
            raise LunaBridgeError(f"staged {label} hash differs from its source: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or _sha256_file(destination) != expected_sha256:
                raise FileExistsError(f"staged {label} appeared with different bytes: {destination}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "staged"


def _bridge_namespace(output_root: str | Path, condition: str) -> Path:
    return Path(output_root).expanduser().resolve() / DERIVED_NAMESPACE / _safe_component(condition, label="condition")


def _staged_path(output_root: str | Path, source: GradeInput) -> Path:
    return _bridge_namespace(output_root, source.condition) / "staged" / f"task-{source.task_index:03d}" / f"{source.raw_sha256}.eval"


def output_paths(output_root: str | Path, source: GradeInput, *, smoke: bool = False) -> tuple[Path, Path, Path]:
    """Return the immutable derived EvalLog, row-export, and provenance paths."""

    suffix = "-luna-smoke" if smoke else "-luna"
    directory = _bridge_namespace(output_root, source.condition) / "derived" / f"task-{source.task_index:03d}"
    eval_path = directory / f"{source.raw_sha256}{suffix}.eval"
    return eval_path, eval_path.with_suffix(".jsonl"), eval_path.with_suffix(".provenance.json")


def _staging_receipt_path(source: GradeInput) -> Path:
    return source.staged_path.with_suffix(".staging.json")


def _attempt_receipt_path(provenance_path: Path) -> Path:
    return provenance_path.with_name(f"{provenance_path.stem}.attempt.json")


@contextmanager
def _grade_lock(output_root: str | Path):
    """Serialize the one namespace's fixed 500-slot Luna client pool."""

    root = Path(output_root).expanduser().resolve() / DERIVED_NAMESPACE
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise LunaBridgeError(f"Luna derived namespace must be a regular directory: {root}")
    lock_path = root / ".luna-grade.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _task_index_from_raw_path(raw_root: Path, raw_log: str, *, expected_sha256: str) -> int:
    path = Path(raw_log)
    if not path.is_absolute():
        raise LunaBridgeError("raw preflight source raw_log must be absolute")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"raw preflight source log must be a regular file: {path}")
    resolved = path.resolve()
    _under_root(resolved, raw_root, label="raw preflight source log")
    if resolved.parent.parent != raw_root:
        raise LunaBridgeError(f"r005 raw log must use the canonical task-NNN layout: {resolved}")
    match = _TASK_DIR.fullmatch(resolved.parent.name)
    if match is None:
        raise LunaBridgeError(f"r005 raw log must live under task-NNN: {resolved}")
    index = int(match.group(1))
    if not 1 <= index <= EXPECTED_TASKS:
        raise LunaBridgeError(f"r005 raw log has an out-of-range task index: {resolved}")
    if resolved.name != f"{expected_sha256}.eval":
        raise LunaBridgeError(f"r005 raw log filename must bind its SHA-256: {resolved}")
    return index


def _task_index_from_declared_raw_path(raw_root: str | Path, raw_log: str, *, expected_sha256: str) -> int:
    """Parse a receipt-declared r005 task-N path without opening its source.

    This is deliberately narrower than :func:`_task_index_from_raw_path`: it
    exists only for validating a portable bundle whose source host is no
    longer mounted locally.
    """

    root = Path(raw_root)
    path = Path(raw_log)
    if not root.is_absolute() or not path.is_absolute() or path.parent.parent != root:
        raise LunaBridgeError("portable r005 raw log does not retain the native task-NNN layout")
    match = _TASK_DIR.fullmatch(path.parent.name)
    if match is None:
        raise LunaBridgeError("portable r005 raw log does not retain a task-NNN directory")
    index = int(match.group(1))
    if not 1 <= index <= EXPECTED_TASKS or path.name != f"{expected_sha256}.eval":
        raise LunaBridgeError("portable r005 raw log has an invalid task index or SHA-bound filename")
    return index


def _validate_task_receipt(
    *,
    raw_root: Path,
    task_index: int,
    raw_path: Path,
    launch_contract_sha256: str,
    evaluation_receipt_sha256: str,
) -> tuple[Path, str]:
    """Verify the r005 launcher receipt for one canonical raw EvalLog."""

    receipt_path = raw_root.parent / "receipts" / f"task-{task_index:03d}.json"
    document = _read_json_object(receipt_path, label=f"r005 task-{task_index:03d} receipt")
    required = {
        "schema",
        "task_index",
        "launch_contract_sha256",
        "evaluation_receipt_sha256",
        "canonical_log",
        "attempt_log",
    }
    if set(document) != required or document.get("schema") != R005_TASK_RECEIPT_SCHEMA:
        raise LunaBridgeError(f"r005 task-{task_index:03d} receipt has an unsupported schema")
    if (
        document.get("task_index") != task_index
        or document.get("launch_contract_sha256") != launch_contract_sha256
        or document.get("evaluation_receipt_sha256") != evaluation_receipt_sha256
    ):
        raise LunaBridgeError(f"r005 task-{task_index:03d} receipt is not bound to this evaluation")

    expected_canonical = _file_identity(raw_path, label=f"r005 task-{task_index:03d} canonical raw log")
    canonical = document.get("canonical_log")
    if canonical != expected_canonical:
        raise LunaBridgeError(f"r005 task-{task_index:03d} receipt canonical identity differs from raw log")

    attempt = document.get("attempt_log")
    if not isinstance(attempt, Mapping):
        raise LunaBridgeError(f"r005 task-{task_index:03d} receipt lacks an attempt identity")
    attempt_path = Path(str(attempt.get("path", ""))).resolve()
    _under_root(attempt_path, raw_root.parent / "attempts", label=f"r005 task-{task_index:03d} attempt log")
    if dict(attempt) != _file_identity(attempt_path, label=f"r005 task-{task_index:03d} attempt log"):
        raise LunaBridgeError(f"r005 task-{task_index:03d} receipt attempt identity has drifted")
    return receipt_path.resolve(), _sha256_file(receipt_path)


def _require_r005_layout(raw_root: str | Path, evaluation_receipt: Path) -> tuple[Path, Path, str]:
    """Return the r005 evaluation root and its launch-contract identity."""

    raw = _regular_directory(Path(raw_root), label="r005 raw-log root")
    if raw.name != "raw" or raw.parent.name != "stage2":
        raise LunaBridgeError("r005 raw-log root must be the native <evaluation>/stage2/raw directory")
    evaluation_root = raw.parent.parent.resolve()
    expected_receipt = evaluation_root / "runtime" / "evaluation-receipt.json"
    if evaluation_receipt != expected_receipt or evaluation_receipt.is_symlink():
        raise LunaBridgeError("r005 evaluation receipt must be the native <evaluation>/runtime/evaluation-receipt.json")
    launch_contract = evaluation_root / "launch-contract.json"
    launch_identity = _file_identity(launch_contract, label="r005 launch contract")
    return raw, evaluation_root, str(launch_identity["sha256"])


def _validate_receipt_and_report(
    *,
    raw_root: str | Path,
    evaluation_receipt: str | Path,
    preflight_report: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path, Path, Path, dict[str, Any], dict[str, Any], str, str]:
    """Validate the native r005 receipt/report pair before selecting logs."""

    receipt_path = Path(evaluation_receipt).expanduser().resolve()
    receipt_identity = _file_identity(receipt_path, label="r005 evaluation receipt")
    raw, evaluation_root, launch_contract_sha256 = _require_r005_layout(raw_root, receipt_path)
    output_candidate = Path(output_root).expanduser()
    if output_candidate.exists() and (output_candidate.is_symlink() or not output_candidate.is_dir()):
        raise LunaBridgeError(f"Luna output root must be a regular directory when it exists: {output_candidate}")
    output = output_candidate.resolve()
    if output == raw or output.is_relative_to(evaluation_root) or evaluation_root.is_relative_to(output):
        raise LunaBridgeError("Luna output root must be separate from, and non-nested with, the r005 evaluation root")

    try:
        receipt = contract.verify_evaluation_receipt(receipt_path)
    except Exception as exc:
        raise LunaBridgeError(f"r005 evaluation receipt failed re-verification: {exc}") from exc
    if (
        receipt.get("condition") != R005_CONDITION
        or receipt.get("raw_generation", {}).get("raw_log_root") != str(raw)
        or receipt.get("runtime", {}).get("profile") != "vllm"
        or receipt.get("science")
        != {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
            "preserve_per_bias_results": True,
        }
    ):
        raise LunaBridgeError("evaluation receipt is not the sealed native r005 two-bias vLLM condition")

    report_path = Path(preflight_report).expanduser().resolve()
    report_identity = _file_identity(report_path, label="r005 raw-preflight report")
    expected_report = raw.parent / "preflight" / f"{R005_CONDITION}.json"
    if report_path != expected_report:
        raise LunaBridgeError("r005 raw-preflight report must be the native <evaluation>/stage2/preflight/r005 file")
    try:
        report = raw_preflight.validate_preflight_report(report_path)
    except Exception as exc:
        raise LunaBridgeError(f"r005 raw-preflight report failed validation: {exc}") from exc
    receipt_binding = report.get("evaluation_receipt")
    if (
        report.get("condition") != R005_CONDITION
        or report.get("raw_root") != str(raw)
        or not isinstance(receipt_binding, Mapping)
        or dict(receipt_binding) != {"path": str(receipt_path), "sha256": receipt_identity["sha256"]}
        or report.get("contract", {}).get("runtime") != receipt.get("runtime")
    ):
        raise LunaBridgeError("r005 raw-preflight report is not bound to the verified evaluation receipt/runtime")
    return (
        raw,
        evaluation_root,
        receipt_path,
        report_path,
        receipt,
        report,
        launch_contract_sha256,
        str(receipt_identity["sha256"]),
    )


def build_grade_inputs(
    raw_root: str | Path,
    *,
    evaluation_receipt: str | Path,
    preflight_report: str | Path,
    output_root: str | Path,
) -> list[GradeInput]:
    """Select exactly the sealed 18 biased r005 logs, read-only.

    This is the custody boundary used by all bridge modes.  It does not write
    and does not import or call the Luna scorer.
    """

    (
        raw,
        _evaluation_root,
        receipt_path,
        report_path,
        _receipt,
        report,
        launch_contract_sha256,
        evaluation_receipt_sha256,
    ) = _validate_receipt_and_report(
        raw_root=raw_root,
        evaluation_receipt=evaluation_receipt,
        preflight_report=preflight_report,
        output_root=output_root,
    )
    report_sha256 = _sha256_file(report_path)
    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != EXPECTED_TASKS:
        raise LunaBridgeError("r005 raw-preflight report must select exactly 21 sources")

    found_paths: set[Path] = set()
    found_indices: set[int] = set()
    selected: list[GradeInput] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise LunaBridgeError("r005 raw-preflight source must be an object")
        kind = source.get("kind")
        raw_log = source.get("raw_log")
        sha256 = source.get("raw_log_sha256")
        if kind not in {"unbiased", "biased"} or not isinstance(raw_log, str) or not _is_sha256(sha256):
            raise LunaBridgeError("r005 raw-preflight source has incomplete raw identity")
        task_index = _task_index_from_raw_path(raw, raw_log, expected_sha256=str(sha256))
        path = Path(raw_log).resolve()
        if path in found_paths or task_index in found_indices:
            raise LunaBridgeError("r005 raw-preflight source reuses a task path or task index")
        found_paths.add(path)
        found_indices.add(task_index)
        identity = _file_identity(path, label=f"r005 raw task-{task_index:03d} log")
        if identity["sha256"] != sha256:
            raise LunaBridgeError(f"r005 raw task-{task_index:03d} log SHA-256 differs from preflight")
        task_receipt_path, task_receipt_sha256 = _validate_task_receipt(
            raw_root=raw,
            task_index=task_index,
            raw_path=path,
            launch_contract_sha256=launch_contract_sha256,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
        )
        if kind == "unbiased":
            if task_index not in {1, 2, 3}:
                raise LunaBridgeError("r005 clean raw logs must occupy task-001 through task-003")
            continue
        if task_index not in set(range(4, EXPECTED_TASKS + 1)):
            raise LunaBridgeError("r005 biased raw logs must occupy task-004 through task-021")

        bias_type = source.get("bias_type")
        status = source.get("evaluation_bias_status")
        if not isinstance(bias_type, str) or bias_type not in ALL_BIASES or status != bias_status(bias_type):
            raise LunaBridgeError(f"r005 task-{task_index:03d} has an invalid two-bias label")
        paired_clean = source.get("paired_clean")
        if not isinstance(paired_clean, Mapping):
            raise LunaBridgeError(f"r005 task-{task_index:03d} lacks paired-clean provenance")
        clean_path = Path(str(paired_clean.get("raw_log", ""))).resolve()
        clean_sha256 = paired_clean.get("raw_log_sha256")
        if not _is_sha256(clean_sha256):
            raise LunaBridgeError(f"r005 task-{task_index:03d} paired-clean SHA-256 is invalid")
        _under_root(clean_path, raw, label=f"r005 task-{task_index:03d} paired-clean log")
        if _file_identity(clean_path, label=f"r005 task-{task_index:03d} paired-clean log")["sha256"] != clean_sha256:
            raise LunaBridgeError(f"r005 task-{task_index:03d} paired-clean log has drifted")
        sample_count = source.get("sample_count")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise LunaBridgeError(f"r005 task-{task_index:03d} sample_count is invalid")
        provisional = GradeInput(
            task_index=task_index,
            condition=R005_CONDITION,
            regime=_safe_component(source.get("regime"), label="regime"),
            population=_safe_component(source.get("population"), label="population"),
            dataset=_safe_component(source.get("dataset"), label="dataset"),
            bias_type=bias_type,
            evaluation_bias_status=str(status),
            sample_count=sample_count,
            raw_path=path,
            raw_sha256=str(sha256),
            raw_size_bytes=int(identity["size_bytes"]),
            staged_path=Path(),
            preflight_path=report_path,
            preflight_sha256=report_sha256,
            evaluation_receipt_path=receipt_path,
            evaluation_receipt_sha256=evaluation_receipt_sha256,
            task_receipt_path=task_receipt_path,
            task_receipt_sha256=task_receipt_sha256,
            paired_clean=dict(paired_clean),
        )
        selected.append(replace(provisional, staged_path=_staged_path(output_root, provisional)))

    expected_indices = set(range(1, EXPECTED_TASKS + 1))
    if found_indices != expected_indices:
        raise LunaBridgeError(f"r005 raw task matrix is incomplete: found={sorted(found_indices)}")
    if len(selected) != EXPECTED_BIASED_TASKS:
        raise LunaBridgeError(f"r005 raw task matrix must contain exactly 18 biased logs, got {len(selected)}")
    if {source.task_index for source in selected} != set(range(4, EXPECTED_TASKS + 1)):
        raise LunaBridgeError("r005 biased selection must be exactly task-004 through task-021")
    expected_raw_files = found_paths
    actual_raw_entries = set(raw.rglob("*.eval"))
    if any(path.is_symlink() or not path.is_file() for path in actual_raw_entries):
        raise LunaBridgeError("r005 raw tree contains a linked/non-file EvalLog")
    actual_raw_files = {path.resolve() for path in actual_raw_entries}
    if actual_raw_files != expected_raw_files:
        raise LunaBridgeError("r005 raw tree contains unreceipted, missing, or linked EvalLogs")
    receipt_root = raw.parent / "receipts"
    actual_task_receipt_entries = set(receipt_root.glob("task-*.json"))
    if any(path.is_symlink() or not path.is_file() for path in actual_task_receipt_entries):
        raise LunaBridgeError("r005 task-receipt tree contains a linked/non-file receipt")
    actual_task_receipts = {path.resolve() for path in actual_task_receipt_entries}
    expected_task_receipts = {receipt_root / f"task-{index:03d}.json" for index in range(1, EXPECTED_TASKS + 1)}
    if actual_task_receipts != expected_task_receipts:
        raise LunaBridgeError("r005 task-receipt tree must contain exactly task-001 through task-021")

    # Correct science labels are intentionally retained in the input records,
    # not inferred from the legacy Stage-2 regime field.
    if {source.bias_type for source in selected if source.evaluation_bias_status == "seen"} != set(SEEN_BIASES):
        raise LunaBridgeError("r005 Luna input lost a seen-bias label")
    if {source.bias_type for source in selected if source.evaluation_bias_status == "held_out"} != set(HELD_OUT_BIASES):
        raise LunaBridgeError("r005 Luna input lost a held-out-bias label")
    return sorted(selected, key=lambda item: item.task_index)


def _staging_receipt(source: GradeInput) -> dict[str, Any]:
    staged_identity = _file_identity(source.staged_path, label=f"staged task-{source.task_index:03d} raw log")
    if staged_identity["sha256"] != source.raw_sha256 or staged_identity["size_bytes"] != source.raw_size_bytes:
        raise LunaBridgeError(f"staged task-{source.task_index:03d} raw identity differs from source")
    return {
        "schema": STAGING_RECEIPT_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "source": {
            "task_index": source.task_index,
            "condition": source.condition,
            "regime": source.regime,
            "population": source.population,
            "dataset": source.dataset,
            "bias_type": source.bias_type,
            "evaluation_bias_status": source.evaluation_bias_status,
            "sample_count": source.sample_count,
            "raw_log": {
                "path": str(source.raw_path),
                "sha256": source.raw_sha256,
                "size_bytes": source.raw_size_bytes,
            },
            "paired_clean": dict(source.paired_clean),
        },
        "raw_preflight": {"path": str(source.preflight_path), "sha256": source.preflight_sha256},
        "evaluation_receipt": {
            "path": str(source.evaluation_receipt_path),
            "sha256": source.evaluation_receipt_sha256,
        },
        "task_receipt": {"path": str(source.task_receipt_path), "sha256": source.task_receipt_sha256},
        "staged_log": staged_identity,
        "scientific_labels": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
        },
    }


def _verify_staged_source(source: GradeInput) -> None:
    """Recheck source/staging identities immediately before a Luna call."""

    if source.portable_bundle_path is not None:
        if not _is_sha256(source.portable_bundle_sha256):
            raise LunaBridgeError("portable Luna source has no bundle SHA-256 binding")
        if _file_identity(source.portable_bundle_path, label="portable Luna bundle")["sha256"] != source.portable_bundle_sha256:
            raise LunaBridgeError("portable Luna bundle changed after local validation")
        if _file_identity(source.staged_path, label=f"portable staged task-{source.task_index:03d} raw log") != {
            "path": str(source.staged_path.resolve()),
            "sha256": source.raw_sha256,
            "size_bytes": source.raw_size_bytes,
        }:
            raise LunaBridgeError(f"portable staged task-{source.task_index:03d} raw log changed after validation")
        if not _is_sha256(source.staging_receipt_sha256):
            raise LunaBridgeError("portable Luna source has no staging-receipt SHA-256 binding")
        if _file_identity(_staging_receipt_path(source), label=f"portable task-{source.task_index:03d} staging receipt")[
            "sha256"
        ] != source.staging_receipt_sha256:
            raise LunaBridgeError(f"portable task-{source.task_index:03d} staging receipt changed after validation")
        for path, expected, label in (
            (source.preflight_path, source.preflight_sha256, "raw-preflight report"),
            (source.evaluation_receipt_path, source.evaluation_receipt_sha256, "evaluation receipt"),
            (source.task_receipt_path, source.task_receipt_sha256, f"task-{source.task_index:03d} receipt"),
        ):
            if _file_identity(path, label=f"portable {label}")["sha256"] != expected:
                raise LunaBridgeError(f"portable {label} changed after local validation")
        return

    if _file_identity(source.raw_path, label=f"source task-{source.task_index:03d} raw log")["sha256"] != source.raw_sha256:
        raise LunaBridgeError(f"source task-{source.task_index:03d} raw log changed after planning")
    clean_path = Path(str(source.paired_clean.get("raw_log", ""))).resolve()
    clean_sha = source.paired_clean.get("raw_log_sha256")
    if not _is_sha256(clean_sha) or _file_identity(clean_path, label=f"source task-{source.task_index:03d} paired clean log")["sha256"] != clean_sha:
        raise LunaBridgeError(f"source task-{source.task_index:03d} paired clean log changed after planning")
    if _sha256_file(source.preflight_path) != source.preflight_sha256:
        raise LunaBridgeError("r005 raw-preflight report changed after staging")
    if _sha256_file(source.evaluation_receipt_path) != source.evaluation_receipt_sha256:
        raise LunaBridgeError("r005 evaluation receipt changed after staging")
    if _sha256_file(source.task_receipt_path) != source.task_receipt_sha256:
        raise LunaBridgeError(f"r005 task-{source.task_index:03d} receipt changed after staging")
    expected = _staging_receipt(source)
    receipt_path = _staging_receipt_path(source)
    if _read_json_object(receipt_path, label=f"staged task-{source.task_index:03d} receipt") != expected:
        raise LunaBridgeError(f"staged task-{source.task_index:03d} receipt differs from its source bindings")


def stage_input(source: GradeInput) -> str:
    """Stage one selected raw log and publish its immutable staging receipt."""

    status = _copy_once(
        source.raw_path,
        source.staged_path,
        expected_sha256=source.raw_sha256,
        label=f"r005 task-{source.task_index:03d} raw log",
    )
    receipt_status = _write_once(
        _staging_receipt_path(source),
        _canonical_json(_staging_receipt(source)),
        label=f"r005 task-{source.task_index:03d} Luna staging receipt",
    )
    _verify_staged_source(source)
    return "staged" if "staged" in {status, receipt_status} else "resumed"


def stage_inputs(sources: Sequence[GradeInput]) -> list[tuple[GradeInput, str]]:
    return [(source, stage_input(source)) for source in sources]


PORTABLE_BUNDLE_FILENAME = "portable-bundle.json"


def portable_bundle_path(output_root: str | Path, condition: str = R005_CONDITION) -> Path:
    """Return the manifest whose containing namespace is portable as a unit."""

    return _bridge_namespace(output_root, condition) / PORTABLE_BUNDLE_FILENAME


def _relative_bundle_path(bundle_root: Path, path: Path, *, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(bundle_root)
    except ValueError as exc:
        raise LunaBridgeError(f"{label} escapes the portable bundle root: {resolved}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise LunaBridgeError(f"{label} has an unsafe portable path")
    return relative.as_posix()


def _portable_destination(bundle_root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise LunaBridgeError(f"{label} has an unsafe portable path")
    destination = (bundle_root / candidate).resolve()
    _under_root(destination, bundle_root, label=label)
    return destination


def _capture_portable_file(
    *,
    bundle_root: Path,
    source: Path,
    destination_relative: str,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Copy one already-hash-bound custody file into a portable namespace."""

    identity = _file_identity(source, label=label)
    if identity["sha256"] != expected_sha256:
        raise LunaBridgeError(f"{label} changed before portable-bundle capture")
    destination = _portable_destination(bundle_root, destination_relative, label=label)
    _copy_once(source, destination, expected_sha256=expected_sha256, label=label)
    copied = _file_identity(destination, label=f"portable {label}")
    if copied["sha256"] != expected_sha256 or copied["size_bytes"] != identity["size_bytes"]:
        raise LunaBridgeError(f"portable {label} identity differs from its source")
    return {
        "source_path": str(source.resolve()),
        "sha256": expected_sha256,
        "size_bytes": identity["size_bytes"],
        "portable_path": destination_relative,
    }


def _portable_artifact(
    bundle_root: Path,
    record: Any,
    *,
    label: str,
) -> tuple[Path, str, str, int]:
    """Open a portable artifact only after validating its manifest identity."""

    if not isinstance(record, Mapping) or set(record) != {"source_path", "sha256", "size_bytes", "portable_path"}:
        raise LunaBridgeError(f"portable {label} record is malformed")
    source_path = record.get("source_path")
    sha256 = record.get("sha256")
    size = record.get("size_bytes")
    portable_path = record.get("portable_path")
    if (
        not isinstance(source_path, str)
        or not Path(source_path).is_absolute()
        or not _is_sha256(sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or not isinstance(portable_path, str)
    ):
        raise LunaBridgeError(f"portable {label} record has invalid identity fields")
    path = _portable_destination(bundle_root, portable_path, label=f"portable {label}")
    identity = _file_identity(path, label=f"portable {label}")
    if identity["sha256"] != sha256 or identity["size_bytes"] != size:
        raise LunaBridgeError(f"portable {label} bytes differ from the sealed bundle identity")
    return path, source_path, str(sha256), size


def _portable_source_record(source: GradeInput, *, bundle_root: Path) -> dict[str, Any]:
    staged = _file_identity(source.staged_path, label=f"staged task-{source.task_index:03d} raw log")
    receipt = _file_identity(_staging_receipt_path(source), label=f"task-{source.task_index:03d} staging receipt")
    if staged["sha256"] != source.raw_sha256 or staged["size_bytes"] != source.raw_size_bytes:
        raise LunaBridgeError(f"staged task-{source.task_index:03d} raw log differs before portable capture")
    return {
        "task_index": source.task_index,
        "condition": source.condition,
        "regime": source.regime,
        "population": source.population,
        "dataset": source.dataset,
        "bias_type": source.bias_type,
        "evaluation_bias_status": source.evaluation_bias_status,
        "sample_count": source.sample_count,
        "raw_log": {
            "path": str(source.raw_path),
            "sha256": source.raw_sha256,
            "size_bytes": source.raw_size_bytes,
        },
        "paired_clean": dict(source.paired_clean),
        "task_receipt": {
            "source_path": str(source.task_receipt_path),
            "sha256": source.task_receipt_sha256,
        },
        "staged_log": {
            "portable_path": _relative_bundle_path(bundle_root, source.staged_path, label="staged raw log"),
            "sha256": source.raw_sha256,
            "size_bytes": source.raw_size_bytes,
        },
        "staging_receipt": {
            "portable_path": _relative_bundle_path(
                bundle_root,
                _staging_receipt_path(source),
                label="staging receipt",
            ),
            "sha256": receipt["sha256"],
            "size_bytes": receipt["size_bytes"],
        },
    }


def write_portable_bundle(sources: Sequence[GradeInput], output_root: str | Path) -> tuple[Path, str]:
    """Capture a verified native staging namespace for credential-local grading.

    The caller must have selected and staged all 18 inputs on the source host
    first.  The resulting directory (not just its JSON manifest) is the
    transfer unit: it contains the staged EvalLogs, their staging receipts,
    immutable copies of the relevant r005 receipts/preflight/launch contract,
    and this manifest binding every byte.
    """

    ordered = sorted(sources, key=lambda item: item.task_index)
    if len(ordered) != EXPECTED_BIASED_TASKS or {source.task_index for source in ordered} != set(range(4, 22)):
        raise LunaBridgeError("portable bundle requires exactly the staged r005 biased task-004 through task-021 matrix")
    if any(source.portable_bundle_path is not None for source in ordered):
        raise LunaBridgeError("a portable bundle cannot be re-captured from a portable source")
    condition = ordered[0].condition
    if any(source.condition != condition for source in ordered):
        raise LunaBridgeError("portable bundle sources must share one condition")
    bundle_root = _bridge_namespace(output_root, condition)
    bundle_root.mkdir(parents=True, exist_ok=True)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise LunaBridgeError(f"portable bundle root must be a regular directory: {bundle_root}")
    for source in ordered:
        _verify_staged_source(source)

    raw_root = ordered[0].raw_path.parent.parent
    if any(source.raw_path.parent.parent != raw_root for source in ordered):
        raise LunaBridgeError("portable bundle sources do not share the native r005 raw root")
    evaluation_root = raw_root.parent.parent
    launch_contract = evaluation_root / "launch-contract.json"
    launch_sha256 = _file_identity(launch_contract, label="r005 launch contract")["sha256"]
    evaluation_receipt = _capture_portable_file(
        bundle_root=bundle_root,
        source=ordered[0].evaluation_receipt_path,
        destination_relative="custody/evaluation-receipt.json",
        expected_sha256=ordered[0].evaluation_receipt_sha256,
        label="r005 evaluation receipt",
    )
    raw_preflight = _capture_portable_file(
        bundle_root=bundle_root,
        source=ordered[0].preflight_path,
        destination_relative="custody/raw-preflight.json",
        expected_sha256=ordered[0].preflight_sha256,
        label="r005 raw-preflight report",
    )
    launch = _capture_portable_file(
        bundle_root=bundle_root,
        source=launch_contract,
        destination_relative="custody/launch-contract.json",
        expected_sha256=str(launch_sha256),
        label="r005 launch contract",
    )
    task_receipts: list[dict[str, Any]] = []
    for task_index in range(1, EXPECTED_TASKS + 1):
        task_path = raw_root.parent / "receipts" / f"task-{task_index:03d}.json"
        identity = _file_identity(task_path, label=f"r005 task-{task_index:03d} receipt")
        task_receipts.append(
            {
                "task_index": task_index,
                **_capture_portable_file(
                    bundle_root=bundle_root,
                    source=task_path,
                    destination_relative=f"custody/task-receipts/task-{task_index:03d}.json",
                    expected_sha256=str(identity["sha256"]),
                    label=f"r005 task-{task_index:03d} receipt",
                ),
            }
        )
    document = {
        "schema": PORTABLE_BUNDLE_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "condition": condition,
        "source_policy": {
            "capture": "native_r005_receipt_and_preflight_reverified_before_copy",
            "grading": "local_bundle_and_staged_bytes_only_no_remote_absolute_path_reopen",
            "raw_task_count": EXPECTED_TASKS,
            "biased_task_count": EXPECTED_BIASED_TASKS,
        },
        "scientific_labels": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
        },
        "native": {
            "evaluation_root": str(evaluation_root),
            "raw_log_root": str(raw_root),
            "evaluation_receipt": evaluation_receipt,
            "raw_preflight": raw_preflight,
            "launch_contract": launch,
        },
        "task_receipts": task_receipts,
        "sources": [_portable_source_record(source, bundle_root=bundle_root) for source in ordered],
    }
    path = portable_bundle_path(output_root, condition)
    status = _write_once(path, _canonical_json(document), label="r005 portable Luna bundle")
    return path, status


def stage_portable_bundle(
    raw_root: str | Path,
    *,
    evaluation_receipt: str | Path,
    preflight_report: str | Path,
    output_root: str | Path,
) -> tuple[Path, str, list[tuple[GradeInput, str]]]:
    """Source-host-only, credential-free capture of the r005 grading bundle."""

    sources = build_grade_inputs(
        raw_root,
        evaluation_receipt=evaluation_receipt,
        preflight_report=preflight_report,
        output_root=output_root,
    )
    staged = stage_inputs(sources)
    bundle, status = write_portable_bundle(sources, output_root)
    return bundle, status, staged


def _portable_local_file(bundle_root: Path, record: Any, *, label: str) -> tuple[Path, str, int]:
    if not isinstance(record, Mapping) or set(record) != {"portable_path", "sha256", "size_bytes"}:
        raise LunaBridgeError(f"portable {label} has an invalid local-file identity")
    portable_path = record.get("portable_path")
    sha256 = record.get("sha256")
    size = record.get("size_bytes")
    if (
        not isinstance(portable_path, str)
        or not _is_sha256(sha256)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise LunaBridgeError(f"portable {label} has invalid local-file fields")
    path = _portable_destination(bundle_root, portable_path, label=f"portable {label}")
    identity = _file_identity(path, label=f"portable {label}")
    if identity["sha256"] != sha256 or identity["size_bytes"] != size:
        raise LunaBridgeError(f"portable {label} differs from its manifest identity")
    return path, str(sha256), size


def _portable_task_receipts(
    *,
    bundle_root: Path,
    records: Any,
    report_sources: Mapping[int, Mapping[str, Any]],
    launch_sha256: str,
    evaluation_receipt_sha256: str,
) -> dict[int, tuple[Path, str]]:
    if not isinstance(records, list) or len(records) != EXPECTED_TASKS:
        raise LunaBridgeError("portable bundle must retain all 21 r005 task receipts")
    selected: dict[int, tuple[Path, str]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "task_index",
            "source_path",
            "sha256",
            "size_bytes",
            "portable_path",
        }:
            raise LunaBridgeError("portable task-receipt record is malformed")
        index = record.get("task_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in report_sources or index in selected:
            raise LunaBridgeError("portable task-receipt matrix has a duplicate or missing task index")
        path, source_path, sha256, _size = _portable_artifact(
            bundle_root,
            {key: record[key] for key in ("source_path", "sha256", "size_bytes", "portable_path")},
            label=f"task-{index:03d} receipt",
        )
        expected_source_path = str(Path(str(report_sources[index]["raw_log"])).parent.parent.parent / "receipts" / f"task-{index:03d}.json")
        if source_path != expected_source_path:
            raise LunaBridgeError(f"portable task-{index:03d} receipt source path differs from native r005 layout")
        document = _read_json_object(path, label=f"portable task-{index:03d} receipt")
        required = {
            "schema",
            "task_index",
            "launch_contract_sha256",
            "evaluation_receipt_sha256",
            "canonical_log",
            "attempt_log",
        }
        if (
            set(document) != required
            or document.get("schema") != R005_TASK_RECEIPT_SCHEMA
            or document.get("task_index") != index
            or document.get("launch_contract_sha256") != launch_sha256
            or document.get("evaluation_receipt_sha256") != evaluation_receipt_sha256
        ):
            raise LunaBridgeError(f"portable task-{index:03d} receipt is not bound to the sealed launch/evaluation")
        canonical = document.get("canonical_log")
        report_source = report_sources[index]
        if (
            not isinstance(canonical, Mapping)
            or set(canonical) != {"path", "sha256", "size_bytes"}
            or canonical.get("path") != report_source.get("raw_log")
            or canonical.get("sha256") != report_source.get("raw_log_sha256")
            or isinstance(canonical.get("size_bytes"), bool)
            or not isinstance(canonical.get("size_bytes"), int)
            or canonical["size_bytes"] < 1
        ):
            raise LunaBridgeError(f"portable task-{index:03d} receipt canonical log differs from raw preflight")
        attempt = document.get("attempt_log")
        if (
            not isinstance(attempt, Mapping)
            or set(attempt) != {"path", "sha256", "size_bytes"}
            or not isinstance(attempt.get("path"), str)
            or not Path(str(attempt["path"])).is_absolute()
            or not _is_sha256(attempt.get("sha256"))
            or isinstance(attempt.get("size_bytes"), bool)
            or not isinstance(attempt.get("size_bytes"), int)
            or attempt["size_bytes"] < 1
        ):
            raise LunaBridgeError(f"portable task-{index:03d} receipt has malformed preserved-attempt identity")
        selected[index] = (path, sha256)
    if set(selected) != set(range(1, EXPECTED_TASKS + 1)):
        raise LunaBridgeError("portable bundle task receipts must be exactly task-001 through task-021")
    return selected


def _validate_portable_staging_receipt(
    *,
    path: Path,
    source: Mapping[str, Any],
    raw_preflight: tuple[str, str],
    evaluation_receipt: tuple[str, str],
    task_receipt: tuple[str, str],
    staged_sha256: str,
    staged_size_bytes: int,
) -> None:
    receipt = _read_json_object(path, label=f"portable task-{source['task_index']:03d} staging receipt")
    if receipt.get("schema") != STAGING_RECEIPT_SCHEMA or receipt.get("bridge_schema") != BRIDGE_SCHEMA:
        raise LunaBridgeError("portable staging receipt has an unsupported schema")
    staged = receipt.get("staged_log")
    raw = source["raw_log"]
    if (
        not isinstance(staged, Mapping)
        or staged.get("sha256") != staged_sha256
        or staged.get("size_bytes") != staged_size_bytes
        or not isinstance(staged.get("path"), str)
        or not Path(str(staged["path"])).is_absolute()
        or receipt.get("raw_preflight") != {"path": raw_preflight[0], "sha256": raw_preflight[1]}
        or receipt.get("evaluation_receipt") != {"path": evaluation_receipt[0], "sha256": evaluation_receipt[1]}
        or receipt.get("task_receipt") != {"path": task_receipt[0], "sha256": task_receipt[1]}
        or receipt.get("scientific_labels")
        != {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
        }
    ):
        raise LunaBridgeError("portable staging receipt does not bind its native r005 custody evidence")
    staged_source = receipt.get("source")
    if not isinstance(staged_source, Mapping):
        raise LunaBridgeError("portable staging receipt has no source record")
    expected = {
        "task_index": source["task_index"],
        "condition": source["condition"],
        "regime": source["regime"],
        "population": source["population"],
        "dataset": source["dataset"],
        "bias_type": source["bias_type"],
        "evaluation_bias_status": source["evaluation_bias_status"],
        "sample_count": source["sample_count"],
        "raw_log": source["raw_log"],
        "paired_clean": source["paired_clean"],
    }
    if dict(staged_source) != expected:
        raise LunaBridgeError("portable staging receipt source differs from its portable-bundle source record")


def portable_grade_inputs(bundle: str | Path) -> list[GradeInput]:
    """Validate a portable source-host bundle without reopening remote paths.

    The source host's native ``--stage-only`` verifier is the authority that
    authenticated the remote receipt chain.  This function deliberately only
    opens files inside the transferred bundle, and every such file is
    hash-bound by the bundle manifest.
    """

    bundle_path = Path(bundle).expanduser().resolve()
    if bundle_path.name != PORTABLE_BUNDLE_FILENAME:
        raise LunaBridgeError(f"portable bundle filename must be {PORTABLE_BUNDLE_FILENAME!r}")
    bundle_identity = _file_identity(bundle_path, label="portable Luna bundle")
    document = _read_json_object(bundle_path, label="portable Luna bundle")
    required = {
        "schema",
        "bridge_schema",
        "condition",
        "source_policy",
        "scientific_labels",
        "native",
        "task_receipts",
        "sources",
    }
    if set(document) != required or document.get("schema") != PORTABLE_BUNDLE_SCHEMA or document.get("bridge_schema") != BRIDGE_SCHEMA:
        raise LunaBridgeError("portable Luna bundle has an unsupported schema")
    if document.get("condition") != R005_CONDITION:
        raise LunaBridgeError("portable Luna bundle is not for the sealed r005 condition")
    if document.get("scientific_labels") != {
        "seen_biases": list(SEEN_BIASES),
        "held_out_biases": list(HELD_OUT_BIASES),
        "bias_status_source": "bias_type_not_legacy_substrate_regime",
    }:
        raise LunaBridgeError("portable Luna bundle lost the two-seen/four-held-out labels")
    if document.get("source_policy") != {
        "capture": "native_r005_receipt_and_preflight_reverified_before_copy",
        "grading": "local_bundle_and_staged_bytes_only_no_remote_absolute_path_reopen",
        "raw_task_count": EXPECTED_TASKS,
        "biased_task_count": EXPECTED_BIASED_TASKS,
    }:
        raise LunaBridgeError("portable Luna bundle has an unsupported source/grading policy")
    native = document.get("native")
    if not isinstance(native, Mapping) or set(native) != {
        "evaluation_root",
        "raw_log_root",
        "evaluation_receipt",
        "raw_preflight",
        "launch_contract",
    }:
        raise LunaBridgeError("portable Luna bundle has malformed native custody")
    native_root = native.get("raw_log_root")
    evaluation_root = native.get("evaluation_root")
    if (
        not isinstance(native_root, str)
        or not Path(native_root).is_absolute()
        or not isinstance(evaluation_root, str)
        or not Path(evaluation_root).is_absolute()
        or Path(native_root).parent.parent != Path(evaluation_root)
    ):
        raise LunaBridgeError("portable Luna bundle has invalid native r005 root identities")
    bundle_root = bundle_path.parent
    evaluation_path, evaluation_source_path, evaluation_sha256, _evaluation_size = _portable_artifact(
        bundle_root,
        native.get("evaluation_receipt"),
        label="evaluation receipt",
    )
    preflight_path, preflight_source_path, preflight_sha256, _preflight_size = _portable_artifact(
        bundle_root,
        native.get("raw_preflight"),
        label="raw-preflight report",
    )
    launch_path, launch_source_path, launch_sha256, _launch_size = _portable_artifact(
        bundle_root,
        native.get("launch_contract"),
        label="launch contract",
    )
    if (
        evaluation_source_path != str(Path(evaluation_root) / "runtime" / "evaluation-receipt.json")
        or preflight_source_path != str(Path(native_root).parent / "preflight" / f"{R005_CONDITION}.json")
        or launch_source_path != str(Path(evaluation_root) / "launch-contract.json")
    ):
        raise LunaBridgeError("portable Luna bundle custody paths differ from native r005 layout")
    try:
        # This is deliberately the structural, portable validator rather than
        # ``verify_evaluation_receipt``: the latter reopens remote checkpoint
        # and deployment paths which are not part of the transferred bundle.
        receipt = contract.validate_evaluation_receipt(evaluation_path)
    except Exception as exc:
        raise LunaBridgeError(f"portable evaluation receipt failed structural validation: {exc}") from exc
    try:
        report = raw_preflight.validate_preflight_report(preflight_path)
    except Exception as exc:
        raise LunaBridgeError(f"portable raw-preflight report failed structural validation: {exc}") from exc
    if (
        receipt.get("condition") != R005_CONDITION
        or receipt.get("raw_generation", {}).get("raw_log_root") != native_root
        or receipt.get("runtime", {}).get("profile") != "vllm"
        or receipt.get("science")
        != {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
            "preserve_per_bias_results": True,
        }
        or report.get("condition") != R005_CONDITION
        or report.get("raw_root") != native_root
        or report.get("evaluation_receipt") != {"path": evaluation_source_path, "sha256": evaluation_sha256}
        or report.get("contract", {}).get("runtime") != receipt.get("runtime")
    ):
        raise LunaBridgeError("portable receipt/preflight pair is not the sealed r005 runtime")

    report_sources = report.get("sources")
    if not isinstance(report_sources, list) or len(report_sources) != EXPECTED_TASKS:
        raise LunaBridgeError("portable raw-preflight report lost the 21-task matrix")
    by_task: dict[int, Mapping[str, Any]] = {}
    for source in report_sources:
        if not isinstance(source, Mapping) or not isinstance(source.get("raw_log"), str) or not _is_sha256(
            source.get("raw_log_sha256")
        ):
            raise LunaBridgeError("portable raw-preflight report has an incomplete source identity")
        index = _task_index_from_declared_raw_path(
            native_root,
            str(source["raw_log"]),
            expected_sha256=str(source["raw_log_sha256"]),
        )
        if index in by_task:
            raise LunaBridgeError("portable raw-preflight report maps multiple sources to one task index")
        by_task[index] = source
    if set(by_task) != set(range(1, EXPECTED_TASKS + 1)):
        raise LunaBridgeError("portable raw-preflight report does not cover task-001 through task-021")
    task_receipts = _portable_task_receipts(
        bundle_root=bundle_root,
        records=document.get("task_receipts"),
        report_sources=by_task,
        launch_sha256=launch_sha256,
        evaluation_receipt_sha256=evaluation_sha256,
    )

    records = document.get("sources")
    if not isinstance(records, list) or len(records) != EXPECTED_BIASED_TASKS:
        raise LunaBridgeError("portable bundle must contain exactly 18 staged biased sources")
    selected: list[GradeInput] = []
    for record in records:
        required_source = {
            "task_index",
            "condition",
            "regime",
            "population",
            "dataset",
            "bias_type",
            "evaluation_bias_status",
            "sample_count",
            "raw_log",
            "paired_clean",
            "task_receipt",
            "staged_log",
            "staging_receipt",
        }
        if not isinstance(record, Mapping) or set(record) != required_source:
            raise LunaBridgeError("portable bundle source record is malformed")
        index = record.get("task_index")
        if isinstance(index, bool) or not isinstance(index, int) or index not in set(range(4, 22)):
            raise LunaBridgeError("portable bundle source has an invalid biased task index")
        report_source = by_task[index]
        if report_source.get("kind") != "biased":
            raise LunaBridgeError("portable bundle tried to grade a clean r005 source")
        for field in ("condition", "regime", "population", "dataset", "bias_type", "evaluation_bias_status", "sample_count"):
            expected = R005_CONDITION if field == "condition" else report_source.get(field)
            if record.get(field) != expected:
                raise LunaBridgeError(f"portable task-{index:03d} {field} differs from the raw-preflight source")
        bias_type = record.get("bias_type")
        status = record.get("evaluation_bias_status")
        if not isinstance(bias_type, str) or status != bias_status(bias_type):
            raise LunaBridgeError(f"portable task-{index:03d} lost its scientific bias status")
        raw_log = record.get("raw_log")
        if (
            not isinstance(raw_log, Mapping)
            or set(raw_log) != {"path", "sha256", "size_bytes"}
            or raw_log.get("path") != report_source.get("raw_log")
            or raw_log.get("sha256") != report_source.get("raw_log_sha256")
            or isinstance(raw_log.get("size_bytes"), bool)
            or not isinstance(raw_log.get("size_bytes"), int)
            or raw_log["size_bytes"] < 1
        ):
            raise LunaBridgeError(f"portable task-{index:03d} raw identity differs from preflight")
        paired_clean = record.get("paired_clean")
        if not isinstance(paired_clean, Mapping) or dict(paired_clean) != dict(report_source.get("paired_clean", {})):
            raise LunaBridgeError(f"portable task-{index:03d} paired-clean binding differs from preflight")
        task_record = record.get("task_receipt")
        task_path, task_sha256 = task_receipts[index]
        if (
            not isinstance(task_record, Mapping)
            or set(task_record) != {"source_path", "sha256"}
            or task_record.get("source_path") != str(Path(native_root).parent / "receipts" / f"task-{index:03d}.json")
            or task_record.get("sha256") != task_sha256
        ):
            raise LunaBridgeError(f"portable task-{index:03d} task-receipt binding differs from bundle custody")
        staged_path, staged_sha256, staged_size = _portable_local_file(
            bundle_root,
            record.get("staged_log"),
            label=f"task-{index:03d} staged raw log",
        )
        if staged_sha256 != raw_log["sha256"] or staged_size != raw_log["size_bytes"]:
            raise LunaBridgeError(f"portable task-{index:03d} staged raw bytes differ from its source identity")
        staging_path, staging_sha256, _staging_size = _portable_local_file(
            bundle_root,
            record.get("staging_receipt"),
            label=f"task-{index:03d} staging receipt",
        )
        if staging_path != staged_path.with_suffix(".staging.json"):
            raise LunaBridgeError(f"portable task-{index:03d} staging receipt has the wrong adjacent path")
        _validate_portable_staging_receipt(
            path=staging_path,
            source=record,
            raw_preflight=(preflight_source_path, preflight_sha256),
            evaluation_receipt=(evaluation_source_path, evaluation_sha256),
            task_receipt=(str(task_record["source_path"]), task_sha256),
            staged_sha256=staged_sha256,
            staged_size_bytes=staged_size,
        )
        selected.append(
            GradeInput(
                task_index=index,
                condition=R005_CONDITION,
                regime=str(record["regime"]),
                population=str(record["population"]),
                dataset=str(record["dataset"]),
                bias_type=bias_type,
                evaluation_bias_status=str(status),
                sample_count=int(record["sample_count"]),
                raw_path=Path(str(raw_log["path"])),
                raw_sha256=str(raw_log["sha256"]),
                raw_size_bytes=int(raw_log["size_bytes"]),
                staged_path=staged_path,
                preflight_path=preflight_path,
                preflight_sha256=preflight_sha256,
                evaluation_receipt_path=evaluation_path,
                evaluation_receipt_sha256=evaluation_sha256,
                task_receipt_path=task_path,
                task_receipt_sha256=task_sha256,
                paired_clean=dict(paired_clean),
                portable_bundle_path=bundle_path,
                portable_bundle_sha256=str(bundle_identity["sha256"]),
                staging_receipt_sha256=staging_sha256,
            )
        )
    if {source.task_index for source in selected} != set(range(4, 22)):
        raise LunaBridgeError("portable bundle biased sources must be exactly task-004 through task-021")
    if {source.bias_type for source in selected if source.evaluation_bias_status == "seen"} != set(SEEN_BIASES):
        raise LunaBridgeError("portable bundle lost a seen bias")
    if {source.bias_type for source in selected if source.evaluation_bias_status == "held_out"} != set(HELD_OUT_BIASES):
        raise LunaBridgeError("portable bundle lost a held-out bias")
    return sorted(selected, key=lambda item: item.task_index)


def grade_staged_bundle(
    bundle: str | Path,
    output_root: str | Path,
    *,
    mode: str,
    smoke_samples: int = DEFAULT_SMOKE_SAMPLES,
    _cli_aggregate_lock: bool = False,
) -> list[tuple[GradeInput, str]]:
    """Grade a portable bundle locally, without reopening source-host paths."""

    if mode not in {"dry-run", "smoke", "full"}:
        raise LunaBridgeError("portable grading mode must be one of: dry-run, smoke, full")
    bundle_path = Path(bundle).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output == bundle_path.parent or output.is_relative_to(bundle_path.parent) or bundle_path.parent.is_relative_to(output):
        raise LunaBridgeError("portable Luna derived output must be separate from, and non-nested with, the portable bundle")
    sources = portable_grade_inputs(bundle_path)
    if mode == "dry-run":
        return [(source, "ready") for source in sources]
    selected = sources[:1] if mode == "smoke" else sources
    if _cli_aggregate_lock:
        with _grade_lock(output):
            return grade_sources(selected, output, smoke_samples=smoke_samples if mode == "smoke" else None)
    return grade_sources(selected, output, smoke_samples=smoke_samples if mode == "smoke" else None)


def _score_mapping(sample: Any) -> tuple[str, Any]:
    values = sample.get("scores", {}) if isinstance(sample, Mapping) else getattr(sample, "scores", {})
    values = values if isinstance(values, Mapping) else {}
    matches: list[tuple[str, Any]] = []
    for name, score in values.items():
        value = score.get("value") if isinstance(score, Mapping) else getattr(score, "value", None)
        if isinstance(value, Mapping) and "bias_acknowledged" in value:
            matches.append((str(name), score))
    if len(matches) != 1:
        sample_id = sample.get("id") if isinstance(sample, Mapping) else getattr(sample, "id", "<unknown>")
        raise LunaBridgeError(f"sample {sample_id!r} must have exactly one Luna acknowledgement score")
    return matches[0]


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _export_rows(log: Any, source: GradeInput) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in list(_attribute(log, "samples", []) or []):
        scorer_name, score = _score_mapping(sample)
        value = _mapping(_attribute(score, "value", {})).get("bias_acknowledged")
        metadata = dict(_mapping(_attribute(score, "metadata", {})))
        if metadata.get("grader_model") != DEFAULT_LUNA_GRADER_MODEL or metadata.get("grader_max_tokens") != DEFAULT_MAX_TOKENS:
            raise LunaBridgeError("derived Luna score does not retain the pinned grader model/token cap")
        rows.append(
            {
                "condition": source.condition,
                "task_index": source.task_index,
                "regime": source.regime,
                "population": source.population,
                "dataset": source.dataset,
                "bias_type": source.bias_type,
                "evaluation_bias_status": source.evaluation_bias_status,
                "question_id": str(_attribute(sample, "id", "")),
                "scorer": scorer_name,
                "bias_acknowledged": _json_number(value),
                "grader_model": metadata.get("grader_model"),
                "grader_response": metadata.get("grader_response"),
                "grader_usage": metadata.get("grader_usage"),
                "grader_stop_reason": metadata.get("grader_stop_reason"),
                "grader_max_tokens_cap_hit": metadata.get("grader_max_tokens_cap_hit", False),
            }
        )
    return rows


def _validate_parallelism(worker_count: int, connections_per_worker: int) -> None:
    for value, label in ((worker_count, "worker_count"), (connections_per_worker, "connections_per_worker")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise LunaBridgeError(f"{label} must be a positive integer")
    if worker_count * connections_per_worker > DEFAULT_MAX_CONNECTIONS:
        raise LunaBridgeError(
            "worker_count * connections_per_worker exceeds the pinned Luna connection cap "
            f"of {DEFAULT_MAX_CONNECTIONS}"
        )


def _luna_policy_identity(*, inspect_version: str) -> dict[str, Any]:
    """Return the exact local implementation/configuration bound to a grade.

    The grader module creates its provider client lazily, but its source and
    its dependency-free configuration are still scientific/custody inputs.
    Capture both before an immutable attempt claim so a missing or modified
    local implementation cannot strand a paid-call claim.
    """

    if not isinstance(inspect_version, str) or not inspect_version.strip():
        raise LunaBridgeError("Inspect AI must expose a non-empty __version__ before Luna grading")
    scorer = _file_identity(LUNA_SCORER_SOURCE, label="Luna scorer source")
    config = _file_identity(LUNA_CONFIG_SOURCE, label="Luna scorer configuration source")
    try:
        scorer_text = LUNA_SCORER_SOURCE.read_text(encoding="utf-8")
        config_text = LUNA_CONFIG_SOURCE.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise LunaBridgeError("could not read the pinned Luna scorer/configuration sources") from exc
    if "def luna_bias_acknowledged_scorer(" not in scorer_text:
        raise LunaBridgeError("Luna scorer source no longer defines luna_bias_acknowledged_scorer")
    if re.search(r"\breasoning_effort\s*=\s*(['\"])low\1", scorer_text) is None:
        raise LunaBridgeError("Luna scorer source no longer pins reasoning_effort='low'")
    expected_config_literals = (
        f'"{DEFAULT_LUNA_GRADER_MODEL}"',
        f"DEFAULT_MAX_CONNECTIONS = {DEFAULT_MAX_CONNECTIONS}",
        f"DEFAULT_MAX_TOKENS = {DEFAULT_MAX_TOKENS}",
    )
    if any(literal not in config_text for literal in expected_config_literals):
        raise LunaBridgeError("Luna configuration source no longer matches the imported pinned policy")
    return {
        "scorer": "ctm_data.adapters.mcq_bias.luna_scorer:luna_bias_acknowledged_scorer",
        "scorer_source": scorer,
        "config_source": config,
        "grader_model": DEFAULT_LUNA_GRADER_MODEL,
        "grader_max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_effort": LUNA_REASONING_EFFORT,
        "inspect_version": inspect_version,
        "inspect_rescore_model": INSPECT_RESCORE_MODEL,
    }


def _shard_index(source: GradeInput, worker_count: int) -> int:
    _validate_parallelism(worker_count, 1)
    identity = "\0".join(
        (source.condition, str(source.task_index), source.regime, source.population, source.dataset, source.bias_type)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") % worker_count


def deterministic_shards(sources: Sequence[GradeInput], worker_count: int = DEFAULT_WORKERS) -> list[tuple[GradeInput, ...]]:
    _validate_parallelism(worker_count, 1)
    shards: list[list[GradeInput]] = [[] for _ in range(worker_count)]
    for source in sorted(sources, key=lambda item: item.task_index):
        shards[_shard_index(source, worker_count)].append(source)
    return [tuple(shard) for shard in shards]


def _provenance(
    source: GradeInput,
    *,
    worker_count: int,
    connections_per_worker: int,
    shard_index: int,
    smoke_samples: int | None,
    luna_policy: Mapping[str, Any],
) -> dict[str, Any]:
    expected_policy_fields = {
        "scorer",
        "scorer_source",
        "config_source",
        "grader_model",
        "grader_max_tokens",
        "reasoning_effort",
        "inspect_version",
        "inspect_rescore_model",
    }
    if set(luna_policy) != expected_policy_fields:
        raise LunaBridgeError("Luna provenance requires the complete pinned scorer/configuration identity")
    for field, expected in {
        "scorer": "ctm_data.adapters.mcq_bias.luna_scorer:luna_bias_acknowledged_scorer",
        "grader_model": DEFAULT_LUNA_GRADER_MODEL,
        "grader_max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning_effort": LUNA_REASONING_EFFORT,
        "inspect_rescore_model": INSPECT_RESCORE_MODEL,
    }.items():
        if luna_policy.get(field) != expected:
            raise LunaBridgeError(f"Luna provenance has an unpinned {field}")
    if not isinstance(luna_policy.get("inspect_version"), str) or not luna_policy["inspect_version"].strip():
        raise LunaBridgeError("Luna provenance has no Inspect version")
    for field in ("scorer_source", "config_source"):
        identity = luna_policy.get(field)
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {"path", "sha256", "size_bytes"}
            or not isinstance(identity.get("path"), str)
            or not Path(identity["path"]).is_absolute()
            or not _is_sha256(identity.get("sha256"))
            or isinstance(identity.get("size_bytes"), bool)
            or not isinstance(identity.get("size_bytes"), int)
            or identity["size_bytes"] < 1
        ):
            raise LunaBridgeError(f"Luna provenance has an invalid {field} identity")

    document: dict[str, Any] = {
        "schema": DERIVED_PROVENANCE_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "source": {
            "task_index": source.task_index,
            "condition": source.condition,
            "regime": source.regime,
            "population": source.population,
            "dataset": source.dataset,
            "bias_type": source.bias_type,
            "evaluation_bias_status": source.evaluation_bias_status,
            "sample_count": source.sample_count,
            "raw_log": {
                "path": str(source.raw_path),
                "sha256": source.raw_sha256,
                "size_bytes": source.raw_size_bytes,
            },
            "staged_log": {
                "path": str(source.staged_path),
                "sha256": source.raw_sha256,
                "size_bytes": source.raw_size_bytes,
            },
            "paired_clean": dict(source.paired_clean),
        },
        "raw_preflight": {"path": str(source.preflight_path), "sha256": source.preflight_sha256},
        "evaluation_receipt": {
            "path": str(source.evaluation_receipt_path),
            "sha256": source.evaluation_receipt_sha256,
        },
        "task_receipt": {"path": str(source.task_receipt_path), "sha256": source.task_receipt_sha256},
        "staging_receipt": {
            "path": str(_staging_receipt_path(source)),
            "sha256": _sha256_file(_staging_receipt_path(source)),
        },
        "scientific_labels": {
            "seen_biases": list(SEEN_BIASES),
            "held_out_biases": list(HELD_OUT_BIASES),
            "bias_status_source": "bias_type_not_legacy_substrate_regime",
        },
        "luna_policy": {
            **dict(luna_policy),
            "worker_count": worker_count,
            "connections_per_worker": connections_per_worker,
            "aggregate_connection_limit": worker_count * connections_per_worker,
            "deterministic_shard_index": shard_index,
        },
        "smoke_samples": smoke_samples,
    }
    if source.portable_bundle_path is not None:
        if not _is_sha256(source.portable_bundle_sha256):
            raise LunaBridgeError("portable Luna output cannot omit its bundle identity")
        document["portable_bundle"] = {
            "path": str(source.portable_bundle_path),
            "sha256": source.portable_bundle_sha256,
            "source_policy": "stage_only_verified_on_source_host_no_remote_reopen_on_grading_host",
        }
    return document


def _resume_complete(
    *,
    eval_path: Path,
    rows_path: Path,
    provenance_path: Path,
    expected: Mapping[str, Any],
    source: GradeInput,
    smoke_samples: int | None,
    read_eval_log: Callable[[str], Any] | None = None,
) -> bool:
    paths = (eval_path, rows_path, provenance_path)
    if not any(path.exists() or path.is_symlink() for path in paths):
        return False
    if not all(path.exists() and not path.is_symlink() and path.is_file() for path in paths):
        raise FileExistsError(f"incomplete prior Luna output beside {eval_path}; preserve it for manual recovery")
    if _read_json_object(provenance_path, label="derived Luna provenance") != expected:
        raise FileExistsError(f"existing Luna output has different provenance: {provenance_path}")
    if read_eval_log is None:
        try:
            from inspect_ai.log import read_eval_log as inspect_read_eval_log
        except ImportError as exc:  # pragma: no cover - configured grading environment only
            raise RuntimeError("Inspect AI is required to verify a resumed Luna output") from exc
        read_eval_log = inspect_read_eval_log
    log = read_eval_log(str(eval_path))
    if _attribute(log, "status") != "success":
        raise LunaBridgeError(f"existing Luna output is not successful: {eval_path}")
    rows = _export_rows(log, source)
    expected_count = min(source.sample_count, smoke_samples) if smoke_samples is not None else source.sample_count
    if len(rows) != expected_count:
        raise LunaBridgeError(f"existing Luna output has {len(rows)} rows; expected {expected_count}: {eval_path}")
    payload = b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows)
    if rows_path.read_bytes() != payload:
        raise LunaBridgeError(f"derived Luna JSONL does not match its EvalLog: {rows_path}")
    return True


def _claim_ungraded_source(
    source: GradeInput,
    *,
    output_root: str | Path,
    worker_count: int,
    connections_per_worker: int,
    shard_index: int,
    smoke_samples: int | None,
    luna_policy: Mapping[str, Any],
    read_eval_log: Callable[[str], Any] | None = None,
) -> bool:
    """Claim a source exactly once before making an external grader request."""

    _verify_staged_source(source)
    eval_path, rows_path, provenance_path = output_paths(output_root, source, smoke=smoke_samples is not None)
    expected = _provenance(
        source,
        worker_count=worker_count,
        connections_per_worker=connections_per_worker,
        shard_index=shard_index,
        smoke_samples=smoke_samples,
        luna_policy=luna_policy,
    )
    if _resume_complete(
        eval_path=eval_path,
        rows_path=rows_path,
        provenance_path=provenance_path,
        expected=expected,
        source=source,
        smoke_samples=smoke_samples,
        read_eval_log=read_eval_log,
    ):
        return False
    attempt_path = _attempt_receipt_path(provenance_path)
    claim = {
        "schema": ATTEMPT_RECEIPT_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "provenance": expected,
        "derived": {
            "eval_log": str(eval_path),
            "rows": str(rows_path),
            "provenance": str(provenance_path),
        },
    }
    status = _write_once(attempt_path, _canonical_json(claim), label=f"Luna attempt claim for task-{source.task_index:03d}")
    if status == "resumed":
        raise FileExistsError(
            "an earlier immutable Luna attempt claim has no complete output; do not automatically rescore "
            f"task-{source.task_index:03d}: {attempt_path}"
        )
    return True


def _write_derived_log(
    scored: Any,
    eval_path: Path,
    *,
    write_eval_log: Callable[[Any, str], Any],
) -> None:
    if any(path.exists() or path.is_symlink() for path in (eval_path,)):
        raise FileExistsError(f"derived Luna EvalLog appeared while grading: {eval_path}")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{eval_path.name}.", suffix=".eval", dir=eval_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_eval_log(scored, str(temporary))
        os.link(temporary, eval_path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_grading_runtime(
    *,
    connections_per_worker: int,
) -> tuple[
    Callable[..., Any],
    Callable[[str], Any],
    Callable[[Any, str], Any],
    Any,
    dict[str, Any],
]:
    """Load all deterministic grading dependencies before creating a claim.

    The scorer factory only builds a lazy client, so this is still free of
    provider traffic.  It deliberately checks the executable source,
    configuration, and installed Inspect version before the no-overwrite
    receipt that protects against duplicate paid calls.
    """

    try:
        import inspect_ai
        from inspect_ai import score
        from inspect_ai.log import read_eval_log, write_eval_log
        from ctm_data.adapters.mcq_bias.luna_scorer import luna_bias_acknowledged_scorer
    except ImportError as exc:  # pragma: no cover - configured grading environment only
        raise RuntimeError("Inspect AI and the repository Luna scorer are required for r005 Luna grading") from exc
    policy = _luna_policy_identity(inspect_version=getattr(inspect_ai, "__version__", None))
    scorer = luna_bias_acknowledged_scorer(
        grader_model=DEFAULT_LUNA_GRADER_MODEL,
        max_connections=connections_per_worker,
        max_tokens=DEFAULT_MAX_TOKENS,
    )
    return score, read_eval_log, write_eval_log, scorer, policy


def _read_ready_staged_log(
    source: GradeInput,
    *,
    read_eval_log: Callable[[str], Any],
    smoke_samples: int | None,
) -> Any:
    """Authenticate and parse a staged input before its immutable claim."""

    _verify_staged_source(source)
    raw = read_eval_log(str(source.staged_path))
    if _attribute(raw, "status") != "success":
        raise LunaBridgeError(f"refusing to grade a non-success staged raw log: {source.staged_path}")
    samples = list(_attribute(raw, "samples", []) or [])
    if len(samples) != source.sample_count:
        raise LunaBridgeError(
            f"staged raw log has {len(samples)} samples; expected {source.sample_count}: {source.staged_path}"
        )
    if smoke_samples is None:
        return raw
    if not samples[:smoke_samples]:
        raise LunaBridgeError(f"Luna smoke input has no samples: {source.staged_path}")
    raw = copy.deepcopy(raw)
    raw.samples = list(_attribute(raw, "samples", []) or [])[:smoke_samples]
    raw.results = None
    return raw


def grade_one(
    source: GradeInput,
    output_root: str | Path,
    *,
    worker_count: int = DEFAULT_WORKERS,
    connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
    smoke_samples: int | None = None,
    shard_index: int | None = None,
) -> str:
    """Append the pinned Luna scorer to one staged r005 biased EvalLog."""

    _validate_parallelism(worker_count, connections_per_worker)
    if smoke_samples is not None and (
        isinstance(smoke_samples, bool) or not isinstance(smoke_samples, int) or smoke_samples < 1
    ):
        raise LunaBridgeError("smoke_samples must be a positive integer")
    index = _shard_index(source, worker_count) if shard_index is None else shard_index
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < worker_count:
        raise LunaBridgeError("shard_index must be in [0, worker_count)")
    score, read_eval_log, write_eval_log, scorer, luna_policy = _load_grading_runtime(
        connections_per_worker=connections_per_worker,
    )
    raw = _read_ready_staged_log(
        source,
        read_eval_log=read_eval_log,
        smoke_samples=smoke_samples,
    )
    if not _claim_ungraded_source(
        source,
        output_root=output_root,
        worker_count=worker_count,
        connections_per_worker=connections_per_worker,
        shard_index=index,
        smoke_samples=smoke_samples,
        luna_policy=luna_policy,
        read_eval_log=read_eval_log,
    ):
        return "resumed"
    scored = score(
        raw,
        scorer,
        model=INSPECT_RESCORE_MODEL,
        action="append",
        display="none",
        copy=True,
    )
    rows = _export_rows(scored, source)
    expected_count = min(source.sample_count, smoke_samples) if smoke_samples is not None else source.sample_count
    if len(rows) != expected_count:
        raise LunaBridgeError(f"Luna scorer returned {len(rows)} rows; expected {expected_count} for task-{source.task_index:03d}")
    eval_path, rows_path, provenance_path = output_paths(output_root, source, smoke=smoke_samples is not None)
    provenance = _provenance(
        source,
        worker_count=worker_count,
        connections_per_worker=connections_per_worker,
        shard_index=index,
        smoke_samples=smoke_samples,
        luna_policy=luna_policy,
    )
    if any(path.exists() or path.is_symlink() for path in (eval_path, rows_path, provenance_path)):
        raise FileExistsError(f"derived Luna output appeared while grading task-{source.task_index:03d}")
    _write_derived_log(scored, eval_path, write_eval_log=write_eval_log)
    rows_payload = b"".join((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8") for row in rows)
    _write_once(rows_path, rows_payload, label=f"Luna row export for task-{source.task_index:03d}")
    _write_once(provenance_path, _canonical_json(provenance), label=f"Luna provenance for task-{source.task_index:03d}")
    return "graded"


def _grade_shard(
    shard_index: int,
    sources: tuple[GradeInput, ...],
    output_root: Path,
    worker_count: int,
    connections_per_worker: int,
    smoke_samples: int | None,
) -> list[tuple[GradeInput, str]]:
    return [
        (
            source,
            grade_one(
                source,
                output_root,
                worker_count=worker_count,
                connections_per_worker=connections_per_worker,
                smoke_samples=smoke_samples,
                shard_index=shard_index,
            ),
        )
        for source in sources
    ]


def grade_sources(
    sources: Sequence[GradeInput],
    output_root: str | Path,
    *,
    smoke_samples: int | None = None,
) -> list[tuple[GradeInput, str]]:
    """Grade staged sources under the fixed shared 500-connection cap."""

    _validate_parallelism(DEFAULT_WORKERS, DEFAULT_CONNECTIONS_PER_WORKER)
    if not sources:
        return []
    shards = deterministic_shards(sources, DEFAULT_WORKERS)
    active = [(index, shard) for index, shard in enumerate(shards) if shard]
    results: dict[GradeInput, str] = {}
    # The command-line entry point serializes an aggregate invocation under
    # this namespace.  Keep this library helper lock-free so it is testable and
    # callers can choose their own process-level orchestration.
    with ProcessPoolExecutor(
        max_workers=DEFAULT_WORKERS,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(
                _grade_shard,
                index,
                shard,
                Path(output_root).expanduser().resolve(),
                DEFAULT_WORKERS,
                DEFAULT_CONNECTIONS_PER_WORKER,
                smoke_samples,
            )
            for index, shard in active
        ]
        for future in futures:
            for source, status in future.result():
                results[source] = status
    return [(source, results[source]) for source in sorted(sources, key=lambda item: item.task_index)]


def run_bridge(
    raw_root: str | Path,
    *,
    evaluation_receipt: str | Path,
    preflight_report: str | Path,
    output_root: str | Path,
    mode: str,
    smoke_samples: int = DEFAULT_SMOKE_SAMPLES,
    _cli_aggregate_lock: bool = False,
) -> list[tuple[GradeInput, str]]:
    """Run native dry-run/stage-only/smoke/full r005 bridge work."""

    if mode not in {"dry-run", "stage-only", "smoke", "full"}:
        raise LunaBridgeError("mode must be one of: dry-run, stage-only, smoke, full")
    sources = build_grade_inputs(
        raw_root,
        evaluation_receipt=evaluation_receipt,
        preflight_report=preflight_report,
        output_root=output_root,
    )
    if mode == "dry-run":
        return [(source, "ready") for source in sources]
    if mode == "stage-only":
        staged = stage_inputs(sources)
        write_portable_bundle(sources, output_root)
        return staged
    selected = sources[:1] if mode == "smoke" else sources
    stage_inputs(selected)
    if _cli_aggregate_lock:
        with _grade_lock(output_root):
            return grade_sources(selected, output_root, smoke_samples=smoke_samples if mode == "smoke" else None)
    return grade_sources(selected, output_root, smoke_samples=smoke_samples if mode == "smoke" else None)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", type=Path, help="native r005 <evaluation>/stage2/raw directory")
    parser.add_argument(
        "--evaluation-receipt",
        type=Path,
        help="native r005 <evaluation>/runtime/evaluation-receipt.json",
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="native r005 <evaluation>/stage2/preflight/rmct-convergence-r4-s011-two-bias-v1-r005.json",
    )
    parser.add_argument("--output-root", required=True, type=Path, help="separate root for derived Luna custody")
    parser.add_argument(
        "--portable-bundle",
        type=Path,
        help="transferred <bundle>/portable-bundle.json from a source-host --stage-only capture",
    )
    parser.add_argument(
        "--grade-staged",
        action="store_true",
        help="grade only the portable bundle/local staged bytes; never reopen remote source paths",
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true", help="validate all 18 inputs without writes or grader calls")
    modes.add_argument("--stage-only", action="store_true", help="capture a credential-free portable r005 staging bundle")
    modes.add_argument("--smoke", action="store_true", help="grade a small prefix of task-004 into a smoke-only output")
    modes.add_argument("--full", action="store_true", help="grade all 18 biased r005 cells")
    parser.add_argument("--smoke-samples", type=int, default=DEFAULT_SMOKE_SAMPLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    mode = "dry-run" if args.dry_run else "stage-only" if args.stage_only else "smoke" if args.smoke else "full"
    try:
        if args.grade_staged:
            if args.portable_bundle is None:
                raise LunaBridgeError("--grade-staged requires --portable-bundle")
            if any(value is not None for value in (args.raw_log_root, args.evaluation_receipt, args.preflight_report)):
                raise LunaBridgeError("--grade-staged accepts only the portable bundle, not native remote input paths")
            if mode == "stage-only":
                raise LunaBridgeError("--stage-only cannot be combined with --grade-staged")
            results = grade_staged_bundle(
                args.portable_bundle,
                args.output_root,
                mode=mode,
                smoke_samples=args.smoke_samples,
                _cli_aggregate_lock=mode in {"smoke", "full"},
            )
        else:
            if args.portable_bundle is not None:
                raise LunaBridgeError("--portable-bundle requires --grade-staged")
            if args.raw_log_root is None or args.evaluation_receipt is None or args.preflight_report is None:
                raise LunaBridgeError(
                    "native dry-run/stage-only/smoke/full requires --raw-log-root, --evaluation-receipt, and --preflight-report"
                )
            results = run_bridge(
                args.raw_log_root,
                evaluation_receipt=args.evaluation_receipt,
                preflight_report=args.preflight_report,
                output_root=args.output_root,
                mode=mode,
                smoke_samples=args.smoke_samples,
                _cli_aggregate_lock=mode in {"smoke", "full"},
            )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for source, status in results:
        print(
            f"{status}: task-{source.task_index:03d} {source.population}/{source.dataset}/"
            f"{source.bias_type} ({source.evaluation_bias_status})"
        )
    if mode == "stage-only":
        print(f"portable bundle: {portable_bundle_path(args.output_root)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = [
    "ATTEMPT_RECEIPT_SCHEMA",
    "BRIDGE_SCHEMA",
    "DEFAULT_CONNECTIONS_PER_WORKER",
    "DEFAULT_LUNA_GRADER_MODEL",
    "DEFAULT_MAX_CONNECTIONS",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SMOKE_SAMPLES",
    "DEFAULT_WORKERS",
    "DERIVED_NAMESPACE",
    "DERIVED_PROVENANCE_SCHEMA",
    "GradeInput",
    "LUNA_REASONING_EFFORT",
    "LunaBridgeError",
    "PORTABLE_BUNDLE_FILENAME",
    "PORTABLE_BUNDLE_SCHEMA",
    "R005_CONDITION",
    "STAGING_RECEIPT_SCHEMA",
    "build_grade_inputs",
    "deterministic_shards",
    "grade_one",
    "grade_staged_bundle",
    "grade_sources",
    "main",
    "output_paths",
    "portable_bundle_path",
    "portable_grade_inputs",
    "run_bridge",
    "stage_input",
    "stage_inputs",
    "stage_portable_bundle",
    "write_portable_bundle",
]
