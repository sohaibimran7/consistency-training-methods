"""Grade completed Stage 2 native-HF/PEFT cells before a matrix is complete.

The regular :mod:`experiments.stage2_ood_hle.grade_luna` entrypoint accepts a
single completed 21-cell raw-preflight report.  Native-HF/PEFT resume workers
also write a narrower, hash-bound handoff next to each successful biased log
after all three clean cells and the paired switch scores have been validated.

This module is the deliberately separate consumer for those receipts.  It
does not discover arbitrary ``.eval`` files: every source must have an
adjacent ``.resume-handoff.json`` written by ``hf_peft_resume``, whose staged,
raw, clean, frozen-manifest, checkpoint, and protocol bindings still validate
at grading time.  It then reuses the shared pinned Luna scoring machinery.

The derived EvalLogs are immutable copies.  Raw generation logs, staged raw
logs, handoff receipts, and the resume contract are read-only inputs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from collections.abc import Mapping, Sequence
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from experiments.stage2_ood_hle import grade_luna, raw_preflight
from experiments.stage2_ood_hle.hf_peft_resume import HANDOFF_SCHEMA, EXPECTED_MAX_CONNECTIONS
from experiments.stage2_ood_hle.hf_peft_runner import validate_raw_hf_peft_checkpoint
from experiments.stage2_ood_hle.materialize import PROMPT_STYLE, validate_manifest
from experiments.stage2_ood_hle.tasks import ood_task_specs


INCREMENTAL_PROVENANCE_BINDING = "incremental_handoff"
ATTEMPT_RECEIPT_SCHEMA = "stage2-ood-hle-incremental-luna-attempt-v1"
EXPECTED_LUNA_GRADER_MODEL = "openrouter/openai/gpt-5.6-luna-20260709"
EXPECTED_LUNA_WORKERS = 5
EXPECTED_LUNA_CONNECTIONS_PER_WORKER = 100
EXPECTED_LUNA_MAX_CONNECTIONS = 500
EXPECTED_LUNA_MAX_TOKENS = 256
_SHA256_HEX = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA256_HEX


def _require_sha256(value: Any, *, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"incremental Stage 2 Luna handoff has invalid {label}")
    return str(value)


def _safe_component(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"incremental Stage 2 Luna handoff has invalid {label}")
    return value


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f"incremental Stage 2 Luna handoff has no absolute {label}")
    return Path(value).resolve()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"incremental Stage 2 Luna handoff has invalid {label}")
    return value


def _under_root(path: Path, root: Path, *, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"incremental Stage 2 Luna {label} escapes its declared root: {path}") from exc
    return path


def _read_receipt(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"incremental Stage 2 Luna handoff receipt is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"incremental Stage 2 Luna handoff receipt is not valid JSON: {path}") from exc
    return _mapping(value, label="receipt")


def _receipt_paths(
    staged_root: Path,
    *,
    condition: str,
    receipt_root: str | Path | None,
    receipts: Sequence[str | Path],
) -> list[Path]:
    condition_root = (staged_root / condition).resolve()
    if receipt_root is not None and receipts:
        raise ValueError("choose --receipt-root or --receipt, not both")
    if receipt_root is not None:
        root = Path(receipt_root).resolve()
        if root != condition_root:
            raise ValueError("incremental receipt root must be the condition directory under --staged-raw-root")
        if not root.is_dir():
            raise FileNotFoundError(f"incremental Stage 2 Luna receipt root does not exist: {root}")
        selected = sorted(root.rglob("*.resume-handoff.json"))
    else:
        selected = [Path(path).resolve() for path in receipts]
    if not selected:
        raise FileNotFoundError("no incremental Stage 2 Luna handoff receipts were selected")
    result: list[Path] = []
    seen: set[Path] = set()
    for receipt in selected:
        if receipt in seen:
            raise ValueError(f"duplicate incremental Stage 2 Luna handoff receipt: {receipt}")
        seen.add(receipt)
        _under_root(receipt, condition_root, label="receipt")
        if receipt.name.endswith(".resume-handoff.json") is False:
            raise ValueError(f"incremental Stage 2 Luna receipt has the wrong filename: {receipt}")
        result.append(receipt)
    return result


def _identity(value: Any) -> tuple[str, str, str, str, str]:
    record = _mapping(value, label="identity")
    kind = _safe_component(record.get("kind"), label="identity.kind")
    regime = _safe_component(record.get("regime"), label="identity.regime")
    population = _safe_component(record.get("population"), label="identity.population")
    dataset = _safe_component(record.get("dataset"), label="identity.dataset")
    bias_type = _safe_component(record.get("bias_type"), label="identity.bias_type")
    if kind != "biased":
        raise ValueError("incremental Stage 2 Luna handoff may grade biased cells only")
    return kind, regime, population, dataset, bias_type


def _path_and_sha(value: Any, *, label: str) -> tuple[Path, str]:
    record = _mapping(value, label=label)
    return (
        _absolute_path(record.get("path"), label=f"{label}.path"),
        _require_sha256(record.get("sha256"), label=f"{label}.sha256"),
    )


def _check_regular_hash(path: Path, expected_sha256: str, *, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"incremental Stage 2 Luna {label} is not a regular file: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"incremental Stage 2 Luna {label} SHA-256 does not match its handoff receipt: {path}")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Write a canonical JSON receipt once, accepting only exact resumes."""

    encoded = (json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
            return "resumed"
        raise FileExistsError(f"incremental Stage 2 Luna receipt already exists and differs: {path}")
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
            if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
                raise FileExistsError(f"incremental Stage 2 Luna receipt appeared and differs: {path}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "written"


def _attempt_receipt_path(provenance_path: Path) -> Path:
    """Keep a pre-score claim beside the immutable output provenance."""

    return provenance_path.with_name(f"{provenance_path.stem}.attempt.json")


def _claim_ungraded_source(
    source: grade_luna.GradeInput,
    *,
    output_root: Path,
    worker_count: int,
    connections_per_worker: int,
    grader_max_tokens: int,
    shard_index: int,
) -> bool:
    """Return whether ``source`` may be scored exactly once.

    The normal derived EvalLog/rows/provenance trio makes successful work
    resumable.  A process crash after an external scorer call but before that
    trio is sealed would otherwise make a retry indistinguishable from a new
    grading attempt.  This immutable pre-score receipt closes that gap:
    automatic consumers may never retry a claimed-but-incomplete cell.
    """

    source_sha256 = _sha256(source.path)
    if source_sha256 != _require_sha256(source.expected_sha256, label="staged_log.sha256"):
        raise ValueError(f"incremental Stage 2 Luna staged raw log changed before attempt claim: {source.path}")
    eval_path, rows_path, provenance_path = grade_luna.output_paths(output_root, source)
    provenance = grade_luna._provenance(
        source,
        source_sha256=source_sha256,
        smoke_samples=None,
        worker_count=worker_count,
        connections_per_worker=connections_per_worker,
        grader_max_tokens=grader_max_tokens,
        shard_index=shard_index,
    )
    if grade_luna._resume_complete(eval_path, rows_path, provenance_path, provenance):
        return False
    receipt_path = _attempt_receipt_path(provenance_path)
    status = _write_immutable_json(
        receipt_path,
        {
            "schema": ATTEMPT_RECEIPT_SCHEMA,
            "source": {
                "condition": source.condition,
                "regime": source.regime,
                "population": source.population,
                "dataset": source.dataset,
                "bias_type": source.bias_type,
                "staged_log": str(source.path),
                "staged_log_sha256": source_sha256,
            },
            "incremental_handoff": {
                "schema": source.binding_schema,
                "receipt_path": source.binding_path,
                "receipt_sha256": source.binding_sha256,
            },
            "policy": {
                "grader_model": EXPECTED_LUNA_GRADER_MODEL,
                "worker_count": worker_count,
                "connections_per_worker": connections_per_worker,
                "aggregate_connection_limit": worker_count * connections_per_worker,
                "grader_max_tokens": grader_max_tokens,
                "deterministic_shard_index": shard_index,
            },
            "derived": {
                "eval_log": str(eval_path),
                "rows": str(rows_path),
                "provenance": str(provenance_path),
            },
        },
    )
    if status == "resumed":
        raise FileExistsError(
            "incremental Stage 2 Luna found an earlier immutable attempt claim without a complete derived output; "
            f"manual recovery is required before this cell can be considered again: {receipt_path}"
        )
    return True


def _expected_cells(manifest: Path) -> tuple[dict[tuple[str, str, str, str, str | None], Any], dict[int, tuple[str, str, str, str, str | None]]]:
    validate_manifest(manifest)
    specs = ood_task_specs(manifest)
    expected = raw_preflight._expected_cell_specs(specs)
    by_index = {
        index: raw_preflight._task_identity(spec)
        for index, spec in enumerate(specs, start=1)
    }
    if len(by_index) != raw_preflight.EXPECTED_TASKS:
        raise ValueError("frozen Stage 2 OOD manifest does not provide the exact 21-cell matrix")
    return expected, by_index


def _validate_protocol(value: Any) -> None:
    protocol = _mapping(value, label="protocol")
    expected = {
        "runtime_profile": "hf-peft",
        "prompt_style": PROMPT_STYLE,
        "include_bias_acknowledged": False,
        "max_tokens": 20480,
        "max_connections": EXPECTED_MAX_CONNECTIONS,
        "validated_with_full_paired_switch_scores": True,
    }
    if dict(protocol) != expected:
        raise ValueError("incremental Stage 2 Luna handoff protocol differs from the validated native-HF/PEFT policy")


def _validate_exact_luna_policy() -> None:
    """Keep this incremental path pinned even if generic CLI defaults drift."""

    if grade_luna.DEFAULT_LUNA_GRADER_MODEL != EXPECTED_LUNA_GRADER_MODEL:
        raise ValueError("incremental Stage 2 Luna grader model pin differs from the approved policy")
    if (
        grade_luna.DEFAULT_MAX_CONNECTIONS,
        grade_luna.DEFAULT_WORKERS,
        grade_luna.DEFAULT_CONNECTIONS_PER_WORKER,
        grade_luna.DEFAULT_MAX_TOKENS,
    ) != (
        EXPECTED_LUNA_MAX_CONNECTIONS,
        EXPECTED_LUNA_WORKERS,
        EXPECTED_LUNA_CONNECTIONS_PER_WORKER,
        EXPECTED_LUNA_MAX_TOKENS,
    ):
        raise ValueError("incremental Stage 2 Luna worker or token cap differs from the approved policy")
    grade_luna._validate_parallelism(EXPECTED_LUNA_WORKERS, EXPECTED_LUNA_CONNECTIONS_PER_WORKER)


@contextmanager
def _global_grade_lock(staged_root: Path):
    """Serialize this owner with the canonical postprocesser's 500-slot pool."""

    lock_path = staged_root.parent / ".luna-grade.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def handoff_bound_logs(
    staged_raw_root: str | Path,
    *,
    condition: str,
    manifest: str | Path,
    checkpoint: str | Path,
    receipt_root: str | Path | None = None,
    receipts: Sequence[str | Path] = (),
) -> list[grade_luna.GradeInput]:
    """Return only receipt-, manifest-, checkpoint-, and byte-bound cells.

    This does not invoke Inspect scoring or make any network request.  It
    deliberately requires live raw and paired-clean files in addition to the
    staged copy, because the native-HF resume host retains them and that makes
    the incremental handoff independently re-checkable before paid grading.
    """

    _validate_exact_luna_policy()
    staged_root = Path(staged_raw_root).resolve()
    if not staged_root.is_dir():
        raise FileNotFoundError(f"incremental Stage 2 Luna staged raw root does not exist: {staged_root}")
    condition = _safe_component(condition, label="condition")
    manifest_path = Path(manifest).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"incremental Stage 2 Luna frozen manifest does not exist: {manifest_path}")
    manifest_sha256 = _sha256(manifest_path)
    expected_cells, index_to_identity = _expected_cells(manifest_path)
    actual_checkpoint = validate_raw_hf_peft_checkpoint(condition, checkpoint)
    selected: list[grade_luna.GradeInput] = []
    seen_cells: set[tuple[str, str, str, str, str]] = set()
    seen_indices: set[int] = set()
    seen_staged: set[Path] = set()

    for receipt_path in _receipt_paths(
        staged_root,
        condition=condition,
        receipt_root=receipt_root,
        receipts=receipts,
    ):
        receipt = _read_receipt(receipt_path)
        if receipt.get("schema") != HANDOFF_SCHEMA:
            raise ValueError(f"incremental Stage 2 Luna handoff has an unexpected schema: {receipt_path}")
        if receipt.get("condition") != condition:
            raise ValueError(f"incremental Stage 2 Luna handoff condition differs from this invocation: {receipt_path}")
        task_index = receipt.get("task_index")
        if isinstance(task_index, bool) or not isinstance(task_index, int) or task_index not in index_to_identity:
            raise ValueError(f"incremental Stage 2 Luna handoff has an invalid task index: {receipt_path}")
        identity = _identity(receipt.get("identity"))
        expected_identity = index_to_identity[task_index]
        if identity != expected_identity or identity not in expected_cells:
            raise ValueError(f"incremental Stage 2 Luna handoff identity disagrees with the frozen manifest: {receipt_path}")
        if task_index in seen_indices or identity in seen_cells:
            raise ValueError(f"duplicate incremental Stage 2 Luna cell: task {task_index}")

        raw_root = _absolute_path(receipt.get("raw_log_dir"), label="raw_log_dir")
        raw_path, raw_sha256 = _path_and_sha(receipt.get("raw_log"), label="raw_log")
        staged_path, staged_sha256 = _path_and_sha(receipt.get("staged_log"), label="staged_log")
        clean_path, clean_sha256 = _path_and_sha(receipt.get("paired_clean"), label="paired_clean")
        _under_root(raw_path, raw_root, label="raw log")
        _under_root(clean_path, raw_root, label="paired clean log")
        if raw_sha256 != staged_sha256:
            raise ValueError(f"incremental Stage 2 Luna staged/raw hashes differ in handoff: {receipt_path}")
        source = {
            "population": identity[2],
            "bias_type": identity[4],
            "dataset": identity[3],
            "raw_log": str(raw_path),
        }
        expected_staged = grade_luna._staged_path(staged_root, condition=condition, source=source)
        if staged_path != expected_staged:
            raise ValueError(f"incremental Stage 2 Luna handoff staged path is not canonical: {receipt_path}")
        if receipt_path != staged_path.with_suffix(".resume-handoff.json"):
            raise ValueError(f"incremental Stage 2 Luna handoff is not adjacent to its staged log: {receipt_path}")
        _check_regular_hash(raw_path, raw_sha256, label="raw log")
        _check_regular_hash(staged_path, staged_sha256, label="staged raw log")
        _check_regular_hash(clean_path, clean_sha256, label="paired clean log")

        manifest_record = _mapping(receipt.get("manifest"), label="manifest")
        _absolute_path(manifest_record.get("path"), label="manifest.path")
        if _require_sha256(manifest_record.get("sha256"), label="manifest.sha256") != manifest_sha256:
            raise ValueError(f"incremental Stage 2 Luna manifest hash differs from its handoff: {receipt_path}")
        checkpoint_record = _mapping(receipt.get("checkpoint"), label="checkpoint")
        if dict(checkpoint_record) != actual_checkpoint:
            raise ValueError(f"incremental Stage 2 Luna checkpoint binding differs from its handoff: {receipt_path}")
        _validate_protocol(receipt.get("protocol"))

        seen_indices.add(task_index)
        seen_cells.add(identity)
        if staged_path in seen_staged:
            raise ValueError(f"multiple incremental Stage 2 Luna handoffs map to one staged log: {staged_path}")
        seen_staged.add(staged_path)
        receipt_sha256 = _sha256(receipt_path)
        selected.append(
            grade_luna.GradeInput(
                path=staged_path,
                condition=condition,
                regime=identity[1],
                population=identity[2],
                dataset=identity[3],
                bias_type=identity[4],
                created="",
                expected_sha256=staged_sha256,
                # A common GradeInput field is retained for compatibility;
                # the actual provenance below explicitly calls this a receipt.
                preflight_report_sha256=receipt_sha256,
                manifest_sha256=manifest_sha256,
                paired_clean={"raw_log": str(clean_path), "raw_log_sha256": clean_sha256},
                binding_kind=INCREMENTAL_PROVENANCE_BINDING,
                binding_sha256=receipt_sha256,
                binding_path=str(receipt_path),
                binding_schema=HANDOFF_SCHEMA,
            )
        )
    return sorted(
        selected,
        key=lambda item: (item.condition, item.regime, item.population, item.dataset, item.bias_type),
    )


def grade_incremental(
    staged_raw_root: str | Path,
    output_root: str | Path,
    *,
    condition: str,
    manifest: str | Path,
    checkpoint: str | Path,
    receipt_root: str | Path | None = None,
    receipts: Sequence[str | Path] = (),
    dry_run: bool = False,
) -> list[tuple[grade_luna.GradeInput, str]]:
    """Grade selected handoff cells under exactly one 500-connection pool."""

    staged = Path(staged_raw_root).resolve()
    output = Path(output_root).resolve()
    if output == staged or output.is_relative_to(staged) or staged.is_relative_to(output):
        raise ValueError("incremental staged raw and derived Luna roots must be separate, non-nested directories")
    if dry_run:
        selected = handoff_bound_logs(
            staged,
            condition=condition,
            manifest=manifest,
            checkpoint=checkpoint,
            receipt_root=receipt_root,
            receipts=receipts,
        )
        return [(source, "ready") for source in selected]

    # Do not expose tuning flags here: this incremental owner has one approved
    # dated model pin, a single 5 x 100 = 500 aggregate connection ceiling,
    # and a 256-token grader cap.
    with _global_grade_lock(staged):
        # Receipts may have appeared or a raw file may have changed while this
        # invocation was waiting for the shared 500-connection slot.  Bind
        # the exact inputs again immediately before spawning any grader.
        selected = handoff_bound_logs(
            staged,
            condition=condition,
            manifest=manifest,
            checkpoint=checkpoint,
            receipt_root=receipt_root,
            receipts=receipts,
        )
        shards = grade_luna.deterministic_shards(selected, EXPECTED_LUNA_WORKERS)
        by_source: dict[grade_luna.GradeInput, str] = {}
        active: list[tuple[int, tuple[grade_luna.GradeInput, ...]]] = []
        for index, shard in enumerate(shards):
            claimed: list[grade_luna.GradeInput] = []
            for source in shard:
                if _claim_ungraded_source(
                    source,
                    output_root=output,
                    worker_count=EXPECTED_LUNA_WORKERS,
                    connections_per_worker=EXPECTED_LUNA_CONNECTIONS_PER_WORKER,
                    grader_max_tokens=EXPECTED_LUNA_MAX_TOKENS,
                    shard_index=index,
                ):
                    claimed.append(source)
                else:
                    by_source[source] = "resumed"
            if claimed:
                active.append((index, tuple(claimed)))
        if active:
            with grade_luna.ProcessPoolExecutor(
                max_workers=EXPECTED_LUNA_WORKERS,
                mp_context=get_context("spawn"),
            ) as executor:
                futures = [
                    executor.submit(
                        grade_luna._grade_shard,
                        index,
                        shard,
                        output,
                        EXPECTED_LUNA_WORKERS,
                        EXPECTED_LUNA_CONNECTIONS_PER_WORKER,
                        EXPECTED_LUNA_MAX_TOKENS,
                        None,
                    )
                    for index, shard in active
                ]
                for future in futures:
                    for source, status in future.result():
                        by_source[source] = status
        return [(source, by_source[source]) for source in selected]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--receipt-root", type=Path)
    source.add_argument("--receipt", action="append", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate receipts and byte bindings only; do not invoke the grader",
    )
    args = parser.parse_args(argv)
    try:
        results = grade_incremental(
            args.staged_raw_root,
            args.output_root,
            condition=args.condition,
            manifest=args.manifest,
            checkpoint=args.checkpoint,
            receipt_root=args.receipt_root,
            receipts=tuple(args.receipt or ()),
            dry_run=args.dry_run,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for source_input, status in results:
        print(
            f"{status}: {source_input.condition}/{source_input.regime}/{source_input.population}/"
            f"{source_input.dataset}/{source_input.bias_type}: {source_input.path}"
        )


__all__ = [
    "EXPECTED_LUNA_CONNECTIONS_PER_WORKER",
    "EXPECTED_LUNA_GRADER_MODEL",
    "EXPECTED_LUNA_MAX_CONNECTIONS",
    "EXPECTED_LUNA_MAX_TOKENS",
    "EXPECTED_LUNA_WORKERS",
    "ATTEMPT_RECEIPT_SCHEMA",
    "INCREMENTAL_PROVENANCE_BINDING",
    "grade_incremental",
    "handoff_bound_logs",
]


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
