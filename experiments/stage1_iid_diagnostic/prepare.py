"""Freeze exact-training and held-out Stage 1 IID diagnostic populations.

This module is deliberately offline.  It only accepts the byte-exact 3,000-row
Stage 1 paired-prompt source, validates its scientific identity, and publishes
two small no-overwrite JSONL files plus a fail-closed manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1
MANIFEST_KIND = "stage1_iid_diagnostic_manifest"
SOURCE_SHA256 = "dfc10d55e51de48566488d107cdc90a0965ac5a73a80f3af68e605e40eff24ad"
SOURCE_ROWS = 3000
DATASETS = ("logiqa", "hellaswag")
SOURCE_COUNTS = {"logiqa": 1500, "hellaswag": 1500}
BIAS_TYPE = "wrong_argument"
PROMPT_STYLE = "encourage_cot"

TRAIN_EVAL_ROWS = 200
TRAIN_EVAL_COUNTS = {"logiqa": 100, "hellaswag": 100}
RMCT_ROWS = 64
RMCT_COUNTS = {"logiqa": 32, "hellaswag": 32}
TRAINING_PREFIX_ROWS = 2048
HELDOUT_ROWS = 200
HELDOUT_COUNTS = {"logiqa": 100, "hellaswag": 100}
HELDOUT_SOURCE_START = TRAINING_PREFIX_ROWS  # zero-based; source row 2,049
HELDOUT_SOURCE_STOP = HELDOUT_SOURCE_START + HELDOUT_ROWS

TRAIN_EVAL_FILENAME = "train-eval-n200.jsonl"
HELDOUT_FILENAME = "heldout-in-domain-n200.jsonl"
DEFAULT_MANIFEST_FILENAME = "manifest.json"

_REQUIRED_FIELDS = frozenset(
    {
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "unbiased_messages",
        "biased_messages",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    }
)


@dataclass(frozen=True, slots=True)
class _SourceRow:
    value: dict[str, Any]
    raw_line: bytes


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_messages(value: object, *, location: str, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty message list")
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"{location}: {field}[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"{location}: {field}[{index}].role must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{location}: {field}[{index}].content must be a non-empty string")


def _validate_row(row: object, *, path: Path, line_number: int) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{location}: row must be a JSON object")
    missing = sorted(_REQUIRED_FIELDS - row.keys())
    if missing:
        raise ValueError(f"{location}: missing frozen field(s): {', '.join(missing)}")
    for field in (
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    ):
        if not isinstance(row[field], str):
            raise ValueError(f"{location}: {field} must be a string")
    if not row["question_id"].strip():
        raise ValueError(f"{location}: question_id must not be empty")
    if row["source_dataset"] not in DATASETS:
        raise ValueError(f"{location}: source_dataset must be one of {list(DATASETS)}")
    if row["bias_type"] != BIAS_TYPE:
        raise ValueError(f"{location}: bias_type must be {BIAS_TYPE!r}")
    if row["prompt_style"] != PROMPT_STYLE:
        raise ValueError(f"{location}: prompt_style must be {PROMPT_STYLE!r}")
    if not row["biased_option"].strip() or row["biased_option"] == row["ground_truth"]:
        raise ValueError(f"{location}: biased_option must be a non-empty distractor")
    _validate_messages(row["unbiased_messages"], location=location, field="unbiased_messages")
    _validate_messages(row["biased_messages"], location=location, field="biased_messages")
    return row


def _read_rows(path: Path, payload: bytes) -> list[_SourceRow]:
    rows: list[_SourceRow] = []
    try:
        lines = payload.splitlines()
    except (TypeError, UnicodeError) as exc:  # pragma: no cover - bytes.splitlines is total
        raise ValueError(f"cannot split source {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(_SourceRow(_validate_row(row, path=path, line_number=line_number), raw_line))
    return rows


def _validate_population(
    rows: Iterable[_SourceRow],
    *,
    path: Path,
    expected_rows: int,
    expected_counts: Mapping[str, int],
    label: str,
) -> list[_SourceRow]:
    materialized = list(rows)
    if len(materialized) != expected_rows:
        raise ValueError(f"{path}: expected exactly {expected_rows} rows in {label}, found {len(materialized)}")
    ids = [row.value["question_id"] for row in materialized]
    if len(ids) != len(set(ids)):
        duplicates = sorted(question_id for question_id, count in Counter(ids).items() if count > 1)
        raise ValueError(f"{path}: duplicate question_id(s) in {label}: {duplicates[:5]}")
    counts = Counter(row.value["source_dataset"] for row in materialized)
    actual = {dataset: counts.get(dataset, 0) for dataset in DATASETS}
    if actual != dict(expected_counts):
        raise ValueError(f"{path}: expected {label} counts {dict(expected_counts)}, got {actual}")
    return materialized


def _jsonl_payload(rows: Iterable[_SourceRow]) -> bytes:
    # Preserve each source JSON object byte-for-byte; only normalize the record
    # separator to one trailing LF in the derived artifact.
    return b"".join(row.raw_line + b"\n" for row in rows)


def _ids_sha256(ids: Iterable[str]) -> str:
    return _sha256("".join(f"{question_id}\n" for question_id in ids).encode("utf-8"))


def _artifact_entry(
    path: Path,
    rows: list[_SourceRow],
    payload: bytes,
    *,
    source_start: int,
    source_stop: int,
) -> dict[str, Any]:
    ids = [row.value["question_id"] for row in rows]
    counts = Counter(row.value["source_dataset"] for row in rows)
    return {
        "path": str(path.resolve()),
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "counts_by_dataset": {dataset: counts.get(dataset, 0) for dataset in DATASETS},
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
        "source_rows_1_based_inclusive": [source_start + 1, source_stop],
    }


def _write_new_atomic(path: Path, payload: bytes) -> None:
    """Publish one file with a race-safe no-overwrite boundary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_iid_diagnostic(
    source: str | Path,
    output_dir: str | Path,
    manifest_output: str | Path | None = None,
    *,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the exact source and publish the two diagnostic populations."""

    source_path = Path(source)
    output_path = Path(output_dir)
    manifest_path = Path(manifest_output) if manifest_output is not None else output_path / DEFAULT_MANIFEST_FILENAME
    train_path = output_path / TRAIN_EVAL_FILENAME
    heldout_path = output_path / HELDOUT_FILENAME
    targets = (train_path, heldout_path, manifest_path)
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("train, held-out, and manifest outputs must be distinct paths")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Stage 1 paired source does not exist: {source_path}")

    source_payload = source_path.read_bytes()
    expected_hash = SOURCE_SHA256 if expected_source_sha256 is None else expected_source_sha256
    actual_hash = _sha256(source_payload)
    if actual_hash != expected_hash:
        raise ValueError(f"{source_path}: source SHA-256 mismatch; expected {expected_hash}, got {actual_hash}")
    source_rows = _validate_population(
        _read_rows(source_path, source_payload),
        path=source_path,
        expected_rows=SOURCE_ROWS,
        expected_counts=SOURCE_COUNTS,
        label="Stage 1 source",
    )

    train_rows = _validate_population(
        source_rows[:TRAIN_EVAL_ROWS],
        path=source_path,
        expected_rows=TRAIN_EVAL_ROWS,
        expected_counts=TRAIN_EVAL_COUNTS,
        label="exact-training diagnostic",
    )
    rmct_rows = _validate_population(
        source_rows[:RMCT_ROWS],
        path=source_path,
        expected_rows=RMCT_ROWS,
        expected_counts=RMCT_COUNTS,
        label="RMCT exact-training prefix",
    )
    heldout_rows = _validate_population(
        source_rows[HELDOUT_SOURCE_START:HELDOUT_SOURCE_STOP],
        path=source_path,
        expected_rows=HELDOUT_ROWS,
        expected_counts=HELDOUT_COUNTS,
        label="held-out in-domain diagnostic",
    )
    train_ids = [row.value["question_id"] for row in train_rows]
    heldout_ids = [row.value["question_id"] for row in heldout_rows]
    if not set(train_ids).isdisjoint(heldout_ids):
        raise AssertionError("exact-training and held-out diagnostic IDs overlap")

    train_payload = _jsonl_payload(train_rows)
    heldout_payload = _jsonl_payload(heldout_rows)
    rmct_ids = [row.value["question_id"] for row in rmct_rows]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "source": {
            "path": str(source_path.resolve()),
            "content_sha256": actual_hash,
            "byte_count": len(source_payload),
            "row_count": len(source_rows),
            "counts_by_dataset": SOURCE_COUNTS,
            "bias_type": BIAS_TYPE,
            "prompt_style": PROMPT_STYLE,
        },
        "selection": {
            "method": "fixed_source_offsets_without_shuffle",
            "training_prefix_rows": TRAINING_PREFIX_ROWS,
            "train_eval_source_rows_1_based_inclusive": [1, TRAIN_EVAL_ROWS],
            "heldout_pool_source_rows_1_based_inclusive": [TRAINING_PREFIX_ROWS + 1, SOURCE_ROWS],
            "heldout_selected_source_rows_1_based_inclusive": [
                HELDOUT_SOURCE_START + 1,
                HELDOUT_SOURCE_STOP,
            ],
            "rmct_source_rows_1_based_inclusive": [1, RMCT_ROWS],
        },
        "splits": {
            "train_eval": _artifact_entry(
                train_path,
                train_rows,
                train_payload,
                source_start=0,
                source_stop=TRAIN_EVAL_ROWS,
            ),
            "heldout_in_domain": _artifact_entry(
                heldout_path,
                heldout_rows,
                heldout_payload,
                source_start=HELDOUT_SOURCE_START,
                source_stop=HELDOUT_SOURCE_STOP,
            ),
        },
        "rmct_first64": {
            "row_count": RMCT_ROWS,
            "counts_by_dataset": RMCT_COUNTS,
            "question_ids": rmct_ids,
            "question_ids_sha256": _ids_sha256(rmct_ids),
            "source_rows_1_based_inclusive": [1, RMCT_ROWS],
        },
        "assertions": {
            "source_ids_unique": True,
            "train_eval_ids_unique": True,
            "heldout_in_domain_ids_unique": True,
            "train_eval_disjoint_from_heldout_in_domain": True,
            "rmct_first64_is_train_eval_prefix": rmct_ids == train_ids[:RMCT_ROWS],
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    _write_new_atomic(train_path, train_payload)
    _write_new_atomic(heldout_path, heldout_payload)
    _write_new_atomic(manifest_path, manifest_payload)
    validate_manifest(manifest_path, expected_source_sha256=expected_hash, verify_source=True)
    return manifest


def _load_document(manifest: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(manifest, Mapping):
        return dict(manifest), None
    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"Stage 1 IID manifest does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Stage 1 IID manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"Stage 1 IID manifest {path} must contain an object")
    return document, path.resolve()


def _checked_artifact_entry(
    entry: object,
    *,
    split: str,
    expected_rows: int,
    expected_counts: Mapping[str, int],
) -> tuple[Path, list[_SourceRow]]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest split {split!r} must be an object")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"manifest split {split!r} has no path")
    path = Path(raw_path)
    payload = path.read_bytes()
    if entry.get("content_sha256") != _sha256(payload):
        raise ValueError(f"manifest split {split!r} content SHA-256 mismatch: {path}")
    if entry.get("byte_count") != len(payload):
        raise ValueError(f"manifest split {split!r} byte_count mismatch: {path}")
    rows = _validate_population(
        _read_rows(path, payload),
        path=path,
        expected_rows=expected_rows,
        expected_counts=expected_counts,
        label=split,
    )
    ids = [row.value["question_id"] for row in rows]
    if entry.get("row_count") != len(rows):
        raise ValueError(f"manifest split {split!r} row_count mismatch: {path}")
    if entry.get("counts_by_dataset") != dict(expected_counts):
        raise ValueError(f"manifest split {split!r} counts_by_dataset mismatch: {path}")
    if entry.get("question_ids") != ids:
        raise ValueError(f"manifest split {split!r} question_ids mismatch: {path}")
    if entry.get("question_ids_sha256") != _ids_sha256(ids):
        raise ValueError(f"manifest split {split!r} question_ids_sha256 mismatch: {path}")
    return path, rows


def validate_manifest(
    manifest: str | Path | Mapping[str, Any],
    *,
    expected_source_sha256: str | None = None,
    verify_source: bool = False,
) -> dict[str, Any]:
    """Fail closed on the manifest, selected files, IDs, and optional source."""

    document, _ = _load_document(manifest)
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != MANIFEST_KIND:
        raise ValueError("unsupported Stage 1 IID diagnostic manifest schema")
    expected_hash = SOURCE_SHA256 if expected_source_sha256 is None else expected_source_sha256
    source = document.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("manifest source must be an object")
    expected_source = {
        "content_sha256": expected_hash,
        "row_count": SOURCE_ROWS,
        "counts_by_dataset": SOURCE_COUNTS,
        "bias_type": BIAS_TYPE,
        "prompt_style": PROMPT_STYLE,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ValueError(f"manifest source.{field} must be {expected!r}, got {source.get(field)!r}")

    splits = document.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train_eval", "heldout_in_domain"}:
        raise ValueError("manifest must contain exactly train_eval and heldout_in_domain splits")
    _, train_rows = _checked_artifact_entry(
        splits["train_eval"],
        split="train_eval",
        expected_rows=TRAIN_EVAL_ROWS,
        expected_counts=TRAIN_EVAL_COUNTS,
    )
    _, heldout_rows = _checked_artifact_entry(
        splits["heldout_in_domain"],
        split="heldout_in_domain",
        expected_rows=HELDOUT_ROWS,
        expected_counts=HELDOUT_COUNTS,
    )
    train_ids = [row.value["question_id"] for row in train_rows]
    heldout_ids = [row.value["question_id"] for row in heldout_rows]
    if not set(train_ids).isdisjoint(heldout_ids):
        raise ValueError("manifest diagnostic splits overlap")

    rmct = document.get("rmct_first64")
    if not isinstance(rmct, Mapping):
        raise ValueError("manifest rmct_first64 must be an object")
    expected_rmct_ids = train_ids[:RMCT_ROWS]
    if rmct.get("row_count") != RMCT_ROWS or rmct.get("counts_by_dataset") != RMCT_COUNTS:
        raise ValueError("manifest rmct_first64 count metadata is invalid")
    if rmct.get("question_ids") != expected_rmct_ids:
        raise ValueError("manifest rmct_first64 IDs are not the train_eval prefix")
    if rmct.get("question_ids_sha256") != _ids_sha256(expected_rmct_ids):
        raise ValueError("manifest rmct_first64 question_ids_sha256 mismatch")

    selection = document.get("selection")
    expected_selection = {
        "method": "fixed_source_offsets_without_shuffle",
        "training_prefix_rows": TRAINING_PREFIX_ROWS,
        "train_eval_source_rows_1_based_inclusive": [1, TRAIN_EVAL_ROWS],
        "heldout_pool_source_rows_1_based_inclusive": [TRAINING_PREFIX_ROWS + 1, SOURCE_ROWS],
        "heldout_selected_source_rows_1_based_inclusive": [HELDOUT_SOURCE_START + 1, HELDOUT_SOURCE_STOP],
        "rmct_source_rows_1_based_inclusive": [1, RMCT_ROWS],
    }
    if selection != expected_selection:
        raise ValueError("manifest selection contract is invalid")

    if verify_source:
        raw_path = source.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("manifest source.path must be a non-empty string")
        source_path = Path(raw_path)
        payload = source_path.read_bytes()
        if _sha256(payload) != expected_hash or source.get("byte_count") != len(payload):
            raise ValueError(f"manifest source file identity mismatch: {source_path}")
        source_rows = _validate_population(
            _read_rows(source_path, payload),
            path=source_path,
            expected_rows=SOURCE_ROWS,
            expected_counts=SOURCE_COUNTS,
            label="Stage 1 source",
        )
        source_ids = [row.value["question_id"] for row in source_rows]
        if train_ids != source_ids[:TRAIN_EVAL_ROWS]:
            raise ValueError("train_eval IDs do not match source rows 1-200")
        if heldout_ids != source_ids[HELDOUT_SOURCE_START:HELDOUT_SOURCE_STOP]:
            raise ValueError("heldout_in_domain IDs do not match source rows 2049-2248")
    return document


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the exact Stage 1 training-domain diagnostic populations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_iid_diagnostic(args.source, args.output_dir, args.manifest_output)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared Stage 1 IID diagnostic: train_eval={manifest['splits']['train_eval']['row_count']}, heldout_in_domain={manifest['splits']['heldout_in_domain']['row_count']}; no model or remote service was called.")


if __name__ == "__main__":
    main()
