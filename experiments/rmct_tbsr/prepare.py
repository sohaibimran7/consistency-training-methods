"""Freeze a balanced RMCT training-domain evaluation subset without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.rmct_tbsr.constants import (
    AUTHORITATIVE_TRAINING_COUNTS,
    AUTHORITATIVE_TRAINING_ROWS,
    DEFAULT_SOURCE,
    DEFAULT_TRAINING_MANIFEST,
    DEFAULT_TRAINING_OUTPUT,
    SOURCE_ROWS,
    SOURCE_SHA256,
    TRAINING_COUNTS,
    TRAINING_ROWS,
    TRAINING_SELECTION_SEED,
    TRAINING_SHA256,
)

SCHEMA_VERSION = 1
MANIFEST_KIND = "rmct_tbsr_training_manifest"

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
class TrainingArtifact:
    """A completely verified frozen training population and its provenance."""

    path: Path
    rows: tuple[dict[str, Any], ...]
    manifest_identity: dict[str, Any]
    source_identity: dict[str, Any]
    training_identity: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _identity(path: Path, payload: bytes, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "content_sha256": _sha256(payload),
        "bytes": len(payload),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _validate_messages(value: object, *, location: str, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{location}: {field} must be a non-empty message list")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise TypeError(f"{location}: {field}[{index}] must be an object")
        for key in ("role", "content"):
            item = message.get(key)
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{location}: {field}[{index}].{key} must be a non-empty string")


def _validate_training_row(row: object, *, path: Path, line_number: int) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    if not isinstance(row, dict):
        raise TypeError(f"{location}: row must be a JSON object")
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
            raise TypeError(f"{location}: {field} must be a string")
    if not row["question_id"].strip():
        raise ValueError(f"{location}: question_id must not be empty")
    if row["source_dataset"] not in TRAINING_COUNTS:
        raise ValueError(f"{location}: source_dataset must be one of {list(TRAINING_COUNTS)}")
    if row["bias_type"] != "wrong_argument":
        raise ValueError(f"{location}: bias_type must be 'wrong_argument'")
    if row["prompt_style"] != "none":
        raise ValueError(f"{location}: prompt_style must be 'none'")
    if not row["biased_option"].strip() or row["biased_option"] == row["ground_truth"]:
        raise ValueError(f"{location}: biased_option must be a non-empty distractor")
    _validate_messages(row["unbiased_messages"], location=location, field="unbiased_messages")
    _validate_messages(row["biased_messages"], location=location, field="biased_messages")
    return row


def _read_source(
    path: Path,
    payload: bytes,
    *,
    expected_rows: int,
) -> list[tuple[dict[str, Any], bytes]]:
    authoritative_rows: list[tuple[dict[str, Any], bytes]] = []
    row_count = 0
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        row_count += 1
        if len(authoritative_rows) >= AUTHORITATIVE_TRAINING_ROWS:
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        authoritative_rows.append((_validate_training_row(row, path=path, line_number=line_number), raw_line))
    if row_count != expected_rows:
        raise ValueError(f"{path}: expected exactly {expected_rows} non-empty rows, found {row_count}")
    if len(authoritative_rows) != AUTHORITATIVE_TRAINING_ROWS:
        raise ValueError(
            f"{path}: expected at least {AUTHORITATIVE_TRAINING_ROWS} authoritative rows, "
            f"found {len(authoritative_rows)}"
        )
    _validate_population(
        [row for row, _ in authoritative_rows],
        path=path,
        expected_counts=AUTHORITATIVE_TRAINING_COUNTS,
        expected_rows=AUTHORITATIVE_TRAINING_ROWS,
        label="authoritative RMCT prefix",
    )
    return authoritative_rows


def _selection_rank(row: dict[str, Any]) -> tuple[bytes, str]:
    payload = f"{TRAINING_SELECTION_SEED}:{row['source_dataset']}:{row['question_id']}".encode("utf-8")
    return hashlib.sha256(payload).digest(), row["question_id"]


def _select_training_subset(
    authoritative_rows: list[tuple[dict[str, Any], bytes]],
) -> tuple[list[dict[str, Any]], bytes]:
    selected_ids: set[str] = set()
    for dataset, count in TRAINING_COUNTS.items():
        candidates = [row for row, _ in authoritative_rows if row["source_dataset"] == dataset]
        ranked = sorted(candidates, key=_selection_rank)
        if len(ranked) < count:
            raise ValueError(f"authoritative RMCT prefix has fewer than {count} rows for {dataset}")
        selected_ids.update(row["question_id"] for row in ranked[:count])

    selected = [(row, raw_line) for row, raw_line in authoritative_rows if row["question_id"] in selected_ids]
    rows = [row for row, _ in selected]
    _validate_population(
        rows,
        path=Path("<deterministic-selection>"),
        expected_counts=TRAINING_COUNTS,
        expected_rows=TRAINING_ROWS,
        label="balanced RMCT evaluation subset",
    )
    payload = b"".join(raw_line + b"\n" for _, raw_line in selected)
    return rows, payload


def _read_training_file(path: Path, payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        rows.append(_validate_training_row(value, path=path, line_number=line_number))
    _validate_population(
        rows,
        path=path,
        expected_counts=TRAINING_COUNTS,
        expected_rows=TRAINING_ROWS,
        label="balanced RMCT evaluation subset",
    )
    return rows


def _validate_population(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    expected_counts: dict[str, int],
    expected_rows: int,
    label: str,
) -> None:
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected exactly {expected_rows} rows in {label}, found {len(rows)}")
    ids = [row["question_id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(question_id for question_id, count in Counter(ids).items() if count > 1)
        raise ValueError(f"{path}: duplicate question_id(s): {duplicates[:5]}")
    counts = Counter(row["source_dataset"] for row in rows)
    actual = {dataset: counts.get(dataset, 0) for dataset in expected_counts}
    if actual != expected_counts:
        raise ValueError(f"{path}: expected dataset counts {expected_counts}, got {actual}")


def _write_new_atomic(path: Path, payload: bytes) -> None:
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


def prepare_training_population(
    source: str | Path = DEFAULT_SOURCE,
    output: str | Path = DEFAULT_TRAINING_OUTPUT,
    manifest_output: str | Path = DEFAULT_TRAINING_MANIFEST,
    *,
    expected_source_sha256: str | None = None,
    expected_source_rows: int = SOURCE_ROWS,
) -> dict[str, Any]:
    """Publish the deterministic balanced subset and a fail-closed manifest."""

    source_path = Path(source)
    output_path = Path(output)
    manifest_path = Path(manifest_output)
    if not source_path.is_file():
        raise FileNotFoundError(f"RMCT source does not exist: {source_path}")
    targets = (output_path, manifest_path)
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("training output and manifest output must be distinct paths")
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output(s): {existing}")

    expected_hash = SOURCE_SHA256 if expected_source_sha256 is None else expected_source_sha256
    source_payload = source_path.read_bytes()
    actual_hash = _sha256(source_payload)
    if actual_hash != expected_hash:
        raise ValueError(f"{source_path}: source SHA-256 mismatch; expected {expected_hash}, got {actual_hash}")
    authoritative_rows = _read_source(source_path, source_payload, expected_rows=expected_source_rows)
    rows, training_payload = _select_training_subset(authoritative_rows)

    counts = Counter(row["source_dataset"] for row in rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "source": {
            **_identity(source_path, source_payload, rows=expected_source_rows),
            "authoritative_prefix_rows": AUTHORITATIVE_TRAINING_ROWS,
            "authoritative_counts_by_dataset": AUTHORITATIVE_TRAINING_COUNTS,
        },
        "selection": {
            "method": "lowest_sha256_rank_within_dataset_then_source_order",
            "rank_input": "{seed}:{source_dataset}:{question_id}",
            "seed": TRAINING_SELECTION_SEED,
            "row_count": TRAINING_ROWS,
            "counts_by_dataset": TRAINING_COUNTS,
            "bias_type": "wrong_argument",
            "prompt_style": "none",
        },
        "training": {
            **_identity(output_path, training_payload, rows=TRAINING_ROWS),
            "counts_by_dataset": {dataset: counts[dataset] for dataset in TRAINING_COUNTS},
            "question_ids": [row["question_id"] for row in rows],
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_new_atomic(output_path, training_payload)
    _write_new_atomic(manifest_path, manifest_payload)
    return manifest


def validate_training_manifest(
    manifest: str | Path,
    *,
    expected_source_sha256: str | None = None,
    expected_training_sha256: str | None = None,
) -> TrainingArtifact:
    """Resolve and verify the source, manifest, frozen bytes, IDs, and counts."""

    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RMCT training manifest does not exist: {manifest_path}")
    manifest_payload = manifest_path.read_bytes()
    try:
        document = json.loads(manifest_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid RMCT training manifest {manifest_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"{manifest_path}: manifest must be a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != MANIFEST_KIND:
        raise ValueError(f"{manifest_path}: unsupported RMCT training manifest schema")

    source_entry = document.get("source")
    training_entry = document.get("training")
    if not isinstance(source_entry, dict) or not isinstance(training_entry, dict):
        raise TypeError(f"{manifest_path}: manifest lacks source/training entries")
    expected_source_hash = SOURCE_SHA256 if expected_source_sha256 is None else expected_source_sha256
    expected_training_hash = TRAINING_SHA256 if expected_training_sha256 is None else expected_training_sha256
    if source_entry.get("content_sha256") != expected_source_hash or source_entry.get("rows") != SOURCE_ROWS:
        raise ValueError(f"{manifest_path}: source identity does not match the RMCT paper source")
    if (
        source_entry.get("authoritative_prefix_rows") != AUTHORITATIVE_TRAINING_ROWS
        or source_entry.get("authoritative_counts_by_dataset") != AUTHORITATIVE_TRAINING_COUNTS
    ):
        raise ValueError(
            f"{manifest_path}: authoritative prefix must contain the exact "
            f"{AUTHORITATIVE_TRAINING_COUNTS} population"
        )

    def resolve(entry: dict[str, Any], label: str) -> Path:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{manifest_path}: {label} entry has no path")
        path = Path(raw_path)
        return path if path.is_absolute() else manifest_path.resolve().parent / path

    source_path = resolve(source_entry, "source")
    training_path = resolve(training_entry, "training")
    if not source_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(f"{manifest_path}: source and frozen training files must both exist")
    source_payload = source_path.read_bytes()
    if _sha256(source_payload) != expected_source_hash:
        raise ValueError(f"{source_path}: source SHA-256 does not match the manifest")
    authoritative_rows = _read_source(source_path, source_payload, expected_rows=SOURCE_ROWS)
    selected_source_rows, selected_source_payload = _select_training_subset(authoritative_rows)
    training_payload = training_path.read_bytes()
    actual_training_hash = _sha256(training_payload)
    if training_entry.get("content_sha256") != actual_training_hash:
        raise ValueError(f"{training_path}: content SHA-256 does not match the manifest")
    if actual_training_hash != expected_training_hash:
        raise ValueError(f"{training_path}: content does not match the exact balanced RMCT subset")
    if training_payload != selected_source_payload:
        raise ValueError(f"{training_path}: bytes do not equal the deterministic balanced source selection")
    rows = _read_training_file(training_path, training_payload)
    if rows != selected_source_rows:
        raise ValueError(f"{training_path}: rows do not equal the deterministic balanced source selection")

    ids = [row["question_id"] for row in rows]
    counts = Counter(row["source_dataset"] for row in rows)
    manifest_counts = {dataset: counts[dataset] for dataset in TRAINING_COUNTS}
    if training_entry.get("rows") != TRAINING_ROWS:
        raise ValueError(f"{manifest_path}: training row count does not match the frozen file")
    if training_entry.get("question_ids") != ids:
        raise ValueError(f"{manifest_path}: ordered question IDs do not match the frozen file")
    if training_entry.get("counts_by_dataset") != manifest_counts:
        raise ValueError(f"{manifest_path}: dataset counts do not match the frozen file")
    if document.get("selection") != {
        "method": "lowest_sha256_rank_within_dataset_then_source_order",
        "rank_input": "{seed}:{source_dataset}:{question_id}",
        "seed": TRAINING_SELECTION_SEED,
        "row_count": TRAINING_ROWS,
        "counts_by_dataset": TRAINING_COUNTS,
        "bias_type": "wrong_argument",
        "prompt_style": "none",
    }:
        raise ValueError(f"{manifest_path}: selection declaration is not the exact balanced RMCT rule")
    return TrainingArtifact(
        path=training_path.resolve(),
        rows=tuple(rows),
        manifest_identity=_identity(manifest_path, manifest_payload),
        source_identity=_identity(source_path, source_payload, rows=SOURCE_ROWS),
        training_identity=_identity(training_path, training_payload, rows=TRAINING_ROWS),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_TRAINING_OUTPUT))
    parser.add_argument("--manifest", type=Path, default=Path(DEFAULT_TRAINING_MANIFEST))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = prepare_training_population(args.source, args.output, args.manifest)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"Prepared immutable RMCT training population n={manifest['training']['rows']} at "
        f"{manifest['training']['path']}; no model service was called."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
