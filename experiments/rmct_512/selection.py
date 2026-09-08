"""Freeze the disjoint second 256-example segment for RMCT-512.

``RMCT-512`` is intentionally *sequential data scaling*, not a fresh
512-example run.  The parent RMCT-256 segment consumes canonical rows 1--256;
this artifact contains only rows 257--512.  A protected continuation must
therefore restore the parent optimizer state and train this JSONL exactly once.

The source proof delegates to the already audited RMCT-256 selector, then adds
the two properties that matter for a continuation: byte-level disjointness
from the parent and the exact contiguous canonical range of the new segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctm.artifacts import plain_file_identity
from experiments.rmct_256 import selection as parent


SCHEMA_VERSION = 1
MANIFEST_KIND = "rmct512_continuation_selection_manifest"
DATA_FILENAME_PREFIX = "rmct-512-continuation-"
MANIFEST_FILENAME_PREFIX = "rmct-512-continuation-manifest-"
PARENT_ROWS = 256
CONTINUATION_ROWS = 256
TOTAL_ROWS = PARENT_ROWS + CONTINUATION_ROWS
CONTINUATION_COUNTS = {"logiqa": 128, "hellaswag": 128}


@dataclass(frozen=True, slots=True)
class MaterializedContinuation:
    data_path: Path
    manifest_path: Path
    data_sha256: str
    manifest_sha256: str
    data_status: str
    manifest_status: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _ids_sha256(ids: Sequence[str]) -> str:
    return _sha256("".join(f"{item}\n" for item in ids).encode("utf-8"))


def _publish_immutable(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing immutable artifact: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing immutable artifact: {path}") from None
        return "resumed"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "written"


def _continuation_entry(rows: Sequence[parent._RawRow], payload: bytes) -> dict[str, Any]:
    ids = [str(row.value["question_id"]) for row in rows]
    return {
        "filename": f"{DATA_FILENAME_PREFIX}{_sha256(payload)}.jsonl",
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "counts_by_dataset": {
            dataset: sum(row.value["source_dataset"] == dataset for row in rows) for dataset in parent.DATASETS
        },
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
        "source_rows_1_based_inclusive": [PARENT_ROWS + 1, TOTAL_ROWS],
        "selection_method": "exact_ordered_canonical_source_continuation_without_shuffle_or_reserialization",
    }


def _validate_parent_selection(
    *,
    parent_selection: str | Path,
    parent_selection_manifest: str | Path,
    canonical_source: str | Path,
    canonical_source_manifest: str | Path,
    recovered_none_source: str | Path,
    recovered_none_manifest: str | Path,
    original64_reference_manifest: str | Path,
    stage2_manifest: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Verify the parent fully, then return its immutable identities."""

    selection_path = Path(parent_selection).resolve()
    manifest_path = Path(parent_selection_manifest).resolve()
    document = parent.verify_selection_manifest(
        manifest_path,
        selection_path=selection_path,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
        verify_sources=True,
    )
    if document.get("selection", {}).get("row_count") != PARENT_ROWS:
        raise ValueError("parent RMCT-256 selection does not contain exactly 256 rows")
    return selection_path, manifest_path, document


def _validate_rows(rows: Sequence[parent._RawRow], *, label: str) -> list[str]:
    return parent._validated_ids(
        rows,
        path=Path(f"<{label}>"),
        label=label,
        expected_rows=CONTINUATION_ROWS,
        expected_counts=CONTINUATION_COUNTS,
        require_canonical_pair=True,
    )


def _build_manifest(
    *,
    source_proof: Mapping[str, Any],
    parent_selection_path: Path,
    parent_manifest_path: Path,
    parent_document: Mapping[str, Any],
    continuation_rows: Sequence[parent._RawRow],
    continuation_payload: bytes,
    stage2_proof: Mapping[str, Any],
) -> dict[str, Any]:
    entry = _continuation_entry(continuation_rows, continuation_payload)
    parent_selection = parent_document["selection"]
    assert isinstance(parent_selection, Mapping)  # proved by the parent verifier
    parent_ids = list(parent_selection["question_ids"])
    continuation_ids = list(entry["question_ids"])
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "model": parent.MODEL,
        "source": dict(source_proof),
        "parent_rmct256": {
            "selection": plain_file_identity(parent_selection_path),
            "selection_manifest": plain_file_identity(parent_manifest_path),
            "selection_manifest_document_sha256": _sha256(_canonical_json(parent_document)),
            "row_count": PARENT_ROWS,
            "question_ids_sha256": _ids_sha256(parent_ids),
            "source_rows_1_based_inclusive": [1, PARENT_ROWS],
        },
        "continuation": entry,
        "stage2_in_domain_exclusion": {
            **dict(stage2_proof),
            "overlap_count": 0,
            "overlap_question_ids": [],
        },
        "assertions": {
            "parent_rmct256_is_fully_verified": True,
            "parent_and_continuation_question_ids_are_disjoint": True,
            "continuation_is_exact_canonical_rows_257_to_512": True,
            "continuation_ids_unique": True,
            "continuation_counts_are_128_logiqa_and_128_hellaswag": True,
            "stage2_in_domain_has_zero_overlap_with_continuation": True,
            "sequential_total_unique_rows": TOTAL_ROWS,
        },
    }


def materialize_rmct512_continuation(
    *,
    parent_selection: str | Path,
    parent_selection_manifest: str | Path,
    canonical_source: str | Path,
    canonical_source_manifest: str | Path,
    recovered_none_source: str | Path,
    recovered_none_manifest: str | Path,
    original64_reference_manifest: str | Path,
    stage2_manifest: str | Path,
    output_dir: str | Path,
) -> MaterializedContinuation:
    """Publish immutable canonical rows 257--512 after proving all ancestry."""

    source_path, source_rows, _, source_proof = parent._verify_canonical_source(
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
    )
    if len(source_rows) < TOTAL_ROWS:
        raise ValueError(f"canonical source has {len(source_rows)} rows; RMCT-512 requires {TOTAL_ROWS}")
    parent_path, parent_manifest_path, parent_document = _validate_parent_selection(
        parent_selection=parent_selection,
        parent_selection_manifest=parent_selection_manifest,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
    )
    parent_rows_path, parent_rows, parent_payload = parent._read_jsonl(parent_path, label="parent RMCT-256 selection")
    del parent_rows_path
    expected_parent_payload = b"".join(row.raw_line for row in source_rows[:PARENT_ROWS])
    if parent_payload != expected_parent_payload:
        raise ValueError("parent RMCT-256 JSONL does not byte-match canonical source rows 1--256")
    parent_ids = [str(row.value["question_id"]) for row in parent_rows]
    continuation_rows = source_rows[PARENT_ROWS:TOTAL_ROWS]
    continuation_payload = b"".join(row.raw_line for row in continuation_rows)
    continuation_ids = _validate_rows(continuation_rows, label="RMCT-512 continuation")
    if set(parent_ids).intersection(continuation_ids):
        raise ValueError("RMCT-512 continuation overlaps parent RMCT-256 question IDs")
    if parent_ids + continuation_ids != [str(row.value["question_id"]) for row in source_rows[:TOTAL_ROWS]]:
        raise ValueError("parent plus continuation IDs do not reconstruct canonical rows 1--512")
    stage2_ids, stage2_proof = parent._verify_stage2_in_domain(stage2_manifest)
    overlap = sorted(set(continuation_ids).intersection(stage2_ids))
    if overlap:
        raise ValueError(f"RMCT-512 continuation overlaps frozen Stage-2 in-domain IDs: {overlap[:5]}")

    document = _build_manifest(
        source_proof=source_proof,
        parent_selection_path=parent_path,
        parent_manifest_path=parent_manifest_path,
        parent_document=parent_document,
        continuation_rows=continuation_rows,
        continuation_payload=continuation_payload,
        stage2_proof=stage2_proof,
    )
    payload = _canonical_json(document)
    data_digest = _sha256(continuation_payload)
    manifest_digest = _sha256(payload)
    output = Path(output_dir).resolve()
    data_path = output / f"{DATA_FILENAME_PREFIX}{data_digest}.jsonl"
    manifest_path = output / f"{MANIFEST_FILENAME_PREFIX}{manifest_digest}.json"
    data_status = _publish_immutable(data_path, continuation_payload)
    manifest_status = _publish_immutable(manifest_path, payload)
    verify_rmct512_continuation_manifest(
        manifest_path,
        continuation_path=data_path,
        parent_selection=parent_selection,
        parent_selection_manifest=parent_selection_manifest,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
    )
    return MaterializedContinuation(
        data_path=data_path,
        manifest_path=manifest_path,
        data_sha256=data_digest,
        manifest_sha256=manifest_digest,
        data_status=data_status,
        manifest_status=manifest_status,
    )


def verify_rmct512_continuation_manifest(
    manifest_path: str | Path,
    *,
    continuation_path: str | Path | None = None,
    parent_selection: str | Path | None = None,
    parent_selection_manifest: str | Path | None = None,
    canonical_source: str | Path | None = None,
    canonical_source_manifest: str | Path | None = None,
    recovered_none_source: str | Path | None = None,
    recovered_none_manifest: str | Path | None = None,
    original64_reference_manifest: str | Path | None = None,
    stage2_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Fully verify an RMCT-512 continuation selection, fail closed on drift."""

    required = {
        "parent_selection": parent_selection,
        "parent_selection_manifest": parent_selection_manifest,
        "canonical_source": canonical_source,
        "canonical_source_manifest": canonical_source_manifest,
        "recovered_none_source": recovered_none_source,
        "recovered_none_manifest": recovered_none_manifest,
        "original64_reference_manifest": original64_reference_manifest,
        "stage2_manifest": stage2_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("full RMCT-512 continuation proof requires explicit input path(s): " + ", ".join(missing))
    path = Path(manifest_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"RMCT-512 continuation manifest must be a regular file: {path}")
    payload = path.read_bytes()
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"RMCT-512 continuation manifest is not valid JSON: {path}") from exc
    if not isinstance(document, dict) or _canonical_json(document) != payload:
        raise ValueError("RMCT-512 continuation manifest is not canonical JSON bytes")
    digest = _sha256(payload)
    expected_name = f"{MANIFEST_FILENAME_PREFIX}{digest}.json"
    if path.name != expected_name:
        raise ValueError(f"RMCT-512 continuation manifest filename must be content-addressed {expected_name}")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != MANIFEST_KIND:
        raise ValueError("RMCT-512 continuation manifest has an unsupported schema/kind")
    if document.get("model") != parent.MODEL:
        raise ValueError("RMCT-512 continuation manifest model differs from the frozen Qwen3.5 contract")

    source_path, source_rows, _, source_proof = parent._verify_canonical_source(
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
    )
    del source_path
    if document.get("source") != source_proof:
        raise ValueError("RMCT-512 continuation source proof differs from the canonical source")
    parent_path, parent_manifest_path, parent_document = _validate_parent_selection(
        parent_selection=parent_selection,
        parent_selection_manifest=parent_selection_manifest,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
    )
    parent_record = document.get("parent_rmct256")
    if not isinstance(parent_record, Mapping):
        raise ValueError("RMCT-512 continuation manifest has no parent_rmct256 record")
    expected_parent_record = {
        "selection": plain_file_identity(parent_path),
        "selection_manifest": plain_file_identity(parent_manifest_path),
        "selection_manifest_document_sha256": _sha256(_canonical_json(parent_document)),
        "row_count": PARENT_ROWS,
        "question_ids_sha256": parent_document["selection"]["question_ids_sha256"],
        "source_rows_1_based_inclusive": [1, PARENT_ROWS],
    }
    if dict(parent_record) != expected_parent_record:
        raise ValueError("RMCT-512 continuation manifest parent selection record differs from verified RMCT-256")

    continuation = document.get("continuation")
    if not isinstance(continuation, Mapping):
        raise ValueError("RMCT-512 continuation manifest has no continuation record")
    name = continuation.get("filename")
    if not isinstance(name, str) or not name:
        raise ValueError("RMCT-512 continuation filename is missing")
    data_path = Path(continuation_path).resolve() if continuation_path is not None else path.parent / name
    if data_path.name != name or data_path.is_symlink() or not data_path.is_file():
        raise ValueError("RMCT-512 continuation JSONL path is invalid")
    _, rows, data_payload = parent._read_jsonl(data_path, label="RMCT-512 continuation JSONL")
    expected_rows = source_rows[PARENT_ROWS:TOTAL_ROWS]
    expected_payload = b"".join(row.raw_line for row in expected_rows)
    if data_payload != expected_payload:
        raise ValueError("RMCT-512 continuation JSONL does not byte-match canonical source rows 257--512")
    ids = _validate_rows(rows, label="RMCT-512 continuation")
    expected_entry = _continuation_entry(expected_rows, expected_payload)
    if dict(continuation) != expected_entry:
        raise ValueError("RMCT-512 continuation record differs from canonical rows 257--512")
    parent_ids = list(parent_document["selection"]["question_ids"])
    if set(parent_ids).intersection(ids):
        raise ValueError("RMCT-512 continuation has overlap with the verified RMCT-256 parent")
    stage2_ids, stage2_proof = parent._verify_stage2_in_domain(stage2_manifest)
    if set(ids).intersection(stage2_ids):
        raise ValueError("RMCT-512 continuation has overlap with Stage-2 in-domain IDs")
    if document.get("stage2_in_domain_exclusion") != {
        **stage2_proof,
        "overlap_count": 0,
        "overlap_question_ids": [],
    }:
        raise ValueError("RMCT-512 continuation Stage-2 exclusion proof differs")
    expected_assertions = {
        "parent_rmct256_is_fully_verified": True,
        "parent_and_continuation_question_ids_are_disjoint": True,
        "continuation_is_exact_canonical_rows_257_to_512": True,
        "continuation_ids_unique": True,
        "continuation_counts_are_128_logiqa_and_128_hellaswag": True,
        "stage2_in_domain_has_zero_overlap_with_continuation": True,
        "sequential_total_unique_rows": TOTAL_ROWS,
    }
    if document.get("assertions") != expected_assertions:
        raise ValueError("RMCT-512 continuation assertions differ from the fixed contract")
    return document


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent-selection", required=True, type=Path)
    parser.add_argument("--parent-selection-manifest", required=True, type=Path)
    parser.add_argument("--canonical-source", required=True, type=Path)
    parser.add_argument("--canonical-source-manifest", required=True, type=Path)
    parser.add_argument("--recovered-none-source", required=True, type=Path)
    parser.add_argument("--recovered-none-manifest", required=True, type=Path)
    parser.add_argument("--original64-reference-manifest", required=True, type=Path)
    parser.add_argument("--stage2-manifest", required=True, type=Path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize")
    _add_source_arguments(materialize)
    materialize.add_argument("--output-dir", required=True, type=Path)
    verify = commands.add_parser("verify")
    _add_source_arguments(verify)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--continuation", type=Path)
    args = parser.parse_args(argv)
    common = {
        "parent_selection": args.parent_selection,
        "parent_selection_manifest": args.parent_selection_manifest,
        "canonical_source": args.canonical_source,
        "canonical_source_manifest": args.canonical_source_manifest,
        "recovered_none_source": args.recovered_none_source,
        "recovered_none_manifest": args.recovered_none_manifest,
        "original64_reference_manifest": args.original64_reference_manifest,
        "stage2_manifest": args.stage2_manifest,
    }
    try:
        if args.command == "materialize":
            result = materialize_rmct512_continuation(output_dir=args.output_dir, **common)
            print(json.dumps({
                "data_path": str(result.data_path),
                "data_sha256": result.data_sha256,
                "data_status": result.data_status,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "manifest_status": result.manifest_status,
            }, sort_keys=True))
        else:
            document = verify_rmct512_continuation_manifest(
                args.manifest,
                continuation_path=args.continuation,
                **common,
            )
            print(json.dumps({
                "manifest": str(args.manifest.resolve()),
                "continuation_sha256": document["continuation"]["content_sha256"],
                "row_count": document["continuation"]["row_count"],
                "verified": True,
            }, sort_keys=True))
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "CONTINUATION_COUNTS",
    "CONTINUATION_ROWS",
    "DATA_FILENAME_PREFIX",
    "MANIFEST_FILENAME_PREFIX",
    "MANIFEST_KIND",
    "MaterializedContinuation",
    "PARENT_ROWS",
    "TOTAL_ROWS",
    "materialize_rmct512_continuation",
    "verify_rmct512_continuation_manifest",
]
