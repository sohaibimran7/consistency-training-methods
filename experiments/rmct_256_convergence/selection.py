"""Materialize and verify four exact convergence blocks from RMCT-256.

The convergence protocol consumes the already attested, immutable 256-row
RMCT selection directly.  It does not create substitute training JSONLs:
each 64-row block is a byte-exact contiguous range of the parent file, loaded
at runtime through ``row_offset``.  This manifest pins the four ranges and
their raw-byte/QID identities so a launch can fail closed before sampling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.rmct_256 import selection as parent


SCHEMA_VERSION = 1
MANIFEST_KIND = "rmct256_convergence_segments_manifest"
MANIFEST_FILENAME_PREFIX = "rmct-256-convergence-segments-manifest-"
PARENT_ROWS = 256
SEGMENT_ROWS = 64
SEGMENT_COUNT = 4
SEGMENT_COUNTS = {"logiqa": 32, "hellaswag": 32}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MaterializedConvergenceSegments:
    """Identity of one immutable, content-addressed segment manifest."""

    manifest_path: Path
    manifest_sha256: str
    manifest_status: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _ids_sha256(ids: Sequence[str]) -> str:
    return _sha256("".join(f"{question_id}\n" for question_id in ids).encode("utf-8"))


def _require_regular_file(path: str | Path, *, label: str) -> Path:
    supplied = Path(path)
    resolved = supplied.resolve()
    if supplied.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {supplied}")
    return resolved


def _read_document(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any], bytes]:
    resolved = _require_regular_file(path, label=label)
    payload = resolved.read_bytes()
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain one JSON object: {resolved}")
    return resolved, document, payload


def _portable_file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "filename": path.name,
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": sum(1 for line in payload.splitlines() if line.strip()),
    }


def _validate_expected_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError("expected_manifest_sha256 must be a lowercase 64-character SHA-256 digest")
    return value


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


def _require_full_parent_inputs(
    *,
    canonical_source: str | Path | None,
    canonical_source_manifest: str | Path | None,
    recovered_none_source: str | Path | None,
    recovered_none_manifest: str | Path | None,
    original64_reference_manifest: str | Path | None,
    stage2_manifest: str | Path | None,
) -> dict[str, str | Path]:
    supplied = {
        "canonical_source": canonical_source,
        "canonical_source_manifest": canonical_source_manifest,
        "recovered_none_source": recovered_none_source,
        "recovered_none_manifest": recovered_none_manifest,
        "original64_reference_manifest": original64_reference_manifest,
        "stage2_manifest": stage2_manifest,
    }
    missing = [name for name, value in supplied.items() if value is None]
    if missing:
        raise ValueError("full convergence-segment proof requires explicit input path(s): " + ", ".join(missing))
    return {name: value for name, value in supplied.items() if value is not None}


def _validate_parent_selection(
    *,
    parent_selection: str | Path,
    parent_selection_manifest: str | Path,
    verify_parent_sources: bool,
    canonical_source: str | Path | None = None,
    canonical_source_manifest: str | Path | None = None,
    recovered_none_source: str | Path | None = None,
    recovered_none_manifest: str | Path | None = None,
    original64_reference_manifest: str | Path | None = None,
    stage2_manifest: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Any], list[parent._RawRow], bytes, list[str]]:
    """Validate the parent selection and return its raw, immutable records.

    ``verify_parent_sources=False`` is deliberately a transport check only:
    it validates the content-addressed parent data/manifest binding. Protected
    target attestation must use the full source proof.
    """

    selection_path = _require_regular_file(parent_selection, label="parent RMCT-256 selection")
    manifest_path = _require_regular_file(parent_selection_manifest, label="parent RMCT-256 selection manifest")
    if verify_parent_sources:
        inputs = _require_full_parent_inputs(
            canonical_source=canonical_source,
            canonical_source_manifest=canonical_source_manifest,
            recovered_none_source=recovered_none_source,
            recovered_none_manifest=recovered_none_manifest,
            original64_reference_manifest=original64_reference_manifest,
            stage2_manifest=stage2_manifest,
        )
        document = parent.verify_selection_manifest(
            manifest_path,
            selection_path=selection_path,
            verify_sources=True,
            **inputs,
        )
    else:
        document = parent.verify_selection_manifest(
            manifest_path,
            selection_path=selection_path,
            verify_sources=False,
        )

    _, rows, payload = parent._read_jsonl(selection_path, label="parent RMCT-256 selection")
    ids = parent._validated_ids(
        rows,
        path=selection_path,
        label="parent RMCT-256 selection",
        expected_rows=PARENT_ROWS,
        expected_counts=parent.SELECTION_COUNTS,
        require_canonical_pair=True,
    )
    selection_entry = document.get("selection")
    if not isinstance(selection_entry, Mapping):
        raise ValueError("verified RMCT-256 selection manifest lacks a selection record")
    expected_selection = {
        "filename": selection_path.name,
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": PARENT_ROWS,
        "counts_by_dataset": dict(parent.SELECTION_COUNTS),
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
        "source_rows_1_based_inclusive": [1, PARENT_ROWS],
        "selection_method": "exact_ordered_source_prefix_without_shuffle_or_reserialization",
    }
    if dict(selection_entry) != expected_selection:
        raise ValueError("parent RMCT-256 selection record differs from its actual immutable JSONL")
    return selection_path, manifest_path, document, rows, payload, ids


def _segment_entry(index: int, rows: Sequence[parent._RawRow]) -> dict[str, Any]:
    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < SEGMENT_COUNT:
        raise ValueError(f"segment index must be an integer in [0, {SEGMENT_COUNT - 1}]")
    if len(rows) != SEGMENT_ROWS:
        raise ValueError(f"segment {index} has {len(rows)} rows; expected {SEGMENT_ROWS}")
    offset = index * SEGMENT_ROWS
    payload = b"".join(row.raw_line for row in rows)
    ids = parent._validated_ids(
        rows,
        path=Path(f"<RMCT-256 convergence segment {index}>"),
        label=f"RMCT-256 convergence segment {index}",
        expected_rows=SEGMENT_ROWS,
        expected_counts=SEGMENT_COUNTS,
        require_canonical_pair=True,
    )
    return {
        "index": index,
        "row_offset": offset,
        "row_count": SEGMENT_ROWS,
        "source_rows_1_based_inclusive": [offset + 1, offset + SEGMENT_ROWS],
        "byte_count": len(payload),
        "content_sha256": _sha256(payload),
        "counts_by_dataset": dict(SEGMENT_COUNTS),
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
        "selection_method": "exact_ordered_parent_rows_without_shuffle_or_reserialization",
    }


def _parent_record(
    *,
    parent_selection_path: Path,
    parent_manifest_path: Path,
    parent_document: Mapping[str, Any],
    parent_ids: Sequence[str],
) -> dict[str, Any]:
    return {
        "selection": _portable_file_record(parent_selection_path),
        "selection_manifest": _portable_file_record(parent_manifest_path),
        "selection_manifest_document_sha256": _sha256(_canonical_json(parent_document)),
        "row_count": PARENT_ROWS,
        "question_ids_sha256": _ids_sha256(parent_ids),
        "source_rows_1_based_inclusive": [1, PARENT_ROWS],
    }


def _build_manifest(
    *,
    parent_selection_path: Path,
    parent_manifest_path: Path,
    parent_document: Mapping[str, Any],
    parent_rows: Sequence[parent._RawRow],
    parent_payload: bytes,
    parent_ids: Sequence[str],
) -> dict[str, Any]:
    if len(parent_rows) != PARENT_ROWS:
        raise ValueError(f"parent RMCT-256 selection has {len(parent_rows)} rows; expected {PARENT_ROWS}")
    segments = [
        _segment_entry(index, parent_rows[index * SEGMENT_ROWS : (index + 1) * SEGMENT_ROWS])
        for index in range(SEGMENT_COUNT)
    ]
    joined_payload = b"".join(
        b"".join(row.raw_line for row in parent_rows[index * SEGMENT_ROWS : (index + 1) * SEGMENT_ROWS])
        for index in range(SEGMENT_COUNT)
    )
    flattened_ids = [question_id for segment in segments for question_id in segment["question_ids"]]
    if joined_payload != parent_payload:
        raise ValueError("RMCT-256 convergence segments do not byte-reconstruct the parent selection")
    if flattened_ids != list(parent_ids):
        raise ValueError("RMCT-256 convergence segments do not preserve the parent QID order")
    if len(flattened_ids) != len(set(flattened_ids)):
        raise ValueError("RMCT-256 convergence segments have duplicate question IDs")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "model": parent.MODEL,
        "parent_rmct256": _parent_record(
            parent_selection_path=parent_selection_path,
            parent_manifest_path=parent_manifest_path,
            parent_document=parent_document,
            parent_ids=parent_ids,
        ),
        "segments": segments,
        "assertions": {
            "parent_rmct256_is_fully_verified": True,
            "segments_are_exact_ordered_parent_slices": True,
            "segments_cover_parent_rows_1_to_256_exactly_once": True,
            "segments_reconstruct_parent_ordered_question_ids": True,
            "segment_question_ids_are_globally_unique": True,
            "segments_are_pairwise_question_id_disjoint": True,
            "each_segment_has_32_logiqa_and_32_hellaswag": True,
        },
    }


def materialize_rmct256_convergence_segments(
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
) -> MaterializedConvergenceSegments:
    """Publish a manifest after fully proving the attested RMCT-256 parent."""

    parent_path, parent_manifest_path, parent_document, rows, payload, ids = _validate_parent_selection(
        parent_selection=parent_selection,
        parent_selection_manifest=parent_selection_manifest,
        verify_parent_sources=True,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
    )
    document = _build_manifest(
        parent_selection_path=parent_path,
        parent_manifest_path=parent_manifest_path,
        parent_document=parent_document,
        parent_rows=rows,
        parent_payload=payload,
        parent_ids=ids,
    )
    manifest_payload = _canonical_json(document)
    manifest_sha256 = _sha256(manifest_payload)
    manifest_path = Path(output_dir).resolve() / f"{MANIFEST_FILENAME_PREFIX}{manifest_sha256}.json"
    status = _publish_immutable(manifest_path, manifest_payload)
    verify_rmct256_convergence_segments_manifest(
        manifest_path,
        parent_selection=parent_selection,
        parent_selection_manifest=parent_selection_manifest,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
        verify_parent_sources=True,
        expected_manifest_sha256=manifest_sha256,
    )
    return MaterializedConvergenceSegments(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        manifest_status=status,
    )


def verify_rmct256_convergence_segments_manifest(
    manifest_path: str | Path,
    *,
    parent_selection: str | Path,
    parent_selection_manifest: str | Path,
    canonical_source: str | Path | None = None,
    canonical_source_manifest: str | Path | None = None,
    recovered_none_source: str | Path | None = None,
    recovered_none_manifest: str | Path | None = None,
    original64_reference_manifest: str | Path | None = None,
    stage2_manifest: str | Path | None = None,
    verify_parent_sources: bool = True,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Fail closed on a content-addressed RMCT-256 convergence manifest.

    A protected launch must retain ``verify_parent_sources=True``.  The
    explicit false mode exists only for runtime transport binding after an
    earlier target attestation has already proved the immutable source chain.
    """

    expected_digest = _validate_expected_digest(expected_manifest_sha256)
    path, document, payload = _read_document(manifest_path, label="RMCT-256 convergence segment manifest")
    if _canonical_json(document) != payload:
        raise ValueError("RMCT-256 convergence segment manifest is not canonical JSON bytes")
    digest = _sha256(payload)
    expected_name = f"{MANIFEST_FILENAME_PREFIX}{digest}.json"
    if path.name != expected_name:
        raise ValueError(f"RMCT-256 convergence segment manifest filename must be content-addressed {expected_name}")
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            "RMCT-256 convergence segment manifest digest differs from the expected launch contract: "
            f"expected {expected_digest}, got {digest}"
        )
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != MANIFEST_KIND:
        raise ValueError("RMCT-256 convergence segment manifest has an unsupported schema/kind")
    if document.get("model") != parent.MODEL:
        raise ValueError("RMCT-256 convergence segment manifest model differs from the frozen Qwen3.5 contract")

    parent_path, parent_manifest_path, parent_document, rows, parent_payload, ids = _validate_parent_selection(
        parent_selection=parent_selection,
        parent_selection_manifest=parent_selection_manifest,
        verify_parent_sources=verify_parent_sources,
        canonical_source=canonical_source,
        canonical_source_manifest=canonical_source_manifest,
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        original64_reference_manifest=original64_reference_manifest,
        stage2_manifest=stage2_manifest,
    )
    expected_document = _build_manifest(
        parent_selection_path=parent_path,
        parent_manifest_path=parent_manifest_path,
        parent_document=parent_document,
        parent_rows=rows,
        parent_payload=parent_payload,
        parent_ids=ids,
    )
    if document != expected_document:
        raise ValueError("RMCT-256 convergence segment manifest differs from the verified parent selection")
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
            result = materialize_rmct256_convergence_segments(output_dir=args.output_dir, **common)
            print(
                json.dumps(
                    {
                        "manifest_path": str(result.manifest_path),
                        "manifest_sha256": result.manifest_sha256,
                        "manifest_status": result.manifest_status,
                    },
                    sort_keys=True,
                )
            )
        else:
            document = verify_rmct256_convergence_segments_manifest(args.manifest, **common)
            print(
                json.dumps(
                    {
                        "manifest": str(args.manifest.resolve()),
                        "parent_selection_sha256": document["parent_rmct256"]["selection"]["content_sha256"],
                        "segment_count": len(document["segments"]),
                        "verified": True,
                    },
                    sort_keys=True,
                )
            )
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "MANIFEST_FILENAME_PREFIX",
    "MANIFEST_KIND",
    "MaterializedConvergenceSegments",
    "PARENT_ROWS",
    "SCHEMA_VERSION",
    "SEGMENT_COUNT",
    "SEGMENT_COUNTS",
    "SEGMENT_ROWS",
    "materialize_rmct256_convergence_segments",
    "verify_rmct256_convergence_segments_manifest",
]
