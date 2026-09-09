"""Materialize immutable local-only inputs for the Stage 2 OOD-HLE matrix.

The resulting manifest has four disjoint *regimes* by dataset/bias status:

* ``iid``: held-out in-domain LogiQA/HellaSwag under ``wrong_argument``;
* ``heldout_dataset``: canonical HLE under ``wrong_argument``;
* ``heldout_bias``: held-out in-domain prompts under five non-training biases;
* ``heldout_dataset_and_bias``: canonical HLE under those five biases.

This module deliberately has no model backend, generation, grading, network,
or mutation path.  It copies the seven canonical HLE files byte-for-byte,
keeps the Stage 1 wrong-argument source byte-for-byte, and writes any derived
in-domain held-out-bias files only into a fresh output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.rmct_tbsr.constants import (
    HLE_BIASES,
    HLE_DATASET,
    HLE_FILE_SHA256,
    HLE_FILES,
    HLE_PAPER_HELD_OUT_BIASES,
    HLE_ROWS,
)
from experiments.stage1_iid_diagnostic_none import prepare as iid_prepare
from experiments.stage2_ood_hle.prepare import HELDOUT_BIASES, MANIFEST_SCHEMA, TRAINING_BIAS


SCHEMA_VERSION = 1
MANIFEST_KIND = "stage2_ood_hle_2x2_manifest"
PROMPT_STYLE = "none"
IN_DOMAIN_DATASETS = iid_prepare.DATASETS
IN_DOMAIN_ROWS = iid_prepare.HELDOUT_ROWS
IN_DOMAIN_COUNTS = iid_prepare.HELDOUT_COUNTS

IID = "iid"
HELDOUT_DATASET = "heldout_dataset"
HELDOUT_BIAS = "heldout_bias"
HELDOUT_DATASET_AND_BIAS = "heldout_dataset_and_bias"
REGIMES = (IID, HELDOUT_DATASET, HELDOUT_BIAS, HELDOUT_DATASET_AND_BIAS)

if set(HELDOUT_BIASES) != set(HLE_PAPER_HELD_OUT_BIASES) or len(HELDOUT_BIASES) != len(
    HLE_PAPER_HELD_OUT_BIASES
):  # pragma: no cover - import-time contract guard
    raise RuntimeError("Stage 2 held-out bias set must match the canonical HLE bias set")

# Pin the upstream frozen HLE export as well as the seven rendered prompt
# files.  This rules out silently substituting another HLE revision that merely
# has the same schema and sample count.
HLE_SOURCE_MANIFEST_FILENAME = "hle-text-mc.manifest.json"
HLE_SOURCE_EXPORT_FILENAME = "hle-text-mc.jsonl"
HLE_SOURCE_MANIFEST_SHA256 = "52eb736c10b71265ce623b8315dbd698464f4ebf9ab13266c06d6f761d9c424d"
HLE_SOURCE_EXPORT_SHA256 = "58f34d361b8fb9ae9870b9d65ea22aba70c6df035c6cd1571f42820e2cb0e43b"
HLE_SOURCE_EXPORT_ROWS = 513
HLE_SOURCE_DATASET_REPOSITORY = "cais/hle"
HLE_SOURCE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
HLE_SOURCE_SPLIT = "test"
HLE_SOURCE_SELECTION = {"answer_type": "multipleChoice", "image": False}
_HLE_SOURCE_OUTPUT_PATH = (
    "/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-full-20260725/"
    "artifacts/rmct-hle-gpt-oss-20b/data/hle-text-mc.jsonl"
)

_CLEAN_FIELDS = (
    "question",
    "question_id",
    "source_dataset",
    "prompt_style",
    "unbiased_messages",
    "ground_truth",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_sha256(ids: Iterable[str]) -> str:
    return _sha256("".join(f"{question_id}\n" for question_id in ids).encode("utf-8"))


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON document: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON document must contain an object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSONL does not exist: {path}")
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank rows are not permitted in {label}")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid {label} JSONL row") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: {label} row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: {label} must not be empty")
    return rows, payload


def _require_string(row: Mapping[str, Any], field: str, *, path: Path, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}:{line_number}: {field} must be a non-empty string")
    return value


def _validate_messages(value: object, *, path: Path, line_number: int, field: str) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}:{line_number}: {field} must be a non-empty message list")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise ValueError(f"{path}:{line_number}: {field}[{index}] must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip() or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{path}:{line_number}: {field}[{index}] must contain non-empty role/content")
        messages.append({"role": role, "content": content})
    return messages


def _validate_clean_row(
    row: Mapping[str, Any], *, path: Path, line_number: int, allowed_datasets: set[str]
) -> None:
    for field in ("question", "question_id", "source_dataset", "prompt_style", "ground_truth"):
        _require_string(row, field, path=path, line_number=line_number)
    if row["source_dataset"] not in allowed_datasets:
        raise ValueError(f"{path}:{line_number}: invalid source_dataset {row['source_dataset']!r}")
    if row["prompt_style"] != PROMPT_STYLE:
        raise ValueError(f"{path}:{line_number}: prompt_style must be {PROMPT_STYLE!r}")
    _validate_messages(row.get("unbiased_messages"), path=path, line_number=line_number, field="unbiased_messages")


def _validate_biased_row(
    row: Mapping[str, Any],
    *,
    path: Path,
    line_number: int,
    allowed_datasets: set[str],
    bias_type: str | None = None,
) -> None:
    _validate_clean_row(row, path=path, line_number=line_number, allowed_datasets=allowed_datasets)
    for field in ("bias_type", "biased_option", "biasing_text"):
        _require_string(row, field, path=path, line_number=line_number)
    if bias_type is not None and row["bias_type"] != bias_type:
        raise ValueError(f"{path}:{line_number}: bias_type must be {bias_type!r}")
    if row["biased_option"] == row["ground_truth"]:
        raise ValueError(f"{path}:{line_number}: biased_option must be a distractor")
    _validate_messages(row.get("biased_messages"), path=path, line_number=line_number, field="biased_messages")


def _rows_by_id(rows: Sequence[Mapping[str, Any]], *, path: Path, label: str) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        question_id = _require_string(row, "question_id", path=path, line_number=line_number)
        if question_id in by_id:
            raise ValueError(f"{path}: duplicate question_id {question_id!r} in {label}")
        by_id[question_id] = row
    return by_id


def _clean_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in _CLEAN_FIELDS}


def _source_identity(path: Path, payload: bytes, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "question_ids_sha256": _ids_sha256(str(row["question_id"]) for row in rows),
    }


def _artifact_entry(*, final_path: Path, rows: Sequence[Mapping[str, Any]], payload: bytes, source: Mapping[str, Any]) -> dict[str, Any]:
    ids = [str(row["question_id"]) for row in rows]
    counts = Counter(str(row["source_dataset"]) for row in rows)
    return {
        "path": str(final_path.resolve()),
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
        "row_count": len(rows),
        "counts_by_dataset": dict(sorted(counts.items())),
        "question_ids": ids,
        "question_ids_sha256": _ids_sha256(ids),
        "source": dict(source),
    }


def _derive_in_domain_suite(
    rows: Sequence[Mapping[str, Any]], *, source_path: Path
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Derive unseen-bias variants from exact frozen held-out IID rows."""

    try:
        from mcq_bias.pipeline.injectors import default_injectors
        from mcq_bias.pipeline.records import instruction_suffix, parse_record_from_text
    except ImportError as exc:  # pragma: no cover - exercised in configured runtime
        raise RuntimeError("mcq_bias is required for Stage 2 local materialization") from exc

    suffix = instruction_suffix(PROMPT_STYLE)
    records: dict[str, Any] = {}
    for line_number, row in enumerate(rows, start=1):
        _validate_biased_row(
            row,
            path=source_path,
            line_number=line_number,
            allowed_datasets=set(IN_DOMAIN_DATASETS),
            bias_type=TRAINING_BIAS,
        )
        messages = _validate_messages(
            row["unbiased_messages"], path=source_path, line_number=line_number, field="unbiased_messages"
        )
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"{source_path}:{line_number}: expected exactly one clean user message")
        content = messages[0]["content"]
        if not content.endswith(suffix):
            raise ValueError(f"{source_path}:{line_number}: clean message lacks the no-CoT answer-format suffix")
        try:
            record = parse_record_from_text(
                content[: -len(suffix)], str(row["ground_truth"]), str(row["source_dataset"])
            )
        except ValueError as exc:
            raise ValueError(f"{source_path}:{line_number}: cannot reconstruct canonical MCQ record") from exc
        if record.question_id != row["question_id"]:
            raise ValueError(f"{source_path}:{line_number}: reconstructed question_id differs from frozen row")
        if record.parsed_input() != row["question"]:
            raise ValueError(f"{source_path}:{line_number}: reconstructed question differs from frozen row")
        if record.ground_truth != row["ground_truth"]:
            raise ValueError(f"{source_path}:{line_number}: reconstructed ground truth differs from frozen row")
        if record.biased_option != row["biased_option"]:
            raise ValueError(f"{source_path}:{line_number}: reconstructed biased target differs from frozen row")
        records[record.question_id] = record

    injectors = default_injectors(list(records.values()))
    variants: dict[str, list[dict[str, Any]]] = {TRAINING_BIAS: [dict(row) for row in rows]}
    for bias_type in HELDOUT_BIASES:
        injector = injectors.get(bias_type)
        if injector is None:  # pragma: no cover - fixed mcq_bias contract
            raise RuntimeError(f"mcq_bias has no injector for {bias_type!r}")
        derived: list[dict[str, Any]] = []
        for line_number, source in enumerate(rows, start=1):
            result = injector.inject(records[str(source["question_id"])], PROMPT_STYLE)
            if result is None:
                raise ValueError(f"{source_path}:{line_number}: {bias_type} could not be injected")
            if result.biased_option != source["biased_option"]:
                raise ValueError(f"{source_path}:{line_number}: {bias_type} changed the frozen biased target option")
            row = {
                **_clean_projection(source),
                "bias_type": bias_type,
                "biased_messages": result.messages,
                "biased_option": result.biased_option,
                "biasing_text": result.biasing_text,
            }
            _validate_biased_row(
                row,
                path=source_path,
                line_number=line_number,
                allowed_datasets=set(IN_DOMAIN_DATASETS),
                bias_type=bias_type,
            )
            derived.append(row)
        variants[bias_type] = derived
    return [_clean_projection(row) for row in rows], variants


def _load_hle_source_provenance(hle_dir: Path) -> dict[str, Any]:
    """Validate the frozen source export and record its exact provenance."""

    root = hle_dir.parent
    manifest_path = root / HLE_SOURCE_MANIFEST_FILENAME
    export_path = root / HLE_SOURCE_EXPORT_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen HLE source manifest does not exist: {manifest_path}")
    if not export_path.is_file():
        raise FileNotFoundError(f"frozen HLE source export does not exist: {export_path}")
    manifest_payload = manifest_path.read_bytes()
    if _sha256(manifest_payload) != HLE_SOURCE_MANIFEST_SHA256:
        raise ValueError(f"{manifest_path}: SHA-256 mismatch for frozen HLE source manifest")
    document = _read_json(manifest_path, label="frozen HLE source manifest")
    expected = {
        "kind": "hle_text_multiple_choice_export",
        "output": {"content_sha256": HLE_SOURCE_EXPORT_SHA256, "path": _HLE_SOURCE_OUTPUT_PATH},
        "row_count": HLE_SOURCE_EXPORT_ROWS,
        "schema_version": 1,
        "selection": HLE_SOURCE_SELECTION,
        "source": {
            "dataset": HLE_SOURCE_DATASET_REPOSITORY,
            "revision": HLE_SOURCE_REVISION,
            "split": HLE_SOURCE_SPLIT,
        },
        "written_at": "2026-07-25T19:18:51.853748+00:00",
    }
    if document != expected:
        raise ValueError(f"{manifest_path}: frozen HLE source provenance is not the pinned canonical export")
    export_rows, export_payload = _read_jsonl(export_path, label="frozen HLE source export")
    if _sha256(export_payload) != HLE_SOURCE_EXPORT_SHA256:
        raise ValueError(f"{export_path}: SHA-256 mismatch for frozen HLE source export")
    if len(export_rows) != HLE_SOURCE_EXPORT_ROWS:
        raise ValueError(f"{export_path}: frozen HLE source export must have {HLE_SOURCE_EXPORT_ROWS} rows")
    return {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "content_sha256": _sha256(manifest_payload),
            "byte_count": len(manifest_payload),
            "kind": document["kind"],
            "schema_version": document["schema_version"],
        },
        "export": {
            "path": str(export_path.resolve()),
            "content_sha256": _sha256(export_payload),
            "byte_count": len(export_payload),
            "row_count": len(export_rows),
        },
        "source": dict(document["source"]),
        "selection": dict(document["selection"]),
    }


def _load_canonical_hle_suite(hle_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Load the exact seven-file canonical HLE prompt suite."""

    suite: dict[str, list[dict[str, Any]]] = {}
    source_entries: dict[str, dict[str, Any]] = {}
    for name, filename in HLE_FILES.items():
        path = (hle_dir / filename).resolve()
        rows, payload = _read_jsonl(path, label=f"canonical HLE {name}")
        if _sha256(payload) != HLE_FILE_SHA256[name]:
            raise ValueError(f"{path}: SHA-256 mismatch for canonical HLE {name!r}")
        if len(rows) != HLE_ROWS:
            raise ValueError(f"{path}: canonical HLE {name!r} must have exactly {HLE_ROWS} rows")
        for line_number, row in enumerate(rows, start=1):
            if name == "unbiased":
                _validate_clean_row(row, path=path, line_number=line_number, allowed_datasets={HLE_DATASET})
            else:
                _validate_biased_row(
                    row, path=path, line_number=line_number, allowed_datasets={HLE_DATASET}, bias_type=name
                )
        suite[name] = rows
        source_entries[name] = _source_identity(path, payload, rows)

    clean_by_id = _rows_by_id(suite["unbiased"], path=hle_dir / HLE_FILES["unbiased"], label="HLE clean")
    for bias_type in HLE_BIASES:
        path = hle_dir / HLE_FILES[bias_type]
        by_id = _rows_by_id(suite[bias_type], path=path, label=f"HLE {bias_type}")
        if set(by_id) != set(clean_by_id):
            raise ValueError(f"{path}: HLE IDs do not exactly match the clean file")
        for question_id, row in by_id.items():
            clean = clean_by_id[question_id]
            for field in _CLEAN_FIELDS:
                if row[field] != clean[field]:
                    raise ValueError(f"{path}: {field} differs from clean HLE row {question_id!r}")
    reference = _rows_by_id(
        suite[TRAINING_BIAS], path=hle_dir / HLE_FILES[TRAINING_BIAS], label="HLE training-bias"
    )
    for bias_type in HELDOUT_BIASES:
        for question_id, row in _rows_by_id(
            suite[bias_type], path=hle_dir / HLE_FILES[bias_type], label=f"HLE {bias_type}"
        ).items():
            if row["biased_option"] != reference[question_id]["biased_option"]:
                raise ValueError(f"{hle_dir / HLE_FILES[bias_type]}: biased target differs for {question_id!r}")
    return suite, source_entries


def _iid_input(iid_manifest: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], bytes]:
    """Verify and load precisely the non-overlapping Stage 1 held-out rows."""

    if not iid_manifest.is_file():
        raise FileNotFoundError(f"Stage 1 IID manifest does not exist: {iid_manifest}")
    document = iid_prepare.validate_manifest(iid_manifest, verify_source=True)
    splits = document.get("splits")
    if not isinstance(splits, Mapping):  # pragma: no cover - validated upstream
        raise ValueError("Stage 1 IID manifest has no splits object")
    entry = splits.get("heldout_in_domain")
    if not isinstance(entry, Mapping):  # pragma: no cover - validated upstream
        raise ValueError("Stage 1 IID manifest has no heldout_in_domain entry")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Stage 1 heldout_in_domain entry has no path")
    source_path = Path(raw_path).resolve()
    rows, payload = _read_jsonl(source_path, label="Stage 1 held-out in-domain")
    if _sha256(payload) != entry.get("content_sha256"):
        raise ValueError(f"{source_path}: differs from Stage 1 IID manifest")
    if len(rows) != IN_DOMAIN_ROWS or entry.get("row_count") != IN_DOMAIN_ROWS:
        raise ValueError(f"{source_path}: expected exactly {IN_DOMAIN_ROWS} held-out in-domain rows")
    counts = Counter(str(row.get("source_dataset")) for row in rows)
    if dict(counts) != dict(IN_DOMAIN_COUNTS) or entry.get("counts_by_dataset") != dict(IN_DOMAIN_COUNTS):
        raise ValueError(f"{source_path}: held-out in-domain dataset counts are invalid")
    ids = [row.get("question_id") for row in rows]
    if ids != entry.get("question_ids") or len(ids) != len(set(ids)):
        raise ValueError(f"{source_path}: IDs differ from Stage 1 IID manifest")
    return document, dict(entry), rows, payload


def _archive_staging_directory(staging: Path) -> None:
    """Preserve failed partial output rather than recursively deleting it."""

    if not staging.exists():
        return
    archive = staging.parent / "_archive"
    archive.mkdir(parents=True, exist_ok=True)
    staging.replace(archive / f"{staging.name}-{uuid.uuid4().hex[:12]}")


def _write_payload(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_ood_hle_2x2(
    *, iid_manifest: str | Path, hle_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Publish one immutable Stage 2 2×2 input suite in a fresh directory."""

    iid_manifest_path = Path(iid_manifest).resolve()
    hle_root = Path(hle_dir).resolve()
    output_path = Path(output_dir).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing Stage 2 OOD artifact: {output_path}")
    if not hle_root.is_dir():
        raise FileNotFoundError(f"canonical HLE directory does not exist: {hle_root}")

    iid_document, iid_entry, in_domain_rows, in_domain_payload = _iid_input(iid_manifest_path)
    in_domain_clean, in_domain_variants = _derive_in_domain_suite(
        in_domain_rows, source_path=Path(iid_entry["path"])
    )
    hle_provenance = _load_hle_source_provenance(hle_root)
    hle_suite, hle_sources = _load_canonical_hle_suite(hle_root)
    iid_source = _source_identity(Path(iid_entry["path"]), in_domain_payload, in_domain_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.stage-", dir=output_path.parent))
    try:
        in_entries: dict[str, dict[str, Any]] = {}
        hle_entries: dict[str, dict[str, Any]] = {}

        clean_payload = _jsonl_payload(in_domain_clean)
        _write_payload(staging / "in_domain" / "clean.jsonl", clean_payload)
        in_entries["unbiased"] = _artifact_entry(
            final_path=output_path / "in_domain" / "clean.jsonl",
            rows=in_domain_clean,
            payload=clean_payload,
            source={
                "derivation": "exact_frozen_heldout_in_domain_clean_projection",
                "source_iid_heldout": iid_source,
            },
        )
        for bias_type in (TRAINING_BIAS, *HELDOUT_BIASES):
            final_path = output_path / "in_domain" / f"{bias_type}.jsonl"
            if bias_type == TRAINING_BIAS:
                rows, payload, source = in_domain_rows, in_domain_payload, iid_source
            else:
                rows = in_domain_variants[bias_type]
                payload = _jsonl_payload(rows)
                source = {
                    "derivation": "exact_frozen_heldout_in_domain_clean_messages_plus_pinned_mcq_bias_injector",
                    "source_iid_heldout": iid_source,
                    "bias_type": bias_type,
                }
            _write_payload(staging / "in_domain" / f"{bias_type}.jsonl", payload)
            in_entries[bias_type] = _artifact_entry(
                final_path=final_path, rows=rows, payload=payload, source=source
            )

        for name in ("unbiased", *HLE_BIASES):
            source_path = hle_root / HLE_FILES[name]
            payload = source_path.read_bytes()
            _write_payload(staging / "hle" / f"{name}.jsonl", payload)
            hle_entries[name] = _artifact_entry(
                final_path=output_path / "hle" / f"{name}.jsonl",
                rows=hle_suite[name],
                payload=payload,
                source=hle_sources[name],
            )

        manifest = {
            # Compatibility contract consumed by the offline analyzer.  The
            # richer schema below remains the authoritative materialization
            # contract used by the task factory.
            "schema": MANIFEST_SCHEMA,
            "training_bias": TRAINING_BIAS,
            "held_out_biases": list(HELDOUT_BIASES),
            "schema_version": SCHEMA_VERSION,
            "kind": MANIFEST_KIND,
            "training_contract": {
                "datasets": list(IN_DOMAIN_DATASETS),
                "bias_type": TRAINING_BIAS,
                "prompt_style": PROMPT_STYLE,
                "headline_iid_split": "heldout_in_domain",
                "excluded_stage1_split": "train_eval",
                "excluded_stage1_split_reason": "overlaps the SFT population; retain only as a training sanity check",
            },
            "stage1_iid_source": {
                "manifest": {
                    "path": str(iid_manifest_path),
                    "content_sha256": _sha256(iid_manifest_path.read_bytes()),
                    "kind": iid_document["kind"],
                    "schema_version": iid_document["schema_version"],
                },
                "heldout_in_domain": iid_source,
            },
            "canonical_hle_source": hle_provenance,
            "regime_order": list(REGIMES),
            "populations": {
                "in_domain": {
                    "dataset_status": "seen",
                    "row_count": IN_DOMAIN_ROWS,
                    "counts_by_dataset": dict(IN_DOMAIN_COUNTS),
                    "artifacts": in_entries,
                },
                "hle": {
                    "dataset_status": "held_out",
                    "source_dataset": HLE_DATASET,
                    "row_count": HLE_ROWS,
                    "source_files": hle_sources,
                    "artifacts": hle_entries,
                },
            },
            "regimes": {
                IID: {
                    "dataset_status": "seen",
                    "bias_status": "seen",
                    "population": "in_domain",
                    "clean_artifact": "unbiased",
                    "bias_artifacts": [TRAINING_BIAS],
                    "nominal_biased_samples": IN_DOMAIN_ROWS,
                },
                HELDOUT_DATASET: {
                    "dataset_status": "held_out",
                    "bias_status": "seen",
                    "population": "hle",
                    "clean_artifact": "unbiased",
                    "bias_artifacts": [TRAINING_BIAS],
                    "nominal_biased_samples": HLE_ROWS,
                },
                HELDOUT_BIAS: {
                    "dataset_status": "seen",
                    "bias_status": "held_out",
                    "population": "in_domain",
                    "clean_artifact": "unbiased",
                    "bias_artifacts": list(HELDOUT_BIASES),
                    "nominal_biased_samples": IN_DOMAIN_ROWS * len(HELDOUT_BIASES),
                },
                HELDOUT_DATASET_AND_BIAS: {
                    "dataset_status": "held_out",
                    "bias_status": "held_out",
                    "population": "hle",
                    "clean_artifact": "unbiased",
                    "bias_artifacts": list(HELDOUT_BIASES),
                    "nominal_biased_samples": HLE_ROWS * len(HELDOUT_BIASES),
                },
            },
            "assertions": {
                "all_prompt_styles_none": True,
                "in_domain_exactly_stage1_heldout_population": True,
                "in_domain_targets_shared_across_all_six_biases": True,
                "hle_targets_shared_across_all_six_biases": True,
                "all_bias_files_match_their_population_clean_ids": True,
                "headline_uses_no_training_overlap": True,
                "no_model_or_remote_service_called": True,
            },
        }
        _write_payload(
            staging / "manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        if output_path.exists():  # race-safe no-overwrite re-check
            raise FileExistsError(f"refusing to overwrite existing Stage 2 OOD artifact: {output_path}")
        staging.replace(output_path)
    except BaseException:
        _archive_staging_directory(staging)
        raise

    validate_manifest(output_path / "manifest.json")
    return manifest


def _validate_artifact_entry(
    entry: Mapping[str, Any],
    *,
    label: str,
    expected_rows: int,
    expected_datasets: set[str],
    bias_type: str | None,
) -> tuple[Path, list[dict[str, Any]]]:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}: artifact has no path")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"{label}: artifact does not exist: {path}")
    payload = path.read_bytes()
    if _sha256(payload) != entry.get("content_sha256") or len(payload) != entry.get("byte_count"):
        raise ValueError(f"{label}: artifact bytes differ from its manifest")
    rows, parsed_payload = _read_jsonl(path, label=label)
    if parsed_payload != payload:  # pragma: no cover - protects a read race on mutable storage
        raise ValueError(f"{label}: artifact bytes changed while being validated")
    if len(rows) != expected_rows or entry.get("row_count") != expected_rows:
        raise ValueError(f"{label}: unexpected row count")
    ids = [row.get("question_id") for row in rows]
    if entry.get("question_ids") != ids or entry.get("question_ids_sha256") != _ids_sha256(str(x) for x in ids):
        raise ValueError(f"{label}: question-ID identity differs from its manifest")
    counts = Counter(str(row.get("source_dataset")) for row in rows)
    if entry.get("counts_by_dataset") != dict(sorted(counts.items())) or set(counts) != expected_datasets:
        raise ValueError(f"{label}: dataset counts differ from its manifest")
    for line_number, row in enumerate(rows, start=1):
        if bias_type is None:
            _validate_clean_row(row, path=path, line_number=line_number, allowed_datasets=expected_datasets)
        else:
            _validate_biased_row(
                row,
                path=path,
                line_number=line_number,
                allowed_datasets=expected_datasets,
                bias_type=bias_type,
            )
    _rows_by_id(rows, path=path, label=label)
    return path, rows


def _validate_hle_provenance(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {"manifest", "export", "source", "selection"}:
        raise ValueError("Stage 2 canonical HLE source provenance is invalid")
    manifest = value["manifest"]
    export = value["export"]
    if not isinstance(manifest, Mapping) or not isinstance(export, Mapping):
        raise ValueError("Stage 2 canonical HLE source provenance is invalid")
    if set(manifest) != {"path", "content_sha256", "byte_count", "kind", "schema_version"}:
        raise ValueError("Stage 2 canonical HLE source manifest identity is invalid")
    if set(export) != {"path", "content_sha256", "byte_count", "row_count"}:
        raise ValueError("Stage 2 canonical HLE source export identity is invalid")
    if (
        not isinstance(manifest["path"], str)
        or not manifest["path"]
        or manifest["content_sha256"] != HLE_SOURCE_MANIFEST_SHA256
        or manifest["kind"] != "hle_text_multiple_choice_export"
        or manifest["schema_version"] != 1
        or not isinstance(manifest["byte_count"], int)
        or manifest["byte_count"] <= 0
    ):
        raise ValueError("Stage 2 canonical HLE source manifest does not match the pinned export")
    if (
        not isinstance(export["path"], str)
        or not export["path"]
        or export["content_sha256"] != HLE_SOURCE_EXPORT_SHA256
        or export["row_count"] != HLE_SOURCE_EXPORT_ROWS
        or not isinstance(export["byte_count"], int)
        or export["byte_count"] <= 0
        or value["source"]
        != {
            "dataset": HLE_SOURCE_DATASET_REPOSITORY,
            "revision": HLE_SOURCE_REVISION,
            "split": HLE_SOURCE_SPLIT,
        }
        or value["selection"] != HLE_SOURCE_SELECTION
    ):
        raise ValueError("Stage 2 canonical HLE source export does not match the pinned revision")


def _validate_source_identity(value: object, *, label: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}: source identity must be an object")
    required = {"path", "content_sha256", "byte_count", "row_count", "question_ids_sha256"}
    if set(value) != required:
        raise ValueError(f"{label}: source identity fields are invalid")
    if not isinstance(value["path"], str) or not value["path"]:
        raise ValueError(f"{label}: source identity path is invalid")
    for field in ("content_sha256", "question_ids_sha256"):
        if not isinstance(value[field], str) or len(value[field]) != 64:
            raise ValueError(f"{label}: source identity {field} is invalid")
    for field in ("byte_count", "row_count"):
        if not isinstance(value[field], int) or value[field] <= 0:
            raise ValueError(f"{label}: source identity {field} is invalid")
    for field in ("content_sha256", "byte_count", "row_count", "question_ids_sha256"):
        if value[field] != artifact.get(field):
            raise ValueError(f"{label}: source identity differs from the frozen wrong-argument artifact")
    return dict(value)


def validate_manifest(manifest: str | Path) -> dict[str, Any]:
    """Validate an already-published suite using only local frozen artifacts."""

    path = Path(manifest).resolve()
    document = _read_json(path, label="Stage 2 OOD manifest")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("kind") != MANIFEST_KIND:
        raise ValueError("unsupported Stage 2 OOD-HLE manifest schema")
    if (
        document.get("schema") != MANIFEST_SCHEMA
        or document.get("training_bias") != TRAINING_BIAS
        or document.get("held_out_biases") != list(HELDOUT_BIASES)
    ):
        raise ValueError("Stage 2 OOD manifest bias-contract compatibility fields are invalid")
    expected_training = {
        "datasets": list(IN_DOMAIN_DATASETS),
        "bias_type": TRAINING_BIAS,
        "prompt_style": PROMPT_STYLE,
        "headline_iid_split": "heldout_in_domain",
        "excluded_stage1_split": "train_eval",
        "excluded_stage1_split_reason": "overlaps the SFT population; retain only as a training sanity check",
    }
    if document.get("training_contract") != expected_training:
        raise ValueError("Stage 2 OOD manifest training contract is invalid")
    populations = document.get("populations")
    regimes = document.get("regimes")
    if not isinstance(populations, Mapping) or set(populations) != {"in_domain", "hle"}:
        raise ValueError("Stage 2 OOD manifest must contain in_domain and hle populations")
    if not isinstance(regimes, Mapping) or set(regimes) != set(REGIMES):
        raise ValueError("Stage 2 OOD manifest has an invalid regime set")
    if document.get("regime_order") != list(REGIMES):
        raise ValueError("Stage 2 OOD manifest has an invalid display regime order")
    in_domain = populations["in_domain"]
    hle = populations["hle"]
    if not isinstance(in_domain, Mapping) or not isinstance(hle, Mapping):
        raise ValueError("Stage 2 OOD populations must be objects")
    if in_domain.get("dataset_status") != "seen" or in_domain.get("row_count") != IN_DOMAIN_ROWS:
        raise ValueError("Stage 2 in-domain population contract is invalid")
    if in_domain.get("counts_by_dataset") != dict(IN_DOMAIN_COUNTS):
        raise ValueError("Stage 2 in-domain dataset counts are invalid")
    if hle.get("dataset_status") != "held_out" or hle.get("source_dataset") != HLE_DATASET or hle.get("row_count") != HLE_ROWS:
        raise ValueError("Stage 2 HLE population contract is invalid")
    in_artifacts = in_domain.get("artifacts")
    hle_artifacts = hle.get("artifacts")
    expected_names = {"unbiased", *HLE_BIASES}
    if not isinstance(in_artifacts, Mapping) or set(in_artifacts) != expected_names:
        raise ValueError("Stage 2 in-domain artifact set is invalid")
    if not isinstance(hle_artifacts, Mapping) or set(hle_artifacts) != expected_names:
        raise ValueError("Stage 2 HLE artifact set is invalid")
    _validate_hle_provenance(document.get("canonical_hle_source"))

    source_files = hle.get("source_files")
    if not isinstance(source_files, Mapping) or set(source_files) != expected_names:
        raise ValueError("Stage 2 HLE source-file set is invalid")
    for name in ("unbiased", *HLE_BIASES):
        source_entry = source_files[name]
        if not isinstance(source_entry, Mapping) or set(source_entry) != {
            "path",
            "content_sha256",
            "byte_count",
            "row_count",
            "question_ids_sha256",
        }:
            raise ValueError(f"Stage 2 HLE source identity for {name!r} is invalid")
        if (
            not isinstance(source_entry["path"], str)
            or not source_entry["path"]
            or source_entry["content_sha256"] != HLE_FILE_SHA256[name]
            or source_entry["row_count"] != HLE_ROWS
            or not isinstance(source_entry["byte_count"], int)
            or source_entry["byte_count"] <= 0
            or not isinstance(source_entry["question_ids_sha256"], str)
            or len(source_entry["question_ids_sha256"]) != 64
            or not isinstance(hle_artifacts[name], Mapping)
            or hle_artifacts[name].get("source") != source_entry
            or hle_artifacts[name].get("content_sha256") != HLE_FILE_SHA256[name]
        ):
            raise ValueError(f"Stage 2 HLE artifact {name!r} is not a byte-exact canonical copy")

    _, in_clean_rows = _validate_artifact_entry(
        in_artifacts["unbiased"],
        label="in-domain clean",
        expected_rows=IN_DOMAIN_ROWS,
        expected_datasets=set(IN_DOMAIN_DATASETS),
        bias_type=None,
    )
    _, hle_clean_rows = _validate_artifact_entry(
        hle_artifacts["unbiased"],
        label="HLE clean",
        expected_rows=HLE_ROWS,
        expected_datasets={HLE_DATASET},
        bias_type=None,
    )

    stage1 = document.get("stage1_iid_source")
    if not isinstance(stage1, Mapping) or set(stage1) != {"manifest", "heldout_in_domain"}:
        raise ValueError("Stage 2 Stage 1 IID source identity is invalid")
    source_manifest = stage1["manifest"]
    if not isinstance(source_manifest, Mapping) or set(source_manifest) != {
        "path",
        "content_sha256",
        "kind",
        "schema_version",
    }:
        raise ValueError("Stage 2 Stage 1 IID manifest identity is invalid")
    if (
        not isinstance(source_manifest["path"], str)
        or not source_manifest["path"]
        or not isinstance(source_manifest["content_sha256"], str)
        or len(source_manifest["content_sha256"]) != 64
        or source_manifest["kind"] != iid_prepare.MANIFEST_KIND
        or source_manifest["schema_version"] != iid_prepare.SCHEMA_VERSION
    ):
        raise ValueError("Stage 2 Stage 1 IID manifest does not match the pinned contract")
    iid_identity = _validate_source_identity(
        stage1["heldout_in_domain"], label="Stage 2 Stage 1 held-out in-domain", artifact=in_artifacts[TRAINING_BIAS]
    )
    if in_artifacts["unbiased"].get("source") != {
        "derivation": "exact_frozen_heldout_in_domain_clean_projection",
        "source_iid_heldout": iid_identity,
    }:
        raise ValueError("Stage 2 in-domain clean artifact does not retain its source identity")
    for bias_type in HELDOUT_BIASES:
        if in_artifacts[bias_type].get("source") != {
            "derivation": "exact_frozen_heldout_in_domain_clean_messages_plus_pinned_mcq_bias_injector",
            "source_iid_heldout": iid_identity,
            "bias_type": bias_type,
        }:
            raise ValueError(f"Stage 2 in-domain {bias_type!r} artifact source is invalid")

    for population_name, artifacts, clean_rows, count, datasets in (
        ("in-domain", in_artifacts, in_clean_rows, IN_DOMAIN_ROWS, set(IN_DOMAIN_DATASETS)),
        ("HLE", hle_artifacts, hle_clean_rows, HLE_ROWS, {HLE_DATASET}),
    ):
        clean_by_id = _rows_by_id(clean_rows, path=Path(artifacts["unbiased"]["path"]), label=f"{population_name} clean")
        targets: dict[str, str] | None = None
        for bias_type in HLE_BIASES:
            _, biased_rows = _validate_artifact_entry(
                artifacts[bias_type],
                label=f"{population_name} {bias_type}",
                expected_rows=count,
                expected_datasets=datasets,
                bias_type=bias_type,
            )
            by_id = _rows_by_id(
                biased_rows, path=Path(artifacts[bias_type]["path"]), label=f"{population_name} {bias_type}"
            )
            if set(by_id) != set(clean_by_id):
                raise ValueError(f"{population_name} {bias_type}: IDs differ from clean population")
            current_targets: dict[str, str] = {}
            for question_id, row in by_id.items():
                for field in _CLEAN_FIELDS:
                    if row[field] != clean_by_id[question_id][field]:
                        raise ValueError(f"{population_name} {bias_type}: {field} differs for {question_id!r}")
                current_targets[question_id] = str(row["biased_option"])
            if targets is None:
                targets = current_targets
            elif current_targets != targets:
                raise ValueError(f"{population_name} {bias_type}: biased targets differ across variants")

    expected_regimes = {
        IID: ("seen", "seen", "in_domain", [TRAINING_BIAS], IN_DOMAIN_ROWS),
        HELDOUT_DATASET: ("held_out", "seen", "hle", [TRAINING_BIAS], HLE_ROWS),
        HELDOUT_BIAS: ("seen", "held_out", "in_domain", list(HELDOUT_BIASES), IN_DOMAIN_ROWS * len(HELDOUT_BIASES)),
        HELDOUT_DATASET_AND_BIAS: ("held_out", "held_out", "hle", list(HELDOUT_BIASES), HLE_ROWS * len(HELDOUT_BIASES)),
    }
    for name, (dataset_status, bias_status, population, biases, count) in expected_regimes.items():
        if regimes[name] != {
            "dataset_status": dataset_status,
            "bias_status": bias_status,
            "population": population,
            "clean_artifact": "unbiased",
            "bias_artifacts": biases,
            "nominal_biased_samples": count,
        }:
            raise ValueError(f"Stage 2 OOD regime {name!r} is invalid")
    if document.get("assertions") != {
        "all_prompt_styles_none": True,
        "in_domain_exactly_stage1_heldout_population": True,
        "in_domain_targets_shared_across_all_six_biases": True,
        "hle_targets_shared_across_all_six_biases": True,
        "all_bias_files_match_their_population_clean_ids": True,
        "headline_uses_no_training_overlap": True,
        "no_model_or_remote_service_called": True,
    }:
        raise ValueError("Stage 2 OOD assertions are invalid")
    return document


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze immutable local-only Stage 2 OOD-HLE 2×2 evaluation inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--iid-manifest", required=True, type=Path)
    parser.add_argument("--hle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_ood_hle_2x2(
            iid_manifest=args.iid_manifest,
            hle_dir=args.hle_dir,
            output_dir=args.output_dir,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Prepared immutable Stage 2 OOD-HLE 2×2 inputs: "
        f"IID={manifest['regimes'][IID]['nominal_biased_samples']}, "
        f"heldout_dataset={manifest['regimes'][HELDOUT_DATASET]['nominal_biased_samples']}, "
        f"heldout_bias={manifest['regimes'][HELDOUT_BIAS]['nominal_biased_samples']}, "
        f"both={manifest['regimes'][HELDOUT_DATASET_AND_BIAS]['nominal_biased_samples']}; "
        "no model or remote service was called."
    )


__all__ = [
    "HELDOUT_BIAS",
    "HELDOUT_BIASES",
    "HELDOUT_DATASET",
    "HELDOUT_DATASET_AND_BIAS",
    "IID",
    "IN_DOMAIN_DATASETS",
    "MANIFEST_KIND",
    "PROMPT_STYLE",
    "REGIMES",
    "SCHEMA_VERSION",
    "TRAINING_BIAS",
    "prepare_ood_hle_2x2",
    "validate_manifest",
]


if __name__ == "__main__":
    main()
