"""Materialize and verify the maximum-disjoint ACT training population.

ACT-Max is a data-scaling condition, not a new prompt or target-generation
condition.  It consumes the already attested 3,000-row Qwen3.5 no-reasoning
wrong-argument store, reserves the established 200-row Stage-2 held-out IID
population unchanged, and uses every other source row exactly once.  The
result therefore contains 2,800 examples: 1,400 LogiQA and 1,400 HellaSwag.

This module is deliberately offline: it makes no model, provider, or GPU call.
Both the selected JSONL and its manifest are content addressed and writes are
idempotent only when an existing file is byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# This is the repository's dependency-free copy of the strict Qwen3.5 source
# recovery proof.  Importing the public mcq_bias adapter would transitively
# require the optional live injector package on a custody host; this audit must
# remain runnable with only the frozen files present.
from experiments.rmct_256 import selection as _qwen35_source_contract

CANONICAL_PAIR_SCHEMA = _qwen35_source_contract.CANONICAL_PAIR_SCHEMA
CANONICAL_PAIR_TRANSFORM = _qwen35_source_contract.CANONICAL_PAIR_TRANSFORM


SCHEMA_VERSION = 1
MANIFEST_KIND = "act_max_training_selection_manifest"
MODEL = "Qwen/Qwen3.5-9B"

DATASETS = ("logiqa", "hellaswag")
RECOVERED_ROWS = 3000
RECOVERED_COUNTS = {"logiqa": 1500, "hellaswag": 1500}
CANONICAL_PREFIX_ROWS = 2048
CANONICAL_PREFIX_COUNTS = {"logiqa": 1024, "hellaswag": 1024}
IID_ROWS = 200
IID_COUNTS = {"logiqa": 100, "hellaswag": 100}
ACT_MAX_ROWS = RECOVERED_ROWS - IID_ROWS
ACT_MAX_COUNTS = {"logiqa": 1400, "hellaswag": 1400}
IID_SOURCE_ROWS = (2049, 2248)
SELECTED_SOURCE_SPANS = ((1, 2048), (2249, 3000))

# These identities are the attested Qwen3.5 no-CoT source chain.  They are
# intentionally literal, so this module cannot accept a same-shaped but
# different prompt store.
RECOVERED_NONE_SOURCE_SHA256 = "7d113ee1858426721d09b23a78f4bbb0e9b16e7576b3ee5ab7da4924c3a0ef3b"
RECOVERED_NONE_MANIFEST_SHA256 = "a67a7a76a4f4a875e19aac966d2ec4038cc2ed2e9226ac2e72f3b5a0a70e86d9"
CANONICAL_PREFIX_SHA256 = "a6e0a554ecdc40974d16f0504f0d30e783d88ac442e91697f07f7c0c8495a08d"
CANONICAL_PREFIX_MANIFEST_SHA256 = "233e14f5d15a4820dec48e1ca0e5523969e0be8b8265872990a79a1700961d23"
IID_REFERENCE_MANIFEST_SHA256 = "8cdd4da0575a125b01b2b9b62d9c0e07176142eec6194e72e8abd04401ca2fec"
IID_HELDOUT_SHA256 = "375158a369e3f040c562136c280ba1cb3f6cf456ffc0874cdb387de0ea44c9c4"
IID_IDS_SHA256 = "499106c1c45cc422d8b231d17a0b87d6cd0636a843fc0c222cdb04bed0198ae1"
STAGE2_MANIFEST_SHA256 = "147b1739ad407058e71385f473d8f050e4689166f9a97122c40de506372d7850"
STAGE2_IN_DOMAIN_CONTENT_SHA256 = "3f37efd16fd1bb0ace2121ea3756e86abe29ee5b17f4c0bfc9a45f4a9e602ee8"
STAGE2_SCHEMA = "stage2-ood-hle-manifest-v1"
STAGE2_KIND = "stage2_ood_hle_2x2_manifest"
STAGE2_HLE_ROWS = 100
STAGE2_IN_DOMAIN_ARTIFACTS = frozenset(
    {
        "unbiased",
        "wrong_argument",
        "suggested_answer",
        "distractor_fact",
        "post_hoc",
        "spurious_few_shot_squares",
        "wrong_few_shot",
    }
)

DATA_FILENAME_PREFIX = "act-max-training-"
MANIFEST_FILENAME_PREFIX = "act-max-training-manifest-"


@dataclass(frozen=True, slots=True)
class MaterializedSelection:
    """Identities returned after a successful immutable selection publish."""

    data_path: Path
    manifest_path: Path
    data_sha256: str
    manifest_sha256: str
    data_status: str
    manifest_status: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ids_sha256(question_ids: Sequence[str]) -> str:
    return _sha256("".join(f"{question_id}\n" for question_id in question_ids).encode("utf-8"))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def _regular_file(path: str | Path, *, label: str) -> Path:
    supplied = Path(path)
    resolved = supplied.resolve()
    if supplied.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file: {supplied}")
    return resolved


def _read_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any], bytes]:
    resolved = _regular_file(path, label=label)
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {resolved}")
    return resolved, value, payload


def _read_jsonl(path: str | Path, *, label: str) -> tuple[Path, list[dict[str, Any]], bytes]:
    resolved = _regular_file(path, label=label)
    payload = resolved.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{label} must be a non-empty LF-terminated JSONL file: {resolved}")
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        if not raw_line.endswith(b"\n") or raw_line.endswith(b"\r\n") or not raw_line[:-1].strip():
            raise ValueError(f"{label} has a non-canonical line at {resolved}:{line_number}")
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} has invalid JSON at {resolved}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row must be an object at {resolved}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} has no JSON rows: {resolved}")
    return resolved, rows, payload


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _require_digest(payload: bytes, expected: str, *, label: str) -> None:
    actual = _sha256(payload)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _dataset_counts(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, int]:
    counts = Counter(str(row.get("source_dataset", "")) for row in rows)
    unknown = sorted(set(counts) - set(DATASETS))
    if unknown:
        raise ValueError(f"{label} has unsupported dataset(s): {unknown}")
    return {dataset: counts.get(dataset, 0) for dataset in DATASETS}


def _portable_identity(path: Path, payload: bytes, *, row_count: int | None = None) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "filename": path.name,
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
    }
    if row_count is not None:
        identity["row_count"] = row_count
    return identity


def _verify_recovered_and_canonical(
    *,
    recovered_none_source: str | Path,
    recovered_none_manifest: str | Path,
    canonical_prefix: str | Path,
    canonical_prefix_manifest: str | Path,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any], bytes]:
    """Verify the complete source plus the pre-existing n=2,048 pool."""

    source_path = _regular_file(recovered_none_source, label="recovered no-CoT source")
    recovery_manifest_path = _regular_file(recovered_none_manifest, label="recovered no-CoT manifest")
    source_payload = source_path.read_bytes()
    recovery_manifest_payload = recovery_manifest_path.read_bytes()
    _require_digest(source_payload, RECOVERED_NONE_SOURCE_SHA256, label="recovered no-CoT source")
    _require_digest(recovery_manifest_payload, RECOVERED_NONE_MANIFEST_SHA256, label="recovered no-CoT manifest")

    # Reuse the historical content-level conversion gate, not merely its
    # metadata. It checks the legacy prompt body and the replacement none
    # terminal on every source row.
    source_rows = _qwen35_source_contract._verify_recovered_none_pairs(source_path, recovery_manifest_path)
    _require_equal(len(source_rows), RECOVERED_ROWS, label="recovered no-CoT row count")
    _require_equal(
        _dataset_counts(source_rows, label="recovered no-CoT source"),
        RECOVERED_COUNTS,
        label="recovered no-CoT dataset counts",
    )
    source_ids = [str(row.get("question_id", "")) for row in source_rows]
    if any(not question_id for question_id in source_ids) or len(source_ids) != len(set(source_ids)):
        raise ValueError("recovered no-CoT source must have 3,000 unique non-empty question IDs")

    canonical_path, canonical_document, canonical_payload = _read_json(
        canonical_prefix_manifest, label="canonical n=2048 manifest"
    )
    _require_digest(canonical_payload, CANONICAL_PREFIX_MANIFEST_SHA256, label="canonical n=2048 manifest")
    _require_equal(canonical_document.get("artifact_schema"), CANONICAL_PAIR_SCHEMA, label="canonical n=2048 schema")
    _require_equal(canonical_document.get("schema_version"), 1, label="canonical n=2048 schema version")
    _require_equal(canonical_document.get("row_count"), CANONICAL_PREFIX_ROWS, label="canonical n=2048 row count")
    _require_equal(canonical_document.get("content_sha256"), CANONICAL_PREFIX_SHA256, label="canonical n=2048 digest")
    provenance = canonical_document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("canonical n=2048 manifest provenance must be an object")
    source_identity = provenance.get("source")
    selection = provenance.get("selection")
    transform = provenance.get("transform")
    if not isinstance(source_identity, Mapping) or not isinstance(selection, Mapping) or not isinstance(transform, Mapping):
        raise ValueError("canonical n=2048 manifest provenance is incomplete")
    _require_equal(source_identity.get("content_sha256"), RECOVERED_NONE_SOURCE_SHA256, label="canonical source digest")
    _require_equal(source_identity.get("row_count"), RECOVERED_ROWS, label="canonical source row count")
    _require_equal(selection.get("limit"), CANONICAL_PREFIX_ROWS, label="canonical n=2048 source limit")
    _require_equal(transform.get("name"), CANONICAL_PAIR_TRANSFORM, label="canonical transform")

    canonical_data_path, canonical_rows, canonical_data_payload = _read_jsonl(
        canonical_prefix, label="canonical n=2048 data"
    )
    _require_digest(canonical_data_payload, CANONICAL_PREFIX_SHA256, label="canonical n=2048 data")
    _require_equal(len(canonical_rows), CANONICAL_PREFIX_ROWS, label="canonical n=2048 physical row count")
    _require_equal(
        _dataset_counts(canonical_rows, label="canonical n=2048 data"),
        CANONICAL_PREFIX_COUNTS,
        label="canonical n=2048 dataset counts",
    )
    reconstructed_prefix = _canonical_jsonl(
        [
            _qwen35_source_contract._canonicalize_wrong_argument_row(row, path=source_path, line_number=index)
            for index, row in enumerate(source_rows[:CANONICAL_PREFIX_ROWS], start=1)
        ]
    )
    if canonical_data_payload != reconstructed_prefix:
        raise ValueError("canonical n=2048 data does not byte-match the attested recovered-no-CoT transform")

    source_proof = {
        "recovered_none": {
            **_portable_identity(source_path, source_payload, row_count=RECOVERED_ROWS),
            "counts_by_dataset": dict(RECOVERED_COUNTS),
            "manifest": _portable_identity(recovery_manifest_path, recovery_manifest_payload),
            "prompt_style": "none",
            "bias_type": "wrong_argument",
        },
        "canonical_prefix_n2048": {
            **_portable_identity(canonical_data_path, canonical_data_payload, row_count=CANONICAL_PREFIX_ROWS),
            "counts_by_dataset": dict(CANONICAL_PREFIX_COUNTS),
            "manifest": _portable_identity(canonical_path, canonical_payload),
            "transform": CANONICAL_PAIR_TRANSFORM,
        },
    }
    return source_path, source_rows, source_proof, canonical_data_payload


def _verify_iid_and_stage2(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    iid_reference_manifest: str | Path,
    iid_heldout: str | Path,
    stage2_manifest: str | Path,
) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    """Verify the frozen IID reservation and its Stage-2/HLE binding."""

    iid_manifest_path, iid_document, iid_manifest_payload = _read_json(
        iid_reference_manifest, label="Stage-1 IID reference manifest"
    )
    _require_digest(iid_manifest_payload, IID_REFERENCE_MANIFEST_SHA256, label="Stage-1 IID reference manifest")
    _require_equal(iid_document.get("schema_version"), 1, label="Stage-1 IID manifest schema version")
    _require_equal(iid_document.get("kind"), "stage1_iid_diagnostic_none_manifest", label="Stage-1 IID manifest kind")
    source = iid_document.get("source")
    selection = iid_document.get("selection")
    splits = iid_document.get("splits")
    if not isinstance(source, Mapping) or not isinstance(selection, Mapping) or not isinstance(splits, Mapping):
        raise ValueError("Stage-1 IID manifest source, selection, and splits must be objects")
    _require_equal(source.get("content_sha256"), RECOVERED_NONE_SOURCE_SHA256, label="Stage-1 IID recovered source hash")
    _require_equal(source.get("row_count"), RECOVERED_ROWS, label="Stage-1 IID recovered source rows")
    _require_equal(
        selection.get("heldout_selected_source_rows_1_based_inclusive"),
        list(IID_SOURCE_ROWS),
        label="Stage-1 IID held-out source rows",
    )
    heldout_entry = splits.get("heldout_in_domain")
    if not isinstance(heldout_entry, Mapping):
        raise ValueError("Stage-1 IID manifest lacks heldout_in_domain")
    _require_equal(heldout_entry.get("row_count"), IID_ROWS, label="Stage-1 IID held-out rows")
    _require_equal(heldout_entry.get("counts_by_dataset"), IID_COUNTS, label="Stage-1 IID held-out counts")
    _require_equal(heldout_entry.get("content_sha256"), IID_HELDOUT_SHA256, label="Stage-1 IID held-out hash")
    _require_equal(heldout_entry.get("question_ids_sha256"), IID_IDS_SHA256, label="Stage-1 IID held-out ID hash")

    iid_data_path, iid_rows, iid_payload = _read_jsonl(iid_heldout, label="Stage-1 IID held-out data")
    _require_digest(iid_payload, IID_HELDOUT_SHA256, label="Stage-1 IID held-out data")
    _require_equal(len(iid_rows), IID_ROWS, label="Stage-1 IID physical row count")
    _require_equal(_dataset_counts(iid_rows, label="Stage-1 IID held-out data"), IID_COUNTS, label="Stage-1 IID physical counts")
    iid_ids = [str(row.get("question_id", "")) for row in iid_rows]
    if any(not question_id for question_id in iid_ids) or len(iid_ids) != len(set(iid_ids)):
        raise ValueError("Stage-1 IID held-out data must have exactly 200 unique non-empty question IDs")
    _require_equal(_ids_sha256(iid_ids), IID_IDS_SHA256, label="Stage-1 IID calculated ID hash")
    _require_equal(heldout_entry.get("question_ids"), iid_ids, label="Stage-1 IID manifest IDs")

    expected_iid_rows = list(source_rows[IID_SOURCE_ROWS[0] - 1 : IID_SOURCE_ROWS[1]])
    if iid_rows != expected_iid_rows:
        raise ValueError("Stage-1 IID held-out rows do not exactly match source rows 2,049--2,248")

    stage2_path, stage2_document, stage2_payload = _read_json(stage2_manifest, label="Stage-2 manifest")
    _require_digest(stage2_payload, STAGE2_MANIFEST_SHA256, label="Stage-2 manifest")
    _require_equal(stage2_document.get("schema"), STAGE2_SCHEMA, label="Stage-2 schema")
    _require_equal(stage2_document.get("schema_version"), 1, label="Stage-2 schema version")
    _require_equal(stage2_document.get("kind"), STAGE2_KIND, label="Stage-2 kind")
    stage2_iid_source = stage2_document.get("stage1_iid_source")
    populations = stage2_document.get("populations")
    if not isinstance(stage2_iid_source, Mapping) or not isinstance(populations, Mapping):
        raise ValueError("Stage-2 manifest lacks stage1_iid_source or populations")
    stage2_iid_manifest = stage2_iid_source.get("manifest")
    stage2_iid_data = stage2_iid_source.get("heldout_in_domain")
    if not isinstance(stage2_iid_manifest, Mapping) or not isinstance(stage2_iid_data, Mapping):
        raise ValueError("Stage-2 manifest has incomplete Stage-1 IID provenance")
    _require_equal(
        stage2_iid_manifest.get("content_sha256"), IID_REFERENCE_MANIFEST_SHA256, label="Stage-2 IID manifest hash"
    )
    _require_equal(stage2_iid_data.get("content_sha256"), IID_HELDOUT_SHA256, label="Stage-2 IID data hash")
    _require_equal(stage2_iid_data.get("question_ids_sha256"), IID_IDS_SHA256, label="Stage-2 IID ID hash")

    in_domain = populations.get("in_domain")
    hle = populations.get("hle")
    if not isinstance(in_domain, Mapping) or not isinstance(hle, Mapping):
        raise ValueError("Stage-2 manifest must contain in_domain and hle populations")
    _require_equal(in_domain.get("row_count"), IID_ROWS, label="Stage-2 in-domain rows")
    _require_equal(in_domain.get("counts_by_dataset"), IID_COUNTS, label="Stage-2 in-domain counts")
    in_domain_artifacts = in_domain.get("artifacts")
    if not isinstance(in_domain_artifacts, Mapping) or set(in_domain_artifacts) != STAGE2_IN_DOMAIN_ARTIFACTS:
        raise ValueError("Stage-2 in-domain artifact set differs from the frozen seven-artifact population")
    unbiased = in_domain_artifacts.get("unbiased")
    if not isinstance(unbiased, Mapping):
        raise ValueError("Stage-2 in-domain population lacks unbiased artifact")
    _require_equal(unbiased.get("content_sha256"), STAGE2_IN_DOMAIN_CONTENT_SHA256, label="Stage-2 in-domain clean hash")
    _require_equal(unbiased.get("question_ids"), iid_ids, label="Stage-2 in-domain clean IDs")
    _require_equal(unbiased.get("question_ids_sha256"), IID_IDS_SHA256, label="Stage-2 in-domain clean ID hash")
    for name, artifact in in_domain_artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Stage-2 in-domain {name!r} artifact is not an object")
        _require_equal(artifact.get("question_ids"), iid_ids, label=f"Stage-2 in-domain {name} IDs")
        _require_equal(artifact.get("question_ids_sha256"), IID_IDS_SHA256, label=f"Stage-2 in-domain {name} ID hash")
        _require_equal(artifact.get("row_count"), IID_ROWS, label=f"Stage-2 in-domain {name} rows")
        _require_equal(artifact.get("counts_by_dataset"), IID_COUNTS, label=f"Stage-2 in-domain {name} counts")

    _require_equal(hle.get("row_count"), STAGE2_HLE_ROWS, label="Stage-2 HLE rows")
    _require_equal(hle.get("source_dataset"), "hle-text-mc", label="Stage-2 HLE source dataset")
    hle_artifacts = hle.get("artifacts")
    if not isinstance(hle_artifacts, Mapping) or set(hle_artifacts) != STAGE2_IN_DOMAIN_ARTIFACTS:
        raise ValueError("Stage-2 HLE artifact set differs from the frozen seven-artifact population")
    hle_unbiased = hle_artifacts.get("unbiased")
    if not isinstance(hle_unbiased, Mapping):
        raise ValueError("Stage-2 HLE population lacks unbiased artifact")
    hle_ids = hle_unbiased.get("question_ids")
    if not isinstance(hle_ids, list) or len(hle_ids) != STAGE2_HLE_ROWS or len(hle_ids) != len(set(hle_ids)):
        raise ValueError("Stage-2 HLE clean IDs must be exactly 100 unique IDs")
    if any(not isinstance(question_id, str) or not question_id for question_id in hle_ids):
        raise ValueError("Stage-2 HLE clean IDs must be non-empty strings")
    hle_ids_hash = _ids_sha256(hle_ids)
    _require_equal(hle_unbiased.get("question_ids_sha256"), hle_ids_hash, label="Stage-2 HLE clean ID hash")
    for name, artifact in hle_artifacts.items():
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Stage-2 HLE {name!r} artifact is not an object")
        _require_equal(artifact.get("question_ids"), hle_ids, label=f"Stage-2 HLE {name} IDs")
        _require_equal(artifact.get("question_ids_sha256"), hle_ids_hash, label=f"Stage-2 HLE {name} ID hash")
        _require_equal(artifact.get("row_count"), STAGE2_HLE_ROWS, label=f"Stage-2 HLE {name} rows")

    iid_proof = {
        "reference_manifest": _portable_identity(iid_manifest_path, iid_manifest_payload),
        "heldout_data": {
            **_portable_identity(iid_data_path, iid_payload, row_count=IID_ROWS),
            "counts_by_dataset": dict(IID_COUNTS),
            "question_ids_sha256": IID_IDS_SHA256,
            "source_rows_1_based_inclusive": list(IID_SOURCE_ROWS),
        },
    }
    stage2_proof = {
        "manifest": _portable_identity(stage2_path, stage2_payload),
        "in_domain": {
            "row_count": IID_ROWS,
            "counts_by_dataset": dict(IID_COUNTS),
            "question_ids_sha256": IID_IDS_SHA256,
            "clean_content_sha256": STAGE2_IN_DOMAIN_CONTENT_SHA256,
            "all_seven_artifacts_share_question_ids": True,
        },
        "hle": {
            "row_count": STAGE2_HLE_ROWS,
            "source_dataset": "hle-text-mc",
            "question_ids_sha256": hle_ids_hash,
            "all_seven_artifacts_share_question_ids": True,
        },
    }
    return iid_ids, iid_proof, stage2_proof


def _derive_selection(
    *,
    recovered_none_source: str | Path,
    recovered_none_manifest: str | Path,
    canonical_prefix: str | Path,
    canonical_prefix_manifest: str | Path,
    iid_reference_manifest: str | Path,
    iid_heldout: str | Path,
    stage2_manifest: str | Path,
) -> tuple[bytes, dict[str, Any]]:
    """Construct the expected payload/document without writing either file."""

    source_path, source_rows, source_proof, canonical_prefix_payload = _verify_recovered_and_canonical(
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        canonical_prefix=canonical_prefix,
        canonical_prefix_manifest=canonical_prefix_manifest,
    )
    iid_ids, iid_proof, stage2_proof = _verify_iid_and_stage2(
        source_rows=source_rows,
        iid_reference_manifest=iid_reference_manifest,
        iid_heldout=iid_heldout,
        stage2_manifest=stage2_manifest,
    )
    iid_id_set = set(iid_ids)
    source_ids = [str(row["question_id"]) for row in source_rows]
    selected_source_rows = [
        (index, row)
        for index, row in enumerate(source_rows, start=1)
        if str(row["question_id"]) not in iid_id_set
    ]
    selected_positions = [index for index, _ in selected_source_rows]
    expected_positions = [
        *range(SELECTED_SOURCE_SPANS[0][0], SELECTED_SOURCE_SPANS[0][1] + 1),
        *range(SELECTED_SOURCE_SPANS[1][0], SELECTED_SOURCE_SPANS[1][1] + 1),
    ]
    _require_equal(selected_positions, expected_positions, label="ACT-Max selected source positions")
    selected_rows = [
        _qwen35_source_contract._canonicalize_wrong_argument_row(row, path=source_path, line_number=index)
        for index, row in selected_source_rows
    ]
    selected_ids = [str(row["question_id"]) for row in selected_rows]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("ACT-Max selection contains duplicate question IDs")
    _require_equal(len(selected_rows), ACT_MAX_ROWS, label="ACT-Max selected row count")
    _require_equal(
        _dataset_counts(selected_rows, label="ACT-Max selection"),
        ACT_MAX_COUNTS,
        label="ACT-Max selected dataset counts",
    )
    if set(selected_ids) & iid_id_set:
        raise ValueError("ACT-Max selection overlaps the frozen IID headline population")

    payload = _canonical_jsonl(selected_rows)
    # This protects the existing frozen n=2,048 canonical pool as a literal
    # prefix of ACT-Max, rather than only checking equivalent JSON objects.
    if not payload.startswith(canonical_prefix_payload):
        raise ValueError("ACT-Max's first 2,048 rows do not byte-match the canonical frozen pool")
    data_sha256 = _sha256(payload)
    # HLE IDs are stored in the Stage-2 proof only as a digest. Re-read the
    # stage2 document's exact IDs after its hash validation to prove this
    # additional non-leakage invariant without reintroducing remote I/O.
    _, stage2_document, _ = _read_json(stage2_manifest, label="Stage-2 manifest")
    hle_ids = stage2_document["populations"]["hle"]["artifacts"]["unbiased"]["question_ids"]
    source_hle_overlap = sorted(set(source_ids) & set(hle_ids))
    if source_hle_overlap:
        raise ValueError(f"training source overlaps frozen HLE IDs: {source_hle_overlap[:8]}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": MANIFEST_KIND,
        "model": MODEL,
        "source": source_proof,
        "selection": {
            "filename": f"{DATA_FILENAME_PREFIX}{data_sha256}.jsonl",
            "content_sha256": data_sha256,
            "byte_count": len(payload),
            "row_count": ACT_MAX_ROWS,
            "counts_by_dataset": dict(ACT_MAX_COUNTS),
            "question_ids": selected_ids,
            "question_ids_sha256": _ids_sha256(selected_ids),
            "source_rows_included_1_based": [list(span) for span in SELECTED_SOURCE_SPANS],
            "source_rows_excluded_1_based_inclusive": list(IID_SOURCE_ROWS),
            "selection_method": "all_verified_source_rows_except_fixed_stage2_iid_holdout",
            "canonical_transform": CANONICAL_PAIR_TRANSFORM,
        },
        "iid_headline_reservation": {
            **iid_proof,
            "overlap_count": 0,
            "overlap_question_ids": [],
        },
        "stage2": {
            **stage2_proof,
            "source_hle_overlap_count": 0,
            "source_hle_overlap_question_ids": source_hle_overlap,
        },
        "assertions": {
            "selected_rows_are_all_and_only_non_iid_source_rows": True,
            "selected_question_ids_unique": True,
            "selected_counts_are_1400_logiqa_and_1400_hellaswag": True,
            "canonical_n2048_is_byte_identical_prefix": True,
            "zero_overlap_with_stage2_iid_headline": True,
            "zero_overlap_between_training_source_and_stage2_hle": True,
            "all_selected_rows_use_none_style_canonical_wrong_argument_pairs": True,
            "no_model_or_remote_service_called": True,
        },
    }
    return payload, manifest


def _publish_immutable(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise FileExistsError(f"refusing to replace non-regular artifact: {path}")
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing immutable artifact: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except Exception:
        # A failed publication leaves the temporary file recoverable for audit.
        raise
    return "written"


def materialize_act_max_selection(
    *,
    recovered_none_source: str | Path,
    recovered_none_manifest: str | Path,
    canonical_prefix: str | Path,
    canonical_prefix_manifest: str | Path,
    iid_reference_manifest: str | Path,
    iid_heldout: str | Path,
    stage2_manifest: str | Path,
    output_dir: str | Path,
) -> MaterializedSelection:
    """Publish the exact 2,800-row ACT-Max selection and proof bundle."""

    payload, document = _derive_selection(
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        canonical_prefix=canonical_prefix,
        canonical_prefix_manifest=canonical_prefix_manifest,
        iid_reference_manifest=iid_reference_manifest,
        iid_heldout=iid_heldout,
        stage2_manifest=stage2_manifest,
    )
    output = Path(output_dir).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise FileExistsError(f"output_dir must be a directory, not {output}")
    data_sha256 = _sha256(payload)
    data_path = output / str(document["selection"]["filename"])
    manifest_payload = _canonical_json(document)
    manifest_sha256 = _sha256(manifest_payload)
    manifest_path = output / f"{MANIFEST_FILENAME_PREFIX}{manifest_sha256}.json"
    data_status = _publish_immutable(data_path, payload)
    manifest_status = _publish_immutable(manifest_path, manifest_payload)
    return MaterializedSelection(
        data_path=data_path,
        manifest_path=manifest_path,
        data_sha256=data_sha256,
        manifest_sha256=manifest_sha256,
        data_status=data_status,
        manifest_status=manifest_status,
    )


def verify_act_max_selection(
    *,
    selection: str | Path,
    selection_manifest: str | Path,
    recovered_none_source: str | Path,
    recovered_none_manifest: str | Path,
    canonical_prefix: str | Path,
    canonical_prefix_manifest: str | Path,
    iid_reference_manifest: str | Path,
    iid_heldout: str | Path,
    stage2_manifest: str | Path,
) -> dict[str, Any]:
    """Replay the full proof and fail closed on any input or output drift."""

    expected_payload, expected_document = _derive_selection(
        recovered_none_source=recovered_none_source,
        recovered_none_manifest=recovered_none_manifest,
        canonical_prefix=canonical_prefix,
        canonical_prefix_manifest=canonical_prefix_manifest,
        iid_reference_manifest=iid_reference_manifest,
        iid_heldout=iid_heldout,
        stage2_manifest=stage2_manifest,
    )
    selection_path = _regular_file(selection, label="ACT-Max selection")
    manifest_path, document, manifest_payload = _read_json(selection_manifest, label="ACT-Max selection manifest")
    if selection_path.read_bytes() != expected_payload:
        raise ValueError("ACT-Max selection bytes do not match the complete frozen-source proof")
    if document != expected_document or manifest_payload != _canonical_json(expected_document):
        raise ValueError("ACT-Max selection manifest does not match the complete frozen-source proof")
    expected_data_name = str(expected_document["selection"]["filename"])
    if selection_path.name != expected_data_name:
        raise ValueError(f"ACT-Max selection filename must be content addressed as {expected_data_name}")
    expected_manifest_name = f"{MANIFEST_FILENAME_PREFIX}{_sha256(_canonical_json(expected_document))}.json"
    if manifest_path.name != expected_manifest_name:
        raise ValueError(f"ACT-Max manifest filename must be content addressed as {expected_manifest_name}")
    return document


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recovered-none-source", required=True, type=Path)
    parser.add_argument("--recovered-none-manifest", required=True, type=Path)
    parser.add_argument("--canonical-prefix", required=True, type=Path)
    parser.add_argument("--canonical-prefix-manifest", required=True, type=Path)
    parser.add_argument("--iid-reference-manifest", required=True, type=Path)
    parser.add_argument("--iid-heldout", required=True, type=Path)
    parser.add_argument("--stage2-manifest", required=True, type=Path)


def _kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "recovered_none_source": args.recovered_none_source,
        "recovered_none_manifest": args.recovered_none_manifest,
        "canonical_prefix": args.canonical_prefix,
        "canonical_prefix_manifest": args.canonical_prefix_manifest,
        "iid_reference_manifest": args.iid_reference_manifest,
        "iid_heldout": args.iid_heldout,
        "stage2_manifest": args.stage2_manifest,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize", help="write the immutable ACT-Max training selection")
    _add_source_arguments(materialize_parser)
    materialize_parser.add_argument("--output-dir", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify", help="replay the ACT-Max proof without writing")
    _add_source_arguments(verify_parser)
    verify_parser.add_argument("--selection", required=True, type=Path)
    verify_parser.add_argument("--selection-manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "materialize":
            result = materialize_act_max_selection(output_dir=args.output_dir, **_kwargs(args))
            print(
                json.dumps(
                    {
                        "data_path": str(result.data_path),
                        "manifest_path": str(result.manifest_path),
                        "data_sha256": result.data_sha256,
                        "manifest_sha256": result.manifest_sha256,
                        "data_status": result.data_status,
                        "manifest_status": result.manifest_status,
                    },
                    sort_keys=True,
                )
            )
        else:
            document = verify_act_max_selection(
                selection=args.selection,
                selection_manifest=args.selection_manifest,
                **_kwargs(args),
            )
            print(
                json.dumps(
                    {
                        "kind": document["kind"],
                        "row_count": document["selection"]["row_count"],
                        "counts_by_dataset": document["selection"]["counts_by_dataset"],
                        "selection_sha256": document["selection"]["content_sha256"],
                        "verified": True,
                    },
                    sort_keys=True,
                )
            )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()


__all__ = [
    "ACT_MAX_COUNTS",
    "ACT_MAX_ROWS",
    "DATASETS",
    "IID_COUNTS",
    "IID_ROWS",
    "MaterializedSelection",
    "materialize_act_max_selection",
    "verify_act_max_selection",
]
