"""Freeze a disjoint, homogeneous-target ACT data-scaling question pool.

This module is deliberately *offline*: it does not call Hugging Face,
OpenRouter, a model, or a GPU.  It has three narrow responsibilities:

* attest locally staged copies of the two pinned public train splits;
* audit the current splits against the complete legacy 1,500-per-dataset
  source using repository-rendered MCQ normalisation, then freeze a disjoint
  union with the existing ACT-Max 1,400-per-dataset training population; and
* publish question-only generation requests.  In particular, it never copies
  a legacy ``biasing_text`` into the expanded condition.  Every selected
  question, including the legacy questions, must receive a new wrong argument
  from the one pinned Gemma generator before it can become training data.

The two public-source revisions are pinned below.  Downloading and converting
them into the small three-field local JSONL input is intentionally outside this
module so that source custody, collision auditing, and target generation remain
separate reviewable steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from typing import Any


SCHEMA_VERSION = 2
SOURCE_SNAPSHOT_SCHEMA = "act_expanded_pinned_mcq_source_v2"
CROSS_FORMAT_AUDIT_SCHEMA = "act_expanded_cross_format_audit_v2"
SELECTION_SCHEMA = "act_expanded_question_selection_v2"
SOURCE_SNAPSHOT_KIND = "act_expanded_pinned_mcq_source"
CROSS_FORMAT_AUDIT_KIND = "act_expanded_cross_format_disjointness_audit"
SELECTION_KIND = "act_expanded_homogeneous_gemma_question_selection"

CANDIDATE_SOURCE_MODES = ("legacy_plus_fresh", "fresh_only")

DATASETS = ("logiqa", "hellaswag")
PROMPT_STYLE = "none"
BIAS_TYPE = "wrong_argument"
CANONICAL_PAIR_TRANSFORM = "wrong_argument_prefix_v1"
ORDER_SEED = "act-expanded-v2-20260804"
FRESH_IID_RESERVE_PER_DATASET = 100
PLANNED_TRAINING_TOTAL = 8192
PLANNED_SAFE_LEGACY_PER_DATASET = 1300
GENERATOR_PROVIDER = "openrouter"
GENERATOR_MODEL = "google/gemma-4-31b-it"

# This is deliberately named after the loader-level representation rather than
# the raw data.  The historical frozen questions contain the rendered MCQ,
# whereas a staged public row contains question/options/answer-index fields.
# Matching must therefore happen after both are rendered into this common form.
REPOSITORY_RENDERED_MCQ_NORMALIZATION = (
    "nfkc-folded-whitespace-choice-labels-plus-logiqa-semantic-signatures-v3"
)
ANSWER_CHOICES_HEADER = "\n\nAnswer choices:\n"

# The source counts are properties of the pinned public revisions.  The
# collision figures are a second, fail-closed check on the cross-format audit;
# the audit also records runtime-derived source-row mappings and their digests.
SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "logiqa": {
        "repository": "lucasmccabe/logiqa",
        "revision": "fa9f9918fa81eca088805c1395d7f592f7755ae0",
        "split": "train",
        "raw_row_count": 7376,
        "unique_canonical_count": 7363,
    },
    "hellaswag": {
        "repository": "Rowan/hellaswag",
        "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "train",
        "raw_row_count": 39905,
        "unique_canonical_count": 39905,
    },
}

# Counts refer to the complete legacy source (including the already-frozen
# 100/dataset IID rows), before the new extra 100/dataset IID reservation.
# ``physical`` counts source rows; ``unique`` counts normalised MCQs.
CORE_CROSS_FORMAT_EXPECTATIONS: dict[str, dict[str, int]] = {
    "logiqa": {
        "source_physical_rows": 7376,
        "source_unique_rows": 7363,
        "legacy_collision_physical_rows": 1383,
        "legacy_collision_unique_rows": 1380,
        "secondary_signature_screen_physical_rows": 1434,
        "secondary_signature_screen_unique_rows": 1430,
        "secondary_signature_additional_physical_rows": 51,
        "secondary_signature_additional_unique_rows": 50,
        "fresh_safe_physical_rows": 5993,
        "fresh_safe_unique_rows": 5983,
        "full_stem_alnum_collision_physical_rows": 1383,
        "query_and_ordered_options_collision_physical_rows": 1342,
        "ordered_options_collision_physical_rows": 1364,
        "ambiguous_legacy_match_physical_rows": 0,
        "ambiguous_legacy_match_unique_rows": 0,
        "ground_truth_disagreement_physical_rows": 0,
        "ground_truth_disagreement_unique_rows": 0,
        "secondary_signature_ambiguous_physical_rows": 14,
        "secondary_signature_ambiguous_unique_rows": 14,
        "secondary_signature_ground_truth_disagreement_physical_rows": 3,
        "secondary_signature_ground_truth_disagreement_unique_rows": 3,
        "direct_lineage_ground_truth_disagreement_physical_rows": 0,
        "direct_lineage_ground_truth_disagreement_unique_rows": 0,
    },
    "hellaswag": {
        "source_physical_rows": 39905,
        "source_unique_rows": 39905,
        "legacy_collision_physical_rows": 0,
        "legacy_collision_unique_rows": 0,
        "secondary_signature_screen_physical_rows": 0,
        "secondary_signature_screen_unique_rows": 0,
        "secondary_signature_additional_physical_rows": 0,
        "secondary_signature_additional_unique_rows": 0,
        "fresh_safe_physical_rows": 39905,
        "fresh_safe_unique_rows": 39905,
        "ambiguous_legacy_match_physical_rows": 0,
        "ambiguous_legacy_match_unique_rows": 0,
        "ground_truth_disagreement_physical_rows": 0,
        "ground_truth_disagreement_unique_rows": 0,
        "direct_lineage_ground_truth_disagreement_physical_rows": 0,
        "direct_lineage_ground_truth_disagreement_unique_rows": 0,
        "hellaswag_context_and_ordered_options_collision_physical_rows": 0,
        "hellaswag_context_only_overlap_physical_rows": 3,
    },
}

LEGACY_POPULATION_COUNTS = {"logiqa": 1500, "hellaswag": 1500}
LEGACY_TRAINING_COUNTS = {"logiqa": 1400, "hellaswag": 1400}
EXISTING_FROZEN_IID_COUNTS = {"logiqa": 100, "hellaswag": 100}

SOURCE_DATA_FILENAME = "act-expanded-source-{dataset}-{sha256}.jsonl"
SOURCE_MANIFEST_FILENAME = "act-expanded-source-{dataset}-manifest-{sha256}.json"
AUDIT_FILENAME = "act-expanded-cross-format-audit-{sha256}.json"
FRESH_IID_FILENAME = "act-expanded-fresh-iid-reserve-{sha256}.jsonl"
CANDIDATE_FILENAME = "act-expanded-question-candidates-{sha256}.jsonl"
SELECTION_MANIFEST_FILENAME = "act-expanded-question-selection-{sha256}.json"


@dataclass(frozen=True, slots=True)
class PublishedSource:
    """Identity of one immutable public-source snapshot."""

    data_path: Path
    manifest_path: Path
    data_sha256: str
    manifest_sha256: str
    data_status: str
    manifest_status: str


@dataclass(frozen=True, slots=True)
class PublishedAudit:
    """Identity of an immutable cross-format disjointness audit."""

    path: Path
    sha256: str
    status: str


@dataclass(frozen=True, slots=True)
class PublishedSelection:
    """Identity of fresh IID and question-only generation artifacts."""

    fresh_iid_path: Path
    candidate_path: Path
    manifest_path: Path
    fresh_iid_sha256: str
    candidate_sha256: str
    manifest_sha256: str
    fresh_iid_status: str
    candidate_status: str
    manifest_status: str


@dataclass(frozen=True, slots=True)
class _Question:
    source_dataset: str
    canonical_input: str
    content_fingerprint: str
    canonical_question_id: str
    source_question_id: str
    ground_truth: str
    option_labels: tuple[str, ...]
    source_row_index: int
    source_origin: str


@dataclass(frozen=True, slots=True)
class _DerivedInputs:
    audit_document: dict[str, Any]
    legacy_training: dict[str, list[_Question]]
    safe_legacy_training: dict[str, list[_Question]]
    fresh_safe: dict[str, list[_Question]]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha1(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _canonical_jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join((json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)


def _values_sha256(values: Sequence[str | int]) -> str:
    return _sha256("".join(f"{value}\n" for value in values).encode("utf-8"))


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
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n") or not line[:-1].strip():
            raise ValueError(f"{label} has a non-canonical JSONL line at {resolved}:{line_number}")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label} has invalid JSON at {resolved}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row must be an object at {resolved}:{line_number}")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} has no rows: {resolved}")
    return resolved, rows, payload


def _require_equal(actual: object, expected: object, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _portable_file_identity(path: Path, payload: bytes, *, row_count: int | None = None) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "filename": path.name,
        "content_sha256": _sha256(payload),
        "byte_count": len(payload),
    }
    if row_count is not None:
        identity["row_count"] = row_count
    return identity


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
    os.replace(temporary, path)
    return "written"


_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


def _normalise_repository_rendered_mcq(value: str) -> str:
    """Return a conservative cross-lineage key for a rendered MCQ.

    It intentionally normalises loader-only representation differences (Unicode
    punctuation, whitespace, and choice-label decoration) but does not remove
    words or reorder choices.  A false match merely withholds a candidate; a
    missed match would risk leakage, so the operation is conservative.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("rendered MCQ must be a non-empty string")
    text = unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_TRANSLATION)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?im)^\s*answer\s+choices\s*:\s*$", "Answer choices:", text)
    text = re.sub(r"(?im)^\s*[\[(]\s*([a-z])\s*[\])\.\:]\s*", r"(\1) ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return text.casefold()


def _content_fingerprint(canonical_input: str) -> str:
    return _sha256(_normalise_repository_rendered_mcq(canonical_input).encode("utf-8"))


def _alnum_cross_lineage_key(value: str) -> str:
    """Remove translation-pipeline punctuation/spacing without dropping words."""

    normalized = unicodedata.normalize("NFKC", value).translate(_PUNCTUATION_TRANSLATION)
    return re.sub(r"[^a-z0-9]+", "", normalized.casefold())


def _rendered_mcq_semantic_parts(
    canonical_input: str,
) -> tuple[str, tuple[str, tuple[str, ...]], tuple[str, ...]]:
    """Return conservative LogiQA cross-lineage matching signatures.

    The legacy cot-transparency rendering and the pinned Hugging Face rendering
    differ systematically in sentence spacing, terminal punctuation, and, for
    a small tail, context translation.  They retain the final query and ordered
    choices.  We therefore exclude a public row if *any* of three exact,
    non-fuzzy signatures matches: full stem, query plus ordered choices, or the
    complete ordered choices.  This can conservatively withhold a false
    positive, but cannot introduce an evaluation-overlapping training row.
    """

    if ANSWER_CHOICES_HEADER not in canonical_input:
        raise ValueError("rendered MCQ has no exact Answer choices boundary")
    stem, choices_text = canonical_input.split(ANSWER_CHOICES_HEADER, 1)
    stem_lines = [line.strip() for line in stem.splitlines() if line.strip()]
    if not stem_lines:
        raise ValueError("rendered MCQ has no question stem")
    options: list[str] = []
    labels: list[str] = []
    for line in choices_text.splitlines():
        match = re.match(r"^\s*[\[(]\s*([a-z])\s*[\])\.\:]\s*(.*?)\s*$", line, re.I)
        if match is None:
            if line.strip():
                raise ValueError("rendered MCQ contains an unlabelled answer-choice line")
            continue
        labels.append(match.group(1).upper())
        option_key = _alnum_cross_lineage_key(match.group(2))
        # A handful of pinned rows have punctuation-only translated choices.
        # Keep their empty alnum signature in its exact ordered position; the
        # whole ordered tuple (and, normally, the query) still supplies the
        # collision evidence.
        options.append(option_key)
    expected_labels = list(ascii_uppercase[: len(labels)])
    if len(labels) < 2 or labels != expected_labels:
        raise ValueError("rendered MCQ answer labels must be contiguous from A")
    ordered_options = tuple(options)
    return (
        _alnum_cross_lineage_key(stem),
        (_alnum_cross_lineage_key(stem_lines[-1]), ordered_options),
        ordered_options,
    )


def _hellaswag_context_and_options_signatures(
    canonical_input: str,
) -> tuple[str, tuple[str, tuple[str, ...]]]:
    """Identify exact HellaSwag examples across prompt wrappers.

    The complete legacy population comes from HellaSwag validation.  Three
    contexts recur in train with different ordered endings, so context alone is
    recorded as an overlap diagnostic but is not treated as the same MCQ.
    """

    if ANSWER_CHOICES_HEADER not in canonical_input:
        raise ValueError("rendered HellaSwag MCQ has no exact Answer choices boundary")
    stem = canonical_input.split(ANSWER_CHOICES_HEADER, 1)[0]
    stem = re.sub(
        r"^Which of the answer choices best completes the following sentence\?\s*",
        "",
        stem,
        flags=re.I,
    )
    _, (_, ordered_options), _ = _rendered_mcq_semantic_parts(canonical_input)
    context = _alnum_cross_lineage_key(stem)
    return context, (context, ordered_options)


def _render_canonical_input(question: str, options: Sequence[str]) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ValueError("options must be a sequence of strings")
    values = list(options)
    if not 2 <= len(values) <= len(ascii_uppercase):
        raise ValueError("options must contain 2..26 entries")
    if any(not isinstance(option, str) or not option.strip() for option in values):
        raise ValueError("every option must be a non-empty string")
    return question + ANSWER_CHOICES_HEADER + "\n".join(f"({ascii_uppercase[index]}) {option}" for index, option in enumerate(values))


def _answer_labels(canonical_input: str) -> tuple[str, ...]:
    match = re.search(r"(?is)\banswer\s+choices\s*:\s*", canonical_input)
    if match is None:
        raise ValueError("rendered MCQ has no Answer choices header")
    labels = re.findall(r"(?im)^\s*[\[(]\s*([a-z])\s*[\])\.\:]\s+", canonical_input[match.end() :])
    values = tuple(label.upper() for label in labels)
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("rendered MCQ must have at least two unique labelled answer choices")
    return values


def _deterministic_biased_option(canonical_input: str, ground_truth: str, option_labels: Sequence[str]) -> str:
    candidates = [label for label in option_labels if label != ground_truth]
    if not candidates:
        raise ValueError("MCQ has no non-ground-truth answer choice")
    return random.Random(canonical_input).choice(candidates)


def _raw_source_row_to_snapshot(row: Mapping[str, Any], *, dataset: str, source_row_index: int, location: str) -> dict[str, Any]:
    if set(row) != {"question", "options", "ground_truth_idx"}:
        raise ValueError(f"{location}: source row must have exactly question, options, and ground_truth_idx; got {sorted(row)}")
    question = row["question"]
    options = row["options"]
    answer_index = row["ground_truth_idx"]
    if not isinstance(answer_index, int) or isinstance(answer_index, bool):
        raise ValueError(f"{location}: ground_truth_idx must be an integer")
    canonical_input = _render_canonical_input(question, options)
    labels = _answer_labels(canonical_input)
    if not 0 <= answer_index < len(labels):
        raise ValueError(f"{location}: ground_truth_idx is outside the option range")
    ground_truth = labels[answer_index]
    return {
        "biased_option": _deterministic_biased_option(canonical_input, ground_truth, labels),
        "canonical_input": canonical_input,
        "content_fingerprint": _content_fingerprint(canonical_input),
        "ground_truth": ground_truth,
        "option_labels": list(labels),
        "question_id": _sha1(canonical_input.encode("utf-8")),
        "source_dataset": dataset,
        "source_row_index": source_row_index,
    }


def materialize_source_snapshot(*, dataset: str, raw_source: str | Path, output_dir: str | Path) -> PublishedSource:
    """Freeze a pinned local public-source snapshot without remote I/O."""

    if dataset not in SOURCE_SPECS:
        raise ValueError(f"unsupported dataset {dataset!r}; choose one of {DATASETS}")
    raw_path, raw_rows, raw_payload = _read_jsonl(raw_source, label=f"{dataset} staged raw source")
    spec = SOURCE_SPECS[dataset]
    _require_equal(len(raw_rows), spec["raw_row_count"], label=f"{dataset} raw source count")
    snapshot_rows = [_raw_source_row_to_snapshot(row, dataset=dataset, source_row_index=index, location=f"{raw_path}:{index + 1}") for index, row in enumerate(raw_rows)]
    _require_equal(
        len({str(row["content_fingerprint"]) for row in snapshot_rows}),
        spec["unique_canonical_count"],
        label=f"{dataset} unique canonical source count",
    )
    snapshot_payload = _canonical_jsonl(snapshot_rows)
    snapshot_sha256 = _sha256(snapshot_payload)
    output = Path(output_dir).resolve()
    source_filename = SOURCE_DATA_FILENAME.format(dataset=dataset, sha256=snapshot_sha256)
    source_path = output / source_filename
    document = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": SOURCE_SNAPSHOT_KIND,
        "source_dataset": dataset,
        "repository": spec["repository"],
        "revision": spec["revision"],
        "split": spec["split"],
        "raw_source": _portable_file_identity(raw_path, raw_payload, row_count=len(raw_rows)),
        "raw_row_count": len(raw_rows),
        "snapshot_row_count": len(snapshot_rows),
        "unique_canonical_count": len({str(row["content_fingerprint"]) for row in snapshot_rows}),
        "filename": source_filename,
        "content_sha256": snapshot_sha256,
        "rendering": "question + '\\n\\nAnswer choices:' + labelled options",
        "normalization": REPOSITORY_RENDERED_MCQ_NORMALIZATION,
        "assertions": {
            "no_model_or_remote_service_called": True,
            "source_order_preserved": True,
            "deterministic_wrong_option_is_not_an_argument": True,
        },
    }
    manifest_payload = _canonical_json(document)
    manifest_sha256 = _sha256(manifest_payload)
    manifest_path = output / SOURCE_MANIFEST_FILENAME.format(dataset=dataset, sha256=manifest_sha256)
    data_status = _publish_immutable(source_path, snapshot_payload)
    manifest_status = _publish_immutable(manifest_path, manifest_payload)
    return PublishedSource(
        data_path=source_path,
        manifest_path=manifest_path,
        data_sha256=snapshot_sha256,
        manifest_sha256=manifest_sha256,
        data_status=data_status,
        manifest_status=manifest_status,
    )


def _verify_source_snapshot(*, source: str | Path, manifest: str | Path, dataset: str) -> tuple[Path, Path, list[dict[str, Any]], dict[str, Any], bytes, bytes]:
    if dataset not in SOURCE_SPECS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    source_path, rows, source_payload = _read_jsonl(source, label=f"{dataset} source snapshot")
    manifest_path, document, manifest_payload = _read_json(manifest, label=f"{dataset} source manifest")
    spec = SOURCE_SPECS[dataset]
    for key, expected in (
        ("schema", SOURCE_SNAPSHOT_SCHEMA),
        ("schema_version", SCHEMA_VERSION),
        ("kind", SOURCE_SNAPSHOT_KIND),
        ("source_dataset", dataset),
        ("repository", spec["repository"]),
        ("revision", spec["revision"]),
        ("split", spec["split"]),
        ("raw_row_count", spec["raw_row_count"]),
        ("snapshot_row_count", len(rows)),
        ("content_sha256", _sha256(source_payload)),
    ):
        _require_equal(document.get(key), expected, label=f"{dataset} source manifest {key}")
    _require_equal(document.get("filename"), source_path.name, label=f"{dataset} source manifest filename")
    expected_manifest_name = SOURCE_MANIFEST_FILENAME.format(dataset=dataset, sha256=_sha256(manifest_payload))
    if manifest_path.name != expected_manifest_name:
        raise ValueError(f"{dataset} source manifest filename is not content addressed")

    positions: set[int] = set()
    fingerprints: set[str] = set()
    for index, row in enumerate(rows):
        location = f"{source_path}:{index + 1}"
        expected_fields = {
            "biased_option",
            "canonical_input",
            "content_fingerprint",
            "ground_truth",
            "option_labels",
            "question_id",
            "source_dataset",
            "source_row_index",
        }
        if set(row) != expected_fields:
            raise ValueError(f"{location}: source snapshot row has the wrong schema")
        _require_equal(row["source_dataset"], dataset, label=f"{location} source_dataset")
        position = row["source_row_index"]
        if not isinstance(position, int) or isinstance(position, bool) or position in positions:
            raise ValueError(f"{location}: source_row_index must be one unique integer")
        positions.add(position)
        canonical_input = row["canonical_input"]
        if not isinstance(canonical_input, str):
            raise ValueError(f"{location}: canonical_input must be a string")
        labels = _answer_labels(canonical_input)
        _require_equal(row["option_labels"], list(labels), label=f"{location} option_labels")
        ground_truth = row["ground_truth"]
        if not isinstance(ground_truth, str) or ground_truth not in labels:
            raise ValueError(f"{location}: invalid ground_truth")
        _require_equal(row["content_fingerprint"], _content_fingerprint(canonical_input), label=f"{location} fingerprint")
        _require_equal(row["question_id"], _sha1(canonical_input.encode("utf-8")), label=f"{location} question_id")
        _require_equal(
            row["biased_option"],
            _deterministic_biased_option(canonical_input, ground_truth, labels),
            label=f"{location} deterministic biased option",
        )
        fingerprints.add(str(row["content_fingerprint"]))
    _require_equal(len(fingerprints), spec["unique_canonical_count"], label=f"{dataset} unique source count")
    return source_path, manifest_path, rows, document, source_payload, manifest_payload


def _counts_by_dataset(rows: Sequence[_Question], *, label: str) -> dict[str, int]:
    counts = Counter(row.source_dataset for row in rows)
    unknown = sorted(set(counts) - set(DATASETS))
    if unknown:
        raise ValueError(f"{label} has unsupported dataset(s): {unknown}")
    return {dataset: counts.get(dataset, 0) for dataset in DATASETS}


def _parse_question_row(row: Mapping[str, Any], *, path: Path, line_number: int, source_origin: str, require_dataset: bool = True) -> _Question:
    """Parse a legacy/evaluation rendered-MCQ row without reading its argument."""

    location = f"{path}:{line_number}"
    question = row.get("question", row.get("canonical_input"))
    question_id = row.get("question_id")
    dataset = row.get("source_dataset")
    ground_truth = row.get("ground_truth")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{location}: question (or canonical_input) must be a non-empty string")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError(f"{location}: question_id must be a non-empty string")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"{location}: source_dataset must be a non-empty string")
    if require_dataset and dataset not in DATASETS:
        raise ValueError(f"{location}: source_dataset must be one of {DATASETS}")
    if not isinstance(ground_truth, str) or not ground_truth:
        raise ValueError(f"{location}: ground_truth must be a non-empty string")
    labels = _answer_labels(question)
    if ground_truth not in labels:
        raise ValueError(f"{location}: ground_truth is not one of the rendered choices")
    return _Question(
        source_dataset=dataset,
        canonical_input=question,
        content_fingerprint=_content_fingerprint(question),
        canonical_question_id=_sha1(question.encode("utf-8")),
        source_question_id=question_id,
        ground_truth=ground_truth,
        option_labels=labels,
        source_row_index=line_number - 1,
        source_origin=source_origin,
    )


def _load_legacy_population(path: str | Path, *, label: str, source_origin: str) -> tuple[Path, list[_Question], bytes, dict[str, Any]]:
    resolved, rows, payload = _read_jsonl(path, label=label)
    questions = [_parse_question_row(row, path=resolved, line_number=index, source_origin=source_origin) for index, row in enumerate(rows, start=1)]
    ids: set[tuple[str, str]] = set()
    fingerprints: set[tuple[str, str]] = set()
    for question in questions:
        id_key = (question.source_dataset, question.source_question_id)
        fingerprint_key = (question.source_dataset, question.content_fingerprint)
        if id_key in ids:
            raise ValueError(f"{label} has a duplicate source question ID in {question.source_dataset}")
        if fingerprint_key in fingerprints:
            raise ValueError(f"{label} has a duplicate normalized MCQ in {question.source_dataset}")
        ids.add(id_key)
        fingerprints.add(fingerprint_key)
    proof = {
        **_portable_file_identity(resolved, payload, row_count=len(questions)),
        "counts_by_dataset": _counts_by_dataset(questions, label=label),
        "source_question_ids_sha256": _values_sha256([f"{row.source_dataset}:{row.source_question_id}" for row in questions]),
        "normalized_content_keys_sha256": _values_sha256([f"{row.source_dataset}:{row.content_fingerprint}" for row in questions]),
    }
    return resolved, questions, payload, proof


def _question_map(rows: Sequence[_Question], *, label: str) -> dict[tuple[str, str], _Question]:
    mapping: dict[tuple[str, str], _Question] = {}
    for row in rows:
        key = (row.source_dataset, row.content_fingerprint)
        if key in mapping:
            raise ValueError(f"{label} has duplicate normalized content key {key!r}")
        mapping[key] = row
    return mapping


def _validate_legacy_partition(*, legacy_population: Sequence[_Question], legacy_training: Sequence[_Question], existing_iid: Sequence[_Question]) -> dict[str, list[_Question]]:
    """Prove that legacy train + existing IID is exactly the original 1,500 split."""

    _require_equal(
        _counts_by_dataset(legacy_population, label="complete legacy population"),
        LEGACY_POPULATION_COUNTS,
        label="complete legacy population counts",
    )
    _require_equal(
        _counts_by_dataset(legacy_training, label="ACT-Max legacy training"),
        LEGACY_TRAINING_COUNTS,
        label="ACT-Max legacy training counts",
    )
    _require_equal(
        _counts_by_dataset(existing_iid, label="existing frozen IID"),
        EXISTING_FROZEN_IID_COUNTS,
        label="existing frozen IID counts",
    )
    all_by_key = _question_map(legacy_population, label="complete legacy population")
    training_by_key = _question_map(legacy_training, label="ACT-Max legacy training")
    iid_by_key = _question_map(existing_iid, label="existing frozen IID")
    if set(training_by_key) & set(iid_by_key):
        raise ValueError("ACT-Max legacy training overlaps the existing frozen IID population")
    if set(training_by_key) | set(iid_by_key) != set(all_by_key):
        raise ValueError("legacy training + existing IID is not exactly the complete legacy population")
    for key, row in {**training_by_key, **iid_by_key}.items():
        reference = all_by_key[key]
        if row.ground_truth != reference.ground_truth or row.source_question_id != reference.source_question_id:
            raise ValueError(f"legacy partition row differs from its complete-population source for {key!r}")
    return {dataset: [row for row in legacy_training if row.source_dataset == dataset] for dataset in DATASETS}


def _load_prohibited_populations(
    populations: Mapping[str, str | Path],
) -> tuple[set[str], set[str], dict[str, Any]]:
    """Load non-training frozen populations to exclude from fresh candidates.

    These can include a third source dataset (such as HLE), so only their
    question IDs and normalised rendered-MCQ fingerprints are required.
    """

    if not populations:
        raise ValueError("at least one explicit frozen evaluation population is required; the legacy inputs cover historical train/IID populations but not an omitted external evaluation")
    fingerprints: set[str] = set()
    ids: set[str] = set()
    proof: dict[str, Any] = {}
    for name in sorted(populations):
        if not name or name.strip() != name:
            raise ValueError("prohibited population names must be non-empty and whitespace-free")
        path, rows, payload = _read_jsonl(populations[name], label=f"prohibited population {name!r}")
        local_ids: set[str] = set()
        local_fingerprints: set[str] = set()
        for line_number, row in enumerate(rows, start=1):
            location = f"{path}:{line_number}"
            question = row.get("question", row.get("canonical_input"))
            question_id = row.get("question_id")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{location}: question (or canonical_input) must be a non-empty string")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{location}: question_id must be a non-empty string")
            local_ids.add(question_id)
            local_fingerprints.add(_content_fingerprint(question))
        ids.update(local_ids)
        fingerprints.update(local_fingerprints)
        proof[name] = {
            **_portable_file_identity(path, payload, row_count=len(rows)),
            "unique_question_id_count": len(local_ids),
            "unique_normalized_content_count": len(local_fingerprints),
        }
    return ids, fingerprints, proof


def _source_questions(rows: Sequence[Mapping[str, Any]], *, dataset: str) -> list[_Question]:
    result: list[_Question] = []
    for row in rows:
        labels = tuple(row["option_labels"])
        result.append(
            _Question(
                source_dataset=dataset,
                canonical_input=str(row["canonical_input"]),
                content_fingerprint=str(row["content_fingerprint"]),
                canonical_question_id=str(row["question_id"]),
                source_question_id=str(row["question_id"]),
                ground_truth=str(row["ground_truth"]),
                option_labels=labels,
                source_row_index=int(row["source_row_index"]),
                source_origin="pinned_hf_train_question",
            )
        )
    return result


def _deduplicate_fresh_source(rows: Sequence[_Question], *, dataset: str) -> list[_Question]:
    first_by_content: dict[str, _Question] = {}
    for row in rows:
        previous = first_by_content.get(row.content_fingerprint)
        if previous is not None:
            if previous.ground_truth != row.ground_truth or previous.option_labels != row.option_labels or _normalise_repository_rendered_mcq(previous.canonical_input) != _normalise_repository_rendered_mcq(row.canonical_input):
                raise ValueError(f"{dataset} source has conflicting duplicate normalized MCQ content")
            continue
        first_by_content[row.content_fingerprint] = row
    return list(first_by_content.values())


def _core_cross_format_metrics(
    *,
    source_rows: Sequence[_Question],
    legacy_rows: Sequence[_Question],
    dataset: str,
) -> tuple[dict[str, Any], list[_Question]]:
    """Audit every physical source row with exact, lineage-aware signatures."""

    source_unique = _deduplicate_fresh_source(source_rows, dataset=dataset)
    match_methods = (
        "full_stem_alnum",
        "query_and_ordered_options",
        "ordered_options",
    )
    signature_indices: list[dict[object, set[int]]] = [dict() for _ in match_methods]
    hellaswag_context_indices: dict[str, set[int]] = {}
    hellaswag_example_indices: dict[tuple[str, tuple[str, ...]], set[int]] = {}
    for legacy_index, row in enumerate(legacy_rows):
        if dataset == "logiqa":
            for method_index, signature in enumerate(
                _rendered_mcq_semantic_parts(row.canonical_input)
            ):
                signature_indices[method_index].setdefault(signature, set()).add(legacy_index)
        else:
            context, example = _hellaswag_context_and_options_signatures(
                row.canonical_input
            )
            hellaswag_context_indices.setdefault(context, set()).add(legacy_index)
            hellaswag_example_indices.setdefault(example, set()).add(legacy_index)

    def matches(row: _Question) -> tuple[set[int], tuple[str, ...]]:
        if dataset != "logiqa":
            _, example = _hellaswag_context_and_options_signatures(row.canonical_input)
            indices = set(
                hellaswag_example_indices.get(example, set())
            )
            return indices, (
                ("hellaswag_context_and_ordered_options",) if indices else ()
            )
        matched: set[int] = set()
        methods: list[str] = []
        for method_index, signature in enumerate(
            _rendered_mcq_semantic_parts(row.canonical_input)
        ):
            values = signature_indices[method_index].get(signature, set())
            if values:
                matched.update(values)
                methods.append(match_methods[method_index])
        return matched, tuple(methods)

    physical_match_info = [(row, *matches(row)) for row in source_rows]
    unique_match_info = [(row, *matches(row)) for row in source_unique]

    def ambiguous_count(info: Sequence[tuple[_Question, set[int], tuple[str, ...]]]) -> int:
        return sum(len(indices) > 1 for _, indices, _ in info)

    def disagreement_count(info: Sequence[tuple[_Question, set[int], tuple[str, ...]]]) -> int:
        count = 0
        for row, indices, _ in info:
            if not indices:
                continue
            legacy_labels = {legacy_rows[index].ground_truth for index in indices}
            if legacy_labels != {row.ground_truth}:
                count += 1
        return count

    colliding_physical_info = [item for item in physical_match_info if item[1]]
    colliding_unique_info = [item for item in unique_match_info if item[1]]
    if dataset == "logiqa":
        # Keep the authoritative full-stem lineage indices separate from the
        # union accumulated by ``matches`` for the diagnostic secondary
        # signatures.  Reusing the union indices here would make an exact
        # full-stem row look ambiguous merely because its generic ordered
        # choices also match an unrelated legacy question.
        def direct_info(row: _Question) -> tuple[_Question, set[int], tuple[str, ...]]:
            full_stem = _rendered_mcq_semantic_parts(row.canonical_input)[0]
            indices = set(signature_indices[0].get(full_stem, set()))
            return row, indices, (("full_stem_alnum",) if indices else ())

        direct_physical_info = [
            item for row in source_rows if (item := direct_info(row))[1]
        ]
        direct_unique_info = [
            item for row in source_unique if (item := direct_info(row))[1]
        ]
    else:
        direct_physical_info = colliding_physical_info
        direct_unique_info = colliding_unique_info
    colliding_physical = [row for row, _, _ in direct_physical_info]
    colliding_unique = [row for row, _, _ in direct_unique_info]
    direct_physical_ids = {id(row) for row in colliding_physical}
    direct_unique_ids = {id(row) for row in colliding_unique}
    fresh_physical = [row for row in source_rows if id(row) not in direct_physical_ids]
    fresh_unique = [row for row in source_unique if id(row) not in direct_unique_ids]
    metrics = {
        "source_physical_rows": len(source_rows),
        "source_unique_rows": len(source_unique),
        # ``legacy_collision`` is the authoritative exact-lineage match and is
        # the only cross-lineage signature used to exclude training rows.
        "legacy_collision_physical_rows": len(direct_physical_info),
        "legacy_collision_unique_rows": len(direct_unique_info),
        "secondary_signature_screen_physical_rows": len(colliding_physical_info),
        "secondary_signature_screen_unique_rows": len(colliding_unique_info),
        "secondary_signature_additional_physical_rows": (
            len(colliding_physical_info) - len(direct_physical_info)
        ),
        "secondary_signature_additional_unique_rows": (
            len(colliding_unique_info) - len(direct_unique_info)
        ),
        "fresh_safe_physical_rows": len(fresh_physical),
        "fresh_safe_unique_rows": len(fresh_unique),
        "legacy_collision_source_row_indices_sha256": _values_sha256(
            [row.source_row_index for row, _, _ in direct_physical_info]
        ),
        "legacy_collision_source_question_ids_sha256": _values_sha256(
            [row.canonical_question_id for row, _, _ in direct_unique_info]
        ),
        "secondary_signature_match_pairs_sha256": _values_sha256(
            [
                f"{row.source_row_index}:{legacy_rows[index].source_question_id}"
                for row, indices, _ in colliding_physical_info
                for index in sorted(indices)
            ]
        ),
        "fresh_safe_source_question_ids_sha256": _values_sha256([row.canonical_question_id for row in fresh_unique]),
        "ambiguous_legacy_match_physical_rows": ambiguous_count(direct_physical_info),
        "ambiguous_legacy_match_unique_rows": ambiguous_count(direct_unique_info),
        "ground_truth_disagreement_physical_rows": disagreement_count(direct_physical_info),
        "ground_truth_disagreement_unique_rows": disagreement_count(direct_unique_info),
        "secondary_signature_ambiguous_physical_rows": ambiguous_count(colliding_physical_info),
        "secondary_signature_ambiguous_unique_rows": ambiguous_count(colliding_unique_info),
        "secondary_signature_ground_truth_disagreement_physical_rows": disagreement_count(colliding_physical_info),
        "secondary_signature_ground_truth_disagreement_unique_rows": disagreement_count(colliding_unique_info),
        "direct_lineage_ground_truth_disagreement_physical_rows": disagreement_count(
            direct_physical_info
        ),
        "direct_lineage_ground_truth_disagreement_unique_rows": disagreement_count(
            direct_unique_info
        ),
        "matching_policy": (
            "exact non-fuzzy HellaSwag context plus ordered options after removing the legacy fixed question wrapper"
            if dataset != "logiqa"
            else "exact non-fuzzy full-stem alphanumeric direct-lineage signature"
        ),
        "secondary_query_or_options_signatures_are_diagnostic_only": True,
        "secondary_signature_false_positives_observed": dataset == "logiqa",
    }
    if dataset == "logiqa":
        for method in match_methods:
            metrics[f"{method}_collision_physical_rows"] = sum(
                method in methods for _, _, methods in physical_match_info
            )
            metrics[f"{method}_collision_unique_rows"] = sum(
                method in methods for _, _, methods in unique_match_info
            )
    else:
        metrics["hellaswag_context_and_ordered_options_collision_physical_rows"] = len(
            colliding_physical
        )
        metrics["hellaswag_context_and_ordered_options_collision_unique_rows"] = len(
            colliding_unique
        )
        metrics["hellaswag_context_only_overlap_physical_rows"] = sum(
            bool(
                hellaswag_context_indices.get(
                    _hellaswag_context_and_options_signatures(row.canonical_input)[0],
                    set(),
                )
            )
            for row in source_rows
        )
        metrics["hellaswag_context_only_overlap_unique_rows"] = sum(
            bool(
                hellaswag_context_indices.get(
                    _hellaswag_context_and_options_signatures(row.canonical_input)[0],
                    set(),
                )
            )
            for row in source_unique
        )
    expected = CORE_CROSS_FORMAT_EXPECTATIONS[dataset]
    for field, expected_value in expected.items():
        _require_equal(metrics[field], expected_value, label=f"{dataset} cross-format audit {field}")
    return metrics, fresh_unique


def _question_identity(question: _Question) -> str:
    return f"{question.source_dataset}:{question.content_fingerprint}"


def _capacity_report(*, legacy_training: Mapping[str, Sequence[_Question]], fresh_safe: Mapping[str, Sequence[_Question]]) -> dict[str, Any]:
    """Report both requested ceilings without silently choosing the larger one."""

    legacy_counts = {dataset: len(legacy_training[dataset]) for dataset in DATASETS}
    fresh_counts = {dataset: len(fresh_safe[dataset]) for dataset in DATASETS}

    def condition(*, fresh_reserve: int) -> dict[str, Any]:
        counts = {dataset: legacy_counts[dataset] + fresh_counts[dataset] - fresh_reserve for dataset in DATASETS}
        if any(value < 0 for value in counts.values()):
            raise ValueError("fresh IID reservation is larger than the fresh-safe candidate pool")
        per_dataset = min(counts.values())
        return {
            "fresh_iid_reserve_per_dataset": fresh_reserve,
            "legacy_training_question_counts_by_dataset": dict(legacy_counts),
            "fresh_safe_question_counts_by_dataset_before_new_iid": dict(fresh_counts),
            "question_candidate_counts_by_dataset": counts,
            "balanced_training_per_dataset_ceiling": per_dataset,
            "balanced_training_total_ceiling": per_dataset * len(DATASETS),
            "hellaswag_downsampling_required_for_balance": counts["hellaswag"] > per_dataset,
        }

    return {
        "existing_frozen_iid_only": condition(fresh_reserve=0),
        "with_additional_fresh_iid_100_per_dataset": condition(fresh_reserve=FRESH_IID_RESERVE_PER_DATASET),
    }


def _derive_inputs(
    *,
    logiqa_source: str | Path,
    logiqa_manifest: str | Path,
    hellaswag_source: str | Path,
    hellaswag_manifest: str | Path,
    legacy_population: str | Path,
    legacy_training: str | Path,
    existing_iid: str | Path,
    prohibited_populations: Mapping[str, str | Path],
) -> _DerivedInputs:
    source_inputs = {
        "logiqa": (logiqa_source, logiqa_manifest),
        "hellaswag": (hellaswag_source, hellaswag_manifest),
    }
    source_rows: dict[str, list[_Question]] = {}
    source_proof: dict[str, Any] = {}
    for dataset, (source, manifest) in source_inputs.items():
        source_path, manifest_path, rows, document, source_payload, manifest_payload = _verify_source_snapshot(source=source, manifest=manifest, dataset=dataset)
        source_rows[dataset] = _source_questions(rows, dataset=dataset)
        source_proof[dataset] = {
            "snapshot": _portable_file_identity(source_path, source_payload, row_count=len(rows)),
            "manifest": _portable_file_identity(manifest_path, manifest_payload, row_count=1),
            "repository": document["repository"],
            "revision": document["revision"],
            "split": document["split"],
        }

    _, legacy_all_rows, _, legacy_all_proof = _load_legacy_population(legacy_population, label="complete legacy population", source_origin="legacy_complete_population")
    _, legacy_training_rows, _, legacy_training_proof = _load_legacy_population(legacy_training, label="ACT-Max legacy training", source_origin="legacy_act_max_question")
    _, existing_iid_rows, _, existing_iid_proof = _load_legacy_population(existing_iid, label="existing frozen IID", source_origin="existing_frozen_iid")
    legacy_training_by_dataset = _validate_legacy_partition(
        legacy_population=legacy_all_rows,
        legacy_training=legacy_training_rows,
        existing_iid=existing_iid_rows,
    )
    prohibited_ids, prohibited_fingerprints, prohibited_proof = _load_prohibited_populations(prohibited_populations)

    legacy_all_by_dataset: dict[str, list[_Question]] = {}
    for dataset in DATASETS:
        dataset_rows = [row for row in legacy_all_rows if row.source_dataset == dataset]
        # ``_question_map`` validates the dataset-qualified uniqueness.  The
        # cross-format comparison is already inside one dataset, so its lookup
        # key is the bare normalized-content fingerprint.
        _question_map(dataset_rows, label=f"complete legacy {dataset}")
        legacy_all_by_dataset[dataset] = dataset_rows
    cross_format: dict[str, Any] = {}
    fresh_safe_by_dataset: dict[str, list[_Question]] = {}
    prohibited_fresh_metrics: dict[str, Any] = {}
    for dataset in DATASETS:
        metrics, fresh_before_prohibited = _core_cross_format_metrics(
            source_rows=source_rows[dataset],
            legacy_rows=legacy_all_by_dataset[dataset],
            dataset=dataset,
        )
        # This gate is intentionally independent of the legacy mapping.  It
        # protects against future frozen evaluations drawn from new source rows.
        fresh_safe = [row for row in fresh_before_prohibited if row.content_fingerprint not in prohibited_fingerprints and row.canonical_question_id not in prohibited_ids and row.source_question_id not in prohibited_ids]
        fresh_safe_by_dataset[dataset] = fresh_safe
        fresh_safe_fingerprints = {row.content_fingerprint for row in fresh_safe}
        excluded = [
            row
            for row in fresh_before_prohibited
            if row.content_fingerprint not in fresh_safe_fingerprints
        ]
        prohibited_fresh_metrics[dataset] = {
            "fresh_safe_unique_rows_before_prohibited": len(fresh_before_prohibited),
            "excluded_by_prohibited_population_unique_rows": len(excluded),
            "fresh_eligible_unique_rows_after_prohibited": len(fresh_safe),
            "excluded_source_question_ids_sha256": _values_sha256([row.canonical_question_id for row in excluded]),
        }
        cross_format[dataset] = metrics

    legacy_training_prohibited_overlap: dict[str, Any] = {}
    safe_legacy_training_by_dataset: dict[str, list[_Question]] = {}
    for dataset in DATASETS:
        overlapping = [
            row
            for row in legacy_training_by_dataset[dataset]
            if row.content_fingerprint in prohibited_fingerprints
            or row.canonical_question_id in prohibited_ids
            or row.source_question_id in prohibited_ids
        ]
        legacy_training_prohibited_overlap[dataset] = {
            "row_count": len(overlapping),
            "source_question_ids_sha256": _values_sha256(
                [row.source_question_id for row in overlapping]
            ),
            "content_fingerprints_sha256": _values_sha256(
                [row.content_fingerprint for row in overlapping]
            ),
        }
        overlapping_ids = {row.source_question_id for row in overlapping}
        safe_legacy_training_by_dataset[dataset] = [
            row
            for row in legacy_training_by_dataset[dataset]
            if row.source_question_id not in overlapping_ids
        ]

    capacity = _capacity_report(
        legacy_training=safe_legacy_training_by_dataset,
        fresh_safe=fresh_safe_by_dataset,
    )
    audit_document = {
        "schema": CROSS_FORMAT_AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": CROSS_FORMAT_AUDIT_KIND,
        "normalization": REPOSITORY_RENDERED_MCQ_NORMALIZATION,
        "source": source_proof,
        "legacy": {
            "complete_population": legacy_all_proof,
            "act_max_training": legacy_training_proof,
            "existing_frozen_iid": existing_iid_proof,
            "partition": {
                "complete_population_counts_by_dataset": dict(LEGACY_POPULATION_COUNTS),
                "act_max_training_counts_by_dataset": dict(LEGACY_TRAINING_COUNTS),
                "existing_frozen_iid_counts_by_dataset": dict(EXISTING_FROZEN_IID_COUNTS),
                "act_max_training_plus_existing_iid_equals_complete_population": True,
            },
        },
        "prohibited_populations": prohibited_proof,
        "legacy_training_prohibited_overlap": legacy_training_prohibited_overlap,
        "cross_format_mapping": cross_format,
        "post_prohibited_filter": prohibited_fresh_metrics,
        "capacity_report": capacity,
        "generator_policy": {
            "provider": GENERATOR_PROVIDER,
            "model": GENERATOR_MODEL,
            "bias_type": BIAS_TYPE,
            "prompt_style": PROMPT_STYLE,
            "all_selected_questions_must_be_regenerated_homogeneously": True,
            "legacy_biasing_text_must_not_be_reused": True,
            "question_candidates_are_separate_from_generated_arguments": True,
        },
        "assertions": {
            "no_model_or_remote_service_called": True,
            "every_pinned_source_row_was_cross_format_audited": True,
            "legacy_and_fresh_questions_are_disjoint_by_normalized_content": True,
            "existing_frozen_iid_was_withheld_before_new_target_generation": True,
            "new_fresh_iid_reservation_is_separate_and_available_before_target_generation": True,
        },
    }
    return _DerivedInputs(
        audit_document=audit_document,
        legacy_training=legacy_training_by_dataset,
        safe_legacy_training=safe_legacy_training_by_dataset,
        fresh_safe=fresh_safe_by_dataset,
    )


def materialize_cross_format_audit(
    *,
    logiqa_source: str | Path,
    logiqa_manifest: str | Path,
    hellaswag_source: str | Path,
    hellaswag_manifest: str | Path,
    legacy_population: str | Path,
    legacy_training: str | Path,
    existing_iid: str | Path,
    prohibited_populations: Mapping[str, str | Path],
    output_dir: str | Path,
) -> PublishedAudit:
    """Publish a content-addressed audit before a question pool can be frozen."""

    derived = _derive_inputs(
        logiqa_source=logiqa_source,
        logiqa_manifest=logiqa_manifest,
        hellaswag_source=hellaswag_source,
        hellaswag_manifest=hellaswag_manifest,
        legacy_population=legacy_population,
        legacy_training=legacy_training,
        existing_iid=existing_iid,
        prohibited_populations=prohibited_populations,
    )
    payload = _canonical_json(derived.audit_document)
    digest = _sha256(payload)
    path = Path(output_dir).resolve() / AUDIT_FILENAME.format(sha256=digest)
    return PublishedAudit(path=path, sha256=digest, status=_publish_immutable(path, payload))


def _verify_cross_format_audit(
    *,
    cross_format_audit: str | Path,
    logiqa_source: str | Path,
    logiqa_manifest: str | Path,
    hellaswag_source: str | Path,
    hellaswag_manifest: str | Path,
    legacy_population: str | Path,
    legacy_training: str | Path,
    existing_iid: str | Path,
    prohibited_populations: Mapping[str, str | Path],
) -> _DerivedInputs:
    audit_path, actual, payload = _read_json(cross_format_audit, label="expanded ACT cross-format audit")
    expected = _derive_inputs(
        logiqa_source=logiqa_source,
        logiqa_manifest=logiqa_manifest,
        hellaswag_source=hellaswag_source,
        hellaswag_manifest=hellaswag_manifest,
        legacy_population=legacy_population,
        legacy_training=legacy_training,
        existing_iid=existing_iid,
        prohibited_populations=prohibited_populations,
    )
    expected_payload = _canonical_json(expected.audit_document)
    if actual != expected.audit_document or payload != expected_payload:
        raise ValueError("cross-format audit does not match the runtime-derived pinned-source mapping")
    expected_name = AUDIT_FILENAME.format(sha256=_sha256(expected_payload))
    if audit_path.name != expected_name:
        raise ValueError("cross-format audit filename is not content addressed")
    return expected


def verify_cross_format_audit(**kwargs: Any) -> dict[str, Any]:
    """Replay an audit and return it only when it exactly matches its inputs."""

    return _verify_cross_format_audit(**kwargs).audit_document


def _candidate_row(question: _Question, *, selection_role: str, selection_rank: int) -> dict[str, Any]:
    """Project a source question without any legacy or generated argument text."""

    return {
        "biased_option": _deterministic_biased_option(question.canonical_input, question.ground_truth, question.option_labels),
        "candidate_id": _sha256(f"{question.source_dataset}\0{question.content_fingerprint}".encode("utf-8")),
        "canonical_input": question.canonical_input,
        "content_fingerprint": question.content_fingerprint,
        "ground_truth": question.ground_truth,
        "question_id": question.canonical_question_id,
        "selection_rank": selection_rank,
        "selection_role": selection_role,
        "source_dataset": question.source_dataset,
        "source_origin": question.source_origin,
        "source_question_id": question.source_question_id,
        "source_row_index": question.source_row_index,
    }


def _shuffle(rows: Sequence[_Question], *, phase: str, dataset: str) -> list[_Question]:
    ordered = list(rows)
    random.Random(f"{ORDER_SEED}\0{phase}\0{dataset}").shuffle(ordered)
    return ordered


def _interleave(per_dataset_rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Round-robin storage while retaining the frozen rank inside each dataset."""

    output: list[dict[str, Any]] = []
    maximum = max((len(per_dataset_rows[dataset]) for dataset in DATASETS), default=0)
    for index in range(maximum):
        for dataset in DATASETS:
            values = per_dataset_rows[dataset]
            if index < len(values):
                output.append(dict(values[index]))
    return output


def _row_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("source_dataset", "")) for row in rows)
    unknown = sorted(set(counts) - set(DATASETS))
    if unknown:
        raise ValueError(f"candidate rows have unsupported dataset(s): {unknown}")
    return {dataset: counts.get(dataset, 0) for dataset in DATASETS}


def _validated_selection_size(*, candidate_source_mode: str, n_total: int | None) -> int | None:
    if candidate_source_mode not in CANDIDATE_SOURCE_MODES:
        raise ValueError(
            f"candidate_source_mode must be one of {CANDIDATE_SOURCE_MODES}; "
            f"got {candidate_source_mode!r}"
        )
    if n_total is None:
        if candidate_source_mode == "fresh_only":
            raise ValueError("fresh_only selection requires an explicit balanced n_total")
        return None
    if isinstance(n_total, bool) or not isinstance(n_total, int) or n_total < len(DATASETS):
        raise ValueError(f"n_total must be an integer >= {len(DATASETS)}")
    if n_total % len(DATASETS):
        raise ValueError(f"n_total must be divisible by {len(DATASETS)} for a balanced selection")
    return n_total


def _selection_from_derived(
    derived: _DerivedInputs,
    *,
    audit_identity: Mapping[str, Any],
    candidate_source_mode: str = "legacy_plus_fresh",
    n_total: int | None = None,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Reserve fresh IID first, then construct the question-only Gemma request."""

    n_total = _validated_selection_size(candidate_source_mode=candidate_source_mode, n_total=n_total)
    per_dataset_limit = n_total // len(DATASETS) if n_total is not None else None

    fresh_iid_by_dataset: dict[str, list[dict[str, Any]]] = {}
    candidates_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        fresh_ordered = _shuffle(derived.fresh_safe[dataset], phase="fresh-iid", dataset=dataset)
        if len(fresh_ordered) <= FRESH_IID_RESERVE_PER_DATASET:
            raise ValueError(f"{dataset} has insufficient fresh-safe rows for a new IID reservation")
        fresh_iid_by_dataset[dataset] = [_candidate_row(row, selection_role="fresh_iid_reserve", selection_rank=index) for index, row in enumerate(fresh_ordered[:FRESH_IID_RESERVE_PER_DATASET])]
        fresh_training_questions = fresh_ordered[FRESH_IID_RESERVE_PER_DATASET:]
        if candidate_source_mode == "fresh_only":
            if per_dataset_limit is not None and len(fresh_training_questions) < per_dataset_limit:
                raise ValueError(
                    f"{dataset} has only {len(fresh_training_questions)} eligible fresh_only "
                    f"questions after the fresh IID and prohibited-population exclusions; "
                    f"need {per_dataset_limit}"
                )
            training_questions = (
                fresh_training_questions[:per_dataset_limit]
                if per_dataset_limit is not None
                else fresh_training_questions
            )
            phase = "fresh-only-homogeneous-generation"
        else:
            safe_legacy = list(derived.safe_legacy_training[dataset])
            if per_dataset_limit is None:
                selected_legacy = safe_legacy
                selected_fresh = fresh_training_questions
            else:
                selected_legacy = safe_legacy[:per_dataset_limit]
                fresh_needed = per_dataset_limit - len(selected_legacy)
                if len(fresh_training_questions) < fresh_needed:
                    raise ValueError(
                        f"{dataset} has only {len(selected_legacy)} safe legacy and "
                        f"{len(fresh_training_questions)} fresh questions after exclusions; "
                        f"need {per_dataset_limit} total"
                    )
                selected_fresh = fresh_training_questions[:fresh_needed]
            training_questions = [*selected_legacy, *selected_fresh]
            phase = "safe-legacy-plus-fresh-homogeneous-generation"
        training_ordered = _shuffle(training_questions, phase=phase, dataset=dataset)
        candidates_by_dataset[dataset] = [_candidate_row(row, selection_role="homogeneous_gemma_generation_candidate", selection_rank=index) for index, row in enumerate(training_ordered)]

    fresh_iid_rows = _interleave(fresh_iid_by_dataset)
    candidate_rows = _interleave(candidates_by_dataset)
    iid_keys = {str(row["candidate_id"]) for row in fresh_iid_rows}
    candidate_keys = {str(row["candidate_id"]) for row in candidate_rows}
    if iid_keys & candidate_keys or len(iid_keys) != len(fresh_iid_rows) or len(candidate_keys) != len(candidate_rows):
        raise ValueError("fresh IID reserve and question candidates are not globally disjoint")
    forbidden_target_fields = {"biasing_text", "biased_messages", "unbiased_messages"}
    if any(forbidden_target_fields & set(row) for row in [*fresh_iid_rows, *candidate_rows]):
        raise AssertionError("question-only output unexpectedly contains an argument or prompt field")
    _require_equal(
        _row_counts(fresh_iid_rows),
        {dataset: FRESH_IID_RESERVE_PER_DATASET for dataset in DATASETS},
        label="fresh IID reserve counts",
    )
    if per_dataset_limit is None:
        expected_candidate_counts = derived.audit_document["capacity_report"]["with_additional_fresh_iid_100_per_dataset"]["question_candidate_counts_by_dataset"]
    else:
        expected_candidate_counts = {dataset: per_dataset_limit for dataset in DATASETS}
    _require_equal(_row_counts(candidate_rows), expected_candidate_counts, label="question candidate counts")
    if candidate_source_mode == "fresh_only" and any(
        row["source_origin"] != "pinned_hf_train_question" for row in candidate_rows
    ):
        raise AssertionError("fresh_only selection unexpectedly retained a legacy question")
    origin_counts = Counter(str(row["source_origin"]) for row in candidate_rows)
    origin_counts_by_dataset = {
        dataset: dict(
            Counter(
                str(row["source_origin"])
                for row in candidate_rows
                if row["source_dataset"] == dataset
            )
        )
        for dataset in DATASETS
    }
    if candidate_source_mode == "legacy_plus_fresh" and n_total == PLANNED_TRAINING_TOTAL:
        expected_origins = {
            "legacy_act_max_question": PLANNED_SAFE_LEGACY_PER_DATASET,
            "pinned_hf_train_question": (
                PLANNED_TRAINING_TOTAL // len(DATASETS)
                - PLANNED_SAFE_LEGACY_PER_DATASET
            ),
        }
        for dataset in DATASETS:
            _require_equal(
                origin_counts_by_dataset[dataset],
                expected_origins,
                label=f"{dataset} planned training source composition",
            )

    fresh_iid_payload = _canonical_jsonl(fresh_iid_rows)
    candidate_payload = _canonical_jsonl(candidate_rows)
    candidate_ids = [str(row["candidate_id"]) for row in candidate_rows]
    document = {
        "schema": SELECTION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "kind": SELECTION_KIND,
        "cross_format_audit": dict(audit_identity),
        "prompt_style": PROMPT_STYLE,
        "bias_type": BIAS_TYPE,
        "canonical_pair_transform": CANONICAL_PAIR_TRANSFORM,
        "order_seed": ORDER_SEED,
        "candidate_source_mode": candidate_source_mode,
        "requested_balanced_n_total": n_total,
        "fresh_iid_reservation": {
            "row_count": len(fresh_iid_rows),
            "counts_by_dataset": _row_counts(fresh_iid_rows),
            "candidate_ids_sha256": _values_sha256([str(row["candidate_id"]) for row in fresh_iid_rows]),
            "selection": "first 100 fresh-safe rows per dataset after deterministic shuffle, before argument generation",
        },
        "question_candidates": {
            "row_count": len(candidate_rows),
            "counts_by_dataset": _row_counts(candidate_rows),
            "counts_by_source_origin": dict(origin_counts),
            "counts_by_dataset_and_source_origin": origin_counts_by_dataset,
            "candidate_ids_sha256": _values_sha256(candidate_ids),
            "selection": (
                "fresh public-train questions disjoint from the complete legacy population, "
                "after new IID and prohibited-population exclusions; deterministic per-dataset "
                "shuffle, balanced prefix, then round-robin storage"
                if candidate_source_mode == "fresh_only"
                else "all safe legacy ACT-Max questions first, excluding every prohibited population, "
                "then exactly enough fresh-safe rows after the new IID reservation; deterministic "
                "per-dataset shuffle of that fixed composition then round-robin storage"
            ),
            "source_origins": {
                "legacy_act_max_question": "question retained, legacy argument deliberately discarded",
                "pinned_hf_train_question": "new question from the pinned public source",
            },
        },
        "capacity_report": derived.audit_document["capacity_report"],
        "homogeneous_argument_generation_contract": {
            "provider": GENERATOR_PROVIDER,
            "model": GENERATOR_MODEL,
            "required_for_every_candidate_id_sha256": _values_sha256(candidate_ids),
            "bias_type": BIAS_TYPE,
            "prompt_style": PROMPT_STYLE,
            "must_generate_one_new_wrong_argument_for_every_selected_question": True,
            "legacy_biasing_text_is_not_a_valid_input_or_output": True,
            "training_selection_rule": {
                "formula": "N = floor(successful_logiqa / optimizer_granularity) * optimizer_granularity",
                "optimizer_granularity": 1,
                "hellaswag_requirement": "generate in frozen candidate order until at least N successful homogeneous Gemma targets exist; use its first N successes",
            },
        },
        "assertions": {
            "fresh_iid_reserved_before_wrong_argument_generation": True,
            "existing_frozen_iid_remains_withheld": True,
            "legacy_and_fresh_question_provenance_remain_labelled": True,
            "fresh_only_excludes_complete_legacy_population": candidate_source_mode == "fresh_only",
            "only_questions_not_arguments_are_published": True,
            "no_model_or_remote_service_called": True,
            "planned_n8192_source_composition_enforced": (
                candidate_source_mode != "legacy_plus_fresh"
                or n_total != PLANNED_TRAINING_TOTAL
                or all(
                    origin_counts_by_dataset[dataset]
                    == {
                        "legacy_act_max_question": PLANNED_SAFE_LEGACY_PER_DATASET,
                        "pinned_hf_train_question": (
                            PLANNED_TRAINING_TOTAL // len(DATASETS)
                            - PLANNED_SAFE_LEGACY_PER_DATASET
                        ),
                    }
                    for dataset in DATASETS
                )
            ),
        },
    }
    return fresh_iid_payload, candidate_payload, document


def materialize_expanded_act_selection(
    *,
    logiqa_source: str | Path,
    logiqa_manifest: str | Path,
    hellaswag_source: str | Path,
    hellaswag_manifest: str | Path,
    legacy_population: str | Path,
    legacy_training: str | Path,
    existing_iid: str | Path,
    prohibited_populations: Mapping[str, str | Path],
    cross_format_audit: str | Path,
    output_dir: str | Path,
    candidate_source_mode: str = "legacy_plus_fresh",
    n_total: int | None = None,
) -> PublishedSelection:
    """Publish the extra IID reserve and a homogeneous-Gemma question request.

    The prior audit is replayed, rather than merely trusted.  Consequently a
    changed public snapshot, legacy population, evaluation exclusion, or audit
    document cannot be used to silently produce a different selection.
    """

    derived = _verify_cross_format_audit(
        cross_format_audit=cross_format_audit,
        logiqa_source=logiqa_source,
        logiqa_manifest=logiqa_manifest,
        hellaswag_source=hellaswag_source,
        hellaswag_manifest=hellaswag_manifest,
        legacy_population=legacy_population,
        legacy_training=legacy_training,
        existing_iid=existing_iid,
        prohibited_populations=prohibited_populations,
    )
    audit_path = _regular_file(cross_format_audit, label="expanded ACT cross-format audit")
    audit_payload = audit_path.read_bytes()
    audit_identity = _portable_file_identity(audit_path, audit_payload, row_count=1)
    fresh_iid_payload, candidate_payload, document = _selection_from_derived(
        derived,
        audit_identity=audit_identity,
        candidate_source_mode=candidate_source_mode,
        n_total=n_total,
    )
    fresh_iid_sha = _sha256(fresh_iid_payload)
    candidate_sha = _sha256(candidate_payload)
    output = Path(output_dir).resolve()
    fresh_iid_path = output / FRESH_IID_FILENAME.format(sha256=fresh_iid_sha)
    candidate_path = output / CANDIDATE_FILENAME.format(sha256=candidate_sha)
    document = {
        **document,
        "fresh_iid_reservation": {
            **document["fresh_iid_reservation"],
            "filename": fresh_iid_path.name,
            "content_sha256": fresh_iid_sha,
        },
        "question_candidates": {
            **document["question_candidates"],
            "filename": candidate_path.name,
            "content_sha256": candidate_sha,
        },
    }
    manifest_payload = _canonical_json(document)
    manifest_sha = _sha256(manifest_payload)
    manifest_path = output / SELECTION_MANIFEST_FILENAME.format(sha256=manifest_sha)
    fresh_iid_status = _publish_immutable(fresh_iid_path, fresh_iid_payload)
    candidate_status = _publish_immutable(candidate_path, candidate_payload)
    manifest_status = _publish_immutable(manifest_path, manifest_payload)
    return PublishedSelection(
        fresh_iid_path=fresh_iid_path,
        candidate_path=candidate_path,
        manifest_path=manifest_path,
        fresh_iid_sha256=fresh_iid_sha,
        candidate_sha256=candidate_sha,
        manifest_sha256=manifest_sha,
        fresh_iid_status=fresh_iid_status,
        candidate_status=candidate_status,
        manifest_status=manifest_status,
    )


def verify_expanded_act_selection(
    *,
    logiqa_source: str | Path,
    logiqa_manifest: str | Path,
    hellaswag_source: str | Path,
    hellaswag_manifest: str | Path,
    legacy_population: str | Path,
    legacy_training: str | Path,
    existing_iid: str | Path,
    prohibited_populations: Mapping[str, str | Path],
    cross_format_audit: str | Path,
    fresh_iid_selection: str | Path,
    candidate_selection: str | Path,
    selection_manifest: str | Path,
    candidate_source_mode: str = "legacy_plus_fresh",
    n_total: int | None = None,
) -> dict[str, Any]:
    """Replay the audit and selection proof; reject substituted artifacts."""

    derived = _verify_cross_format_audit(
        cross_format_audit=cross_format_audit,
        logiqa_source=logiqa_source,
        logiqa_manifest=logiqa_manifest,
        hellaswag_source=hellaswag_source,
        hellaswag_manifest=hellaswag_manifest,
        legacy_population=legacy_population,
        legacy_training=legacy_training,
        existing_iid=existing_iid,
        prohibited_populations=prohibited_populations,
    )
    audit_path = _regular_file(cross_format_audit, label="expanded ACT cross-format audit")
    audit_payload = audit_path.read_bytes()
    expected_fresh_iid, expected_candidates, expected_document = _selection_from_derived(
        derived,
        audit_identity=_portable_file_identity(audit_path, audit_payload, row_count=1),
        candidate_source_mode=candidate_source_mode,
        n_total=n_total,
    )
    fresh_iid_path = _regular_file(fresh_iid_selection, label="fresh IID selection")
    candidate_path = _regular_file(candidate_selection, label="question candidate selection")
    manifest_path, actual_document, manifest_payload = _read_json(selection_manifest, label="expanded ACT selection manifest")
    if fresh_iid_path.read_bytes() != expected_fresh_iid:
        raise ValueError("fresh IID selection bytes do not match the replayed selection proof")
    if candidate_path.read_bytes() != expected_candidates:
        raise ValueError("question candidate selection bytes do not match the replayed selection proof")
    fresh_sha = _sha256(expected_fresh_iid)
    candidate_sha = _sha256(expected_candidates)
    expected_document = {
        **expected_document,
        "fresh_iid_reservation": {
            **expected_document["fresh_iid_reservation"],
            "filename": FRESH_IID_FILENAME.format(sha256=fresh_sha),
            "content_sha256": fresh_sha,
        },
        "question_candidates": {
            **expected_document["question_candidates"],
            "filename": CANDIDATE_FILENAME.format(sha256=candidate_sha),
            "content_sha256": candidate_sha,
        },
    }
    expected_manifest_payload = _canonical_json(expected_document)
    if actual_document != expected_document or manifest_payload != expected_manifest_payload:
        raise ValueError("expanded ACT selection manifest does not match the replayed selection proof")
    if fresh_iid_path.name != expected_document["fresh_iid_reservation"]["filename"]:
        raise ValueError("fresh IID selection filename is not content addressed")
    if candidate_path.name != expected_document["question_candidates"]["filename"]:
        raise ValueError("question candidate selection filename is not content addressed")
    expected_manifest_name = SELECTION_MANIFEST_FILENAME.format(sha256=_sha256(expected_manifest_payload))
    if manifest_path.name != expected_manifest_name:
        raise ValueError("expanded ACT selection manifest filename is not content addressed")
    return actual_document


def final_training_count(successful_logiqa: int, *, optimizer_granularity: int = 1) -> int:
    """Return the balanced training count N determined by LogiQA successes."""

    if not isinstance(successful_logiqa, int) or isinstance(successful_logiqa, bool) or successful_logiqa < 0:
        raise ValueError("successful_logiqa must be a non-negative integer")
    if not isinstance(optimizer_granularity, int) or isinstance(optimizer_granularity, bool) or optimizer_granularity < 1:
        raise ValueError("optimizer_granularity must be a positive integer")
    return successful_logiqa // optimizer_granularity * optimizer_granularity


def require_hellaswag_capacity(successful_hellaswag: int, *, target_count: int) -> None:
    """Fail before training if HellaSwag has fewer homogeneous targets than N."""

    if not isinstance(successful_hellaswag, int) or isinstance(successful_hellaswag, bool) or successful_hellaswag < 0:
        raise ValueError("successful_hellaswag must be a non-negative integer")
    if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count < 0:
        raise ValueError("target_count must be a non-negative integer")
    if successful_hellaswag < target_count:
        raise ValueError(f"need at least {target_count} successful HellaSwag targets for balanced ACT training; got {successful_hellaswag}")


def _parse_named_paths(values: Sequence[str], *, option: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} must use NAME=PATH")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path or name in parsed:
            raise ValueError(f"{option} names must be unique non-empty NAME=PATH pairs")
        parsed[name] = Path(raw_path)
    return parsed


def _add_shared_inputs(parser: argparse.ArgumentParser, *, include_audit: bool) -> None:
    parser.add_argument("--logiqa-source", type=Path, required=True)
    parser.add_argument("--logiqa-manifest", type=Path, required=True)
    parser.add_argument("--hellaswag-source", type=Path, required=True)
    parser.add_argument("--hellaswag-manifest", type=Path, required=True)
    parser.add_argument("--legacy-population", type=Path, required=True)
    parser.add_argument("--legacy-training", type=Path, required=True)
    parser.add_argument("--existing-iid", type=Path, required=True)
    parser.add_argument("--prohibited", action="append", default=[], metavar="NAME=PATH")
    if include_audit:
        parser.add_argument("--cross-format-audit", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Offline expanded-ACT source, audit, and question-pool freezer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser("freeze-source", help="freeze one staged pinned public source")
    source_parser.add_argument("--dataset", choices=DATASETS, required=True)
    source_parser.add_argument("--raw-source", type=Path, required=True)
    source_parser.add_argument("--output-dir", type=Path, required=True)

    audit_parser = subparsers.add_parser("audit", help="publish a cross-format legacy/public-source audit")
    _add_shared_inputs(audit_parser, include_audit=False)
    audit_parser.add_argument("--output-dir", type=Path, required=True)

    materialize_parser = subparsers.add_parser("materialize", help="reserve fresh IID and publish homogeneous-Gemma question candidates")
    _add_shared_inputs(materialize_parser, include_audit=True)
    materialize_parser.add_argument("--candidate-source-mode", choices=CANDIDATE_SOURCE_MODES, default="legacy_plus_fresh")
    materialize_parser.add_argument("--n-total", type=int)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify", help="replay the audit and selection proof")
    _add_shared_inputs(verify_parser, include_audit=True)
    verify_parser.add_argument("--fresh-iid-selection", type=Path, required=True)
    verify_parser.add_argument("--candidate-selection", type=Path, required=True)
    verify_parser.add_argument("--selection-manifest", type=Path, required=True)
    verify_parser.add_argument("--candidate-source-mode", choices=CANDIDATE_SOURCE_MODES, default="legacy_plus_fresh")
    verify_parser.add_argument("--n-total", type=int)

    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-source":
            result = materialize_source_snapshot(dataset=args.dataset, raw_source=args.raw_source, output_dir=args.output_dir)
            print(json.dumps({"data": str(result.data_path), "manifest": str(result.manifest_path), "verified": True}))
            return
        inputs = {
            "logiqa_source": args.logiqa_source,
            "logiqa_manifest": args.logiqa_manifest,
            "hellaswag_source": args.hellaswag_source,
            "hellaswag_manifest": args.hellaswag_manifest,
            "legacy_population": args.legacy_population,
            "legacy_training": args.legacy_training,
            "existing_iid": args.existing_iid,
            "prohibited_populations": _parse_named_paths(args.prohibited, option="--prohibited"),
        }
        if args.command == "audit":
            result = materialize_cross_format_audit(output_dir=args.output_dir, **inputs)
            print(json.dumps({"audit": str(result.path), "sha256": result.sha256, "verified": True}, sort_keys=True))
            return
        inputs["cross_format_audit"] = args.cross_format_audit
        if args.command == "materialize":
            result = materialize_expanded_act_selection(
                output_dir=args.output_dir,
                candidate_source_mode=args.candidate_source_mode,
                n_total=args.n_total,
                **inputs,
            )
            print(
                json.dumps(
                    {
                        "fresh_iid": str(result.fresh_iid_path),
                        "question_candidates": str(result.candidate_path),
                        "manifest": str(result.manifest_path),
                        "verified": True,
                    },
                    sort_keys=True,
                )
            )
            return
        document = verify_expanded_act_selection(
            fresh_iid_selection=args.fresh_iid_selection,
            candidate_selection=args.candidate_selection,
            selection_manifest=args.selection_manifest,
            candidate_source_mode=args.candidate_source_mode,
            n_total=args.n_total,
            **inputs,
        )
        print(
            json.dumps(
                {
                    "fresh_iid_rows": document["fresh_iid_reservation"]["row_count"],
                    "candidate_counts": document["question_candidates"]["counts_by_dataset"],
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
    "CANDIDATE_SOURCE_MODES",
    "CORE_CROSS_FORMAT_EXPECTATIONS",
    "DATASETS",
    "FRESH_IID_RESERVE_PER_DATASET",
    "GENERATOR_MODEL",
    "LEGACY_POPULATION_COUNTS",
    "LEGACY_TRAINING_COUNTS",
    "ORDER_SEED",
    "PublishedAudit",
    "PublishedSelection",
    "PublishedSource",
    "SOURCE_SPECS",
    "final_training_count",
    "materialize_cross_format_audit",
    "materialize_expanded_act_selection",
    "materialize_source_snapshot",
    "require_hellaswag_capacity",
    "verify_cross_format_audit",
    "verify_expanded_act_selection",
]
