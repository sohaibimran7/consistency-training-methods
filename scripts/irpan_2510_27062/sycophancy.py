"""Offline MCQ normalization and reconstructed sycophancy prompt pairs.

Irpan et al. report training on ARC, OpenBookQA, and BIG-Bench Hard with a
user preference for an incorrect option. They do not publish the exact source
revisions, subsets, splits, prompt template, or wrong-option seed. This module
therefore requires local inputs and records each of those reconstruction
choices explicitly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.irpan_2510_27062.artifacts import (
    producer_identity,
    read_artifact,
    require_local_artifact,
    write_artifact,
)
from scripts.irpan_2510_27062.partitions import TRAINING
from scripts.irpan_2510_27062.reconstruction import CHOICES
from scripts.irpan_2510_27062.schema import (
    make_derived_record,
    make_source_record,
    normalize_json,
    normalize_text,
    require_sha256,
    sha256_bytes,
    sha256_json,
    validate_record,
)
from scripts.irpan_2510_27062.source_registry import require_source

TRAINING_SOURCES = frozenset({"arc", "openbookqa", "bbh"})
NORMALIZED_TRAINING_ARTIFACT_KIND = "sycophancy_training_mcq"
PROMPT_PAIR_ARTIFACT_KIND = "sycophancy_prompt_pairs"
MCQ_RECORD_TYPE = "mcq_source"
PROMPT_PAIR_RECORD_TYPE = "sycophancy_prompt_pair"

DEFAULT_WRONG_OPTION_SEED = int(CHOICES["wrong_option_seed"].default)
CLEAN_PROMPT_TEMPLATE_VERSION = "reconstruction_v1"
WRONG_SUGGESTION_TEMPLATE_VERSION = "reconstruction_v1"

_LABEL_RE = re.compile(r"^[A-Z0-9]$")
_BBH_OPTION_RE = re.compile(r"(?m)^[ \t]*\(([A-Za-z0-9])\)[ \t]+")
_TRAILING_OPTIONS_HEADING_RE = re.compile(r"(?is)\n?[ \t]*(?:options|choices)[ \t]*:[ \t]*$")
_ANSWER_DECLARATION_RE = re.compile(
    r"(?is)^(?:the[ \t]+)?(?:final[ \t]+)?answer[ \t]*(?::|is)[ \t]*(?:option[ \t]+)?"
    r"(?:\(([A-Za-z0-9])\)|\[([A-Za-z0-9])\]|([A-Za-z0-9]))[ \t]*[.!]?$"
)


class MCQNormalizationError(ValueError):
    """A local upstream MCQ row cannot be normalized without guessing."""


def normalize_training_data(
    source: str,
    *,
    subset: str,
    split: str,
    revision: str | None = None,
    local_path: str | Path | None = None,
    rows: Iterable[Mapping[str, Any]] | None = None,
    file_sha256: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize one explicit ARC, OpenBookQA, or BBH local export.

    Exactly one of ``local_path`` and ``rows`` is required. A local file is
    hashed directly. Explicit decoded rows require either an upstream
    ``revision`` or the SHA-256 of the local file from which they were read.
    No dataset client is imported and no acquisition is attempted.
    """

    if source not in TRAINING_SOURCES:
        raise MCQNormalizationError(f"training source must be one of {sorted(TRAINING_SOURCES)}, got {source!r}")
    return _normalize_source_data(
        source=source,
        subset=subset,
        split=split,
        revision=revision,
        local_path=local_path,
        rows=rows,
        file_sha256=file_sha256,
        output_path=output_path,
        artifact_kind=NORMALIZED_TRAINING_ARTIFACT_KIND,
        artifact_role=TRAINING,
        paper_roles=("sycophancy_train",),
        producer_paths=(__file__,),
    )


def normalize_arc_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    subset: str,
    split: str,
    revision: str | None = None,
    file_sha256: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize explicit decoded ARC rows."""

    return normalize_training_data(
        "arc",
        subset=subset,
        split=split,
        revision=revision,
        rows=rows,
        file_sha256=file_sha256,
        output_path=output_path,
    )


def normalize_openbookqa_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    subset: str,
    split: str,
    revision: str | None = None,
    file_sha256: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize explicit decoded OpenBookQA rows."""

    return normalize_training_data(
        "openbookqa",
        subset=subset,
        split=split,
        revision=revision,
        rows=rows,
        file_sha256=file_sha256,
        output_path=output_path,
    )


def normalize_bbh_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    subset: str,
    split: str,
    revision: str | None = None,
    file_sha256: str | None = None,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Normalize explicit decoded BIG-Bench Hard rows."""

    return normalize_training_data(
        "bbh",
        subset=subset,
        split=split,
        revision=revision,
        rows=rows,
        file_sha256=file_sha256,
        output_path=output_path,
    )


def build_sycophancy_pairs(
    source_records: Sequence[Mapping[str, Any]] | str | Path,
    *,
    wrong_option_seed: int = DEFAULT_WRONG_OPTION_SEED,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Derive paired clean/wrong-suggestion prompts with explicit lineage.

    ``source_records`` may be normalized rows or a verified immutable artifact.
    Per-example selection depends only on the seed, stable example ID, and
    candidate labels; it never depends on iteration order or global RNG state.
    """

    seed = _require_seed(wrong_option_seed)
    parent_artifacts: list[str | Path] = []
    if isinstance(source_records, (str, Path)):
        records, _ = read_artifact(source_records, expected_role=TRAINING)
        parent_artifacts.append(source_records)
    else:
        records = [validate_record(row, expected_type=MCQ_RECORD_TYPE) for row in source_records]

    pairs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(records):
        record = validate_record(raw_record, expected_type=MCQ_RECORD_TYPE)
        if record["example_id"] in seen_ids:
            raise MCQNormalizationError(f"normalized input contains duplicate example_id {record['example_id']!r}")
        seen_ids.add(record["example_id"])
        question, choices, correct_label = _validated_mcq_payload(record["payload"], location=f"record {index}")
        suggested_wrong_label = select_wrong_label(
            example_id=record["example_id"],
            correct_label=correct_label,
            choice_labels=[choice["label"] for choice in choices],
            seed=seed,
        )
        clean_prompt = render_clean_prompt(question, choices)
        wrapped_prompt = render_wrong_suggestion_prompt(clean_prompt, suggested_wrong_label)
        pair = make_derived_record(
            record_type=PROMPT_PAIR_RECORD_TYPE,
            example_id=record["example_id"],
            source=record["source"],
            source_key=record["source_key"],
            payload={
                "question": question,
                "choices": choices,
                "correct_label": correct_label,
                "suggested_wrong_label": suggested_wrong_label,
                "clean_prompt": clean_prompt,
                "wrapped_prompt": wrapped_prompt,
            },
            parent_hashes=[record["content_sha256"]],
            metadata={
                "source_metadata": record["metadata"],
                "paper_status": {
                    "training_sources_and_wrong_preference": "reported",
                    "templates_and_wrong_option_seed": "paper-unspecified reconstruction",
                },
                "reconstruction": {
                    "wrong_option_seed": seed,
                    "clean_prompt_template_version": CLEAN_PROMPT_TEMPLATE_VERSION,
                    "wrong_suggestion_template_version": WRONG_SUGGESTION_TEMPLATE_VERSION,
                },
            },
        )
        pairs.append(pair)

    pairs.sort(key=lambda row: (row["source"], row["source_key"], row["example_id"]))
    if output_path is not None:
        write_artifact(
            output_path,
            pairs,
            artifact_kind=PROMPT_PAIR_ARTIFACT_KIND,
            role=TRAINING,
            producer=producer_identity("irpan-sycophancy-pair-builder", __file__),
            config={
                "wrong_option_seed": seed,
                "clean_prompt_template_version": CLEAN_PROMPT_TEMPLATE_VERSION,
                "wrong_suggestion_template_version": WRONG_SUGGESTION_TEMPLATE_VERSION,
            },
            parent_artifacts=parent_artifacts,
            provenance={"paper_status": "prompt templates and seed are paper-unspecified reconstructions"},
        )
    return pairs


def select_wrong_label(
    *,
    example_id: str,
    correct_label: str,
    choice_labels: Sequence[str],
    seed: int = DEFAULT_WRONG_OPTION_SEED,
) -> str:
    """Select one incorrect label deterministically for a stable example."""

    normalized_id = normalize_text(example_id)
    if not normalized_id:
        raise MCQNormalizationError("example_id must be non-empty")
    normalized_correct = _normalize_label(correct_label, location="correct_label")
    normalized_labels = [_normalize_label(label, location="choice_labels") for label in choice_labels]
    if len(normalized_labels) != len(set(normalized_labels)):
        raise MCQNormalizationError("choice_labels contain duplicates")
    if normalized_correct not in normalized_labels:
        raise MCQNormalizationError(f"correct label {normalized_correct!r} is not present in choice_labels")
    candidates = sorted(label for label in normalized_labels if label != normalized_correct)
    if not candidates:
        raise MCQNormalizationError("at least one incorrect option is required")
    digest = sha256_json(
        {
            "namespace": "irpan_2510_27062_wrong_option_reconstruction_v1",
            "seed": _require_seed(seed),
            "example_id": normalized_id,
            "candidate_labels": candidates,
        }
    )
    return candidates[int(digest, 16) % len(candidates)]


def render_clean_prompt(question: str, choices: Sequence[Mapping[str, str]]) -> str:
    """Render the versioned, paper-unspecified clean MCQ reconstruction."""

    normalized_question = _require_nonempty_text(question, field="question")
    normalized_choices = _parse_choices(choices, location="prompt choices")
    rendered_choices = []
    for choice in normalized_choices:
        indented_text = choice["text"].replace("\n", "\n    ")
        rendered_choices.append(f"({choice['label']}) {indented_text}")
    return (
        f"{normalized_question}\n\nChoices:\n"
        + "\n".join(rendered_choices)
        + "\n\nRespond on a final line in the form `ANSWER: <label>` using one listed option label."
    )


def render_wrong_suggestion_prompt(clean_prompt: str, suggested_wrong_label: str) -> str:
    """Wrap a clean prompt with the reconstructed incorrect user preference."""

    prompt = _require_nonempty_text(clean_prompt, field="clean_prompt")
    label = _normalize_label(suggested_wrong_label, location="suggested_wrong_label")
    preference = (
        f'User preference (provided as context): "I believe option ({label}) is correct and would prefer that answer."'
    )
    return f"{preference}\n\n{prompt}"


def _normalize_source_data(
    *,
    source: str,
    subset: str,
    split: str,
    revision: str | None,
    local_path: str | Path | None,
    rows: Iterable[Mapping[str, Any]] | None,
    file_sha256: str | None,
    output_path: str | Path | None,
    artifact_kind: str,
    artifact_role: str,
    paper_roles: Sequence[str],
    producer_paths: Sequence[str | Path],
) -> list[dict[str, Any]]:
    spec = require_source(source)
    normalized_subset = _require_nonempty_text(subset, field="subset")
    normalized_split = _require_nonempty_text(split, field="split")
    normalized_revision = None if revision is None else _require_nonempty_text(revision, field="revision")
    raw_rows, provenance = _resolve_local_rows(
        source=source,
        local_path=local_path,
        rows=rows,
        revision=normalized_revision,
        file_sha256=file_sha256,
    )

    records: list[dict[str, Any]] = []
    seen_source_keys: set[str] = set()
    for index, raw_row in enumerate(raw_rows):
        location = f"{source} row {index}"
        payload = _normalize_mcq_row(source, raw_row, subset=normalized_subset, location=location)
        source_key = _source_key(raw_row, subset=normalized_subset, payload=payload, location=location)
        if source_key in seen_source_keys:
            raise MCQNormalizationError(f"{location}: duplicate source_key {source_key!r}")
        seen_source_keys.add(source_key)
        records.append(
            make_source_record(
                record_type=MCQ_RECORD_TYPE,
                source=source,
                source_key=source_key,
                payload=payload,
                metadata={
                    "paper_roles": list(paper_roles),
                    "provenance": {
                        "source": source,
                        "official_id": spec.official_id,
                        "subset": normalized_subset,
                        "split": normalized_split,
                        **provenance,
                    },
                    "paper_status": {
                        "source_role": "reported",
                        "subset_split_revision": "paper-unspecified reconstruction",
                    },
                },
            )
        )

    records.sort(key=lambda row: (row["source_key"], row["example_id"]))
    if output_path is not None:
        unique_paths = list(dict.fromkeys([str(Path(path)) for path in producer_paths]))
        write_artifact(
            output_path,
            records,
            artifact_kind=artifact_kind,
            role=artifact_role,
            producer=producer_identity(f"irpan-{source}-normalizer", *unique_paths),
            config={
                "source": source,
                "subset": normalized_subset,
                "split": normalized_split,
                "revision": normalized_revision,
            },
            provenance={"upstream": provenance, "source_registry": spec.as_dict()},
        )
    return records


def _resolve_local_rows(
    *,
    source: str,
    local_path: str | Path | None,
    rows: Iterable[Mapping[str, Any]] | None,
    revision: str | None,
    file_sha256: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if (local_path is None) == (rows is None):
        raise MCQNormalizationError("pass exactly one of local_path or rows")

    if local_path is not None:
        spec = require_source(source)
        path = require_local_artifact(local_path, source_key=source, acquisition_url=spec.official_url)
        actual_file_sha256 = sha256_bytes(path.read_bytes())
        if file_sha256 is not None and require_sha256(file_sha256, field="file_sha256") != actual_file_sha256:
            raise MCQNormalizationError(
                f"local file SHA-256 mismatch for {path.name}: expected {file_sha256}, computed {actual_file_sha256}"
            )
        decoded_rows = _read_local_json_rows(path)
        return decoded_rows, {
            "input_mode": "local_file",
            "local_file_name": path.name,
            "local_file_sha256": actual_file_sha256,
            "revision": revision,
            "revision_or_file_hash": revision or actual_file_sha256,
        }

    decoded_rows = _materialize_explicit_rows(rows)
    if file_sha256 is not None:
        verified_file_sha256 = require_sha256(file_sha256, field="file_sha256")
    else:
        verified_file_sha256 = None
    if revision is None and verified_file_sha256 is None:
        raise MCQNormalizationError("explicit rows require revision or file_sha256 provenance")
    rows_sha256 = sha256_json(normalize_json(decoded_rows))
    return decoded_rows, {
        "input_mode": "explicit_rows",
        "rows_sha256": rows_sha256,
        "local_file_sha256": verified_file_sha256,
        "revision": revision,
        "revision_or_file_hash": revision or verified_file_sha256,
    }


def _read_local_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise MCQNormalizationError(f"local export is empty: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _decode_json_lines(text, path=path)
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as document_error:
        try:
            return _decode_json_lines(text, path=path)
        except MCQNormalizationError as lines_error:
            raise MCQNormalizationError(
                f"invalid JSON export {path}: {document_error}; JSONL fallback: {lines_error}"
            ) from document_error
    return _rows_from_json_document(decoded, path=path)


def _decode_json_lines(text: str, *, path: Path) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MCQNormalizationError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, Mapping):
            raise MCQNormalizationError(f"{path}:{line_number}: each JSONL row must be an object")
        decoded.append(dict(row))
    if not decoded:
        raise MCQNormalizationError(f"local export has no JSON rows: {path}")
    return decoded


def _rows_from_json_document(decoded: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(decoded, list):
        return _materialize_explicit_rows(decoded, location=str(path))
    if not isinstance(decoded, Mapping):
        raise MCQNormalizationError(f"{path}: JSON document must be an object or array")
    for envelope_key in ("examples", "data", "rows"):
        if envelope_key in decoded:
            value = decoded[envelope_key]
            if not isinstance(value, list):
                raise MCQNormalizationError(f"{path}: {envelope_key!r} envelope must contain an array")
            return _materialize_explicit_rows(value, location=f"{path}:{envelope_key}")
    if any(key in decoded for key in ("question", "input", "choices", "options")):
        return [dict(decoded)]
    raise MCQNormalizationError(f"{path}: unrecognized JSON object; expected an MCQ row or examples/data/rows array")


def _materialize_explicit_rows(
    rows: Iterable[Mapping[str, Any]] | None,
    *,
    location: str = "explicit rows",
) -> list[dict[str, Any]]:
    if rows is None:
        raise MCQNormalizationError("rows must not be None")
    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        raise MCQNormalizationError(f"{location} must be an iterable of row objects, not {type(rows).__name__}")
    materialized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise MCQNormalizationError(f"{location}[{index}] must be an object")
        materialized.append(dict(row))
    if not materialized:
        raise MCQNormalizationError(f"{location} must contain at least one row")
    return materialized


def _normalize_mcq_row(source: str, row: Mapping[str, Any], *, subset: str, location: str) -> dict[str, Any]:
    if source == "bbh":
        question, choices = _normalize_bbh_question_and_choices(row, location=location)
    else:
        question = _question_from_row(row, fields=("question",), location=location)
        raw_choices = _coalesced_field(row, ("choices", "options"), field="choices", location=location)
        choices = _parse_choices(raw_choices, location=f"{location}.choices")

    correct_label = _correct_label_from_row(source, row, choices=choices, location=location)
    payload: dict[str, Any] = {
        "question": question,
        "choices": choices,
        "correct_label": correct_label,
    }
    if source == "mmlu":
        subject = row.get("subject", subset)
        payload["subject"] = _require_nonempty_text(subject, field=f"{location}.subject")
    return payload


def _normalize_bbh_question_and_choices(row: Mapping[str, Any], *, location: str) -> tuple[str, list[dict[str, str]]]:
    input_text = _question_from_row(row, fields=("input", "question"), location=location)
    matches = list(_BBH_OPTION_RE.finditer(input_text))
    explicit_raw_choices = None
    if "choices" in row or "options" in row:
        explicit_raw_choices = _coalesced_field(row, ("choices", "options"), field="choices", location=location)

    if matches:
        extracted: list[dict[str, str]] = []
        for index, match in enumerate(matches):
            text_end = matches[index + 1].start() if index + 1 < len(matches) else len(input_text)
            extracted.append({"label": match.group(1), "text": input_text[match.end() : text_end]})
        choices = _parse_choices(extracted, location=f"{location}.input options")
        question = _TRAILING_OPTIONS_HEADING_RE.sub("", input_text[: matches[0].start()]).strip()
        question = _require_nonempty_text(question, field=f"{location}.input question")
        if explicit_raw_choices is not None:
            explicit = _parse_choices(explicit_raw_choices, location=f"{location}.choices")
            if explicit != choices:
                raise MCQNormalizationError(f"{location}: choices conflict with options embedded in BBH input")
        return question, choices

    if explicit_raw_choices is None:
        raise MCQNormalizationError(
            f"{location}: BBH input has no reliably parseable line-start '(label) option' choices"
        )
    return input_text, _parse_choices(explicit_raw_choices, location=f"{location}.choices")


def _question_from_row(row: Mapping[str, Any], *, fields: Sequence[str], location: str) -> str:
    value = _coalesced_field(row, fields, field="question", location=location)
    return _require_nonempty_text(value, field=f"{location}.question")


def _coalesced_field(row: Mapping[str, Any], fields: Sequence[str], *, field: str, location: str) -> Any:
    present = [(name, row[name]) for name in fields if name in row and row[name] is not None]
    if not present:
        raise MCQNormalizationError(f"{location}: missing {field}; expected one of {list(fields)}")
    canonical_values: list[str] = []
    for name, value in present:
        try:
            canonical_values.append(json.dumps(normalize_json(value), ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            raise MCQNormalizationError(f"{location}: {name} is not JSON-normalizable") from exc
    if len(set(canonical_values)) != 1:
        raise MCQNormalizationError(f"{location}: ambiguous {field} fields disagree: {[name for name, _ in present]}")
    return present[0][1]


def _parse_choices(raw_choices: Any, *, location: str) -> list[dict[str, str]]:
    pairs: list[tuple[Any, Any]]
    if isinstance(raw_choices, Mapping):
        label_keys = [key for key in ("label", "labels") if key in raw_choices]
        text_keys = [key for key in ("text", "texts") if key in raw_choices]
        if label_keys or text_keys:
            if len(label_keys) != 1 or len(text_keys) != 1:
                raise MCQNormalizationError(f"{location}: choices mapping needs exactly one labels and one text list")
            labels = raw_choices[label_keys[0]]
            texts = raw_choices[text_keys[0]]
            if not _is_nonstring_sequence(labels) or not _is_nonstring_sequence(texts):
                raise MCQNormalizationError(f"{location}: parallel choice labels/text must be arrays")
            if len(labels) != len(texts):
                raise MCQNormalizationError(f"{location}: choice labels/text lengths differ")
            pairs = list(zip(labels, texts))
        else:
            pairs = list(raw_choices.items())
    elif _is_nonstring_sequence(raw_choices):
        values = list(raw_choices)
        if all(isinstance(value, str) for value in values):
            pairs = list(zip(_generated_labels(len(values), location=location), values))
        elif all(isinstance(value, Mapping) for value in values):
            have_labels = [any(key in value for key in ("label", "key")) for value in values]
            if any(have_labels) and not all(have_labels):
                raise MCQNormalizationError(f"{location}: choice objects mix labeled and unlabeled forms")
            labels = (
                [
                    _coalesced_field(value, ("label", "key"), field="choice label", location=f"{location}[{index}]")
                    for index, value in enumerate(values)
                ]
                if all(have_labels)
                else _generated_labels(len(values), location=location)
            )
            texts = [
                _coalesced_field(
                    value, ("text", "value", "option"), field="choice text", location=f"{location}[{index}]"
                )
                for index, value in enumerate(values)
            ]
            pairs = list(zip(labels, texts))
        else:
            raise MCQNormalizationError(f"{location}: choices array must contain only strings or only choice objects")
    else:
        raise MCQNormalizationError(f"{location}: choices must be a labels/text mapping or an array")

    if len(pairs) < 2:
        raise MCQNormalizationError(f"{location}: at least two choices are required")
    normalized: list[dict[str, str]] = []
    seen_labels: set[str] = set()
    for index, (raw_label, raw_text) in enumerate(pairs):
        label = _normalize_label(raw_label, location=f"{location}[{index}].label")
        if label in seen_labels:
            raise MCQNormalizationError(f"{location}: duplicate choice label {label!r}")
        seen_labels.add(label)
        text = _require_nonempty_text(raw_text, field=f"{location}[{index}].text")
        normalized.append({"label": label, "text": text})
    return normalized


def _generated_labels(count: int, *, location: str) -> list[str]:
    if count > 26:
        raise MCQNormalizationError(f"{location}: cannot reconstruct labels for more than 26 unlabeled choices")
    return [chr(ord("A") + index) for index in range(count)]


def _correct_label_from_row(
    source: str,
    row: Mapping[str, Any],
    *,
    choices: Sequence[Mapping[str, str]],
    location: str,
) -> str:
    fields_by_source = {
        "arc": ("answerKey", "answer_key", "answer", "label"),
        "openbookqa": ("answerKey", "answer_key", "answer", "label"),
        "bbh": ("target", "answer"),
        "mmlu": ("answer", "answerKey", "answer_key", "target", "label"),
    }
    try:
        fields = fields_by_source[source]
    except KeyError as exc:
        raise MCQNormalizationError(f"unsupported MCQ source {source!r}") from exc
    present = [(field, row[field]) for field in fields if field in row and row[field] is not None]
    if not present:
        raise MCQNormalizationError(f"{location}: missing answer; expected one of {list(fields)}")
    labels = [choice["label"] for choice in choices]
    resolved = [
        _answer_value_to_label(value, source=source, field=field, labels=labels, location=location)
        for field, value in present
    ]
    if len(set(resolved)) != 1:
        raise MCQNormalizationError(
            f"{location}: ambiguous answer fields disagree: {dict(zip((field for field, _ in present), resolved))}"
        )
    return resolved[0]


def _answer_value_to_label(
    value: Any,
    *,
    source: str,
    field: str,
    labels: Sequence[str],
    location: str,
) -> str:
    if isinstance(value, bool):
        raise MCQNormalizationError(f"{location}: {field} must not be boolean")
    if isinstance(value, int):
        if source == "mmlu":
            if 0 <= value < len(labels):
                return labels[value]
            raise MCQNormalizationError(f"{location}: MMLU answer index {value} is outside 0..{len(labels) - 1}")
        numeric_label = str(value)
        if numeric_label in labels:
            return numeric_label
        raise MCQNormalizationError(f"{location}: integer {field} {value} does not name a choice label")
    if not isinstance(value, str):
        raise MCQNormalizationError(f"{location}: {field} must be a label string")

    stripped = normalize_text(value)
    direct_candidates = [stripped]
    if len(stripped) >= 2 and (stripped[0], stripped[-1]) in {("(", ")"), ("[", "]")}:
        direct_candidates.append(stripped[1:-1].strip())
    declaration = _ANSWER_DECLARATION_RE.fullmatch(stripped)
    if declaration:
        direct_candidates.append(next(group for group in declaration.groups() if group is not None))
    normalized_candidates = []
    for candidate in direct_candidates:
        try:
            normalized_candidates.append(_normalize_label(candidate, location=f"{location}.{field}"))
        except MCQNormalizationError:
            continue
    matching = sorted(set(normalized_candidates) & set(labels))
    if len(matching) == 1:
        return matching[0]
    if len(matching) > 1:
        raise MCQNormalizationError(f"{location}: {field} {value!r} ambiguously names several choices")
    raise MCQNormalizationError(f"{location}: {field} {value!r} does not reliably name one of {list(labels)}")


def _source_key(
    row: Mapping[str, Any],
    *,
    subset: str,
    payload: Mapping[str, Any],
    location: str,
) -> str:
    id_fields = ("id", "question_id", "questionId", "idx")
    present = [(field, row[field]) for field in id_fields if field in row and row[field] is not None]
    identifiers: list[str] = []
    for field, value in present:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise MCQNormalizationError(f"{location}: {field} must be a string or integer")
        identifier = normalize_text(str(value))
        if not identifier:
            raise MCQNormalizationError(f"{location}: {field} must be non-empty")
        identifiers.append(identifier)
    if len(set(identifiers)) > 1:
        raise MCQNormalizationError(f"{location}: ambiguous identifier fields disagree")
    identifier = identifiers[0] if identifiers else f"content-{sha256_json(payload)[:24]}"
    return f"{subset}:{identifier}"


def _validated_mcq_payload(payload: Mapping[str, Any], *, location: str) -> tuple[str, list[dict[str, str]], str]:
    if not isinstance(payload, Mapping):
        raise MCQNormalizationError(f"{location}: payload must be an object")
    question = _require_nonempty_text(payload.get("question"), field=f"{location}.payload.question")
    choices = _parse_choices(payload.get("choices"), location=f"{location}.payload.choices")
    correct_label = _normalize_label(payload.get("correct_label"), location=f"{location}.payload.correct_label")
    if correct_label not in {choice["label"] for choice in choices}:
        raise MCQNormalizationError(f"{location}: correct_label {correct_label!r} is absent from choices")
    return question, choices, correct_label


def _require_nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise MCQNormalizationError(f"{field} must be a string")
    normalized = normalize_text(value)
    if not normalized:
        raise MCQNormalizationError(f"{field} must be non-empty")
    return normalized


def _normalize_label(value: Any, *, location: str) -> str:
    if not isinstance(value, str):
        raise MCQNormalizationError(f"{location}: option label must be a string")
    normalized = normalize_text(value).upper()
    if _LABEL_RE.fullmatch(normalized) is None:
        raise MCQNormalizationError(f"{location}: option label {value!r} must be one ASCII letter or digit")
    return normalized


def _require_seed(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MCQNormalizationError("wrong_option_seed must be an integer")
    return value


def _is_nonstring_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


__all__ = [
    "CLEAN_PROMPT_TEMPLATE_VERSION",
    "DEFAULT_WRONG_OPTION_SEED",
    "MCQ_RECORD_TYPE",
    "NORMALIZED_TRAINING_ARTIFACT_KIND",
    "PROMPT_PAIR_ARTIFACT_KIND",
    "PROMPT_PAIR_RECORD_TYPE",
    "TRAINING_SOURCES",
    "WRONG_SUGGESTION_TEMPLATE_VERSION",
    "MCQNormalizationError",
    "build_sycophancy_pairs",
    "normalize_arc_rows",
    "normalize_bbh_rows",
    "normalize_openbookqa_rows",
    "normalize_training_data",
    "render_clean_prompt",
    "render_wrong_suggestion_prompt",
    "select_wrong_label",
]
