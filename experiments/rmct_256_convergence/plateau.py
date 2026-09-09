"""Immutable, training-primary RMCT-256 convergence accounting.

The RMCT-256 continuation is deliberately segmented into four fixed 64-row
blocks per pass.  Each segment consumes the trainer's authoritative 16
per-update ``train/consistency_gap_abs_sum_1``, ``..._count_1``, and
``..._mean_1`` records.  Every update must cover all four questions; a missing
parsed rate is a hard failure rather than a convenient zero.  The block score
is therefore the sample-weighted mean:

    sum(update.abs_sum for 16 updates) / 64

Four such block scores form one complete-pass pooled score.  Blocks are
matched by their ordered question IDs across passes, so a later pass cannot
look better merely because it happened to see easier rows.

Only the training metric participates in plateau decisions.  A metric document
may carry held-out diagnostics, but they are hash-bound provenance only and
are never read by the decision logic.

All persisted inputs and outputs are content-addressed JSON files.  The guard
recomputes a decision from every source metric before accepting a receipt,
which makes missing, changed, incomplete, duplicated, or out-of-order metrics
fail closed.  A valid terminal decision exits successfully so an ``afterok``
successor can read it and no-op; only a ``fail`` decision exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

SEGMENT_METRICS_SCHEMA = "rmct256-convergence-segment-metrics-v2"
EXTRACTED_METRICS_SOURCE_SCHEMA = "rmct256-convergence-extracted-metrics-source-v1"
DECISION_RECEIPT_SCHEMA = "rmct256-convergence-decision-receipt-v1"

UPDATES_PER_SEGMENT = 16
QUESTIONS_PER_UPDATE = 4
QUESTIONS_PER_SEGMENT = UPDATES_PER_SEGMENT * QUESTIONS_PER_UPDATE
SEGMENTS_PER_PASS = 4
UPDATES_PER_PASS = UPDATES_PER_SEGMENT * SEGMENTS_PER_PASS
MIN_FULL_PASSES = 1
TRAINING_PERTURBATION_INDEX = 1
_MEAN_TOLERANCE = Decimal("1e-12")

DEFAULT_MIN_DELTA = 0.01
DEFAULT_PATIENCE = 2
DEFAULT_HARD_CAP_PASSES = 4

TRAINING_METRIC_NAME = "mean_absolute_biased_minus_clean_rate"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_METRIC_FILENAME_RE = re.compile(
    r"segment-metrics-p0*(?P<pass>[1-9][0-9]*)-b0*(?P<block>[0-3])-(?P<sha>[0-9a-f]{64})\.json\Z"
)
_DECISION_FILENAME_RE = re.compile(
    r"decision-p0*(?P<pass>[1-9][0-9]*)-b0*(?P<block>[0-3])-(?P<sha>[0-9a-f]{64})\.json\Z"
)


class PlateauError(ValueError):
    """Base class for a rejectable convergence-contract violation."""


class MetricValidationError(PlateauError):
    """A segment metric is absent, malformed, mismatched, or changed."""


class DecisionReceiptError(PlateauError):
    """A decision receipt cannot safely control an afterok successor."""


class NonContinuationDecision(DecisionReceiptError):
    """A valid receipt intentionally says that training must not continue."""


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricValidationError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
    error_type: type[PlateauError] = MetricValidationError,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise error_type(f"{label} has {'; '.join(details)}")


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetricValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise MetricValidationError(f"{label} must be at least {minimum}")
    return value


def _require_nonempty_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetricValidationError(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: Any, *, label: str, error_type: type[PlateauError] = MetricValidationError) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise error_type(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _ensure_json_value(value: Any, *, label: str) -> None:
    """Reject NaN/Infinity in optional diagnostic provenance too."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise MetricValidationError(f"{label} must not contain NaN or infinity")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_json_value(item, label=f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MetricValidationError(f"{label} has a non-string object key")
            _ensure_json_value(item, label=f"{label}.{key}")
        return
    raise MetricValidationError(f"{label} is not JSON-compatible")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _json_object(
    payload: bytes, *, label: str, error_type: type[PlateauError] = MetricValidationError
) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise error_type(f"{label} is not valid finite JSON") from exc
    if not isinstance(value, dict):
        raise error_type(f"{label} must contain one JSON object")
    return value


def _regular_file(path: str | Path, *, label: str, error_type: type[PlateauError] = MetricValidationError) -> Path:
    supplied = Path(path)
    resolved = supplied.resolve()
    if supplied.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise error_type(f"{label} must be a regular non-symlink file: {supplied}")
    return resolved


def _decimal_from_number(value: Any, *, label: str, minimum: Decimal, maximum: Decimal) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricValidationError(f"{label} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise MetricValidationError(f"{label} must be finite")
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:  # pragma: no cover - protected by numeric type check
        raise MetricValidationError(f"{label} must be a finite number") from exc
    if not decimal.is_finite() or not minimum <= decimal <= maximum:
        raise MetricValidationError(f"{label} must be in [{minimum}, {maximum}]")
    return decimal


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _decimal_record(value: Decimal) -> dict[str, str]:
    return {"decimal": _decimal_text(value)}


def _decimal_from_min_delta(value: float) -> Decimal:
    # str() preserves the intended human configuration (e.g. 0.01), unlike a
    # binary-float conversion that would make the threshold non-reproducible.
    return Decimal(str(value))


@dataclass(frozen=True, slots=True, order=True)
class SegmentPosition:
    """The unique location of one 16-update, 64-question training segment."""

    pass_index: int
    segment_index: int

    def __post_init__(self) -> None:
        if isinstance(self.pass_index, bool) or not isinstance(self.pass_index, int) or self.pass_index < 1:
            raise MetricValidationError("pass_index must be a positive integer")
        if (
            isinstance(self.segment_index, bool)
            or not isinstance(self.segment_index, int)
            or not 0 <= self.segment_index < SEGMENTS_PER_PASS
        ):
            raise MetricValidationError(f"segment_index must be in [0, {SEGMENTS_PER_PASS - 1}]")

    @property
    def checkpoint_step(self) -> int:
        return (self.pass_index - 1) * UPDATES_PER_PASS + (self.segment_index + 1) * UPDATES_PER_SEGMENT

    @property
    def row_offset(self) -> int:
        return self.segment_index * QUESTIONS_PER_SEGMENT

    def as_dict(self) -> dict[str, int]:
        return {
            "checkpoint_step": self.checkpoint_step,
            "pass_index": self.pass_index,
            "segment_index": self.segment_index,
        }


@dataclass(frozen=True, slots=True)
class PlateauConfig:
    """The explicit stopping policy; geometry stays fixed by the RMCT-256 plan."""

    min_delta: float = DEFAULT_MIN_DELTA
    patience: int = DEFAULT_PATIENCE
    hard_cap_passes: int = DEFAULT_HARD_CAP_PASSES

    def __post_init__(self) -> None:
        if isinstance(self.min_delta, bool) or not isinstance(self.min_delta, (int, float)):
            raise MetricValidationError("min_delta must be a finite number")
        if not math.isfinite(float(self.min_delta)) or not 0.0 <= float(self.min_delta) <= 1.0:
            raise MetricValidationError("min_delta must be finite and in [0, 1]")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1:
            raise MetricValidationError("patience must be a positive integer")
        if (
            isinstance(self.hard_cap_passes, bool)
            or not isinstance(self.hard_cap_passes, int)
            or self.hard_cap_passes < MIN_FULL_PASSES
        ):
            raise MetricValidationError(f"hard_cap_passes must be an integer of at least {MIN_FULL_PASSES}")

    @property
    def min_delta_decimal(self) -> Decimal:
        return _decimal_from_min_delta(float(self.min_delta))

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_every_updates": UPDATES_PER_SEGMENT,
            "hard_cap_passes": self.hard_cap_passes,
            "min_delta": float(self.min_delta),
            "min_full_passes": MIN_FULL_PASSES,
            "patience": self.patience,
            "questions_per_segment": QUESTIONS_PER_SEGMENT,
            "questions_per_update": QUESTIONS_PER_UPDATE,
            "segments_per_pass": SEGMENTS_PER_PASS,
            "training_perturbation_index": TRAINING_PERTURBATION_INDEX,
            "updates_per_segment": UPDATES_PER_SEGMENT,
        }

    @classmethod
    def from_document(cls, value: Any, *, error_type: type[PlateauError] = DecisionReceiptError) -> "PlateauConfig":
        if not isinstance(value, Mapping):
            raise error_type("receipt config must be an object")
        expected = {
            "checkpoint_every_updates",
            "hard_cap_passes",
            "min_delta",
            "min_full_passes",
            "patience",
            "questions_per_segment",
            "questions_per_update",
            "segments_per_pass",
            "training_perturbation_index",
            "updates_per_segment",
        }
        _require_exact_keys(value, label="receipt config", required=expected, error_type=error_type)
        try:
            config = cls(
                min_delta=value["min_delta"],
                patience=value["patience"],
                hard_cap_passes=value["hard_cap_passes"],
            )
        except MetricValidationError as exc:
            raise error_type(str(exc)) from exc
        if config.as_dict() != dict(value):
            raise error_type("receipt config differs from the fixed RMCT-256 convergence geometry")
        return config


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    """Opaque, hash-bound metadata for a saved checkpoint at one boundary."""

    path: str
    sha256: str
    step: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "step": self.step}


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """One regular local evidence file that is immutable by SHA-256 binding."""

    path: Path
    sha256: str

    def as_dict(self) -> dict[str, str]:
        return {"content_sha256": self.sha256, "path": str(self.path)}


@dataclass(frozen=True, slots=True)
class UpdateGap:
    """The authoritative absolute-gap aggregation for one four-question update."""

    update_index: int
    global_step: int
    absolute_sum: Decimal
    absolute_count: int
    absolute_mean: Decimal

    def as_dict(self) -> dict[str, Any]:
        return {
            "abs_count": self.absolute_count,
            "abs_mean": float(self.absolute_mean),
            "abs_sum": float(self.absolute_sum),
            "global_step": self.global_step,
            "update_index": self.update_index,
        }


@dataclass(frozen=True, slots=True)
class ExtractedMetricsSource:
    """A source receipt and all evidence it cryptographically binds."""

    selection_sha256: str
    position: SegmentPosition
    question_ids: tuple[str, ...]
    checkpoint: CheckpointReceipt
    raw_metrics_jsonl: FileIdentity
    normalized_metrics_jsonl: FileIdentity
    segment_completion_receipt: FileIdentity
    source_receipt: FileIdentity
    updates: tuple[UpdateGap, ...]
    heldout_diagnostics: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class SegmentMetric:
    """A validated, derived score from exactly one immutable metric document."""

    position: SegmentPosition
    selection_sha256: str
    question_ids: tuple[str, ...]
    updates: tuple[UpdateGap, ...]
    mean_absolute_gap: Decimal
    checkpoint: CheckpointReceipt
    source_receipt: FileIdentity
    raw_metrics_jsonl: FileIdentity
    normalized_metrics_jsonl: FileIdentity
    segment_completion_receipt: FileIdentity
    heldout_diagnostics_present: bool
    source_path: Path | None = None
    source_sha256: str | None = None

    @property
    def question_ids_sha256(self) -> str:
        return _sha256("".join(f"{question_id}\n" for question_id in self.question_ids).encode("utf-8"))

    def score_record(self) -> dict[str, str]:
        return _decimal_record(self.mean_absolute_gap)


@dataclass(frozen=True, slots=True)
class PassSummary:
    """One complete, matched four-block pass and its plateau state."""

    pass_index: int
    blocks: tuple[SegmentMetric, ...]
    pooled_score: Decimal
    matched_block_reductions: tuple[Decimal, ...] | None
    reference_before_score: Decimal | None
    reference_before_pass_index: int | None
    reduction_vs_reference: Decimal | None
    qualifies_as_improvement: bool | None
    nonimproving_passes: int
    reference_after_score: Decimal
    reference_after_pass_index: int

    @property
    def endpoint(self) -> SegmentMetric:
        return self.blocks[-1]


@dataclass(frozen=True, slots=True)
class PlateauAnalysis:
    """Validated metric chain plus all completed-pass accounting."""

    target: SegmentPosition
    config: PlateauConfig
    metrics: tuple[SegmentMetric, ...]
    complete_passes: tuple[PassSummary, ...]
    raw_best_pass: PassSummary | None
    plateau_reference_pass: PassSummary | None

    @property
    def target_is_complete_pass(self) -> bool:
        return self.target.segment_index == SEGMENTS_PER_PASS - 1


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    path: Path
    sha256: str
    status: str


@dataclass(frozen=True, slots=True)
class PublishedSegmentMetrics(PublishedArtifact):
    position: SegmentPosition
    mean_absolute_gap: Decimal


@dataclass(frozen=True, slots=True)
class PublishedDecision(PublishedArtifact):
    decision: str
    receipt: Mapping[str, Any]

    @property
    def guard_exit_code(self) -> int:
        return 1 if self.decision == "fail" else 0


def _parse_checkpoint(value: Any, *, position: SegmentPosition) -> CheckpointReceipt:
    checkpoint = _require_mapping(value, label="checkpoint")
    _require_exact_keys(checkpoint, label="checkpoint", required={"path", "sha256", "step"})
    step = _require_int(checkpoint["step"], label="checkpoint.step", minimum=1)
    if step != position.checkpoint_step:
        raise MetricValidationError(
            f"checkpoint.step must equal segment checkpoint_step {position.checkpoint_step}, got {step}"
        )
    return CheckpointReceipt(
        path=_require_nonempty_string(checkpoint["path"], label="checkpoint.path"),
        sha256=_require_sha256(checkpoint["sha256"], label="checkpoint.sha256"),
        step=step,
    )


def _parse_selection(value: Any, *, label: str = "selection") -> str:
    selection = _require_mapping(value, label=label)
    _require_exact_keys(selection, label=label, required={"content_sha256"})
    return _require_sha256(selection["content_sha256"], label=f"{label}.content_sha256")


def _parse_segment(value: Any, *, label: str = "segment") -> tuple[SegmentPosition, tuple[str, ...]]:
    segment = _require_mapping(value, label=label)
    _require_exact_keys(
        segment,
        label=label,
        required={
            "checkpoint_step",
            "pass_index",
            "question_ids",
            "questions",
            "row_offset",
            "segment_index",
            "updates",
        },
    )
    position = SegmentPosition(
        pass_index=_require_int(segment["pass_index"], label=f"{label}.pass_index", minimum=1),
        segment_index=_require_int(segment["segment_index"], label=f"{label}.segment_index", minimum=0),
    )
    checkpoint_step = _require_int(segment["checkpoint_step"], label=f"{label}.checkpoint_step", minimum=1)
    updates = _require_int(segment["updates"], label=f"{label}.updates", minimum=1)
    questions = _require_int(segment["questions"], label=f"{label}.questions", minimum=1)
    row_offset = _require_int(segment["row_offset"], label=f"{label}.row_offset", minimum=0)
    if checkpoint_step != position.checkpoint_step:
        raise MetricValidationError(
            f"{label}.checkpoint_step must be {position.checkpoint_step} for pass/block "
            f"{position.pass_index}/{position.segment_index}"
        )
    if updates != UPDATES_PER_SEGMENT or questions != QUESTIONS_PER_SEGMENT or row_offset != position.row_offset:
        raise MetricValidationError(
            f"{label} must declare fixed geometry updates={UPDATES_PER_SEGMENT}, "
            f"questions={QUESTIONS_PER_SEGMENT}, row_offset={position.row_offset}"
        )
    raw_question_ids = segment["question_ids"]
    if isinstance(raw_question_ids, (str, bytes)) or not isinstance(raw_question_ids, list):
        raise MetricValidationError(f"{label}.question_ids must be an ordered list")
    question_ids = tuple(
        _require_nonempty_string(question_id, label=f"{label}.question_ids[{index}]")
        for index, question_id in enumerate(raw_question_ids)
    )
    if len(question_ids) != QUESTIONS_PER_SEGMENT or len(set(question_ids)) != QUESTIONS_PER_SEGMENT:
        raise MetricValidationError(f"{label}.question_ids must contain exactly 64 unique ordered IDs")
    return position, question_ids


def _parse_file_identity(value: Any, *, label: str) -> FileIdentity:
    identity = _require_mapping(value, label=label)
    _require_exact_keys(identity, label=label, required={"content_sha256", "path"})
    path = _regular_file(_require_nonempty_string(identity["path"], label=f"{label}.path"), label=label)
    expected = _require_sha256(identity["content_sha256"], label=f"{label}.content_sha256")
    actual = _sha256(path.read_bytes())
    if actual != expected:
        raise MetricValidationError(f"{label} SHA-256 differs from its declared immutable identity: {path}")
    return FileIdentity(path=path, sha256=actual)


def _parse_update_gap(value: Any, *, label: str, position: SegmentPosition, expected_update_index: int) -> UpdateGap:
    update = _require_mapping(value, label=label)
    _require_exact_keys(
        update, label=label, required={"abs_count", "abs_mean", "abs_sum", "global_step", "update_index"}
    )
    update_index = _require_int(update["update_index"], label=f"{label}.update_index", minimum=1)
    if update_index != expected_update_index:
        raise MetricValidationError(
            f"{label}.update_index must be {expected_update_index}; segment records must be ordered 1..16"
        )
    expected_step = position.checkpoint_step - UPDATES_PER_SEGMENT + update_index
    global_step = _require_int(update["global_step"], label=f"{label}.global_step", minimum=1)
    if global_step != expected_step:
        raise MetricValidationError(f"{label}.global_step must be {expected_step}, got {global_step}")
    absolute_count = _require_int(update["abs_count"], label=f"{label}.abs_count", minimum=0)
    if absolute_count != QUESTIONS_PER_UPDATE:
        raise MetricValidationError(
            f"{label}.abs_count must be exactly {QUESTIONS_PER_UPDATE}; incomplete/parse-failed updates are not valid"
        )
    absolute_sum = _decimal_from_number(
        update["abs_sum"], label=f"{label}.abs_sum", minimum=Decimal("0"), maximum=Decimal(QUESTIONS_PER_UPDATE)
    )
    absolute_mean = _decimal_from_number(
        update["abs_mean"], label=f"{label}.abs_mean", minimum=Decimal("0"), maximum=Decimal("1")
    )
    if abs(absolute_mean - absolute_sum / absolute_count) > _MEAN_TOLERANCE:
        raise MetricValidationError(f"{label}.abs_mean must equal abs_sum / abs_count")
    return UpdateGap(
        update_index=update_index,
        global_step=global_step,
        absolute_sum=absolute_sum,
        absolute_count=absolute_count,
        absolute_mean=absolute_mean,
    )


def _normalized_metric_keys() -> tuple[str, str, str]:
    suffix = TRAINING_PERTURBATION_INDEX
    return (
        f"train/consistency_gap_abs_sum_{suffix}",
        f"train/consistency_gap_abs_count_{suffix}",
        f"train/consistency_gap_abs_mean_{suffix}",
    )


def _load_normalized_updates(identity: FileIdentity, *, position: SegmentPosition) -> tuple[UpdateGap, ...]:
    payload = identity.path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise MetricValidationError(f"normalized metrics JSONL must be non-empty and LF-terminated: {identity.path}")
    lines = payload.splitlines(keepends=True)
    if len(lines) != UPDATES_PER_SEGMENT:
        raise MetricValidationError(f"normalized metrics JSONL must contain exactly {UPDATES_PER_SEGMENT} records")
    sum_key, count_key, mean_key = _normalized_metric_keys()
    updates: list[UpdateGap] = []
    for expected_index, line in enumerate(lines, start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n") or not line[:-1].strip():
            raise MetricValidationError(
                f"normalized metrics JSONL has a non-canonical line at {identity.path}:{expected_index}"
            )
        row = _json_object(line, label=f"normalized metrics JSONL {identity.path}:{expected_index}")
        _require_exact_keys(
            row, label=f"normalized metrics JSONL {identity.path}:{expected_index}", required={"global_step", "metrics"}
        )
        metrics = _require_mapping(
            row["metrics"], label=f"normalized metrics JSONL {identity.path}:{expected_index}.metrics"
        )
        _require_exact_keys(
            metrics,
            label=f"normalized metrics JSONL {identity.path}:{expected_index}.metrics",
            required={sum_key, count_key, mean_key},
        )
        updates.append(
            _parse_update_gap(
                {
                    "abs_count": metrics[count_key],
                    "abs_mean": metrics[mean_key],
                    "abs_sum": metrics[sum_key],
                    "global_step": row["global_step"],
                    "update_index": expected_index,
                },
                label=f"normalized metrics JSONL {identity.path}:{expected_index}",
                position=position,
                expected_update_index=expected_index,
            )
        )
    return tuple(updates)


def _read_extracted_metrics_source(path: str | Path) -> ExtractedMetricsSource:
    source_path = _regular_file(path, label="extracted metrics source receipt")
    source_payload = source_path.read_bytes()
    source_identity = FileIdentity(path=source_path, sha256=_sha256(source_payload))
    document = _json_object(source_payload, label=f"extracted metrics source receipt {source_path}")
    _require_exact_keys(
        document,
        label="extracted metrics source receipt",
        required={
            "checkpoint",
            "normalized_metrics_jsonl",
            "raw_metrics_jsonl",
            "schema",
            "segment",
            "segment_completion_receipt",
            "selection",
        },
        optional={"heldout_diagnostics"},
    )
    if document["schema"] != EXTRACTED_METRICS_SOURCE_SCHEMA:
        raise MetricValidationError(
            f"extracted metrics source receipt.schema must be {EXTRACTED_METRICS_SOURCE_SCHEMA!r}"
        )
    selection_sha256 = _parse_selection(document["selection"])
    position, question_ids = _parse_segment(document["segment"])
    checkpoint = _parse_checkpoint(document["checkpoint"], position=position)
    raw_metrics = _parse_file_identity(document["raw_metrics_jsonl"], label="raw_metrics_jsonl")
    normalized_metrics = _parse_file_identity(document["normalized_metrics_jsonl"], label="normalized_metrics_jsonl")
    completion = _parse_file_identity(document["segment_completion_receipt"], label="segment_completion_receipt")
    diagnostics: Mapping[str, Any] | None = None
    if "heldout_diagnostics" in document:
        diagnostics = _require_mapping(document["heldout_diagnostics"], label="heldout_diagnostics")
        _ensure_json_value(diagnostics, label="heldout_diagnostics")
    return ExtractedMetricsSource(
        selection_sha256=selection_sha256,
        position=position,
        question_ids=question_ids,
        checkpoint=checkpoint,
        raw_metrics_jsonl=raw_metrics,
        normalized_metrics_jsonl=normalized_metrics,
        segment_completion_receipt=completion,
        source_receipt=source_identity,
        updates=_load_normalized_updates(normalized_metrics, position=position),
        heldout_diagnostics=diagnostics,
    )


def _source_metric_document(source: ExtractedMetricsSource) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema": SEGMENT_METRICS_SCHEMA,
        "selection": {"content_sha256": source.selection_sha256},
        "segment": {
            "checkpoint_step": source.position.checkpoint_step,
            "pass_index": source.position.pass_index,
            "question_ids": list(source.question_ids),
            "questions": QUESTIONS_PER_SEGMENT,
            "row_offset": source.position.row_offset,
            "segment_index": source.position.segment_index,
            "updates": UPDATES_PER_SEGMENT,
        },
        "checkpoint": source.checkpoint.as_dict(),
        "provenance": {
            "normalized_metrics_jsonl": source.normalized_metrics_jsonl.as_dict(),
            "raw_metrics_jsonl": source.raw_metrics_jsonl.as_dict(),
            "segment_completion_receipt": source.segment_completion_receipt.as_dict(),
            "source_receipt": source.source_receipt.as_dict(),
        },
        "training_gap": {
            "metric": TRAINING_METRIC_NAME,
            "perturbation_index": TRAINING_PERTURBATION_INDEX,
            "updates": [update.as_dict() for update in source.updates],
        },
    }
    if source.heldout_diagnostics is not None:
        document["heldout_diagnostics"] = dict(source.heldout_diagnostics)
    return document


def _source_from_metric_document(document: Mapping[str, Any]) -> ExtractedMetricsSource:
    provenance = _require_mapping(document["provenance"], label="provenance")
    _require_exact_keys(
        provenance,
        label="provenance",
        required={"normalized_metrics_jsonl", "raw_metrics_jsonl", "segment_completion_receipt", "source_receipt"},
    )
    source_receipt = _parse_file_identity(provenance["source_receipt"], label="provenance.source_receipt")
    raw_metrics = _parse_file_identity(provenance["raw_metrics_jsonl"], label="provenance.raw_metrics_jsonl")
    normalized_metrics = _parse_file_identity(
        provenance["normalized_metrics_jsonl"], label="provenance.normalized_metrics_jsonl"
    )
    completion = _parse_file_identity(
        provenance["segment_completion_receipt"], label="provenance.segment_completion_receipt"
    )
    source = _read_extracted_metrics_source(source_receipt.path)
    if (
        source.source_receipt != source_receipt
        or source.raw_metrics_jsonl != raw_metrics
        or source.normalized_metrics_jsonl != normalized_metrics
        or source.segment_completion_receipt != completion
    ):
        raise MetricValidationError(
            "segment metric provenance differs from its hash-bound extracted metrics source receipt"
        )
    return source


def parse_segment_metrics(document: Mapping[str, Any]) -> SegmentMetric:
    """Derive and validate a segment score from extracted trainer metric evidence."""

    document = _require_mapping(document, label="segment metric document")
    _require_exact_keys(
        document,
        label="segment metric document",
        required={"schema", "selection", "segment", "checkpoint", "provenance", "training_gap"},
        optional={"heldout_diagnostics"},
    )
    if document["schema"] != SEGMENT_METRICS_SCHEMA:
        raise MetricValidationError(f"segment metric document.schema must be {SEGMENT_METRICS_SCHEMA!r}")
    selection_sha256 = _parse_selection(document["selection"])
    position, question_ids = _parse_segment(document["segment"])
    checkpoint = _parse_checkpoint(document["checkpoint"], position=position)
    training = _require_mapping(document["training_gap"], label="training_gap")
    _require_exact_keys(training, label="training_gap", required={"metric", "perturbation_index", "updates"})
    if training["metric"] != TRAINING_METRIC_NAME:
        raise MetricValidationError(f"training_gap.metric must be {TRAINING_METRIC_NAME!r}")
    perturbation_index = _require_int(
        training["perturbation_index"], label="training_gap.perturbation_index", minimum=0
    )
    if perturbation_index != TRAINING_PERTURBATION_INDEX:
        raise MetricValidationError(f"training_gap.perturbation_index must be {TRAINING_PERTURBATION_INDEX}")
    raw_updates = training["updates"]
    if not isinstance(raw_updates, list) or len(raw_updates) != UPDATES_PER_SEGMENT:
        raise MetricValidationError(f"training_gap.updates must contain exactly {UPDATES_PER_SEGMENT} records")
    updates = tuple(
        _parse_update_gap(
            raw_update,
            label=f"training_gap.updates[{index - 1}]",
            position=position,
            expected_update_index=index,
        )
        for index, raw_update in enumerate(raw_updates, start=1)
    )
    source = _source_from_metric_document(document)
    if (
        source.selection_sha256 != selection_sha256
        or source.position != position
        or source.question_ids != question_ids
        or source.checkpoint != checkpoint
        or source.updates != updates
    ):
        raise MetricValidationError(
            "segment metric document differs from its hash-bound extracted metrics source receipt"
        )
    if "heldout_diagnostics" in document:
        diagnostics = _require_mapping(document["heldout_diagnostics"], label="heldout_diagnostics")
        _ensure_json_value(diagnostics, label="heldout_diagnostics")
        if source.heldout_diagnostics is None or dict(diagnostics) != dict(source.heldout_diagnostics):
            raise MetricValidationError("heldout diagnostics differ from the hash-bound source receipt")
    elif source.heldout_diagnostics is not None:
        raise MetricValidationError("segment metric omits heldout diagnostics present in its source receipt")
    return SegmentMetric(
        position=position,
        selection_sha256=selection_sha256,
        question_ids=question_ids,
        updates=updates,
        mean_absolute_gap=sum((update.absolute_sum for update in updates), Decimal("0")) / QUESTIONS_PER_SEGMENT,
        checkpoint=checkpoint,
        source_receipt=source.source_receipt,
        raw_metrics_jsonl=source.raw_metrics_jsonl,
        normalized_metrics_jsonl=source.normalized_metrics_jsonl,
        segment_completion_receipt=source.segment_completion_receipt,
        heldout_diagnostics_present="heldout_diagnostics" in document,
    )


def _metric_filename(position: SegmentPosition, digest: str) -> str:
    return f"segment-metrics-p{position.pass_index:03d}-b{position.segment_index:02d}-{digest}.json"


def _decision_filename(position: SegmentPosition, digest: str) -> str:
    return f"decision-p{position.pass_index:03d}-b{position.segment_index:02d}-{digest}.json"


def _publish_immutable(path: Path, payload: bytes) -> str:
    """Atomically create one content-addressed artifact, or resume exact bytes."""

    if path.exists() or path.is_symlink():
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
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"immutable artifact appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "written"


def publish_segment_metrics(directory: str | Path, document: Mapping[str, Any]) -> PublishedSegmentMetrics:
    """Validate and content-address one segment metric document.

    The source document is canonicalized before publishing.  Later guard reads
    require this filename/hash binding, so direct edits are detected instead of
    being silently treated as fresh observations.
    """

    metric = parse_segment_metrics(document)
    payload = _canonical_json(document)
    digest = _sha256(payload)
    path = Path(directory).resolve() / _metric_filename(metric.position, digest)
    status = _publish_immutable(path, payload)
    return PublishedSegmentMetrics(
        path=path,
        sha256=digest,
        status=status,
        position=metric.position,
        mean_absolute_gap=metric.mean_absolute_gap,
    )


def publish_metrics_from_source_receipt(
    directory: str | Path,
    source_receipt: str | Path,
) -> PublishedSegmentMetrics:
    """Verify an extractor receipt and publish its content-addressed metric artifact.

    ``source_receipt`` is the only CLI input accepted from an Isambard segment
    extractor.  It binds the sealed original logger JSONL, the normalized 16
    records consumed here, and the segment completion/checkpoint receipt.
    """

    return publish_segment_metrics(directory, _source_metric_document(_read_extracted_metrics_source(source_receipt)))


def load_segment_metrics(path: str | Path) -> SegmentMetric:
    """Load a content-addressed metric document and reject any byte drift."""

    resolved = _regular_file(path, label="segment metric")
    match = _METRIC_FILENAME_RE.fullmatch(resolved.name)
    if match is None:
        raise MetricValidationError(
            "segment metric filename must be content-addressed as " "segment-metrics-p###-b##-<sha256>.json"
        )
    payload = resolved.read_bytes()
    digest = _sha256(payload)
    if digest != match.group("sha"):
        raise MetricValidationError(f"segment metric content SHA-256 disagrees with its filename: {resolved}")
    document = _json_object(payload, label=f"segment metric {resolved}")
    metric = parse_segment_metrics(document)
    if metric.position.pass_index != int(match.group("pass")) or metric.position.segment_index != int(
        match.group("block")
    ):
        raise MetricValidationError(f"segment metric filename position disagrees with document: {resolved}")
    return replace(metric, source_path=resolved, source_sha256=digest)


def discover_segment_metrics(directory: str | Path) -> list[Path]:
    """Return all metric artifacts in one non-symlink directory, sorted by name."""

    root = Path(directory)
    resolved = root.resolve()
    if root.is_symlink() or resolved.is_symlink() or not resolved.is_dir():
        raise MetricValidationError(f"metrics directory must be a non-symlink directory: {root}")
    paths: list[Path] = []
    for child in sorted(resolved.iterdir(), key=lambda item: item.name):
        if child.name.startswith("segment-metrics-"):
            if not _METRIC_FILENAME_RE.fullmatch(child.name):
                raise MetricValidationError(f"metrics directory contains a malformed metric filename: {child.name}")
            paths.append(child)
    if not paths:
        raise MetricValidationError(f"metrics directory contains no segment metrics: {resolved}")
    return paths


def _metric_position_from_path(path: Path) -> SegmentPosition:
    match = _METRIC_FILENAME_RE.fullmatch(path.name)
    if match is None:  # pragma: no cover - discover_segment_metrics already enforces this
        raise MetricValidationError(f"not a content-addressed segment metric filename: {path}")
    return SegmentPosition(int(match.group("pass")), int(match.group("block")))


def _expected_positions(target: SegmentPosition) -> tuple[SegmentPosition, ...]:
    positions: list[SegmentPosition] = []
    for pass_index in range(1, target.pass_index + 1):
        final_segment = target.segment_index if pass_index == target.pass_index else SEGMENTS_PER_PASS - 1
        positions.extend(SegmentPosition(pass_index, segment_index) for segment_index in range(final_segment + 1))
    return tuple(positions)


def _normalize_expected_question_ids(
    expected_question_ids: Mapping[SegmentPosition, Sequence[str]] | None,
    *,
    expected_positions: Sequence[SegmentPosition],
) -> dict[SegmentPosition, tuple[str, ...]]:
    if expected_question_ids is None:
        return {}
    expected_set = set(expected_positions)
    normalized: dict[SegmentPosition, tuple[str, ...]] = {}
    for position, question_ids in expected_question_ids.items():
        if not isinstance(position, SegmentPosition):
            raise MetricValidationError("expected question-ID contracts must be keyed by SegmentPosition")
        if position not in expected_set:
            raise MetricValidationError(f"question-ID contract is outside the requested metric prefix: {position}")
        if isinstance(question_ids, (str, bytes)) or not isinstance(question_ids, Sequence):
            raise MetricValidationError(f"question-ID contract for {position} must be a sequence")
        ids = tuple(
            _require_nonempty_string(item, label=f"expected question ID for {position}") for item in question_ids
        )
        if len(ids) != QUESTIONS_PER_SEGMENT or len(set(ids)) != QUESTIONS_PER_SEGMENT:
            raise MetricValidationError(f"question-ID contract for {position} must contain 64 unique IDs")
        normalized[position] = ids
    return normalized


def _summarize_complete_passes(
    metrics_by_position: Mapping[SegmentPosition, SegmentMetric],
    *,
    target: SegmentPosition,
    config: PlateauConfig,
) -> tuple[tuple[PassSummary, ...], PassSummary | None, PassSummary | None]:
    summaries: list[PassSummary] = []
    raw_best: PassSummary | None = None
    reference_score: Decimal | None = None
    reference_pass_index: int | None = None
    reference_summary: PassSummary | None = None
    previous: PassSummary | None = None
    nonimproving = 0

    for pass_index in range(1, target.pass_index + 1):
        positions = tuple(SegmentPosition(pass_index, index) for index in range(SEGMENTS_PER_PASS))
        if any(position not in metrics_by_position for position in positions):
            break
        blocks = tuple(metrics_by_position[position] for position in positions)
        pooled = sum((block.mean_absolute_gap for block in blocks), Decimal("0")) / SEGMENTS_PER_PASS
        if previous is None:
            matched_reductions: tuple[Decimal, ...] | None = None
            before_score = None
            before_pass_index = None
            reduction = None
            qualifies: bool | None = None
            reference_score = pooled
            reference_pass_index = pass_index
            nonimproving = 0
        else:
            matched_reductions = tuple(
                previous.blocks[index].mean_absolute_gap - blocks[index].mean_absolute_gap
                for index in range(SEGMENTS_PER_PASS)
            )
            assert reference_score is not None and reference_pass_index is not None
            before_score = reference_score
            before_pass_index = reference_pass_index
            reduction = before_score - pooled
            qualifies = reduction >= config.min_delta_decimal
            if qualifies:
                reference_score = pooled
                reference_pass_index = pass_index
                nonimproving = 0
            else:
                nonimproving += 1
        assert reference_score is not None and reference_pass_index is not None
        summary = PassSummary(
            pass_index=pass_index,
            blocks=blocks,
            pooled_score=pooled,
            matched_block_reductions=matched_reductions,
            reference_before_score=before_score,
            reference_before_pass_index=before_pass_index,
            reduction_vs_reference=reduction,
            qualifies_as_improvement=qualifies,
            nonimproving_passes=nonimproving,
            reference_after_score=reference_score,
            reference_after_pass_index=reference_pass_index,
        )
        summaries.append(summary)
        if raw_best is None or summary.pooled_score < raw_best.pooled_score:
            # A tie keeps the earlier complete pass, which is deterministic and
            # avoids inventing a preference for a later checkpoint.
            raw_best = summary
        if reference_pass_index == pass_index:
            reference_summary = summary
        previous = summary
    return tuple(summaries), raw_best, reference_summary


def _decision_for_completed_pass(summary: PassSummary, *, config: PlateauConfig) -> tuple[str, str]:
    """Return the terminal/continuation state at one complete pass boundary."""

    if summary.pass_index > config.hard_cap_passes:
        raise MetricValidationError("metrics extend past the configured hard pass cap")
    if summary.pass_index == config.hard_cap_passes:
        # A hard cap is a scientifically distinct outcome: even a still
        # improving fourth pass cannot be labelled as convergence.
        if summary.qualifies_as_improvement is True:
            return "capped", "hard_cap_reached_while_improving"
        if summary.nonimproving_passes >= config.patience:
            return "converged", "plateau_patience_exhausted_at_hard_cap"
        return "capped", "hard_cap_reached"
    if summary.pass_index >= MIN_FULL_PASSES and summary.nonimproving_passes >= config.patience:
        return "converged", "plateau_patience_exhausted"
    return "continue", "complete_pass_requires_next_pass"


def analyze_segment_metrics(
    metrics: Sequence[SegmentMetric],
    *,
    target: SegmentPosition,
    config: PlateauConfig = PlateauConfig(),
    expected_selection_sha256: str | None = None,
    expected_question_ids: Mapping[SegmentPosition, Sequence[str]] | None = None,
) -> PlateauAnalysis:
    """Validate the exact metric prefix through ``target`` and summarize it.

    A valid prefix is contiguous from pass 1/block 0 to the requested segment.
    It cannot contain a duplicate, skip a block, switch selections, change a
    matched block's question order, or continue after an earlier terminal pass.
    """

    if not isinstance(config, PlateauConfig):
        raise MetricValidationError("config must be a PlateauConfig")
    if target.pass_index > config.hard_cap_passes:
        raise MetricValidationError(
            f"target pass {target.pass_index} exceeds configured hard cap {config.hard_cap_passes}"
        )
    required_positions = _expected_positions(target)
    expected_set = set(required_positions)
    normalized_question_ids = _normalize_expected_question_ids(
        expected_question_ids, expected_positions=required_positions
    )
    by_position: dict[SegmentPosition, SegmentMetric] = {}
    for metric in metrics:
        if not isinstance(metric, SegmentMetric):
            raise MetricValidationError("metrics must contain SegmentMetric values")
        if metric.position in by_position:
            raise MetricValidationError(f"duplicate segment metric for {metric.position}")
        if metric.position not in expected_set:
            raise MetricValidationError(f"unexpected metric outside requested prefix: {metric.position}")
        by_position[metric.position] = metric
    if set(by_position) != expected_set:
        missing = sorted(expected_set - set(by_position))
        unexpected = sorted(set(by_position) - expected_set)
        raise MetricValidationError(f"incomplete contiguous metric prefix; missing={missing}, unexpected={unexpected}")

    selection_hashes = {metric.selection_sha256 for metric in by_position.values()}
    if len(selection_hashes) != 1:
        raise MetricValidationError("all segment metrics must bind the same immutable training selection SHA-256")
    selection_sha256 = next(iter(selection_hashes))
    if expected_selection_sha256 is not None:
        required_selection_sha256 = _require_sha256(expected_selection_sha256, label="expected_selection_sha256")
        if selection_sha256 != required_selection_sha256:
            raise MetricValidationError("segment metrics do not bind the expected immutable training selection SHA-256")

    first_pass_ids: dict[int, tuple[str, ...]] = {}
    for position in required_positions:
        metric = by_position[position]
        expected_ids = normalized_question_ids.get(position)
        if expected_ids is not None and metric.question_ids != expected_ids:
            raise MetricValidationError(f"segment metric question IDs differ from the supplied contract: {position}")
        baseline = first_pass_ids.get(position.segment_index)
        if baseline is None:
            first_pass_ids[position.segment_index] = metric.question_ids
        elif metric.question_ids != baseline:
            raise MetricValidationError(
                f"matched block question IDs differ from pass 1 for segment index {position.segment_index}"
            )

    summaries, raw_best, reference = _summarize_complete_passes(by_position, target=target, config=config)
    for summary in summaries:
        if summary.pass_index >= target.pass_index:
            break
        prior_decision, _ = _decision_for_completed_pass(summary, config=config)
        if prior_decision != "continue":
            raise MetricValidationError(
                f"metric prefix continues after terminal pass {summary.pass_index} decision {prior_decision!r}"
            )

    return PlateauAnalysis(
        target=target,
        config=config,
        metrics=tuple(by_position[position] for position in required_positions),
        complete_passes=summaries,
        raw_best_pass=raw_best,
        plateau_reference_pass=reference,
    )


def analyze_metric_paths(
    paths: Sequence[str | Path],
    *,
    target: SegmentPosition,
    config: PlateauConfig = PlateauConfig(),
    expected_selection_sha256: str | None = None,
    expected_question_ids: Mapping[SegmentPosition, Sequence[str]] | None = None,
) -> PlateauAnalysis:
    """Load content-addressed metric files, then apply :func:`analyze_segment_metrics`."""

    return analyze_segment_metrics(
        [load_segment_metrics(path) for path in paths],
        target=target,
        config=config,
        expected_selection_sha256=expected_selection_sha256,
        expected_question_ids=expected_question_ids,
    )


def _afterok_record(decision: str) -> dict[str, Any]:
    if decision == "continue":
        return {"permit_training": True, "successor_action": "launch"}
    if decision in {"converged", "capped"}:
        return {"permit_training": False, "successor_action": "no_op"}
    if decision == "fail":
        return {"permit_training": False, "successor_action": "block"}
    raise AssertionError(f"unknown decision {decision!r}")


def _source_metric_record(metric: SegmentMetric) -> dict[str, Any]:
    if metric.source_path is None or metric.source_sha256 is None:
        raise MetricValidationError("immutable decision receipts require file-backed source metrics")
    return {
        "checkpoint_step": metric.position.checkpoint_step,
        "content_sha256": metric.source_sha256,
        "pass_index": metric.position.pass_index,
        "path": str(metric.source_path),
        "segment_index": metric.position.segment_index,
    }


def _checkpoint_record(metric: SegmentMetric) -> dict[str, Any]:
    return {
        "checkpoint": metric.checkpoint.as_dict(),
        "checkpoint_step": metric.position.checkpoint_step,
        "pass_index": metric.position.pass_index,
        "segment_index": metric.position.segment_index,
        "source_metric_content_sha256": metric.source_sha256,
    }


def _endpoint_record(summary: PassSummary) -> dict[str, Any]:
    endpoint = summary.endpoint
    return {
        "checkpoint": endpoint.checkpoint.as_dict(),
        "checkpoint_step": endpoint.position.checkpoint_step,
        "pass_index": summary.pass_index,
        "pooled_training_mean_absolute_gap": _decimal_record(summary.pooled_score),
        "segment_index": endpoint.position.segment_index,
        "source_metric_content_sha256": endpoint.source_sha256,
    }


def _pass_record(summary: PassSummary) -> dict[str, Any]:
    return {
        "block_scores": [
            {
                "mean_absolute_training_gap": metric.score_record(),
                "question_ids_sha256": metric.question_ids_sha256,
                "segment_index": metric.position.segment_index,
                "source_metric_content_sha256": metric.source_sha256,
            }
            for metric in summary.blocks
        ],
        "consecutive_nonimproving_passes": summary.nonimproving_passes,
        "matched_block_reductions_from_previous_pass": (
            None
            if summary.matched_block_reductions is None
            else [_decimal_record(value) for value in summary.matched_block_reductions]
        ),
        "pass_index": summary.pass_index,
        "plateau_reference_after": {
            "pass_index": summary.reference_after_pass_index,
            "pooled_training_mean_absolute_gap": _decimal_record(summary.reference_after_score),
        },
        "plateau_reference_before": (
            None
            if summary.reference_before_score is None
            else {
                "pass_index": summary.reference_before_pass_index,
                "pooled_training_mean_absolute_gap": _decimal_record(summary.reference_before_score),
            }
        ),
        "pooled_training_mean_absolute_gap": _decimal_record(summary.pooled_score),
        "qualifies_as_improvement": summary.qualifies_as_improvement,
        "reduction_vs_plateau_reference": (
            None if summary.reduction_vs_reference is None else _decimal_record(summary.reduction_vs_reference)
        ),
    }


def build_decision_receipt(analysis: PlateauAnalysis) -> dict[str, Any]:
    """Build the complete immutable decision payload from a validated prefix."""

    if not analysis.metrics:
        raise MetricValidationError("cannot build a decision receipt without metric sources")
    target_metric = analysis.metrics[-1]
    if target_metric.position != analysis.target:
        raise MetricValidationError("analysis target does not match its final source metric")

    if analysis.target_is_complete_pass:
        summary = analysis.complete_passes[-1]
        if summary.pass_index != analysis.target.pass_index:
            raise MetricValidationError("complete target pass has no complete-pass summary")
        decision, reason = _decision_for_completed_pass(summary, config=analysis.config)
    else:
        decision, reason = "continue", "awaiting_complete_pass_boundary"

    heldout_positions = [
        {"pass_index": metric.position.pass_index, "segment_index": metric.position.segment_index}
        for metric in analysis.metrics
        if metric.heldout_diagnostics_present
    ]
    return {
        "afterok": _afterok_record(decision),
        "checkpoint_receipts": [_checkpoint_record(metric) for metric in analysis.metrics],
        "complete_passes": [_pass_record(summary) for summary in analysis.complete_passes],
        "config": analysis.config.as_dict(),
        "decision": decision,
        "decision_metric": {
            "heldout_diagnostics_used_for_decision": False,
            "name": TRAINING_METRIC_NAME,
        },
        "heldout_diagnostics": {
            "source_segments_present": heldout_positions,
            "used_for_decision": False,
        },
        "plateau_reference_pass_endpoint": (
            None if analysis.plateau_reference_pass is None else _endpoint_record(analysis.plateau_reference_pass)
        ),
        "raw_best_pass_endpoint": None if analysis.raw_best_pass is None else _endpoint_record(analysis.raw_best_pass),
        "reason": reason,
        "schema": DECISION_RECEIPT_SCHEMA,
        "selection_content_sha256": target_metric.selection_sha256,
        "source_metrics": [_source_metric_record(metric) for metric in analysis.metrics],
        "target": analysis.target.as_dict(),
    }


def _candidate_metric_evidence(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for raw_path in sorted((Path(path) for path in paths), key=lambda item: str(item)):
        record: dict[str, Any] = {"path": str(raw_path.resolve())}
        try:
            if raw_path.is_symlink() or not raw_path.is_file():
                record["status"] = "missing_or_not_regular"
            else:
                record["content_sha256"] = _sha256(raw_path.read_bytes())
                record["status"] = "observed_untrusted_bytes"
        except OSError:
            record["status"] = "unreadable"
        evidence.append(record)
    return evidence


def build_failure_receipt(
    *,
    target: SegmentPosition,
    config: PlateauConfig,
    expected_selection_sha256: str | None,
    error: BaseException,
    candidate_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Record a fail-closed guard result without pretending incomplete data is valid."""

    selection = expected_selection_sha256 if expected_selection_sha256 is not None else None
    # A malformed expected selection is itself the failure being recorded.  Do
    # not let that prevent the guard from leaving an immutable fail receipt.
    if selection is not None and (not isinstance(selection, str) or not _SHA256_RE.fullmatch(selection)):
        selection = None
    return {
        "afterok": _afterok_record("fail"),
        "checkpoint_receipts": [],
        "complete_passes": [],
        "config": config.as_dict(),
        "decision": "fail",
        "decision_metric": {
            "heldout_diagnostics_used_for_decision": False,
            "name": TRAINING_METRIC_NAME,
        },
        "failure": {
            "candidate_metrics": _candidate_metric_evidence(candidate_paths),
            "error_type": type(error).__name__,
            "message": str(error),
        },
        "heldout_diagnostics": {"source_segments_present": [], "used_for_decision": False},
        "plateau_reference_pass_endpoint": None,
        "raw_best_pass_endpoint": None,
        "reason": "metric_validation_failed",
        "schema": DECISION_RECEIPT_SCHEMA,
        "selection_content_sha256": selection,
        "source_metrics": [],
        "target": target.as_dict(),
    }


def publish_decision_receipt(directory: str | Path, receipt: Mapping[str, Any]) -> PublishedDecision:
    """Content-address a constructed decision receipt without overwriting it."""

    if not isinstance(receipt, Mapping):
        raise DecisionReceiptError("decision receipt must be an object")
    target_raw = receipt.get("target")
    if not isinstance(target_raw, Mapping):
        raise DecisionReceiptError("decision receipt has no target object")
    try:
        target = SegmentPosition(pass_index=target_raw.get("pass_index"), segment_index=target_raw.get("segment_index"))
    except MetricValidationError as exc:
        raise DecisionReceiptError(str(exc)) from exc
    decision = receipt.get("decision")
    if decision not in {"continue", "converged", "capped", "fail"}:
        raise DecisionReceiptError("decision receipt has an invalid decision")
    payload = _canonical_json(receipt)
    digest = _sha256(payload)
    path = Path(directory).resolve() / _decision_filename(target, digest)
    status = _publish_immutable(path, payload)
    return PublishedDecision(path=path, sha256=digest, status=status, decision=str(decision), receipt=dict(receipt))


def _parse_target(value: Any, *, error_type: type[PlateauError] = DecisionReceiptError) -> SegmentPosition:
    if not isinstance(value, Mapping):
        raise error_type("receipt target must be an object")
    _require_exact_keys(
        value,
        label="receipt target",
        required={"checkpoint_step", "pass_index", "segment_index"},
        error_type=error_type,
    )
    try:
        position = SegmentPosition(value["pass_index"], value["segment_index"])
    except MetricValidationError as exc:
        raise error_type(str(exc)) from exc
    if value["checkpoint_step"] != position.checkpoint_step:
        raise error_type("receipt target checkpoint_step is inconsistent with pass/block")
    return position


def _validate_afterok(value: Any, *, decision: str) -> None:
    if not isinstance(value, Mapping):
        raise DecisionReceiptError("receipt afterok must be an object")
    _require_exact_keys(
        value,
        label="receipt afterok",
        required={"permit_training", "successor_action"},
        error_type=DecisionReceiptError,
    )
    if dict(value) != _afterok_record(decision):
        raise DecisionReceiptError("receipt afterok policy does not match its decision")


def _validate_basic_decision_receipt(document: Mapping[str, Any]) -> tuple[SegmentPosition, PlateauConfig, str]:
    required = {
        "afterok",
        "checkpoint_receipts",
        "complete_passes",
        "config",
        "decision",
        "decision_metric",
        "heldout_diagnostics",
        "plateau_reference_pass_endpoint",
        "raw_best_pass_endpoint",
        "reason",
        "schema",
        "selection_content_sha256",
        "source_metrics",
        "target",
    }
    optional = {"failure"}
    _require_exact_keys(
        document,
        label="decision receipt",
        required=required,
        optional=optional,
        error_type=DecisionReceiptError,
    )
    if document["schema"] != DECISION_RECEIPT_SCHEMA:
        raise DecisionReceiptError(f"decision receipt.schema must be {DECISION_RECEIPT_SCHEMA!r}")
    decision = document["decision"]
    if decision not in {"continue", "converged", "capped", "fail"}:
        raise DecisionReceiptError("decision receipt has an invalid decision")
    if (decision == "fail") != ("failure" in document):
        raise DecisionReceiptError("only a fail receipt may contain failure evidence")
    target = _parse_target(document["target"])
    config = PlateauConfig.from_document(document["config"])
    _validate_afterok(document["afterok"], decision=decision)
    metric = document["decision_metric"]
    if not isinstance(metric, Mapping) or dict(metric) != {
        "heldout_diagnostics_used_for_decision": False,
        "name": TRAINING_METRIC_NAME,
    }:
        raise DecisionReceiptError("receipt decision metric must be the training mean absolute gap only")
    diagnostics = document["heldout_diagnostics"]
    if not isinstance(diagnostics, Mapping) or diagnostics.get("used_for_decision") is not False:
        raise DecisionReceiptError("held-out diagnostics may not drive a plateau decision")
    if decision == "fail":
        if document["selection_content_sha256"] is not None:
            _require_sha256(
                document["selection_content_sha256"],
                label="failure receipt selection_content_sha256",
                error_type=DecisionReceiptError,
            )
    else:
        _require_sha256(
            document["selection_content_sha256"],
            label="receipt selection_content_sha256",
            error_type=DecisionReceiptError,
        )
    return target, config, str(decision)


def load_decision_receipt(path: str | Path) -> dict[str, Any]:
    """Load a content-addressed decision receipt and validate its outer contract."""

    resolved = _regular_file(path, label="decision receipt", error_type=DecisionReceiptError)
    match = _DECISION_FILENAME_RE.fullmatch(resolved.name)
    if match is None:
        raise DecisionReceiptError("decision filename must be content-addressed as decision-p###-b##-<sha256>.json")
    payload = resolved.read_bytes()
    if _sha256(payload) != match.group("sha"):
        raise DecisionReceiptError(f"decision receipt content SHA-256 disagrees with its filename: {resolved}")
    document = _json_object(payload, label=f"decision receipt {resolved}", error_type=DecisionReceiptError)
    target, _, _ = _validate_basic_decision_receipt(document)
    if target.pass_index != int(match.group("pass")) or target.segment_index != int(match.group("block")):
        raise DecisionReceiptError("decision receipt filename position disagrees with its target")
    return document


def _metric_paths_from_receipt(document: Mapping[str, Any]) -> list[Path]:
    values = document.get("source_metrics")
    if not isinstance(values, list) or not values:
        raise DecisionReceiptError("non-fail receipt has no source metrics")
    paths: list[Path] = []
    seen_positions: set[SegmentPosition] = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise DecisionReceiptError(f"source_metrics[{index}] must be an object")
        _require_exact_keys(
            raw,
            label=f"source_metrics[{index}]",
            required={"checkpoint_step", "content_sha256", "pass_index", "path", "segment_index"},
            error_type=DecisionReceiptError,
        )
        position = SegmentPosition(raw["pass_index"], raw["segment_index"])
        if raw["checkpoint_step"] != position.checkpoint_step or position in seen_positions:
            raise DecisionReceiptError("receipt source metric positions are incomplete or inconsistent")
        seen_positions.add(position)
        digest = _require_sha256(
            raw["content_sha256"], label=f"source_metrics[{index}].content_sha256", error_type=DecisionReceiptError
        )
        path = _regular_file(raw["path"], label=f"source_metrics[{index}]", error_type=DecisionReceiptError)
        if _sha256(path.read_bytes()) != digest:
            raise DecisionReceiptError(f"source metric changed after the decision receipt was written: {path}")
        paths.append(path)
    return paths


def verify_decision_receipt(path: str | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    """Fully replay a successful receipt from its metric sources when requested.

    Replaying is the key afterok guard: it catches both a changed receipt and a
    metric file modified after a previously-valid convergence decision.
    """

    document = load_decision_receipt(path)
    target, config, decision = _validate_basic_decision_receipt(document)
    if decision == "fail":
        return document
    if not verify_sources:
        return document
    paths = _metric_paths_from_receipt(document)
    try:
        analysis = analyze_metric_paths(
            paths,
            target=target,
            config=config,
            expected_selection_sha256=document["selection_content_sha256"],
        )
        expected = build_decision_receipt(analysis)
    except (MetricValidationError, OSError, json.JSONDecodeError) as exc:
        raise DecisionReceiptError(f"source metric is no longer valid: {exc}") from exc
    if expected != document:
        raise DecisionReceiptError("decision receipt does not match a deterministic replay of its metric sources")
    return document


def require_continue(path: str | Path) -> dict[str, Any]:
    """Validate a guard receipt and allow a successor only for ``continue``."""

    receipt = verify_decision_receipt(path, verify_sources=True)
    if receipt["decision"] != "continue":
        raise NonContinuationDecision(f"decision {receipt['decision']!r} requires an afterok successor to no-op")
    return receipt


def guard_from_paths(
    paths: Sequence[str | Path],
    *,
    output_directory: str | Path,
    target: SegmentPosition,
    expected_selection_sha256: str,
    config: PlateauConfig = PlateauConfig(),
) -> PublishedDecision:
    """Emit a content-addressed continue/converged/capped/fail guard receipt."""

    try:
        required_selection_sha256 = _require_sha256(expected_selection_sha256, label="expected_selection_sha256")
        analysis = analyze_metric_paths(
            paths,
            target=target,
            config=config,
            expected_selection_sha256=required_selection_sha256,
        )
        receipt = build_decision_receipt(analysis)
    except Exception as exc:  # A guard must fail closed even on an incidental input/read failure.
        receipt = build_failure_receipt(
            target=target,
            config=config,
            expected_selection_sha256=expected_selection_sha256,
            error=exc,
            candidate_paths=paths,
        )
    return publish_decision_receipt(output_directory, receipt)


def guard_from_directory(
    metrics_directory: str | Path,
    *,
    output_directory: str | Path,
    target: SegmentPosition,
    expected_selection_sha256: str,
    config: PlateauConfig = PlateauConfig(),
) -> PublishedDecision:
    """Directory form of :func:`guard_from_paths` for a scheduler boundary job."""

    paths: list[Path] = []
    try:
        # The scheduler asks about an earlier completed pass before every later
        # segment starts.  Its metric directory therefore legitimately contains
        # a newer partial pass.  Select the exact contiguous prefix relevant to
        # this guard; analyze_metric_paths still rejects every gap/duplicate in
        # that prefix.  A later metric is checked when *its* pass is guarded.
        paths = [
            path for path in discover_segment_metrics(metrics_directory) if _metric_position_from_path(path) <= target
        ]
    except Exception as exc:
        receipt = build_failure_receipt(
            target=target,
            config=config,
            expected_selection_sha256=expected_selection_sha256,
            error=exc,
            candidate_paths=paths,
        )
        return publish_decision_receipt(output_directory, receipt)
    return guard_from_paths(
        paths,
        output_directory=output_directory,
        target=target,
        expected_selection_sha256=expected_selection_sha256,
        config=config,
    )


def _cli_config(args: argparse.Namespace) -> PlateauConfig:
    return PlateauConfig(
        min_delta=args.min_delta,
        patience=args.patience,
        hard_cap_passes=args.hard_cap_passes,
    )


def _print_json(value: Mapping[str, Any]) -> None:
    print(_canonical_json(value).decode("utf-8"), end="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish-metrics", help="verify an extractor source receipt and content-address its segment metrics"
    )
    publish.add_argument("--input", required=True, type=Path)
    publish.add_argument("--output-directory", required=True, type=Path)

    guard = subparsers.add_parser("guard", help="write a pass-boundary afterok decision receipt")
    guard.add_argument("--metrics-directory", required=True, type=Path)
    guard.add_argument("--output-directory", required=True, type=Path)
    guard.add_argument("--target-pass", required=True, type=int)
    guard.add_argument("--target-segment", required=True, type=int)
    guard.add_argument("--selection-sha256", required=True)
    guard.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    guard.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    guard.add_argument("--hard-cap-passes", type=int, default=DEFAULT_HARD_CAP_PASSES)

    verify = subparsers.add_parser(
        "verify-receipt", help="replay and verify a decision receipt before a successor acts"
    )
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--no-verify-sources", action="store_true")

    continue_parser = subparsers.add_parser(
        "require-continue", help="exit zero only for a fully verified continue receipt"
    )
    continue_parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by the scheduler's post-segment boundary guard."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "publish-metrics":
            result = publish_metrics_from_source_receipt(args.output_directory, args.input)
            _print_json(
                {
                    "mean_absolute_training_gap": _decimal_record(result.mean_absolute_gap),
                    "path": str(result.path),
                    "position": result.position.as_dict(),
                    "sha256": result.sha256,
                    "status": result.status,
                }
            )
            return 0
        if args.command == "guard":
            target = SegmentPosition(args.target_pass, args.target_segment)
            result = guard_from_directory(
                args.metrics_directory,
                output_directory=args.output_directory,
                target=target,
                expected_selection_sha256=args.selection_sha256,
                config=_cli_config(args),
            )
            _print_json(
                {
                    "afterok": result.receipt["afterok"],
                    "decision": result.decision,
                    "guard_exit_code": result.guard_exit_code,
                    "path": str(result.path),
                    "sha256": result.sha256,
                    "status": result.status,
                }
            )
            return result.guard_exit_code
        if args.command == "verify-receipt":
            receipt = verify_decision_receipt(args.receipt, verify_sources=not args.no_verify_sources)
            _print_json({"afterok": receipt["afterok"], "decision": receipt["decision"], "valid": True})
            return 0
        if args.command == "require-continue":
            receipt = require_continue(args.receipt)
            _print_json({"decision": receipt["decision"], "continue": True})
            return 0
        raise AssertionError(f"unknown command {args.command!r}")
    except NonContinuationDecision as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (DecisionReceiptError, FileExistsError, MetricValidationError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - exercised through main() in focused tests
    raise SystemExit(main())


__all__ = [
    "DECISION_RECEIPT_SCHEMA",
    "DEFAULT_HARD_CAP_PASSES",
    "DEFAULT_MIN_DELTA",
    "DEFAULT_PATIENCE",
    "DecisionReceiptError",
    "EXTRACTED_METRICS_SOURCE_SCHEMA",
    "MetricValidationError",
    "NonContinuationDecision",
    "PlateauAnalysis",
    "PlateauConfig",
    "PublishedDecision",
    "PublishedSegmentMetrics",
    "QUESTIONS_PER_SEGMENT",
    "QUESTIONS_PER_UPDATE",
    "SEGMENT_METRICS_SCHEMA",
    "SEGMENTS_PER_PASS",
    "SegmentMetric",
    "SegmentPosition",
    "UPDATES_PER_SEGMENT",
    "analyze_metric_paths",
    "analyze_segment_metrics",
    "build_decision_receipt",
    "discover_segment_metrics",
    "guard_from_directory",
    "guard_from_paths",
    "load_decision_receipt",
    "load_segment_metrics",
    "main",
    "parse_segment_metrics",
    "publish_decision_receipt",
    "publish_metrics_from_source_receipt",
    "publish_segment_metrics",
    "require_continue",
    "verify_decision_receipt",
]
