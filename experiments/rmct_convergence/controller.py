#!/usr/bin/env python3
"""Fail-closed convergence decisions for sealed RMCT checkpoint windows.

Each invocation evaluates exactly one consecutive, sealed 16-optimizer-step
segment.  The source metrics must report both training perturbations (1 and
2) for every optimizer update.  The decision statistic is sample weighted:

``sum(abs_sum) / sum(abs_count)`` across both perturbations and all updates.

The first segment always continues.  Later segments converge only when both
the current and immediately preceding segments have complete pooled and
per-bias parse/valid coverage, the current weighted gap is small enough, and
the two weighted gaps are stable.  Every decision is content addressed and
replayable from its source, checkpoint, completion, predecessor, and fixed
threshold configuration evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SOURCE_METRICS_SCHEMA = "rmct-convergence-source-metrics-v1"
CHECKPOINT_RECEIPT_SCHEMA = "rmct-convergence-checkpoint-receipt-v1"
COMPLETION_RECEIPT_SCHEMA = "rmct-convergence-completion-receipt-v1"
DECISION_RECEIPT_SCHEMA = "rmct-convergence-checkpoint-window-decision-v1"

UPDATES_PER_SEGMENT = 16
PERTURBATION_INDICES = ("1", "2")
BIAS_BY_PERTURBATION = {"1": "wrong_argument", "2": "suggested_answer"}
DEFAULT_MIN_COVERAGE = Decimal("0.85")
DEFAULT_MAX_WEIGHTED_ABS_GAP = Decimal("0.10")
DEFAULT_MAX_ABS_CHANGE = Decimal("0.01")
DEFAULT_MAX_OPTIMIZER_STEPS = 512

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DECISION_FILENAME_RE = re.compile(
    r"checkpoint-window-decision-s(?P<segment>[0-9]+)-(?P<sha>[0-9a-f]{64})\.json\Z"
)


class ConvergenceError(ValueError):
    """A checkpoint-window input or receipt cannot safely control training."""


class SourceMetricsError(ConvergenceError):
    """A segment metric source is missing, partial, or inconsistent."""


class DecisionReceiptError(ConvergenceError):
    """A persisted convergence decision cannot be replayed safely."""


class NonContinuationDecision(DecisionReceiptError):
    """A valid terminal receipt was presented to a training-only successor."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_object(payload: bytes, *, label: str, error_type: type[ConvergenceError] = SourceMetricsError) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise error_type(f"{label} must be one finite JSON object") from exc
    if not isinstance(value, dict):
        raise error_type(f"{label} must be a JSON object")
    return value


def _regular_file(
    value: str | Path, *, label: str, error_type: type[ConvergenceError] = SourceMetricsError
) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise error_type(f"{label} must be a regular file: {path}")
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise error_type(f"{label} must resolve to a regular file: {path}")
    return resolved


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
    error_type: type[ConvergenceError] = SourceMetricsError,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - (optional or set()))
    if missing or unknown:
        parts: list[str] = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise error_type(f"{label} has {'; '.join(parts)}")


def _mapping(value: Any, *, label: str, error_type: type[ConvergenceError] = SourceMetricsError) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise error_type(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, *, label: str, error_type: type[ConvergenceError] = SourceMetricsError) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{label} must be a non-empty string")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    error_type: type[ConvergenceError] = SourceMetricsError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise error_type(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise error_type(f"{label} must be at least {minimum}")
    return value


def _decimal(
    value: Any,
    *,
    label: str,
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    error_type: type[ConvergenceError] = SourceMetricsError,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise error_type(f"{label} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise error_type(f"{label} must be finite")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise error_type(f"{label} must be a finite decimal") from exc
    if not result.is_finite():
        raise error_type(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise error_type(f"{label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise error_type(f"{label} must be at most {maximum}")
    return result


def _decimal_float(value: Decimal) -> float:
    """Use JSON numbers in receipts while doing all decision math in Decimal."""

    return float(value)


def _identity(path: str | Path, *, label: str, error_type: type[ConvergenceError] = SourceMetricsError) -> dict[str, str]:
    resolved = _regular_file(path, label=label, error_type=error_type)
    return {"content_sha256": _sha256(resolved.read_bytes()), "path": str(resolved)}


def _validate_identity(
    value: Any, *, label: str, error_type: type[ConvergenceError] = DecisionReceiptError
) -> tuple[Path, dict[str, str]]:
    record = _mapping(value, label=label, error_type=error_type)
    _require_exact_keys(record, label=label, required={"content_sha256", "path"}, error_type=error_type)
    digest = record["content_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise error_type(f"{label}.content_sha256 must be a lowercase SHA-256 digest")
    path = _regular_file(_nonempty_string(record["path"], label=f"{label}.path", error_type=error_type), label=label, error_type=error_type)
    actual = _sha256(path.read_bytes())
    if actual != digest:
        raise error_type(f"{label} changed after the decision receipt was written: {path}")
    return path, {"content_sha256": actual, "path": str(path)}


@dataclass(frozen=True, slots=True)
class WindowThresholds:
    """The fixed decision thresholds and sealed checkpoint geometry."""

    minimum_coverage: Decimal | float | str = DEFAULT_MIN_COVERAGE
    maximum_weighted_abs_gap: Decimal | float | str = DEFAULT_MAX_WEIGHTED_ABS_GAP
    maximum_abs_change: Decimal | float | str = DEFAULT_MAX_ABS_CHANGE
    max_optimizer_steps: int = DEFAULT_MAX_OPTIMIZER_STEPS

    def __post_init__(self) -> None:
        minimum = _decimal(self.minimum_coverage, label="minimum_coverage", minimum=Decimal("0"), maximum=Decimal("1"))
        weighted = _decimal(
            self.maximum_weighted_abs_gap,
            label="maximum_weighted_abs_gap",
            minimum=Decimal("0"),
        )
        change = _decimal(self.maximum_abs_change, label="maximum_abs_change", minimum=Decimal("0"))
        cap = _integer(self.max_optimizer_steps, label="max_optimizer_steps", minimum=UPDATES_PER_SEGMENT)
        if cap % UPDATES_PER_SEGMENT:
            raise SourceMetricsError(f"max_optimizer_steps must be divisible by {UPDATES_PER_SEGMENT}")
        object.__setattr__(self, "minimum_coverage", minimum)
        object.__setattr__(self, "maximum_weighted_abs_gap", weighted)
        object.__setattr__(self, "maximum_abs_change", change)

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_optimizer_steps": self.max_optimizer_steps,
            "maximum_abs_change": _decimal_float(self.maximum_abs_change),
            "maximum_weighted_abs_gap": _decimal_float(self.maximum_weighted_abs_gap),
            "minimum_coverage": _decimal_float(self.minimum_coverage),
            "perturbation_indices": list(PERTURBATION_INDICES),
            "updates_per_segment": UPDATES_PER_SEGMENT,
        }

    @classmethod
    def from_document(cls, value: Any, *, error_type: type[ConvergenceError] = DecisionReceiptError) -> "WindowThresholds":
        record = _mapping(value, label="threshold_config", error_type=error_type)
        required = {
            "max_optimizer_steps",
            "maximum_abs_change",
            "maximum_weighted_abs_gap",
            "minimum_coverage",
            "perturbation_indices",
            "updates_per_segment",
        }
        _require_exact_keys(record, label="threshold_config", required=required, error_type=error_type)
        if record["perturbation_indices"] != list(PERTURBATION_INDICES):
            raise error_type("threshold_config must use perturbation indices [1, 2]")
        if record["updates_per_segment"] != UPDATES_PER_SEGMENT:
            raise error_type(f"threshold_config.updates_per_segment must be {UPDATES_PER_SEGMENT}")
        try:
            config = cls(
                minimum_coverage=record["minimum_coverage"],
                maximum_weighted_abs_gap=record["maximum_weighted_abs_gap"],
                maximum_abs_change=record["maximum_abs_change"],
                max_optimizer_steps=record["max_optimizer_steps"],
            )
        except ConvergenceError as exc:
            raise error_type(str(exc)) from exc
        if config.as_dict() != dict(record):
            raise error_type("threshold_config is not canonical")
        return config


# The longer name is convenient in plans and keeps the public API readable.
CheckpointWindowConfig = WindowThresholds


@dataclass(frozen=True, slots=True)
class SegmentIdentity:
    segment_index: int
    optimizer_step_start: int
    optimizer_step_end: int

    @property
    def optimizer_steps(self) -> int:
        return UPDATES_PER_SEGMENT

    def as_dict(self) -> dict[str, int]:
        return {
            "optimizer_step_end": self.optimizer_step_end,
            "optimizer_step_start": self.optimizer_step_start,
            "optimizer_steps": UPDATES_PER_SEGMENT,
            "segment_index": self.segment_index,
        }

    @classmethod
    def parse(cls, value: Any, *, label: str = "segment") -> "SegmentIdentity":
        record = _mapping(value, label=label)
        _require_exact_keys(
            record,
            label=label,
            required={"segment_index", "optimizer_step_start", "optimizer_step_end", "optimizer_steps"},
        )
        index = _integer(record["segment_index"], label=f"{label}.segment_index", minimum=0)
        start = _integer(record["optimizer_step_start"], label=f"{label}.optimizer_step_start", minimum=1)
        end = _integer(record["optimizer_step_end"], label=f"{label}.optimizer_step_end", minimum=1)
        updates = _integer(record["optimizer_steps"], label=f"{label}.optimizer_steps", minimum=1)
        expected_start = index * UPDATES_PER_SEGMENT + 1
        expected_end = (index + 1) * UPDATES_PER_SEGMENT
        if updates != UPDATES_PER_SEGMENT or start != expected_start or end != expected_end:
            raise SourceMetricsError(
                f"{label} must be consecutive sealed optimizer steps {expected_start}..{expected_end} "
                f"with optimizer_steps={UPDATES_PER_SEGMENT}"
            )
        return cls(segment_index=index, optimizer_step_start=start, optimizer_step_end=end)


@dataclass(frozen=True, slots=True)
class Coverage:
    parse_count: int
    valid_count: int
    total_count: int

    @property
    def parse_coverage(self) -> Decimal:
        return Decimal(self.parse_count) / Decimal(self.total_count)

    @property
    def valid_coverage(self) -> Decimal:
        return Decimal(self.valid_count) / Decimal(self.total_count)

    def as_dict(self, *, config: WindowThresholds) -> dict[str, Any]:
        parse_coverage = self.parse_coverage
        valid_coverage = self.valid_coverage
        return {
            "parse_count": self.parse_count,
            "parse_coverage": _decimal_float(parse_coverage),
            "passes": parse_coverage >= config.minimum_coverage and valid_coverage >= config.minimum_coverage,
            "total_count": self.total_count,
            "valid_count": self.valid_count,
            "valid_coverage": _decimal_float(valid_coverage),
        }


@dataclass(frozen=True, slots=True)
class SegmentSummary:
    identity: SegmentIdentity
    absolute_sum: Decimal
    absolute_count: int
    pooled_coverage: Coverage
    coverage_by_bias: Mapping[str, Coverage]

    @property
    def weighted_abs_gap(self) -> Decimal:
        return self.absolute_sum / Decimal(self.absolute_count)

    def as_dict(self, *, config: WindowThresholds) -> dict[str, Any]:
        return {
            "absolute_count": self.absolute_count,
            "absolute_sum": _decimal_float(self.absolute_sum),
            "coverage": {
                "by_bias": {name: self.coverage_by_bias[name].as_dict(config=config) for name in sorted(self.coverage_by_bias)},
                "pooled": self.pooled_coverage.as_dict(config=config),
            },
            "segment": self.identity.as_dict(),
            "weighted_abs_gap": _decimal_float(self.weighted_abs_gap),
        }


@dataclass(frozen=True, slots=True)
class ParsedSource:
    identity: SegmentIdentity
    summary: SegmentSummary


@dataclass(frozen=True, slots=True)
class PublishedDecision:
    path: Path
    sha256: str
    status: str
    decision: str
    receipt: Mapping[str, Any]

    @property
    def guard_exit_code(self) -> int:
        return 1 if self.decision == "fail" else 0


def _parse_coverage(value: Any, *, label: str) -> Coverage:
    record = _mapping(value, label=label)
    _require_exact_keys(record, label=label, required={"parse_count", "valid_count", "total_count"})
    parsed = _integer(record["parse_count"], label=f"{label}.parse_count", minimum=0)
    valid = _integer(record["valid_count"], label=f"{label}.valid_count", minimum=0)
    total = _integer(record["total_count"], label=f"{label}.total_count", minimum=1)
    if valid > parsed or parsed > total:
        raise SourceMetricsError(f"{label} must satisfy valid_count <= parse_count <= total_count")
    return Coverage(parse_count=parsed, valid_count=valid, total_count=total)


def parse_source_metrics(document: Mapping[str, Any]) -> ParsedSource:
    """Validate and aggregate one strict 16-update source metrics document."""

    document = _mapping(document, label="source metrics")
    _require_exact_keys(
        document,
        label="source metrics",
        required={"schema", "segment", "updates"},
        optional={"provenance"},
    )
    if document["schema"] != SOURCE_METRICS_SCHEMA:
        raise SourceMetricsError(f"source metrics.schema must be {SOURCE_METRICS_SCHEMA!r}")
    if "provenance" in document:
        provenance = _mapping(document["provenance"], label="source metrics.provenance")
        _require_exact_keys(
            provenance,
            label="source metrics.provenance",
            required={"metrics_jsonl"},
        )
        metrics_path, _ = _validate_identity(
            provenance["metrics_jsonl"],
            label="source metrics.provenance.metrics_jsonl",
            error_type=SourceMetricsError,
        )
        # The concrete input must be a JSONL metrics file, rather than a receipt
        # copied under an arbitrary name.  Its bytes are independently hash-bound
        # above; extraction owns the semantic line-by-line validation.
        if metrics_path.suffix != ".jsonl":
            raise SourceMetricsError("source metrics provenance must bind a .jsonl trainer metrics file")
    identity = SegmentIdentity.parse(document["segment"])
    raw_updates = document["updates"]
    if not isinstance(raw_updates, list) or len(raw_updates) != UPDATES_PER_SEGMENT:
        raise SourceMetricsError(f"source metrics.updates must contain exactly {UPDATES_PER_SEGMENT} records")

    absolute_sum = Decimal("0")
    absolute_count = 0
    pooled_parse = pooled_valid = pooled_total = 0
    bias_counts: dict[str, list[int]] = {}
    expected_biases: set[str] | None = None
    for offset, raw_update in enumerate(raw_updates):
        label = f"source metrics.updates[{offset}]"
        update = _mapping(raw_update, label=label)
        _require_exact_keys(update, label=label, required={"optimizer_step", "batch_count", "perturbations"})
        expected_step = identity.optimizer_step_start + offset
        optimizer_step = _integer(update["optimizer_step"], label=f"{label}.optimizer_step", minimum=1)
        if optimizer_step != expected_step:
            raise SourceMetricsError(
                f"{label}.optimizer_step must be {expected_step}; missing, duplicate, or out-of-order optimizer steps are unsafe"
            )
        # An empty data batch would otherwise look like a very good zero-gap update.
        _integer(update["batch_count"], label=f"{label}.batch_count", minimum=1)
        perturbations = _mapping(update["perturbations"], label=f"{label}.perturbations")
        _require_exact_keys(perturbations, label=f"{label}.perturbations", required=set(PERTURBATION_INDICES))
        update_biases: set[str] = set()
        update_pooled_parse = update_pooled_valid = update_pooled_total = 0
        for perturbation_index in PERTURBATION_INDICES:
            perturbation_label = f"{label}.perturbations[{perturbation_index!r}]"
            perturbation = _mapping(perturbations[perturbation_index], label=perturbation_label)
            _require_exact_keys(
                perturbation,
                label=perturbation_label,
                required={"abs_sum", "abs_count", "parse_count", "valid_count", "total_count", "biases"},
            )
            current_absolute_sum = _decimal(
                perturbation["abs_sum"], label=f"{perturbation_label}.abs_sum", minimum=Decimal("0")
            )
            current_absolute_count = _integer(
                perturbation["abs_count"], label=f"{perturbation_label}.abs_count", minimum=0
            )
            if current_absolute_sum > current_absolute_count:
                raise SourceMetricsError(f"{perturbation_label}.abs_sum must not exceed abs_count")
            coverage = _parse_coverage(
                {
                    "parse_count": perturbation["parse_count"],
                    "valid_count": perturbation["valid_count"],
                    "total_count": perturbation["total_count"],
                },
                label=perturbation_label,
            )
            biases = _mapping(perturbation["biases"], label=f"{perturbation_label}.biases")
            if not biases:
                raise SourceMetricsError(f"{perturbation_label}.biases must not be empty")
            names: set[str] = set()
            bias_parse = bias_valid = bias_total = 0
            for bias_name, raw_bias in biases.items():
                name = _nonempty_string(bias_name, label=f"{perturbation_label}.biases key")
                names.add(name)
                bias = _parse_coverage(raw_bias, label=f"{perturbation_label}.biases[{name!r}]")
                bias_parse += bias.parse_count
                bias_valid += bias.valid_count
                bias_total += bias.total_count
                counts = bias_counts.setdefault(name, [0, 0, 0])
                counts[0] += bias.parse_count
                counts[1] += bias.valid_count
                counts[2] += bias.total_count
            if (bias_parse, bias_valid, bias_total) != (
                coverage.parse_count,
                coverage.valid_count,
                coverage.total_count,
            ):
                raise SourceMetricsError(
                    f"{perturbation_label}.biases must partition its pooled parse/valid/total counts"
                )
            expected_bias = BIAS_BY_PERTURBATION[perturbation_index]
            if names != {expected_bias}:
                raise SourceMetricsError(
                    f"{perturbation_label}.biases must contain only the canonical bias {expected_bias!r}"
                )
            update_biases.update(names)
            absolute_sum += current_absolute_sum
            absolute_count += current_absolute_count
            update_pooled_parse += coverage.parse_count
            update_pooled_valid += coverage.valid_count
            update_pooled_total += coverage.total_count
        if update_biases != set(BIAS_BY_PERTURBATION.values()):
            raise SourceMetricsError(f"{label} has partial bias conditions")
        if expected_biases is None:
            expected_biases = update_biases
        elif update_biases != expected_biases:
            raise SourceMetricsError("source metrics have partial or mismatched bias conditions across optimizer updates")
        pooled_parse += update_pooled_parse
        pooled_valid += update_pooled_valid
        pooled_total += update_pooled_total

    if absolute_count == 0:
        raise SourceMetricsError("source metrics have no absolute-gap observations across perturbation indices 1 and 2")
    if not expected_biases:
        raise SourceMetricsError("source metrics have no complete bias conditions")
    summary = SegmentSummary(
        identity=identity,
        absolute_sum=absolute_sum,
        absolute_count=absolute_count,
        pooled_coverage=Coverage(parse_count=pooled_parse, valid_count=pooled_valid, total_count=pooled_total),
        coverage_by_bias={
            name: Coverage(parse_count=counts[0], valid_count=counts[1], total_count=counts[2])
            for name, counts in sorted(bias_counts.items())
        },
    )
    return ParsedSource(identity=identity, summary=summary)


def _load_source(path: str | Path, *, error_type: type[ConvergenceError] = SourceMetricsError) -> tuple[ParsedSource, dict[str, str]]:
    identity = _identity(path, label="source metrics", error_type=error_type)
    try:
        document = _json_object(Path(identity["path"]).read_bytes(), label="source metrics", error_type=error_type)
        parsed = parse_source_metrics(document)
    except SourceMetricsError as exc:
        if error_type is SourceMetricsError:
            raise
        raise error_type(str(exc)) from exc
    return parsed, identity


def _read_metrics_jsonl(path: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Read a canonical local trainer JSONL file without accepting blank records."""

    resolved = _regular_file(path, label="trainer metrics JSONL")
    payload = resolved.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise SourceMetricsError("trainer metrics JSONL must be non-empty and LF-terminated")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n") or line.endswith(b"\r\n") or not line[:-1].strip():
            raise SourceMetricsError(f"trainer metrics JSONL has a blank or non-canonical line at {line_number}")
        records.append(_json_object(line, label=f"trainer metrics JSONL line {line_number}"))
    return resolved, records


def _metric_number(record: Mapping[str, Any], key: str, *, step: int) -> Decimal:
    return _decimal(record[key], label=f"trainer metrics optimizer step {step} {key}", minimum=Decimal("0"))


def _metric_count(record: Mapping[str, Any], key: str, *, step: int, maximum: int) -> int:
    value = _integer(record[key], label=f"trainer metrics optimizer step {step} {key}", minimum=0)
    if value > maximum:
        raise SourceMetricsError(
            f"trainer metrics optimizer step {step} {key} exceeds the fixed per-bias batch size {maximum}"
        )
    return value


def extract_source_metrics(
    *,
    metrics_jsonl: str | Path,
    segment_index: int,
    output: str | Path,
) -> Path:
    """Extract one canonical source document from a local RL ``metrics.jsonl``.

    The RMCT setting fixes two batch items: ``wrong_argument`` at perturbation
    index 1 and ``suggested_answer`` at index 2.  A valid rate observation for
    either perturbation is the existing trainer's absolute-gap count, so the
    same count supplies parse and valid coverage.  Its planned denominator is
    two per optimizer update (32 per bias over a sealed segment).
    """

    identity = SegmentIdentity.parse(
        {
            "segment_index": segment_index,
            "optimizer_step_start": _integer(segment_index, label="segment_index", minimum=0) * UPDATES_PER_SEGMENT + 1,
            "optimizer_step_end": (_integer(segment_index, label="segment_index", minimum=0) + 1) * UPDATES_PER_SEGMENT,
            "optimizer_steps": UPDATES_PER_SEGMENT,
        }
    )
    metrics_path, records = _read_metrics_jsonl(metrics_jsonl)
    expected_steps = set(range(identity.optimizer_step_start, identity.optimizer_step_end + 1))
    metric_keys = {
        "train/optimizer_step",
        "train/skipped_empty_batch",
        *(
            key
            for index in PERTURBATION_INDICES
            for key in (
                f"train/consistency_gap_abs_sum_{index}",
                f"train/consistency_gap_abs_count_{index}",
                f"train/consistency_gap_abs_mean_{index}",
            )
        ),
    }
    selected: dict[int, dict[str, Any]] = {}
    for line_number, record in enumerate(records, start=1):
        present = set(record) & metric_keys
        if not present:
            continue
        if "train/optimizer_step" not in record:
            raise SourceMetricsError(
                f"trainer metrics line {line_number} has a partial convergence record without train/optimizer_step"
            )
        optimizer_step = _integer(
            record["train/optimizer_step"], label=f"trainer metrics line {line_number} train/optimizer_step", minimum=0
        )
        if optimizer_step not in expected_steps:
            raise SourceMetricsError(
                f"trainer metrics line {line_number} has convergence data outside sealed optimizer steps "
                f"{identity.optimizer_step_start}..{identity.optimizer_step_end}: {optimizer_step}"
            )
        if optimizer_step in selected:
            raise SourceMetricsError(f"trainer metrics JSONL has duplicate optimizer step {optimizer_step}")
        expected_next = identity.optimizer_step_start + len(selected)
        if optimizer_step != expected_next:
            raise SourceMetricsError(
                f"trainer metrics JSONL has out-of-order optimizer step {optimizer_step}; expected {expected_next}"
            )
        global_step = _integer(record.get("step"), label=f"trainer metrics line {line_number} step", minimum=1)
        if global_step != optimizer_step:
            raise SourceMetricsError(
                f"trainer metrics line {line_number} global step {global_step} does not equal optimizer step {optimizer_step}"
            )
        skipped = record.get("train/skipped_empty_batch")
        if skipped != 0 or isinstance(skipped, bool):
            raise SourceMetricsError(
                f"trainer metrics optimizer step {optimizer_step} is a skipped or empty batch, not a sealed optimizer mutation"
            )
        update: dict[str, Any] = {"optimizer_step": optimizer_step, "batch_count": 2, "perturbations": {}}
        for perturbation_index in PERTURBATION_INDICES:
            sum_key = f"train/consistency_gap_abs_sum_{perturbation_index}"
            count_key = f"train/consistency_gap_abs_count_{perturbation_index}"
            mean_key = f"train/consistency_gap_abs_mean_{perturbation_index}"
            if sum_key not in record or count_key not in record:
                raise SourceMetricsError(
                    f"trainer metrics optimizer step {optimizer_step} has partial {perturbation_index} absolute-gap fields"
                )
            absolute_sum = _metric_number(record, sum_key, step=optimizer_step)
            absolute_count = _metric_count(record, count_key, step=optimizer_step, maximum=2)
            if absolute_sum > absolute_count:
                raise SourceMetricsError(
                    f"trainer metrics optimizer step {optimizer_step} {sum_key} exceeds its observed count"
                )
            if absolute_count:
                if mean_key not in record:
                    raise SourceMetricsError(
                        f"trainer metrics optimizer step {optimizer_step} lacks {mean_key} for a nonzero count"
                    )
                mean = _metric_number(record, mean_key, step=optimizer_step)
                if mean > Decimal("1") or abs(mean - absolute_sum / Decimal(absolute_count)) > Decimal("1e-12"):
                    raise SourceMetricsError(
                        f"trainer metrics optimizer step {optimizer_step} {mean_key} does not equal sum/count"
                    )
            elif mean_key in record:
                raise SourceMetricsError(
                    f"trainer metrics optimizer step {optimizer_step} has an unavailable-gap mean despite zero count"
                )
            bias = BIAS_BY_PERTURBATION[perturbation_index]
            coverage = {"parse_count": absolute_count, "valid_count": absolute_count, "total_count": 2}
            update["perturbations"][perturbation_index] = {
                "abs_sum": _decimal_float(absolute_sum),
                "abs_count": absolute_count,
                **coverage,
                "biases": {bias: coverage},
            }
        selected[optimizer_step] = update
    missing = sorted(expected_steps - set(selected))
    if missing:
        raise SourceMetricsError(f"trainer metrics JSONL is missing sealed optimizer steps: {missing}")
    source = {
        "schema": SOURCE_METRICS_SCHEMA,
        "provenance": {"metrics_jsonl": _identity(metrics_path, label="trainer metrics JSONL")},
        "segment": identity.as_dict(),
        "updates": [selected[step] for step in range(identity.optimizer_step_start, identity.optimizer_step_end + 1)],
    }
    # Validate our own derived form before persistence; this protects the
    # producer/controller boundary from a future accidental schema drift.
    parse_source_metrics(source)
    destination = Path(output).resolve()
    _publish_immutable(destination, _canonical_json(source))
    return destination


def _validate_checkpoint_receipt(path: str | Path, *, identity: SegmentIdentity) -> dict[str, str]:
    file_identity = _identity(path, label="checkpoint receipt")
    document = _json_object(Path(file_identity["path"]).read_bytes(), label="checkpoint receipt")
    _require_exact_keys(document, label="checkpoint receipt", required={"schema", "segment_index", "optimizer_step", "sealed"})
    if document["schema"] != CHECKPOINT_RECEIPT_SCHEMA:
        raise SourceMetricsError(f"checkpoint receipt.schema must be {CHECKPOINT_RECEIPT_SCHEMA!r}")
    if document["sealed"] is not True:
        raise SourceMetricsError("checkpoint receipt must state sealed=true")
    if _integer(document["segment_index"], label="checkpoint receipt.segment_index", minimum=0) != identity.segment_index:
        raise SourceMetricsError("checkpoint receipt.segment_index does not match source metrics")
    if _integer(document["optimizer_step"], label="checkpoint receipt.optimizer_step", minimum=1) != identity.optimizer_step_end:
        raise SourceMetricsError("checkpoint receipt.optimizer_step does not match the sealed segment endpoint")
    return file_identity


def _validate_completion_receipt(path: str | Path, *, identity: SegmentIdentity) -> dict[str, str]:
    file_identity = _identity(path, label="completion receipt")
    document = _json_object(Path(file_identity["path"]).read_bytes(), label="completion receipt")
    _require_exact_keys(
        document,
        label="completion receipt",
        required={"schema", "segment_index", "optimizer_step_start", "optimizer_step_end", "optimizer_steps", "sealed"},
    )
    if document["schema"] != COMPLETION_RECEIPT_SCHEMA:
        raise SourceMetricsError(f"completion receipt.schema must be {COMPLETION_RECEIPT_SCHEMA!r}")
    if document["sealed"] is not True:
        raise SourceMetricsError("completion receipt must state sealed=true")
    try:
        completion_identity = SegmentIdentity.parse(
            {
                "segment_index": document["segment_index"],
                "optimizer_step_start": document["optimizer_step_start"],
                "optimizer_step_end": document["optimizer_step_end"],
                "optimizer_steps": document["optimizer_steps"],
            },
            label="completion receipt",
        )
    except SourceMetricsError:
        raise
    if completion_identity != identity:
        raise SourceMetricsError("completion receipt does not match source metric segment identity")
    return file_identity


def _coverage_passes(summary: SegmentSummary, *, config: WindowThresholds) -> bool:
    pooled = summary.pooled_coverage
    if pooled.parse_coverage < config.minimum_coverage or pooled.valid_coverage < config.minimum_coverage:
        return False
    return all(
        coverage.parse_coverage >= config.minimum_coverage and coverage.valid_coverage >= config.minimum_coverage
        for coverage in summary.coverage_by_bias.values()
    )


def _decision_afterok(decision: str) -> dict[str, Any]:
    if decision == "continue":
        return {"permit_training": True, "successor_action": "launch"}
    if decision in {"converged", "capped"}:
        return {"permit_training": False, "successor_action": "no_op"}
    if decision == "fail":
        return {"permit_training": False, "successor_action": "block"}
    raise AssertionError(f"unknown decision {decision!r}")


def _window_conditions(
    current: SegmentSummary,
    previous: SegmentSummary | None,
    *,
    config: WindowThresholds,
) -> tuple[str, dict[str, Any]]:
    if previous is None:
        return "continue", {
            "absolute_gap_change": None,
            "current_gap_at_or_below_threshold": None,
            "current_coverage_passes": None,
            "first_segment": True,
            "previous_coverage_passes": None,
            "stable_vs_previous": None,
        }
    if set(current.coverage_by_bias) != set(previous.coverage_by_bias):
        raise SourceMetricsError("current and predecessor segments do not have the same complete bias conditions")
    difference = abs(current.weighted_abs_gap - previous.weighted_abs_gap)
    current_coverage = _coverage_passes(current, config=config)
    previous_coverage = _coverage_passes(previous, config=config)
    gap_at_threshold = current.weighted_abs_gap <= config.maximum_weighted_abs_gap
    stable = difference <= config.maximum_abs_change
    eligible = current_coverage and previous_coverage and gap_at_threshold and stable
    if eligible:
        decision = "converged"
    elif current.identity.optimizer_step_end >= config.max_optimizer_steps:
        decision = "capped"
    else:
        decision = "continue"
    return decision, {
        "absolute_gap_change": _decimal_float(difference),
        "current_gap_at_or_below_threshold": gap_at_threshold,
        "current_coverage_passes": current_coverage,
        "first_segment": False,
        "previous_coverage_passes": previous_coverage,
        "stable_vs_previous": stable,
    }


def _predecessor_summary(
    path: str | Path,
    *,
    current: SegmentIdentity,
    config: WindowThresholds,
    _seen: set[Path] | None = None,
) -> tuple[SegmentSummary, dict[str, str]]:
    document = verify_decision_receipt(path)
    if document["decision"] != "continue":
        raise SourceMetricsError(
            f"cannot evaluate segment {current.segment_index} after terminal predecessor decision {document['decision']!r}"
        )
    previous_target = SegmentIdentity.parse(document["target"], label="predecessor target")
    if previous_target.segment_index != current.segment_index - 1 or previous_target.optimizer_step_end + 1 != current.optimizer_step_start:
        raise SourceMetricsError("predecessor receipt is not the immediately preceding sealed optimizer segment")
    predecessor_config = WindowThresholds.from_document(document["threshold_config"])
    if predecessor_config != config:
        raise SourceMetricsError("predecessor receipt uses a different threshold configuration")
    source_path, _ = _validate_identity(document["source_metrics"], label="predecessor source metrics")
    previous, _ = _load_source(source_path, error_type=DecisionReceiptError)
    return previous.summary, _identity(path, label="predecessor receipt", error_type=DecisionReceiptError)


def _build_receipt(
    *,
    source: ParsedSource,
    source_identity: dict[str, str],
    checkpoint_identity: dict[str, str],
    completion_identity: dict[str, str],
    predecessor_identity: dict[str, str] | None,
    previous_summary: SegmentSummary | None,
    config: WindowThresholds,
) -> dict[str, Any]:
    current = source.summary
    decision, conditions = _window_conditions(current, previous_summary, config=config)
    threshold_config = config.as_dict()
    return {
        "afterok": _decision_afterok(decision),
        "checkpoint_receipt": checkpoint_identity,
        "completion_receipt": completion_identity,
        "decision": decision,
        "predecessor_receipt": predecessor_identity,
        "schema": DECISION_RECEIPT_SCHEMA,
        "source_metrics": source_identity,
        "target": current.identity.as_dict(),
        "threshold_config": threshold_config,
        "threshold_config_sha256": _sha256(_canonical_json(threshold_config)),
        "window": {
            "conditions": conditions,
            "current": current.as_dict(config=config),
            "previous": None if previous_summary is None else previous_summary.as_dict(config=config),
        },
    }


def _decision_filename(segment_index: int, digest: str) -> str:
    return f"checkpoint-window-decision-s{segment_index:03d}-{digest}.json"


def _publish_immutable(path: Path, payload: bytes) -> str:
    """Atomically create a receipt once, or resume exact bytes."""

    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise DecisionReceiptError(f"refusing to overwrite different immutable receipt: {path}")
        return "resumed"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise DecisionReceiptError(f"decision output directory must be a regular directory: {path.parent}")
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
                raise DecisionReceiptError(f"immutable receipt appeared with different bytes: {path}") from None
            return "resumed"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "written"


def publish_decision_receipt(directory: str | Path, receipt: Mapping[str, Any]) -> PublishedDecision:
    """Content-address an already validated decision receipt exactly once."""

    target = SegmentIdentity.parse(receipt.get("target"), label="decision target")
    payload = _canonical_json(receipt)
    digest = _sha256(payload)
    output = Path(directory).resolve()
    if output.exists() and (output.is_symlink() or not output.is_dir()):
        raise DecisionReceiptError(f"decision output directory must be a regular directory: {output}")
    # A second, different decision for the same sealed boundary is ambiguous on
    # resume, even if both files are individually content addressed.
    if output.is_dir():
        prefix = f"checkpoint-window-decision-s{target.segment_index:03d}-"
        competitors = [item for item in output.iterdir() if item.name.startswith(prefix) and item.name != _decision_filename(target.segment_index, digest)]
        if competitors:
            raise DecisionReceiptError(
                f"refusing a competing decision receipt for sealed segment {target.segment_index}: {competitors[0]}"
            )
    path = output / _decision_filename(target.segment_index, digest)
    status = _publish_immutable(path, payload)
    return PublishedDecision(path=path, sha256=digest, status=status, decision=str(receipt["decision"]), receipt=dict(receipt))


def guard_from_paths(
    *,
    source_metrics: str | Path,
    checkpoint_receipt: str | Path,
    completion_receipt: str | Path,
    output_directory: str | Path,
    predecessor_receipt: str | Path | None = None,
    config: WindowThresholds = WindowThresholds(),
) -> PublishedDecision:
    """Evaluate one sealed segment and emit an immutable checkpoint-window receipt.

    Segment zero must not have a predecessor.  Every later segment must name
    the receipt for precisely its immediately preceding segment, which is
    replayed before it is trusted.
    """

    if not isinstance(config, WindowThresholds):
        raise SourceMetricsError("config must be a WindowThresholds instance")
    source, source_identity = _load_source(source_metrics)
    target = source.identity
    if target.optimizer_step_end > config.max_optimizer_steps:
        raise SourceMetricsError(
            f"segment endpoint {target.optimizer_step_end} exceeds max_optimizer_steps={config.max_optimizer_steps}"
        )
    checkpoint_identity = _validate_checkpoint_receipt(checkpoint_receipt, identity=target)
    completion_identity = _validate_completion_receipt(completion_receipt, identity=target)
    if target.segment_index == 0:
        if predecessor_receipt is not None:
            raise SourceMetricsError("first sealed segment must not have a predecessor receipt")
        predecessor_identity = None
        previous_summary = None
    else:
        if predecessor_receipt is None:
            raise SourceMetricsError("non-first sealed segment requires its immediately preceding receipt")
        previous_summary, predecessor_identity = _predecessor_summary(
            predecessor_receipt,
            current=target,
            config=config,
        )
    receipt = _build_receipt(
        source=source,
        source_identity=source_identity,
        checkpoint_identity=checkpoint_identity,
        completion_identity=completion_identity,
        predecessor_identity=predecessor_identity,
        previous_summary=previous_summary,
        config=config,
    )
    return publish_decision_receipt(output_directory, receipt)


def _load_decision_receipt(path: str | Path) -> tuple[dict[str, Any], Path, str]:
    resolved = _regular_file(path, label="decision receipt", error_type=DecisionReceiptError)
    match = _DECISION_FILENAME_RE.fullmatch(resolved.name)
    if match is None:
        raise DecisionReceiptError("decision receipt filename must be content-addressed as checkpoint-window-decision-sNNN-<sha>.json")
    payload = resolved.read_bytes()
    digest = _sha256(payload)
    if digest != match.group("sha"):
        raise DecisionReceiptError(f"decision receipt content SHA-256 disagrees with its filename: {resolved}")
    document = _json_object(payload, label="decision receipt", error_type=DecisionReceiptError)
    return document, resolved, digest


def _parse_decision_outer(document: Mapping[str, Any]) -> tuple[SegmentIdentity, WindowThresholds, str]:
    required = {
        "afterok",
        "checkpoint_receipt",
        "completion_receipt",
        "decision",
        "predecessor_receipt",
        "schema",
        "source_metrics",
        "target",
        "threshold_config",
        "threshold_config_sha256",
        "window",
    }
    _require_exact_keys(document, label="decision receipt", required=required, error_type=DecisionReceiptError)
    if document["schema"] != DECISION_RECEIPT_SCHEMA:
        raise DecisionReceiptError(f"decision receipt.schema must be {DECISION_RECEIPT_SCHEMA!r}")
    target = SegmentIdentity.parse(document["target"], label="decision target")
    config = WindowThresholds.from_document(document["threshold_config"])
    expected_config_sha = _sha256(_canonical_json(config.as_dict()))
    if document["threshold_config_sha256"] != expected_config_sha:
        raise DecisionReceiptError("decision receipt threshold configuration hash is inconsistent")
    decision = document["decision"]
    if decision not in {"continue", "converged", "capped", "fail"}:
        raise DecisionReceiptError("decision receipt has an invalid decision")
    afterok = _mapping(document["afterok"], label="decision receipt.afterok", error_type=DecisionReceiptError)
    _require_exact_keys(
        afterok,
        label="decision receipt.afterok",
        required={"permit_training", "successor_action"},
        error_type=DecisionReceiptError,
    )
    if dict(afterok) != _decision_afterok(str(decision)):
        raise DecisionReceiptError("decision receipt afterok policy does not match decision")
    return target, config, str(decision)


def verify_decision_receipt(path: str | Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Replay a receipt recursively and reject source, predecessor, or config drift."""

    document, resolved, _ = _load_decision_receipt(path)
    seen = set() if _seen is None else _seen
    if resolved in seen:
        raise DecisionReceiptError("decision receipt predecessor chain contains a cycle")
    seen.add(resolved)
    try:
        target, config, _ = _parse_decision_outer(document)
        source_path, source_identity = _validate_identity(document["source_metrics"], label="source metrics")
        try:
            source, actual_source_identity = _load_source(source_path, error_type=DecisionReceiptError)
        except DecisionReceiptError:
            raise
        if source_identity != actual_source_identity or source.identity != target:
            raise DecisionReceiptError("decision receipt source metrics do not match target")
        checkpoint_path, checkpoint_identity = _validate_identity(document["checkpoint_receipt"], label="checkpoint receipt")
        completion_path, completion_identity = _validate_identity(document["completion_receipt"], label="completion receipt")
        try:
            expected_checkpoint_identity = _validate_checkpoint_receipt(checkpoint_path, identity=target)
            expected_completion_identity = _validate_completion_receipt(completion_path, identity=target)
        except SourceMetricsError as exc:
            raise DecisionReceiptError(str(exc)) from exc
        if checkpoint_identity != expected_checkpoint_identity or completion_identity != expected_completion_identity:
            raise DecisionReceiptError("decision receipt checkpoint/completion identity is inconsistent")
        if target.optimizer_step_end > config.max_optimizer_steps:
            raise DecisionReceiptError("decision target exceeds its threshold configuration cap")
        if target.segment_index == 0:
            if document["predecessor_receipt"] is not None:
                raise DecisionReceiptError("first segment decision receipt must not bind a predecessor")
            predecessor_identity = None
            previous_summary = None
        else:
            predecessor_path, predecessor_identity = _validate_identity(
                document["predecessor_receipt"], label="predecessor receipt"
            )
            previous_summary, expected_predecessor_identity = _predecessor_summary(
                predecessor_path,
                current=target,
                config=config,
            )
            if predecessor_identity != expected_predecessor_identity:
                raise DecisionReceiptError("decision receipt predecessor identity is inconsistent")
        expected = _build_receipt(
            source=source,
            source_identity=actual_source_identity,
            checkpoint_identity=expected_checkpoint_identity,
            completion_identity=expected_completion_identity,
            predecessor_identity=predecessor_identity,
            previous_summary=previous_summary,
            config=config,
        )
        if expected != document:
            raise DecisionReceiptError("decision receipt does not match deterministic replay of its evidence")
        return document
    finally:
        seen.remove(resolved)


def successor_action(path: str | Path) -> dict[str, Any]:
    """Return a verified successor policy; terminal receipts intentionally no-op."""

    receipt = verify_decision_receipt(path)
    return dict(receipt["afterok"])


def require_continue(path: str | Path) -> dict[str, Any]:
    """Strict variant for callers that must launch only after a continue receipt."""

    receipt = verify_decision_receipt(path)
    if receipt["decision"] != "continue":
        raise NonContinuationDecision(f"decision {receipt['decision']!r} requires a successor no-op")
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    guard = subparsers.add_parser("guard", help="evaluate one sealed 16-optimizer-step checkpoint window")
    guard.add_argument("--source-metrics", required=True, type=Path)
    guard.add_argument("--checkpoint-receipt", required=True, type=Path)
    guard.add_argument("--completion-receipt", required=True, type=Path)
    guard.add_argument("--output-directory", required=True, type=Path)
    guard.add_argument("--predecessor-receipt", type=Path)
    guard.add_argument("--min-coverage", default=float(DEFAULT_MIN_COVERAGE), type=float)
    guard.add_argument("--max-weighted-abs-gap", default=float(DEFAULT_MAX_WEIGHTED_ABS_GAP), type=float)
    guard.add_argument("--max-abs-change", default=float(DEFAULT_MAX_ABS_CHANGE), type=float)
    guard.add_argument("--max-optimizer-steps", default=DEFAULT_MAX_OPTIMIZER_STEPS, type=int)

    extract = subparsers.add_parser(
        "extract-source", help="extract a strict 16-optimizer-step source document from local trainer metrics.jsonl"
    )
    extract.add_argument("--metrics-jsonl", required=True, type=Path)
    extract.add_argument("--segment-index", required=True, type=int)
    extract.add_argument("--output", required=True, type=Path)

    verify = subparsers.add_parser("verify-receipt", help="replay a checkpoint-window receipt")
    verify.add_argument("--receipt", required=True, type=Path)

    successor = subparsers.add_parser(
        "successor", help="replay a receipt and print launch/no-op policy; terminal receipts exit successfully"
    )
    successor.add_argument("--receipt", required=True, type=Path)

    continue_parser = subparsers.add_parser("require-continue", help="exit zero only for a verified continue receipt")
    continue_parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "guard":
            config = WindowThresholds(
                minimum_coverage=args.min_coverage,
                maximum_weighted_abs_gap=args.max_weighted_abs_gap,
                maximum_abs_change=args.max_abs_change,
                max_optimizer_steps=args.max_optimizer_steps,
            )
            result = guard_from_paths(
                source_metrics=args.source_metrics,
                checkpoint_receipt=args.checkpoint_receipt,
                completion_receipt=args.completion_receipt,
                output_directory=args.output_directory,
                predecessor_receipt=args.predecessor_receipt,
                config=config,
            )
            print(_canonical_json({"decision": result.decision, "path": str(result.path), "status": result.status}).decode(), end="")
            return result.guard_exit_code
        if args.command == "extract-source":
            output = extract_source_metrics(
                metrics_jsonl=args.metrics_jsonl,
                segment_index=args.segment_index,
                output=args.output,
            )
            print(_canonical_json({"path": str(output), "status": "written_or_resumed"}).decode(), end="")
            return 0
        if args.command == "verify-receipt":
            print(_canonical_json(verify_decision_receipt(args.receipt)).decode(), end="")
            return 0
        if args.command == "successor":
            print(_canonical_json(successor_action(args.receipt)).decode(), end="")
            return 0
        if args.command == "require-continue":
            require_continue(args.receipt)
            return 0
    except (ConvergenceError, OSError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"unknown command {args.command!r}")


if __name__ == "__main__":  # pragma: no cover - exercised through the module entry point
    raise SystemExit(main())


__all__ = [
    "CHECKPOINT_RECEIPT_SCHEMA",
    "COMPLETION_RECEIPT_SCHEMA",
    "DECISION_RECEIPT_SCHEMA",
    "SOURCE_METRICS_SCHEMA",
    "DEFAULT_MAX_ABS_CHANGE",
    "DEFAULT_MAX_OPTIMIZER_STEPS",
    "DEFAULT_MAX_WEIGHTED_ABS_GAP",
    "DEFAULT_MIN_COVERAGE",
    "PERTURBATION_INDICES",
    "UPDATES_PER_SEGMENT",
    "CheckpointWindowConfig",
    "ConvergenceError",
    "DecisionReceiptError",
    "extract_source_metrics",
    "NonContinuationDecision",
    "PublishedDecision",
    "SourceMetricsError",
    "WindowThresholds",
    "guard_from_paths",
    "parse_source_metrics",
    "publish_decision_receipt",
    "require_continue",
    "successor_action",
    "verify_decision_receipt",
]
