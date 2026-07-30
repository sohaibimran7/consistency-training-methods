"""Collect fail-closed model-selection observations from Inspect eval logs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Real
from pathlib import Path
from typing import Any

from ctm.artifacts import write_atomic_bytes
from scripts.irpan_2510_27062.analysis import (
    SCORED,
    SELECTION_DOMAINS,
    SELECTION_OBSERVATION_SCHEMA,
    UNSCORED,
    VALIDATION,
    BenchmarkRoute,
    CandidateLocator,
    get_benchmark_route,
    validate_selection_observation,
)
from scripts.irpan_2510_27062.mcq_bias_metrics import (
    MCQMetricAggregationError,
    aggregate_mcq_bias_sample_values,
)

_MISSING = object()


class SelectionLogError(ValueError):
    """An Inspect log cannot be mapped unambiguously to a selection observation."""


@dataclass(frozen=True, slots=True)
class _MetricResult:
    status: str
    value: float | None
    numerator: int
    denominator: int
    unscored_count: int
    unscored_reason: str | None


@dataclass(frozen=True, slots=True)
class _LogObservation:
    route_id: str
    candidate_id: str
    candidate_method: str
    identity: str
    timestamp: float | None
    log_name: str
    observation: dict[str, Any]


def collect_validation_observations(
    log_dir: str | Path,
    *,
    domain: str,
    method: str | None = None,
    schema: str = SELECTION_OBSERVATION_SCHEMA,
) -> list[dict[str, Any]]:
    """Read Inspect logs and return the latest typed row for each candidate route.

    Inspect is imported inside this function so inventory, selection, and all
    other offline CLI commands remain usable in dependency-minimal installs.
    """

    _validate_request(domain=domain, method=method, schema=schema)
    list_eval_logs, read_eval_log = _load_inspect_log_api()
    try:
        log_infos = list(list_eval_logs(str(log_dir)))
    except (OSError, TypeError, ValueError) as exc:
        raise SelectionLogError(f"could not list Inspect logs in {log_dir}: {exc}") from exc

    collected: list[_LogObservation] = []
    for log_info in sorted(log_infos, key=_log_name):
        name = _log_name(log_info)
        try:
            log = read_eval_log(log_info)
        except (OSError, TypeError, ValueError) as exc:
            raise SelectionLogError(f"could not read Inspect log {name}: {exc}") from exc
        record = _log_observation(
            log,
            log_info=log_info,
            log_name=name,
            requested_domain=domain,
            requested_method=method,
            schema=schema,
        )
        if record is not None:
            collected.append(record)

    if not collected:
        suffix = f" and method {method!r}" if method is not None else ""
        raise SelectionLogError(f"no successful validation logs matched domain {domain!r}{suffix} in {log_dir}")

    selected: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[_LogObservation]] = {}
    for record in collected:
        grouped.setdefault((record.candidate_id, record.route_id), []).append(record)
    for key in sorted(grouped):
        retries = grouped[key]
        identities = {retry.identity for retry in retries}
        if len(identities) != 1:
            names = sorted(retry.log_name for retry in retries)
            raise SelectionLogError(
                f"ambiguous successful retries for candidate {key[0]!r}, route {key[1]!r}: "
                f"candidate or route metadata differ across {names}"
            )
        if len(retries) == 1:
            selected.append(retries[0].observation)
            continue
        if any(retry.timestamp is None for retry in retries):
            names = sorted(retry.log_name for retry in retries)
            raise SelectionLogError(
                f"ambiguous successful retries for candidate {key[0]!r}, route {key[1]!r}: "
                f"a creation time is missing in {names}"
            )
        latest_timestamp = max(retry.timestamp for retry in retries if retry.timestamp is not None)
        latest = [retry for retry in retries if retry.timestamp == latest_timestamp]
        if len(latest) != 1:
            names = sorted(retry.log_name for retry in latest)
            raise SelectionLogError(f"latest successful retry tie for candidate {key[0]!r}, route {key[1]!r}: {names}")
        selected.append(latest[0].observation)

    selected.sort(key=_observation_sort_key)
    return selected


def materialize_validation_observations(
    log_dir: str | Path,
    output: str | Path,
    *,
    domain: str,
    method: str | None = None,
    schema: str = SELECTION_OBSERVATION_SCHEMA,
) -> list[dict[str, Any]]:
    """Publish deterministic JSONL observations without replacing prior output."""

    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite validation observations: {target}")
    observations = collect_validation_observations(
        log_dir,
        domain=domain,
        method=method,
        schema=schema,
    )
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in observations
    )
    write_atomic_bytes(target, payload)
    return observations


def _load_inspect_log_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from inspect_ai.log import list_eval_logs, read_eval_log
    except ImportError as exc:  # pragma: no cover - depends on the installed runtime
        raise SelectionLogError("Inspect AI is required to collect validation observations") from exc
    return list_eval_logs, read_eval_log


def _validate_request(*, domain: str, method: str | None, schema: str) -> None:
    if domain not in SELECTION_DOMAINS:
        raise SelectionLogError(f"domain must be one of {sorted(SELECTION_DOMAINS)}, got {domain!r}")
    if method is not None and (not isinstance(method, str) or not method or method != method.strip()):
        raise SelectionLogError("method must be None or a non-empty, exactly formatted string")
    if schema != SELECTION_OBSERVATION_SCHEMA:
        raise SelectionLogError(
            f"unsupported selection observation schema {schema!r}; expected {SELECTION_OBSERVATION_SCHEMA!r}"
        )


def _log_observation(
    log: Any,
    *,
    log_info: Any,
    log_name: str,
    requested_domain: str,
    requested_method: str | None,
    schema: str,
) -> _LogObservation | None:
    if _field(log, "status", None) != "success":
        return None
    eval_spec = _field(log, "eval", None)
    metadata = _field(eval_spec, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    candidate = metadata.get("selection_candidate")
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise SelectionLogError(f"{log_name}: eval.metadata.selection_candidate must be an object")
    if candidate.get("domain") != requested_domain:
        return None
    if requested_method is not None and candidate.get("method") != requested_method:
        return None
    if metadata.get("stage") != VALIDATION:
        return None

    candidate_method = candidate.get("method")
    if not isinstance(candidate_method, str) or not candidate_method or candidate_method != candidate_method.strip():
        raise SelectionLogError(f"{log_name}: selection_candidate.method must be a non-empty exact string")
    benchmark = _metadata_alias(metadata, "benchmark", "source", log_name=log_name)
    metric = _metadata_alias(metadata, "primary_metric", "metric", log_name=log_name)
    condition = metadata.get("condition")
    subset = metadata.get("subset")
    annotation_source = metadata.get("annotation_source")
    try:
        route = get_benchmark_route(
            benchmark,
            stage=VALIDATION,
            condition=condition,
            subset=subset,
            annotation_source=annotation_source,
        )
    except ValueError as exc:
        raise SelectionLogError(f"{log_name}: {exc}") from exc
    if route.domain != requested_domain or not route.selection_input:
        raise SelectionLogError(f"{log_name}: route {route.route_id!r} is not a {requested_domain!r} selection route")
    allowed_metrics = {route.metric}
    if route.derived_metric is not None:
        allowed_metrics.add(route.derived_metric)
    if metric not in allowed_metrics:
        raise SelectionLogError(
            f"{log_name}: route {route.route_id!r} requires one of {sorted(allowed_metrics)}, got {metric!r}"
        )

    details = candidate.get("candidate_details")
    if isinstance(details, Mapping) and "method" in details and details["method"] != candidate_method:
        raise SelectionLogError(f"{log_name}: selection_candidate.method conflicts with candidate_details.method")
    metric_result = _extract_metric(log, route=route, metric=metric)
    raw: dict[str, Any] = {
        "schema": schema,
        "domain": requested_domain,
        "candidate_id": candidate.get("candidate_id"),
        "candidate_locator": candidate.get("candidate_locator"),
        "candidate_details": details,
        "benchmark": benchmark,
        "stage": VALIDATION,
        "condition": condition,
        "metric": metric,
        "status": metric_result.status,
        "value": metric_result.value,
        "numerator": metric_result.numerator,
        "denominator": metric_result.denominator,
        "unscored_count": metric_result.unscored_count,
    }
    if subset is not None:
        raw["subset"] = subset
    if annotation_source is not None:
        raw["annotation_source"] = annotation_source
    if metric_result.unscored_reason is not None:
        raw["unscored_reason"] = metric_result.unscored_reason
    try:
        observation = _normalized_observation(raw)
    except ValueError as exc:
        raise SelectionLogError(f"{log_name}: invalid selection observation: {exc}") from exc

    identity = json.dumps(
        {
            "schema": schema,
            "domain": requested_domain,
            "candidate_id": observation["candidate_id"],
            "candidate_locator": observation["candidate_locator"],
            "candidate_details": observation["candidate_details"],
            "method": candidate_method,
            "benchmark": benchmark,
            "stage": VALIDATION,
            "condition": condition,
            "subset": subset,
            "annotation_source": annotation_source,
            "metric": metric,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _LogObservation(
        route_id=route.route_id,
        candidate_id=observation["candidate_id"],
        candidate_method=candidate_method,
        identity=identity,
        timestamp=_retry_timestamp(log, log_info=log_info, log_name=log_name),
        log_name=log_name,
        observation=observation,
    )


def _extract_metric(log: Any, *, route: BenchmarkRoute, metric: str) -> _MetricResult:
    if route.benchmark == "mmlu":
        native_result = _extract_native_mcq_bias_metric(log, route=route, metric=metric)
        if native_result is not None:
            return native_result

    results = _field(log, "results", None)
    raw_scores = _field(results, "scores", ())
    scores = raw_scores if isinstance(raw_scores, Sequence) and not isinstance(raw_scores, (str, bytes)) else ()
    metric_key = metric if route.benchmark == "mmlu" else "mean"
    matches: list[tuple[Any, Mapping[str, Any]]] = []
    for score in scores:
        score_metrics = _field(score, "metrics", None)
        if isinstance(score_metrics, Mapping) and metric_key in score_metrics:
            matches.append((score, score_metrics))
    if not matches:
        return _unscored_result(f"missing metric {metric_key!r}")
    if len(matches) != 1:
        return _unscored_result(f"ambiguous metric {metric_key!r}: found in {len(matches)} EvalScore values")

    score, score_metrics = matches[0]
    problems: list[str] = []
    raw_value = _metric_value(score_metrics[metric_key])
    value = _unit_interval_or_reason(raw_value, label=f"metric {metric_key!r}", problems=problems)

    unscored_counts = _count_values(
        (
            ("metrics.unscored_count", _mapping_metric(score_metrics, "unscored_count")),
            ("score.unscored_samples", _field(score, "unscored_samples", _MISSING)),
        ),
        problems=problems,
    )
    scored_counts = _count_values(
        (
            ("metrics.scored_count", _mapping_metric(score_metrics, "scored_count")),
            ("score.scored_samples", _field(score, "scored_samples", _MISSING)),
        ),
        problems=problems,
    )
    sample_counts = _count_values(
        (
            ("metrics.sample_count", _mapping_metric(score_metrics, "sample_count")),
            ("results.completed_samples", _field(results, "completed_samples", _MISSING)),
        ),
        problems=problems,
        ignore_zero_fallback=True,
    )
    fractions = _fraction_values(
        (("metrics.unscored_fraction", _mapping_metric(score_metrics, "unscored_fraction")),),
        problems=problems,
    )

    unscored_count = _consistent_count(unscored_counts, "unscored counts", problems)
    denominator = _consistent_count(scored_counts, "scored counts", problems)
    sample_count = _consistent_count(sample_counts, "sample counts", problems)
    unscored_fraction = _consistent_float(fractions, "unscored fractions", problems)
    if unscored_count is None and unscored_fraction is not None and sample_count is not None:
        inferred = unscored_fraction * sample_count
        rounded = round(inferred)
        if math.isclose(inferred, rounded, rel_tol=0.0, abs_tol=1e-9):
            unscored_count = int(rounded)
        else:
            problems.append("unscored_fraction is incompatible with sample_count")
    if unscored_count is None and unscored_fraction == 0.0:
        unscored_count = 0
    if denominator is None and sample_count is not None and unscored_count is not None:
        denominator = sample_count - unscored_count
        if denominator < 0:
            problems.append("unscored_count exceeds sample_count")
            denominator = None
    if unscored_count is None and denominator is not None and sample_count is not None:
        unscored_count = sample_count - denominator
        if unscored_count < 0:
            problems.append("scored count exceeds sample_count")
            unscored_count = None
    if unscored_count is None:
        unscored_count = 0
    if unscored_fraction is not None and unscored_fraction > 0.0:
        problems.append(f"unscored_fraction={unscored_fraction:g}")
    if unscored_count > 0:
        problems.append(f"unscored_count={unscored_count}")
    if denominator is None or denominator <= 0:
        problems.append("missing positive scored-sample denominator")

    if problems or value is None or denominator is None or denominator <= 0:
        reason = "; ".join(dict.fromkeys(problems)) or f"invalid metric {metric_key!r}"
        return _unscored_result(reason, unscored_count=max(unscored_count, 1))
    numerator_float = value * denominator
    numerator = round(numerator_float)
    if not math.isclose(numerator_float, numerator, rel_tol=1e-12, abs_tol=1e-9):
        return _unscored_result(
            f"metric {metric_key!r} is incompatible with scored-sample denominator {denominator}",
            unscored_count=1,
        )
    return _MetricResult(
        status=SCORED,
        value=value,
        numerator=int(numerator),
        denominator=denominator,
        unscored_count=0,
        unscored_reason=None,
    )


def _extract_native_mcq_bias_metric(
    log: Any,
    *,
    route: BenchmarkRoute,
    metric: str,
) -> _MetricResult | None:
    """Use native sample values, retaining fallback only for named legacy tasks."""

    raw_samples = _field(log, "samples", _MISSING)
    if raw_samples is _MISSING or raw_samples is None:
        return (
            None if _is_legacy_irpan_mmlu_log(log) else _unscored_result("native mcq_bias log is missing sample scores")
        )
    if not isinstance(raw_samples, Sequence) or isinstance(raw_samples, (str, bytes)):
        return _unscored_result("Inspect samples are not a sequence")
    completed_samples = _field(_field(log, "results", None), "completed_samples", _MISSING)
    if isinstance(completed_samples, bool) or not isinstance(completed_samples, int) or completed_samples < 0:
        return _unscored_result("native mcq_bias log is missing a valid completed-sample count")
    if completed_samples != len(raw_samples):
        return _unscored_result(
            f"native sample coverage mismatch: {len(raw_samples)} values for " f"{completed_samples} completed samples",
            unscored_count=max(completed_samples - len(raw_samples), 1),
        )

    values: list[Mapping[str, object]] = []
    missing_scorer = 0
    saw_native_scorer = False
    for index, sample in enumerate(raw_samples, start=1):
        raw_scores = _field(sample, "scores", None)
        if not isinstance(raw_scores, Mapping):
            missing_scorer += 1
            continue
        matches = [
            score
            for name, score in raw_scores.items()
            if isinstance(name, str) and name.split("/")[-1] == "mcq_bias_scorer"
        ]
        if not matches:
            missing_scorer += 1
            continue
        saw_native_scorer = True
        if len(matches) != 1:
            return _unscored_result(f"sample {index} has ambiguous mcq_bias_scorer values")

        metadata = _field(sample, "metadata", None)
        if not isinstance(metadata, Mapping):
            return _unscored_result(f"sample {index} is missing mcq_bias metadata")
        if route.condition == "clean":
            if metadata.get("variant") != "unbiased":
                return _unscored_result(f"sample {index} is not an unbiased mcq_bias variant")
        elif route.condition == "wrong_suggestion":
            if metadata.get("variant") != "biased" or metadata.get("bias_type") != "suggested_answer":
                return _unscored_result(f"sample {index} is not a biased suggested_answer variant")
        else:
            return _unscored_result(f"unsupported MMLU condition {route.condition!r}")

        raw_value = _field(matches[0], "value", _MISSING)
        if raw_value is _MISSING:
            as_dict = getattr(matches[0], "as_dict", None)
            raw_value = as_dict() if callable(as_dict) else _MISSING
        if not isinstance(raw_value, Mapping):
            return _unscored_result(f"sample {index} mcq_bias_scorer value is not a mapping")
        values.append(raw_value)

    if not saw_native_scorer:
        return (
            None
            if _is_legacy_irpan_mmlu_log(log)
            else _unscored_result("native mcq_bias log has no mcq_bias_scorer sample values")
        )
    if missing_scorer:
        return _unscored_result(
            f"{missing_scorer} of {len(raw_samples)} samples are missing mcq_bias_scorer",
            unscored_count=missing_scorer,
        )
    try:
        aggregate = aggregate_mcq_bias_sample_values(values, condition=route.condition)
    except MCQMetricAggregationError as exc:
        return _unscored_result(str(exc), unscored_count=max(len(values), 1))
    if aggregate.value is None:
        return _unscored_result(
            "all wrong-suggestion responses were unparsed",
            unscored_count=aggregate.unparsed_count,
        )
    numerator = aggregate.numerator
    value = aggregate.value
    if metric == route.derived_metric:
        numerator = aggregate.denominator - numerator
        value = 1.0 - value
    elif metric != route.metric:
        return _unscored_result(f"unsupported native MMLU metric {metric!r}")
    return _MetricResult(
        status=SCORED,
        value=value,
        numerator=numerator,
        denominator=aggregate.denominator,
        unscored_count=aggregate.unparsed_count if route.condition == "wrong_suggestion" else 0,
        unscored_reason=None,
    )


def _is_legacy_irpan_mmlu_log(log: Any) -> bool:
    task_name = str(_field(_field(log, "eval", None), "task", ""))
    return "scripts.irpan_2510_27062.mmlu_tasks" in task_name


def _unscored_result(reason: str, *, unscored_count: int = 1) -> _MetricResult:
    return _MetricResult(
        status=UNSCORED,
        value=None,
        numerator=0,
        denominator=0,
        unscored_count=max(unscored_count, 1),
        unscored_reason=reason,
    )


def _normalized_observation(raw: Mapping[str, Any]) -> dict[str, Any]:
    observation = validate_selection_observation(raw)
    if not isinstance(observation.candidate_locator, CandidateLocator):
        raise SelectionLogError("typed observation validator did not return a candidate locator")
    payload: dict[str, Any] = {
        "schema": observation.schema,
        "domain": observation.domain,
        "candidate_id": observation.candidate_id,
        "candidate_locator": observation.candidate_locator.as_dict(),
        "candidate_details": dict(observation.candidate_details or {}),
        "benchmark": observation.benchmark,
        "stage": observation.stage,
        "condition": observation.condition,
        "metric": observation.metric,
        "status": observation.status,
        "value": observation.value,
        "numerator": observation.numerator,
        "denominator": observation.denominator,
        "unscored_count": observation.unscored_count,
    }
    if observation.subset is not None:
        payload["subset"] = observation.subset
    if observation.annotation_source is not None:
        payload["annotation_source"] = observation.annotation_source
    if observation.unscored_reason is not None:
        payload["unscored_reason"] = observation.unscored_reason
    validate_selection_observation(payload)
    return payload


def _metadata_alias(metadata: Mapping[str, Any], first: str, second: str, *, log_name: str) -> str:
    values = [(key, metadata[key]) for key in (first, second) if metadata.get(key) is not None]
    if not values:
        raise SelectionLogError(f"{log_name}: eval metadata requires {first!r} or {second!r}")
    if any(value != values[0][1] for _, value in values[1:]):
        raise SelectionLogError(f"{log_name}: eval metadata fields {first!r} and {second!r} conflict")
    value = values[0][1]
    if not isinstance(value, str) or not value or value != value.strip():
        raise SelectionLogError(f"{log_name}: eval metadata {values[0][0]!r} must be a non-empty exact string")
    return value


def _retry_timestamp(log: Any, *, log_info: Any, log_name: str) -> float | None:
    eval_spec = _field(log, "eval", None)
    raw = _field(eval_spec, "created", _MISSING)
    if raw is _MISSING or raw in (None, ""):
        raw = _field(_field(log, "stats", None), "started_at", _MISSING)
    if raw is _MISSING or raw in (None, ""):
        raw = _field(log_info, "mtime", _MISSING)
    if raw is _MISSING or raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        parsed = raw
    elif isinstance(raw, Real) and not isinstance(raw, bool):
        timestamp = float(raw)
        if not math.isfinite(timestamp):
            raise SelectionLogError(f"{log_name}: retry timestamp must be finite")
        return timestamp
    elif isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise SelectionLogError(f"{log_name}: invalid Inspect creation timestamp {raw!r}") from exc
    else:
        raise SelectionLogError(f"{log_name}: invalid Inspect creation timestamp {raw!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _mapping_metric(metrics: Mapping[str, Any], key: str) -> Any:
    if key not in metrics:
        return _MISSING
    return _metric_value(metrics[key])


def _metric_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("value", _MISSING)
    nested = _field(value, "value", _MISSING)
    return value if nested is _MISSING else nested


def _unit_interval_or_reason(raw: Any, *, label: str, problems: list[str]) -> float | None:
    if isinstance(raw, bool) or not isinstance(raw, Real):
        problems.append(f"{label} is not numeric")
        return None
    value = float(raw)
    if not math.isfinite(value):
        problems.append(f"{label} is non-finite")
        return None
    if not 0.0 <= value <= 1.0:
        problems.append(f"{label} is outside [0, 1]")
        return None
    return value


def _count_values(
    values: Sequence[tuple[str, Any]],
    *,
    problems: list[str],
    ignore_zero_fallback: bool = False,
) -> list[int]:
    normalized: list[int] = []
    for label, raw in values:
        if raw is _MISSING or raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, Real) or not math.isfinite(float(raw)):
            problems.append(f"{label} is not a non-negative integer")
            continue
        integer = int(raw)
        if float(raw) != integer or integer < 0:
            problems.append(f"{label} is not a non-negative integer")
            continue
        if ignore_zero_fallback and integer == 0:
            continue
        normalized.append(integer)
    return normalized


def _fraction_values(values: Sequence[tuple[str, Any]], *, problems: list[str]) -> list[float]:
    normalized: list[float] = []
    for label, raw in values:
        if raw is _MISSING or raw is None:
            continue
        value = _unit_interval_or_reason(raw, label=label, problems=problems)
        if value is not None:
            normalized.append(value)
    return normalized


def _consistent_count(values: Sequence[int], label: str, problems: list[str]) -> int | None:
    if not values:
        return None
    if len(set(values)) != 1:
        problems.append(f"conflicting {label}: {sorted(set(values))}")
    return values[0]


def _consistent_float(values: Sequence[float], label: str, problems: list[str]) -> float | None:
    if not values:
        return None
    if any(not math.isclose(value, values[0], rel_tol=1e-12, abs_tol=1e-12) for value in values[1:]):
        problems.append(f"conflicting {label}")
    return values[0]


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _log_name(log_info: Any) -> str:
    name = _field(log_info, "name", None)
    return name if isinstance(name, str) and name else str(log_info)


def _observation_sort_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        row["candidate_id"],
        row["benchmark"],
        row.get("condition") or "",
        row.get("subset") or "",
        row.get("annotation_source") or "",
        row["metric"],
    )


__all__ = [
    "SelectionLogError",
    "collect_validation_observations",
    "materialize_validation_observations",
]
