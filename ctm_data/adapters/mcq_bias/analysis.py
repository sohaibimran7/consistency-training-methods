"""Aggregate official ``mcq_bias`` Inspect logs into chart-ready JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from ctm.artifacts import write_atomic_bytes


@dataclass(frozen=True)
class _Summary:
    mean: float
    stderr: float
    n: int
    estimate_method: str
    stderr_method: str
    source_mean_metric: str | None = None
    source_stderr_metric: str | None = None


def _attribute(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _metric_value(sample: Any, metric: str) -> float | None:
    for score in (getattr(sample, "scores", None) or {}).values():
        value = getattr(score, "value", None)
        if isinstance(value, Mapping) and metric in value:
            candidate = value[metric]
        else:
            continue
        return _finite_float(candidate)
    return None


_PREDICATE_OPERATORS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "is_finite", "is_missing"})


def _validated_predicate(predicate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a declarative per-sample metric predicate."""

    if not isinstance(predicate, Mapping) or not predicate:
        raise ValueError("where predicate must be a non-empty object")
    compound = [key for key in ("all", "any", "not") if key in predicate]
    if compound:
        if len(compound) != 1 or len(predicate) != 1:
            raise ValueError("where predicate must contain exactly one of all, any, not, or a metric comparison")
        operator = compound[0]
        value = predicate[operator]
        if operator == "not":
            if not isinstance(value, Mapping):
                raise ValueError("where.not must be an object")
            return {"not": _validated_predicate(value)}
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
            raise ValueError(f"where.{operator} must be a non-empty array")
        return {operator: [_validated_predicate(item) for item in value]}

    allowed = {"metric", "op", "value"}
    unknown = sorted(set(predicate) - allowed)
    if unknown:
        raise ValueError(f"where comparison has unknown fields: {unknown}")
    metric = predicate.get("metric")
    operator = predicate.get("op", "eq")
    if not isinstance(metric, str) or not metric:
        raise ValueError("where comparison metric must be a non-empty string")
    if operator not in _PREDICATE_OPERATORS:
        raise ValueError(f"where comparison op must be one of {sorted(_PREDICATE_OPERATORS)}")
    if operator not in {"is_finite", "is_missing"} and "value" not in predicate:
        raise ValueError(f"where comparison op {operator!r} requires value")
    if operator in {"in", "not_in"}:
        value = predicate.get("value")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"where comparison op {operator!r} requires an array value")
        normalized_value: Any = [_finite_float(item) for item in value]
        if any(item is None for item in normalized_value):
            raise ValueError(f"where comparison op {operator!r} requires finite numeric values")
    elif operator in {"is_finite", "is_missing"}:
        normalized_value = None
    else:
        normalized_value = _finite_float(predicate.get("value"))
        if normalized_value is None:
            raise ValueError(f"where comparison op {operator!r} requires a finite numeric value")
    result = {"metric": metric, "op": operator}
    if operator not in {"is_finite", "is_missing"}:
        result["value"] = normalized_value
    return result


def _sample_matches(sample: Any, predicate: Mapping[str, Any]) -> bool:
    if "all" in predicate:
        return all(_sample_matches(sample, item) for item in predicate["all"])
    if "any" in predicate:
        return any(_sample_matches(sample, item) for item in predicate["any"])
    if "not" in predicate:
        return not _sample_matches(sample, predicate["not"])

    value = _metric_value(sample, str(predicate["metric"]))
    operator = str(predicate["op"])
    target = predicate.get("value")
    if operator == "is_missing":
        return value is None
    if operator == "is_finite":
        return value is not None
    if value is None:
        return False
    if operator == "eq":
        return value == target
    if operator == "ne":
        return value != target
    if operator == "lt":
        return value < target
    if operator == "le":
        return value <= target
    if operator == "gt":
        return value > target
    if operator == "ge":
        return value >= target
    if operator == "in":
        return value in target
    if operator == "not_in":
        return value not in target
    raise AssertionError(f"unhandled predicate operator: {operator}")


def _inspect_summary(log: Any, metric: str) -> _Summary | None:
    """Read an atomic estimate from Inspect's ``EvalResults`` when present."""

    results = _attribute(log, "results")
    for score in _attribute(results, "scores", []) or []:
        if str(_attribute(score, "name", "")) != metric:
            continue
        metrics = _attribute(score, "metrics", {}) or {}
        mean_name = next(
            (key for name in ("nanmean", "mean") if (key := _metric_result_key(metrics, name)) is not None),
            None,
        )
        stderr_name = next(
            (key for name in ("nanstderr", "stderr") if (key := _metric_result_key(metrics, name)) is not None),
            None,
        )
        if mean_name is None or stderr_name is None:
            continue
        mean = _finite_float(_attribute(metrics[mean_name], "value"))
        standard_error = _finite_float(_attribute(metrics[stderr_name], "value"))
        n = _attribute(score, "scored_samples")
        if mean is None or standard_error is None or not isinstance(n, int) or n <= 0:
            continue
        return _Summary(
            mean=mean,
            stderr=standard_error,
            n=n,
            estimate_method="inspect",
            stderr_method="inspect",
            source_mean_metric=mean_name,
            source_stderr_metric=stderr_name,
        )
    return None


def _metric_result_key(metrics: Mapping[str, Any], basename: str) -> str | None:
    """Match both Inspect's qualified and legacy unqualified metric keys."""

    return next((str(key) for key in metrics if str(key).rsplit("/", 1)[-1] == basename), None)


def _sample_summary(values: Sequence[float], method: Literal["sample", "binomial"]) -> _Summary:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    if method == "binomial":
        if any(value not in {0.0, 1.0} for value in values):
            raise ValueError("binomial stderr requires binary scores")
        standard_error = math.sqrt(mean * (1.0 - mean) / len(values))
    else:
        standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return _Summary(mean, standard_error, len(values), "sample", method)


def _pool_summaries(summaries: Sequence[_Summary]) -> _Summary:
    """Pool means and SEMs with sample-count weighting and total-variance propagation."""

    if not summaries:
        raise ValueError("cannot pool an empty collection")
    total = sum(summary.n for summary in summaries)
    mean = sum(summary.n * summary.mean for summary in summaries) / total
    stderr_methods = {summary.stderr_method for summary in summaries}
    if stderr_methods == {"binomial"}:
        standard_error = math.sqrt(mean * (1.0 - mean) / total)
    elif total == 1:
        standard_error = 0.0
    else:
        within_sse = sum(summary.n * (summary.n - 1) * summary.stderr**2 for summary in summaries if summary.n > 1)
        between_sse = sum(summary.n * (summary.mean - mean) ** 2 for summary in summaries)
        standard_error = math.sqrt(((within_sse + between_sse) / (total - 1)) / total)
    estimate_methods = {summary.estimate_method for summary in summaries}
    return _Summary(
        mean=mean,
        stderr=standard_error,
        n=total,
        estimate_method=next(iter(estimate_methods)) if len(estimate_methods) == 1 else "mixed",
        stderr_method=(
            next(iter(stderr_methods))
            if len(summaries) == 1 and len(stderr_methods) == 1
            else _pooled_method_label(stderr_methods)
        ),
        source_mean_metric=_common_value(summary.source_mean_metric for summary in summaries),
        source_stderr_metric=_common_value(summary.source_stderr_metric for summary in summaries),
    )


def _common_value(values: Iterable[str | None]) -> str | None:
    distinct = set(values)
    return next(iter(distinct)) if len(distinct) == 1 else None


def _pooled_method_label(methods: Iterable[str]) -> str:
    components = {component for method in methods for component in method.removeprefix("pooled:").split("+")}
    return f"pooled:{'+'.join(sorted(components))}"


def _task_identity(log: Any) -> tuple[str, str, str, str, int | None]:
    args = log.eval.task_args
    return (
        str(args.get("bias_type", "")),
        str(args.get("dataset", "")),
        str(args.get("prompt_style", "none")),
        str(args.get("seed", "42")),
        args.get("n_questions"),
    )


def aggregate_logs(
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
    metric: str,
    where_metric: str | None = None,
    where_value: float = 1.0,
    where: Mapping[str, Any] | None = None,
    stderr: str = "inspect",
    variant: str = "biased",
    metadata: Mapping[str, Any] | None = None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate one log group per condition.

    Within a group, only the latest successful attempt for each task identity is
    retained. Use :func:`aggregate_log_groups` when a condition has independent
    replicate directories that must be pooled.
    """

    return aggregate_log_groups(
        {condition: [logs] for condition, logs in logs_by_condition.items()},
        metric=metric,
        where_metric=where_metric,
        where_value=where_value,
        where=where,
        stderr=stderr,
        variant=variant,
        metadata=metadata,
        condition_metadata=condition_metadata,
    )


def aggregate_log_groups(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    metric: str,
    where_metric: str | None = None,
    where_value: float = 1.0,
    where: Mapping[str, Any] | None = None,
    stderr: str = "inspect",
    variant: str = "biased",
    metadata: Mapping[str, Any] | None = None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Pool independent log groups while deduplicating retries within each group.

    Unfiltered atomic estimates use Inspect's supplied mean and stderr by
    default. Aggregation across tasks and replicates is sample-count weighted.
    Conditional estimates fall back to sample scores because Inspect does not
    provide aggregate metrics for arbitrary post-hoc subsets.
    """

    if not log_groups_by_condition:
        raise ValueError("at least one condition is required")
    if not metric:
        raise ValueError("metric must be non-empty")
    if stderr not in {"inspect", "sample", "binomial"}:
        raise ValueError("stderr must be 'inspect', 'sample', or 'binomial'")
    if variant not in {"biased", "unbiased"}:
        raise ValueError("variant must be 'biased' or 'unbiased'")
    if where is not None and where_metric is not None:
        raise ValueError("use either where or where_metric/where_value, not both")
    predicate = (
        _validated_predicate(where)
        if where is not None
        else _validated_predicate({"metric": where_metric, "op": "eq", "value": where_value})
        if where_metric is not None
        else None
    )

    latest: dict[str, list[dict[tuple[str, str, str, str, int | None], Any]]] = {}
    for condition, log_groups in log_groups_by_condition.items():
        if not condition:
            raise ValueError("condition names must be non-empty")
        if not log_groups:
            raise ValueError(f"condition {condition!r} has no log groups")
        latest[condition] = []
        for replicate_index, logs in enumerate(log_groups, start=1):
            selected: dict[tuple[str, str, str, str, int | None], Any] = {}
            for log in logs:
                if getattr(log, "status", None) != "success":
                    continue
                identity = _task_identity(log)
                is_biased = bool(identity[0])
                if (variant == "biased" and not is_biased) or (variant == "unbiased" and is_biased):
                    continue
                previous = selected.get(identity)
                if previous is None or str(log.eval.created) > str(previous.eval.created):
                    selected[identity] = log
            if not selected:
                raise ValueError(
                    f"condition {condition!r}, replicate {replicate_index} has no successful {variant} mcq_bias logs"
                )
            latest[condition].append(selected)

    first_groups = next(iter(latest.values()))
    expected_tasks = set(first_groups[0])
    for condition, groups in latest.items():
        for replicate_index, selected in enumerate(groups, start=1):
            missing = sorted(expected_tasks - set(selected))
            extra = sorted(set(selected) - expected_tasks)
            if missing or extra:
                raise ValueError(
                    f"condition {condition!r}, replicate {replicate_index} does not have the same evaluation "
                    f"tasks as the first group; missing={missing}, extra={extra}"
                )

    pooled: dict[tuple[str, str], list[_Summary]] = defaultdict(list)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    replicate_counts: dict[tuple[str, str], int] = {}
    for condition, groups in latest.items():
        for selected in groups:
            for identity, log in selected.items():
                bias_type, dataset, _, _, _ = identity
                bias_type = bias_type or "unbiased"
                key = (condition, bias_type)
                samples = list(log.samples or [])
                totals[key] += len(samples)
                datasets[key].add(dataset)
                replicate_counts[key] = len(groups)
                values = []
                for sample in samples:
                    if predicate is not None and not _sample_matches(sample, predicate):
                        continue
                    value = _metric_value(sample, metric)
                    if value is not None:
                        values.append(value)
                if not values:
                    continue
                inspect_summary = (
                    _inspect_summary(log, metric) if stderr == "inspect" and predicate is None else None
                )
                if inspect_summary is not None:
                    pooled[key].append(inspect_summary)
                else:
                    fallback = "sample" if stderr == "inspect" else stderr
                    summary = _sample_summary(values, fallback)
                    if stderr == "inspect":
                        summary = _Summary(
                            **{
                                **summary.__dict__,
                                "estimate_method": "sample_conditional" if predicate else "sample_fallback",
                                "stderr_method": "sample_conditional" if predicate else "sample_fallback",
                            }
                        )
                    pooled[key].append(summary)

    if not pooled:
        raise ValueError(f"no finite {metric!r} scores remain after filtering")

    rows = []
    for condition, bias_type in sorted(pooled):
        summary = _pool_summaries(pooled[(condition, bias_type)])
        row = {
            "schema_version": 2,
            "condition": condition,
            "bias_type": bias_type,
            "metric": metric,
            "mean": summary.mean,
            "stderr": summary.stderr,
            "estimate_method": summary.estimate_method,
            "stderr_method": summary.stderr_method,
            "n_replicates": replicate_counts[(condition, bias_type)],
            "n_scored": summary.n,
            "n_total": totals[(condition, bias_type)],
            "datasets": sorted(datasets[(condition, bias_type)]),
            "variant": variant,
            **({"where": predicate} if predicate is not None else {}),
            **({"where_metric": where_metric, "where_value": where_value} if where_metric is not None else {}),
            **({"source_mean_metric": summary.source_mean_metric} if summary.source_mean_metric else {}),
            **({"source_stderr_metric": summary.source_stderr_metric} if summary.source_stderr_metric else {}),
        }
        if metadata:
            row.update(metadata)
        if condition_metadata and condition in condition_metadata:
            row.update(condition_metadata[condition])
        rows.append(row)
    return rows


def append_held_out_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_biases: Sequence[str] | None = None,
    label: str = "held_out_mean",
) -> list[dict[str, Any]]:
    """Append a sample-count-weighted held-out summary within every facet cell.

    When ``excluded_biases`` is omitted, each row group's ``training_biases``
    metadata defines the excluded set. This keeps held-out averages local when
    several models or training-bias regimes share one chart-ready document.
    """

    excluded = set(excluded_biases or [])
    if excluded_biases is not None and not excluded:
        raise ValueError("excluded_biases must be non-empty when supplied")
    output = [dict(row) for row in rows]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        row_excluded = excluded or {str(value) for value in (row.get("training_biases", []) or [])}
        if not row_excluded:
            raise ValueError("held-out auto mode requires non-empty training_biases metadata")
        if row["bias_type"] not in row_excluded and row["bias_type"] != label:
            grouped[_held_out_group_key(row)].append(row)
    for _, selected in sorted(grouped.items(), key=lambda item: repr(item[0])):
        if not selected:
            continue
        pooled = _pool_summaries(
            [
                _Summary(
                    mean=float(row["mean"]),
                    stderr=float(row["stderr"]),
                    n=int(row["n_scored"]),
                    estimate_method=str(row.get("estimate_method", "unknown")),
                    stderr_method=str(row.get("stderr_method", "unknown")),
                    source_mean_metric=row.get("source_mean_metric"),
                    source_stderr_metric=row.get("source_stderr_metric"),
                )
                for row in selected
            ]
        )
        summary = {
            **{
                key: value
                for key, value in selected[0].items()
                if key
                not in {
                    "bias_type",
                    "mean",
                    "stderr",
                    "n_scored",
                    "n_total",
                    "datasets",
                    "source_mean_metric",
                    "source_stderr_metric",
                    "p_value",
                    "significance",
                    "significance_baseline",
                    "significance_unavailable_reason",
                }
            },
            "condition": selected[0]["condition"],
            "bias_type": label,
            "metric": selected[0]["metric"],
            "mean": pooled.mean,
            "stderr": pooled.stderr,
            "estimate_method": pooled.estimate_method,
            "stderr_method": pooled.stderr_method,
            "n_scored": pooled.n,
            "n_total": sum(int(row["n_total"]) for row in selected),
            "datasets": sorted({dataset for row in selected for dataset in row["datasets"]}),
            "component_biases": sorted(str(row["bias_type"]) for row in selected),
            "sample_count_weighted": True,
            **({"source_mean_metric": pooled.source_mean_metric} if pooled.source_mean_metric else {}),
            **({"source_stderr_metric": pooled.source_stderr_metric} if pooled.source_stderr_metric else {}),
        }
        if summary.get("transform") == "percent_change":
            p_value = _one_sample_p_value(float(summary["mean"]), float(summary["stderr"]))
            summary.update(
                {
                    "p_value": p_value,
                    "significance": _significance_label(p_value) if p_value is not None else "",
                    "significance_baseline": "zero",
                }
            )
        output.append(summary)
    return output


def _held_out_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("condition", "")),
        str(row.get("model", "")),
        str(row.get("prompt_style", "")),
        str(row.get("training_regime", "")),
        tuple(str(value) for value in (row.get("training_biases", []) or [])),
        str(row.get("method", "")),
        bool(row.get("is_control", False)),
        str(row.get("control_for", "")),
        str(row.get("metric", "")),
        str(row.get("variant", "")),
        json.dumps(row.get("where"), sort_keys=True),
    )


def append_percent_change(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_condition: str,
    include_baseline: bool = False,
) -> list[dict[str, Any]]:
    """Convert estimates to percentage change from a matching baseline.

    The input means and standard errors are used directly, so Inspect-supplied
    uncertainty remains authoritative. Error propagation assumes independent
    numerator and baseline estimates, matching ``viz_refactor``.
    """

    if not baseline_condition:
        raise ValueError("baseline_condition must be non-empty")
    baselines = {_significance_key(row): row for row in rows if row.get("condition") == baseline_condition}
    output: list[dict[str, Any]] = []
    for source in rows:
        if source.get("condition") == baseline_condition and not include_baseline:
            continue
        baseline = baselines.get(_significance_key(source))
        if baseline is None:
            continue
        baseline_mean = float(baseline["mean"])
        if not math.isfinite(baseline_mean) or baseline_mean == 0.0:
            continue
        mean = float(source["mean"])
        stderr = float(source["stderr"])
        baseline_stderr = float(baseline["stderr"])
        ratio = mean / baseline_mean
        transformed_stderr = 100.0 * math.sqrt(
            (stderr / baseline_mean) ** 2 + (mean * baseline_stderr / baseline_mean**2) ** 2
        )
        transformed_mean = 100.0 * (ratio - 1.0)
        p_value = _one_sample_p_value(transformed_mean, transformed_stderr)
        row = {
            **dict(source),
            "source_metric": source["metric"],
            "source_estimate_method": source.get("estimate_method"),
            "source_stderr_method": source.get("stderr_method"),
            "metric": f"{source['metric']}_percent_change",
            "mean": transformed_mean,
            "stderr": transformed_stderr,
            "estimate_method": "derived:percent_change",
            "stderr_method": (
                f"delta:{source.get('stderr_method', 'unknown')}+"
                f"{baseline.get('stderr_method', 'unknown')}"
            ),
            "transform": "percent_change",
            "ratio_baseline": baseline_condition,
            "baseline_mean": baseline_mean,
            "baseline_stderr": baseline_stderr,
            "baseline_n_scored": int(baseline["n_scored"]),
            "p_value": p_value,
            "significance": _significance_label(p_value) if p_value is not None else "",
            "significance_baseline": "zero",
        }
        if source.get("condition") == baseline_condition:
            row.update({"mean": 0.0, "stderr": 0.0, "p_value": None, "significance": ""})
        output.append(row)
    return output


def _one_sample_p_value(mean: float, stderr: float) -> float | None:
    if not math.isfinite(mean) or not math.isfinite(stderr) or stderr <= 0.0:
        return None
    return 2.0 * (1.0 - NormalDist().cdf(abs(mean) / stderr))


def append_significance(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_condition: str,
) -> list[dict[str, Any]]:
    """Attach two-sided two-proportion z-tests against one condition.

    The renderer only displays these precomputed annotations; it never performs
    statistical inference.
    """

    baselines = {_significance_key(row): row for row in rows if row["condition"] == baseline_condition}
    if not baselines:
        return [
            {
                **dict(row),
                "significance_baseline": baseline_condition,
                "p_value": None,
                "significance": "",
                "significance_unavailable_reason": "baseline_condition_absent",
            }
            for row in rows
        ]
    output = []
    for source in rows:
        row = dict(source)
        key = _significance_key(row)
        baseline = baselines.get(key)
        if baseline is None:
            row.update(
                {
                    "p_value": None,
                    "significance": "",
                    "significance_unavailable_reason": "missing_baseline_cell",
                }
            )
            output.append(row)
            continue
        row["significance_baseline"] = baseline_condition
        if row["condition"] == baseline_condition:
            row.update({"p_value": None, "significance": ""})
        else:
            p_value = _two_proportion_p_value(
                float(row["mean"]),
                int(row["n_scored"]),
                float(baseline["mean"]),
                int(baseline["n_scored"]),
            )
            row.update({"p_value": p_value, "significance": _significance_label(p_value)})
        output.append(row)
    return output


def _significance_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("model", "")),
        str(row.get("prompt_style", "")),
        str(row.get("training_regime", "")),
        tuple(str(value) for value in (row.get("training_biases", []) or [])),
        str(row["bias_type"]),
        str(row.get("metric", "")),
        str(row.get("variant", "")),
        json.dumps(row.get("where"), sort_keys=True),
    )


def _two_proportion_p_value(mean_a: float, n_a: int, mean_b: float, n_b: int) -> float:
    if n_a <= 0 or n_b <= 0 or not (0.0 <= mean_a <= 1.0 and 0.0 <= mean_b <= 1.0):
        raise ValueError("two-proportion significance requires positive counts and means in [0, 1]")
    pooled = (mean_a * n_a + mean_b * n_b) / (n_a + n_b)
    standard_error = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_a + 1.0 / n_b))
    if standard_error == 0.0:
        return 1.0 if mean_a == mean_b else 0.0
    return 2.0 * (1.0 - NormalDist().cdf(abs(mean_a - mean_b) / standard_error))


def _significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def aggregate_sycophancy_tradeoff(
    logs_by_condition: Mapping[str, Sequence[Any]],
    *,
    bias_type: str = "suggested_answer",
) -> list[dict[str, Any]]:
    """Combine biased non-compliance and clean accuracy into the paper's F1."""

    output = []
    expected_pairs: set[tuple[str, str, str, int | None]] | None = None
    for condition, logs in logs_by_condition.items():
        latest: dict[tuple[str, str, str, str, int | None], Any] = {}
        for log in logs:
            if getattr(log, "status", None) != "success":
                continue
            identity = _task_identity(log)
            previous = latest.get(identity)
            if previous is None or str(log.eval.created) > str(previous.eval.created):
                latest[identity] = log

        biased = {identity: log for identity, log in latest.items() if identity[0] == bias_type}
        if not biased:
            raise ValueError(f"condition {condition!r} has no successful {bias_type!r} logs")
        pair_keys = {(identity[1], identity[2], identity[3], identity[4]) for identity in biased}
        if expected_pairs is None:
            expected_pairs = pair_keys
        elif pair_keys != expected_pairs:
            raise ValueError(f"condition {condition!r} does not have the same biased task set")

        not_sycophantic = []
        clean_accuracy = []
        for identity, biased_log in biased.items():
            key = (identity[1], identity[2], identity[3], identity[4])
            clean_identity = ("", *key)
            clean_log = latest.get(clean_identity)
            if clean_log is None:
                raise ValueError(f"condition {condition!r} has no matching unbiased log for dataset={identity[1]!r}")
            not_sycophantic.extend(
                1.0 - value
                for sample in (biased_log.samples or [])
                if (value := _metric_value(sample, "matches_bias")) is not None
            )
            clean_accuracy.extend(
                value for sample in (clean_log.samples or []) if (value := _metric_value(sample, "correct")) is not None
            )
        if not not_sycophantic or not clean_accuracy:
            raise ValueError(f"condition {condition!r} has no scored sycophancy/accuracy samples")
        not_syco_mean = statistics.fmean(not_sycophantic)
        accuracy_mean = statistics.fmean(clean_accuracy)
        harmonic_mean = (
            2 * not_syco_mean * accuracy_mean / (not_syco_mean + accuracy_mean)
            if not_syco_mean + accuracy_mean > 0
            else 0.0
        )
        output.append(
            {
                "condition": condition,
                "bias_type": bias_type,
                "not_sycophantic": not_syco_mean,
                "clean_accuracy": accuracy_mean,
                "f1": harmonic_mean,
                "n_not_sycophantic": len(not_sycophantic),
                "n_clean_accuracy": len(clean_accuracy),
            }
        )
    return sorted(output, key=lambda row: row["condition"])


def _parse_runs(values: Sequence[str]) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be NAME=LOG_DIR, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"--run must be NAME=LOG_DIR, got {value!r}")
        path = Path(raw_path)
        if not path.is_dir():
            raise ValueError(f"log directory for {name!r} does not exist: {path}")
        runs[name].append(path)
    return dict(runs)


def _read_logs(path: Path) -> list[Any]:
    from inspect_ai.log import read_eval_log

    files = sorted(path.rglob("*.eval"))
    if not files:
        raise ValueError(f"no .eval logs found under {path}")
    return [read_eval_log(str(file)) for file in files]


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate official mcq_bias sample scores by condition and bias",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", nargs="+", required=True, metavar="NAME=LOG_DIR")
    parser.add_argument("--report", choices=["metric", "sycophancy_tradeoff"], default="metric")
    parser.add_argument("--metric", default="matches_bias")
    parser.add_argument("--where-metric")
    parser.add_argument("--where-value", type=float, default=1.0)
    parser.add_argument(
        "--where",
        type=_json_object,
        help="Declarative sample predicate, e.g. '{\"metric\":\"correct\",\"op\":\"eq\",\"value\":1}'",
    )
    parser.add_argument(
        "--stderr",
        choices=["inspect", "sample", "binomial"],
        default="inspect",
        help="Use Inspect's metric when available, or explicitly recompute from samples",
    )
    parser.add_argument("--variant", choices=["biased", "unbiased"], default="biased")
    parser.add_argument("--metadata", type=_json_object, default={})
    parser.add_argument("--condition-metadata", type=_json_object, default={})
    parser.add_argument("--significance-baseline")
    parser.add_argument(
        "--ratio-baseline",
        help="Emit percentage change from this condition using propagated input standard errors",
    )
    parser.add_argument("--bias-type", default="suggested_answer")
    parser.add_argument(
        "--held-out-exclude",
        nargs="+",
        help="For metric reports, append a sample-count-weighted mean over every bias except these",
    )
    parser.add_argument(
        "--held-out-auto",
        action="store_true",
        help="Derive held-out exclusions independently from each row group's training_biases",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        runs = _parse_runs(args.run)
    except ValueError as exc:
        parser.error(str(exc))

    print("\nmcq_bias aggregation:")
    for name, paths in runs.items():
        for replicate_index, path in enumerate(paths, start=1):
            print(f"  {name} replicate {replicate_index}: {path}")
    print(f"  metric={args.metric}")
    print(f"  report={args.report}")
    if args.where:
        print(f"  filter={json.dumps(args.where, sort_keys=True)}")
    elif args.where_metric:
        print(f"  filter={args.where_metric} == {args.where_value}")
    print(f"  output={args.output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    try:
        log_groups_by_condition = {name: [_read_logs(path) for path in paths] for name, paths in runs.items()}
        if args.report == "sycophancy_tradeoff":
            if args.where or args.where_metric or args.held_out_exclude or args.held_out_auto or args.ratio_baseline:
                raise ValueError("filter, held-out, and ratio options apply only to --report metric")
            repeated = [name for name, groups in log_groups_by_condition.items() if len(groups) > 1]
            if repeated:
                raise ValueError(
                    "repeated --run condition names are currently supported only for metric reports; " f"got {repeated}"
                )
            logs_by_condition = {name: groups[0] for name, groups in log_groups_by_condition.items()}
            rows = aggregate_sycophancy_tradeoff(logs_by_condition, bias_type=args.bias_type)
        else:
            rows = aggregate_log_groups(
                log_groups_by_condition,
                metric=args.metric,
                where_metric=args.where_metric,
                where_value=args.where_value,
                where=args.where,
                stderr=args.stderr,
                variant=args.variant,
                metadata=args.metadata,
                condition_metadata=args.condition_metadata,
            )
            if args.held_out_exclude and args.held_out_auto:
                raise ValueError("use either --held-out-exclude or --held-out-auto, not both")
            if args.ratio_baseline:
                rows = append_percent_change(rows, baseline_condition=args.ratio_baseline)
            if args.held_out_exclude or args.held_out_auto:
                rows = append_held_out_summary(
                    rows,
                    excluded_biases=args.held_out_exclude if not args.held_out_auto else None,
                )
            if args.significance_baseline and not args.ratio_baseline:
                rows = append_significance(rows, baseline_condition=args.significance_baseline)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_atomic_bytes(args.output, (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
