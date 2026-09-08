"""Shared chart-row aggregation for Inspect-native benchmark adapters.

The common layer owns retry selection, replicate pooling, denominators, and
condition metadata.  Each adapter remains responsible for selecting benchmark
rows and naming the scientific dimensions placed on a chart.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctm.artifacts import write_atomic_bytes

RowDimensions = Callable[[Mapping[str, Any]], Mapping[str, Any] | None]
SampleDimensions = Callable[[Mapping[str, Any], Any], Sequence[Mapping[str, Any]]]

_PROPENSITY_GAP_METRICS = {
    "mcq_bias": "net_bias_switch",
    "evalawarebench": "net_refusal_switch",
    "wildjailbreak": "net_refusal_switch",
}


@dataclass(frozen=True)
class _Summary:
    mean: float
    stderr: float
    n: int


def _eval_log_metadata(log: Any) -> dict[str, Any]:
    value = getattr(getattr(log, "eval", None), "metadata", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _log_identity(log: Any) -> str:
    metadata = _eval_log_metadata(log)
    artifact = metadata.get("artifact") if isinstance(metadata.get("artifact"), Mapping) else {}
    selection = artifact.get("selection") if isinstance(artifact.get("selection"), Mapping) else {}
    value = {
        "benchmark": metadata.get("benchmark"),
        "task": str(getattr(getattr(log, "eval", None), "task", "")),
        "model": str(getattr(getattr(log, "eval", None), "model", "")),
        "valence": metadata.get("valence"),
        "prompt_type": metadata.get("prompt_type"),
        "bias_type": metadata.get("bias_type"),
        "dataset": metadata.get("source_dataset", metadata.get("dataset")),
        "factors": metadata.get("factors"),
        "artifact": artifact.get("content_sha256"),
        "selection": selection.get("selected_source_ids_sha256"),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _latest_successful_logs(logs: Sequence[Any]) -> list[Any]:
    """Keep the latest successful attempt for each explicit task identity."""

    latest: dict[str, Any] = {}
    for log in logs:
        if getattr(log, "status", None) != "success":
            continue
        identity = _log_identity(log)
        previous = latest.get(identity)
        created = str(getattr(getattr(log, "eval", None), "created", ""))
        previous_created = str(getattr(getattr(previous, "eval", None), "created", ""))
        if previous is None or created > previous_created:
            latest[identity] = log
    return [latest[key] for key in sorted(latest)]


def _sample_metrics(sample: Any) -> dict[str, float]:
    """Flatten all finite numeric values stored by Inspect scorers."""

    values: dict[str, float] = {}
    for score in (getattr(sample, "scores", None) or {}).values():
        score_value = getattr(score, "value", None)
        if not isinstance(score_value, Mapping):
            continue
        for name, candidate in score_value.items():
            if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
                continue
            number = float(candidate)
            if math.isfinite(number):
                values.setdefault(str(name), number)
    return values


def _describe_eval_log(log: Any) -> dict[str, Any]:
    """Return the adapter-neutral descriptor for one Inspect log."""

    metadata = _eval_log_metadata(log)
    artifact = metadata.get("artifact") if isinstance(metadata.get("artifact"), Mapping) else None
    evaluation = getattr(log, "eval", None)
    return {
        "benchmark": metadata.get("benchmark"),
        "task": str(getattr(evaluation, "task", "")),
        "model": str(getattr(evaluation, "model", "")),
        "created": str(getattr(evaluation, "created", "")),
        "valence": metadata.get("valence"),
        "prompt_type": metadata.get("prompt_type"),
        "bias_type": metadata.get("bias_type"),
        "dataset": metadata.get("source_dataset", metadata.get("dataset")),
        "factors": metadata.get("factors"),
        "artifact": artifact,
        "log": str(getattr(log, "location", "")) or None,
    }


def _expects_propensity_gap(metadata: Mapping[str, Any]) -> bool:
    benchmark = metadata.get("benchmark")
    if benchmark == "mcq_bias":
        return bool(metadata.get("bias_type"))
    if benchmark == "evalawarebench":
        return metadata.get("prompt_type") == "factor"
    if benchmark == "wildjailbreak":
        return metadata.get("prompt_type") == "adversarial"
    return False


def _build_eval_report(logs: Sequence[Any], *, require_propensity_gaps: bool = True) -> dict[str, Any]:
    """Aggregate finite sample metrics without depending on a CTM report module."""

    selected = _latest_successful_logs(logs)
    if not selected:
        raise ValueError("no successful Inspect logs were supplied")
    metric_rows: list[dict[str, Any]] = []
    propensity_rows: list[dict[str, Any]] = []
    missing_gaps: list[str] = []
    for log in selected:
        descriptor = _describe_eval_log(log)
        samples = list(getattr(log, "samples", None) or [])
        by_metric: dict[str, list[float]] = {}
        for sample in samples:
            for name, value in _sample_metrics(sample).items():
                by_metric.setdefault(name, []).append(value)
        for name, values in sorted(by_metric.items()):
            row = {
                **descriptor,
                "metric": name,
                "mean": statistics.fmean(values),
                "stderr": statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0,
                "n_valid": len(values),
                "n_total": len(samples),
            }
            metric_rows.append(row)
            expected = _PROPENSITY_GAP_METRICS.get(str(descriptor["benchmark"]))
            if name == expected:
                propensity_rows.append({**row, "interpretation": "evaluated propensity minus reference propensity"})
        metadata = _eval_log_metadata(log)
        expected = _PROPENSITY_GAP_METRICS.get(str(metadata.get("benchmark")))
        if require_propensity_gaps and _expects_propensity_gap(metadata) and expected not in by_metric:
            missing_gaps.append(
                f"{descriptor['benchmark']}/{descriptor['task']}"
                f" valence={descriptor['valence']!r} bias_type={descriptor['bias_type']!r}"
            )
    if missing_gaps:
        raise ValueError("propensity-gap scoring is missing from evaluated task log(s): " + "; ".join(missing_gaps))
    return {
        "schema_version": 1,
        "n_logs": len(selected),
        "propensity_gaps": propensity_rows,
        "metrics": metric_rows,
    }


def _pool(summaries: Sequence[_Summary]) -> _Summary:
    if not summaries:
        raise ValueError("cannot pool an empty collection")
    total = sum(summary.n for summary in summaries)
    mean = sum(summary.n * summary.mean for summary in summaries) / total
    if total == 1:
        return _Summary(mean, 0.0, total)
    within_sse = sum(summary.n * (summary.n - 1) * summary.stderr**2 for summary in summaries if summary.n > 1)
    between_sse = sum(summary.n * (summary.mean - mean) ** 2 for summary in summaries)
    return _Summary(mean, math.sqrt(((within_sse + between_sse) / (total - 1)) / total), total)


def aggregate_chart_rows(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    benchmark: str,
    metric: str,
    dimensions: RowDimensions,
    cell_fields: Sequence[str],
    display_fields: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate chart cells across conditions and independent replicates."""

    if not log_groups_by_condition:
        raise ValueError("at least one condition is required")
    if not benchmark or not metric:
        raise ValueError("benchmark and metric must be non-empty")
    if not cell_fields or any(not field for field in cell_fields):
        raise ValueError("cell_fields must contain explicit non-empty field names")

    grouped: dict[str, list[dict[tuple[Any, ...], list[tuple[dict[str, Any], _Summary]]]]] = {}
    for condition, log_groups in log_groups_by_condition.items():
        if not condition:
            raise ValueError("condition names must be non-empty")
        if not log_groups:
            raise ValueError(f"condition {condition!r} has no log groups")
        grouped[condition] = []
        for replicate_index, logs in enumerate(log_groups, start=1):
            report = _build_eval_report(logs, require_propensity_gaps=False)
            cells: dict[tuple[Any, ...], list[tuple[dict[str, Any], _Summary]]] = defaultdict(list)
            for source in report["metrics"]:
                if source.get("benchmark") != benchmark or source.get("metric") != metric:
                    continue
                adapter_dimensions = dimensions(source)
                if adapter_dimensions is None:
                    continue
                row = {**source, **adapter_dimensions}
                missing = [field for field in cell_fields if field not in row]
                if missing:
                    raise ValueError(f"adapter dimensions omitted chart cell fields: {missing}")
                key = tuple(_hashable(row[field]) for field in cell_fields)
                cells[key].append(
                    (
                        row,
                        _Summary(float(row["mean"]), float(row["stderr"]), int(row["n_valid"])),
                    )
                )
            if not cells:
                raise ValueError(
                    f"condition {condition!r}, replicate {replicate_index} has no finite "
                    f"{benchmark}/{metric} rows after adapter filtering"
                )
            grouped[condition].append(cells)

    return _finalize_chart_rows(
        grouped,
        benchmark=benchmark,
        metric=metric,
        cell_fields=cell_fields,
        display_fields=display_fields,
        metadata=metadata,
        condition_metadata=condition_metadata,
        require_complete=require_complete,
    )


def aggregate_sample_chart_rows(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    benchmark: str,
    metric: str,
    dimensions: SampleDimensions,
    cell_fields: Sequence[str],
    display_fields: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Aggregate finite sample metrics into adapter-defined, possibly multi-label cells."""

    if not log_groups_by_condition:
        raise ValueError("at least one condition is required")
    if not benchmark or not metric:
        raise ValueError("benchmark and metric must be non-empty")
    if not cell_fields or any(not field for field in cell_fields):
        raise ValueError("cell_fields must contain explicit non-empty field names")

    grouped: dict[str, list[dict[tuple[Any, ...], list[tuple[dict[str, Any], _Summary]]]]] = {}
    for condition, log_groups in log_groups_by_condition.items():
        if not condition:
            raise ValueError("condition names must be non-empty")
        if not log_groups:
            raise ValueError(f"condition {condition!r} has no log groups")
        grouped[condition] = []
        for replicate_index, logs in enumerate(log_groups, start=1):
            cells: dict[tuple[Any, ...], list[tuple[dict[str, Any], _Summary]]] = defaultdict(list)
            for log in _latest_successful_logs(logs):
                descriptor = _describe_eval_log(log)
                if descriptor.get("benchmark") != benchmark:
                    continue
                values_by_key: dict[tuple[Any, ...], list[float]] = defaultdict(list)
                totals_by_key: dict[tuple[Any, ...], int] = defaultdict(int)
                exemplar_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
                for sample in list(getattr(log, "samples", None) or []):
                    adapter_dimensions = list(dimensions(descriptor, sample))
                    sample_value = _sample_metrics(sample).get(metric)
                    seen_keys: set[tuple[Any, ...]] = set()
                    for dimension in adapter_dimensions:
                        row = {**descriptor, **dimension}
                        missing = [field for field in cell_fields if field not in row]
                        if missing:
                            raise ValueError(f"adapter dimensions omitted chart cell fields: {missing}")
                        key = tuple(_hashable(row[field]) for field in cell_fields)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        totals_by_key[key] += 1
                        exemplar_by_key[key] = row
                        if sample_value is not None:
                            values_by_key[key].append(sample_value)
                for key, values in values_by_key.items():
                    if not values:
                        continue
                    mean = statistics.fmean(values)
                    stderr = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
                    cells[key].append(
                        (
                            {**exemplar_by_key[key], "n_total": totals_by_key[key]},
                            _Summary(mean, stderr, len(values)),
                        )
                    )
            if not cells:
                raise ValueError(
                    f"condition {condition!r}, replicate {replicate_index} has no finite "
                    f"{benchmark}/{metric} sample rows after adapter filtering"
                )
            grouped[condition].append(cells)

    return _finalize_chart_rows(
        grouped,
        benchmark=benchmark,
        metric=metric,
        cell_fields=cell_fields,
        display_fields=display_fields,
        metadata=metadata,
        condition_metadata=condition_metadata,
        require_complete=require_complete,
    )


def _finalize_chart_rows(
    grouped: Mapping[str, Sequence[Mapping[tuple[Any, ...], Sequence[tuple[dict[str, Any], _Summary]]]]],
    *,
    benchmark: str,
    metric: str,
    cell_fields: Sequence[str],
    display_fields: Sequence[str],
    metadata: Mapping[str, Any] | None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None,
    require_complete: bool,
) -> list[dict[str, Any]]:
    expected: set[tuple[Any, ...]] | None = None
    if require_complete:
        for condition, replicates in grouped.items():
            for replicate_index, cells in enumerate(replicates, start=1):
                actual = set(cells)
                if expected is None:
                    expected = actual
                    continue
                if actual != expected:
                    raise ValueError(
                        f"condition {condition!r}, replicate {replicate_index} has a different chart-cell matrix; "
                        f"missing={sorted(expected - actual, key=repr)}, extra={sorted(actual - expected, key=repr)}"
                    )

    output: list[dict[str, Any]] = []
    for condition, replicates in grouped.items():
        keys = sorted({key for cells in replicates for key in cells}, key=repr)
        for key in keys:
            sources: list[dict[str, Any]] = []
            summaries: list[_Summary] = []
            n_total = 0
            contributing_replicates = 0
            for cells in replicates:
                values = cells.get(key, [])
                if not values:
                    continue
                contributing_replicates += 1
                sources.extend(row for row, _ in values)
                summaries.append(_pool([summary for _, summary in values]))
                n_total += sum(int(row["n_total"]) for row, _ in values)
            summary = _pool(summaries)
            exemplar = sources[0]
            row = {
                "schema_version": 1,
                "condition": condition,
                "benchmark": benchmark,
                "metric": metric,
                "mean": summary.mean,
                "stderr": summary.stderr,
                "estimate_method": "sample" if len(summaries) == 1 else "pooled_sample",
                "stderr_method": "sample" if len(summaries) == 1 else "pooled_sample",
                "n_replicates": contributing_replicates,
                "n_scored": summary.n,
                "n_total": n_total,
                "model": exemplar.get("model"),
                **{field: exemplar[field] for field in cell_fields},
            }
            source_logs = sorted({str(source["log"]) for source in sources if source.get("log")})
            source_tasks = sorted({str(source["task"]) for source in sources if source.get("task")})
            artifacts = _artifact_summaries(sources)
            row.update(
                {
                    "n_source_logs": len(source_logs),
                    "source_logs": source_logs,
                    "source_tasks": source_tasks,
                    "artifacts": artifacts,
                }
            )
            for name in (*display_fields, "task", "prompt_type", "valence", "factors", "dataset"):
                if name in exemplar and name not in row:
                    row[name] = exemplar[name]
            if metadata:
                row.update(metadata)
            if condition_metadata and condition in condition_metadata:
                row.update(condition_metadata[condition])
            output.append(row)
    return sorted(output, key=lambda row: (str(row["condition"]), *(repr(row[field]) for field in cell_fields)))


def _artifact_summaries(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for source in sources:
        artifact = source.get("artifact")
        if not isinstance(artifact, Mapping):
            continue
        selection = artifact.get("selection")
        summary = {
            name: artifact[name]
            for name in (
                "artifact_schema",
                "schema_version",
                "path",
                "manifest_path",
                "content_sha256",
                "row_count",
            )
            if name in artifact
        }
        if isinstance(selection, Mapping):
            summary["selection"] = {
                name: selection[name]
                for name in (
                    "n_variants",
                    "selected_row_count",
                    "selected_source_ids_sha256",
                )
                if name in selection
            }
        identity = json.dumps(summary, sort_keys=True, separators=(",", ":"), default=str)
        by_identity[identity] = summary
    return [by_identity[key] for key in sorted(by_identity)]


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_hashable(item) for item in value)
    return value


def parse_runs(values: Sequence[str]) -> dict[str, list[Path]]:
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


def read_logs(path: Path) -> list[Any]:
    from inspect_ai.log import read_eval_log

    files = sorted(path.rglob("*.eval"))
    if not files:
        raise ValueError(f"no .eval logs found under {path}")
    return [read_eval_log(str(file)) for file in files]


def json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"expected a JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def write_chart_rows(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"refusing to overwrite existing output: {path}")
    payload = (json.dumps(list(rows), indent=2, sort_keys=True, default=str) + "\n").encode()
    write_atomic_bytes(path, payload)


__all__ = [
    "aggregate_chart_rows",
    "aggregate_sample_chart_rows",
    "json_object",
    "parse_runs",
    "read_logs",
    "write_chart_rows",
]
