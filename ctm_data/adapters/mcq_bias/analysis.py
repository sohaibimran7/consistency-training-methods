"""Aggregate official ``mcq_bias`` Inspect logs into chart-ready JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.artifacts import write_atomic_bytes


def _metric_value(sample: Any, metric: str) -> float | None:
    for score in (getattr(sample, "scores", None) or {}).values():
        value = getattr(score, "value", None)
        if isinstance(value, Mapping) and metric in value:
            candidate = value[metric]
        else:
            continue
        if candidate is None or isinstance(candidate, bool):
            return None
        try:
            result = float(candidate)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None
    return None


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
    stderr: str = "sample",
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
        stderr=stderr,
    )


def aggregate_log_groups(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    metric: str,
    where_metric: str | None = None,
    where_value: float = 1.0,
    stderr: str = "sample",
) -> list[dict[str, Any]]:
    """Pool independent log groups while deduplicating retries within each group."""

    if not log_groups_by_condition:
        raise ValueError("at least one condition is required")
    if not metric:
        raise ValueError("metric must be non-empty")
    if stderr not in {"sample", "binomial"}:
        raise ValueError("stderr must be 'sample' or 'binomial'")

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
                if not identity[0]:  # shared unbiased tasks are not graph rows
                    continue
                previous = selected.get(identity)
                if previous is None or str(log.eval.created) > str(previous.eval.created):
                    selected[identity] = log
            if not selected:
                raise ValueError(
                    f"condition {condition!r}, replicate {replicate_index} has no successful biased mcq_bias logs"
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

    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    replicate_counts: dict[tuple[str, str], int] = {}
    for condition, groups in latest.items():
        for selected in groups:
            for identity, log in selected.items():
                bias_type, dataset, _, _, _ = identity
                key = (condition, bias_type)
                samples = list(log.samples or [])
                totals[key] += len(samples)
                datasets[key].add(dataset)
                replicate_counts[key] = len(groups)
                for sample in samples:
                    if where_metric is not None:
                        predicate_value = _metric_value(sample, where_metric)
                        if predicate_value is None or predicate_value != where_value:
                            continue
                    value = _metric_value(sample, metric)
                    if value is not None:
                        pooled[key].append(value)

    rows = []
    for condition, bias_type in sorted(pooled):
        values = pooled[(condition, bias_type)]
        if not values:
            raise ValueError(f"condition {condition!r}, bias {bias_type!r} has no finite {metric!r} scores")
        mean = statistics.fmean(values)
        if stderr == "binomial":
            if any(value not in {0.0, 1.0} for value in values):
                raise ValueError(f"binomial stderr requires binary {metric!r} scores")
            standard_error = math.sqrt(mean * (1.0 - mean) / len(values))
        else:
            standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "condition": condition,
                "bias_type": bias_type,
                "metric": metric,
                "mean": mean,
                "stderr": standard_error,
                "stderr_method": stderr,
                "n_replicates": replicate_counts[(condition, bias_type)],
                "n_scored": len(values),
                "n_total": totals[(condition, bias_type)],
                "datasets": sorted(datasets[(condition, bias_type)]),
                **(
                    {"where_metric": where_metric, "where_value": where_value}
                    if where_metric is not None
                    else {}
                ),
            }
        )
    return rows


def append_held_out_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_biases: Sequence[str],
    label: str = "held_out_mean",
) -> list[dict[str, Any]]:
    """Append an equal-bias-weighted summary row for every condition."""

    excluded = set(excluded_biases)
    if not excluded:
        raise ValueError("excluded_biases must be non-empty")
    output = [dict(row) for row in rows]
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["bias_type"] not in excluded:
            by_condition[str(row["condition"])].append(row)
    for condition, selected in sorted(by_condition.items()):
        if not selected:
            raise ValueError(f"condition {condition!r} has no held-out bias rows")
        means = [float(row["mean"]) for row in selected]
        standard_errors = [float(row["stderr"]) for row in selected]
        summary = {
            "condition": condition,
            "bias_type": label,
            "metric": selected[0]["metric"],
            "mean": statistics.fmean(means),
            "stderr": math.sqrt(sum(value * value for value in standard_errors)) / len(selected),
            "n_scored": sum(int(row["n_scored"]) for row in selected),
            "n_total": sum(int(row["n_total"]) for row in selected),
            "datasets": sorted({dataset for row in selected for dataset in row["datasets"]}),
            "component_biases": sorted(str(row["bias_type"]) for row in selected),
            "equal_bias_weighted": True,
        }
        if "where_metric" in selected[0]:
            summary["where_metric"] = selected[0]["where_metric"]
            summary["where_value"] = selected[0]["where_value"]
        output.append(summary)
    return output


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
                raise ValueError(
                    f"condition {condition!r} has no matching unbiased log for dataset={identity[1]!r}"
                )
            not_sycophantic.extend(
                1.0 - value
                for sample in (biased_log.samples or [])
                if (value := _metric_value(sample, "matches_bias")) is not None
            )
            clean_accuracy.extend(
                value
                for sample in (clean_log.samples or [])
                if (value := _metric_value(sample, "correct")) is not None
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
        "--stderr",
        choices=["sample", "binomial"],
        default="sample",
        help="Standard-error estimator; paper switch metrics use binomial",
    )
    parser.add_argument("--bias-type", default="suggested_answer")
    parser.add_argument(
        "--held-out-exclude",
        nargs="+",
        help="For metric reports, append an equal-weighted mean over every bias except these",
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
    if args.where_metric:
        print(f"  filter={args.where_metric} == {args.where_value}")
    print(f"  output={args.output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    try:
        log_groups_by_condition = {
            name: [_read_logs(path) for path in paths]
            for name, paths in runs.items()
        }
        if args.report == "sycophancy_tradeoff":
            if args.where_metric or args.held_out_exclude:
                raise ValueError("--where-metric and --held-out-exclude apply only to --report metric")
            repeated = [name for name, groups in log_groups_by_condition.items() if len(groups) > 1]
            if repeated:
                raise ValueError(
                    "repeated --run condition names are currently supported only for metric reports; "
                    f"got {repeated}"
                )
            logs_by_condition = {
                name: groups[0] for name, groups in log_groups_by_condition.items()
            }
            rows = aggregate_sycophancy_tradeoff(logs_by_condition, bias_type=args.bias_type)
        else:
            rows = aggregate_log_groups(
                log_groups_by_condition,
                metric=args.metric,
                where_metric=args.where_metric,
                where_value=args.where_value,
                stderr=args.stderr,
            )
            if args.held_out_exclude:
                rows = append_held_out_summary(rows, excluded_biases=args.held_out_exclude)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_atomic_bytes(args.output, (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
