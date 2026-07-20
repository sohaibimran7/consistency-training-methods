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


def aggregate_logs(logs_by_condition: Mapping[str, Sequence[Any]], *, metric: str) -> list[dict[str, Any]]:
    """Pool sample metrics by condition and held-out bias, checking task parity."""

    if not logs_by_condition:
        raise ValueError("at least one condition is required")
    if not metric:
        raise ValueError("metric must be non-empty")

    latest: dict[str, dict[tuple[str, str, str, str, int | None], Any]] = {}
    for condition, logs in logs_by_condition.items():
        if not condition:
            raise ValueError("condition names must be non-empty")
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
            raise ValueError(f"condition {condition!r} has no successful biased mcq_bias logs")
        latest[condition] = selected

    expected_tasks = set(next(iter(latest.values())))
    for condition, selected in latest.items():
        missing = sorted(expected_tasks - set(selected))
        extra = sorted(set(selected) - expected_tasks)
        if missing or extra:
            raise ValueError(
                f"condition {condition!r} does not have the same evaluation tasks as the first condition; "
                f"missing={missing}, extra={extra}"
            )

    pooled: dict[tuple[str, str], list[float]] = defaultdict(list)
    totals: dict[tuple[str, str], int] = defaultdict(int)
    datasets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for condition, selected in latest.items():
        for identity, log in selected.items():
            bias_type, dataset, _, _, _ = identity
            key = (condition, bias_type)
            samples = list(log.samples or [])
            totals[key] += len(samples)
            datasets[key].add(dataset)
            pooled[key].extend(value for sample in samples if (value := _metric_value(sample, metric)) is not None)

    rows = []
    for condition, bias_type in sorted(pooled):
        values = pooled[(condition, bias_type)]
        if not values:
            raise ValueError(f"condition {condition!r}, bias {bias_type!r} has no finite {metric!r} scores")
        stderr = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(
            {
                "condition": condition,
                "bias_type": bias_type,
                "metric": metric,
                "mean": statistics.fmean(values),
                "stderr": stderr,
                "n_scored": len(values),
                "n_total": totals[(condition, bias_type)],
                "datasets": sorted(datasets[(condition, bias_type)]),
            }
        )
    return rows


def _parse_runs(values: Sequence[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be NAME=LOG_DIR, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or not raw_path:
            raise ValueError(f"--run must be NAME=LOG_DIR, got {value!r}")
        if name in runs:
            raise ValueError(f"duplicate condition name {name!r}")
        path = Path(raw_path)
        if not path.is_dir():
            raise ValueError(f"log directory for {name!r} does not exist: {path}")
        runs[name] = path
    return runs


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
    parser.add_argument("--metric", default="matches_bias")
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
    for name, path in runs.items():
        print(f"  {name}: {path}")
    print(f"  metric={args.metric}")
    print(f"  output={args.output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    try:
        rows = aggregate_logs({name: _read_logs(path) for name, path in runs.items()}, metric=args.metric)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    write_atomic_bytes(args.output, (json.dumps(rows, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()
