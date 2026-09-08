"""Aggregate EvalAwareBench Inspect logs into factor-explicit chart rows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters._analysis import (
    aggregate_chart_rows,
    json_object,
    parse_runs,
    read_logs,
    write_chart_rows,
)


def _factor_set(value: Any) -> str:
    if not value:
        return "baseline"
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("EvalAwareBench report factors must be an array")
    factors = [str(factor) for factor in value]
    return "+".join(factors)


def aggregate_logs(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    metric: str = "net_refusal_switch",
    prompt_types: Sequence[str] = ("factor",),
    valences: Sequence[str] | None = None,
    factor_sets: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Pool conditions while keeping factor configuration and valence explicit."""

    selected_prompt_types = set(prompt_types)
    if not selected_prompt_types or not selected_prompt_types <= {"baseline", "factor"}:
        raise ValueError("prompt_types must contain baseline and/or factor")
    selected_valences = set(valences or ())
    selected_factor_sets = set(factor_sets or ())

    def dimensions(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        prompt_type = str(row.get("prompt_type", ""))
        valence = str(row.get("valence", ""))
        factor_set = _factor_set(row.get("factors"))
        if prompt_type not in selected_prompt_types:
            return None
        if selected_valences and valence not in selected_valences:
            return None
        if selected_factor_sets and factor_set not in selected_factor_sets:
            return None
        return {
            "factor_set": factor_set,
            "factor_label": "Baseline" if factor_set == "baseline" else factor_set.replace("+", " + "),
            "valence": valence,
            "prompt_type": prompt_type,
        }

    return aggregate_chart_rows(
        log_groups_by_condition,
        benchmark="evalawarebench",
        metric=metric,
        dimensions=dimensions,
        cell_fields=("factor_set", "valence", "prompt_type", "model"),
        display_fields=("factor_label",),
        metadata=metadata,
        condition_metadata=condition_metadata,
        require_complete=require_complete,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate EvalAwareBench Inspect scores by factor configuration and valence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", nargs="+", required=True, metavar="NAME=LOG_DIR")
    parser.add_argument("--metric", default="net_refusal_switch")
    parser.add_argument("--prompt-type", action="append", choices=["baseline", "factor"], dest="prompt_types")
    parser.add_argument("--valence", action="append", choices=["safety", "capability"], dest="valences")
    parser.add_argument("--factor-set", action="append", dest="factor_sets")
    parser.add_argument("--metadata", type=json_object, default={})
    parser.add_argument("--condition-metadata", type=json_object, default={})
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        runs = parse_runs(args.run)
    except ValueError as exc:
        parser.error(str(exc))
    print("\nEvalAwareBench aggregation:")
    for name, paths in runs.items():
        for replicate_index, path in enumerate(paths, start=1):
            print(f"  {name} replicate {replicate_index}: {path}")
    print(f"  metric={args.metric}")
    print(f"  prompt_types={args.prompt_types or ['factor']}")
    print(f"  output={args.output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return
    try:
        rows = aggregate_logs(
            {name: [read_logs(path) for path in paths] for name, paths in runs.items()},
            metric=args.metric,
            prompt_types=args.prompt_types or ("factor",),
            valences=args.valences,
            factor_sets=args.factor_sets,
            metadata=args.metadata,
            condition_metadata=args.condition_metadata,
            require_complete=not args.allow_incomplete,
        )
        write_chart_rows(args.output, rows)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["aggregate_logs", "main"]
