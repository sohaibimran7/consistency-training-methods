"""Aggregate WildJailbreak Inspect logs into valence-explicit chart rows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters._analysis import (
    aggregate_chart_rows,
    aggregate_sample_chart_rows,
    json_object,
    parse_runs,
    read_logs,
    write_chart_rows,
)


def aggregate_logs(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    metric: str = "net_refusal_switch",
    prompt_types: Sequence[str] = ("adversarial",),
    valences: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    require_complete: bool = True,
    group_by: str = "valence",
    include_untagged: bool = True,
) -> list[dict[str, Any]]:
    """Pool conditions while retaining harmful/benign task semantics."""

    selected_prompt_types = set(prompt_types)
    if not selected_prompt_types or not selected_prompt_types <= {"vanilla", "adversarial"}:
        raise ValueError("prompt_types must contain vanilla and/or adversarial")
    selected_valences = set(valences or ())
    if group_by not in {"valence", "tactic"}:
        raise ValueError("group_by must be valence or tactic")

    if group_by == "tactic":
        return _aggregate_tactics(
            log_groups_by_condition,
            metric=metric,
            selected_prompt_types=selected_prompt_types,
            selected_valences=selected_valences,
            include_untagged=include_untagged,
            metadata=metadata,
            condition_metadata=condition_metadata,
            require_complete=require_complete,
        )

    def dimensions(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
        prompt_type = str(row.get("prompt_type", ""))
        valence = str(row.get("valence", ""))
        if prompt_type not in selected_prompt_types:
            return None
        if selected_valences and valence not in selected_valences:
            return None
        return {
            "valence": valence,
            "valence_label": valence.title(),
            "prompt_type": prompt_type,
        }

    return aggregate_chart_rows(
        log_groups_by_condition,
        benchmark="wildjailbreak",
        metric=metric,
        dimensions=dimensions,
        cell_fields=("valence", "prompt_type", "model"),
        display_fields=("valence_label",),
        metadata=metadata,
        condition_metadata=condition_metadata,
        require_complete=require_complete,
    )


def _aggregate_tactics(
    log_groups_by_condition: Mapping[str, Sequence[Sequence[Any]]],
    *,
    metric: str,
    selected_prompt_types: set[str],
    selected_valences: set[str],
    include_untagged: bool,
    metadata: Mapping[str, Any] | None,
    condition_metadata: Mapping[str, Mapping[str, Any]] | None,
    require_complete: bool,
) -> list[dict[str, Any]]:
    def dimensions(descriptor: Mapping[str, Any], sample: Any) -> Sequence[Mapping[str, Any]]:
        prompt_type = str(descriptor.get("prompt_type", ""))
        valence = str(descriptor.get("valence", ""))
        if prompt_type not in selected_prompt_types:
            return ()
        if selected_valences and valence not in selected_valences:
            return ()
        sample_metadata = getattr(sample, "metadata", None)
        if not isinstance(sample_metadata, Mapping):
            sample_metadata = {}
        raw_tactics = sample_metadata.get("tactics", [])
        if not isinstance(raw_tactics, Sequence) or isinstance(raw_tactics, (str, bytes)):
            raise TypeError("WildJailbreak sample metadata.tactics must be an array")
        tactics = sorted({str(tactic).strip() for tactic in raw_tactics if str(tactic).strip()})
        if not tactics and include_untagged:
            tactics = ["untagged"]
        return [
            {
                "tactic": tactic,
                "tactic_label": tactic.replace("_", " ").replace("-", " ").title(),
                "tactic_grouping": "multi_label_marginal",
                "valence": valence,
                "prompt_type": prompt_type,
            }
            for tactic in tactics
        ]

    return aggregate_sample_chart_rows(
        log_groups_by_condition,
        benchmark="wildjailbreak",
        metric=metric,
        dimensions=dimensions,
        cell_fields=("tactic", "valence", "prompt_type", "model"),
        display_fields=("tactic_label", "tactic_grouping"),
        metadata=metadata,
        condition_metadata=condition_metadata,
        require_complete=require_complete,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate WildJailbreak Inspect scores by explicit valence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run", nargs="+", required=True, metavar="NAME=LOG_DIR")
    parser.add_argument("--metric", default="net_refusal_switch")
    parser.add_argument("--group-by", choices=["valence", "tactic"], default="valence")
    parser.add_argument("--prompt-type", action="append", choices=["vanilla", "adversarial"], dest="prompt_types")
    parser.add_argument("--valence", action="append", choices=["harmful", "benign"], dest="valences")
    parser.add_argument("--metadata", type=json_object, default={})
    parser.add_argument("--condition-metadata", type=json_object, default={})
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--exclude-untagged",
        action="store_true",
        help="For tactic grouping, omit adversarial samples without a tactic label",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        runs = parse_runs(args.run)
    except ValueError as exc:
        parser.error(str(exc))
    print("\nWildJailbreak aggregation:")
    for name, paths in runs.items():
        for replicate_index, path in enumerate(paths, start=1):
            print(f"  {name} replicate {replicate_index}: {path}")
    print(f"  metric={args.metric}")
    print(f"  group_by={args.group_by}")
    print(f"  prompt_types={args.prompt_types or ['adversarial']}")
    print(f"  output={args.output}")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return
    try:
        rows = aggregate_logs(
            {name: [read_logs(path) for path in paths] for name, paths in runs.items()},
            metric=args.metric,
            prompt_types=args.prompt_types or ("adversarial",),
            valences=args.valences,
            metadata=args.metadata,
            condition_metadata=args.condition_metadata,
            require_complete=not args.allow_incomplete,
            group_by=args.group_by,
            include_untagged=not args.exclude_untagged,
        )
        write_chart_rows(args.output, rows)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {len(rows)} aggregate rows to {args.output}")


if __name__ == "__main__":
    main()


__all__ = ["aggregate_logs", "main"]
