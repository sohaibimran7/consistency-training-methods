#!/usr/bin/env python3
"""Compare two sealed Figure 6 runtime-sentinel judge screens content-free."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ctm_data.adapters.eval_awareness.figure6_runtime_comparison import (
    RuntimeComparisonError,
    RuntimeScreenPaths,
    write_runtime_comparison,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-round-manifest", type=Path, required=True)
    parser.add_argument("--reference-source-manifest", type=Path, required=True)
    parser.add_argument("--reference-judge-lifecycle", type=Path, required=True)
    parser.add_argument("--reference-private-verdicts", type=Path, required=True)
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--candidate-round-manifest", type=Path, required=True)
    parser.add_argument("--candidate-source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-judge-lifecycle", type=Path, required=True)
    parser.add_argument("--candidate-private-verdicts", type=Path, required=True)
    parser.add_argument("--candidate-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = write_runtime_comparison(
            RuntimeScreenPaths(
                round_manifest=args.reference_round_manifest,
                source_manifest=args.reference_source_manifest,
                judge_lifecycle=args.reference_judge_lifecycle,
                private_verdicts=args.reference_private_verdicts,
                aggregate=args.reference_aggregate,
            ),
            RuntimeScreenPaths(
                round_manifest=args.candidate_round_manifest,
                source_manifest=args.candidate_source_manifest,
                judge_lifecycle=args.candidate_judge_lifecycle,
                private_verdicts=args.candidate_private_verdicts,
                aggregate=args.candidate_aggregate,
            ),
            output_path=args.output,
        )
    except (OSError, RuntimeComparisonError, ValueError):
        # Deliberately never render source paths, raw rows, verdict content,
        # prompts, completions, reasoning, or identifiers in errors.
        print(json.dumps({"error_type": "RuntimeComparisonError"}, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
