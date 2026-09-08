"""Stream completed Stage 2 Inspect EvalLogs into compact analysis JSONL.

This is the memory-safe handoff before ``experiments.stage2_ood_hle.analyze``:
each full EvalLog is reduced to the paired outcomes used by the estimands, its
sample graph is released, and the output is provenance-bound to every selected
source log.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from experiments.stage2_ood_hle.analyze import extract_inspect_runs_to_jsonl, parse_runs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="CONDITION=LOCAL_LOGS")
    parser.add_argument("--expected-prompt-style", default="none")
    parser.add_argument("--output", required=True, type=Path, help="new compact observation JSONL")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        sidecar = extract_inspect_runs_to_jsonl(
            parse_runs(args.run),
            args.output,
            expected_prompt_style=args.expected_prompt_style,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"written: {args.output.resolve()} "
        f"({sidecar['observations']['rows']} observations, {len(sidecar['input_sources'])} source logs)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
