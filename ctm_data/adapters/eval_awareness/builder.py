#!/usr/bin/env python3
"""Freeze explicitly selected EvalAwareBench rows for consistency training."""

from __future__ import annotations

import argparse
from pathlib import Path

from ctm_data.adapters.eval_awareness.data import (
    DATASET_CONFIGS,
    DATASET_LICENSE,
    materialize_eval_awareness,
    read_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-variants", type=int, default=4)
    parser.add_argument(
        "--factors",
        nargs="+",
        help="Select rows whose factors_varied exactly match this set, e.g. --factors F6",
    )
    parser.add_argument("--seed", default="42")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-config", choices=DATASET_CONFIGS, default="prompts")
    parser.add_argument("--source-license", default=DATASET_LICENSE)
    args = parser.parse_args(argv)
    manifest = materialize_eval_awareness(
        read_jsonl(args.input_jsonl),
        args.output,
        n_variants=args.n_variants,
        seed=args.seed,
        source_revision=args.source_revision,
        source_config=args.source_config,
        source_license=args.source_license,
        factors=args.factors,
    )
    print(f"Wrote {manifest['row_count']} training families to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
