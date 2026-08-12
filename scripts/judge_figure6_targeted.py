#!/usr/bin/env python3
"""Freeze and screen immutable Figure 6 targeted-generation sentinels.

The executable judge path is deliberately limited to the initial DeepSeek V4
Pro/user/k=1 screen.  ``confirmation-design`` only emits an auditable k=3
proposal; it cannot make requests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ctm_data.adapters.eval_awareness.figure6_targeted_judge import (
    build_k3_confirmation_design,
    freeze_immutable_paired_sentinel_source,
    freeze_immutable_sentinel_source,
    judge_targeted_sentinels,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze-source", help="Pin completed generator output without copying it.")
    freeze.add_argument("--records", type=Path, required=True)
    freeze.add_argument("--generator-manifest", type=Path, required=True)
    freeze.add_argument("--output-manifest", type=Path, required=True)

    paired_freeze = subparsers.add_parser(
        "freeze-paired-source", help="Pin a completed paired-v2 round without copying raw records."
    )
    paired_freeze.add_argument("--records", type=Path, action="append", required=True)
    paired_freeze.add_argument("--round-manifest", type=Path, required=True)
    paired_freeze.add_argument("--arm-manifest", type=Path, action="append", required=True)
    paired_freeze.add_argument("--output-manifest", type=Path, required=True)

    screen = subparsers.add_parser("screen", help="Run the paper-compatible DeepSeek/user/k=1 screen.")
    screen.add_argument("--records", type=Path, action="append", required=True)
    screen.add_argument("--source-manifest", type=Path, required=True)
    screen.add_argument("--judge-template", type=Path, required=True)
    screen.add_argument("--attempt-log", type=Path, required=True)
    screen.add_argument("--manifest", type=Path, required=True)
    screen.add_argument("--private-verdicts", type=Path, required=True)
    screen.add_argument("--aggregate", type=Path, required=True)
    screen.add_argument("--concurrency", type=int, default=4)
    screen.add_argument("--expected-plan-sha256")
    approval = screen.add_mutually_exclusive_group(required=True)
    approval.add_argument("--dry-run", action="store_true")
    approval.add_argument("--yes", action="store_true")

    confirmation = subparsers.add_parser(
        "confirmation-design", help="Describe a separately authorized DeepSeek/user/k=3 confirmation only."
    )
    confirmation.add_argument("--initial-lifecycle-manifest", type=Path, required=True)
    confirmation.add_argument("--custom-id", action="append", dest="custom_ids")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-source":
            result = freeze_immutable_sentinel_source(args.records, args.generator_manifest, args.output_manifest)
        elif args.command == "freeze-paired-source":
            result = freeze_immutable_paired_sentinel_source(
                args.records, args.round_manifest, args.arm_manifest, args.output_manifest
            )
        elif args.command == "confirmation-design":
            lifecycle = json.loads(args.initial_lifecycle_manifest.read_text(encoding="utf-8"))
            if not isinstance(lifecycle, dict) or not isinstance(lifecycle.get("plan"), dict):
                parser.error("--initial-lifecycle-manifest must contain a targeted judge plan")
            result = build_k3_confirmation_design(lifecycle["plan"], selected_custom_ids=args.custom_ids)
        else:
            if args.yes and not args.expected_plan_sha256:
                parser.error("paid screening requires --expected-plan-sha256 from a reviewed dry run")
            result = asyncio.run(
                judge_targeted_sentinels(
                    args.records[0] if len(args.records) == 1 else args.records,
                    args.source_manifest,
                    judge_template_path=args.judge_template,
                    attempt_log_path=args.attempt_log,
                    lifecycle_manifest_path=args.manifest,
                    private_verdicts_path=args.private_verdicts,
                    aggregate_path=args.aggregate,
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    expected_plan_sha256=args.expected_plan_sha256,
                    confirm_paid=args.yes,
                    dry_run=args.dry_run,
                    concurrency=args.concurrency,
                )
            )
    except (OSError, ValueError, RuntimeError) as exc:
        # No prompts, generations, raw responses, or credential-bearing values
        # are rendered by this CLI.
        print(json.dumps({"error_type": type(exc).__name__}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
