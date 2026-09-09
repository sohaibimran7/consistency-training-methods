#!/usr/bin/env python3
"""Plan or run one registered paired Figure 6 midtrained sentinel round."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ctm_data.adapters.eval_awareness.figure6_generate import _safe_error
from ctm_data.adapters.eval_awareness.figure6_sentinel import (
    COMPARISON_ROUNDS,
    DEFAULT_API_KEY_ENV,
    run_comparison_round,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round", dest="round_id", choices=sorted(COMPARISON_ROUNDS), required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--server-attestation", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.yes:
        parser.error("--dry-run and --yes cannot be combined")
    if not args.dry_run and (not args.yes or args.expected_plan_sha256 is None):
        parser.error("real requests require --yes and --expected-plan-sha256 from a dry run")
    try:
        result = asyncio.run(
            run_comparison_round(
                args.artifact,
                args.output_root,
                round_id=args.round_id,
                prompt_path=args.prompt_path,
                server_attestation_path=args.server_attestation,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                expected_plan_sha256=args.expected_plan_sha256,
                confirm_requests=args.yes,
                dry_run=args.dry_run,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {_safe_error(exc)}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if args.dry_run or result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
