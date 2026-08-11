#!/usr/bin/env python3
"""Run the audited qwen_mo_mid safety-baseline request-contract diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ctm_data.adapters.eval_awareness.figure6_generate import _safe_error
from ctm_data.adapters.eval_awareness.figure6_request_contract import (
    DEFAULT_MAX_CONCURRENCY,
    SCOPES,
    run_request_contract_gate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument(
        "--server-attestation",
        type=Path,
        required=True,
        help="Sanitized, pinned serving-stack attestation stored under artifacts/.",
    )
    parser.add_argument("--scope", choices=sorted(SCOPES), default="wire-smoke")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="FIGURE6_LOCAL_ENDPOINT_TOKEN")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--yes", action="store_true", help="Approve requests for the reviewed plan hash.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.yes:
        parser.error("--dry-run and --yes cannot be combined")
    if not args.dry_run and not args.yes:
        parser.error("real target requests require --yes; use --dry-run to review the plan")
    if not args.dry_run and args.expected_plan_sha256 is None:
        parser.error("real target requests require --expected-plan-sha256 from a reviewed dry run")
    try:
        summary = asyncio.run(
            run_request_contract_gate(
                args.artifact,
                args.output,
                prompt_path=args.prompt_path,
                server_attestation_path=args.server_attestation,
                scope=args.scope,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                max_concurrency=args.max_concurrency,
                expected_plan_sha256=args.expected_plan_sha256,
                confirm_requests=args.yes,
                dry_run=args.dry_run,
            )
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {_safe_error(exc)}")
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.dry_run or summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
