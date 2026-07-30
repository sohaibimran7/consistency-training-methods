#!/usr/bin/env python3
"""Run resumable EvalAwareBench Figure 6 judging through OpenRouter."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

from ctm_data.adapters.eval_awareness.figure6_judge import (
    PAPER_JUDGE_TEMPLATE_SHA256,
    _read_jsonl,
    load_judge_template,
)
from ctm_data.adapters.eval_awareness.figure6_openrouter import (
    CURRENT_QWEN_MODEL_KEYS,
    DEFAULT_JUDGE_PROFILE,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_RETRY_AFTER,
    JUDGE_PROFILES,
    judge_generations,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=Path, nargs="+", required=True)
    parser.add_argument("--judge-template", type=Path, required=True)
    parser.add_argument("--expected-template-sha256", default=PAPER_JUDGE_TEMPLATE_SHA256)
    parser.add_argument("--attempt-log", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Audit manifest (default: ATTEMPT_LOG.manifest.json).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--judge-profile",
        choices=sorted(JUDGE_PROFILES),
        default=DEFAULT_JUDGE_PROFILE,
        help=(
            "Exact registered paid judge profile. The default is the direct DeepSeek V3.2 alternative judge; "
            "models and request settings cannot be overridden independently."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--expected-model-key",
        action="append",
        choices=CURRENT_QWEN_MODEL_KEYS,
        required=True,
        dest="expected_model_keys",
        help="Registered Qwen model required in this exact paid matrix; repeat in registry order.",
    )
    parser.add_argument(
        "--proxy",
        help="Proxy URL. Direct profiles prohibit it; the Muse alternate requires loopback socks5h.",
    )
    parser.add_argument("--expected-exit-instance-id", help="Positive numeric U.S. Vast instance ID.")
    parser.add_argument("--expected-exit-ssh-host", help="Concrete Vast SSH host; never a credential.")
    parser.add_argument("--expected-exit-ssh-port", type=int, help="Concrete Vast SSH port.")
    parser.add_argument("--route-country-code", help="Attested route country; Muse requires exactly US.")
    parser.add_argument("--route-attested-at", help="ISO 8601 UTC timestamp for the route evidence.")
    parser.add_argument("--route-attested-by", help="Non-secret operator/reviewer identity.")
    parser.add_argument(
        "--route-attestation-sha256",
        help="SHA-256 of the archived, non-secret Vast/egress evidence JSON.",
    )
    parser.add_argument(
        "--route-attestation-evidence",
        type=Path,
        help="Archived non-secret route evidence JSON; its bytes must match --route-attestation-sha256.",
    )
    parser.add_argument(
        "--max-retry-after",
        type=float,
        default=DEFAULT_MAX_RETRY_AFTER,
        help="Pinned registered Retry-After ceiling; provider delay is never truncated.",
    )
    parser.add_argument(
        "--expected-plan-sha256",
        help="Require the paid run to match a previously reviewed deterministic dry-run hash.",
    )
    parser.add_argument(
        "--amend-attempt-ceiling",
        action="store_true",
        help="Authorize a reviewed, strictly higher retry ceiling while preserving all prior attempts/successes.",
    )
    parser.add_argument(
        "--rescore-paid-errors",
        action="store_true",
        help="Explicitly authorize retrying preserved malformed HTTP-200 paid responses.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Explicitly approve paid requests for the calculated plan. Required unless --dry-run is used.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.yes:
        parser.error("--dry-run and --yes cannot be combined")
    if args.dry_run and args.amend_attempt_ceiling:
        parser.error("--dry-run reports amendment requirements but cannot authorize --amend-attempt-ceiling")
    if not args.dry_run and not args.yes:
        parser.error("paid OpenRouter judging requires explicit --yes; use --dry-run to inspect the plan")
    if not args.dry_run and args.expected_plan_sha256 is None:
        parser.error("paid OpenRouter judging requires --expected-plan-sha256 from a reviewed dry run")
    route_attestation_evidence = None
    if args.route_attestation_evidence is not None:
        route_attestation_evidence = args.route_attestation_evidence.read_bytes()
        evidence_sha256 = hashlib.sha256(route_attestation_evidence).hexdigest()
        if args.route_attestation_sha256 is not None and args.route_attestation_sha256 != evidence_sha256:
            parser.error("--route-attestation-sha256 does not match the exact --route-attestation-evidence bytes")
        args.route_attestation_sha256 = evidence_sha256
    generations = _read_jsonl(args.generations, label="generation")
    template = load_judge_template(args.judge_template, expected_sha256=args.expected_template_sha256)
    summary = asyncio.run(
        judge_generations(
            generations,
            template=template,
            attempt_log_path=args.attempt_log,
            output_path=args.output,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            manifest_path=args.manifest,
            judge_template_sha256=args.expected_template_sha256,
            judge_profile=args.judge_profile,
            max_attempts=args.max_attempts,
            proxy=args.proxy,
            expected_exit_instance_id=args.expected_exit_instance_id,
            expected_exit_ssh_host=args.expected_exit_ssh_host,
            expected_exit_ssh_port=args.expected_exit_ssh_port,
            route_country_code=args.route_country_code,
            route_attested_at=args.route_attested_at,
            route_attested_by=args.route_attested_by,
            route_attestation_sha256=args.route_attestation_sha256,
            route_attestation_evidence=route_attestation_evidence,
            expected_model_keys=args.expected_model_keys,
            confirm_paid=args.yes,
            expected_plan_sha256=args.expected_plan_sha256,
            amend_attempt_ceiling=args.amend_attempt_ceiling,
            rescore_paid_errors=args.rescore_paid_errors,
            max_retry_after=args.max_retry_after,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
