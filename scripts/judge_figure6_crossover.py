#!/usr/bin/env python3
"""Run the audited Figure 6 midtrained crossover diagnostics."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from ctm.artifacts import write_atomic_bytes
from ctm_data.adapters.eval_awareness.figure6_crossover import (
    DEEPSEEK_ALLOWED_RESPONSE_MODELS,
    DEEPSEEK_OUTPUT_SCHEMA,
    DEEPSEEK_PROTOCOL_ID,
    IGOR_MODEL_KEY,
    OUR_MODEL_KEY,
    PROBE_PROTOCOLS,
    SCOPE_CONFIGS,
    aggregate_judgments,
    aggregate_probe_attempts,
    historical_igor_judgments,
    import_igor_logs,
    judge_deepseek_scope,
    judge_luna_scope,
    judge_probe_scope,
    scoped_generations,
    select_our_paired,
)
from ctm_data.adapters.eval_awareness.figure6_judge import (
    PAPER_JUDGE_TEMPLATE_SHA256,
    custom_id_for_generation,
    load_judge_template,
)
from ctm_data.adapters.eval_awareness.figure6_openrouter import (
    OPENROUTER_GPT_56_LUNA_DIRECT_PROFILE,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (REPOSITORY_ROOT / "artifacts").resolve()


def _require_artifact_path(path: Path, parser: argparse.ArgumentParser) -> None:
    try:
        path.resolve().relative_to(ARTIFACT_ROOT)
    except ValueError:
        parser.error(f"crossover outputs containing model or judge data must stay under {ARTIFACT_ROOT}")


def _add_paid_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--judge-template", type=Path, required=True)
    parser.add_argument("--scope", choices=sorted(SCOPE_CONFIGS), default="baseline")
    parser.add_argument("--attempt-log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256")
    approval = parser.add_mutually_exclusive_group(required=True)
    approval.add_argument("--dry-run", action="store_true")
    approval.add_argument("--yes", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    importer = subparsers.add_parser("import-igor", help="Import the six hash-pinned historical Inspect logs.")
    importer.add_argument("--logs-dir", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--manifest", type=Path, required=True)

    selector = subparsers.add_parser("select-our-paired", help="Freeze our matching midtrained replicate-1 rows.")
    selector.add_argument("--igor-generations", type=Path, required=True)
    selector.add_argument("--our-generations", type=Path, required=True)
    selector.add_argument("--output", type=Path, required=True)
    selector.add_argument("--manifest", type=Path, required=True)

    luna = subparsers.add_parser("luna", help="Judge a frozen scope with Luna/system/strict k=1.")
    _add_paid_arguments(luna)

    deepseek = subparsers.add_parser("deepseek", help="Judge a frozen scope with Igor's DeepSeek/user/k=3.")
    _add_paid_arguments(deepseek)

    probe = subparsers.add_parser("probe", help="Run one immutable minimal-change judge-isolation probe.")
    _add_paid_arguments(probe)
    probe.add_argument("--protocol", choices=sorted(PROBE_PROTOCOLS), required=True)

    summarize = subparsers.add_parser("summarize", help="Summarize the four crossover cells without content.")
    summarize.add_argument("--scope", choices=sorted(SCOPE_CONFIGS), default="baseline")
    summarize.add_argument("--igor-generations", type=Path, required=True)
    summarize.add_argument("--igor-luna", type=Path)
    summarize.add_argument("--our-luna", type=Path)
    summarize.add_argument("--our-deepseek", type=Path)
    summarize.add_argument("--output", type=Path)

    probe_summary = subparsers.add_parser(
        "summarize-probes", help="Summarize complete or intentionally stopped probe attempts without content."
    )
    probe_summary.add_argument(
        "--attempt",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Named probe attempt JSONL; repeat for each immutable protocol.",
    )
    probe_summary.add_argument("--output", type=Path, required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no records in {path}")
    return rows


def _paid_preflight(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.yes and not args.expected_plan_sha256:
        parser.error("paid crossover runs require --expected-plan-sha256 from a reviewed dry run")


def _summary(args: argparse.Namespace) -> dict:
    igor_records = _read_jsonl(args.igor_generations)
    igor_scoped = scoped_generations(igor_records, args.scope)
    expected_identities = {
        (row["pair_id"], row["valence"], row["config_name"]) for row in igor_scoped
    }
    expected_igor_custom_ids = {custom_id_for_generation(row) for row in igor_scoped}
    result: dict[str, object] = {
        "schema": "ctm.eval_awareness.figure6_crossover.summary.v1",
        "scope": args.scope,
        "cells": {
            "igor_outputs__historic_deepseek_user_k3": aggregate_judgments(
                historical_igor_judgments(igor_records, args.scope)
            )
        },
    }
    optional = {
        "igor_outputs__luna_system_k1": args.igor_luna,
        "our_outputs__historic_luna_system_k1": args.our_luna,
        "our_outputs__deepseek_user_k3": args.our_deepseek,
    }
    cells = result["cells"]
    assert isinstance(cells, dict)
    for label, path in optional.items():
        if path is None:
            continue
        rows = _read_jsonl(path)
        if label == "our_outputs__historic_luna_system_k1":
            rows = [
                row
                for row in rows
                if row.get("model_key") == "qwen_mo_mid"
                and row.get("replicate") == 1
                and row.get("valence") == "safety"
                and row.get("config_name") in SCOPE_CONFIGS[args.scope]
            ]
        else:
            rows = [row for row in rows if row.get("config_name") in SCOPE_CONFIGS[args.scope]]
        expected = {"baseline": 100, "non-f3": 495, "full": 594}[args.scope]
        if len(rows) != expected:
            raise ValueError(f"{label} must contain {expected} scoped judgments, got {len(rows)}")
        identities = {(row.get("pair_id"), row.get("valence"), row.get("config_name")) for row in rows}
        if identities != expected_identities:
            raise ValueError(f"{label} identities differ from the frozen historical cohort")
        custom_ids = [row.get("custom_id") for row in rows]
        if len(set(custom_ids)) != len(custom_ids) or not all(isinstance(value, str) for value in custom_ids):
            raise ValueError(f"{label} custom IDs must be present and unique")
        if label == "igor_outputs__luna_system_k1":
            if set(custom_ids) != expected_igor_custom_ids:
                raise ValueError("Igor/Luna custom IDs differ from the frozen imported generations")
            if any(
                row.get("model_key") != IGOR_MODEL_KEY
                or row.get("judge_profile") != OPENROUTER_GPT_56_LUNA_DIRECT_PROFILE
                or row.get("judge_status") != "ok"
                or row.get("judge_template_sha256") != PAPER_JUDGE_TEMPLATE_SHA256
                for row in rows
            ):
                raise ValueError("Igor/Luna judgments have unexpected provenance")
            plan_hashes = {row.get("judge_plan_sha256") for row in rows}
        elif label == "our_outputs__historic_luna_system_k1":
            if any(
                row.get("model_key") != OUR_MODEL_KEY
                or row.get("judge_profile") != OPENROUTER_GPT_56_LUNA_DIRECT_PROFILE
                or row.get("judge_status") != "ok"
                or row.get("judge_template_sha256") != PAPER_JUDGE_TEMPLATE_SHA256
                for row in rows
            ):
                raise ValueError("our Luna judgments have unexpected provenance")
            plan_hashes = {row.get("judge_plan_sha256") for row in rows}
        else:
            allowed_protocols = {
                DEEPSEEK_PROTOCOL_ID,
                "igor-deepseek-v4-pro-user-k3-v1",
                "igor-deepseek-v4-pro-user-k3-v2",
            }
            if any(
                row.get("schema") != DEEPSEEK_OUTPUT_SCHEMA
                or row.get("model_key") != OUR_MODEL_KEY
                or row.get("protocol_id") not in allowed_protocols
                or not set(row.get("response_models") or ()) <= DEEPSEEK_ALLOWED_RESPONSE_MODELS
                for row in rows
            ):
                raise ValueError("our DeepSeek judgments have unexpected provenance")
            plan_hashes = {row.get("plan_sha256") for row in rows}
        if len(plan_hashes) != 1 or not all(isinstance(value, str) and len(value) == 64 for value in plan_hashes):
            raise ValueError(f"{label} must bind to exactly one paid plan hash")
        cells[label] = aggregate_judgments(rows)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "import-igor":
        _require_artifact_path(args.output, parser)
        _require_artifact_path(args.manifest, parser)
        result = import_igor_logs(args.logs_dir, output_path=args.output, manifest_path=args.manifest)
    elif args.command == "select-our-paired":
        _require_artifact_path(args.output, parser)
        _require_artifact_path(args.manifest, parser)
        result = select_our_paired(
            args.igor_generations,
            args.our_generations,
            output_path=args.output,
            manifest_path=args.manifest,
        )
    elif args.command in {"luna", "deepseek", "probe"}:
        _paid_preflight(args, parser)
        for target in (args.attempt_log, args.manifest, args.output):
            _require_artifact_path(target, parser)
        records = _read_jsonl(args.generations)
        # Validate the exact expected scope before loading a credential.
        scoped_generations(records, args.scope)
        template = load_judge_template(
            args.judge_template,
            expected_sha256=PAPER_JUDGE_TEMPLATE_SHA256,
        )
        common = {
            "scope": args.scope,
            "template": template,
            "attempt_log_path": args.attempt_log,
            "output_path": args.output,
            "manifest_path": args.manifest,
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "expected_plan_sha256": args.expected_plan_sha256,
            "confirm_paid": args.yes,
            "dry_run": args.dry_run,
        }
        if args.command == "luna":
            result = asyncio.run(judge_luna_scope(records, **common))
        elif args.command == "deepseek":
            result = asyncio.run(judge_deepseek_scope(records, **common))
        else:
            result = asyncio.run(judge_probe_scope(records, protocol_id=args.protocol, **common))
    elif args.command == "summarize":
        if args.output is not None:
            _require_artifact_path(args.output, parser)
        result = _summary(args)
        if args.output is not None:
            payload = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
            write_atomic_bytes(args.output, payload)
    else:
        _require_artifact_path(args.output, parser)
        probes: dict[str, object] = {}
        for item in args.attempt:
            if "=" not in item:
                parser.error("--attempt must use LABEL=PATH")
            label, path_text = item.split("=", 1)
            if not label or label in probes:
                parser.error("probe labels must be non-empty and unique")
            path = Path(path_text)
            _require_artifact_path(path, parser)
            probes[label] = aggregate_probe_attempts(_read_jsonl(path))
        result = {"schema": "ctm.eval_awareness.figure6_crossover.probe_summary.v1", "probes": probes}
        payload = (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        write_atomic_bytes(args.output, payload)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
