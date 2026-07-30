"""Single offline CLI for the paper-specific reconstruction and audit stages."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from ctm.artifacts import write_atomic_bytes
from ctm.cli_safety import parse_json_object, reject_inline_secrets
from scripts.irpan_2510_27062.analysis import SELECTION_OBSERVATION_SCHEMA, rank_validation_candidates
from scripts.irpan_2510_27062.artifacts import read_artifact
from scripts.irpan_2510_27062.filtering import (
    materialize_retained_prompt_pairs,
    materialize_vulnerability_filter,
)
from scripts.irpan_2510_27062.jailbreak_sources import (
    DEFAULT_ID_FIELD,
    DEFAULT_PROMPT_FIELD,
    materialize_harmbench_source,
)
from scripts.irpan_2510_27062.judge import (
    DEFAULT_MAX_PARSE_RETRIES,
    materialize_external_judgments,
    materialize_judgment_requests,
)
from scripts.irpan_2510_27062.partitions import partition_registry_payload
from scripts.irpan_2510_27062.reconstruction import reconstruction_ledger
from scripts.irpan_2510_27062.safety_tasks import materialize_eval_artifact
from scripts.irpan_2510_27062.selection_logs import materialize_validation_observations
from scripts.irpan_2510_27062.smoke_fixtures import (
    materialize_bct_result_fixture,
    materialize_smoke_fixtures,
)
from scripts.irpan_2510_27062.source_registry import source_registry_payload
from scripts.irpan_2510_27062.wrappers import (
    materialize_completion_requests,
    materialize_external_completions,
    materialize_wrapper_candidates,
    read_external_result_export,
)

_EVAL_SOURCES = ("harmbench", "or_bench", "clearharm", "wildguardtest", "xstest", "wildjailbreak")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="Print the source registry and reconstruction ledger")
    inventory.add_argument("--output", type=Path, help="Optional immutable JSON output; stdout is always printed")

    materialize = commands.add_parser("materialize-eval", help="Normalize one explicit local evaluation export")
    materialize.add_argument("--source-path", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    materialize.add_argument("--source", required=True, choices=_EVAL_SOURCES)
    materialize.add_argument("--subset", required=True)
    materialize.add_argument("--split", required=True)
    materialize.add_argument("--revision")
    materialize.add_argument("--expected-count", type=int)
    materialize.add_argument("--expected-count-mode", choices=["off", "warn", "strict"], default="warn")

    verify = commands.add_parser("verify", help="Verify an immutable paper artifact without printing its rows")
    verify.add_argument("path", type=Path)
    verify.add_argument("--kind")
    verify.add_argument("--role", choices=["training", "validation", "final_eval"])

    retained_pairs = commands.add_parser(
        "materialize-retained-pairs",
        help="Adapt retained jailbreak candidates to the shared ctm.prompt_pairs schema",
    )
    retained_pairs.add_argument("--retained", type=Path, required=True)
    retained_pairs.add_argument("--output", type=Path, required=True)

    harmbench = commands.add_parser(
        "normalize-harmbench-training",
        help="Normalize one explicitly partitioned local HarmBench training export",
    )
    harmbench.add_argument("--input", type=Path, required=True)
    harmbench.add_argument("--output", type=Path, required=True)
    harmbench.add_argument("--subset", required=True)
    harmbench.add_argument("--split", required=True)
    harmbench.add_argument("--source-revision")
    harmbench.add_argument("--expected-file-sha256")
    harmbench.add_argument("--id-field", default=DEFAULT_ID_FIELD)
    harmbench.add_argument("--prompt-field", default=DEFAULT_PROMPT_FIELD)

    wrappers = commands.add_parser(
        "build-wrappers",
        help="Apply the frozen reconstructed wrapper catalog",
    )
    wrappers.add_argument("--source", type=Path, required=True)
    wrappers.add_argument("--output", type=Path, required=True)

    completion_requests = commands.add_parser(
        "build-completion-requests",
        help="Build immutable clean/wrapped requests for external execution",
    )
    completion_requests.add_argument("--candidates", type=Path, required=True)
    completion_requests.add_argument("--output", type=Path, required=True)
    completion_requests.add_argument("--generator", required=True, help="Canonical JSON generator identity")
    completion_requests.add_argument("--decoding-params", required=True, help="Canonical JSON decoding options")

    completion_import = commands.add_parser(
        "import-completions",
        help="Strictly join an external completion export to immutable requests",
    )
    completion_import.add_argument("--requests", type=Path, required=True)
    completion_import.add_argument("--results", type=Path, required=True)
    completion_import.add_argument("--output", type=Path, required=True)

    judgment_requests = commands.add_parser(
        "build-judgment-requests",
        help="Build immutable Gemini judgment requests for external execution",
    )
    judgment_requests.add_argument("--completions", type=Path, required=True)
    judgment_requests.add_argument("--output", type=Path, required=True)
    judgment_requests.add_argument("--judge", required=True, help="Canonical JSON judge identity")
    judgment_requests.add_argument("--decoding-params", required=True, help="Canonical JSON decoding options")

    judgment_import = commands.add_parser(
        "import-judgments",
        help="Strictly parse and join an external judgment export",
    )
    judgment_import.add_argument("--requests", type=Path, required=True)
    judgment_import.add_argument("--results", type=Path, required=True)
    judgment_import.add_argument("--output", type=Path, required=True)
    judgment_import.add_argument("--max-parse-retries", type=int, default=DEFAULT_MAX_PARSE_RETRIES)

    filtering = commands.add_parser(
        "filter-vulnerabilities",
        help="Apply the clean-refused/wrapped-fulfilled retention rule",
    )
    filtering.add_argument("--candidates", type=Path, required=True)
    filtering.add_argument("--judgments", type=Path, required=True)
    filtering.add_argument("--audit-output", type=Path, required=True)
    filtering.add_argument("--retained-output", type=Path, required=True)

    selection = commands.add_parser(
        "select-validation",
        help="Rank candidate observations using validation routes only",
    )
    selection.add_argument("--domain", required=True, choices=["sycophancy", "jailbreak"])
    selection.add_argument("--method", help="Require every observation to carry this candidate method")
    selection.add_argument("--input", type=Path, required=True, help="JSON array or JSONL observations")
    selection.add_argument("--output", type=Path, required=True)

    collection = commands.add_parser(
        "collect-validation-observations",
        help="Collect typed selection observations from successful Inspect validation logs",
    )
    collection.add_argument("--domain", required=True, choices=["sycophancy", "jailbreak"])
    collection.add_argument("--method", help="Collect only this exact candidate method")
    collection.add_argument("--log-dir", type=Path, required=True)
    collection.add_argument("--schema", required=True, choices=[SELECTION_OBSERVATION_SCHEMA])
    collection.add_argument("--output", type=Path, required=True)

    fixtures = commands.add_parser(
        "materialize-smoke-fixtures",
        help="Publish deterministic synthetic artifacts for the offline smoke graph",
    )
    fixtures.add_argument("--output-dir", type=Path, required=True)

    fixture_results = commands.add_parser(
        "build-smoke-bct-results",
        help="Create deterministic synthetic responses for verified prompt pairs",
    )
    fixture_results.add_argument("--pairs", type=Path, required=True)
    fixture_results.add_argument("--output", type=Path, required=True)
    return parser


def _print_manifest(path: Path, manifest: dict) -> None:
    print(
        json.dumps(
            {
                "artifact": str(path),
                "manifest": str(path) + ".manifest.json",
                "artifact_kind": manifest["provenance"]["artifact_kind"],
                "role": manifest["provenance"]["role"],
                "row_count": manifest["row_count"],
                "content_sha256": manifest["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _json_options(value: str, *, label: str) -> dict:
    options = parse_json_object(value, label=label)
    reject_inline_secrets(options, path=label)
    return options


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "inventory":
        payload = {
            "paper_id": "irpan_2510_27062",
            "sources": source_registry_payload(),
            "partitions": partition_registry_payload(),
            "reproduction_boundary": reconstruction_ledger(),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            if args.output.exists():
                parser.error(f"refusing to overwrite inventory: {args.output}")
            write_atomic_bytes(args.output, text.encode("utf-8"))
        print(text, end="")
        return
    if args.command == "materialize-eval":
        if args.expected_count is not None and args.expected_count < 1:
            parser.error("--expected-count must be >= 1")
        try:
            manifest = materialize_eval_artifact(
                args.source_path,
                args.output,
                source=args.source,
                subset=args.subset,
                split=args.split,
                revision=args.revision,
                expected_count=args.expected_count,
                expected_count_mode=args.expected_count_mode,
            )
        except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "verify":
        try:
            _rows, manifest = read_artifact(
                args.path,
                expected_kind=args.kind,
                expected_role=args.role,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.path, manifest)
        return
    if args.command == "materialize-retained-pairs":
        try:
            manifest = materialize_retained_prompt_pairs(args.retained, args.output)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "artifact": str(args.output),
                    "manifest": str(args.output) + ".manifest.json",
                    "artifact_schema": manifest["artifact_schema"],
                    "row_count": manifest["row_count"],
                    "content_sha256": manifest["content_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "normalize-harmbench-training":
        try:
            manifest = materialize_harmbench_source(
                args.input,
                args.output,
                subset=args.subset,
                split=args.split,
                source_revision=args.source_revision,
                expected_file_sha256=args.expected_file_sha256,
                id_field=args.id_field,
                prompt_field=args.prompt_field,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "build-wrappers":
        try:
            manifest = materialize_wrapper_candidates(args.source, args.output)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "build-completion-requests":
        try:
            manifest = materialize_completion_requests(
                args.candidates,
                args.output,
                generator=_json_options(args.generator, label="generator"),
                decoding_params=_json_options(args.decoding_params, label="decoding_params"),
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "import-completions":
        try:
            manifest = materialize_external_completions(
                args.requests,
                args.results,
                args.output,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "build-judgment-requests":
        try:
            manifest = materialize_judgment_requests(
                args.completions,
                args.output,
                judge=_json_options(args.judge, label="judge"),
                decoding_params=_json_options(args.decoding_params, label="decoding_params"),
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "import-judgments":
        try:
            manifest = materialize_external_judgments(
                args.requests,
                args.results,
                args.output,
                max_parse_retries=args.max_parse_retries,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.output, manifest)
        return
    if args.command == "filter-vulnerabilities":
        try:
            audit_manifest, retained_manifest = materialize_vulnerability_filter(
                args.candidates,
                args.judgments,
                args.audit_output,
                args.retained_output,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        _print_manifest(args.audit_output, audit_manifest)
        _print_manifest(args.retained_output, retained_manifest)
        return
    if args.command == "collect-validation-observations":
        try:
            observations = materialize_validation_observations(
                args.log_dir,
                args.output,
                domain=args.domain,
                method=args.method,
                schema=args.schema,
            )
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {
                    "domain": args.domain,
                    "method": args.method,
                    "output": str(args.output),
                    "row_count": len(observations),
                    "schema": args.schema,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "select-validation":
        if args.output.exists():
            parser.error(f"refusing to overwrite selection audit: {args.output}")
        try:
            observations = read_external_result_export(args.input)
            _verify_selection_method(observations, args.method)
            audit = rank_validation_candidates(observations, domain=args.domain).as_dict()
            payload = (json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
            write_atomic_bytes(args.output, payload)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if args.command == "materialize-smoke-fixtures":
        try:
            paths = materialize_smoke_fixtures(args.output_dir)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))
        return
    if args.command == "build-smoke-bct-results":
        try:
            row_count = materialize_bct_result_fixture(args.pairs, args.output)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        print(json.dumps({"output": str(args.output), "row_count": row_count}, indent=2, sort_keys=True))
        return
    raise AssertionError(f"unhandled command: {args.command}")


def _verify_selection_method(observations: list[dict], method: str | None) -> None:
    if method is None:
        return
    if not method or method != method.strip():
        raise ValueError("--method must be a non-empty, exactly formatted string")
    mismatches: list[str] = []
    for index, observation in enumerate(observations, start=1):
        details = observation.get("candidate_details")
        if not isinstance(details, Mapping) or details.get("method") != method:
            candidate_id = observation.get("candidate_id", f"row-{index}")
            mismatches.append(str(candidate_id))
    if mismatches:
        raise ValueError(
            f"--method {method!r} does not match candidate_details.method for candidates {sorted(mismatches)}"
        )


if __name__ == "__main__":
    main()


__all__ = ["main"]
