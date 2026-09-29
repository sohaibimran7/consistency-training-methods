"""Grade only raw logs proven to belong to the final no-CoT diagnostic.

The shared Stage 1 Luna runner intentionally accepts a generic raw-preflight
v1 report.  That is useful for historical data, but it is not sufficient at
this final-result boundary: a report made for the old CoT population has the
same schema.  This wrapper requires and validates the no-CoT report contract
before delegating the actual API work to the shared, connection-capped grader.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.stage1_iid_diagnostic import grade_luna as shared_grade_luna
from experiments.stage1_iid_diagnostic.raw_preflight import PREFLIGHT_SCHEMA
from experiments.stage1_iid_diagnostic_none.prepare import BIAS_TYPE, PROMPT_STYLE, SOURCE_SHA256


def validate_none_preflight_report(report: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Reject a generic, historical-CoT, or incomplete preflight report."""

    if isinstance(report, Mapping):
        document = dict(report)
    else:
        path = Path(report)
        if not path.is_file():
            raise FileNotFoundError(f"no-CoT raw preflight report does not exist: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid no-CoT raw preflight JSON report: {path}") from exc
    if document.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("unsupported raw preflight report schema")
    contract = document.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("no-CoT raw preflight report has no contract object")
    expected = {
        "source_sha256": SOURCE_SHA256,
        "bias_type": BIAS_TYPE,
        "prompt_style": PROMPT_STYLE,
        "native_variant_file": None,
        "include_bias_acknowledged": False,
        "grader_model": None,
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"raw preflight report does not bind the final no-CoT contract: "
                f"{field}={contract.get(field)!r}"
            )
    sources = document.get("sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise ValueError("no-CoT raw preflight report must contain exactly four split/dataset sources")
    expected_cells = {
        ("train_eval", "logiqa"),
        ("train_eval", "hellaswag"),
        ("heldout_in_domain", "logiqa"),
        ("heldout_in_domain", "hellaswag"),
    }
    cells: set[tuple[str, str]] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError("no-CoT raw preflight report source must be an object")
        cell = (source.get("split"), source.get("dataset"))
        cells.add(cell)
        if source.get("prompt_style") != PROMPT_STYLE or source.get("source_identity_digest") != SOURCE_SHA256:
            raise ValueError(f"raw preflight source does not bind the final no-CoT contract: {cell!r}")
        if source.get("variant_file") is not None:
            raise ValueError(f"raw preflight source has an unexpected alternate prompt file: {cell!r}")
    if cells != expected_cells:
        raise ValueError(f"no-CoT raw preflight report has incomplete or unexpected cells: {sorted(cells)!r}")
    return document


def grade_all(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    preflight_report: str | Path,
    smoke: bool = False,
    smoke_samples: int = 2,
    workers: int = shared_grade_luna.DEFAULT_WORKERS,
    connections_per_worker: int = shared_grade_luna.DEFAULT_CONNECTIONS_PER_WORKER,
    grader_max_tokens: int = shared_grade_luna.DEFAULT_MAX_TOKENS,
) -> list[tuple[shared_grade_luna.GradeInput, str]]:
    """Validate the final no-CoT handoff, then run the shared Luna grader."""

    validate_none_preflight_report(preflight_report)
    return shared_grade_luna.grade_all(
        raw_root,
        output_root,
        preflight_report=preflight_report,
        smoke=smoke,
        smoke_samples=smoke_samples,
        workers=workers,
        connections_per_worker=connections_per_worker,
        grader_max_tokens=grader_max_tokens,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-log-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--preflight-report",
        required=True,
        type=Path,
        help="no-CoT raw-preflight report; generic historical-CoT reports are rejected",
    )
    parser.add_argument("--smoke", action="store_true", help="grade only the first two samples from one biased log")
    parser.add_argument("--smoke-samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=shared_grade_luna.DEFAULT_WORKERS)
    parser.add_argument("--connections-per-worker", type=int, default=shared_grade_luna.DEFAULT_CONNECTIONS_PER_WORKER)
    parser.add_argument("--grader-max-tokens", type=int, default=shared_grade_luna.DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)
    try:
        results = grade_all(
            args.raw_log_root,
            args.output_root,
            preflight_report=args.preflight_report,
            smoke=args.smoke,
            smoke_samples=args.smoke_samples,
            workers=args.workers,
            connections_per_worker=args.connections_per_worker,
            grader_max_tokens=args.grader_max_tokens,
        )
    except (FileExistsError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    if not results:
        raise SystemExit("no successful biased no-CoT Stage 1 IID logs found")
    for source, status in results:
        print(f"{status}: {source.condition}/{source.split}/{source.dataset}: {source.path}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
