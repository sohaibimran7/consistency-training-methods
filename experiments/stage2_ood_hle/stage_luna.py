"""Stage only raw logs bound by a Stage 2 OOD preflight report.

This is a local filesystem operation: it makes no model or network requests.
It copies the report's eighteen biased raw EvalLogs to the canonical handoff
layout used by :mod:`experiments.stage2_ood_hle.grade_luna`, verifies source
and copied SHA-256 values, and never overwrites differing files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from experiments.stage2_ood_hle.grade_luna import _staged_path
from experiments.stage2_ood_hle.raw_preflight import validate_preflight_report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_new(source: Path, destination: Path, *, expected_sha256: str) -> str:
    """Copy one raw log atomically, or resume only an exact prior copy."""

    if destination.exists():
        if destination.is_file() and _sha256(destination) == expected_sha256:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing staged Stage 2 raw log: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"copied Stage 2 raw log has the wrong SHA-256: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or _sha256(destination) != expected_sha256:
                raise FileExistsError(f"staged Stage 2 raw log appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "staged"


def stage_from_preflight(preflight_report: str | Path, output_root: str | Path) -> list[tuple[Path, str]]:
    """Hash-verify and stage exactly the report-listed eighteen biased logs."""

    report_path = Path(preflight_report).resolve()
    report = validate_preflight_report(report_path)
    source_root = Path(str(report["raw_root"])).resolve()
    destination_root = Path(output_root).resolve()
    if destination_root == source_root or destination_root.is_relative_to(source_root) or source_root.is_relative_to(destination_root):
        raise ValueError("Stage 2 Luna staging root must be separate and non-nested from the generation raw-log root")
    condition = str(report["condition"])
    sources = report["sources"]
    assert isinstance(sources, list)  # guaranteed by validate_preflight_report
    results: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for entry in sources:
        assert isinstance(entry, Mapping)  # guaranteed by validate_preflight_report
        if entry["kind"] != "biased":
            continue
        source = Path(str(entry["raw_log"])).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"raw Stage 2 log named by preflight report does not exist: {source}")
        expected_sha256 = str(entry["raw_log_sha256"])
        if _sha256(source) != expected_sha256:
            raise ValueError(f"raw Stage 2 log changed after preflight: {source}")
        destination = _staged_path(destination_root, condition=condition, source=entry)
        if destination in seen:
            raise ValueError(f"raw preflight report maps multiple cells to one stage destination: {destination}")
        seen.add(destination)
        results.append((destination, _copy_new(source, destination, expected_sha256=expected_sha256)))
    if len(results) != 18:
        raise ValueError(f"raw preflight report must contain exactly 18 biased logs to stage, got {len(results)}")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        results = stage_from_preflight(args.preflight_report, args.output_root)
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    counts = {status: sum(1 for _, observed in results if observed == status) for status in {status for _, status in results}}
    print(f"Stage 2 Luna staging complete: {args.output_root.resolve()} ({counts})")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
