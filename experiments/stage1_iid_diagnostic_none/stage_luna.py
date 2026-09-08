"""Stage hash-bound no-CoT Stage 1 raw logs for posthoc Luna grading.

The Stage 1 grader intentionally operates on a separate, portable staging
tree.  This small CPU-only helper is the missing counterpart to the Stage 2
stager: it copies only the four biased logs named by a validated no-CoT raw
preflight report, verifies their bytes before and after the copy, and never
overwrites a differing destination.  It does not import Inspect, load a model,
or make a network request.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_none_preflight_report(report: str | Path) -> dict[str, object]:
    """Load the stricter no-CoT report validator only when staging is invoked.

    Keeping this import lazy preserves the stager's useful CPU-only boundary:
    merely inspecting its CLI does not initialize the optional Inspect/MCQ
    stack that the grading implementation imports.
    """

    from experiments.stage1_iid_diagnostic_none.grade_luna import (
        validate_none_preflight_report as validate,
    )

    return validate(report)


def _safe_component(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"Stage 1 Luna preflight report has an invalid {field}")
    return value


def _staged_path(root: Path, *, condition: str, source: Mapping[str, object]) -> Path:
    """Return the canonical staging destination for one hash-bound source."""

    split = _safe_component(source.get("split"), field="split")
    raw_log = source.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        raise ValueError("Stage 1 Luna preflight report source has no raw_log")
    filename = Path(raw_log).name
    if not filename or filename in {".", ".."} or not filename.endswith(".eval"):
        raise ValueError("Stage 1 Luna preflight report source raw_log is not an .eval file")
    destination = (root / condition / split / filename).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:  # Defensive; component checks above make escaping impossible.
        raise ValueError("Stage 1 Luna staging destination escapes the supplied output root") from exc
    return destination


def _copy_new(source: Path, destination: Path, *, expected_sha256: str) -> str:
    """Copy one log atomically, or resume only an exact previous staging copy."""

    if destination.exists():
        if destination.is_file() and _sha256(destination) == expected_sha256:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing staged Stage 1 raw log: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _sha256(temporary) != expected_sha256:
            raise ValueError(f"copied Stage 1 raw log has the wrong SHA-256: {source}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not destination.is_file() or _sha256(destination) != expected_sha256:
                raise FileExistsError(f"staged Stage 1 raw log appeared and differs: {destination}")
            return "resumed"
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return "staged"


def stage_from_preflight(preflight_report: str | Path, output_root: str | Path) -> list[tuple[Path, str]]:
    """Stage exactly the report-listed no-CoT biased logs with SHA-256 checks."""

    report_path = Path(preflight_report).resolve()
    report = validate_none_preflight_report(report_path)
    source_root_value = report.get("raw_root")
    if not isinstance(source_root_value, str) or not source_root_value or not Path(source_root_value).is_absolute():
        raise ValueError("Stage 1 Luna preflight report has no absolute raw_root")
    source_root = Path(source_root_value).resolve()
    destination_root = Path(output_root).resolve()
    if destination_root == source_root or destination_root.is_relative_to(source_root) or source_root.is_relative_to(destination_root):
        raise ValueError("Stage 1 Luna staging root must be separate and non-nested from the generation raw-log root")

    condition = _safe_component(report.get("condition"), field="condition")
    sources = report.get("sources")
    if not isinstance(sources, list):  # The wrapper already validates this; keep the boundary self-contained.
        raise ValueError("Stage 1 Luna preflight report has no sources list")
    results: list[tuple[Path, str]] = []
    seen_destinations: set[Path] = set()
    for entry in sources:
        if not isinstance(entry, Mapping):
            raise ValueError("Stage 1 Luna preflight report source is not an object")
        raw_log = entry.get("raw_log")
        expected_sha256 = entry.get("raw_log_sha256")
        if not isinstance(raw_log, str) or not raw_log or not Path(raw_log).is_absolute():
            raise ValueError("Stage 1 Luna preflight report source has no absolute raw_log")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("Stage 1 Luna preflight report source has an invalid raw_log_sha256")
        source = Path(raw_log).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"Stage 1 raw log named by preflight is outside raw_root: {source}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"Stage 1 raw log named by preflight does not exist: {source}")
        if _sha256(source) != expected_sha256:
            raise ValueError(f"Stage 1 raw log changed after preflight: {source}")
        destination = _staged_path(destination_root, condition=condition, source=entry)
        if destination in seen_destinations:
            raise ValueError(f"Stage 1 preflight report maps multiple cells to one staging destination: {destination}")
        seen_destinations.add(destination)
        results.append((destination, _copy_new(source, destination, expected_sha256=expected_sha256)))
    if len(results) != 4:
        raise ValueError(f"Stage 1 raw preflight report must contain exactly four logs to stage, got {len(results)}")
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
    print(f"Stage 1 Luna staging complete: {args.output_root.resolve()} ({counts})")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
