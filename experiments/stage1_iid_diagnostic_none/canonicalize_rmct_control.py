"""Canonically publish an accelerated RMCT-control Stage 1 IID analysis.

The accelerated HF/PEFT evaluation used the operational condition name
``rmct-control-b8-accelerated``.  Its graded EvalLogs are immutable evidence,
so this utility never renames, copies, edits, or re-grades them.  Instead it
re-extracts their observations, binds their IDs to the frozen no-CoT manifest,
and writes a *new* analysis using the paper-facing condition name
``rmct-control``.

The alias analysis is an input rather than a convenience.  The tool proves
that every pooled and per-dataset value is exactly reproduced from the
supplied EvalLogs apart from the condition label, then computes the new
training-prefix ``rmct_first64`` subset directly from frozen manifest IDs.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.stage1_iid_diagnostic import analyze


SCHEMA = "stage1-iid-rmct-control-canonicalization-v1"
ALIAS_CONDITION = "rmct-control-b8-accelerated"
CANONICAL_CONDITION = "rmct-control"
_EXPECTED_CELLS = frozenset(
    (split, dataset) for split in analyze.SPLITS for dataset in analyze.DATASETS
)


def _validate_frozen_manifest(manifest_path: Path) -> tuple[dict[str, Any], int]:
    """Load the no-CoT validator only when canonicalization actually runs.

    The validator imports training-source code whose optional GPU dependencies
    are intentionally absent from lightweight offline report environments.
    Keeping that import here preserves the production validation boundary while
    allowing this read-only tool (and its focused tests) to be imported there.
    """

    from experiments.stage1_iid_diagnostic_none import prepare

    return prepare.validate_manifest(manifest_path, verify_source=False), prepare.RMCT_ROWS


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_payload(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return document


def _relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _expected_ids_by_cell(manifest: Mapping[str, Any]) -> dict[tuple[str, str], tuple[str, ...]]:
    """Read the manifest-bound frozen split files into per-dataset ID lists.

    ``prepare.validate_manifest`` verifies the split files against the
    manifest.  This additional, deliberately small read maps its frozen IDs
    onto the four EvalLog cells, so a set of correctly-sized but wrong-dataset
    logs cannot be canonicalized.
    """

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):  # defensive; validate_manifest has already checked this.
        raise ValueError("frozen no-CoT manifest has no split mapping")
    result: dict[tuple[str, str], tuple[str, ...]] = {}
    for split in analyze.SPLITS:
        entry = splits.get(split)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError(f"frozen no-CoT manifest has no path for {split!r}")
        path = Path(entry["path"])
        by_dataset: dict[str, list[str]] = {dataset: [] for dataset in analyze.DATASETS}
        all_ids: list[str] = []
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw_line.strip():
                raise ValueError(f"frozen {split!r} split contains a blank line: {path}:{line_number}")
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid frozen {split!r} JSONL row: {path}:{line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"frozen {split!r} row is not an object: {path}:{line_number}")
            question_id = row.get("question_id")
            dataset = row.get("source_dataset")
            if not isinstance(question_id, str) or not question_id or dataset not in analyze.DATASETS:
                raise ValueError(f"frozen {split!r} row has invalid identity: {path}:{line_number}")
            all_ids.append(question_id)
            by_dataset[dataset].append(question_id)
        if all_ids != entry.get("question_ids"):
            raise ValueError(f"frozen {split!r} split IDs no longer match its manifest")
        for dataset in analyze.DATASETS:
            result[(split, dataset)] = tuple(by_dataset[dataset])
    if set(result) != _EXPECTED_CELLS:
        raise AssertionError("internal frozen-cell construction error")
    return result


def _validate_alias_report(
    report: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> int:
    """Validate the metadata that binds an alias report to this final suite."""

    if report.get("schema") != analyze.ANALYSIS_SCHEMA:
        raise ValueError("alias analysis has an unsupported schema")
    if report.get("grader_model") != analyze.DEFAULT_LUNA_GRADER_MODEL:
        raise ValueError("alias analysis has an unexpected Luna grader model")
    if report.get("inspect_rescore_model") != analyze.INSPECT_RESCORE_MODEL:
        raise ValueError("alias analysis has an unexpected Inspect rescore model")
    grader_max_tokens = report.get("grader_max_tokens")
    if isinstance(grader_max_tokens, bool) or not isinstance(grader_max_tokens, int) or grader_max_tokens < 1:
        raise ValueError("alias analysis has an invalid grader_max_tokens")
    if report.get("diagnostic_manifest_sha256") != manifest_sha256:
        raise ValueError("alias analysis is bound to a different diagnostic manifest")
    # The path may differ after a legitimate artifact transfer; its content
    # hash, verified above, is the invariant that matters.
    reported_manifest = report.get("diagnostic_manifest")
    if not isinstance(reported_manifest, str) or not reported_manifest:
        raise ValueError("alias analysis has no diagnostic_manifest path")
    if not manifest_path.is_file():  # keeps the argument use explicit and fail-closed.
        raise FileNotFoundError(f"frozen diagnostic manifest does not exist: {manifest_path}")
    expected_metrics = {
        "tbsr": "P(biased answer = bias answer | clean answer != bias answer, jointly parsed)",
        "away_from_bias": "P(biased answer != bias answer | clean answer = bias answer, jointly parsed)",
        "total_switch": "P(biased answer differs from clean answer | jointly parsed)",
        "luna_yes": "P(Luna YES | Luna verdict parsed)",
    }
    if report.get("metric_definitions") != expected_metrics:
        raise ValueError("alias analysis metric definitions do not match the final paired analysis")
    return grader_max_tokens


def _expected_alias_cells(cells: Any) -> dict[str, Any]:
    """Return alias pooled/per-dataset cells relabelled to the canonical name.

    The generic analyzer does not recognize the operational alias as an RMCT
    condition, so its source report correctly has no ``rmct_first64`` block.
    That block is deliberately added only by this canonicalizer, after it has
    been recomputed from the frozen manifest IDs.
    """

    if not isinstance(cells, Mapping):
        raise ValueError("alias analysis has no cells mapping")
    expected_keys = {f"{ALIAS_CONDITION}/{split}" for split in analyze.SPLITS}
    if set(cells) != expected_keys:
        raise ValueError("alias analysis must contain exactly the two accelerated RMCT-control cells")
    canonicalized: dict[str, Any] = {}
    for split in analyze.SPLITS:
        key = f"{ALIAS_CONDITION}/{split}"
        value = cells[key]
        if not isinstance(value, Mapping):
            raise ValueError(f"alias analysis cell {key!r} is not an object")
        if value.get("condition") != ALIAS_CONDITION or value.get("split") != split:
            raise ValueError(f"alias analysis cell identity is inconsistent: {key!r}")
        if set(value) != {"condition", "split", "pooled", "per_dataset"}:
            raise ValueError(f"alias analysis cell has unexpected fields: {key!r}")
        copied = copy.deepcopy(dict(value))
        copied["condition"] = CANONICAL_CONDITION
        canonicalized[f"{CANONICAL_CONDITION}/{split}"] = copied
    return canonicalized


def _validate_rows_against_manifest(
    rows: Sequence[analyze.Observation],
    expected_ids: Mapping[tuple[str, str], Sequence[str]],
) -> None:
    grouped: dict[tuple[str, str], list[str]] = {cell: [] for cell in _EXPECTED_CELLS}
    for row in rows:
        if row.condition != ALIAS_CONDITION:
            raise ValueError(f"graded EvalLog has unexpected condition {row.condition!r}")
        cell = (row.split, row.dataset)
        if cell not in grouped:
            raise ValueError(f"graded EvalLog has unexpected cell {cell!r}")
        grouped[cell].append(row.question_id)
    for cell in sorted(_EXPECTED_CELLS):
        observed = tuple(grouped[cell])
        expected = tuple(expected_ids[cell])
        observed_ids = set(observed)
        expected_ids_for_cell = set(expected)
        # Inspect's EvalLog loader can order samples by their persisted sample
        # identity rather than the source-JSONL order. The immutable contract
        # is therefore exact membership (and uniqueness), not incidental row
        # ordering. Keep the check fail-closed for missing, swapped-dataset,
        # duplicate, or extra question IDs.
        if (
            len(observed) != len(expected)
            or len(observed_ids) != len(observed)
            or observed_ids != expected_ids_for_cell
        ):
            missing = sorted(expected_ids_for_cell - observed_ids)
            unexpected = sorted(observed_ids - expected_ids_for_cell)
            raise ValueError(
                "graded EvalLog IDs do not exactly match the frozen manifest for "
                f"{cell!r}: expected {len(expected)} unique IDs, got {len(observed)} "
                f"rows/{len(observed_ids)} unique IDs; missing={missing[:3]}, "
                f"unexpected={unexpected[:3]}"
            )


def _validate_sources(
    sources: Sequence[Mapping[str, Any]], *, alias_report: Mapping[str, Any]) -> None:
    expected_sources = alias_report.get("sources")
    if not isinstance(expected_sources, list):
        raise ValueError("alias analysis has no sources list")
    if list(sources) != expected_sources:
        raise ValueError("alias analysis sources are not exactly reproducible from the supplied graded EvalLogs")
    cells: set[tuple[str, str]] = set()
    for source in sources:
        if source.get("condition") != ALIAS_CONDITION:
            raise ValueError("graded EvalLog provenance has a non-alias condition")
        split = source.get("split")
        dataset = source.get("dataset")
        if (split, dataset) not in _EXPECTED_CELLS:
            raise ValueError("graded EvalLog provenance has an unexpected diagnostic cell")
        cells.add((split, dataset))
    if cells != _EXPECTED_CELLS or len(sources) != len(_EXPECTED_CELLS):
        raise ValueError("graded EvalLog provenance does not contain exactly four diagnostic cells")


def _canonical_sources(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(source),
            "condition": CANONICAL_CONDITION,
            "source_condition": ALIAS_CONDITION,
        }
        for source in sources
    ]


def build_canonical_report(
    alias_analysis: str | Path,
    graded_root: str | Path,
    manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute and validate a canonical RMCT-control report, offline only.

    Returns ``(report, provenance)`` without writing either object.  All input
    EvalLogs are read through the normal Inspect analysis loader; no call here
    can invoke a model, grader, or generation backend.
    """

    alias_path = Path(alias_analysis).resolve()
    graded_path = Path(graded_root).resolve()
    manifest_path = Path(manifest).resolve()
    alias_payload = alias_path.read_bytes() if alias_path.is_file() else None
    if alias_payload is None:
        raise FileNotFoundError(f"alias analysis does not exist: {alias_path}")
    alias_report = _read_json_object(alias_path, label="alias analysis")
    manifest_payload = manifest_path.read_bytes() if manifest_path.is_file() else None
    if manifest_payload is None:
        raise FileNotFoundError(f"frozen diagnostic manifest does not exist: {manifest_path}")
    manifest_sha256 = _sha256_bytes(manifest_payload)
    grader_max_tokens = _validate_alias_report(
        alias_report,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )

    # validate_manifest proves the split bytes/IDs/counts against the frozen
    # no-CoT contract.  It intentionally does not require the full 3,000-row
    # source to be co-located with this analysis handoff.
    frozen_manifest, rmct_rows = _validate_frozen_manifest(manifest_path)
    expected_ids = _expected_ids_by_cell(frozen_manifest)
    first64 = frozen_manifest["rmct_first64"]["question_ids"]
    if not isinstance(first64, list) or len(first64) != rmct_rows or any(
        not isinstance(question_id, str) for question_id in first64
    ):
        raise ValueError("frozen diagnostic manifest has invalid rmct_first64 IDs")

    rows, sources = analyze.load_graded_logs(graded_path, grader_max_tokens=grader_max_tokens)
    _validate_sources(sources, alias_report=alias_report)
    _validate_rows_against_manifest(rows, expected_ids)
    canonical_rows = [dataclasses.replace(row, condition=CANONICAL_CONDITION) for row in rows]
    canonical_cells = analyze.grouped_report(canonical_rows, set(first64))
    first64_summary = canonical_cells.get(f"{CANONICAL_CONDITION}/train_eval", {}).get("rmct_first64", {})
    if first64_summary.get("pooled", {}).get("counts", {}).get("samples") != len(first64):
        raise AssertionError("canonical RMCT-control analysis did not recompute the complete rmct_first64 subset")
    expected_cells = _expected_alias_cells(alias_report.get("cells"))
    canonical_full_cells = {
        key: {field: value for field, value in cell.items() if field != "rmct_first64"}
        for key, cell in canonical_cells.items()
    }
    if canonical_full_cells != expected_cells:
        raise ValueError(
            "pooled/per-dataset accelerated RMCT-control analysis is not exactly reproducible "
            "from the supplied graded EvalLogs (beyond the condition label)"
        )

    report = {
        "schema": analyze.ANALYSIS_SCHEMA,
        "grader_model": analyze.DEFAULT_LUNA_GRADER_MODEL,
        "grader_max_tokens": grader_max_tokens,
        "inspect_rescore_model": analyze.INSPECT_RESCORE_MODEL,
        "diagnostic_manifest": str(manifest_path),
        "diagnostic_manifest_sha256": manifest_sha256,
        "metric_definitions": alias_report["metric_definitions"],
        "sources": _canonical_sources(sources),
        "cells": canonical_cells,
    }
    report_payload = _json_payload(report)
    alias_cells_payload = _json_payload({"cells": alias_report["cells"]})
    canonical_cells_payload = _json_payload({"cells": canonical_cells})
    provenance = {
        "schema": SCHEMA,
        "operation": "offline_alias_condition_canonicalization",
        "alias_condition": ALIAS_CONDITION,
        "canonical_condition": CANONICAL_CONDITION,
        "alias_analysis": str(alias_path),
        "alias_analysis_sha256": _sha256_bytes(alias_payload),
        "graded_root": str(graded_path),
        "diagnostic_manifest": str(manifest_path),
        "diagnostic_manifest_sha256": manifest_sha256,
        "rmct_first64": {
            "row_count": len(first64),
            "question_ids_sha256": frozen_manifest["rmct_first64"]["question_ids_sha256"],
            "recomputed_from_manifest_ids": True,
        },
        "graded_sources": list(sources),
        "checks": {
            "frozen_manifest_validated": True,
            "graded_eval_log_ids_exactly_match_manifest": True,
            "alias_analysis_sources_exactly_match_graded_eval_logs": True,
            "pooled_and_per_dataset_cells_equivalent_except_condition_label": True,
            "raw_or_graded_logs_modified": False,
        },
        "alias_cells_sha256": _sha256_bytes(alias_cells_payload),
        "canonical_cells_sha256": _sha256_bytes(canonical_cells_payload),
        "canonical_analysis_sha256": _sha256_bytes(report_payload),
    }
    return report, provenance


def _check_output(path: Path, payload: bytes) -> str:
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"canonicalization output is not a file: {path}")
        if path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing canonicalization output: {path}")
        return "resumed"
    return "written"


def write_canonicalization(
    report: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    output: str | Path,
    provenance_output: str | Path,
    graded_root: str | Path,
    alias_analysis: str | Path,
    manifest: str | Path,
) -> tuple[str, str]:
    """Atomically publish only new report/provenance files, never log files."""

    report_path = Path(output).resolve()
    provenance_path = Path(provenance_output).resolve()
    graded_path = Path(graded_root).resolve()
    protected_inputs = {Path(alias_analysis).resolve(), Path(manifest).resolve()}
    graded_sources = provenance.get("graded_sources")
    if isinstance(graded_sources, Sequence) and not isinstance(graded_sources, (str, bytes)):
        for source in graded_sources:
            if not isinstance(source, Mapping):
                continue
            for field in ("graded_log", "source_log"):
                raw_path = source.get(field)
                if isinstance(raw_path, str) and raw_path:
                    protected_inputs.add(Path(raw_path).resolve())
    if report_path == provenance_path:
        raise ValueError("canonical report and canonicalization provenance outputs must be distinct")
    for path in (report_path, provenance_path):
        if path in protected_inputs:
            raise ValueError(f"canonicalization output would overwrite an input: {path}")
        if _relative_to(path, graded_path):
            raise ValueError("canonicalization outputs must be outside the immutable graded-log root")
    report_payload = _json_payload(report)
    provenance_payload = _json_payload(provenance)
    report_status = _check_output(report_path, report_payload)
    provenance_status = _check_output(provenance_path, provenance_payload)
    # Preflight both destinations before writing either, avoiding a partially
    # published pair when a pre-existing conflicting provenance is discovered.
    if report_status == "written":
        analyze.write_report(report_path, report)
    if provenance_status == "written":
        analyze.write_report(provenance_path, provenance)
    return report_status, provenance_status


def canonicalize(
    alias_analysis: str | Path,
    graded_root: str | Path,
    manifest: str | Path,
    *,
    output: str | Path,
    provenance_output: str | Path,
) -> tuple[str, str]:
    """Build, verify, and publish the report/provenance pair."""

    report, provenance = build_canonical_report(alias_analysis, graded_root, manifest)
    return write_canonicalization(
        report,
        provenance,
        output=output,
        provenance_output=provenance_output,
        graded_root=graded_root,
        alias_analysis=alias_analysis,
        manifest=manifest,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alias-analysis", required=True, type=Path)
    parser.add_argument("--graded-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new canonical rmct-control analysis JSON")
    parser.add_argument(
        "--provenance-output",
        required=True,
        type=Path,
        help="new canonicalization provenance JSON",
    )
    args = parser.parse_args(argv)
    try:
        report_status, provenance_status = canonicalize(
            args.alias_analysis,
            args.graded_root,
            args.manifest,
            output=args.output,
            provenance_output=args.provenance_output,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"analysis={report_status}: {args.output.resolve()}")
    print(f"provenance={provenance_status}: {args.provenance_output.resolve()}")


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    main()
