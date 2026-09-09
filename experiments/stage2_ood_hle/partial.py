"""Build an explicitly incomplete, local-only Stage 2 OOD analysis artifact.

The publication Stage 2 analyzer deliberately fails unless every condition
has the full 2 x 6 matrix.  That is the right default for a final result, but
it makes it unnecessarily hard to inspect cells which have already completed
while other conditions are still running.  This module is a separate,
fail-closed path for that situation:

* a canonical complete analysis supplies the fixed Base/ACT/AttCT/MLPCT rows;
* only complete (IID=200, HLE=100) new Inspect cells are admitted;
* incomplete cells are retained only as availability metadata, never rates;
* a held-out-bias micro-pool is emitted only after all five component cells;
* CIs use the regular question-cluster bootstrap and stars use the existing
  uncorrected two-sided two-proportion z-test implementation.

It deliberately has a distinct schema and output namespace.  It must never
be substituted for :mod:`experiments.stage2_ood_hle.analyze`'s strict final
four-column report.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.analysis import _significance_label, _two_proportion_p_value
from experiments.stage2_ood_hle.analyze import (
    ANALYSIS_SCHEMA,
    BIAS_VERBALISED_METRIC,
    COUNT_FIELDS,
    HLE_POPULATION,
    IID_POPULATION,
    POPULATIONS,
    TBSR_METRIC,
    AnalysisConfig,
    BootstrapConfig,
    Observation,
    _cell,
    _headline_cell,
    _ordered_conditions,
    _question_ids_digest,
    _validate_bootstrap,
    _validate_counts,
    _valid_rate,
    observation_from_mapping,
    parse_runs,
    validate_report,
)

PARTIAL_ANALYSIS_SCHEMA = "stage2-ood-hle-partial-analysis-v1"
PARTIAL_HELD_OUT_MEAN = "held_out_mean"
PARTIAL_CELL_SOURCE_CANONICAL = "canonical_complete_analysis"
PARTIAL_CELL_SOURCE_INSPECT = "complete_partial_inspect_luna"
SIGNIFICANCE_METHOD = "two_sided_two_proportion_z_test"
SIGNIFICANCE_MULTIPLICITY = "uncorrected"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_from_complete_report(report: Mapping[str, Any]) -> AnalysisConfig:
    """Reconstruct the exact analysis configuration embedded in a final report."""

    config = report.get("config")
    if not isinstance(config, Mapping):  # defensive; ``validate_report`` has checked this already
        raise ValueError("canonical Stage 2 report has no config")
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("canonical Stage 2 report has no bootstrap config")
    return AnalysisConfig(
        iid_questions=int(config.get("iid_questions", 0)),
        hle_questions=int(config.get("hle_questions", 0)),
        training_bias=str(config.get("training_bias", "")),
        held_out_biases=tuple(str(value) for value in (config.get("held_out_biases") or [])),
        expected_prompt_style=str(config.get("expected_prompt_style", "")),
        bootstrap=BootstrapConfig(
            replicates=int(bootstrap.get("replicates", 0)),
            seed=int(bootstrap.get("seed", -1)),
            chunk_size=int(bootstrap.get("chunk_size", 0)),
        ),
    )


def _cell_key(condition: str, population: str, bias_type: str) -> str:
    return f"{condition}/{population}/{bias_type}"


def _summary_key(condition: str, population: str) -> str:
    return f"{condition}/{population}/{PARTIAL_HELD_OUT_MEAN}"


def _summary_column(population: str) -> str:
    if population == IID_POPULATION:
        return "held_out_bias"
    if population == HLE_POPULATION:
        return "held_out_dataset_and_bias"
    raise ValueError(f"unsupported partial population {population!r}")


def _expected_cell_keys(conditions: Sequence[str], config: AnalysisConfig) -> set[str]:
    return {
        _cell_key(condition, population, bias)
        for condition in conditions
        for population in POPULATIONS
        for bias in (config.training_bias, *config.held_out_biases)
    }


def _copy_complete_cells(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cells = report.get("per_bias_cells")
    if not isinstance(cells, Mapping):  # defensive; final validator owns the public boundary
        raise ValueError("canonical Stage 2 report has no per-bias cells")
    copied: dict[str, dict[str, Any]] = {}
    for key, value in cells.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("canonical Stage 2 report has an invalid per-bias cell")
        cell = copy.deepcopy(dict(value))
        cell["source"] = PARTIAL_CELL_SOURCE_CANONICAL
        # The canonical report is itself built only after Inspect selected a
        # complete Luna-score mapping for every source log.  A verdict may be
        # null (a parse failure/cap hit) without making the log ungraded.
        cell["luna_score_complete"] = True
        copied[key] = cell
    return copied


def _canonical_digest(report: Mapping[str, Any], *, population: str, bias_type: str) -> str:
    cells = report["per_bias_cells"]
    assert isinstance(cells, Mapping)  # checked by ``validate_report``
    cell = cells[_cell_key("untrained", population, bias_type)]
    assert isinstance(cell, Mapping)
    digest = cell.get("question_ids_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"canonical Base {population}/{bias_type} has no question-id digest")
    return digest


def _partial_condition_order(canonical_conditions: Sequence[str], observations: Sequence[Observation]) -> list[str]:
    all_conditions = set(str(condition) for condition in canonical_conditions)
    all_conditions.update(record.condition for record in observations)
    # Match the familiar Stage 2 condition ordering rather than appending
    # newly arriving BCT/RMCT rows after the canonical ACT-family rows.
    proxy = [
        Observation(
            condition=condition,
            population=IID_POPULATION,
            question_id="ordering-proxy",
            bias_type="ordering-proxy",
            joint_parse=False,
            clean_matches_bias=None,
            towards_bias_switch=None,
        )
        for condition in all_conditions
    ]
    return _ordered_conditions(proxy)


def _availability_record(
    records: Sequence[Observation],
    *,
    expected_questions: int,
    require_luna: bool,
    luna_score_complete: bool,
) -> dict[str, Any]:
    ids = [record.question_id for record in records]
    unique_ids = set(ids)
    if len(records) != len(unique_ids):
        status = "invalid_duplicate_question_id"
    elif len(unique_ids) > expected_questions:
        status = "invalid_excess_questions"
    elif len(unique_ids) < expected_questions:
        status = "incomplete"
    elif require_luna and not luna_score_complete:
        status = "ungraded_luna"
    else:
        status = "complete"
    return {
        "status": status,
        "expected_questions": expected_questions,
        "observations": len(records),
        "unique_question_ids": len(unique_ids),
        "luna_score_complete": luna_score_complete,
    }


def _append_significance(
    cell: dict[str, Any],
    *,
    baseline: Mapping[str, Any],
    condition: str,
    baseline_condition: str,
) -> None:
    """Use the publication pipeline's exact z-test and star thresholds."""

    annotations: dict[str, dict[str, Any]] = {}
    for metric, rate_key in ((TBSR_METRIC, "tbsr"), (BIAS_VERBALISED_METRIC, "bias_verbalised")):
        value = cell.get(rate_key)
        base_value = baseline.get(rate_key)
        if not isinstance(value, Mapping) or not isinstance(base_value, Mapping):
            raise ValueError(f"partial cell has no {rate_key} rate mapping")
        rate = value.get("rate")
        denominator = value.get("denominator")
        base_rate = base_value.get("rate")
        base_denominator = base_value.get("denominator")
        annotation: dict[str, Any] = {
            "baseline_condition": baseline_condition,
            "method": SIGNIFICANCE_METHOD,
            "multiplicity": SIGNIFICANCE_MULTIPLICITY,
            "p_value": None,
            "marker": "",
        }
        luna_complete = cell.get("luna_score_complete") is True and baseline.get("luna_score_complete") is True
        if condition == baseline_condition:
            annotation["unavailable_reason"] = "baseline_cell"
        elif metric == BIAS_VERBALISED_METRIC and not luna_complete:
            # A partially graded cell is usable for TBSR but must never become
            # a verbalisation estimate or a significance test.
            annotation["unavailable_reason"] = "incomplete_luna_grading"
        elif (
            rate is None
            or base_rate is None
            or not isinstance(denominator, int)
            or not isinstance(base_denominator, int)
            or denominator <= 0
            or base_denominator <= 0
        ):
            annotation["unavailable_reason"] = "missing_or_zero_denominator"
        else:
            p_value = _two_proportion_p_value(
                float(rate),
                denominator,
                float(base_rate),
                base_denominator,
            )
            annotation["p_value"] = p_value
            annotation["marker"] = _significance_label(p_value)
        annotations[metric] = annotation
    cell["significance"] = annotations


def _validate_signature(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} significance must be an object")
    for metric in (TBSR_METRIC, BIAS_VERBALISED_METRIC):
        annotation = value.get(metric)
        if not isinstance(annotation, Mapping):
            raise ValueError(f"{label} {metric} significance is missing")
        if annotation.get("method") != SIGNIFICANCE_METHOD:
            raise ValueError(f"{label} {metric} uses the wrong significance test")
        if annotation.get("multiplicity") != SIGNIFICANCE_MULTIPLICITY:
            raise ValueError(f"{label} {metric} must disclose uncorrected multiplicity")
        marker = annotation.get("marker")
        if marker not in {"", "*", "**", "***"}:
            raise ValueError(f"{label} {metric} has an invalid significance marker")
        p_value = annotation.get("p_value")
        if p_value is None:
            if marker:
                raise ValueError(f"{label} {metric} has a star without a p-value")
            continue
        if isinstance(p_value, bool) or not isinstance(p_value, (int, float)) or not math.isfinite(float(p_value)):
            raise ValueError(f"{label} {metric} p-value must be finite or null")
        if not 0.0 <= float(p_value) <= 1.0:
            raise ValueError(f"{label} {metric} p-value lies outside [0, 1]")
        if marker != _significance_label(float(p_value)):
            raise ValueError(f"{label} {metric} star conflicts with its p-value")


def _validate_partial_cell(
    cell: Mapping[str, Any],
    *,
    key: str,
    condition: str,
    population: str,
    bias_type: str,
    expected_questions: int,
) -> None:
    if cell.get("condition") != condition or cell.get("population") != population or cell.get("bias_type") != bias_type:
        raise ValueError(f"partial cell {key!r} has a conflicting identity")
    if cell.get("source") not in {PARTIAL_CELL_SOURCE_CANONICAL, PARTIAL_CELL_SOURCE_INSPECT}:
        raise ValueError(f"partial cell {key!r} has no recognized provenance")
    if not isinstance(cell.get("luna_score_complete"), bool):
        raise ValueError(f"partial cell {key!r} must state whether its Luna score mappings are complete")
    _validate_counts(cell.get("counts"), label=key)
    counts = cell["counts"]
    assert isinstance(counts, Mapping)
    if counts.get("attempted_pairs") != expected_questions:
        raise ValueError(f"partial cell {key!r} does not contain the complete frozen population")
    _valid_rate(cell.get("tbsr"), label=f"{key}.tbsr", allow_null=True)
    _validate_bootstrap(cell.get("bootstrap"), label=f"{key}.tbsr", metric=TBSR_METRIC)
    _valid_rate(cell.get("bias_verbalised"), label=f"{key}.bias_verbalised", allow_null=True)
    _validate_bootstrap(
        cell.get("bias_verbalised_bootstrap"), label=f"{key}.bias_verbalised", metric=BIAS_VERBALISED_METRIC
    )
    digest = cell.get("question_ids_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"partial cell {key!r} has an invalid question-id digest")
    _validate_signature(cell.get("significance"), label=key)


def validate_partial_report(report: Mapping[str, Any]) -> None:
    """Validate an exploratory report without weakening final-report validation."""

    if report.get("schema") != PARTIAL_ANALYSIS_SCHEMA:
        raise ValueError(f"partial Stage 2 report schema must be {PARTIAL_ANALYSIS_SCHEMA!r}")
    conditions = report.get("conditions")
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(not isinstance(value, str) or not value for value in conditions)
    ):
        raise ValueError("partial Stage 2 report conditions must be a non-empty string array")
    if len(conditions) != len(set(conditions)) or "untrained" not in conditions:
        raise ValueError("partial Stage 2 report must contain a unique Base/untrained condition")
    config = _config_from_complete_report({"config": report.get("config")})
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ValueError("partial Stage 2 report has no cells")
    allowed = _expected_cell_keys(conditions, config)
    if not set(cells).issubset(allowed):
        raise ValueError("partial Stage 2 report contains an unknown cell")
    baseline_cells = _expected_cell_keys(("untrained",), config)
    if not baseline_cells.issubset(cells):
        raise ValueError("partial Stage 2 report must retain every complete Base cell")
    for key, value in cells.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("partial Stage 2 report has a malformed cell")
        condition, population, bias_type = key.split("/", 2)
        _validate_partial_cell(
            value,
            key=key,
            condition=condition,
            population=population,
            bias_type=bias_type,
            expected_questions=config.expected_questions[population],
        )

    summaries = report.get("held_out_summaries")
    if not isinstance(summaries, Mapping):
        raise ValueError("partial Stage 2 report has no held-out summaries")
    allowed_summaries = {_summary_key(condition, population) for condition in conditions for population in POPULATIONS}
    if not set(summaries).issubset(allowed_summaries):
        raise ValueError("partial Stage 2 report contains an unknown held-out summary")
    for key, value in summaries.items():
        if not isinstance(key, str) or not isinstance(value, Mapping):
            raise ValueError("partial Stage 2 report has a malformed held-out summary")
        condition, population, category = key.split("/", 2)
        if category != PARTIAL_HELD_OUT_MEAN:
            raise ValueError(f"partial held-out summary {key!r} has the wrong category")
        source_keys = value.get("source_cell_keys")
        expected_sources = [_cell_key(condition, population, bias) for bias in config.held_out_biases]
        if source_keys != expected_sources or any(source not in cells for source in expected_sources):
            raise ValueError(f"partial held-out summary {key!r} lacks a complete five-bias source set")
        if value.get("source") not in {PARTIAL_CELL_SOURCE_CANONICAL, PARTIAL_CELL_SOURCE_INSPECT}:
            raise ValueError(f"partial held-out summary {key!r} has no recognized provenance")
        if not isinstance(value.get("luna_score_complete"), bool):
            raise ValueError(f"partial held-out summary {key!r} must state Luna score coverage")
        # Headline cells deliberately retain a ``column`` rather than ``bias_type``.
        if value.get("condition") != condition or value.get("column") != _summary_column(population):
            raise ValueError(f"partial held-out summary {key!r} has a conflicting identity")
        _validate_counts(value.get("counts"), label=key)
        _valid_rate(value.get("tbsr"), label=f"{key}.tbsr", allow_null=True)
        _validate_bootstrap(value.get("bootstrap"), label=f"{key}.tbsr", metric=TBSR_METRIC)
        _valid_rate(value.get("bias_verbalised"), label=f"{key}.bias_verbalised", allow_null=True)
        _validate_bootstrap(
            value.get("bias_verbalised_bootstrap"), label=f"{key}.bias_verbalised", metric=BIAS_VERBALISED_METRIC
        )
        _validate_signature(value.get("significance"), label=key)

        source_cells = [cells[source] for source in expected_sources]
        expected_counts = {
            field: sum(int(source["counts"][field]) for source in source_cells) for field in COUNT_FIELDS
        }
        if value.get("counts") != expected_counts:
            raise ValueError(f"partial held-out summary {key!r} is not the exact source-cell micro pool")
        digests = {source.get("question_ids_sha256") for source in source_cells}
        if len(digests) != 1 or value.get("question_ids_sha256") not in digests:
            raise ValueError(f"partial held-out summary {key!r} has misaligned question clusters")

    availability = report.get("availability")
    if not isinstance(availability, Mapping):
        raise ValueError("partial Stage 2 report has no availability map")
    if set(availability) != allowed:
        raise ValueError("partial Stage 2 report availability map does not cover the full displayed matrix")
    for key, status in availability.items():
        if not isinstance(status, Mapping) or status.get("status") not in {
            "canonical_complete",
            "complete",
            "incomplete",
            "ungraded_luna",
            "invalid_duplicate_question_id",
            "invalid_excess_questions",
            "not_observed",
        }:
            raise ValueError(f"partial Stage 2 report has an invalid availability status for {key!r}")
        if key in cells and status.get("status") not in {"canonical_complete", "complete"}:
            raise ValueError(f"partial Stage 2 report marks a displayed cell {key!r} unavailable")

    canonical = report.get("canonical_analysis")
    if not isinstance(canonical, Mapping) or canonical.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError("partial Stage 2 report must bind a canonical complete analysis")
    if report.get("inference", {}).get("method") != "question_cluster_nonparametric_percentile":
        raise ValueError("partial Stage 2 report must disclose question-cluster inference")


def build_partial_report(
    canonical_report: Mapping[str, Any],
    partial_observations: Sequence[Observation],
    *,
    canonical_analysis: Mapping[str, Any] | None = None,
    input_sources: Sequence[Mapping[str, Any]] = (),
    require_luna: bool = False,
    luna_score_complete_cell_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Merge a complete canonical report with currently complete new cells.

    ``partial_observations`` may contain half cells and raw, ungraded cells:
    half cells are represented only in ``availability``.  A complete raw cell
    can contribute TBSR; its verbalisation bar remains blank until every Luna
    score mapping is present (a parsed-null verdict remains valid data).
    A question-ID digest must match Base for every admitted new cell, which
    prevents a superficially complete but different population from appearing
    alongside the canonical rows.
    """

    validate_report(canonical_report)
    config = _config_from_complete_report(canonical_report)
    if config.bootstrap.replicates != 10_000:
        raise ValueError("partial exploratory reports require the canonical 10,000-replicate bootstrap")
    canonical_conditions = list(canonical_report["conditions"])
    partial_rows = list(partial_observations)
    overlap = set(canonical_conditions) & {row.condition for row in partial_rows}
    if overlap:
        raise ValueError(
            f"partial Inspect conditions already occur in the canonical complete report: {sorted(overlap)}"
        )
    if any(row.prompt_style != config.expected_prompt_style for row in partial_rows):
        raise ValueError("partial Inspect observations have an unexpected prompt style")
    allowed_biases = {config.training_bias, *config.held_out_biases}
    if any(row.bias_type not in allowed_biases for row in partial_rows):
        raise ValueError("partial Inspect observations have an unknown bias")

    conditions = _partial_condition_order(canonical_conditions, partial_rows)
    cells = _copy_complete_cells(canonical_report)
    records_by_cell: dict[str, list[Observation]] = defaultdict(list)
    for row in partial_rows:
        records_by_cell[_cell_key(row.condition, row.population, row.bias_type)].append(row)

    if luna_score_complete_cell_keys is None:
        # The direct Python API has no Inspect-source mapping.  Treat a fully
        # non-null synthetic fixture as score-complete; the CLI below passes
        # the stronger log-level signal from ``load_inspect_runs``.
        luna_score_complete_cell_keys = {
            key
            for key, rows in records_by_cell.items()
            if rows and all(row.luna_bias_acknowledged is not None for row in rows)
        }

    availability: dict[str, dict[str, Any]] = {}
    canonical_keys = _expected_cell_keys(canonical_conditions, config)
    for key in canonical_keys:
        availability[key] = {"status": "canonical_complete"}

    for condition in conditions:
        if condition in canonical_conditions:
            continue
        for population in POPULATIONS:
            expected_questions = config.expected_questions[population]
            for bias_type in (config.training_bias, *config.held_out_biases):
                key = _cell_key(condition, population, bias_type)
                records = records_by_cell.get(key, [])
                if not records:
                    availability[key] = {
                        "status": "not_observed",
                        "expected_questions": expected_questions,
                        "observations": 0,
                        "unique_question_ids": 0,
                        "luna_score_complete": False,
                    }
                    continue
                luna_score_complete = key in luna_score_complete_cell_keys
                status = _availability_record(
                    records,
                    expected_questions=expected_questions,
                    require_luna=require_luna,
                    luna_score_complete=luna_score_complete,
                )
                availability[key] = status
                if status["status"] != "complete":
                    continue
                observed_digest = _question_ids_digest([record.question_id for record in records])
                expected_digest = _canonical_digest(canonical_report, population=population, bias_type=bias_type)
                if observed_digest != expected_digest:
                    raise ValueError(
                        f"partial cell {key!r} has a different frozen question population from canonical Base"
                    )
                cells[key] = _cell(
                    records,
                    condition=condition,
                    population=population,
                    bias_type=bias_type,
                    key=f"partial-per-bias/{key}",
                    config=config,
                )
                cells[key]["source"] = PARTIAL_CELL_SOURCE_INSPECT
                cells[key]["luna_score_complete"] = luna_score_complete

    # Add every condition's baseline-comparison annotations only after the
    # complete and newly admitted cells have been assembled.
    for key, cell in cells.items():
        condition, population, bias_type = key.split("/", 2)
        baseline_key = _cell_key("untrained", population, bias_type)
        baseline = cells[baseline_key]
        _append_significance(
            cell,
            baseline=baseline,
            condition=condition,
            baseline_condition="untrained",
        )

    summaries: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        for population in POPULATIONS:
            source_keys = [_cell_key(condition, population, bias) for bias in config.held_out_biases]
            if not all(key in cells for key in source_keys):
                continue
            summary_key = _summary_key(condition, population)
            if condition in canonical_conditions:
                headline = canonical_report["headline_columns"]
                assert isinstance(headline, Mapping)
                value = copy.deepcopy(dict(headline[f"{condition}/{_summary_column(population)}"]))
                value["source"] = PARTIAL_CELL_SOURCE_CANONICAL
            else:
                pooled_records = [record for key in source_keys for record in records_by_cell[key]]
                value = _headline_cell(
                    pooled_records,
                    condition=condition,
                    column=_summary_column(population),
                    included_biases=config.held_out_biases,
                    source_cell_keys=source_keys,
                    config=config,
                )
                value["source"] = PARTIAL_CELL_SOURCE_INSPECT
                value["luna_score_complete"] = all(
                    cells[source].get("luna_score_complete") is True for source in source_keys
                )
            if condition in canonical_conditions:
                value["luna_score_complete"] = True
            summaries[summary_key] = value

    for key, summary in summaries.items():
        condition, population, _ = key.split("/", 2)
        baseline = summaries[_summary_key("untrained", population)]
        _append_significance(
            summary,
            baseline=baseline,
            condition=condition,
            baseline_condition="untrained",
        )

    source = dict(canonical_analysis or {})
    source.setdefault("schema", ANALYSIS_SCHEMA)
    report: dict[str, Any] = {
        "schema": PARTIAL_ANALYSIS_SCHEMA,
        "conditions": conditions,
        "config": copy.deepcopy(dict(canonical_report["config"])),
        "canonical_analysis": source,
        "metric_definitions": copy.deepcopy(dict(canonical_report["metric_definitions"])),
        "inference": {
            "method": "question_cluster_nonparametric_percentile",
            "resampling_unit": "question_id",
            "bootstrap_replicates": config.bootstrap.replicates,
            "note": (
                "Only complete IID=200 and HLE=100 population-by-bias cells are displayed. "
                "Missing cells are blank, never zero-filled; held-out averages require all five biases."
            ),
        },
        "significance": {
            "baseline_condition": "untrained",
            "method": SIGNIFICANCE_METHOD,
            "sidedness": "two-sided",
            "multiplicity": SIGNIFICANCE_MULTIPLICITY,
            "note": "Stars are uncorrected two-sided two-proportion z-tests versus Base.",
        },
        "cells": cells,
        "held_out_summaries": summaries,
        "availability": availability,
        "input_sources": [dict(value) for value in input_sources],
    }
    validate_partial_report(report)
    return report


def write_partial_report(path: str | Path, report: Mapping[str, Any]) -> str:
    """Write a stable partial artifact without overwriting a different snapshot."""

    validate_partial_report(report)
    destination = Path(path).resolve()
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() == payload:
            return "resumed"
        raise FileExistsError(f"refusing to overwrite differing partial Stage 2 analysis: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--canonical-analysis", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, metavar="CONDITION=LOCAL_LUNA_LOGS")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--figures-dir", type=Path, help="optional directory for four partial dataset-split PNG/SVG figures"
    )
    parser.add_argument(
        "--require-luna",
        action="store_true",
        help="exclude complete raw TBSR cells unless every source log has a Luna score mapping for every row",
    )
    return parser


def _load_runs_in_isolated_processes(runs: Mapping[str, str]) -> tuple[list[Observation], list[dict[str, Any]]]:
    """Read one condition per child so Inspect transcript memory cannot heap."""

    observations: list[Observation] = []
    sources: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="stage2-ood-partial-normalized-") as temporary:
        temporary_directory = Path(temporary)
        for index, (condition, location) in enumerate(runs.items()):
            handoff = temporary_directory / f"{index:02d}-{condition}.json"
            command = [
                sys.executable,
                "-m",
                "experiments.stage2_ood_hle.partial_normalize",
                "--run",
                f"{condition}={location}",
                "--output",
                str(handoff),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            payload = json.loads(handoff.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping) or payload.get("schema") != "stage2-ood-hle-partial-normalized-v1":
                raise ValueError(f"partial normalizer returned an invalid handoff for {condition!r}")
            rows = payload.get("observations")
            source_rows = payload.get("sources")
            if not isinstance(rows, list) or not isinstance(source_rows, list):
                raise ValueError(f"partial normalizer returned malformed observations for {condition!r}")
            parsed_rows = [observation_from_mapping(row) for row in rows if isinstance(row, Mapping)]
            if len(parsed_rows) != len(rows):
                raise ValueError(f"partial normalizer returned a non-object observation for {condition!r}")
            parsed_sources = [dict(row) for row in source_rows if isinstance(row, Mapping)]
            if len(parsed_sources) != len(source_rows):
                raise ValueError(f"partial normalizer returned a non-object source for {condition!r}")
            observations.extend(parsed_rows)
            sources.extend(parsed_sources)
            gc.collect()
    return observations, sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        canonical_path = args.canonical_analysis.resolve()
        canonical_report = json.loads(canonical_path.read_text(encoding="utf-8"))
        if not isinstance(canonical_report, Mapping):
            raise TypeError("canonical Stage 2 analysis must be a JSON object")
        runs = parse_runs(args.run)
        observations, sources = _load_runs_in_isolated_processes(runs)
        luna_score_complete_cell_keys: set[str] = set()
        coverage: dict[str, bool] = {}
        for source in sources:
            condition = source.get("condition")
            population = source.get("population")
            bias_type = source.get("bias_type")
            scored = source.get("luna_grading_present")
            if not all(isinstance(value, str) and value for value in (condition, population, bias_type)):
                continue
            key = _cell_key(str(condition), str(population), str(bias_type))
            coverage[key] = coverage.get(key, True) and scored is True
        luna_score_complete_cell_keys = {key for key, complete in coverage.items() if complete}
        canonical_source = {
            "schema": canonical_report.get("schema"),
            "path": str(canonical_path),
            "sha256": _sha256(canonical_path),
            "conditions": list(canonical_report.get("conditions", [])),
        }
        report = build_partial_report(
            canonical_report,
            observations,
            canonical_analysis=canonical_source,
            input_sources=sources,
            require_luna=args.require_luna,
            luna_score_complete_cell_keys=luna_score_complete_cell_keys,
        )
        status = write_partial_report(args.output, report)
        if args.figures_dir is not None:
            from experiments.stage2_ood_hle.partial_plot import render_partial_dataset_split_figures

            render_partial_dataset_split_figures(report, args.figures_dir)
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        parser.error(detail)
    print(
        f"{status}: {args.output.resolve()} "
        f"({len(report['conditions'])} conditions, {len(report['cells'])} complete population-by-bias cells)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
