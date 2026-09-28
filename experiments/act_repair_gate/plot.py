"""Render the ACT repair gate TBSR comparison with the publication plotter.

The inputs are the immutable raw paired-switch reports produced by
``experiments.stage1_iid_diagnostic.gate_analysis``: one for the untrained
base model and one for the repaired ACT checkpoint.  This module deliberately
does not read Inspect logs or call a model-based grader.  It only adapts the
already-counted conditional TBSR estimates to the shared main-figure renderer.

For example::

    python -m experiments.act_repair_gate.plot \
      --untrained-report artifacts/act-repair-gate/untrained-gate.json \
      --act-report artifacts/act-repair-gate/act-gate.json \
      --output-dir artifacts/act-repair-gate/figures

The x-axis is split-major: training-domain samples appear on the left and
held-out in-domain samples on the right.  Filled bars are the correctly
canonicalized prompts; outlined bars are the native-prompt comparator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.stage1_iid_diagnostic.gate_analysis import ANALYSIS_SCHEMA as RAW_GATE_SCHEMA

MERGED_SCHEMA = "act-repair-gate-plot-input-v1"
CONDITIONS = ("untrained", "act")
# Keep this aligned with the shared renderer's within-method order: a filled
# canonical result first, followed by its outlined native-prompt comparator.
# This makes the legend read in the same order as the bars.
VARIANTS = ("canonical", "native")
SPLITS = ("train_eval", "heldout_in_domain")
DATASETS = ("logiqa", "hellaswag")
MODEL = "Qwen/Qwen3.5-9B"
BIAS_IDENTITIES = {
    "train_eval": "training_domain",
    "heldout_in_domain": "held_out_in_domain",
}
BIAS_LABELS = {
    "training_domain": "Training-domain samples",
    "held_out_in_domain": "Held-out in-domain samples",
}
CONDITION_IDENTITIES = {
    ("untrained", "canonical"): ("untrained-canonical", "none", False, "Base (canonical)"),
    ("untrained", "native"): ("untrained-native", "none", True, "Base (native)"),
    ("act", "canonical"): ("act-canonical", "act", False, "ACT (canonical)"),
    ("act", "native"): ("act-native", "act", True, "ACT (native)"),
}
CONDITION_ORDER = tuple(identity[0] for identity in CONDITION_IDENTITIES.values())
FIGURE_STEM = "towards-bias-switch"

_MAIN_CHART_ROOT = Path(__file__).resolve().parents[1] / "rmct_paper_vast_more_methods"


def _rate(cell: Mapping[str, Any], *, condition: str, prompt_variant: str, split: str) -> tuple[int, int, float]:
    if cell.get("condition") != condition or cell.get("prompt_variant") != prompt_variant or cell.get("split") != split:
        raise ValueError(f"raw gate cell identity conflicts for {condition}/{prompt_variant}/{split}")
    pooled = cell.get("pooled")
    rates = pooled.get("rates") if isinstance(pooled, Mapping) else None
    value = rates.get("tbsr") if isinstance(rates, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"raw gate cell lacks pooled TBSR for {condition}/{prompt_variant}/{split}")
    numerator, denominator, rate = value.get("numerator"), value.get("denominator"), value.get("rate")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator <= 0
        or numerator < 0
        or numerator > denominator
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
    ):
        raise ValueError(f"raw gate TBSR is invalid for {condition}/{prompt_variant}/{split}")
    expected = numerator / denominator
    if not math.isclose(float(rate), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"raw gate TBSR is inconsistent for {condition}/{prompt_variant}/{split}: {rate!r} != {numerator}/{denominator}"
        )
    return numerator, denominator, expected


def _sample_count(cell: Mapping[str, Any], *, condition: str, prompt_variant: str, split: str) -> int:
    pooled = cell.get("pooled")
    counts = pooled.get("counts") if isinstance(pooled, Mapping) else None
    samples = counts.get("samples") if isinstance(counts, Mapping) else None
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError(f"raw gate sample count is invalid for {condition}/{prompt_variant}/{split}")
    return samples


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _source_index(report: Mapping[str, Any], *, condition: str) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    sources = report.get("sources")
    if not isinstance(sources, list):
        raise ValueError(f"{condition} raw gate report sources must be an array")
    found: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for source in sources:
        if not isinstance(source, Mapping):
            raise ValueError(f"{condition} raw gate report contains a non-object source")
        key = (str(source.get("prompt_variant", "")), str(source.get("split", "")), str(source.get("dataset", "")))
        if key in found:
            raise ValueError(f"{condition} raw gate report has duplicate source cell {key!r}")
        question_ids_sha256 = source.get("question_ids_sha256")
        if not _valid_sha256(question_ids_sha256):
            raise ValueError(f"{condition} raw gate source has no valid question_ids_sha256 for {key!r}")
        found[key] = source
    expected = {(variant, split, dataset) for variant in VARIANTS for split in SPLITS for dataset in DATASETS}
    if set(found) != expected:
        raise ValueError(
            f"{condition} raw gate report has incomplete source cells; "
            f"missing={sorted(expected - set(found))}, unexpected={sorted(set(found) - expected)}"
        )
    return found


def validate_raw_gate_report(report: Mapping[str, Any], *, condition: str) -> None:
    """Fail closed unless one report is a complete immutable raw gate matrix."""

    if condition not in CONDITIONS:
        raise ValueError(f"unsupported ACT repair condition: {condition!r}")
    if report.get("schema") != RAW_GATE_SCHEMA:
        raise ValueError(f"raw gate schema must be {RAW_GATE_SCHEMA!r}")
    if report.get("condition") != condition:
        raise ValueError(f"raw gate report condition must be {condition!r}")
    if report.get("analysis_mode") != "raw_local_paired_switch_scores_only":
        raise ValueError("ACT repair plot accepts only raw local paired-switch gate reports")
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ValueError("raw gate report cells must be an object")
    expected = {f"{variant}/{split}" for variant in VARIANTS for split in SPLITS}
    if set(cells) != expected:
        raise ValueError(
            f"{condition} raw gate report has incomplete cells; "
            f"missing={sorted(expected - set(cells))}, unexpected={sorted(set(cells) - expected)}"
        )
    for prompt_variant in VARIANTS:
        for split in SPLITS:
            cell = cells[f"{prompt_variant}/{split}"]
            if not isinstance(cell, Mapping):
                raise ValueError(f"raw gate cell is not an object: {condition}/{prompt_variant}/{split}")
            _, denominator, _ = _rate(cell, condition=condition, prompt_variant=prompt_variant, split=split)
            samples = _sample_count(cell, condition=condition, prompt_variant=prompt_variant, split=split)
            if denominator > samples:
                raise ValueError(f"raw gate TBSR denominator exceeds samples for {condition}/{prompt_variant}/{split}")
    _source_index(report, condition=condition)


def merge_reports(untrained_report: Mapping[str, Any], act_report: Mapping[str, Any]) -> dict[str, Any]:
    """Merge two pinned raw reports after proving their evaluated populations match."""

    reports = {"untrained": untrained_report, "act": act_report}
    for condition, report in reports.items():
        validate_raw_gate_report(report, condition=condition)
    source_indexes = {condition: _source_index(report, condition=condition) for condition, report in reports.items()}
    for key in sorted(source_indexes["untrained"]):
        untrained_source = source_indexes["untrained"][key]
        act_source = source_indexes["act"][key]
        if untrained_source["question_ids_sha256"] != act_source["question_ids_sha256"]:
            raise ValueError(f"untrained/ACT question populations differ for {key!r}")
        if untrained_source.get("source_identity_digest") != act_source.get("source_identity_digest"):
            raise ValueError(f"untrained/ACT source identity differs for {key!r}")
    return {
        "schema": MERGED_SCHEMA,
        "analysis_mode": "raw_local_paired_switch_scores_only",
        "conditions": reports,
    }


def chart_rows(merged: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Adapt gate report counts to chart rows for the shared main renderer."""

    if merged.get("schema") != MERGED_SCHEMA:
        raise ValueError(f"merged report schema must be {MERGED_SCHEMA!r}")
    conditions = merged.get("conditions")
    if not isinstance(conditions, Mapping) or set(conditions) != set(CONDITIONS):
        raise ValueError("merged report must contain exactly untrained and act conditions")
    rows: list[dict[str, Any]] = []
    # Split-major ordering deliberately places training-domain samples on the
    # left and held-out in-domain samples on the right.
    for split in SPLITS:
        for condition in CONDITIONS:
            report = conditions[condition]
            if not isinstance(report, Mapping):
                raise ValueError(f"merged {condition} report is not an object")
            cells = report.get("cells")
            if not isinstance(cells, Mapping):
                raise ValueError(f"merged {condition} report cells are not an object")
            for prompt_variant in VARIANTS:
                cell = cells[f"{prompt_variant}/{split}"]
                if not isinstance(cell, Mapping):
                    raise ValueError(f"merged cell is not an object: {condition}/{prompt_variant}/{split}")
                numerator, denominator, mean = _rate(
                    cell,
                    condition=condition,
                    prompt_variant=prompt_variant,
                    split=split,
                )
                canonical_condition, method, is_control, label = CONDITION_IDENTITIES[(condition, prompt_variant)]
                rows.append(
                    {
                        "condition": canonical_condition,
                        "condition_label": label,
                        "method": method,
                        "is_control": is_control,
                        "bias_type": BIAS_IDENTITIES[split],
                        "metric": "towards_bias_switch",
                        "mean": mean,
                        "stderr": math.sqrt(mean * (1.0 - mean) / (denominator - 1)) if denominator > 1 else 0.0,
                        "n_scored": denominator,
                        "n_total": _sample_count(
                            cell,
                            condition=condition,
                            prompt_variant=prompt_variant,
                            split=split,
                        ),
                        "numerator": numerator,
                        "model": MODEL,
                        "training_biases": ["training_domain"],
                        "split": split,
                        "prompt_variant": prompt_variant,
                        "significance": "",
                    }
                )
    return rows


def chart_spec() -> dict[str, Any]:
    """Load the shared main TBSR recipe and declare the gate's two x-axis groups."""

    recipe = json.loads((_MAIN_CHART_ROOT / "switch_chart.json").read_text())
    return {
        **recipe,
        "metric": "towards_bias_switch",
        "condition_order": list(CONDITION_ORDER),
        "bias_order": ["training_domain", "held_out_in_domain"],
        "bias_labels": BIAS_LABELS,
        "model_order": [MODEL],
        "model_labels": {MODEL: MODEL},
        "held_out_label": "held_out_in_domain",
        "show_significance": False,
        "legend_columns": 4,
        "theme": {"rcparams": {"svg.hashsalt": "act-repair-gate-main-style-v1"}},
    }


@contextmanager
def _deterministic_rendering():
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous


def render_figures(merged: Mapping[str, Any], output_dir: str | Path) -> str:
    """Render PNG/SVG TBSR figures atomically through the shared plot pipeline."""

    rows = chart_rows(merged)
    spec = chart_spec()
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        candidates: dict[Path, Path] = {}
        with _deterministic_rendering():
            for extension in ("png", "svg"):
                candidate = staging / f"{FIGURE_STEM}.{extension}"
                render_publication_plot(rows, spec, candidate)
                candidates[directory / candidate.name] = candidate
        for destination, candidate in candidates.items():
            if destination.exists() and destination.read_bytes() != candidate.read_bytes():
                raise FileExistsError(f"refusing to overwrite differing ACT repair figure: {destination}")
        directory.mkdir(parents=True, exist_ok=True)
        wrote = False
        for destination, candidate in candidates.items():
            if destination.exists():
                continue
            try:
                os.link(candidate, destination)
                wrote = True
            except FileExistsError:
                if destination.read_bytes() != candidate.read_bytes():
                    raise FileExistsError(f"ACT repair figure appeared and differs: {destination}")
        return "written" if wrote else "resumed"


def _read_report(path: Path) -> Mapping[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, Mapping):
        raise TypeError(f"raw gate report must be a JSON object: {path}")
    return document


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--untrained-report", required=True, type=Path)
    parser.add_argument("--act-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        merged = merge_reports(_read_report(args.untrained_report), _read_report(args.act_report))
        status = render_figures(merged, args.output_dir)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output_dir.resolve()} (2 files)")


if __name__ == "__main__":
    main()
