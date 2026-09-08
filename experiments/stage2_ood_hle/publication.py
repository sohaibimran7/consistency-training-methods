"""Adapt a complete Stage 2 OOD report to CTM's standard publication plot.

This module is deliberately a chart-row/spec adapter. Statistical estimates,
asymmetric question-cluster confidence intervals, and Holm-adjusted markers
are all produced by :mod:`experiments.stage2_ood_hle.analyze`; visual drawing
is delegated to :func:`ctm_data.adapters.mcq_bias.plot.render_publication_plot`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from ctm_data.adapters.mcq_bias.plot_registry import load_presentation_registry, registry_labels
from experiments.stage2_ood_hle.analyze import (
    BIAS_VERBALISED_METRIC,
    HLE_POPULATION,
    IID_POPULATION,
    TBSR_METRIC,
    validate_report,
)


HELD_OUT_MEAN = "held_out_mean"
FIGURE_METRICS = {
    TBSR_METRIC: {
        "stem": "ood-tbsr",
        "rate_key": "tbsr",
        "bootstrap_key": "bootstrap",
        "ylabel": "Towards-bias switch rate",
    },
    BIAS_VERBALISED_METRIC: {
        "stem": "ood-bias-verbalised",
        "rate_key": "bias_verbalised",
        "bootstrap_key": "bias_verbalised_bootstrap",
        "ylabel": "Bias verbalised (Luna YES)",
    },
}
POPULATION_PANELS = {
    IID_POPULATION: {
        "suffix": "iid-by-bias",
        "headline_column": "held_out_bias",
        "title": "IID held-out / training-domain dataset",
    },
    HLE_POPULATION: {
        "suffix": "hle-by-bias",
        "headline_column": "held_out_dataset_and_bias",
        "title": "HLE / held-out dataset",
    },
}
CONDITION_METADATA = {
    "untrained": {"method": "none", "is_control": False, "label": "Base"},
    "bct": {"method": "bias_augmented_consistency", "is_control": False, "label": "BCT"},
    "bct-control": {"method": "bias_augmented_consistency", "is_control": True, "label": "BCT Control"},
    "opct": {"method": "opct", "is_control": False, "label": "OPCT"},
    "rmct": {"method": "rate_matching", "is_control": False, "label": "RMCT"},
    "rmct-control": {"method": "rate_matching", "is_control": True, "label": "RMCT Control"},
    "act": {"method": "act", "is_control": False, "label": "ACT"},
    "attct": {"method": "attct", "is_control": False, "label": "AttCT"},
    "mlpct": {"method": "mlpct", "is_control": False, "label": "MLPCT"},
}


def _metric_cell_fields(metric: str) -> tuple[str, str]:
    specification = FIGURE_METRICS.get(metric)
    if specification is None:
        raise ValueError(f"unsupported Stage 2 publication metric {metric!r}")
    return str(specification["rate_key"]), str(specification["bootstrap_key"])


def _categories(report: Mapping[str, Any]) -> tuple[str, ...]:
    config = report["config"]
    assert isinstance(config, Mapping)  # validated by validate_report
    training_bias = config["training_bias"]
    held_out_biases = config["held_out_biases"]
    assert isinstance(training_bias, str) and isinstance(held_out_biases, list)
    return (training_bias, *(str(bias) for bias in held_out_biases), HELD_OUT_MEAN)


def _cell(
    report: Mapping[str, Any],
    *,
    condition: str,
    population: str,
    category: str,
) -> Mapping[str, Any]:
    if category == HELD_OUT_MEAN:
        panel = POPULATION_PANELS[population]
        cells = report["headline_columns"]
        assert isinstance(cells, Mapping)
        value = cells[f"{condition}/{panel['headline_column']}"]
    else:
        cells = report["per_bias_cells"]
        assert isinstance(cells, Mapping)
        value = cells[f"{condition}/{population}/{category}"]
    assert isinstance(value, Mapping)
    return value


def _significance_note(report: Mapping[str, Any]) -> str:
    value = report.get("significance")
    if not isinstance(value, Mapping):
        raise ValueError("standard Stage 2 publication figures require paired Base comparison annotations")
    permutations = value.get("permutations")
    if not isinstance(permutations, Mapping) or not isinstance(permutations.get("permutations"), int):
        raise ValueError("Stage 2 paired comparison annotations have no permutation count")
    family_size = value.get("family_size")
    if not isinstance(family_size, int):
        raise ValueError("Stage 2 paired comparison annotations have no Holm family size")
    return (
        "Stars: * Holm-adjusted p<0.05, ** p<0.01, *** p<0.001.\n"
        "Two-sided paired whole-question label-swap randomization test vs Base; "
        f"Holm-adjusted across {family_size} conditions per cell/metric "
        f"({permutations['permutations']:,} permutations)."
    )


def chart_rows(
    report: Mapping[str, Any],
    *,
    population: str,
    metric: str,
) -> list[dict[str, Any]]:
    """Convert one IID/HLE metric panel into standard chart-ready rows."""

    validate_report(report)
    if population not in POPULATION_PANELS:
        raise ValueError(f"unsupported Stage 2 publication population {population!r}")
    rate_key, bootstrap_key = _metric_cell_fields(metric)
    conditions = report["conditions"]
    assert isinstance(conditions, list)
    unknown_conditions = [condition for condition in conditions if condition not in CONDITION_METADATA]
    if unknown_conditions:
        raise ValueError(f"no standard presentation metadata for conditions: {unknown_conditions}")
    categories = _categories(report)
    config = report["config"]
    assert isinstance(config, Mapping)
    training_bias = str(config["training_bias"])
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        presentation = CONDITION_METADATA[condition]
        for category in categories:
            cell = _cell(report, condition=condition, population=population, category=category)
            rate = cell[rate_key]
            bootstrap = cell[bootstrap_key]
            significance = cell.get("significance")
            assert isinstance(rate, Mapping) and isinstance(bootstrap, Mapping)
            if not isinstance(significance, Mapping) or not isinstance(significance.get(metric), Mapping):
                raise ValueError(f"{condition}/{population}/{category} has no paired significance annotation")
            estimate = rate.get("rate")
            ci = bootstrap.get("ci_95")
            if estimate is None or not isinstance(ci, Mapping) or ci.get("lower") is None or ci.get("upper") is None:
                raise ValueError(f"{condition}/{population}/{category} has no renderable {metric} interval")
            annotation = significance[metric]
            marker = annotation.get("marker")
            if marker not in {"", "*", "**", "***"}:
                raise ValueError(f"{condition}/{population}/{category} has an invalid Holm marker")
            rows.append(
                {
                    "condition": condition,
                    "condition_label": presentation["label"],
                    "method": presentation["method"],
                    "is_control": presentation["is_control"],
                    "bias_type": category,
                    "metric": metric,
                    "mean": float(estimate),
                    # Kept for the generic renderer's stable row contract;
                    # ci_lower/ci_upper below take precedence for drawing.
                    "stderr": float(bootstrap["standard_error"]),
                    "ci_lower": float(ci["lower"]),
                    "ci_upper": float(ci["upper"]),
                    "n_scored": int(rate["denominator"]),
                    "model": "stage2-qwen35-9b",
                    "model_label": "Qwen 3.5 9B",
                    "training_biases": [training_bias],
                    "population_label": population,
                    "significance": marker,
                    "p_value_holm": annotation.get("p_value_holm"),
                }
            )
    return rows


def publication_spec(report: Mapping[str, Any], *, population: str, metric: str) -> dict[str, Any]:
    """Return the declarative standard-renderer recipe for one Stage 2 panel."""

    categories = _categories(report)
    panel = POPULATION_PANELS[population]
    labels = registry_labels(load_presentation_registry().biases)
    training_bias = categories[0]
    bias_labels = {
        **labels,
        training_bias: f"{labels.get(training_bias, training_bias.replace('_', ' ').title())}\n(trained bias)",
        HELD_OUT_MEAN: "Held-out avg.\n(micro pool)",
    }
    return {
        "metric": metric,
        "facet": "population_label",
        "facet_labels": {"population_label": {population: str(panel["title"])}},
        "condition_order": list(report["conditions"]),
        "condition_labels": {condition: metadata["label"] for condition, metadata in CONDITION_METADATA.items()},
        "method_colors": {"opct": "#62a9a4"},
        "condition_styles": {
            "bct-control": {"color": "#7aa7d9"},
            "rmct-control": {"color": "#8cc39a"},
        },
        "bias_order": list(categories),
        "bias_labels": bias_labels,
        "held_out_label": HELD_OUT_MEAN,
        "ylabel": str(FIGURE_METRICS[metric]["ylabel"]),
        "percent": True,
        "show_significance": True,
        "significance_note": _significance_note(report),
        "legend_columns": 5,
        "theme": {
            "figure_width_min": 10.4,
            "figure_width_per_bias": 1.28,
            "figure_width_intercept": 2.3,
            "figure_height_per_row": 4.55,
            "figure_height_intercept": 0.25,
            "annotation_fontsize": 7.0,
            "tick_fontsize": 7.0,
        },
    }


def render_standard_figures(report: Mapping[str, Any], output_dir: str | Path) -> str:
    """Render the four requested Stage 2 panels through the standard renderer."""

    validate_report(report)
    _significance_note(report)
    directory = Path(output_dir).resolve()
    expected = {
        f"{FIGURE_METRICS[metric]['stem']}-{POPULATION_PANELS[population]['suffix']}.{extension}"
        for metric in FIGURE_METRICS
        for population in POPULATION_PANELS
        for extension in ("png", "svg")
    }
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing Stage 2 standard publication figures: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        for metric, metric_specification in FIGURE_METRICS.items():
            for population, panel in POPULATION_PANELS.items():
                rows = chart_rows(report, population=population, metric=metric)
                spec = publication_spec(report, population=population, metric=metric)
                stem = f"{metric_specification['stem']}-{panel['suffix']}"
                for extension in ("png", "svg"):
                    render_publication_plot(rows, spec, staging / f"{stem}.{extension}")
        generated = {path.name for path in staging.iterdir()}
        if generated != expected:
            raise RuntimeError("standard Stage 2 adapter did not generate the complete expected figure set")
        directory.mkdir(parents=True, exist_ok=True)
        for candidate in staging.iterdir():
            os.link(candidate, directory / candidate.name)
    return "written"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = json.loads(args.analysis.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise TypeError("Stage 2 OOD analysis must be an object")
        status = render_standard_figures(report, args.output_dir)
    except (FileExistsError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _parser().error(str(exc))
    print(f"{status}: {args.output_dir.resolve()} (8 files)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
