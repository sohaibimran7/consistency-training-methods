"""Render isolated Stage 2 IID/HLE OOD figures.

This renderer consumes only a completed :mod:`.analyze` report.  It never
opens an Inspect log, derives an estimate, or touches the historical Stage-1
two-column figure code.  It renders both TBSR and Luna explicit-bias
acknowledgement. Error bars are question-cluster 95% bootstrap intervals, not
binomial standard errors.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ctm_data.adapters.mcq_bias.plot_registry import load_presentation_registry, registry_labels
from experiments.stage2_ood_hle.analyze import (
    ANALYSIS_SCHEMA,
    BIAS_VERBALISED_METRIC,
    HEADLINE_COLUMNS,
    HEADLINE_COLUMN_METADATA,
    HLE_POPULATION,
    IID_POPULATION,
    TBSR_METRIC,
    validate_report,
)


FIGURES = {
    TBSR_METRIC: {
        "stem": "ood-tbsr-four-column",
        "rate_key": "tbsr",
        "bootstrap_key": "bootstrap",
        "x_label": "TBSR; 95% question-cluster bootstrap CI",
        "y_label": "Towards-bias switch rate",
    },
    BIAS_VERBALISED_METRIC: {
        "stem": "ood-bias-verbalised-four-column",
        "rate_key": "bias_verbalised",
        "bootstrap_key": "bias_verbalised_bootstrap",
        "x_label": "Luna bias-verbalisation rate; 95% question-cluster bootstrap CI",
        "y_label": "Bias verbalised (Luna YES)",
    },
}
CONDITION_LABELS = {
    "untrained": "Base",
    "bct": "BCT",
    "bct-control": "BCT Control",
    "rmct": "RMCT",
    "rmct-control": "RMCT Control",
    "act": "ACT",
    "attct": "AttCT",
    "mlpct": "MLPCT",
    "opct": "OPCT",
    "rmct-vllm": "RMCT (vLLM)",
}
METHOD_COLORS = {
    "untrained": "#9aa0a6",
    "bct": "#7aa7d9",
    "bct-control": "#7aa7d9",
    "rmct": "#8cc39a",
    "rmct-control": "#8cc39a",
    "act": "#d99a6c",
    "attct": "#9e9ac8",
    "mlpct": "#d58aaa",
    "opct": "#62a9a4",
    "rmct-vllm": "#8cc39a",
}
CONTROL_CONDITIONS = frozenset({"bct-control", "rmct-control"})
FALLBACK_COLORS = ("#4f81bd", "#c0504d", "#8064a2", "#4bacc6", "#9bbb59", "#f79646")
HELD_OUT_MEAN = "held_out_mean"

# The existing mcq-bias publication renderer uses these same two background
# roles. Stage 2 retains its own renderer because its percentile bootstrap
# intervals can be asymmetric; silently coercing them to ±2 stderr would
# change the reported uncertainty.
TRAINED_BACKGROUND = "#fbf6ea"
HELD_OUT_BACKGROUND = "#f1f1ee"
SEPARATOR_COLOR = "#d8dadf"
DATASET_SPLIT_POPULATIONS = (
    {
        "population": IID_POPULATION,
        "suffix": "iid-by-bias",
        "title": "IID held-out / training-domain dataset",
        "subtitle": (
            "Training bias at left; held-out biases evaluated separately; "
            "final bar is the micro-pooled held-out result."
        ),
        "summary_column": "held_out_bias",
    },
    {
        "population": HLE_POPULATION,
        "suffix": "hle-by-bias",
        "title": "HLE / held-out dataset",
        "subtitle": (
            "Training bias at left; held-out biases evaluated separately; "
            "final bar is the micro-pooled held-out result."
        ),
        "summary_column": "held_out_dataset_and_bias",
    },
)


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


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _color(condition: str, index: int) -> str:
    return METHOD_COLORS.get(condition, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _headline_cell(report: Mapping[str, Any], *, condition: str, column: str) -> Mapping[str, Any]:
    cells = report["headline_columns"]
    assert isinstance(cells, Mapping)  # validated by the public boundary
    cell = cells[f"{condition}/{column}"]
    assert isinstance(cell, Mapping)
    return cell


def _rate_and_interval(
    cell: Mapping[str, Any], *, rate_key: str, bootstrap_key: str
) -> tuple[float, float, float, int, int]:
    rate_value = cell[rate_key]
    bootstrap = cell[bootstrap_key]
    assert isinstance(rate_value, Mapping) and isinstance(bootstrap, Mapping)
    ci = bootstrap["ci_95"]
    assert isinstance(ci, Mapping)
    rate = float(rate_value["rate"])
    lower, upper = float(ci["lower"]), float(ci["upper"])
    numerator, denominator = int(rate_value["numerator"]), int(rate_value["denominator"])
    return rate, lower, upper, numerator, denominator


def _require_renderable_metric(report: Mapping[str, Any], *, metric: str) -> None:
    """Reject an ungraded raw matrix before any figure is published."""

    specification = FIGURES[metric]
    rate_key = str(specification["rate_key"])
    for condition in report["conditions"]:
        for column in HEADLINE_COLUMNS:
            cell = _headline_cell(report, condition=str(condition), column=column)
            rate = cell[rate_key]
            assert isinstance(rate, Mapping)
            if rate.get("rate") is None:
                readable = "Luna bias-verbalisation" if metric == BIAS_VERBALISED_METRIC else metric
                raise ValueError(
                    f"cannot render {readable}: {condition}/{column} has no parsed verdicts; "
                    "run posthoc Luna grading first"
                )


def _render_metric(
    report: Mapping[str, Any],
    *,
    metric: str,
    candidates: Mapping[str, Path],
) -> None:
    """Render one metric into its already-selected temporary destinations."""

    specification = FIGURES[metric]
    conditions = list(report["conditions"])
    with matplotlib.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.edgecolor": "#141518",
            "axes.linewidth": 0.8,
            "grid.color": "#ececee",
            "grid.linewidth": 0.6,
            "svg.hashsalt": "stage2-ood-hle-four-column-v1",
        }
    ):
        figure, axes = plt.subplots(
            1,
            len(HEADLINE_COLUMNS),
            figsize=(max(10.2, 1.38 * len(conditions) + 3.4), 4.4),
            sharey=True,
            layout="constrained",
        )
        if len(HEADLINE_COLUMNS) == 1:  # pragma: no cover - fixed contract, kept robust
            axes = [axes]
        x = np.arange(len(conditions), dtype=float)
        for axis, column in zip(axes, HEADLINE_COLUMNS, strict=True):
            rates: list[float] = []
            low_errors: list[float] = []
            high_errors: list[float] = []
            labels: list[str] = []
            annotations: list[str] = []
            for index, condition in enumerate(conditions):
                cell = _headline_cell(report, condition=condition, column=column)
                rate, lower, upper, numerator, denominator = _rate_and_interval(
                    cell,
                    rate_key=str(specification["rate_key"]),
                    bootstrap_key=str(specification["bootstrap_key"]),
                )
                rates.append(rate)
                low_errors.append(max(0.0, rate - lower))
                high_errors.append(max(0.0, upper - rate))
                labels.append(_condition_label(condition))
                annotations.append(f"{numerator}/{denominator}")
            bars = axis.bar(
                x,
                rates,
                color=[_color(condition, index) for index, condition in enumerate(conditions)],
                edgecolor="#222429",
                linewidth=0.55,
                yerr=np.asarray([low_errors, high_errors]),
                error_kw={"ecolor": "#24262b", "elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7},
            )
            for bar, condition in zip(bars, conditions, strict=True):
                if condition in CONTROL_CONDITIONS:
                    bar.set_hatch("//")
            axis.set_title(str(HEADLINE_COLUMN_METADATA[column]["label"]), fontsize=9, pad=9)
            axis.text(
                0.5,
                1.005,
                str(HEADLINE_COLUMN_METADATA[column]["subtitle"]),
                transform=axis.transAxes,
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#4a4d54",
            )
            axis.set_xticks(x, labels, rotation=48, ha="right", fontsize=7)
            axis.set_ylim(0.0, 1.03)
            axis.set_yticks(np.linspace(0, 1, 6))
            axis.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)], fontsize=7)
            axis.yaxis.grid(True)
            axis.set_axisbelow(True)
            for index, label in enumerate(annotations):
                axis.text(index, 0.012, label, ha="center", va="bottom", rotation=90, fontsize=5.2, color="#32343a")
            axis.set_xlabel(str(specification["x_label"]), fontsize=7, labelpad=4)
        axes[0].set_ylabel(str(specification["y_label"]), fontsize=8)
        legend_handles = [
            Patch(
                facecolor=_color(condition, index),
                edgecolor="#222429",
                hatch="//" if condition in CONTROL_CONDITIONS else "",
                label=_condition_label(condition),
            )
            for index, condition in enumerate(conditions)
        ]
        figure.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncols=min(len(conditions), 5),
            fontsize=7,
            frameon=False,
            title="Condition",
            title_fontsize=7,
        )
        for extension, candidate in candidates.items():
            figure.savefig(candidate, format=extension, metadata={"Date": None})
        plt.close(figure)


def _per_bias_cell(
    report: Mapping[str, Any], *, condition: str, population: str, bias_type: str
) -> Mapping[str, Any]:
    cells = report["per_bias_cells"]
    assert isinstance(cells, Mapping)  # validated by the public boundary
    cell = cells[f"{condition}/{population}/{bias_type}"]
    assert isinstance(cell, Mapping)
    return cell


def _dataset_split_categories(report: Mapping[str, Any]) -> tuple[str, ...]:
    config = report["config"]
    assert isinstance(config, Mapping)  # validated by the public boundary
    training_bias = config["training_bias"]
    held_out_biases = config["held_out_biases"]
    assert isinstance(training_bias, str) and isinstance(held_out_biases, list)
    return (training_bias, *(str(bias) for bias in held_out_biases), HELD_OUT_MEAN)


def _dataset_split_cell(
    report: Mapping[str, Any],
    *,
    condition: str,
    population: str,
    category: str,
    summary_column: str,
) -> Mapping[str, Any]:
    if category == HELD_OUT_MEAN:
        return _headline_cell(report, condition=condition, column=summary_column)
    return _per_bias_cell(report, condition=condition, population=population, bias_type=category)


def _require_renderable_dataset_split_metric(report: Mapping[str, Any], *, metric: str) -> None:
    """Reject a report missing a per-bias metric before rendering any panel."""

    specification = FIGURES[metric]
    rate_key = str(specification["rate_key"])
    categories = _dataset_split_categories(report)
    for condition in report["conditions"]:
        for panel in DATASET_SPLIT_POPULATIONS:
            population = str(panel["population"])
            summary_column = str(panel["summary_column"])
            for category in categories:
                cell = _dataset_split_cell(
                    report,
                    condition=str(condition),
                    population=population,
                    category=category,
                    summary_column=summary_column,
                )
                rate = cell[rate_key]
                assert isinstance(rate, Mapping)
                if rate.get("rate") is None:
                    readable = "Luna bias-verbalisation" if metric == BIAS_VERBALISED_METRIC else metric
                    raise ValueError(
                        f"cannot render {readable}: {condition}/{population}/{category} has no parsed verdicts; "
                        "run posthoc Luna grading first"
                    )


def _bias_labels() -> Mapping[str, str]:
    """Use the standard publication registry rather than a second label table."""

    return registry_labels(load_presentation_registry().biases)


def _dataset_split_stem(*, metric: str, suffix: str) -> str:
    four_column_stem = str(FIGURES[metric]["stem"])
    base = four_column_stem.removesuffix("-four-column")
    return f"{base}-{suffix}"


def _render_dataset_split_metric(
    report: Mapping[str, Any],
    *,
    metric: str,
    panel: Mapping[str, str],
    candidates: Mapping[str, Path],
) -> None:
    """Render one population with individual held-out-bias bars and a micro-pool."""

    specification = FIGURES[metric]
    conditions = [str(condition) for condition in report["conditions"]]
    categories = _dataset_split_categories(report)
    population = str(panel["population"])
    summary_column = str(panel["summary_column"])
    labels = _bias_labels()
    x_labels = [
        (
            f"{labels.get(category, category.replace('_', ' ').title())}\n(trained bias)"
            if index == 0
            else "Held-out avg.\n(micro pool)"
            if category == HELD_OUT_MEAN
            else labels.get(category, category.replace("_", " ").title())
        )
        for index, category in enumerate(categories)
    ]
    n_conditions = len(conditions)
    bar_width = min(0.13, 0.78 / max(1, n_conditions))
    offsets = (np.arange(n_conditions, dtype=float) - (n_conditions - 1) / 2.0) * bar_width
    x = np.arange(len(categories), dtype=float)

    with matplotlib.rc_context(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "font.family": ["DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.edgecolor": "#141518",
            "axes.linewidth": 0.8,
            "grid.color": "#ececee",
            "grid.linewidth": 0.6,
            "svg.hashsalt": "stage2-ood-hle-dataset-split-v1",
        }
    ):
        figure, axis = plt.subplots(
            figsize=(max(10.4, 1.28 * len(categories) + 2.3), 4.85),
            layout="constrained",
        )
        axis.axvspan(-0.5, 0.5, color=TRAINED_BACKGROUND, alpha=0.88, zorder=0)
        axis.axvspan(
            len(categories) - 1.5,
            len(categories) - 0.5,
            color=HELD_OUT_BACKGROUND,
            alpha=0.98,
            zorder=0,
        )
        for separator in range(1, len(categories)):
            axis.axvline(separator - 0.5, color=SEPARATOR_COLOR, linewidth=0.6, zorder=1)

        for condition_index, condition in enumerate(conditions):
            color = _color(condition, condition_index)
            is_control = condition in CONTROL_CONDITIONS
            for category_index, category in enumerate(categories):
                cell = _dataset_split_cell(
                    report,
                    condition=condition,
                    population=population,
                    category=category,
                    summary_column=summary_column,
                )
                rate, lower, upper, _, _ = _rate_and_interval(
                    cell,
                    rate_key=str(specification["rate_key"]),
                    bootstrap_key=str(specification["bootstrap_key"]),
                )
                position = x[category_index] + offsets[condition_index]
                axis.bar(
                    position,
                    rate,
                    bar_width,
                    color="white" if is_control else color,
                    edgecolor=color if is_control else "#222429",
                    linewidth=1.1 if is_control else 0.45,
                    hatch="//" if is_control else "",
                    zorder=2,
                )
                axis.errorbar(
                    position,
                    rate,
                    yerr=np.asarray([[max(0.0, rate - lower)], [max(0.0, upper - rate)]]),
                    fmt="none",
                    ecolor="#24262b",
                    elinewidth=0.7,
                    capsize=1.8,
                    capthick=0.7,
                    zorder=3,
                )

        axis.set_xlim(-0.5, len(categories) - 0.5)
        axis.set_ylim(0.0, 1.03)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)], fontsize=7)
        axis.set_xticks(x, x_labels, rotation=31, ha="right", fontsize=7)
        axis.yaxis.grid(True)
        axis.grid(axis="x", visible=False)
        axis.set_ylabel(str(specification["y_label"]), fontsize=8)
        axis.set_title(str(panel["title"]), loc="left", fontsize=9, fontweight="semibold", pad=17)
        axis.text(
            0.0,
            1.013,
            str(panel["subtitle"]),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.3,
            color="#4a4d54",
        )
        legend_handles = [
            Patch(
                facecolor="white" if condition in CONTROL_CONDITIONS else _color(condition, index),
                edgecolor=_color(condition, index) if condition in CONTROL_CONDITIONS else "#222429",
                linewidth=1.1 if condition in CONTROL_CONDITIONS else 0.45,
                hatch="//" if condition in CONTROL_CONDITIONS else "",
                label=_condition_label(condition),
            )
            for index, condition in enumerate(conditions)
        ]
        figure.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncols=min(len(legend_handles), 6),
            fontsize=7,
            frameon=False,
            title="Condition",
            title_fontsize=7,
        )
        for extension, candidate in candidates.items():
            figure.savefig(candidate, format=extension, metadata={"Date": None})
        plt.close(figure)


def render_dataset_split_figures(report: Mapping[str, Any], output_dir: str | Path) -> str:
    """Publish four standard-style figures split into IID and HLE populations.

    Each metric gets one figure for IID held-out questions and one for HLE.
    Within a figure, the left category is the training bias, the five middle
    categories retain the frozen held-out-bias order, and the right category is
    the exact question-cluster micro-pool rather than an average of bar heights.
    """

    validate_report(report)
    if report.get("schema") != ANALYSIS_SCHEMA:  # defensive; validate_report is the main check
        raise ValueError("unexpected Stage 2 OOD report schema")
    for metric in FIGURES:
        _require_renderable_dataset_split_metric(report, metric=metric)
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        candidates: dict[Path, Path] = {}
        with _deterministic_rendering():
            for metric in FIGURES:
                for panel in DATASET_SPLIT_POPULATIONS:
                    stem = _dataset_split_stem(metric=metric, suffix=str(panel["suffix"]))
                    per_figure = {extension: staging / f"{stem}.{extension}" for extension in ("png", "svg")}
                    _render_dataset_split_metric(
                        report,
                        metric=metric,
                        panel=panel,
                        candidates=per_figure,
                    )
                    candidates.update({directory / path.name: path for path in per_figure.values()})

        for destination, candidate in candidates.items():
            if destination.exists() and destination.read_bytes() != candidate.read_bytes():
                raise FileExistsError(f"refusing to overwrite differing Stage 2 OOD figure: {destination}")
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
                    raise FileExistsError(f"Stage 2 OOD figure appeared and differs: {destination}")
        return "written" if wrote else "resumed"


def render_figures(report: Mapping[str, Any], output_dir: str | Path) -> str:
    """Publish both four-column PNG/SVG figures from an immutable OOD report."""

    validate_report(report)
    if report.get("schema") != ANALYSIS_SCHEMA:  # defensive; validate_report is the main check
        raise ValueError("unexpected Stage 2 OOD report schema")
    for metric in FIGURES:
        _require_renderable_metric(report, metric=metric)
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        candidates: dict[Path, Path] = {}
        with _deterministic_rendering():
            for metric, specification in FIGURES.items():
                per_metric = {
                    extension: staging / f"{specification['stem']}.{extension}"
                    for extension in ("png", "svg")
                }
                _render_metric(report, metric=metric, candidates=per_metric)
                candidates.update({directory / path.name: path for path in per_metric.values()})

        for destination, candidate in candidates.items():
            if destination.exists() and destination.read_bytes() != candidate.read_bytes():
                raise FileExistsError(f"refusing to overwrite differing Stage 2 OOD figure: {destination}")
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
                    raise FileExistsError(f"Stage 2 OOD figure appeared and differs: {destination}")
        return "written" if wrote else "resumed"


def render_four_column(report: Mapping[str, Any], output_dir: str | Path) -> str:
    """Backward-compatible name for rendering the complete four-column set."""

    return render_figures(report, output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--layout",
        choices=("four-column", "dataset-split", "all"),
        default="four-column",
        help="figure family to render; the default preserves the original four-column CLI contract",
    )
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.analysis.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise TypeError("Stage 2 OOD analysis must be a JSON object")
        statuses = []
        if args.layout in {"four-column", "all"}:
            statuses.append(render_figures(report, args.output_dir))
        if args.layout in {"dataset-split", "all"}:
            statuses.append(render_dataset_split_figures(report, args.output_dir))
    except (FileExistsError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    status = "written" if "written" in statuses else "resumed"
    file_count = 4 if args.layout == "four-column" else 8 if args.layout == "dataset-split" else 12
    print(f"{status}: {args.output_dir.resolve()} ({file_count} files)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
