"""Render explicitly partial Stage 2 dataset-split figures.

This is intentionally separate from :mod:`.plot`: the latter remains strict
and can only render an immutable complete analysis report.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from experiments.stage2_ood_hle.analyze import BIAS_VERBALISED_METRIC, HLE_POPULATION, IID_POPULATION, TBSR_METRIC
from experiments.stage2_ood_hle.partial import PARTIAL_ANALYSIS_SCHEMA, PARTIAL_HELD_OUT_MEAN, validate_partial_report
from experiments.stage2_ood_hle.plot import (
    CONDITION_LABELS,
    CONTROL_CONDITIONS,
    FIGURES,
    HELD_OUT_BACKGROUND,
    SEPARATOR_COLOR,
    TRAINED_BACKGROUND,
    _bias_labels,
    _color,
    _deterministic_rendering,
)

PARTIAL_DATASET_SPLIT_POPULATIONS = (
    {
        "population": IID_POPULATION,
        "suffix": "iid-by-bias-partial",
        "title": "IID held-out / training-domain dataset",
    },
    {
        "population": HLE_POPULATION,
        "suffix": "hle-by-bias-partial",
        "title": "HLE / held-out dataset",
    },
)


def _condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition)


def _categories(report: Mapping[str, Any]) -> tuple[str, ...]:
    config = report["config"]
    assert isinstance(config, Mapping)
    training = config["training_bias"]
    held_out = config["held_out_biases"]
    assert isinstance(training, str) and isinstance(held_out, list)
    return (training, *(str(value) for value in held_out), PARTIAL_HELD_OUT_MEAN)


def _cell(report: Mapping[str, Any], *, condition: str, population: str, category: str) -> Mapping[str, Any] | None:
    if category == PARTIAL_HELD_OUT_MEAN:
        value = report["held_out_summaries"].get(f"{condition}/{population}/{category}")
    else:
        value = report["cells"].get(f"{condition}/{population}/{category}")
    return value if isinstance(value, Mapping) else None


def _rate_interval(cell: Mapping[str, Any], *, rate_key: str, bootstrap_key: str) -> tuple[float, float, float] | None:
    rate = cell.get(rate_key)
    bootstrap = cell.get(bootstrap_key)
    if not isinstance(rate, Mapping) or not isinstance(bootstrap, Mapping) or rate.get("rate") is None:
        return None
    if rate_key == "bias_verbalised":
        if cell.get("luna_score_complete") is not True:
            # A parsed-null Luna verdict remains in the canonical denominator;
            # only an absent score mapping makes the cell ungraded.
            return None
    ci = bootstrap.get("ci_95")
    if not isinstance(ci, Mapping) or ci.get("lower") is None or ci.get("upper") is None:
        return None
    return float(rate["rate"]), float(ci["lower"]), float(ci["upper"])


def _marker(cell: Mapping[str, Any], *, metric: str) -> str:
    value = cell.get("significance")
    if not isinstance(value, Mapping):
        return ""
    annotation = value.get(metric)
    return str(annotation.get("marker", "")) if isinstance(annotation, Mapping) else ""


def _render_metric(
    report: Mapping[str, Any],
    *,
    metric: str,
    panel: Mapping[str, str],
    candidates: Mapping[str, Path],
) -> None:
    specification = FIGURES[metric]
    rate_key = str(specification["rate_key"])
    bootstrap_key = str(specification["bootstrap_key"])
    conditions = [str(value) for value in report["conditions"]]
    categories = _categories(report)
    population = str(panel["population"])
    labels = _bias_labels()
    x_labels = [
        (
            f"{labels.get(category, category.replace('_', ' ').title())}\n(trained bias)"
            if index == 0
            else (
                "Held-out avg.\n(micro pool; complete only)"
                if category == PARTIAL_HELD_OUT_MEAN
                else labels.get(category, category.replace("_", " ").title())
            )
        )
        for index, category in enumerate(categories)
    ]
    width = min(0.13, 0.78 / max(1, len(conditions)))
    offsets = (np.arange(len(conditions), dtype=float) - (len(conditions) - 1) / 2.0) * width
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
            "svg.hashsalt": "stage2-ood-hle-partial-dataset-split-v1",
        }
    ):
        figure, axis = plt.subplots(figsize=(max(10.4, 1.28 * len(categories) + 2.3), 5.0), layout="constrained")
        axis.axvspan(-0.5, 0.5, color=TRAINED_BACKGROUND, alpha=0.88, zorder=0)
        axis.axvspan(len(categories) - 1.5, len(categories) - 0.5, color=HELD_OUT_BACKGROUND, alpha=0.98, zorder=0)
        for separator in range(1, len(categories)):
            axis.axvline(separator - 0.5, color=SEPARATOR_COLOR, linewidth=0.6, zorder=1)

        for condition_index, condition in enumerate(conditions):
            color = _color(condition, condition_index)
            control = condition in CONTROL_CONDITIONS
            for category_index, category in enumerate(categories):
                cell = _cell(report, condition=condition, population=population, category=category)
                if cell is None:  # A truly blank cell; never a zero-filled estimate.
                    continue
                interval = _rate_interval(cell, rate_key=rate_key, bootstrap_key=bootstrap_key)
                if interval is None:
                    continue
                rate, lower, upper = interval
                position = x[category_index] + offsets[condition_index]
                axis.bar(
                    position,
                    rate,
                    width,
                    color="white" if control else color,
                    edgecolor=color if control else "#222429",
                    linewidth=1.1 if control else 0.45,
                    hatch="//" if control else "",
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
                marker = _marker(cell, metric=metric)
                if marker:
                    axis.text(
                        position,
                        min(1.215, upper + 0.016 + condition_index * 0.012),
                        marker,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#141518",
                        zorder=4,
                        clip_on=False,
                    )

        axis.set_xlim(-0.5, len(categories) - 0.5)
        # The small stagger lets multiple significance markers remain legible
        # when several methods are near a 100% verbalisation rate.
        axis.set_ylim(0.0, 1.24)
        axis.set_yticks(np.linspace(0, 1, 6))
        axis.set_yticklabels([f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)], fontsize=7)
        axis.set_xticks(x, x_labels, rotation=31, ha="right", fontsize=7)
        axis.yaxis.grid(True)
        axis.grid(axis="x", visible=False)
        axis.set_ylabel(str(specification["y_label"]), fontsize=8)
        axis.set_title(f"{panel['title']} — partial results", loc="left", fontsize=9, fontweight="semibold", pad=17)
        axis.text(
            0.0,
            1.013,
            "95% question-cluster percentile bootstrap CI. Blank = incomplete or unavailable; never zero-filled.",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.3,
            color="#4a4d54",
        )
        legend = [
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
            handles=legend,
            loc="outside lower center",
            ncols=min(len(legend), 6),
            fontsize=7,
            frameon=False,
            title="Condition",
            title_fontsize=7,
        )
        figure.text(
            1.0,
            1.0,
            "* p<0.05   ** p<0.01   *** p<0.001 vs Base; two-sided two-proportion z-test, uncorrected",
            ha="right",
            va="top",
            fontsize=6.2,
        )
        for extension, candidate in candidates.items():
            figure.savefig(candidate, format=extension, metadata={"Date": None})
        plt.close(figure)


def render_partial_dataset_split_figures(report: Mapping[str, Any], output_dir: str | Path) -> str:
    """Write four audit-labelled partial dataset-split figures (PNG and SVG)."""

    validate_partial_report(report)
    if report.get("schema") != PARTIAL_ANALYSIS_SCHEMA:  # defensive; public validator owns this boundary
        raise ValueError("unexpected partial Stage 2 report schema")
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        candidates: dict[Path, Path] = {}
        with _deterministic_rendering():
            for metric in (TBSR_METRIC, BIAS_VERBALISED_METRIC):
                for panel in PARTIAL_DATASET_SPLIT_POPULATIONS:
                    stem = f"ood-{metric.replace('_', '-')}-{panel['suffix']}"
                    per_figure = {extension: staging / f"{stem}.{extension}" for extension in ("png", "svg")}
                    _render_metric(report, metric=metric, panel=panel, candidates=per_figure)
                    candidates.update({directory / source.name: source for source in per_figure.values()})
        for destination, source in candidates.items():
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise FileExistsError(f"refusing to overwrite differing partial Stage 2 figure: {destination}")
        directory.mkdir(parents=True, exist_ok=True)
        wrote = False
        for destination, source in candidates.items():
            if destination.exists():
                continue
            try:
                os.link(source, destination)
                wrote = True
            except FileExistsError:
                if destination.read_bytes() != source.read_bytes():
                    raise FileExistsError(f"partial Stage 2 figure appeared and differs: {destination}")
        return "written" if wrote else "resumed"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.analysis.read_text(encoding="utf-8"))
        if not isinstance(report, Mapping):
            raise TypeError("partial Stage 2 analysis must be a JSON object")
        status = render_partial_dataset_split_figures(report, args.output_dir)
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output_dir.resolve()} (8 files)")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
