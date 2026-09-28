"""Render Stage 1 IID diagnostics with the main mcq-bias figure pipeline."""

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
from experiments.stage1_iid_diagnostic.analyze import ANALYSIS_SCHEMA

CONDITIONS = (
    "untrained",
    "bct",
    "bct-control",
    "rmct",
    "rmct-control",
    "act",
    "attct",
    "mlpct",
    "opct",
)
SPLITS = ("train_eval", "heldout_in_domain")

# The diagnostic runner uses short directory names. Chart-ready rows use the
# canonical identifiers from the main Stage 1 experiment so the shared registry
# supplies exactly the same labels, colors, control outlines, and legend order.
CONDITION_IDENTITIES = {
    "untrained": ("untrained", "none", False),
    "bct": ("bias-augmented-consistency", "bias_augmented_consistency", False),
    "bct-control": ("bias-augmented-consistency-control", "bias_augmented_consistency", True),
    "rmct": ("rate-matching", "rate_matching", False),
    "rmct-control": ("rate-matching-control", "rate_matching", True),
    "act": ("act", "act", False),
    "attct": ("attct", "attct", False),
    "mlpct": ("mlpct", "mlpct", False),
    "opct": ("opct", "opct", False),
}
CONDITION_ORDER = (
    "untrained",
    "bias-augmented-consistency",
    "bias-augmented-consistency-control",
    "opct",
    "rate-matching",
    "rate-matching-control",
    "act",
    "attct",
    "mlpct",
)
BIAS_IDENTITIES = {
    "train_eval": "training_domain",
    "heldout_in_domain": "held_out_in_domain",
}
BIAS_LABELS = {
    "training_domain": "Training-domain samples",
    "held_out_in_domain": "Held-out in-domain samples",
}
MODEL = "Qwen/Qwen3.5-9B"

_MAIN_CHART_ROOT = Path(__file__).resolve().parents[1] / "rmct_paper_vast_more_methods"
FIGURES = {
    "towards-bias-switch": ("tbsr", "towards_bias_switch", "switch_chart.json"),
    "bias-verbalised": ("luna_yes", "bias_acknowledged", "verbalisation_chart.json"),
}


def _rate(report: Mapping[str, Any], condition: str, split: str, metric: str) -> tuple[int, int, float]:
    key = f"{condition}/{split}"
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ValueError("analysis cells must be an object")
    cell = cells.get(key)
    if not isinstance(cell, Mapping):
        raise ValueError(f"analysis is missing required cell {key!r}")
    if cell.get("condition") != condition or cell.get("split") != split:
        raise ValueError(f"analysis cell {key!r} has conflicting identity fields")
    pooled = cell.get("pooled")
    rates = pooled.get("rates") if isinstance(pooled, Mapping) else None
    value = rates.get(metric) if isinstance(rates, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"analysis cell {key!r} is missing pooled rate {metric!r}")
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
        raise ValueError(f"analysis cell {key!r} has an invalid pooled rate {metric!r}")
    expected = numerator / denominator
    if not math.isclose(float(rate), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"analysis cell {key!r} has inconsistent {metric!r} rate: {rate!r} != {numerator}/{denominator}"
        )
    return numerator, denominator, expected


def validate_report(report: Mapping[str, Any]) -> None:
    """Fail closed unless *report* is the complete final nine-condition matrix."""

    if report.get("schema") != ANALYSIS_SCHEMA:
        raise ValueError(f"analysis schema must be {ANALYSIS_SCHEMA!r}")
    cells = report.get("cells")
    if not isinstance(cells, Mapping):
        raise ValueError("analysis cells must be an object")
    expected = {f"{condition}/{split}" for condition in CONDITIONS for split in SPLITS}
    actual = set(cells)
    if actual != expected:
        raise ValueError(
            "analysis must contain the final nine-condition matrix; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    for _, (analysis_metric, _, _) in FIGURES.items():
        for condition in CONDITIONS:
            for split in SPLITS:
                _rate(report, condition, split, analysis_metric)


def _sample_count(report: Mapping[str, Any], condition: str, split: str) -> int:
    cell = report["cells"][f"{condition}/{split}"]
    pooled = cell.get("pooled") if isinstance(cell, Mapping) else None
    counts = pooled.get("counts") if isinstance(pooled, Mapping) else None
    samples = counts.get("samples") if isinstance(counts, Mapping) else None
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError(f"analysis cell {condition}/{split!s} has an invalid pooled sample count")
    return samples


def chart_rows(report: Mapping[str, Any], analysis_metric: str, chart_metric: str) -> list[dict[str, Any]]:
    """Adapt exact-count diagnostic rates to the main plotter's chart-row boundary."""

    rows: list[dict[str, Any]] = []
    # Split-major order and the explicit bias order put the training-domain
    # group on the left and the held-out in-domain group on the right.
    for split in SPLITS:
        for condition in CONDITIONS:
            numerator, denominator, mean = _rate(report, condition, split, analysis_metric)
            sample_count = _sample_count(report, condition, split)
            canonical_condition, method, is_control = CONDITION_IDENTITIES[condition]
            rows.append(
                {
                    "condition": canonical_condition,
                    "method": method,
                    "is_control": is_control,
                    "bias_type": BIAS_IDENTITIES[split],
                    "metric": chart_metric,
                    "mean": mean,
                    # Match mcq_bias.analysis's Bernoulli sample fallback.
                    "stderr": (
                        math.sqrt(mean * (1.0 - mean) / (denominator - 1))
                        if denominator > 1
                        else 0.0
                    ),
                    "n_scored": denominator,
                    "n_total": sample_count,
                    "numerator": numerator,
                    "model": MODEL,
                    # This is panel-level experiment metadata, including for the
                    # untrained comparator, just as in the main Stage 1 figures.
                    "training_biases": ["training_domain"],
                    "split": split,
                    "significance": "",
                }
            )
    return rows


def chart_spec(chart_metric: str, recipe_name: str) -> dict[str, Any]:
    """Load a main-figure recipe and add only the diagnostic facet declaration."""

    recipe = json.loads((_MAIN_CHART_ROOT / recipe_name).read_text())
    return {
        **recipe,
        "metric": chart_metric,
        "condition_order": list(CONDITION_ORDER),
        "bias_order": ["training_domain", "held_out_in_domain"],
        "bias_labels": BIAS_LABELS,
        "model_order": [MODEL],
        "model_labels": {MODEL: MODEL},
        "held_out_label": "held_out_in_domain",
        "show_significance": False,
        "theme": {"rcparams": {"svg.hashsalt": "stage1-iid-main-style-v1"}},
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


def render_figures(report: Mapping[str, Any], output_dir: str | Path) -> str:
    """Render both formats through the shared publication renderer and publish atomically."""

    validate_report(report)
    directory = Path(output_dir).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{directory.name}.", dir=directory.parent) as temporary:
        staging = Path(temporary)
        candidates: dict[Path, Path] = {}
        with _deterministic_rendering():
            for stem, (analysis_metric, chart_metric, recipe_name) in FIGURES.items():
                rows = chart_rows(report, analysis_metric, chart_metric)
                spec = chart_spec(chart_metric, recipe_name)
                for extension in ("png", "svg"):
                    candidate = staging / f"{stem}.{extension}"
                    render_publication_plot(rows, spec, candidate)
                    candidates[directory / candidate.name] = candidate

        for destination, candidate in candidates.items():
            if destination.exists() and destination.read_bytes() != candidate.read_bytes():
                raise FileExistsError(f"refusing to overwrite differing figure: {destination}")

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
                    raise FileExistsError(f"figure appeared during publication and differs: {destination}")
        return "written" if wrote else "resumed"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.analysis.read_text())
        if not isinstance(report, Mapping):
            raise TypeError("analysis must be a JSON object")
        status = render_figures(report, args.output_dir)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"{status}: {args.output_dir.resolve()} ({len(FIGURES) * 2} files)")


if __name__ == "__main__":
    main()
