"""Render publication grouped-bar figures from adapter-owned chart-ready JSON.

This module deliberately starts at CTM's analysis boundary. It never opens an
Inspect log or computes an estimate, error bar, held-out average, or p-value.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from ctm_data.adapters._plot_registry import (
    PresentationRegistry,
    load_presentation_registry,
    registry_labels,
)

_DEFAULT_COLORS = {
    "none": "#9aa0a6",
    "bias_augmented_consistency": "#7aa7d9",
    "rate_matching": "#8cc39a",
    "act": "#d99a6c",
    "attct": "#9e9ac8",
    "mlpct": "#d58aaa",
}

_DEFAULT_CONDITION_LABELS = {
    "untrained": "Base",
    "bias-augmented-consistency": "BCT",
    "bias-augmented-consistency-control": "BCT Control",
    "rate-matching": "RMCT",
    "rate-matching-control": "RMCT Control",
    "act": "ACT",
    "act-control": "ACT Control",
    "attct": "AttCT",
    "attct-control": "AttCT Control",
    "mlpct": "MLPCT",
    "mlpct-control": "MLPCT Control",
}

_RCPARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "svg.fonttype": "none",
    "font.family": ["DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "axes.edgecolor": "#141518",
    "axes.linewidth": 0.8,
    "grid.color": "#ececee",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
}


@dataclass(frozen=True)
class PlotTheme:
    """Declarative publication theme; callbacks may return a modified copy."""

    rcparams: Mapping[str, Any] = field(default_factory=lambda: dict(_RCPARAMS))
    figure_width_min: float = 5.3
    figure_width_per_category: float = 0.62
    figure_width_intercept: float = 1.7
    figure_height_per_row: float = 2.15
    figure_height_intercept: float = 0.35
    trained_background: str = "#fbf6ea"
    held_out_background: str = "#f1f1ee"
    separator_color: str = "#d8dadf"
    foreground: str = "#141518"
    error_capsize: float = 1.4
    error_linewidth: float = 0.65
    panel_title_fontsize: float = 8.0
    tick_fontsize: float = 7.0
    annotation_fontsize: float = 7.0
    sample_label_fontsize: float = 5.5


@dataclass(frozen=True)
class PlotSchema:
    """Adapter vocabulary and presentation defaults for the shared renderer."""

    category_field: str
    category_label_field: str
    category_section: str
    category_order_key: str
    category_order_spec: str
    category_labels_spec: str
    default_registry_path: Path | None = None
    training_categories_field: str | None = None
    auto_column_field: str | None = None
    held_out_label: str | None = None
    category_label_rotation: float = 40.0
    category_label_alignment: str = "right"


@dataclass(frozen=True)
class PlotFacet:
    title: str
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class FacetLayout:
    panels: tuple[tuple[PlotFacet | None, ...], ...]

    @property
    def nrows(self) -> int:
        return len(self.panels)

    @property
    def ncols(self) -> int:
        return max((len(row) for row in self.panels), default=0)


ThemeCallback = Callable[[PlotTheme, Mapping[str, Any]], PlotTheme | Mapping[str, Any]]
FacetCallback = Callable[[Sequence[Mapping[str, Any]], Mapping[str, Any], PresentationRegistry], FacetLayout]
BarStyleCallback = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


def _theme_from_mapping(base: PlotTheme, values: Mapping[str, Any]) -> PlotTheme:
    values = dict(values)
    legacy_width = values.pop("figure_width_per_bias", None)
    if legacy_width is not None:
        if "figure_width_per_category" in values:
            raise ValueError("theme may use figure_width_per_category or figure_width_per_bias, not both")
        values["figure_width_per_category"] = legacy_width
    allowed = set(PlotTheme.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"theme has unknown fields: {unknown}")
    updates = dict(values)
    if "rcparams" in updates:
        if not isinstance(updates["rcparams"], Mapping):
            raise ValueError("theme.rcparams must be an object")
        updates["rcparams"] = {**base.rcparams, **updates["rcparams"]}
    return replace(base, **updates)


def _import_callback(reference: str, label: str) -> Callable[..., Any]:
    if ":" not in reference:
        raise ValueError(f"{label} must be package.module:function")
    module_name, attribute = reference.split(":", 1)
    callback = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(callback):
        raise TypeError(f"{label} is not callable: {reference}")
    return callback


def _hashable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_hashable(item) for item in value)
    return value


def _field_key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(str(row.get(field, "") or "") if field == "model" else _hashable(row.get(field)) for field in fields)


def _field_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, str) for item in value)
    ):
        return list(value)
    raise ValueError(f"facet.{label} must be a string or array of strings")


def _facet_value_label(
    field_name: str,
    value: Any,
    *,
    registry: PresentationRegistry,
    model_labels: Mapping[str, str],
    category_labels: Mapping[str, str],
    schema: PlotSchema,
    spec: Mapping[str, Any],
) -> str:
    custom = (
        spec.get("facet_labels", {}).get(field_name, {}) if isinstance(spec.get("facet_labels", {}), Mapping) else {}
    )
    lookup_key = "|".join(str(item) for item in value) if isinstance(value, tuple) else str(value)
    if isinstance(custom, Mapping) and lookup_key in custom:
        return str(custom[lookup_key])
    if field_name == "model":
        return model_labels.get(str(value), str(value))
    if schema.training_categories_field and field_name == schema.training_categories_field:
        values = value if isinstance(value, tuple) else (value,)
        names = [category_labels.get(str(item), str(item).replace("_", " ").title()) for item in values]
        return f"Training: {', '.join(names)}" if names else "Untrained"
    if field_name == schema.category_field:
        return category_labels.get(lookup_key, lookup_key.replace("_", " ").title())
    return f"{field_name.replace('_', ' ').title()}: {lookup_key}"


def _default_facet_layout(
    data: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    registry: PresentationRegistry,
    model_labels: Mapping[str, str],
    category_labels: Mapping[str, str],
    schema: PlotSchema,
) -> FacetLayout:
    facet = spec.get("facet", "auto")
    if facet == "auto" or facet is None:
        row_fields = ["model"]
        if schema.auto_column_field:
            distinct_columns = list(dict.fromkeys(_field_key(row, [schema.auto_column_field])[0] for row in data))
            column_fields = [schema.auto_column_field] if len(distinct_columns) > 1 else []
        else:
            column_fields = []
    elif facet is False or facet == "none":
        row_fields, column_fields = [], []
    elif isinstance(facet, str):
        row_fields, column_fields = [facet], []
    elif isinstance(facet, Mapping):
        row_fields = _field_list(facet.get("rows"), "rows")
        column_fields = _field_list(facet.get("columns"), "columns")
    else:
        raise ValueError("facet must be auto, none, a field name, or an object")
    overlap = sorted(set(row_fields) & set(column_fields))
    if overlap:
        raise ValueError(f"facet fields cannot appear in both rows and columns: {overlap}")

    row_keys = list(dict.fromkeys(_field_key(row, row_fields) for row in data)) or [()]
    if row_fields == ["model"]:
        preferred_models = spec.get("model_order", registry.ordering.get("models"))
        row_keys = [(value,) for value in _ordered([str(key[0] or "") for key in row_keys], preferred_models)]
    column_keys = list(dict.fromkeys(_field_key(row, column_fields) for row in data)) or [()]
    panels = []
    for row_key in row_keys:
        panel_row = []
        for column_key in column_keys:
            selected = tuple(
                row
                for row in data
                if _field_key(row, row_fields) == row_key and _field_key(row, column_fields) == column_key
            )
            if not selected:
                panel_row.append(None)
                continue
            title_parts = [
                _facet_value_label(
                    field_name,
                    value,
                    registry=registry,
                    model_labels=model_labels,
                    category_labels=category_labels,
                    schema=schema,
                    spec=spec,
                )
                for field_name, value in [*zip(row_fields, row_key), *zip(column_fields, column_key)]
                if value not in (None, "", ())
            ]
            panel_row.append(PlotFacet(" · ".join(title_parts), selected))
        panels.append(tuple(panel_row))
    return FacetLayout(tuple(panels))


def _ordered(present: Sequence[str], preferred: Sequence[str] | None) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in present))
    if not preferred:
        return values
    preferred_values = [str(value) for value in preferred]
    return [value for value in preferred_values if value in values] + [
        value for value in values if value not in preferred_values
    ]


def _condition_offsets(rows: Sequence[Mapping[str, Any]], conditions: Sequence[str]) -> tuple[dict[str, float], float]:
    by_family: dict[str, list[str]] = defaultdict(list)
    metadata = {str(row["condition"]): row for row in rows}
    for condition in conditions:
        by_family[str(metadata[condition].get("method", condition))].append(condition)
    for family_conditions in by_family.values():
        family_conditions.sort(key=lambda condition: bool(metadata[condition].get("is_control", False)))

    bar_width = min(0.14, 0.72 / max(len(conditions), 1))
    gap = bar_width * 0.40
    family_order = list(dict.fromkeys(str(metadata[condition].get("method", condition)) for condition in conditions))
    total_width = len(conditions) * bar_width + max(0, len(family_order) - 1) * gap
    cursor = -total_width / 2 + bar_width / 2
    offsets: dict[str, float] = {}
    for family_index, family in enumerate(family_order):
        for condition in by_family[family]:
            offsets[condition] = cursor
            cursor += bar_width
        if family_index < len(family_order) - 1:
            cursor += gap
    return offsets, bar_width


def _validate_rows(rows: Any, *, category_field: str) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("data must be a non-empty JSON array")
    required = {"condition", category_field, "metric", "mean", "stderr", "n_scored"}
    output = []
    for index, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise TypeError(f"row {index} must be an object")
        missing = sorted(required - source.keys())
        if missing:
            raise ValueError(f"row {index} is missing {missing}")
        row = dict(source)
        for numeric_field in ("mean", "stderr"):
            value = float(row[numeric_field])
            if not math.isfinite(value):
                raise ValueError(f"row {index} has non-finite {numeric_field}")
            row[numeric_field] = value
        has_ci_lower = "ci_lower" in row
        has_ci_upper = "ci_upper" in row
        if has_ci_lower != has_ci_upper:
            raise ValueError(f"row {index} must provide ci_lower and ci_upper together")
        if has_ci_lower:
            lower, upper = float(row["ci_lower"]), float(row["ci_upper"])
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise ValueError(f"row {index} has non-finite asymmetric confidence interval bounds")
            if lower > row["mean"] or row["mean"] > upper:
                raise ValueError(f"row {index} confidence interval must satisfy ci_lower <= mean <= ci_upper")
            row["ci_lower"] = lower
            row["ci_upper"] = upper
        output.append(row)
    return output


def _interval(row: Mapping[str, Any], *, ci_multiplier: float) -> tuple[float, float]:
    """Return exact asymmetric bounds when supplied, else the legacy symmetric interval."""

    if "ci_lower" in row:
        return float(row["ci_lower"]), float(row["ci_upper"])
    error = ci_multiplier * float(row["stderr"])
    return float(row["mean"]) - error, float(row["mean"]) + error


def _legend_handles_row_major(handles: Sequence[Patch], columns: int) -> list[Patch]:
    """Arrange handles so Matplotlib's column-major legend reads row-major."""

    rows = math.ceil(len(handles) / columns)
    return [
        handles[index]
        for column in range(columns)
        for row in range(rows)
        if (index := row * columns + column) < len(handles)
    ]


def render_publication_plot(
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    output: Path,
    *,
    schema: PlotSchema,
    theme_callback: ThemeCallback | None = None,
    facet_callback: FacetCallback | None = None,
    bar_style_callback: BarStyleCallback | None = None,
) -> None:
    """Render a publication figure without mutating chart-ready statistics.

    Declarative recipe settings cover normal use. The three optional callbacks
    are deliberate Python-level escape hatches for new visual designs and
    computed facets without weakening the JSON data boundary.
    """

    category_field = str(spec.get("category_field", schema.category_field))
    if not category_field:
        raise ValueError("category_field must be non-empty")
    category_label_field = (
        schema.category_label_field if category_field == schema.category_field else f"{category_field}_label"
    )
    data = _validate_rows(list(rows), category_field=category_field)
    if output.exists():
        raise ValueError(f"refusing to overwrite existing output: {output}")
    if output.suffix.lower() not in {".svg", ".pdf", ".png"}:
        raise ValueError("output extension must be .svg, .pdf, or .png")

    registry = load_presentation_registry(
        spec.get("registry"),
        default_path=schema.default_registry_path,
        category_section=schema.category_section,
    )
    metric = str(spec.get("metric", data[0]["metric"]))
    data = [row for row in data if str(row["metric"]) == metric]
    if not data:
        raise ValueError(f"no rows for metric {metric!r}")
    show_controls = bool(spec.get("show_controls", True))
    if not show_controls:
        data = [row for row in data if not bool(row.get("is_control", False))]

    condition_preference = spec.get("condition_order", registry.ordering.get("training_types"))
    category_preference = spec.get(
        schema.category_order_spec,
        spec.get("category_order", registry.ordering.get(schema.category_order_key)),
    )
    model_preference = spec.get("model_order", registry.ordering.get("models"))
    models = _ordered([str(row.get("model", "")) for row in data], model_preference)
    conditions = _ordered([str(row["condition"]) for row in data], condition_preference)
    categories = _ordered([str(row[category_field]) for row in data], category_preference)
    has_explicit_category_order = schema.category_order_spec in spec or "category_order" in spec
    if not has_explicit_category_order and schema.training_categories_field:
        training_sets = [
            {str(value) for value in (row.get(schema.training_categories_field, []) or [])} for row in data
        ]
        training_sets = [values for values in training_sets if values]
        if training_sets:
            intersection = set.intersection(*training_sets)
            union = set.union(*training_sets)
            categories = (
                [category for category in categories if category in intersection]
                + [category for category in categories if category in union and category not in intersection]
                + [category for category in categories if category not in union]
            )
    for mapping_name in (
        "condition_labels",
        schema.category_labels_spec,
        "category_labels",
        "model_labels",
        "method_colors",
        "condition_styles",
    ):
        if mapping_name in spec and not isinstance(spec[mapping_name], Mapping):
            raise ValueError(f"{mapping_name} must be an object")
    condition_styles = spec.get("condition_styles", {})
    if any(not isinstance(value, Mapping) for value in condition_styles.values()):
        raise TypeError("condition_styles entries must be objects")
    row_condition_labels = {
        str(row["condition"]): str(row["condition_label"])
        for row in data
        if isinstance(row.get("condition_label"), str)
    }
    row_category_labels = {
        str(row[category_field]): str(row[category_label_field])
        for row in data
        if isinstance(row.get(category_label_field), str)
    }
    row_model_labels = {
        str(row.get("model", "")): str(row["model_label"]) for row in data if isinstance(row.get("model_label"), str)
    }
    condition_labels = {
        **_DEFAULT_CONDITION_LABELS,
        **row_condition_labels,
        **registry_labels(registry.training_types),
        **{str(key): str(value) for key, value in spec.get("condition_labels", {}).items()},
    }
    category_labels = {
        **row_category_labels,
        **registry_labels(registry.categories),
        **{str(key): str(value) for key, value in spec.get("category_labels", {}).items()},
        **{str(key): str(value) for key, value in spec.get(schema.category_labels_spec, {}).items()},
    }
    model_labels = {
        **row_model_labels,
        **registry_labels(registry.models),
        **{str(key): str(value) for key, value in spec.get("model_labels", {}).items()},
    }
    registry_method_colors = {
        str(key): str(value["color"]) for key, value in registry.methods.items() if isinstance(value.get("color"), str)
    }
    method_colors = {**_DEFAULT_COLORS, **registry_method_colors, **spec.get("method_colors", {})}
    ci_multiplier = float(spec.get("ci_multiplier", 2.0))
    if not math.isfinite(ci_multiplier) or ci_multiplier < 0:
        raise ValueError("ci_multiplier must be a finite non-negative number")
    significance_note = spec.get("significance_note")
    if significance_note is not None and (not isinstance(significance_note, str) or not significance_note.strip()):
        raise ValueError("significance_note must be a non-empty string when provided")
    held_out_value = spec.get("held_out_label", schema.held_out_label)
    held_out_label = str(held_out_value) if held_out_value is not None else None
    sample_labels = spec.get("sample_labels", False)
    if sample_labels is True:
        sample_label_field = "n_scored"
    elif sample_labels is False or sample_labels is None:
        sample_label_field = None
    elif sample_labels in {"n_scored", "n_total"}:
        sample_label_field = str(sample_labels)
    else:
        raise ValueError("sample_labels must be false, true, 'n_scored', or 'n_total'")

    if not models or not conditions or not categories:
        raise ValueError("plot has no model panels, conditions, or categories after filtering")

    theme_value = spec.get("theme", {})
    if not isinstance(theme_value, Mapping):
        raise TypeError("theme must be an object")
    theme = _theme_from_mapping(PlotTheme(), theme_value)
    if theme_callback is None and isinstance(spec.get("theme_callback"), str):
        theme_callback = _import_callback(str(spec["theme_callback"]), "theme_callback")
    if theme_callback is not None:
        customized = theme_callback(theme, spec)
        theme = customized if isinstance(customized, PlotTheme) else _theme_from_mapping(theme, customized)

    if facet_callback is None and isinstance(spec.get("facet_callback"), str):
        facet_callback = _import_callback(str(spec["facet_callback"]), "facet_callback")
    layout = (
        facet_callback(data, spec, registry)
        if facet_callback is not None
        else _default_facet_layout(data, spec, registry, model_labels, category_labels, schema)
    )
    if not isinstance(layout, FacetLayout) or not layout.nrows or not layout.ncols:
        raise ValueError("facet callback must return a non-empty FacetLayout")
    if bar_style_callback is None and isinstance(spec.get("bar_style_callback"), str):
        bar_style_callback = _import_callback(str(spec["bar_style_callback"]), "bar_style_callback")

    plt.rcParams.update(theme.rcparams)
    panel_width = max(
        theme.figure_width_min,
        theme.figure_width_per_category * len(categories) + theme.figure_width_intercept,
    )
    figure_width = panel_width * layout.ncols
    figure_height = theme.figure_height_per_row * layout.nrows + theme.figure_height_intercept
    figure, axes = plt.subplots(
        layout.nrows,
        layout.ncols,
        figsize=(figure_width, figure_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    centers = list(range(len(categories)))
    panel_local_conditions = bool(spec.get("panel_local_conditions", False))
    if panel_local_conditions:
        offsets: dict[str, float] = {}
        bar_width = 0.0
    else:
        offsets, bar_width = _condition_offsets(data, conditions)

    ylim_value = spec.get("ylim")
    if ylim_value is None:
        intervals = [_interval(row, ci_multiplier=ci_multiplier) for row in data]
        upper = max(interval[1] for interval in intervals)
        lower = min(interval[0] for interval in intervals)
        is_percent_change = any(row.get("transform") == "percent_change" for row in data)
        if is_percent_change:
            lower, upper = min(0.0, lower), max(0.0, upper)
            span = max(upper - lower, 1.0)
            annotation_pad = 0.18 if sample_label_field else 0.10 if spec.get("show_significance", True) else 0.05
            ylim = (lower - annotation_pad * span if lower < 0 else 0.0, upper + annotation_pad * span)
        else:
            annotation_headroom = 0.20 if sample_label_field else 0.14
            ylim = (min(0.0, lower - 0.04), max(1.0, upper + annotation_headroom))
    elif isinstance(ylim_value, list) and len(ylim_value) == 2:
        ylim = (float(ylim_value[0]), float(ylim_value[1]))
    else:
        raise ValueError("ylim must be a two-element array")
    if not all(math.isfinite(value) for value in ylim) or ylim[0] >= ylim[1]:
        raise ValueError("ylim values must be finite and increasing")
    y_span = ylim[1] - ylim[0]

    for panel_row_index, panel_row in enumerate(layout.panels):
        for panel_column_index in range(layout.ncols):
            axis = axes[panel_row_index][panel_column_index]
            facet = panel_row[panel_column_index] if panel_column_index < len(panel_row) else None
            if facet is None:
                axis.set_visible(False)
                continue
            panel_data = list(facet.rows)
            panel_conditions = (
                _ordered([str(row["condition"]) for row in panel_data], conditions)
                if panel_local_conditions
                else conditions
            )
            panel_offsets, panel_bar_width = (
                _condition_offsets(panel_data, panel_conditions)
                if panel_local_conditions
                else (offsets, bar_width)
            )
            lookup = {(str(row["condition"]), str(row[category_field])): row for row in panel_data}
            if len(lookup) != len(panel_data):
                raise ValueError(
                    f"multiple rows occupy the same (condition, {category_field}) cell in facet {facet.title!r}; "
                    "include the differing field in facet.rows or facet.columns"
                )
            training_categories = (
                {
                    str(category)
                    for row in panel_data
                    for category in (row.get(schema.training_categories_field, []) or [])
                }
                if schema.training_categories_field
                else set()
            )

            for category_index, category in enumerate(categories):
                if category in training_categories:
                    axis.axvspan(
                        category_index - 0.5,
                        category_index + 0.5,
                        color=theme.trained_background,
                        alpha=0.75,
                        zorder=0,
                    )
                elif held_out_label is not None and category == held_out_label:
                    axis.axvspan(
                        category_index - 0.5,
                        category_index + 0.5,
                        color=theme.held_out_background,
                        alpha=0.95,
                        zorder=0,
                    )
                if category_index:
                    axis.axvline(
                        category_index - 0.5,
                        color=theme.separator_color,
                        linewidth=0.6,
                        zorder=1,
                    )

            for condition in panel_conditions:
                example = next((row for row in panel_data if str(row["condition"]) == condition), None)
                if example is None:
                    continue
                method = str(example.get("method", condition))
                registry_style = registry.training_types.get(condition, {})
                color = str(
                    condition_styles.get(condition, {}).get(
                        "color",
                        registry_style.get(
                            "color",
                            example.get("color", method_colors.get(condition, method_colors.get(method, "#888888"))),
                        ),
                    )
                )
                is_control = bool(example.get("is_control", False))
                for category_index, category in enumerate(categories):
                    row = lookup.get((condition, category))
                    if row is None:
                        continue
                    x = category_index + panel_offsets[condition]
                    style: Mapping[str, Any] = {
                        "facecolor": "white" if is_control else color,
                        "edgecolor": color,
                        "linewidth": 2.4 if is_control else 0.0,
                        "hatch": registry_style.get("hatch", ""),
                    }
                    if bar_style_callback is not None:
                        style = bar_style_callback(row, style)
                        if not isinstance(style, Mapping):
                            raise ValueError("bar_style_callback must return a mapping")
                    bars = axis.bar(x, row["mean"], panel_bar_width, zorder=2, **style)
                    if float(style.get("linewidth", 0.0)):
                        for bar in bars:
                            bar.set_clip_path(bar)
                    lower, upper = _interval(row, ci_multiplier=ci_multiplier)
                    lower_error = max(0.0, row["mean"] - lower)
                    upper_error = max(0.0, upper - row["mean"])
                    if lower_error > 0 or upper_error > 0:
                        axis.errorbar(
                            x,
                            row["mean"],
                            yerr=[[lower_error], [upper_error]],
                            fmt="none",
                            ecolor=theme.foreground,
                            capsize=theme.error_capsize,
                            linewidth=theme.error_linewidth,
                            alpha=0.75,
                            zorder=3,
                        )
                    direction = 1 if row["mean"] >= 0 else -1
                    outside = upper if direction > 0 else lower
                    vertical_alignment = "bottom" if direction > 0 else "top"
                    marker = str(row.get("significance", ""))
                    marker_visible = bool(marker and spec.get("show_significance", True))
                    if marker_visible:
                        axis.text(
                            x,
                            outside + direction * 0.018 * y_span,
                            marker,
                            ha="center",
                            va=vertical_alignment,
                            fontsize=theme.annotation_fontsize,
                            color=theme.foreground,
                            zorder=4,
                            clip_on=False,
                        )
                    if sample_label_field is not None:
                        if sample_label_field not in row:
                            raise ValueError(f"sample label field {sample_label_field!r} is absent from a chart row")
                        marker_space = 0.04 * y_span if marker_visible else 0.0
                        axis.text(
                            x,
                            outside + direction * (0.018 * y_span + marker_space),
                            f"n={int(row[sample_label_field])}",
                            ha="center",
                            va=vertical_alignment,
                            rotation=90,
                            fontsize=theme.sample_label_fontsize,
                            color=theme.foreground,
                            alpha=0.82,
                            zorder=4,
                            clip_on=False,
                        )

            axis.set_xlim(-0.5, len(categories) - 0.5)
            axis.set_ylim(*ylim)
            if panel_column_index == 0:
                axis.set_ylabel(str(spec.get("ylabel", metric)))
            axis.grid(axis="y")
            axis.grid(axis="x", visible=False)
            is_percent_change = any(row.get("transform") == "percent_change" for row in panel_data)
            if bool(spec.get("zero_line", is_percent_change)):
                axis.axhline(0.0, color=theme.foreground, linewidth=0.8, alpha=0.7, zorder=1)
            if is_percent_change:
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}%"))
            elif bool(spec.get("percent", True)):
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
            if facet.title:
                axis.set_title(
                    facet.title,
                    loc="left",
                    fontsize=theme.panel_title_fontsize,
                    fontweight="semibold",
                    pad=2,
                )
            if panel_row_index == layout.nrows - 1:
                label_rotation = float(spec.get("category_label_rotation", schema.category_label_rotation))
                label_alignment = str(spec.get("category_label_alignment", schema.category_label_alignment))
                if not math.isfinite(label_rotation):
                    raise ValueError("category_label_rotation must be finite")
                if label_alignment not in {"left", "center", "right"}:
                    raise ValueError("category_label_alignment must be left, center, or right")
                axis.set_xticks(centers)
                axis.set_xticklabels(
                    [category_labels.get(category, category.replace("_", " ").title()) for category in categories],
                    rotation=label_rotation,
                    ha=label_alignment,
                    fontsize=theme.tick_fontsize,
                )

    legend_handles = []
    for condition in conditions:
        example = next((row for row in data if str(row["condition"]) == condition), None)
        if example is None:
            continue
        method = str(example.get("method", condition))
        registry_style = registry.training_types.get(condition, {})
        color = str(
            condition_styles.get(condition, {}).get(
                "color",
                registry_style.get(
                    "color",
                    example.get("color", method_colors.get(condition, method_colors.get(method, "#888888"))),
                ),
            )
        )
        is_control = bool(example.get("is_control", False))
        style: Mapping[str, Any] = {
            "facecolor": "white" if is_control else color,
            "edgecolor": color,
            "linewidth": 1.2 if is_control else 0.5,
            "hatch": registry_style.get("hatch", ""),
        }
        if bar_style_callback is not None:
            style = bar_style_callback(example, style)
        legend_handles.append(
            Patch(
                label=condition_labels.get(condition, condition.replace("-", " ").title()),
                **style,
            )
        )
    custom_significance_footer = (
        significance_note if bool(spec.get("show_significance", True)) and significance_note is not None else None
    )
    figure.tight_layout(
        pad=0.25,
        h_pad=1.25,
        w_pad=0.8,
        rect=(0.0, 0.075, 1.0, 1.0) if custom_significance_footer is not None else None,
    )
    legend_columns = min(
        len(legend_handles),
        int(spec.get("legend_columns", min(6, len(legend_handles)))),
    )
    if legend_handles and legend_columns < 1:
        raise ValueError("legend_columns must be at least 1")
    if legend_handles:
        figure.legend(
            handles=_legend_handles_row_major(legend_handles, legend_columns),
            loc="lower right",
            bbox_to_anchor=(1.0, 1.0),
            ncol=legend_columns,
            fontsize=theme.tick_fontsize,
            handlelength=1.1,
            columnspacing=0.9,
        )
    if custom_significance_footer is not None:
        figure.text(
            0.01,
            0.012,
            custom_significance_footer,
            ha="left",
            va="bottom",
            fontsize=5.8,
        )
    elif bool(spec.get("show_significance", True)) and any(row.get("significance") for row in data):
        figure.text(1.0, 1.0, "* p<0.05   ** p<0.01   *** p<0.001", ha="right", va="top", fontsize=6.2)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def render_plot_cli(
    argv: list[str] | None,
    *,
    schema: PlotSchema,
    description: str,
    prepare_spec: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> None:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = json.loads(args.data.read_text())
        spec = json.loads(args.spec.read_text())
        if not isinstance(spec, Mapping):
            raise TypeError("spec must be a JSON object")
        if prepare_spec is not None:
            spec = prepare_spec(spec)
        render_publication_plot(rows, spec, args.output, schema=schema)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Rendered {len(rows)} aggregate rows to {args.output}")


__all__ = [
    "BarStyleCallback",
    "FacetCallback",
    "FacetLayout",
    "PlotFacet",
    "PlotSchema",
    "PlotTheme",
    "ThemeCallback",
    "render_plot_cli",
    "render_publication_plot",
]
