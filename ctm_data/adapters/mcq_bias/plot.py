"""Render publication MCQ-bias figures from chart-ready JSON."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ctm_data.adapters._plot import (
    BarStyleCallback,
    FacetCallback,
    FacetLayout,
    PlotFacet,
    PlotSchema,
    PlotTheme,
    ThemeCallback,
    render_plot_cli,
)
from ctm_data.adapters._plot import render_publication_plot as _render_publication_plot

_SCHEMA = PlotSchema(
    category_field="bias_type",
    category_label_field="bias_label",
    category_section="biases",
    category_order_key="biases",
    category_order_spec="bias_order",
    category_labels_spec="bias_labels",
    default_registry_path=Path(__file__).with_name("plot_registry.toml"),
    training_categories_field="training_biases",
    auto_column_field="training_biases",
    held_out_label="held_out_mean",
)


def render_publication_plot(
    rows: Sequence[Mapping[str, object]],
    spec: Mapping[str, object],
    output: Path,
    *,
    theme_callback: ThemeCallback | None = None,
    facet_callback: FacetCallback | None = None,
    bar_style_callback: BarStyleCallback | None = None,
) -> None:
    """Render the established MCQ-bias grouped-bar design."""

    _render_publication_plot(
        rows,
        spec,
        output,
        schema=_SCHEMA,
        theme_callback=theme_callback,
        facet_callback=facet_callback,
        bar_style_callback=bar_style_callback,
    )


def main(argv: list[str] | None = None) -> None:
    render_plot_cli(
        argv,
        schema=_SCHEMA,
        description="Render publication MCQ-bias plots from chart-ready JSON",
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FacetLayout",
    "PlotFacet",
    "PlotTheme",
    "main",
    "render_publication_plot",
]
