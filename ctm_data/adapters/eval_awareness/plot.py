"""Render EvalAwareBench factor/valence publication figures."""

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
    category_field="factor_set",
    category_label_field="factor_label",
    category_section="factors",
    category_order_key="factors",
    category_order_spec="factor_order",
    category_labels_spec="factor_labels",
    default_registry_path=Path(__file__).with_name("plot_registry.toml"),
    auto_column_field="valence",
    category_label_rotation=0.0,
    category_label_alignment="center",
)


def _defaults(spec: Mapping[str, object]) -> dict[str, object]:
    facet_labels = {"valence": {"safety": "Safety", "capability": "Capability"}}
    supplied = spec.get("facet_labels")
    if isinstance(supplied, Mapping):
        facet_labels.update(supplied)
    return {"zero_line": True, **spec, "facet_labels": facet_labels}


def render_publication_plot(
    rows: Sequence[Mapping[str, object]],
    spec: Mapping[str, object],
    output: Path,
    *,
    theme_callback: ThemeCallback | None = None,
    facet_callback: FacetCallback | None = None,
    bar_style_callback: BarStyleCallback | None = None,
) -> None:
    _render_publication_plot(
        rows,
        _defaults(spec),
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
        description="Render EvalAwareBench factor/valence plots from chart-ready JSON",
        prepare_spec=_defaults,
    )


if __name__ == "__main__":
    main()


__all__ = ["FacetLayout", "PlotFacet", "PlotTheme", "main", "render_publication_plot"]
