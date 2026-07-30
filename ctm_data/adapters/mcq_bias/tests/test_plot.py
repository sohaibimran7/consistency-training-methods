from pathlib import Path

import pytest

from ctm_data.adapters.mcq_bias.plot import FacetLayout, PlotFacet, PlotTheme, render_publication_plot


def _rows():
    rows = []
    for condition, method, control, adjustment in (
        ("untrained", "none", False, 0.0),
        ("rmct", "rate_matching", False, 0.2),
        ("rmct-control", "rate_matching", True, 0.05),
    ):
        for bias, mean in (("wrong_argument", 0.35), ("suggested_answer", 0.20), ("held_out_mean", 0.25)):
            rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "is_control": control,
                    "bias_type": bias,
                    "metric": "towards_bias_switch",
                    "mean": mean + adjustment,
                    "stderr": 0.025,
                    "n_scored": 100,
                    "model": "gpt-oss-20b",
                    "training_biases": ["wrong_argument"],
                    "significance": "**" if condition == "rmct" else "",
                }
            )
    return rows


def test_publication_renderer_writes_svg_from_chart_ready_rows(tmp_path: Path):
    output = tmp_path / "figure.svg"
    render_publication_plot(
        _rows(),
        {
            "ylabel": "Towards-bias switch rate",
            "condition_order": ["untrained", "rmct", "rmct-control"],
            "condition_labels": {"untrained": "Base", "rmct": "RMCT", "rmct-control": "RMCT Control"},
            "bias_order": ["wrong_argument", "suggested_answer", "held_out_mean"],
            "bias_labels": {
                "wrong_argument": "Distractor argument",
                "suggested_answer": "Suggested answer",
                "held_out_mean": "Held-out avg.",
            },
            "model_labels": {"gpt-oss-20b": "OpenAI GPT OSS 20B"},
            "ylim": [0, 1.05],
        },
        output,
    )
    content = output.read_text()
    assert content.startswith("<?xml")
    assert "Towards-bias switch rate" in content
    assert output.stat().st_size > 10_000


def test_publication_renderer_refuses_to_overwrite(tmp_path: Path):
    output = tmp_path / "figure.svg"
    output.write_text("existing")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        render_publication_plot(_rows(), {}, output)


def test_renderer_can_label_sample_counts_and_use_registry_defaults(tmp_path: Path):
    output = tmp_path / "counts.svg"

    render_publication_plot(
        _rows(),
        {"sample_labels": "n_scored", "bias_order": ["wrong_argument", "suggested_answer", "held_out_mean"]},
        output,
    )

    content = output.read_text()
    assert "n=100" in content
    assert "OpenAI GPT OSS 20B" in content
    assert "Held-out avg." in content


def test_renderer_auto_facets_distinct_training_bias_sets(tmp_path: Path):
    rows = []
    for training_bias, shift in (("wrong_argument", 0.0), ("suggested_answer", 0.1)):
        for row in _rows():
            rows.append(
                {
                    **row,
                    "mean": min(row["mean"] + shift, 0.95),
                    "training_biases": [training_bias],
                    "training_regime": training_bias,
                }
            )
    output = tmp_path / "facets.svg"

    render_publication_plot(rows, {"show_significance": False}, output)

    content = output.read_text()
    assert "Training: Distractor argument" in content
    assert "Training: Suggested answer" in content
    assert output.stat().st_size > 20_000


def test_renderer_handles_negative_percent_change_and_zero_line(tmp_path: Path):
    rows = [
        {
            **row,
            "condition": "rmct",
            "metric": "matches_bias_percent_change",
            "mean": -25.0 if row["bias_type"] == "wrong_argument" else 40.0,
            "stderr": 5.0,
            "transform": "percent_change",
        }
        for row in _rows()
        if row["condition"] == "rmct"
    ]
    output = tmp_path / "ratio.svg"

    render_publication_plot(rows, {"sample_labels": True}, output)

    content = output.read_text()
    assert "−25%" in content or "-25%" in content
    assert "n=100" in content


def test_renderer_accepts_theme_facet_and_bar_style_callbacks(tmp_path: Path):
    calls = {"theme": 0, "facet": 0, "bar": 0}

    def theme_callback(theme, spec):
        assert isinstance(theme, PlotTheme)
        calls["theme"] += 1
        return {"trained_background": "#ffeeee", "panel_title_fontsize": 9}

    def facet_callback(rows, spec, registry):
        calls["facet"] += 1
        return FacetLayout(((PlotFacet("Custom facet", tuple(rows)),),))

    def bar_style_callback(row, style):
        calls["bar"] += 1
        return {**style, "alpha": 0.9}

    output = tmp_path / "callbacks.svg"
    render_publication_plot(
        _rows(),
        {},
        output,
        theme_callback=theme_callback,
        facet_callback=facet_callback,
        bar_style_callback=bar_style_callback,
    )

    assert calls["theme"] == 1
    assert calls["facet"] == 1
    assert calls["bar"] > len(_rows())
    assert "Custom facet" in output.read_text()
