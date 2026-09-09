from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from ctm_data.adapters.eval_awareness.plot import render_publication_plot as render_evalaware
from ctm_data.adapters.wildjailbreak.plot import render_publication_plot as render_wildjailbreak


def _row(*, condition, metric, mean, **dimensions):
    return {
        "condition": condition,
        "method": "none" if condition == "base" else "rate_matching",
        "metric": metric,
        "mean": mean,
        "stderr": 0.05,
        "n_scored": 100,
        "model": "model-a",
        **dimensions,
    }


def test_evalaware_renderer_uses_factor_axis_and_valence_facets(tmp_path: Path):
    rows = [
        _row(
            condition=condition,
            metric="net_refusal_switch",
            mean=mean,
            factor_set="F6",
            factor_label="F6",
            valence=valence,
        )
        for condition, mean in (("base", 0.2), ("trained", 0.05))
        for valence in ("safety", "capability")
    ]
    output = tmp_path / "evalaware.svg"

    render_evalaware(rows, {"ylim": [-1.05, 1.05], "show_significance": False}, output)

    content = output.read_text()
    assert content.startswith("<?xml")
    assert "F6" in content
    assert "Safety" in content
    assert "Capability" in content


def test_wildjailbreak_renderer_uses_valence_axis(tmp_path: Path):
    rows = [
        _row(
            condition=condition,
            metric="net_refusal_switch",
            mean=mean,
            valence=valence,
            valence_label=valence.title(),
        )
        for condition, mean in (("base", 0.2), ("trained", 0.05))
        for valence in ("harmful", "benign")
    ]
    output = tmp_path / "wildjailbreak.svg"

    render_wildjailbreak(rows, {"ylim": [-1.05, 1.05], "show_significance": False}, output)

    content = output.read_text()
    assert content.startswith("<?xml")
    assert "Harmful" in content
    assert "Benign" in content


def test_wildjailbreak_renderer_can_use_tactic_axis_and_valence_facets(tmp_path: Path):
    rows = [
        _row(
            condition=condition,
            metric="net_refusal_switch",
            mean=mean,
            tactic=tactic,
            tactic_label=tactic.title(),
            valence="harmful",
        )
        for condition, mean in (("base", 0.2), ("trained", 0.05))
        for tactic in ("roleplay", "encoding")
    ]
    output = tmp_path / "wildjailbreak-tactics.svg"

    render_wildjailbreak(
        rows,
        {
            "category_field": "tactic",
            "facet": {"rows": "model", "columns": "valence"},
            "ylim": [-1.05, 1.05],
            "show_significance": False,
        },
        output,
    )

    content = output.read_text()
    assert "Roleplay" in content
    assert "Encoding" in content
