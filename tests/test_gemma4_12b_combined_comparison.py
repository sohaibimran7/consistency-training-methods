from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.gemma4_12b_base_eval import combined_comparison as comparison


def _source_rows(*, model: str, condition: str, metric: str, value_offset: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for population_index, population in enumerate(comparison.POPULATION_ORDER):
        datasets = ["logiqa", "hellaswag"] if population == "held_in_datasets" else ["hle-text-mc"]
        for bias_index, bias in enumerate(comparison.BIAS_ORDER):
            mean = min(0.95, 0.10 + value_offset + 0.01 * population_index + 0.002 * bias_index)
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "model_label": model,
                    "condition": condition,
                    "condition_label": "Source base",
                    "method": "none",
                    "is_control": False,
                    "training_biases": [],
                    "population": population,
                    "population_datasets": datasets,
                    "datasets": datasets,
                    "bias_type": bias,
                    "bias_status": "aggregate" if bias.endswith("_mean") else "seen",
                    "mean": mean,
                    "stderr": 0.03,
                    "ci_method": "wilson",
                    "ci_confidence": 0.95,
                    "ci_lower": max(0.0, mean - 0.04),
                    "ci_upper": min(1.0, mean + 0.04),
                    "n_scored": 50 + population_index,
                    "n_total": 50 + population_index,
                    "success_count": int(round((50 + population_index) * mean)),
                }
            )
    return rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture_sources(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path], Path]:
    qwen: dict[str, Path] = {}
    muse: dict[str, Path] = {}
    gemma = tmp_path / "gemma-publication"
    for metric in comparison.SUPPORTED_METRICS:
        qwen[metric] = _write_rows(
            tmp_path / "qwen" / metric / "chart-rows.json",
            _source_rows(model="qwen3.5-9b", condition="base_archived", metric=metric, value_offset=0.00),
        )
        muse[metric] = _write_rows(
            tmp_path / "muse" / metric / "chart-rows.json",
            _source_rows(model="muse-glimmer-30b", condition="base", metric=metric, value_offset=0.10),
        )
        _write_rows(
            gemma / f"{comparison.OUTPUT_STEMS[metric]}-chart-rows.json",
            _source_rows(model="gemma-4-12b-it", condition="base", metric=metric, value_offset=0.20),
        )
    return qwen, muse, gemma


def test_chart_rows_preserve_source_statistics_but_remove_cross_model_stars(tmp_path: Path) -> None:
    qwen, muse, gemma = _fixture_sources(tmp_path)
    rows = comparison.chart_rows(
        metric="towards_bias_switch",
        qwen_rows=qwen,
        muse_rows=muse,
        gemma_rows=gemma,
    )

    assert len(rows) == 3 * len(comparison.POPULATION_ORDER) * len(comparison.BIAS_ORDER)
    assert {row["condition"] for row in rows} == set(comparison.CONDITION_ORDER)
    assert {row["significance"] for row in rows} == {""}
    assert {row["p_value"] for row in rows} == {None}
    assert {row["significance_unavailable_reason"] for row in rows} == {
        comparison.NO_CROSS_MODEL_SIGNIFICANCE_REASON
    }
    qwen_row = next(
        row
        for row in rows
        if row["condition"] == "qwen_base"
        and row["population"] == "held_in_datasets"
        and row["bias_type"] == "wrong_argument"
    )
    assert qwen_row["n_scored"] == 50
    assert qwen_row["ci_method"] == "wilson"
    assert qwen_row["ci_lower"] == pytest.approx(0.06)
    assert qwen_row["ci_upper"] == pytest.approx(0.14)
    assert qwen_row["source_condition"] == "base_archived"


def test_render_bundle_has_both_metrics_and_visible_protocol_caveat(tmp_path: Path) -> None:
    qwen, muse, gemma = _fixture_sources(tmp_path)
    output = comparison.render_combined_base_comparison(
        qwen_rows=qwen,
        muse_rows=muse,
        gemma_rows=gemma,
        output_dir=tmp_path / "combined-publication",
    )

    expected = {
        "towards-bias-switch-rate-chart-rows.json",
        "towards-bias-switch-rate-chart-spec.json",
        "towards-bias-switch-rate.png",
        "towards-bias-switch-rate.svg",
        "bias-verbalisation-chart-rows.json",
        "bias-verbalisation-chart-spec.json",
        "bias-verbalisation.png",
        "bias-verbalisation.svg",
        "manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    verbalisation_svg = (output / "bias-verbalisation.svg").read_text(encoding="utf-8")
    assert "no cross-model paired significance tests or stars are claimed" in verbalisation_svg
    assert "256-token cap" in verbalisation_svg
    assert "uncapped" in verbalisation_svg
    assert "max_tokens=20480" in verbalisation_svg
    assert "54/1800 biased outputs hit it" in verbalisation_svg
    assert "9/1800 grades hit it; 1791 valid" in verbalisation_svg
    switch_rows = json.loads((output / "towards-bias-switch-rate-chart-rows.json").read_text(encoding="utf-8"))
    assert {row["significance"] for row in switch_rows} == {""}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["comparison_status"] == comparison.COMPARISON_STATUS
    assert manifest["cross_model_significance"]["claimed"] is False
    assert manifest["verbalisation_protocol_compatibility"]["qwen"]["luna_output_token_cap"] == 256
    assert manifest["verbalisation_protocol_compatibility"]["gemma"]["luna_output_token_cap"] is None
    assert manifest["included_models"] == ["qwen", "muse", "gemma"]
    assert set(manifest["models"]) == {"qwen", "muse", "gemma"}
    assert manifest["protocol_caveat"]["qwen_historical_generation"]["max_tokens"] == 20_480
    assert manifest["protocol_caveat"]["qwen_historical_generation"]["cap_hits"] == 54
    assert manifest["protocol_caveat"]["qwen_historical_luna"]["cap_hits"] == 9
    assert manifest["protocol_caveat"]["qwen_historical_luna"]["valid_grades"] == 1_791
    assert manifest["source_inputs"]["bias_acknowledged"]["qwen"]["selected_rows"] == 18
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        comparison.render_combined_base_comparison(
            qwen_rows=qwen,
            muse_rows=muse,
            gemma_rows=gemma,
            output_dir=output,
        )


def test_muse_is_optional_but_qwen_and_gemma_are_the_accurately_listed_two_model_output(tmp_path: Path) -> None:
    qwen, _muse, gemma = _fixture_sources(tmp_path)
    rows = comparison.chart_rows(
        metric="towards_bias_switch",
        qwen_rows=qwen,
        gemma_rows=gemma,
    )
    assert len(rows) == 2 * len(comparison.POPULATION_ORDER) * len(comparison.BIAS_ORDER)
    assert {row["condition"] for row in rows} == {"qwen_base", "gemma_base"}

    output = comparison.render_combined_base_comparison(
        qwen_rows=qwen,
        gemma_rows=gemma,
        output_dir=tmp_path / "two-model-publication",
    )
    spec = json.loads((output / "towards-bias-switch-rate-chart-spec.json").read_text(encoding="utf-8"))
    assert spec["included_models"] == ["qwen", "gemma"]
    assert spec["condition_order"] == ["qwen_base", "gemma_base"]
    assert "Muse" not in json.dumps(spec)
    assert "Qwen 3.5 9B and Gemma 4 12B" in spec["title"]
    switch_svg = (output / "towards-bias-switch-rate.svg").read_text(encoding="utf-8")
    assert "Qwen 3.5 9B base (historical)" in switch_svg
    assert "Gemma 4 12B base" in switch_svg
    assert "Muse Glimmer" not in switch_svg
    assert "max_tokens=20480" in switch_svg
    assert "54/1800 biased outputs hit it" in switch_svg
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["included_models"] == ["qwen", "gemma"]
    assert set(manifest["models"]) == {"qwen", "gemma"}
    assert set(manifest["source_inputs"]["towards_bias_switch"]) == {
        "included_models",
        "comparison_title",
        "qwen",
        "gemma",
    }
    assert manifest["protocol_caveat"]["gemma"]["generation_token_cap"] is None


def test_discovery_prefers_current_qwen_consistency_gap_artifact(tmp_path: Path) -> None:
    qwen_rows = _source_rows(
        model="qwen3.5-9b",
        condition="base_archived",
        metric="towards_bias_switch",
        value_offset=0.0,
    )
    exact = _write_rows(
        tmp_path
        / "rmct-step16-step176-standard-switch-rate-by-dataset-significance-key-r003-20260821"
        / "chart-rows.json",
        qwen_rows,
    )
    _write_rows(tmp_path / "another-complete-qwen-source" / "chart-rows.json", qwen_rows)

    assert comparison.discover_historical_rows(
        "qwen", metric="towards_bias_switch", artifact_root=tmp_path
    ) == exact.resolve()


def test_ambiguous_semantic_discovery_requires_an_explicit_muse_path(tmp_path: Path) -> None:
    rows = _source_rows(
        model="muse-glimmer-30b",
        condition="base",
        metric="bias_acknowledged",
        value_offset=0.1,
    )
    _write_rows(tmp_path / "muse-a" / "chart-rows.json", rows)
    _write_rows(tmp_path / "muse-b" / "chart-rows.json", rows)

    with pytest.raises(comparison.CombinedBaseComparisonError, match="multiple complete muse"):
        comparison.discover_historical_rows("muse", metric="bias_acknowledged", artifact_root=tmp_path)
