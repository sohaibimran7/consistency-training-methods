"""Offline chart-row adapter coverage for Stage 2 standard publication figures."""

from __future__ import annotations

from pathlib import Path

from experiments.stage2_ood_hle.analyze import AnalysisConfig, BootstrapConfig, Observation, RandomizationConfig, build_report
from experiments.stage2_ood_hle.publication import chart_rows, publication_spec, render_standard_figures


def _records() -> tuple[list[Observation], AnalysisConfig]:
    config = AnalysisConfig(
        iid_questions=3,
        hle_questions=2,
        bootstrap=BootstrapConfig(replicates=101, seed=9, chunk_size=31),
        randomization=RandomizationConfig(permutations=10_000, seed=11, chunk_size=127),
    )
    rows: list[Observation] = []
    for condition in ("untrained", "bct"):
        for population, question_count in (("iid", 3), ("hle", 2)):
            for bias in (config.training_bias, *config.held_out_biases):
                for index in range(question_count):
                    rows.append(
                        Observation(
                            condition=condition,
                            population=population,  # type: ignore[arg-type] -- fixed public population labels
                            question_id=f"{population}-{index}",
                            bias_type=bias,
                            joint_parse=True,
                            clean_matches_bias=0,
                            towards_bias_switch=int(condition == "bct"),
                            luna_bias_acknowledged=int(condition == "bct"),
                        )
                    )
    return rows, config


def test_standard_adapter_preserves_exact_bootstrap_bounds_and_holm_markers(tmp_path: Path):
    records, config = _records()
    report = build_report(records, config=config)

    rows = chart_rows(report, population="iid", metric="tbsr")
    assert len(rows) == 2 * 7  # training bias, five held-out biases, and the micro-pool
    source = report["per_bias_cells"]["bct/iid/wrong_argument"]
    adapted = next(row for row in rows if row["condition"] == "bct" and row["bias_type"] == "wrong_argument")
    assert adapted["ci_lower"] == source["bootstrap"]["ci_95"]["lower"]
    assert adapted["ci_upper"] == source["bootstrap"]["ci_95"]["upper"]
    assert adapted["significance"] == source["significance"]["tbsr"]["marker"]
    headline = report["headline_columns"]["bct/held_out_bias"]
    pooled = next(row for row in rows if row["condition"] == "bct" and row["bias_type"] == "held_out_mean")
    assert pooled["mean"] == headline["tbsr"]["rate"]
    assert pooled["ci_lower"] == headline["bootstrap"]["ci_95"]["lower"]

    spec = publication_spec(report, population="iid", metric="tbsr")
    assert "paired whole-question label-swap" in spec["significance_note"]
    assert "Holm-adjusted" in spec["significance_note"]
    assert spec["method_colors"]["opct"] == "#62a9a4"

    output = tmp_path / "standard"
    assert render_standard_figures(report, output) == "written"
    assert len(list(output.iterdir())) == 8
    svg = (output / "ood-tbsr-iid-by-bias.svg").read_text(encoding="utf-8")
    assert "paired whole-question label-swap" in svg
    assert "Held-out avg." in svg
