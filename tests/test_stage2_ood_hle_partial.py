"""Focused offline contracts for the exploratory partial Stage 2 renderer."""

from __future__ import annotations

from pathlib import Path

from experiments.stage2_ood_hle.analyze import AnalysisConfig, BootstrapConfig, Observation, build_report
from experiments.stage2_ood_hle.partial import build_partial_report, validate_partial_report
from experiments.stage2_ood_hle.partial_plot import render_partial_dataset_split_figures


def _record(
    condition: str, population: str, question_id: str, bias_type: str, *, toward: int, luna: int | None
) -> Observation:
    return Observation(
        condition=condition,
        population=population,  # type: ignore[arg-type] -- fixture uses the two public values
        question_id=question_id,
        bias_type=bias_type,
        joint_parse=True,
        clean_matches_bias=0,
        towards_bias_switch=toward,
        luna_bias_acknowledged=luna,
    )


def _rows(condition: str, config: AnalysisConfig, *, include_hle_heldout: bool) -> list[Observation]:
    rows: list[Observation] = []
    for population, count in (("iid", config.iid_questions), ("hle", config.hle_questions)):
        biases = [config.training_bias]
        if population == "iid" or include_hle_heldout:
            biases.extend(config.held_out_biases)
        for bias in biases:
            for index in range(count):
                # A parsed-null verdict is legal on a fully scored Luna log.
                luna = (
                    None
                    if condition == "bct" and population == "iid" and bias == config.training_bias and index == 0
                    else index % 2
                )
                rows.append(
                    _record(
                        condition,
                        population,
                        f"{population}-{index}",
                        bias,
                        toward=int(condition == "untrained"),
                        luna=luna,
                    )
                )
    return rows


def test_partial_report_plots_only_complete_cells_and_keeps_parsed_null_luna(tmp_path: Path):
    # Production requires 10k resamples; use the same count here to prevent a
    # test-only inference path from drifting away from the real renderer.
    config = AnalysisConfig(
        iid_questions=10,
        hle_questions=10,
        bootstrap=BootstrapConfig(replicates=10_000, seed=37, chunk_size=127),
    )
    canonical = build_report(
        _rows("untrained", config, include_hle_heldout=True) + _rows("act", config, include_hle_heldout=True),
        config=config,
    )
    bct_rows = _rows("bct", config, include_hle_heldout=False)
    # This is only 9/10, so it must remain an availability entry rather than a
    # bar in either metric.
    control_rows = [
        row
        for row in _rows("bct-control", config, include_hle_heldout=False)
        if not (row.population == "iid" and row.bias_type == config.training_bias and row.question_id == "iid-9")
    ]
    complete_keys = {
        f"bct/{population}/{bias}"
        for population in ("iid", "hle")
        for bias in ([config.training_bias, *config.held_out_biases] if population == "iid" else [config.training_bias])
    }
    report = build_partial_report(
        canonical,
        bct_rows + control_rows,
        luna_score_complete_cell_keys=complete_keys,
    )
    validate_partial_report(report)

    # IID BCT has all five held-out biases, so it gets a micro-pool.  HLE BCT
    # and every BCT-control category are intentionally incomplete.
    assert "bct/iid/held_out_mean" in report["held_out_summaries"]
    assert "bct/hle/held_out_mean" not in report["held_out_summaries"]
    assert "bct-control/iid/wrong_argument" not in report["cells"]
    assert report["availability"]["bct-control/iid/wrong_argument"]["status"] == "incomplete"

    parsed_null = report["cells"]["bct/iid/wrong_argument"]
    assert parsed_null["luna_score_complete"] is True
    assert parsed_null["counts"]["luna_parsed"] == 9
    assert parsed_null["bias_verbalised"]["denominator"] == 9
    assert parsed_null["significance"]["bias_verbalised"]["p_value"] is not None

    output = tmp_path / "partial-figures"
    assert render_partial_dataset_split_figures(report, output) == "written"
    assert {path.suffix for path in output.iterdir()} == {".png", ".svg"}
    assert len(list(output.iterdir())) == 8
    assert (output / "ood-tbsr-iid-by-bias-partial.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
