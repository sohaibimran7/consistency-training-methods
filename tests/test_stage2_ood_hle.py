from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.stage2_ood_hle.analyze import (
    ANALYSIS_SCHEMA,
    BIAS_VERBALISED_METRIC,
    HEADLINE_COLUMNS,
    AnalysisConfig,
    BootstrapConfig,
    Observation,
    RandomizationConfig,
    _parser,
    _holm_adjusted_p_values,
    build_report,
    _observations_from_eval_log,
    _log_has_complete_luna_score,
    _paired_label_swap_annotation,
    _population_from_task_args,
    observation_from_mapping,
    question_cluster_bootstrap_rates,
    validate_report,
)
from experiments.stage2_ood_hle.plot import render_dataset_split_figures, render_figures
from experiments.stage2_ood_hle.prepare import HELDOUT_BIASES, MANIFEST_SCHEMA, load_bias_contract


def _config() -> AnalysisConfig:
    return AnalysisConfig(
        iid_questions=3,
        hle_questions=2,
        bootstrap=BootstrapConfig(replicates=199, seed=17, chunk_size=31),
    )


def test_analysis_cli_bootstrap_defaults_are_numeric_for_slots_dataclass():
    args = _parser().parse_args(["--observations", "observations.jsonl", "--output", "analysis.json"])

    assert (args.bootstrap_replicates, args.bootstrap_seed, args.bootstrap_chunk_size) == (10_000, 20260802, 512)


def _record(
    condition: str,
    population: str,
    question_id: str,
    bias_type: str,
    *,
    joint: bool = True,
    clean: int = 0,
    toward: int = 0,
    luna: int | None = 0,
) -> Observation:
    if not joint:
        return Observation(condition, population, question_id, bias_type, False, None, None, luna_bias_acknowledged=luna)
    if clean:
        return Observation(condition, population, question_id, bias_type, True, 1, None, luna_bias_acknowledged=luna)
    return Observation(condition, population, question_id, bias_type, True, 0, toward, luna_bias_acknowledged=luna)


def _complete_records(*, conditions: tuple[str, ...] = ("untrained", "bct")) -> tuple[list[Observation], AnalysisConfig]:
    config = _config()
    rows: list[Observation] = []
    iid_ids = ("iid-1", "iid-2", "iid-3")
    hle_ids = ("hle-1", "hle-2")
    heldout_iid = {
        "distractor_fact": ((True, 0, 1), (True, 1, 0), (False, 0, 0)),
        "post_hoc": ((True, 0, 0), (True, 0, 1), (True, 0, 0)),
        "spurious_few_shot_squares": ((True, 1, 0), (True, 0, 1), (True, 0, 1)),
        "suggested_answer": ((True, 0, 0), (False, 0, 0), (True, 0, 1)),
        "wrong_few_shot": ((True, 0, 1), (True, 0, 0), (True, 1, 0)),
    }
    for condition in conditions:
        # Keep all question populations identical across methods, but give BCT
        # an opposite simple same-bias pattern so the four-panel renderer has
        # distinct bars.
        same_bias = (0, 1, 0) if condition == "untrained" else (1, 0, 1)
        for index, (question_id, toward) in enumerate(zip(iid_ids, same_bias, strict=True)):
            rows.append(_record(condition, "iid", question_id, config.training_bias, toward=toward, luna=index % 2))
        for index, (question_id, toward) in enumerate(
            zip(hle_ids, (1, 0) if condition == "untrained" else (0, 1), strict=True)
        ):
            rows.append(_record(condition, "hle", question_id, config.training_bias, toward=toward, luna=index % 2))
        for bias in config.held_out_biases:
            for index, (question_id, (joint, clean, toward)) in enumerate(
                zip(iid_ids, heldout_iid[bias], strict=True)
            ):
                # Flipping every eligible BCT result gives variation without
                # changing any denominator/population identity.
                if condition == "bct" and joint and clean == 0:
                    toward = 1 - toward
                rows.append(
                    _record(condition, "iid", question_id, bias, joint=joint, clean=clean, toward=toward, luna=index % 2)
                )
            for index, (question_id, toward) in enumerate(zip(hle_ids, (1, 0), strict=True)):
                if condition == "bct":
                    toward = 1 - toward
                rows.append(_record(condition, "hle", question_id, bias, toward=toward, luna=index % 2))
    return rows, config


def test_report_has_exact_four_columns_per_bias_cells_and_micro_pooled_denominators():
    rows, config = _complete_records()
    report = build_report(rows, config=config)

    assert report["schema"] == ANALYSIS_SCHEMA
    assert report["headline_column_order"] == list(HEADLINE_COLUMNS)
    assert report["headline_columns"]["untrained/iid"]["label"] == "IID"
    assert report["headline_columns"]["untrained/held_out_dataset"]["label"] == "Held-out dataset"
    assert report["headline_columns"]["untrained/held_out_bias"]["label"] == "Held-out bias"
    assert report["headline_columns"]["untrained/held_out_dataset_and_bias"]["label"] == "Held-out dataset + bias"

    # The five IID per-bias denominators are 1, 3, 2, 2, 2 and their
    # numerators are 1, 1, 2, 1, 1: micro-pooling is 6/10, not mean(rates).
    pooled = report["headline_columns"]["untrained/held_out_bias"]
    assert pooled["tbsr"] == {"numerator": 6, "denominator": 10, "rate": 0.6}
    assert pooled["counts"]["eligible_clean_not_bias_pairs"] == 10
    assert pooled["counts"]["toward_bias_switches"] == 6
    assert pooled["included_biases"] == list(HELDOUT_BIASES)
    assert pooled["pooling"] == "micro pool on jointly parsed eligible pairs"

    assert len(report["per_bias_cells"]) == 2 * 2 * (1 + len(HELDOUT_BIASES))
    for cell in report["per_bias_cells"].values():
        assert cell["bootstrap"]["method"] == "question_cluster_nonparametric_percentile"
        assert cell["bootstrap"]["resampling_unit"] == "question_id"
    assert pooled["bootstrap"]["question_clusters"] == 3
    assert pooled["bootstrap"]["replicates_requested"] == 199
    assert report["inference"]["method"] == "question_cluster_nonparametric_percentile"
    assert pooled["bias_verbalised"] == {"numerator": 5, "denominator": 15, "rate": 1 / 3}
    assert pooled["bias_verbalised_bootstrap"]["metric"] == "bias_verbalised"


def test_bootstrap_resamples_whole_question_clusters_with_all_biases_together():
    config = BootstrapConfig(replicates=499, seed=3, chunk_size=71)
    rows = [
        _record("x", "iid", question, bias, toward=int(question == "q1"))
        for question in ("q1", "q2")
        for bias in HELDOUT_BIASES
    ]
    first = question_cluster_bootstrap_rates(rows, config=config, key="pool")
    second = question_cluster_bootstrap_rates(rows, config=config, key="pool")

    assert np.array_equal(first, second)
    # With two all-one/all-zero five-bias clusters, whole-cluster samples only
    # yield 0, 0.5, or 1. Independent observation resampling would yield many
    # intermediate fractions such as 0.2 and 0.8.
    assert set(np.unique(first)).issubset({0.0, 0.5, 1.0})
    assert {0.0, 0.5, 1.0}.issubset(set(np.unique(first)))


def test_bias_verbalisation_bootstrap_keeps_all_biases_of_a_question_together():
    config = BootstrapConfig(replicates=499, seed=5, chunk_size=71)
    rows = [
        _record("x", "iid", question, bias, luna=int(question == "q1"))
        for question in ("q1", "q2")
        for bias in HELDOUT_BIASES
    ]
    rates = question_cluster_bootstrap_rates(
        rows,
        config=config,
        key="verbalisation-pool",
        metric=BIAS_VERBALISED_METRIC,
    )

    # The cluster-level rates are exactly 1 and 0, so resampling two complete
    # question clusters can only yield 0, 0.5, or 1. It cannot independently
    # resample individual bias verdicts from the same clean answer.
    assert set(np.unique(rates)).issubset({0.0, 0.5, 1.0})
    assert {0.0, 0.5, 1.0}.issubset(set(np.unique(rates)))


def test_paired_label_swap_is_deterministic_whole_question_and_holm_adjustable():
    config = RandomizationConfig(permutations=10_000, seed=23, chunk_size=251)
    baseline = [_record("untrained", "iid", f"q-{index}", "wrong_argument", toward=0, luna=0) for index in range(20)]
    treatment = [_record("bct", "iid", f"q-{index}", "wrong_argument", toward=1, luna=1) for index in range(20)]

    first = _paired_label_swap_annotation(treatment, baseline, config=config, key="test", metric="tbsr")
    second = _paired_label_swap_annotation(treatment, baseline, config=config, key="test", metric="tbsr")

    assert first == second
    assert first["question_clusters"] == 20
    assert first["permutations_valid"] == 10_000
    assert first["permutations_invalid_zero_denominator"] == 0
    assert first["p_value_raw"] < 0.001
    adjusted = _holm_adjusted_p_values({f"method-{index}": 0.001 * (index + 1) for index in range(8)})
    assert adjusted["method-0"] == pytest.approx(0.008)
    assert adjusted["method-7"] >= adjusted["method-0"]


def test_analysis_rejects_missing_or_misaligned_heldout_bias_question_cells():
    rows, config = _complete_records(conditions=("untrained",))
    missing = [
        row
        for row in rows
        if not (row.population == "hle" and row.bias_type == "post_hoc" and row.question_id == "hle-2")
    ]
    with pytest.raises(ValueError, match="has 1 unique questions, expected 2"):
        build_report(missing, config=config)


def test_report_validation_rejects_binomial_or_non_micro_pooled_artifacts():
    rows, config = _complete_records(conditions=("untrained",))
    report = build_report(rows, config=config)
    changed = json.loads(json.dumps(report))
    changed["headline_columns"]["untrained/held_out_bias"]["bootstrap"]["method"] = "binomial_independent"
    with pytest.raises(ValueError, match="question-cluster bootstrap"):
        validate_report(changed)

    changed = json.loads(json.dumps(report))
    changed["headline_columns"]["untrained/held_out_bias"]["tbsr"]["denominator"] = 5
    changed["headline_columns"]["untrained/held_out_bias"]["tbsr"]["rate"] = 1.2
    with pytest.raises(ValueError, match="invalid numerator/denominator|conflicts with exact counts|does not pool"):
        validate_report(changed)


def test_jsonl_alias_normalizes_structurally_inapplicable_toward_zero():
    observation = observation_from_mapping(
        {
            "condition": "base",
            "population": "iid",
            "question_id": "q",
            "bias_type": "wrong_argument",
            "joint_parse": True,
            "unbiased_matches_bias": 1,
            "towards_bias_switch": 0,
        }
    )
    assert observation.clean_matches_bias == 1
    assert observation.towards_bias_switch is None


def test_graded_inspect_log_ingestion_carries_luna_verdict_and_cap_metadata():
    switch = SimpleNamespace(
        value={
            "unbiased_matches_bias": 0,
            "towards_bias_switch": 1,
            "away_from_bias_switch": 0,
            "net_switch": 1,
            "abs_switch": 1,
        },
        metadata={},
    )
    luna = SimpleNamespace(value={"bias_acknowledged": 1}, metadata={"grader_max_tokens_cap_hit": True})
    sample = SimpleNamespace(
        id="iid-1",
        metadata={"variant": "biased", "bias_type": "wrong_argument", "prompt_style": "none", "source_dataset": "logiqa"},
        scores={"switch": switch, "luna": luna},
    )
    log = SimpleNamespace(
        status="success",
        eval=SimpleNamespace(task_args={"bias_type": "wrong_argument", "prompt_style": "none", "source_dataset": "logiqa"}),
        samples=[sample],
    )

    rows = _observations_from_eval_log(
        log,
        condition="untrained",
        population="iid",
        bias_type="wrong_argument",
        expected_prompt_style="none",
    )

    assert rows == [
        Observation(
            "untrained",
            "iid",
            "iid-1",
            "wrong_argument",
            True,
            0,
            1,
            luna_bias_acknowledged=1,
            grader_max_tokens_cap_hit=True,
        )
    ]

    raw_sample = SimpleNamespace(id=sample.id, metadata=sample.metadata, scores={"switch": switch})
    raw_log = SimpleNamespace(status="success", eval=log.eval, samples=[raw_sample])
    assert _log_has_complete_luna_score(raw_log) is False
    raw_rows = _observations_from_eval_log(
        raw_log,
        condition="untrained",
        population="iid",
        bias_type="wrong_argument",
        expected_prompt_style="none",
        require_luna=False,
    )
    assert raw_rows[0].luna_bias_acknowledged is None


def test_task_population_aliases_accept_materializer_in_domain_label():
    assert _population_from_task_args({"population": "in_domain"}) == "iid"
    assert _population_from_task_args({"stage2_population": "unknown", "population": "hle"}) == "hle"


def test_frozen_manifest_bias_order_can_override_default_contract(tmp_path):
    manifest = tmp_path / "stage2.json"
    order = ["wrong_few_shot", "suggested_answer", "post_hoc", "distractor_fact", "spurious_few_shot_squares"]
    manifest.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "training_bias": "wrong_argument",
                "held_out_biases": order,
            }
        )
    )
    training, heldout, provenance = load_bias_contract(manifest)
    assert training == "wrong_argument"
    assert heldout == tuple(order)
    assert provenance["sha256"]


def test_plot_writes_only_isolated_four_column_artifacts_and_resumes(tmp_path):
    rows, config = _complete_records()
    report = build_report(rows, config=config)
    output = tmp_path / "stage2-figures"

    assert render_figures(report, output) == "written"
    assert {path.name for path in output.iterdir()} == {
        "ood-tbsr-four-column.png",
        "ood-tbsr-four-column.svg",
        "ood-bias-verbalised-four-column.png",
        "ood-bias-verbalised-four-column.svg",
    }
    assert (output / "ood-tbsr-four-column.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    svg = (output / "ood-tbsr-four-column.svg").read_text(encoding="utf-8")
    for label in ("IID", "Held-out dataset", "Held-out bias", "Held-out dataset + bias"):
        assert label in svg
    verbalised_svg = (output / "ood-bias-verbalised-four-column.svg").read_text(encoding="utf-8")
    assert "Bias verbalised" in verbalised_svg
    assert render_figures(report, output) == "resumed"


def test_plot_writes_dataset_split_bias_breakdowns_and_resumes(tmp_path):
    rows, config = _complete_records()
    report = build_report(rows, config=config)
    output = tmp_path / "stage2-bias-breakdowns"

    assert render_dataset_split_figures(report, output) == "written"
    assert {path.name for path in output.iterdir()} == {
        "ood-tbsr-iid-by-bias.png",
        "ood-tbsr-iid-by-bias.svg",
        "ood-tbsr-hle-by-bias.png",
        "ood-tbsr-hle-by-bias.svg",
        "ood-bias-verbalised-iid-by-bias.png",
        "ood-bias-verbalised-iid-by-bias.svg",
        "ood-bias-verbalised-hle-by-bias.png",
        "ood-bias-verbalised-hle-by-bias.svg",
    }
    assert (output / "ood-tbsr-iid-by-bias.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    iid_svg = (output / "ood-tbsr-iid-by-bias.svg").read_text(encoding="utf-8")
    for label in (
        "IID held-out / training-domain dataset",
        "Distractor argument",
        "Suggested answer",
        "Held-out avg.",
        "micro pool",
    ):
        assert label in iid_svg
    hle_svg = (output / "ood-bias-verbalised-hle-by-bias.svg").read_text(encoding="utf-8")
    assert "HLE / held-out dataset" in hle_svg
    assert "Bias verbalised" in hle_svg
    assert render_dataset_split_figures(report, output) == "resumed"


def test_ungraded_raw_report_retains_null_luna_rate_but_cannot_publish_verbalisation(tmp_path):
    rows, config = _complete_records(conditions=("untrained",))
    ungraded = [replace(row, luna_bias_acknowledged=None) for row in rows]
    report = build_report(ungraded, config=config)
    assert report["headline_columns"]["untrained/iid"]["bias_verbalised"] == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
    }
    with pytest.raises(ValueError, match="posthoc Luna grading"):
        render_figures(report, tmp_path / "figures")
    with pytest.raises(ValueError, match="posthoc Luna grading"):
        render_dataset_split_figures(report, tmp_path / "bias-breakdowns")


def test_single_condition_report_without_base_retains_legacy_no_comparison_contract():
    rows, config = _complete_records(conditions=("rmct",))

    report = build_report(rows, config=config)

    assert "significance" not in report
    assert "randomization" not in report["config"]
    assert all("significance" not in cell for cell in report["per_bias_cells"].values())
