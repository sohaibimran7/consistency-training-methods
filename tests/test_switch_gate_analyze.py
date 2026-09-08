import json
from types import SimpleNamespace

import pytest

from experiments.switch_gate.analyze import (
    CONFIRMATION_COUNTS,
    HLE_ALL_BIASES,
    HLE_HELD_OUT_BIASES,
    MODELS,
    SCREEN_COUNTS,
    BootstrapConfig,
    Observation,
    _write_json,
    confirmation_analysis,
    holm_step_down,
    missingness_sensitivity_bounds,
    observations_from_log,
    screen_gate,
    studentized_bootstrap_evidence,
    training_ratio_estimate,
    validate_expected_hle,
    validate_expected_training,
    validate_hle_clusters,
)


def _observation(
    *,
    model=MODELS[0],
    question_id,
    dataset,
    bias="wrong_argument",
    parsed=True,
    clean_target=0,
    toward=0,
    away=0,
):
    if not parsed:
        return Observation(
            model=model,
            question_id=question_id,
            source_dataset=dataset,
            bias_type=bias,
            joint_parse=False,
            clean_target=None,
            toward=None,
            away=None,
            net_switch=None,
            abs_switch=None,
        )
    return Observation(
        model=model,
        question_id=question_id,
        source_dataset=dataset,
        bias_type=bias,
        clean_target=clean_target,
        toward=toward,
        away=away,
        net_switch=toward - away,
        abs_switch=int(bool(toward or away)),
    )


def _training_records(*, strong, model=MODELS[0]):
    rows = []
    for dataset, count in CONFIRMATION_COUNTS.items():
        for index in range(count):
            clean_target = int(index % 5 == 0)
            if strong:
                toward = int(not clean_target and index % 2 == 0)
                away = int(clean_target and index % 53 == 0)
            else:
                toward = int(not clean_target and index % 17 == 0)
                away = int(clean_target and index % 19 == 0)
            rows.append(
                _observation(
                    model=model,
                    question_id=f"{dataset}-{index}",
                    dataset=dataset,
                    clean_target=clean_target,
                    toward=toward,
                    away=away,
                )
            )
    return rows


def _hle_records(*, strong, model=MODELS[0], include_descriptive=True):
    rows = []
    biases = HLE_ALL_BIASES if include_descriptive else HLE_HELD_OUT_BIASES
    for bias_index, bias in enumerate(biases):
        for question in range(100):
            clean_target = int(question % 5 == 0)
            if strong:
                toward = int(not clean_target and (question + 2 * bias_index) % 2 == 0)
                away = int(clean_target and question % 47 == 0)
            else:
                toward = int(not clean_target and (question + bias_index) % 19 == 0)
                away = int(clean_target and (question + bias_index) % 17 == 0)
            rows.append(
                _observation(
                    model=model,
                    question_id=f"hle-{question}",
                    dataset="hle-text-mc",
                    bias=bias,
                    clean_target=clean_target,
                    toward=toward,
                    away=away,
                )
            )
    return rows


def test_training_estimates_use_fixed_population_poststratification():
    records = [
        *[_observation(question_id=f"l-{index}", dataset="logiqa", toward=1) for index in range(2)],
        *[_observation(question_id=f"h-{index}", dataset="hellaswag", toward=0) for index in range(8)],
    ]

    estimate = training_ratio_estimate(records, "T")

    assert estimate.estimate == pytest.approx(472 / 2048)
    assert estimate.estimate != pytest.approx(2 / 10)
    assert estimate.denominator == pytest.approx(1.0)


def test_screen_uses_only_coverage_and_pooled_toward_count():
    records = []
    row_number = 0
    for dataset, count in SCREEN_COUNTS.items():
        for index in range(count):
            parsed = row_number < 85
            records.append(
                _observation(
                    question_id=f"{dataset}-{index}",
                    dataset=dataset,
                    parsed=parsed,
                    toward=int(parsed and row_number < 4),
                )
            )
            row_number += 1

    report = screen_gate(records)

    assert report["advance"] is True
    assert report["overall"]["raw_counts"]["toward"] == 4
    assert report["overall"]["joint_parse_coverage"] == 0.85
    assert set(report["by_stratum"]) == {"logiqa", "hellaswag"}


def test_studentized_bootstrap_is_deterministic_one_sided_and_plus_one_corrected():
    records = _training_records(strong=True)
    config = BootstrapConfig(replicates=399, seed=20260729, chunk_size=73)

    first = studentized_bootstrap_evidence(records, "training", config)
    second = studentized_bootstrap_evidence(records, "training", config)

    assert first == second
    assert first["T"]["p_value"] >= 1 / 400
    assert first["D"]["p_value"] >= 1 / 400
    assert first["T"]["p_value"] < 0.05
    assert first["D"]["p_value"] < 0.05
    assert first["T"]["bootstrap_method"] == "stratified studentized bootstrap-t"


def test_hle_bootstrap_clusters_all_bias_observations_by_question():
    records = _hle_records(strong=True, include_descriptive=False)
    config = BootstrapConfig(replicates=299, seed=20260729, chunk_size=61)

    evidence = studentized_bootstrap_evidence(records, "hle", config)

    assert evidence["T"]["p_value"] < 0.05
    assert evidence["D"]["p_value"] < 0.05
    assert evidence["T"]["bootstrap_method"] == "question-cluster studentized bootstrap-t"
    assert evidence["T"]["bootstrap_replicates"] == 299


def test_holm_retains_all_six_cells_when_five_are_absent():
    first_cell = (MODELS[0], "training_wrong_argument")
    correction = holm_step_down({first_cell: 0.008})

    assert correction[first_cell]["holm_reject"] is True
    assert correction[first_cell]["holm_adjusted_p"] == pytest.approx(0.048)
    absent = correction[(MODELS[2], "hle_held_out")]
    assert absent["unadjusted_p"] is None
    assert absent["holm_adjusted_p"] is None
    assert absent["holm_reject"] is False
    assert absent["family_size"] == 6


def test_confirmation_can_be_partial_but_underpowered_hle_nonpass_never_supports_no_go():
    records = [*_training_records(strong=True), *_hle_records(strong=False)]

    report = confirmation_analysis(
        {MODELS[0]: records},
        training_powered=True,
        hle_decision_powered=False,
        bootstrap=BootstrapConfig(replicates=199, seed=20260729, chunk_size=50),
    )

    model = report["models"][MODELS[0]]
    assert model["targets"]["training_wrong_argument"]["gate_pass"] is True
    assert model["targets"]["hle_held_out"]["gate_pass"] is False
    assert model["targets"]["hle_held_out"]["adequately_powered_nonpass"] is False
    assert model["decision"] == "PARTIAL"
    assert model["hle_wrong_argument_descriptive_only"]["confirmatory"] is False
    assert report["models"][MODELS[1]]["decision"] == "NO DECISION"


def test_missingness_bounds_are_explicitly_nonconfirmatory():
    records = [
        _observation(question_id="l-1", dataset="logiqa", toward=1),
        _observation(question_id="l-2", dataset="logiqa", parsed=False),
        _observation(question_id="h-1", dataset="hellaswag", toward=0),
        _observation(question_id="h-2", dataset="hellaswag", parsed=False),
    ]

    bounds = missingness_sensitivity_bounds(records, design="training")

    assert bounds["confirmatory"] is False
    assert bounds["missing_weight_mass"] == pytest.approx(0.5)
    assert bounds["T"][0] <= bounds["T"][1]
    assert bounds["D"][0] <= bounds["D"][1]


def _fake_log(rows, *, model=MODELS[0]):
    samples = []
    for sample_id, values in rows:
        samples.append(
            SimpleNamespace(
                id=sample_id,
                metadata={
                    "source_dataset": "logiqa",
                    "bias_type": "wrong_argument",
                    "variant": "biased",
                    "prompt_style": "none",
                },
                scores={"switch_scorer": SimpleNamespace(value=values)},
            )
        )
    return SimpleNamespace(
        eval=SimpleNamespace(
            model=model,
            task_args={
                "dataset": "logiqa",
                "bias_type": "wrong_argument",
                "prompt_style": "none",
            },
        ),
        samples=samples,
    )


def _scores(*, clean=0, toward=0, away=None, net=0, absolute=0):
    return {
        "unbiased_matches_bias": clean,
        "towards_bias_switch": toward,
        "away_from_bias_switch": away,
        "net_switch": net,
        "abs_switch": absolute,
    }


def test_log_extraction_accepts_conditional_null_and_joint_parse_failure():
    log = _fake_log(
        [
            ("q1", _scores(clean=0, toward=1, away=None, net=1, absolute=1)),
            ("q2", _scores(clean=1, toward=None, away=1, net=-1, absolute=1)),
            ("q3", _scores(clean=None, toward=None, away=None, net=None, absolute=None)),
        ]
    )

    rows = observations_from_log(log, expected_model=MODELS[0])

    assert [(row.toward, row.away, row.joint_parse) for row in rows] == [
        (1, 0, True),
        (0, 1, True),
        (None, None, False),
    ]


def test_log_extraction_accepts_exact_provider_prefixed_model_only():
    log = _fake_log(
        [("q1", _scores())],
        model=f"tinker-sampling/{MODELS[0]}",
    )

    assert observations_from_log(log, expected_model=MODELS[0])[0].model == MODELS[0]
    with pytest.raises(ValueError, match="does not match requested model"):
        observations_from_log(log, expected_model=MODELS[1])


def test_log_extraction_rejects_partial_failure_and_duplicate_ids():
    partial = _fake_log([("q1", _scores(toward=None, away=None, net=None, absolute=0))])
    with pytest.raises(ValueError, match="partial joint-parse failure"):
        observations_from_log(partial)

    duplicated = _fake_log([("q1", _scores()), ("q1", _scores())])
    with pytest.raises(ValueError, match="duplicate question_id"):
        observations_from_log(duplicated)

    impossible_absolute = _fake_log([("q1", _scores(net=0, absolute=1))])
    with pytest.raises(ValueError, match=r"abs_switch must equal abs\(net_switch\)"):
        observations_from_log(impossible_absolute)


def test_hle_cluster_validation_rejects_duplicate_cells_and_clean_disagreement():
    first = _observation(
        question_id="hle-1",
        dataset="hle-text-mc",
        bias=HLE_HELD_OUT_BIASES[0],
        clean_target=0,
    )
    duplicate = _observation(
        question_id="hle-1",
        dataset="hle-text-mc",
        bias=HLE_HELD_OUT_BIASES[0],
        clean_target=0,
    )
    with pytest.raises(ValueError, match="duplicate HLE observation"):
        validate_hle_clusters([first, duplicate])

    disagreement = _observation(
        question_id="hle-1",
        dataset="hle-text-mc",
        bias=HLE_HELD_OUT_BIASES[1],
        clean_target=1,
    )
    with pytest.raises(ValueError, match="disagree on the shared clean target"):
        validate_hle_clusters([first, disagreement])


def test_expected_id_validation_fails_closed_on_incomplete_cells():
    training = _training_records(strong=False)
    expected_training = {
        dataset: {record.question_id for record in training if record.source_dataset == dataset}
        for dataset in CONFIRMATION_COUNTS
    }
    with pytest.raises(ValueError, match="do not match expected split"):
        validate_expected_training(training[:-1], expected_training)

    hle = _hle_records(strong=False)
    expected_hle = {f"hle-{question}" for question in range(100)}
    with pytest.raises(ValueError, match="IDs do not match expected split"):
        validate_expected_hle(hle[:-1], expected_hle)


def test_json_writer_refuses_overwrite_and_emits_stable_json(tmp_path):
    output = tmp_path / "analysis.json"
    _write_json(output, {"schema_version": "v", "value": 1}, force=False)
    assert json.loads(output.read_text()) == {"schema_version": "v", "value": 1}

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json(output, {"value": 2}, force=False)
