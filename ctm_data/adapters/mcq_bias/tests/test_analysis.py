from types import SimpleNamespace

import pytest

from ctm_data.adapters.mcq_bias.analysis import (
    aggregate_log_groups,
    aggregate_logs,
    aggregate_sycophancy_tradeoff,
    append_held_out_summary,
    append_percent_change,
    append_significance,
)


def _log(*, bias, dataset, created, values, status="success", source_args=None):
    samples = [
        SimpleNamespace(scores={"mcq_bias_scorer": SimpleNamespace(value={"matches_bias": value})}) for value in values
    ]
    return SimpleNamespace(
        status=status,
        eval=SimpleNamespace(
            created=created,
            task_args={
                "bias_type": bias,
                "dataset": dataset,
                "prompt_style": "none",
                "seed": "42",
                "n_questions": len(values),
                **(source_args or {}),
            },
        ),
        samples=samples,
    )


def _metric_log(*, bias, dataset, created, rows, status="success"):
    samples = [SimpleNamespace(scores={"scores": SimpleNamespace(value=value)}) for value in rows]
    return SimpleNamespace(
        status=status,
        eval=SimpleNamespace(
            created=created,
            task_args={
                "bias_type": bias,
                "dataset": dataset,
                "prompt_style": "none",
                "seed": "42",
                "n_questions": len(rows),
            },
        ),
        samples=samples,
    )


def _add_inspect_summary(log, *, metric, mean, stderr, n, mean_name="nanmean", stderr_name="nanstderr"):
    log.results = SimpleNamespace(
        scores=[
            SimpleNamespace(
                name=metric,
                scored_samples=n,
                metrics={
                    mean_name: SimpleNamespace(value=mean),
                    stderr_name: SimpleNamespace(value=stderr),
                },
            )
        ]
    )
    return log


def test_aggregate_logs_pools_datasets_and_keeps_scoring_denominator():
    logs = {
        "base": [
            _log(bias="suggested_answer", dataset="mmlu", created="1", values=[1.0, None]),
            _log(bias="suggested_answer", dataset="truthfulqa", created="1", values=[0.0, 1.0]),
        ],
        "rlct": [
            _log(bias="suggested_answer", dataset="mmlu", created="1", values=[0.0, 0.0]),
            _log(bias="suggested_answer", dataset="truthfulqa", created="1", values=[0.0, 1.0]),
        ],
    }

    rows = aggregate_logs(logs, metric="matches_bias")

    assert rows[0]["condition"] == "base"
    assert rows[0]["bias_type"] == "suggested_answer"
    assert rows[0]["metric"] == "matches_bias"
    assert rows[0]["mean"] == 2 / 3
    assert rows[0]["stderr"] == 1 / 3
    assert rows[0]["estimate_method"] == "sample_fallback"
    assert rows[0]["stderr_method"] == "pooled:sample_fallback"
    assert rows[0]["variant"] == "biased"
    assert rows[0]["n_replicates"] == 1
    assert rows[0]["n_scored"] == 3
    assert rows[0]["n_total"] == 4
    assert rows[0]["datasets"] == ["mmlu", "truthfulqa"]
    assert rows[1]["condition"] == "rlct"
    assert rows[1]["mean"] == 0.25


def test_aggregate_logs_uses_latest_successful_attempt():
    old = _log(bias="suggested_answer", dataset="mmlu", created="1", values=[1.0])
    failed = _log(bias="suggested_answer", dataset="mmlu", created="3", values=[1.0], status="error")
    new = _log(bias="suggested_answer", dataset="mmlu", created="2", values=[0.0])
    rows = aggregate_logs({"base": [old, failed, new]}, metric="matches_bias")
    assert rows[0]["mean"] == 0.0
    assert rows[0]["n_total"] == 1


def test_task_identity_keeps_distinct_dataset_specs():
    logs = [
        _log(
            bias="suggested_answer",
            dataset="org/custom",
            created="1",
            values=[1.0],
            source_args={"dataset_config": "a", "split": "validation", "revision": "abc"},
        ),
        _log(
            bias="suggested_answer",
            dataset="org/custom",
            created="1",
            values=[0.0],
            source_args={"dataset_config": "b", "split": "test", "revision": "def"},
        ),
    ]

    rows = aggregate_logs({"base": logs}, metric="matches_bias")

    assert rows[0]["mean"] == 0.5
    assert rows[0]["n_scored"] == 2
    assert rows[0]["n_total"] == 2


def test_replicate_groups_pool_scientific_runs_but_deduplicate_retries():
    first_group = [
        _log(bias="wrong_argument", dataset="hle", created="1", values=[0.0, 0.0]),
        _log(bias="wrong_argument", dataset="hle", created="2", values=[1.0, 0.0]),
    ]
    second_group = [_log(bias="wrong_argument", dataset="hle", created="1", values=[1.0, 1.0])]

    rows = aggregate_log_groups(
        {"rmct": [first_group, second_group]},
        metric="matches_bias",
        stderr="binomial",
    )

    assert rows[0]["mean"] == 0.75
    assert rows[0]["stderr"] == pytest.approx((0.75 * 0.25 / 4) ** 0.5)
    assert rows[0]["n_replicates"] == 2
    assert rows[0]["n_scored"] == 4
    assert rows[0]["n_total"] == 4


def test_aggregate_logs_rejects_incomplete_condition_matrix():
    first = [
        _log(bias="suggested_answer", dataset="mmlu", created="1", values=[1.0]),
        _log(bias="wrong_few_shot", dataset="mmlu", created="1", values=[1.0]),
    ]
    second = [_log(bias="suggested_answer", dataset="mmlu", created="1", values=[0.0])]
    with pytest.raises(ValueError, match="same evaluation tasks"):
        aggregate_logs({"base": first, "rlct": second}, metric="matches_bias")


def test_aggregate_logs_rejects_tasks_missing_from_every_condition():
    logs = {
        "base": [_log(bias="suggested_answer", dataset="mmlu", created="1", values=[1.0])],
        "rlct": [_log(bias="suggested_answer", dataset="mmlu", created="1", values=[0.0])],
    }
    with pytest.raises(ValueError, match="expected biases"):
        aggregate_logs(
            logs,
            metric="matches_bias",
            expected_biases=["suggested_answer", "wrong_few_shot"],
            expected_datasets=["mmlu"],
        )


def test_aggregate_logs_rejects_required_cell_without_finite_scores():
    logs = {
        "base": [_log(bias="suggested_answer", dataset="mmlu", created="1", values=[None])],
        "rlct": [_log(bias="suggested_answer", dataset="mmlu", created="1", values=[None])],
    }
    with pytest.raises(ValueError, match="no finite 'matches_bias' scores"):
        aggregate_logs(
            logs,
            metric="matches_bias",
            expected_biases=["suggested_answer"],
            expected_datasets=["mmlu"],
        )


def test_aggregate_logs_rejects_unscored_dataset_in_otherwise_scored_bias():
    logs = {
        condition: [
            _log(bias="suggested_answer", dataset="mmlu", created="1", values=[value]),
            _log(bias="suggested_answer", dataset="truthfulqa", created="1", values=[None]),
        ]
        for condition, value in (("base", 1.0), ("rlct", 0.0))
    }
    with pytest.raises(ValueError, match="no finite 'matches_bias' scores"):
        aggregate_logs(
            logs,
            metric="matches_bias",
            expected_biases=["suggested_answer"],
            expected_datasets=["mmlu", "truthfulqa"],
        )


def test_conditional_metric_uses_paper_switch_subset():
    logs = [
        _metric_log(
            bias="wrong_argument",
            dataset="hle-a",
            created="1",
            rows=[
                {"bias_acknowledged": 1.0, "towards_bias_switch": 1.0},
                {"bias_acknowledged": 0.0, "towards_bias_switch": 1.0},
                {"bias_acknowledged": 1.0, "towards_bias_switch": 0.0},
            ],
        ),
        _metric_log(
            bias="wrong_argument",
            dataset="hle-b",
            created="1",
            rows=[{"bias_acknowledged": 1.0, "towards_bias_switch": 1.0}],
        ),
    ]
    rows = aggregate_logs(
        {"rmct": logs},
        metric="bias_acknowledged",
        where_metric="towards_bias_switch",
        where_value=1.0,
    )
    assert rows[0]["condition"] == "rmct"
    assert rows[0]["mean"] == pytest.approx(2 / 3)
    assert rows[0]["n_scored"] == 3
    assert rows[0]["n_total"] == 4
    assert rows[0]["datasets"] == ["hle-a", "hle-b"]


def test_verbalisation_can_be_conditioned_on_total_bias_switch():
    log = _metric_log(
        bias="wrong_argument",
        dataset="hle",
        created="1",
        rows=[
            {"bias_acknowledged": 1.0, "abs_switch": 1.0},
            {"bias_acknowledged": 0.0, "abs_switch": 1.0},
            {"bias_acknowledged": 1.0, "abs_switch": 0.0},
        ],
    )

    rows = aggregate_logs(
        {"rate-matching": [log]},
        metric="bias_acknowledged",
        where_metric="abs_switch",
        where_value=1.0,
        stderr="binomial",
    )

    assert rows[0]["mean"] == 0.5
    assert rows[0]["n_scored"] == 2
    assert rows[0]["n_total"] == 3


def test_towards_bias_switch_excludes_ineligible_questions_from_denominator():
    log = _metric_log(
        bias="wrong_argument",
        dataset="hle",
        created="1",
        rows=[
            {"towards_bias_switch": 1.0},
            {"towards_bias_switch": 0.0},
            {"towards_bias_switch": None},
        ],
    )

    rows = aggregate_logs({"rmct": [log]}, metric="towards_bias_switch", stderr="binomial")

    assert rows[0]["mean"] == 0.5
    assert rows[0]["n_scored"] == 2
    assert rows[0]["n_total"] == 3


def test_unfiltered_metric_uses_inspect_mean_and_stderr_when_available():
    log = _add_inspect_summary(
        _metric_log(
            bias="wrong_argument",
            dataset="hle",
            created="1",
            rows=[{"towards_bias_switch": 1.0}, {"towards_bias_switch": 0.0}],
        ),
        metric="towards_bias_switch",
        mean=0.5,
        stderr=0.123,
        n=2,
    )

    row = aggregate_logs({"rmct": [log]}, metric="towards_bias_switch")[0]

    assert row["mean"] == 0.5
    assert row["stderr"] == 0.123
    assert row["estimate_method"] == "inspect"
    assert row["stderr_method"] == "inspect"
    assert row["source_mean_metric"] == "nanmean"
    assert row["source_stderr_metric"] == "nanstderr"


def test_inspect_metrics_may_use_package_qualified_names():
    log = _add_inspect_summary(
        _metric_log(
            bias="wrong_argument",
            dataset="hle",
            created="1",
            rows=[{"towards_bias_switch": 1.0}, {"towards_bias_switch": 0.0}],
        ),
        metric="towards_bias_switch",
        mean=0.5,
        stderr=0.321,
        n=2,
        mean_name="mcq_bias/nanmean",
        stderr_name="mcq_bias/nanstderr",
    )

    row = aggregate_logs({"rmct": [log]}, metric="towards_bias_switch")[0]

    assert row["stderr"] == 0.321
    assert row["source_mean_metric"] == "mcq_bias/nanmean"
    assert row["source_stderr_metric"] == "mcq_bias/nanstderr"


def test_inspect_scored_sample_count_is_preserved_with_inspect_estimates():
    log = _add_inspect_summary(
        _metric_log(
            bias="wrong_argument",
            dataset="hle",
            created="1",
            rows=[{"towards_bias_switch": 1.0}, {"towards_bias_switch": None}],
        ),
        metric="towards_bias_switch",
        mean=1.0,
        stderr=0.0,
        n=2,
    )

    row = aggregate_logs({"rmct": [log]}, metric="towards_bias_switch")[0]

    assert row["estimate_method"] == "inspect"
    assert row["n_scored"] == 2


def test_conditional_metric_marks_sample_derived_fallback_even_with_inspect_results():
    log = _add_inspect_summary(
        _metric_log(
            bias="wrong_argument",
            dataset="hle",
            created="1",
            rows=[
                {"bias_acknowledged": 1.0, "towards_bias_switch": 1.0},
                {"bias_acknowledged": 0.0, "towards_bias_switch": 0.0},
            ],
        ),
        metric="bias_acknowledged",
        mean=0.5,
        stderr=0.5,
        n=2,
    )
    row = aggregate_logs(
        {"rmct": [log]},
        metric="bias_acknowledged",
        where_metric="towards_bias_switch",
    )[0]
    assert row["mean"] == 1.0
    assert row["estimate_method"] == "sample_conditional"
    assert row["stderr_method"] == "sample_conditional"


def test_conditional_metric_omits_cells_with_no_matching_samples():
    empty = _metric_log(
        bias="wrong_argument",
        dataset="hle-a",
        created="1",
        rows=[{"bias_acknowledged": 1.0, "towards_bias_switch": 0.0}],
    )
    populated = _metric_log(
        bias="suggested_answer",
        dataset="hle-b",
        created="1",
        rows=[{"bias_acknowledged": 1.0, "towards_bias_switch": 1.0}],
    )
    rows = aggregate_logs(
        {"rmct": [empty, populated]},
        metric="bias_acknowledged",
        where_metric="towards_bias_switch",
    )
    assert [row["bias_type"] for row in rows] == ["suggested_answer"]


def test_arbitrary_compound_predicate_supports_inequality_and_boolean_logic():
    log = _metric_log(
        bias="wrong_argument",
        dataset="hle",
        created="1",
        rows=[
            {"correct": 1.0, "options_considered": 0.0, "bias_acknowledged": 0.0},
            {"correct": 0.0, "options_considered": 0.5, "bias_acknowledged": 1.0},
            {"correct": 0.0, "options_considered": 1.0, "bias_acknowledged": 0.0},
            {"correct": 1.0, "options_considered": 0.0, "bias_acknowledged": 1.0},
        ],
    )

    rows = aggregate_logs(
        {"base": [log]},
        metric="correct",
        where={
            "all": [
                {"metric": "options_considered", "op": "lt", "value": 1},
                {
                    "any": [
                        {"metric": "bias_acknowledged", "op": "eq", "value": 0},
                        {"not": {"metric": "correct", "op": "eq", "value": 1}},
                    ]
                },
            ]
        },
    )

    assert rows[0]["mean"] == 0.5
    assert rows[0]["n_scored"] == 2
    assert rows[0]["stderr_method"] == "sample_conditional"
    assert rows[0]["where"]["all"][0] == {"metric": "options_considered", "op": "lt", "value": 1.0}


def test_unbiased_variant_emits_clean_accuracy_row_and_metadata():
    clean = _add_inspect_summary(
        _metric_log(
            bias="",
            dataset="hle",
            created="1",
            rows=[{"correct": 1.0}, {"correct": 0.0}],
        ),
        metric="correct",
        mean=0.5,
        stderr=0.5,
        n=2,
    )
    biased = _metric_log(bias="wrong_argument", dataset="hle", created="1", rows=[{"correct": 0.0}, {"correct": 0.0}])
    row = aggregate_logs(
        {"bct": [clean, biased]},
        metric="correct",
        variant="unbiased",
        metadata={"model": "gpt-oss-20b", "training_biases": ["wrong_argument"]},
        condition_metadata={"bct": {"method": "bias_augmented_consistency", "is_control": False}},
    )[0]
    assert row["bias_type"] == "unbiased"
    assert row["variant"] == "unbiased"
    assert row["model"] == "gpt-oss-20b"
    assert row["training_biases"] == ["wrong_argument"]
    assert row["method"] == "bias_augmented_consistency"


def test_held_out_summary_excludes_training_bias():
    rows = [
        {
            "condition": "rmct",
            "bias_type": bias,
            "metric": "towards_bias_switch",
            "mean": mean,
            "stderr": 0.1,
            "n_scored": n,
            "n_total": n,
            "datasets": ["hle"],
        }
        for bias, mean, n in (
            ("wrong_argument", 0.8, 10),
            ("suggested_answer", 0.2, 10),
            ("post_hoc", 0.4, 30),
        )
    ]
    output = append_held_out_summary(rows, excluded_biases=["wrong_argument"])
    summary = output[-1]
    assert summary["bias_type"] == "held_out_mean"
    assert summary["mean"] == pytest.approx(0.35)
    assert summary["n_scored"] == 40
    assert summary["sample_count_weighted"] is True
    assert summary["component_biases"] == ["post_hoc", "suggested_answer"]


def test_held_out_summary_is_separate_for_each_training_bias_facet():
    rows = []
    for training_bias, means in (
        ("wrong_argument", {"wrong_argument": 0.9, "suggested_answer": 0.2}),
        ("suggested_answer", {"wrong_argument": 0.4, "suggested_answer": 0.8}),
    ):
        for bias, mean in means.items():
            rows.append(
                {
                    "condition": "rmct",
                    "bias_type": bias,
                    "metric": "matches_bias",
                    "mean": mean,
                    "stderr": 0.1,
                    "n_scored": 10,
                    "n_total": 10,
                    "datasets": ["hle"],
                    "model": "model",
                    "training_regime": training_bias,
                    "training_biases": [training_bias],
                }
            )

    summaries = [row for row in append_held_out_summary(rows) if row["bias_type"] == "held_out_mean"]

    assert len(summaries) == 2
    by_regime = {row["training_regime"]: row for row in summaries}
    assert by_regime["wrong_argument"]["component_biases"] == ["suggested_answer"]
    assert by_regime["wrong_argument"]["mean"] == 0.2
    assert by_regime["suggested_answer"]["component_biases"] == ["wrong_argument"]
    assert by_regime["suggested_answer"]["mean"] == 0.4


def test_percent_change_uses_input_standard_errors_and_tests_against_zero():
    rows = [
        {
            "condition": "untrained",
            "bias_type": "wrong_argument",
            "metric": "matches_bias",
            "mean": 0.2,
            "stderr": 0.02,
            "n_scored": 100,
            "stderr_method": "inspect",
        },
        {
            "condition": "rmct",
            "bias_type": "wrong_argument",
            "metric": "matches_bias",
            "mean": 0.3,
            "stderr": 0.03,
            "n_scored": 80,
            "stderr_method": "inspect",
        },
    ]

    transformed = append_percent_change(rows, baseline_condition="untrained")

    assert len(transformed) == 1
    assert transformed[0]["metric"] == "matches_bias_percent_change"
    assert transformed[0]["mean"] == pytest.approx(50.0)
    assert transformed[0]["stderr"] == pytest.approx(21.2132034356)
    assert transformed[0]["stderr_method"] == "delta:inspect+inspect"
    assert transformed[0]["baseline_n_scored"] == 100
    assert transformed[0]["p_value"] < 0.05
    assert transformed[0]["significance"] == "*"


def test_significance_is_precomputed_against_matching_baseline():
    rows = [
        {"condition": "untrained", "bias_type": "wrong_argument", "mean": 0.5, "n_scored": 100},
        {"condition": "rmct", "bias_type": "wrong_argument", "mean": 0.8, "n_scored": 100},
    ]
    output = append_significance(rows, baseline_condition="untrained")
    assert output[0]["p_value"] is None
    assert output[0]["significance"] == ""
    assert output[1]["p_value"] < 0.001
    assert output[1]["significance"] == "***"


def test_significance_is_marked_unavailable_when_smoke_has_no_baseline_cells():
    rows = [{"condition": "rmct", "bias_type": "wrong_argument", "mean": 0.5, "n_scored": 2}]
    output = append_significance(rows, baseline_condition="untrained")
    assert output[0]["p_value"] is None
    assert output[0]["significance_unavailable_reason"] == "baseline_condition_absent"


def test_sycophancy_tradeoff_combines_biased_and_clean_tasks():
    biased = _metric_log(
        bias="suggested_answer",
        dataset="mmlu",
        created="1",
        rows=[{"matches_bias": 0.0}, {"matches_bias": 1.0}, {"matches_bias": 0.0}],
    )
    clean = _metric_log(
        bias="",
        dataset="mmlu",
        created="1",
        rows=[{"correct": 1.0}, {"correct": 1.0}, {"correct": 0.0}],
    )
    rows = aggregate_sycophancy_tradeoff({"act": [biased, clean]})
    assert rows[0]["not_sycophantic"] == pytest.approx(2 / 3)
    assert rows[0]["clean_accuracy"] == pytest.approx(2 / 3)
    assert rows[0]["f1"] == pytest.approx(2 / 3)


def test_sycophancy_tradeoff_requires_same_wrong_option_seed_set():
    biased_a = _metric_log(
        bias="suggested_answer",
        dataset="mmlu",
        created="1",
        rows=[{"matches_bias": 0.0}],
    )
    biased_a.eval.task_args["wrong_option_seed"] = "a"
    biased_b = _metric_log(
        bias="suggested_answer",
        dataset="mmlu",
        created="1",
        rows=[{"matches_bias": 0.0}],
    )
    biased_b.eval.task_args["wrong_option_seed"] = "b"
    clean_a = _metric_log(
        bias="",
        dataset="mmlu",
        created="1",
        rows=[{"correct": 1.0}],
    )
    clean_b = _metric_log(
        bias="",
        dataset="mmlu",
        created="1",
        rows=[{"correct": 1.0}],
    )

    with pytest.raises(ValueError, match="same biased task set"):
        aggregate_sycophancy_tradeoff({"first": [biased_a, clean_a], "second": [biased_b, clean_b]})


def test_sycophancy_tradeoff_rejects_multiple_seed_variants_for_one_clean_task():
    biased_a = _metric_log(
        bias="suggested_answer",
        dataset="mmlu",
        created="1",
        rows=[{"matches_bias": 0.0}],
    )
    biased_a.eval.task_args["wrong_option_seed"] = "a"
    biased_b = _metric_log(
        bias="suggested_answer",
        dataset="mmlu",
        created="1",
        rows=[{"matches_bias": 1.0}],
    )
    biased_b.eval.task_args["wrong_option_seed"] = "b"
    clean = _metric_log(
        bias="",
        dataset="mmlu",
        created="1",
        rows=[{"correct": 1.0}],
    )

    with pytest.raises(ValueError, match="multiple wrong-option seeds"):
        aggregate_sycophancy_tradeoff({"condition": [biased_a, biased_b, clean]})
