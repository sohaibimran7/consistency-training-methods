from types import SimpleNamespace

import pytest

from ctm_data.adapters.mcq_bias.analysis import (
    aggregate_log_groups,
    aggregate_logs,
    aggregate_sycophancy_tradeoff,
    append_held_out_summary,
)


def _log(*, bias, dataset, created, values, status="success"):
    samples = [SimpleNamespace(scores={"mcq_bias_scorer": SimpleNamespace(value={"matches_bias": value})}) for value in values]
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

    assert rows[0] == {
        "condition": "base",
        "bias_type": "suggested_answer",
        "metric": "matches_bias",
        "mean": 2 / 3,
        "stderr": 1 / 3,
        "stderr_method": "sample",
        "n_replicates": 1,
        "n_scored": 3,
        "n_total": 4,
        "datasets": ["mmlu", "truthfulqa"],
    }
    assert rows[1]["condition"] == "rlct"
    assert rows[1]["mean"] == 0.25


def test_aggregate_logs_uses_latest_successful_attempt():
    old = _log(bias="suggested_answer", dataset="mmlu", created="1", values=[1.0])
    failed = _log(bias="suggested_answer", dataset="mmlu", created="3", values=[1.0], status="error")
    new = _log(bias="suggested_answer", dataset="mmlu", created="2", values=[0.0])
    rows = aggregate_logs({"base": [old, failed, new]}, metric="matches_bias")
    assert rows[0]["mean"] == 0.0
    assert rows[0]["n_total"] == 1


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


def test_held_out_summary_excludes_training_bias():
    rows = [
        {
            "condition": "rmct",
            "bias_type": bias,
            "metric": "towards_bias_switch",
            "mean": mean,
            "stderr": 0.1,
            "n_scored": 10,
            "n_total": 10,
            "datasets": ["hle"],
        }
        for bias, mean in (("wrong_argument", 0.8), ("suggested_answer", 0.2), ("post_hoc", 0.4))
    ]
    output = append_held_out_summary(rows, excluded_biases=["wrong_argument"])
    summary = output[-1]
    assert summary["bias_type"] == "held_out_mean"
    assert summary["mean"] == pytest.approx(0.3)
    assert summary["component_biases"] == ["post_hoc", "suggested_answer"]


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
