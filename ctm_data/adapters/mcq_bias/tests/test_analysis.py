from types import SimpleNamespace

import pytest

from ctm_data.adapters.mcq_bias.analysis import aggregate_logs


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


def test_aggregate_logs_rejects_incomplete_condition_matrix():
    first = [
        _log(bias="suggested_answer", dataset="mmlu", created="1", values=[1.0]),
        _log(bias="wrong_few_shot", dataset="mmlu", created="1", values=[1.0]),
    ]
    second = [_log(bias="suggested_answer", dataset="mmlu", created="1", values=[0.0])]
    with pytest.raises(ValueError, match="same evaluation tasks"):
        aggregate_logs({"base": first, "rlct": second}, metric="matches_bias")
