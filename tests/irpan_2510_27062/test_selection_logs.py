from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.irpan_2510_27062.analysis import SELECTION_OBSERVATION_SCHEMA, validate_selection_observation
from scripts.irpan_2510_27062.selection_logs import (
    SelectionLogError,
    collect_validation_observations,
    materialize_validation_observations,
)


def _candidate(domain: str, method: str, candidate_id: str) -> dict:
    return {
        "domain": domain,
        "method": method,
        "candidate_id": candidate_id,
        "candidate_locator": {"kind": "model", "value": f"fixture/{candidate_id}"},
        "candidate_details": {"method": method, "fixture": True},
    }


def _log(
    *,
    candidate: dict,
    created: str,
    benchmark: str,
    metric: str,
    metrics: dict,
    condition: str | None = None,
    stage: str = "validation",
    status: str = "success",
    scored_samples: int | None = None,
    unscored_samples: int | None = None,
    completed_samples: int = 10,
    samples: list | None = None,
    task: str | None = None,
) -> SimpleNamespace:
    task_metadata = {
        "selection_candidate": candidate,
        "stage": stage,
        "condition": condition,
    }
    if benchmark == "mmlu":
        task_metadata.update({"benchmark": benchmark, "primary_metric": metric})
    else:
        task_metadata.update({"source": benchmark, "metric": metric})
    return SimpleNamespace(
        status=status,
        eval=SimpleNamespace(
            created=created,
            metadata=task_metadata,
            task=task or ("mcq_bias/mcq_bias" if samples is not None else "legacy/summary-metric"),
        ),
        results=SimpleNamespace(
            completed_samples=completed_samples,
            scores=[
                SimpleNamespace(
                    metrics=metrics,
                    scored_samples=scored_samples,
                    unscored_samples=unscored_samples,
                )
            ],
        ),
        samples=samples,
    )


def _fake_runtime(monkeypatch, logs: dict[str, SimpleNamespace]) -> None:
    infos = [SimpleNamespace(name=name, mtime=index) for index, name in enumerate(reversed(logs))]

    def list_eval_logs(log_dir: str):
        assert log_dir == "fixture-logs"
        return infos

    def read_eval_log(info: SimpleNamespace):
        return logs[info.name]

    monkeypatch.setattr(
        "scripts.irpan_2510_27062.selection_logs._load_inspect_log_api",
        lambda: (list_eval_logs, read_eval_log),
    )


def _native_mcq_sample(*, variant: str, correct, matches_bias, answer_parsed: int) -> SimpleNamespace:
    metadata = {"variant": variant}
    if variant == "biased":
        metadata["bias_type"] = "suggested_answer"
    return SimpleNamespace(
        metadata=metadata,
        scores={
            "mcq_bias_scorer": SimpleNamespace(
                value={
                    "correct": correct,
                    "matches_bias": matches_bias,
                    "answer_parsed": answer_parsed,
                }
            )
        },
    )


def test_collects_latest_successful_exact_method_routes_and_sorts(monkeypatch) -> None:
    candidate = _candidate("sycophancy", "bct", "sycophancy:bct:configured")
    other_method = _candidate("sycophancy", "sft", "sycophancy:sft:configured")
    logs = {
        "old-clean.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 0.6}},
            scored_samples=10,
            unscored_samples=0,
        ),
        "latest-clean.eval": _log(
            candidate=candidate,
            created="2026-01-02T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 0.8}},
            scored_samples=10,
            unscored_samples=0,
        ),
        "failed-newer-clean.eval": _log(
            candidate=candidate,
            created="2026-01-03T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 1.0}},
            status="error",
            scored_samples=10,
            unscored_samples=0,
        ),
        "wrong-suggestion.eval": _log(
            candidate=candidate,
            created="2026-01-02T01:00:00Z",
            benchmark="mmlu",
            metric="followed_wrong_suggestion",
            condition="wrong_suggestion",
            metrics={"followed_wrong_suggestion": {"value": 0.2}},
            scored_samples=10,
            unscored_samples=0,
        ),
        "other-method.eval": _log(
            candidate=other_method,
            created="2026-01-04T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 0.9}},
            scored_samples=10,
            unscored_samples=0,
        ),
        "final.eval": _log(
            candidate=candidate,
            created="2026-01-05T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            stage="final",
            metrics={"mmlu_accuracy": {"value": 1.0}},
            scored_samples=10,
            unscored_samples=0,
        ),
    }
    _fake_runtime(monkeypatch, logs)

    rows = collect_validation_observations(
        "fixture-logs",
        domain="sycophancy",
        method="bct",
        schema=SELECTION_OBSERVATION_SCHEMA,
    )

    assert [(row["condition"], row["value"]) for row in rows] == [
        ("clean", 0.8),
        ("wrong_suggestion", 0.2),
    ]
    assert [(row["numerator"], row["denominator"], row["unscored_count"]) for row in rows] == [
        (8, 10, 0),
        (2, 10, 0),
    ]
    assert all(validate_selection_observation(row).schema == SELECTION_OBSERVATION_SCHEMA for row in rows)


def test_native_clean_scores_count_unparsed_as_incorrect(monkeypatch) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    logs = {
        "clean.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 1.0}},
            scored_samples=1,
            unscored_samples=1,
            completed_samples=2,
            samples=[
                _native_mcq_sample(variant="unbiased", correct=1.0, matches_bias=None, answer_parsed=1),
                _native_mcq_sample(variant="unbiased", correct=None, matches_bias=None, answer_parsed=0),
            ],
        )
    }
    _fake_runtime(monkeypatch, logs)

    row = collect_validation_observations("fixture-logs", domain="sycophancy", method="base")[0]

    assert row["status"] == "scored"
    assert (row["numerator"], row["denominator"], row["value"]) == (1, 2, 0.5)
    assert row["unscored_count"] == 0


def test_native_wrong_suggestion_scores_exclude_unparsed(monkeypatch) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    logs = {
        "wrong.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="followed_wrong_suggestion",
            condition="wrong_suggestion",
            metrics={"followed_wrong_suggestion": {"value": 0.5}},
            scored_samples=1,
            unscored_samples=1,
            completed_samples=2,
            samples=[
                _native_mcq_sample(variant="biased", correct=0.0, matches_bias=1.0, answer_parsed=1),
                _native_mcq_sample(variant="biased", correct=None, matches_bias=None, answer_parsed=0),
            ],
        )
    }
    _fake_runtime(monkeypatch, logs)

    row = collect_validation_observations("fixture-logs", domain="sycophancy", method="base")[0]

    assert row["status"] == "scored"
    assert (row["numerator"], row["denominator"], row["value"]) == (1, 1, 1.0)
    assert row["unscored_count"] == 1
    validate_selection_observation(row)


def test_native_wrong_suggestion_can_emit_non_sycophancy(monkeypatch) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    logs = {
        "wrong.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="non_sycophancy",
            condition="wrong_suggestion",
            metrics={"non_sycophancy": {"value": 0.0}},
            completed_samples=2,
            samples=[
                _native_mcq_sample(variant="biased", correct=0.0, matches_bias=1.0, answer_parsed=1),
                _native_mcq_sample(variant="biased", correct=1.0, matches_bias=0.0, answer_parsed=1),
            ],
        )
    }
    _fake_runtime(monkeypatch, logs)

    row = collect_validation_observations("fixture-logs", domain="sycophancy", method="base")[0]

    assert row["metric"] == "non_sycophancy"
    assert (row["numerator"], row["denominator"], row["value"]) == (1, 2, 0.5)


def test_native_sample_coverage_mismatch_fails_closed(monkeypatch) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    logs = {
        "clean.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 1.0}},
            completed_samples=2,
            samples=[_native_mcq_sample(variant="unbiased", correct=1.0, matches_bias=None, answer_parsed=1)],
        )
    }
    _fake_runtime(monkeypatch, logs)

    row = collect_validation_observations("fixture-logs", domain="sycophancy", method="base")[0]

    assert row["status"] == "unscored"
    assert "coverage mismatch" in row["unscored_reason"]


def test_safety_uses_named_mean_and_marks_any_unscored_coverage(monkeypatch) -> None:
    candidate = _candidate("jailbreak", "rmct", "jailbreak:rmct:configured")
    logs = {
        "harmbench.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="harmbench",
            metric="harmful_asr",
            metrics={
                "mean": {"value": 0.1},
                "sample_count": {"value": 10},
                "scored_count": {"value": 10},
                "unscored_count": {"value": 0},
                "unscored_fraction": {"value": 0.0},
            },
        ),
        "or-bench.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:01:00Z",
            benchmark="or_bench",
            metric="answered_benign",
            metrics={
                "mean": {"value": 0.9},
                "answered_benign": {"value": 0.2},
                "sample_count": {"value": 10},
                "scored_count": {"value": 9},
                "unscored_count": {"value": 1},
                "unscored_fraction": {"value": 0.1},
            },
        ),
    }
    _fake_runtime(monkeypatch, logs)

    rows = collect_validation_observations("fixture-logs", domain="jailbreak", method="rmct")

    assert [row["benchmark"] for row in rows] == ["harmbench", "or_bench"]
    assert rows[0]["value"] == 0.1
    assert (rows[0]["numerator"], rows[0]["denominator"]) == (1, 10)
    assert rows[1]["status"] == "unscored"
    assert rows[1]["value"] is None
    assert rows[1]["unscored_count"] == 1
    assert "unscored" in rows[1]["unscored_reason"]


@pytest.mark.parametrize(
    "metrics, reason",
    [
        ({}, "missing metric"),
        ({"mmlu_accuracy": {"value": float("nan")}}, "non-finite"),
        ({"mmlu_accuracy": {"value": 1.01}}, "outside"),
    ],
)
def test_absent_nonfinite_and_out_of_range_metrics_emit_unscored(monkeypatch, metrics, reason) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    logs = {
        "metric.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics=metrics,
            scored_samples=10,
            unscored_samples=0,
        )
    }
    _fake_runtime(monkeypatch, logs)

    row = collect_validation_observations("fixture-logs", domain="sycophancy", method="base")[0]

    assert row["status"] == "unscored"
    assert row["value"] is None
    assert reason in row["unscored_reason"]
    validate_selection_observation(row)


def test_latest_retry_timestamp_ties_are_rejected(monkeypatch) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    common = {
        "candidate": candidate,
        "created": "2026-01-01T00:00:00Z",
        "benchmark": "mmlu",
        "metric": "mmlu_accuracy",
        "condition": "clean",
        "metrics": {"mmlu_accuracy": {"value": 0.8}},
        "scored_samples": 10,
        "unscored_samples": 0,
    }
    _fake_runtime(monkeypatch, {"one.eval": _log(**common), "two.eval": _log(**common)})

    with pytest.raises(SelectionLogError, match="retry tie"):
        collect_validation_observations("fixture-logs", domain="sycophancy", method="base")


def test_materialized_jsonl_is_immutable(monkeypatch, tmp_path: Path) -> None:
    candidate = _candidate("sycophancy", "base", "sycophancy:base:configured")
    logs = {
        "clean.eval": _log(
            candidate=candidate,
            created="2026-01-01T00:00:00Z",
            benchmark="mmlu",
            metric="mmlu_accuracy",
            condition="clean",
            metrics={"mmlu_accuracy": {"value": 0.8}},
            scored_samples=10,
            unscored_samples=0,
        )
    }
    _fake_runtime(monkeypatch, logs)
    output = tmp_path / "observations.jsonl"

    rows = materialize_validation_observations(
        "fixture-logs",
        output,
        domain="sycophancy",
        method="base",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == rows[0]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_validation_observations(
            "fixture-logs",
            output,
            domain="sycophancy",
            method="base",
        )
