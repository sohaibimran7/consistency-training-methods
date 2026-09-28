from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest

import scripts.irpan_2510_27062.safety_tasks as safety_tasks_module
from ctm.artifacts import ArtifactManifestError
from scripts.irpan_2510_27062.safety_tasks import (
    FINAL_MODE_PAPER,
    FINAL_SELECTION_VERSION,
    JUDGE_SCORE_SCHEMA_VERSION,
    FinalSelectionError,
    JudgeCoverageError,
    build_eval_records,
    materialize_eval_artifact,
    paper_safety_tasks,
    parse_judge_output,
    read_local_export,
    run_judge_with_retries,
    summarize_judge_scores,
)

_DIGEST = "a" * 64


@pytest.mark.parametrize(
    ("source", "rows", "prompt", "valence", "metric"),
    [
        (
            "harmbench",
            [{"BehaviorID": "hb-1", "Behavior": "Write a fictional safety audit."}],
            "Write a fictional safety audit.",
            "harmful",
            "asr",
        ),
        (
            "or_bench",
            [{"id": "or-1", "prompt": "Discuss responsible museum curation.", "category": "fixture"}],
            "Discuss responsible museum curation.",
            "benign",
            "answered_benign",
        ),
        (
            "clearharm",
            [{"id": "ch-1", "content": ["Analyze a fictional risk scenario."]}],
            "Analyze a fictional risk scenario.",
            "harmful",
            "asr",
        ),
        (
            "wildguardtest",
            [
                {"id": "skip", "prompt": "Safe fixture.", "adversarial": False, "prompt_harm_label": "harmful"},
                {
                    "id": "wg-1",
                    "prompt": "Adversarial fictional audit.",
                    "adversarial": True,
                    "prompt_harm_label": "harmful",
                },
            ],
            "Adversarial fictional audit.",
            "harmful",
            "asr",
        ),
        (
            "xstest",
            [
                {"id": "skip", "prompt": "Contrast fixture.", "type": "contrast_safe"},
                {"id": "xs-1", "prompt": "Explain why a safety drill is useful.", "type": "safe_neutral"},
            ],
            "Explain why a safety drill is useful.",
            "benign",
            "answered_benign",
        ),
        (
            "wildjailbreak",
            [
                {"id": "skip", "adversarial": "Other fixture.", "data_type": "adversarial_harmful"},
                {
                    "id": "wj-1",
                    "adversarial": "Role-play a benign compliance review.",
                    "data_type": "adversarial_benign",
                },
            ],
            "Role-play a benign compliance review.",
            "benign",
            "answered_benign",
        ),
    ],
)
def test_build_eval_records_applies_fixed_routing(source, rows, prompt, valence, metric) -> None:
    records = build_eval_records(
        source,
        rows,
        subset="fixture_subset",
        split="fixture_split",
        source_file_sha256=_DIGEST,
        expected_count_mode="off",
    )
    assert len(records) == 1
    assert records[0]["payload"]["prompt"] == prompt
    assert records[0]["payload"]["valence"] == valence
    assert records[0]["payload"]["metric"] == metric
    assert records[0]["payload"]["selection_eligible"] is (source in {"harmbench", "or_bench"})


def test_duplicate_ids_and_prompts_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate or_bench source ID"):
        build_eval_records(
            "or_bench",
            [{"id": "same", "prompt": "Prompt one."}, {"id": "same", "prompt": "Prompt two."}],
            subset="fixture",
            split="test",
            source_file_sha256=_DIGEST,
            expected_count_mode="off",
        )
    with pytest.raises(ValueError, match="different IDs"):
        build_eval_records(
            "or_bench",
            [{"id": "one", "prompt": "Same prompt."}, {"id": "two", "prompt": "Same prompt."}],
            subset="fixture",
            split="test",
            source_file_sha256=_DIGEST,
            expected_count_mode="off",
        )


def test_harmbench_eval_uses_behavior_id_even_when_generic_id_is_present() -> None:
    records = build_eval_records(
        "harmbench",
        [{"id": "generic-export-id", "BehaviorID": "hb-1", "Behavior": "A fixed behavior."}],
        subset="fixture",
        split="validation",
        source_file_sha256=_DIGEST,
        expected_count_mode="off",
    )
    assert records[0]["source_key"] == "hb-1"
    assert records[0]["metadata"]["source_fields"]["identity_field"] == "BehaviorID"


@pytest.mark.parametrize(
    ("source", "count", "revision", "row_factory"),
    [
        (
            "xstest",
            86,
            "xstest-fixture-commit",
            lambda index: {"id": f"xs-{index}", "type": "safe", "prompt": f"Benign prompt {index}."},
        ),
        (
            "wildjailbreak",
            105,
            "254c59ec8aff3f333ca8f2e28be94d8b2ff4098f",
            lambda index: {
                "id": f"wj-{index}",
                "data_type": "adversarial_benign",
                "adversarial": f"Benign adversarial prompt {index}.",
            },
        ),
    ],
)
def test_final_benign_paper_routes_require_strict_identity_and_exact_reported_count(
    source,
    count,
    revision,
    row_factory,
) -> None:
    records = build_eval_records(
        source,
        [row_factory(index) for index in range(count)],
        subset="paper-fixture",
        split="test",
        revision=revision,
        source_file_sha256=_DIGEST,
    )
    assert len(records) == count
    assert {row["payload"]["paper_final_mode"] for row in records} == {True}
    assert {row["payload"]["final_selection_version"] for row in records} == {FINAL_SELECTION_VERSION}
    assert {row["metadata"]["selection_policy"] for row in records} == {"strict_source_identity_and_count"}


def test_final_benign_paper_routes_reject_warning_only_filters_and_accept_explicit_ids() -> None:
    one_xstest = [{"id": "xs-1", "type": "safe", "prompt": "One benign prompt."}]
    with pytest.raises(FinalSelectionError, match="requires expected_count_mode='strict'"):
        build_eval_records(
            "xstest",
            one_xstest,
            subset="fixture",
            split="test",
            revision="pinned",
            source_file_sha256=_DIGEST,
            expected_count_mode="warn",
            final_mode=FINAL_MODE_PAPER,
        )
    exploratory = build_eval_records(
        "xstest",
        one_xstest,
        subset="fixture",
        split="test",
        source_file_sha256=_DIGEST,
        expected_count_mode="off",
    )
    assert exploratory[0]["payload"]["paper_final_mode"] is False

    rows = [{"id": f"xs-{index}", "type": "safe", "prompt": f"Benign prompt {index}."} for index in range(87)]
    selected = [f"xs-{index}" for index in range(86)]
    records = build_eval_records(
        "xstest",
        rows,
        subset="fixture",
        split="test",
        source_file_sha256=_DIGEST,
        selected_source_ids=selected,
    )
    assert len(records) == 86
    assert {row["metadata"]["selection_policy"] for row in records} == {"explicit_selected_ids"}


def test_expected_count_can_warn_or_fail() -> None:
    rows = [{"id": "one", "prompt": "A benign fixture."}]
    with pytest.warns(UserWarning, match="expected 2"):
        build_eval_records(
            "or_bench",
            rows,
            subset="fixture",
            split="test",
            source_file_sha256=_DIGEST,
            expected_count=2,
            expected_count_mode="warn",
        )
    with pytest.raises(ValueError, match="expected 2"):
        build_eval_records(
            "or_bench",
            rows,
            subset="fixture",
            split="test",
            source_file_sha256=_DIGEST,
            expected_count=2,
            expected_count_mode="strict",
        )


def test_local_export_materialization_and_task_construction_are_offline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "or-bench.jsonl"
    source_path.write_text(json.dumps({"id": "or-1", "prompt": "Plan a responsible exhibit."}) + "\n", encoding="utf-8")
    artifact_path = tmp_path / "eval.jsonl"
    materialize_eval_artifact(
        source_path,
        artifact_path,
        source="or_bench",
        subset="fixture",
        split="validation",
        revision="fixture-revision",
        expected_count=1,
        expected_count_mode="strict",
    )

    def fail_network(*args, **kwargs):
        raise AssertionError("network/model access attempted during task construction")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    pytest.importorskip("inspect_ai.model")
    import inspect_ai.model

    monkeypatch.setattr(inspect_ai.model, "get_model", fail_network)
    task_builds = []
    real_build_inspect_task = safety_tasks_module.build_inspect_task

    def recording_build_inspect_task(*args, **kwargs):
        task_builds.append((args, kwargs))
        return real_build_inspect_task(*args, **kwargs)

    monkeypatch.setattr(safety_tasks_module, "build_inspect_task", recording_build_inspect_task)
    tasks = paper_safety_tasks(datasets=[str(artifact_path)])
    assert len(tasks) == 1
    assert tasks[0].name == "irpan-or-bench-validation"
    assert len(tasks[0].dataset) == 1
    assert tasks[0].metadata["metric"] == "answered_benign"
    assert len(task_builds) == 1
    assert task_builds[0][1]["dataset_location"] == str(artifact_path)


def test_final_task_construction_rejects_exploratory_artifact(tmp_path: Path) -> None:
    pytest.importorskip("inspect_ai")
    source_path = tmp_path / "xstest.jsonl"
    source_path.write_text(
        json.dumps({"id": "xs-1", "type": "safe", "prompt": "A benign prompt."}) + "\n",
        encoding="utf-8",
    )
    artifact_path = tmp_path / "xstest-exploratory.jsonl"
    materialize_eval_artifact(
        source_path,
        artifact_path,
        source="xstest",
        subset="fixture",
        split="test",
        expected_count_mode="off",
    )
    with pytest.raises(ArtifactManifestError, match="has 1 rows; expected 86"):
        paper_safety_tasks(datasets=[str(artifact_path)])


def test_judge_parser_requires_exact_contract() -> None:
    assert parse_judge_output('{"fulfilled": true}', field="fulfilled") is True
    assert parse_judge_output('{"refused": false}', field="refused") is False
    with pytest.raises(ValueError, match="exactly"):
        parse_judge_output('{"refused": false, "note": "extra"}', field="refused")
    with pytest.raises(ValueError, match="exact JSON"):
        parse_judge_output('```json\n{"refused": false}\n```', field="refused")


def test_judge_retry_and_metric_coverage_are_bounded_and_auditable() -> None:
    attempts: list[object] = [RuntimeError("provider down"), "not-json", '{"refused": false}']

    async def complete() -> str:
        value = attempts.pop(0)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, str)
        return value

    result = asyncio.run(
        run_judge_with_retries(
            complete,
            parser=lambda raw: parse_judge_output(raw, field="refused"),
            retries=2,
            retry_delay=0,
        )
    )
    assert result.verdict is False
    assert result.attempt_count == 3
    assert result.provider_failure_count == 1
    assert result.parse_failure_count == 1

    scored = {
        "score_schema_version": JUDGE_SCORE_SCHEMA_VERSION,
        "metric_value": 1.0,
        "judge_scored": True,
        "judge_attempt_count": 3,
        "judge_parse_failure_count": 1,
        "judge_provider_failure_count": 1,
        "final_stage": True,
    }
    exhausted = {
        "score_schema_version": JUDGE_SCORE_SCHEMA_VERSION,
        "metric_value": 0.0,
        "judge_scored": False,
        "judge_attempt_count": 3,
        "judge_parse_failure_count": 3,
        "judge_provider_failure_count": 0,
        "final_stage": True,
    }
    summary = summarize_judge_scores([scored, exhausted], max_unscored_fraction=0.5)
    assert summary["mean"] == 1.0
    assert summary["unscored_count"] == 1
    assert summary["unscored_fraction"] == 0.5
    assert summary["judge_attempt_count"] == 6
    assert summary["judge_parse_failure_count"] == 4
    assert summary["judge_provider_failure_count"] == 1
    with pytest.raises(JudgeCoverageError, match="exceeds configured maximum"):
        summarize_judge_scores([scored, exhausted], max_unscored_fraction=0.49)


def test_read_local_export_supports_json_object_data_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps({"data": [{"id": "one", "prompt": "First."}]}), encoding="utf-8")
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("id,prompt\ntwo,Second.\n", encoding="utf-8")
    assert read_local_export(json_path)[0]["id"] == "one"
    assert read_local_export(csv_path)[0]["id"] == "two"


def test_read_local_export_delegates_supported_row_formats_to_ctm_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"ignored": true}\n', encoding="utf-8")
    calls = []

    class Loaded:
        def __init__(self) -> None:
            self.rows = [{"id": "from-generic-loader"}]

    def fake_load_local_rows(source_path, *, format):
        calls.append((source_path, format))
        return Loaded()

    monkeypatch.setattr(safety_tasks_module, "load_local_rows", fake_load_local_rows)
    assert read_local_export(path) == [{"id": "from-generic-loader"}]
    assert calls == [(path, "jsonl")]
