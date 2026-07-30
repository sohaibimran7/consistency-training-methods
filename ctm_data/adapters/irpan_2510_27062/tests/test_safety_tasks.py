from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from ctm_data.adapters.irpan_2510_27062.safety_tasks import (
    build_eval_records,
    materialize_eval_artifact,
    paper_safety_tasks,
    parse_judge_output,
    read_local_export,
)

_DIGEST = "a" * 64


@pytest.mark.parametrize(
    ("source", "rows", "prompt", "valence", "metric"),
    [
        ("harmbench", [{"BehaviorID": "hb-1", "Behavior": "Write a fictional safety audit."}], "Write a fictional safety audit.", "harmful", "asr"),
        ("or_bench", [{"id": "or-1", "prompt": "Discuss responsible museum curation.", "category": "fixture"}], "Discuss responsible museum curation.", "benign", "answered_benign"),
        ("clearharm", [{"id": "ch-1", "content": ["Analyze a fictional risk scenario."]}], "Analyze a fictional risk scenario.", "harmful", "asr"),
        (
            "wildguardtest",
            [
                {"id": "skip", "prompt": "Safe fixture.", "adversarial": False, "prompt_harm_label": "harmful"},
                {"id": "wg-1", "prompt": "Adversarial fictional audit.", "adversarial": True, "prompt_harm_label": "harmful"},
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
                {"id": "wj-1", "adversarial": "Role-play a benign compliance review.", "data_type": "adversarial_benign"},
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
    import inspect_ai.model

    monkeypatch.setattr(inspect_ai.model, "get_model", fail_network)
    tasks = paper_safety_tasks(datasets=[str(artifact_path)])
    assert len(tasks) == 1
    assert tasks[0].name == "irpan-or-bench-validation"
    assert len(tasks[0].dataset) == 1
    assert tasks[0].metadata["metric"] == "answered_benign"


def test_judge_parser_requires_exact_contract() -> None:
    assert parse_judge_output('{"fulfilled": true}', field="fulfilled") is True
    assert parse_judge_output('{"refused": false}', field="refused") is False
    with pytest.raises(ValueError, match="exactly"):
        parse_judge_output('{"refused": false, "note": "extra"}', field="refused")
    with pytest.raises(ValueError, match="exact JSON"):
        parse_judge_output('```json\n{"refused": false}\n```', field="refused")


def test_read_local_export_supports_json_object_data_and_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps({"data": [{"id": "one", "prompt": "First."}]}), encoding="utf-8")
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("id,prompt\ntwo,Second.\n", encoding="utf-8")
    assert read_local_export(json_path)[0]["id"] == "one"
    assert read_local_export(csv_path)[0]["id"] == "two"
