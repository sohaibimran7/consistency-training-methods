import json
from types import SimpleNamespace

from ctm_data.adapters.mcq_bias.materialize import interleave_rows


def test_interleave_rows_keeps_global_prefix_balanced():
    assert interleave_rows([[{"id": "a1"}, {"id": "a2"}], [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}]]) == [
        {"id": "a1"},
        {"id": "b1"},
        {"id": "a2"},
        {"id": "b2"},
        {"id": "b3"},
    ]


def test_materialize_forwards_dataset_specs_and_uses_task_metadata_path(tmp_path, monkeypatch):
    from mcq_bias import tasks

    from ctm_data.adapters.mcq_bias.materialize import main

    calls = []

    def fake_mcq_bias(**kwargs):
        calls.append(kwargs)
        path = tmp_path / f"source-{len(calls)}.jsonl"
        path.write_text(json.dumps({"question_id": str(len(calls))}) + "\n")
        return SimpleNamespace(metadata={"dataset_file": str(path)})

    monkeypatch.setattr(tasks, "mcq_bias", fake_mcq_bias)
    output = tmp_path / "training.jsonl"
    manifest = tmp_path / "training.manifest.json"
    main(
        [
            "--bias-type",
            "suggested_answer",
            "--datasets",
            "mmlu",
            '{"dataset":"org/custom","dataset_config":"challenge","split":"validation",'
            '"revision":"abc123","question_field":"prompt","choices_field":"answers","answer_field":"label"}',
            "--n-questions",
            "1",
            "--prompt-family",
            "irpan",
            "--wrong-option-seed",
            "17",
            "--dataset-dir",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--manifest-output",
            str(manifest),
            "-y",
        ]
    )

    assert calls[0]["dataset"] == "mmlu"
    assert calls[1]["dataset"] == "org/custom"
    assert calls[1]["dataset_config"] == "challenge"
    assert calls[1]["split"] == "validation"
    assert calls[1]["revision"] == "abc123"
    assert calls[1]["question_field"] == "prompt"
    assert calls[1]["choices_field"] == "answers"
    assert calls[1]["answer_field"] == "label"
    assert all(call["prompt_family"] == "irpan" for call in calls)
    assert all(call["wrong_option_seed"] == "17" for call in calls)
    assert [json.loads(line)["question_id"] for line in output.read_text().splitlines()] == ["1", "2"]
    selection = json.loads(manifest.read_text())["selection"]
    assert selection["datasets"] == ["mmlu", "org/custom"]
    assert selection["dataset_specs"][1]["revision"] == "abc123"


def test_eval_materializer_forwards_dataset_specs(tmp_path, monkeypatch):
    from mcq_bias import tasks

    from ctm_data.adapters.mcq_bias.materialize_eval import main

    captured = {}

    def fake_suite_tasks(**kwargs):
        captured.update(kwargs)
        return [object(), object()]

    monkeypatch.setattr(tasks, "suite_tasks", fake_suite_tasks)
    main(
        [
            "--bias-types",
            "suggested_answer",
            "--datasets",
            '{"dataset":"org/custom","split":"validation","revision":"abc123"}',
            "--n-questions",
            "1",
            "--dataset-dir",
            str(tmp_path),
            "-y",
        ]
    )

    assert captured["datasets"] == [{"dataset": "org/custom", "split": "validation", "revision": "abc123"}]
