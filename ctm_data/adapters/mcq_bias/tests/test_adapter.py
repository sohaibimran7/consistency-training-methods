"""Offline tests for the thin native ``mcq_bias`` training adapter."""

import json

import pytest

from ctm_data.adapters.mcq_bias.data import file_identity, load_paths
from ctm_data.adapters.mcq_bias.setting import SycophancySetting


def _frozen_row(
    *,
    bias_type: str = "suggested_answer",
    dataset: str = "unit",
    question_id: str = "q1",
    prompt_style: str = "none",
) -> dict:
    return {
        "question": f"Question {question_id}?",
        "question_id": question_id,
        "source_dataset": dataset,
        "prompt_style": prompt_style,
        "unbiased_messages": [{"role": "user", "content": f"clean {question_id}"}],
        "biased_messages": [{"role": "user", "content": f"biased {question_id}"}],
        "bias_type": bias_type,
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": "The user suggests B.",
    }


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_native_rows_are_validated_without_renaming_fields(tmp_path):
    path = _write_rows(tmp_path / "chosen.jsonl", [_frozen_row()])

    loaded = load_paths([path], n_datapoints=1)

    assert loaded == [_frozen_row()]


def test_invalid_native_rows_fail_at_the_file_and_line(tmp_path):
    row = _frozen_row()
    del row["biased_messages"]
    path = _write_rows(tmp_path / "broken.jsonl", [row])

    with pytest.raises(ValueError, match=r"broken.jsonl:1: missing mcq_bias frozen field.*biased_messages"):
        load_paths([path], n_datapoints=1)


def test_are_you_sure_fails_until_multiturn_training_is_implemented(tmp_path):
    path = _write_rows(
        tmp_path / "are_you_sure.jsonl",
        [_frozen_row(bias_type="are_you_sure")],
    )

    with pytest.raises(NotImplementedError, match="requires staged multi-turn generation"):
        load_paths([path], n_datapoints=1)


@pytest.mark.parametrize("biased_option", ["", "   ", "NOT", "NOT ", "NOT  A", "NOTA"])
def test_invalid_biased_options_fail_during_loading(tmp_path, biased_option):
    row = _frozen_row()
    row["biased_option"] = biased_option
    path = _write_rows(tmp_path / "invalid-option.jsonl", [row])

    with pytest.raises(ValueError, match="biased_option"):
        load_paths([path], n_datapoints=1)


def test_total_count_is_split_across_exact_selected_files(tmp_path):
    first = _write_rows(
        tmp_path / "first.jsonl",
        [_frozen_row(question_id=f"a{i}") for i in range(3)],
    )
    second = _write_rows(
        tmp_path / "second.jsonl",
        [_frozen_row(question_id=f"b{i}") for i in range(3)],
    )

    loaded = load_paths([first, second], n_datapoints=3)

    assert [row["question_id"] for row in loaded] == ["a0", "a1", "b0"]


def test_each_selected_file_can_have_an_independent_limit(tmp_path):
    first = _write_rows(
        tmp_path / "first.jsonl",
        [_frozen_row(question_id=f"a{i}") for i in range(3)],
    )
    second = _write_rows(
        tmp_path / "second.jsonl",
        [_frozen_row(question_id=f"b{i}") for i in range(3)],
    )

    loaded = load_paths(
        [first, second],
        path_limits={str(first): 1, str(second): 2},
    )

    assert [row["question_id"] for row in loaded] == ["a0", "b0", "b1"]


def test_selection_errors_are_explicit(tmp_path):
    path = _write_rows(tmp_path / "one.jsonl", [_frozen_row()])

    with pytest.raises(ValueError, match="at least one data_path"):
        load_paths([], n_datapoints=1)
    with pytest.raises(ValueError, match="at least the number of data_paths"):
        load_paths([path, path], n_datapoints=1)
    with pytest.raises(ValueError, match="only 1/2 requested rows"):
        load_paths([path], n_datapoints=2)
    with pytest.raises(ValueError, match="not present in data_paths"):
        load_paths([path], path_limits={"somewhere-else.jsonl": 1})


def test_setting_uses_only_explicit_files_and_records_their_identity(tmp_path):
    path = _write_rows(
        tmp_path / "chosen.jsonl",
        [_frozen_row(bias_type="post_hoc", dataset="truthfulqa")],
    )
    setting = SycophancySetting(data_paths=[path])

    loaded = setting.load_datapoints(n_datapoints=1)

    assert [row["question_id"] for row in loaded] == ["q1"]
    assert setting.bias_types == ["post_hoc"]
    assert setting.datasets == ["truthfulqa"]
    assert setting.training_artifact_identity() == [file_identity(path)]


def test_setting_does_not_choose_training_data_implicitly():
    with pytest.raises(ValueError, match="requires at least one data_path"):
        SycophancySetting().load_datapoints(n_datapoints=1)
