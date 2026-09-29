"""Offline tests for the thin native ``mcq_bias`` training adapter."""

import json
from collections import Counter

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


def test_empty_message_content_fails_during_loading(tmp_path):
    row = _frozen_row()
    row["biased_messages"][0]["content"] = ""
    path = _write_rows(tmp_path / "empty-message.jsonl", [row])

    with pytest.raises(ValueError, match="non-empty string role/content"):
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


def test_row_offset_loads_four_exact_balanced_disjoint_blocks_without_rewriting_source(tmp_path):
    rows = [
        _frozen_row(
            dataset="logiqa" if index % 2 == 0 else "hellaswag",
            question_id=f"q{index:03d}",
        )
        for index in range(256)
    ]
    path = _write_rows(tmp_path / "full-256.jsonl", rows)
    original_payload = path.read_bytes()

    blocks = [load_paths([path], n_datapoints=64, row_offset=offset) for offset in range(0, 256, 64)]

    assert [[row["question_id"] for row in block] for block in blocks] == [
        [f"q{index:03d}" for index in range(offset, offset + 64)] for offset in range(0, 256, 64)
    ]
    assert all(Counter(row["source_dataset"] for row in block) == {"logiqa": 32, "hellaswag": 32} for block in blocks)
    block_ids = [set(row["question_id"] for row in block) for block in blocks]
    assert all(first.isdisjoint(second) for index, first in enumerate(block_ids) for second in block_ids[index + 1 :])
    assert set().union(*block_ids) == {f"q{index:03d}" for index in range(256)}
    assert path.read_bytes() == original_payload


def test_row_offset_records_exact_slice_provenance(tmp_path):
    path = _write_rows(
        tmp_path / "full-256.jsonl",
        [
            _frozen_row(
                dataset="logiqa" if index % 2 == 0 else "hellaswag",
                question_id=f"q{index:03d}",
            )
            for index in range(256)
        ],
    )
    setting = SycophancySetting(data_paths=[path])

    loaded = setting.load_datapoints(n_datapoints=64, row_offset=128)

    expected_selection = {
        "method": "exact_contiguous_source_rows_without_reserialization",
        "row_offset": 128,
        "row_count": 64,
        "source_rows_1_based_inclusive": [129, 192],
    }
    assert [row["question_id"] for row in loaded] == [f"q{index:03d}" for index in range(128, 192)]
    assert setting.training_artifact_identity()[0]["provenance"]["selection"] == expected_selection
    assert setting.run_metadata()["row_selection"] == expected_selection


@pytest.mark.parametrize("row_offset", [-1, True, "0", 1.5])
def test_row_offset_rejects_invalid_types_and_values(tmp_path, row_offset):
    path = _write_rows(tmp_path / "one.jsonl", [_frozen_row()])

    with pytest.raises(ValueError, match="row_offset must be a non-negative integer"):
        load_paths([path], n_datapoints=1, row_offset=row_offset)


def test_row_offset_rejects_ambiguous_and_out_of_bounds_slices(tmp_path):
    rows = [_frozen_row(question_id=f"q{index}") for index in range(256)]
    first = _write_rows(tmp_path / "first.jsonl", rows)
    second = _write_rows(tmp_path / "second.jsonl", rows)

    with pytest.raises(ValueError, match="row_offset requires exactly one data_path"):
        load_paths([first, second], n_datapoints=64, row_offset=0)
    with pytest.raises(ValueError, match="row_offset cannot be combined with path_limits"):
        load_paths([first], path_limits={str(first): 64}, row_offset=0)
    with pytest.raises(ValueError, match=r"only 63/64 requested rows after row_offset 193"):
        load_paths([first], n_datapoints=64, row_offset=193)


def test_setting_binds_convergence_offset_to_the_verified_segment_manifest(tmp_path, monkeypatch):
    path = _write_rows(
        tmp_path / "full-256.jsonl",
        [
            _frozen_row(
                dataset="logiqa" if index % 2 == 0 else "hellaswag",
                question_id=f"q{index:03d}",
            )
            for index in range(256)
        ],
    )
    expected_digest = "a" * 64
    expected_segment = {
        "row_offset": 128,
        "row_count": 64,
        "source_rows_1_based_inclusive": [129, 192],
        "content_sha256": "b" * 64,
        "question_ids_sha256": "c" * 64,
    }
    captured = {}

    def fake_verifier(manifest, **kwargs):
        captured["manifest"] = manifest
        captured.update(kwargs)
        return {"segments": [{}, {}, expected_segment, {}]}

    import experiments.rmct_256_convergence.selection as convergence_selection

    monkeypatch.setattr(convergence_selection, "verify_rmct256_convergence_segments_manifest", fake_verifier)
    setting = SycophancySetting(data_paths=[path])

    loaded = setting.load_datapoints(
        n_datapoints=64,
        row_offset=128,
        selection_manifest="rmct256-parent.manifest.json",
        rmct256_convergence_manifest="convergence.manifest.json",
        rmct256_convergence_manifest_sha256=expected_digest,
        rmct256_segment_index=2,
        rmct256_convergence_metadata={"ignored_by_loader": True},
    )

    assert [row["question_id"] for row in loaded] == [f"q{index:03d}" for index in range(128, 192)]
    assert captured == {
        "manifest": "convergence.manifest.json",
        "parent_selection": path,
        "parent_selection_manifest": "rmct256-parent.manifest.json",
        "verify_parent_sources": False,
        "expected_manifest_sha256": expected_digest,
    }
    provenance = setting.training_artifact_identity()[0]["provenance"]["selection"]
    assert provenance["rmct256_convergence_segment"] == {
        "index": 2,
        "manifest": {"filename": "convergence.manifest.json", "content_sha256": expected_digest},
        **expected_segment,
    }


def test_setting_rejects_incomplete_convergence_contract_without_affecting_ordinary_selection_manifest(tmp_path):
    path = _write_rows(tmp_path / "one.jsonl", [_frozen_row()])
    setting = SycophancySetting(data_paths=[path])

    assert setting.load_datapoints(n_datapoints=1, selection_manifest="ordinary-parent.manifest.json") == [_frozen_row()]
    with pytest.raises(ValueError, match="complete immutable contract; missing"):
        setting.load_datapoints(
            n_datapoints=1,
            row_offset=0,
            rmct256_convergence_manifest="convergence.manifest.json",
        )
