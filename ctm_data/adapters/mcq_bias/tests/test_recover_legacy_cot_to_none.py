"""Focused tests for the fail-closed legacy-G4 CoT-to-none recovery boundary."""

import copy
import hashlib
import json

import pytest

from ctm_data.adapters.mcq_bias.data import load_paths
from ctm_data.adapters.mcq_bias.recover_legacy_cot_to_none import (
    ARTIFACT_KIND,
    LEGACY_COT_TERMINAL,
    LEGACY_G4_WRONG_ARGUMENT_TEMPLATE,
    NONE_ANSWER_FORMAT_TERMINAL,
    TRANSFORM_VERSION,
    convert_legacy_g4_cot_row_to_none,
    recover_legacy_g4_cot_pairs_to_none,
)


def _legacy_cot_row(source_dataset: str, index: int) -> dict:
    question = (
        f"Question {source_dataset}-{index}?\n\n" "Answer choices:\n(A) apples\n(B) bananas\n(C) carrots\n(D) dates"
    )
    argument = f"An untrusted argument for {source_dataset}-{index}."
    return {
        "question": question,
        "question_id": f"{source_dataset}-{index}",
        "source_dataset": source_dataset,
        "prompt_style": "encourage_cot",
        "unbiased_messages": [{"role": "user", "content": question + LEGACY_COT_TERMINAL}],
        "biased_messages": [
            {
                "role": "user",
                "content": LEGACY_G4_WRONG_ARGUMENT_TEMPLATE.format(
                    argument=argument,
                    question=question,
                )
                + LEGACY_COT_TERMINAL,
            }
        ],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": argument,
        # Recovery must retain metadata outside the native frozen-row schema too.
        "provenance_note": {"nested": ["preserve", 7]},
    }


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_single_row_conversion_changes_only_prompt_style_and_message_content():
    source = _legacy_cot_row("logiqa", 3)
    converted = convert_legacy_g4_cot_row_to_none(source)

    assert source["prompt_style"] == "encourage_cot"
    assert source["unbiased_messages"][0]["content"].endswith(LEGACY_COT_TERMINAL)
    assert source["biased_messages"][0]["content"].endswith(LEGACY_COT_TERMINAL)

    expected = copy.deepcopy(source)
    expected["prompt_style"] = "none"
    expected["unbiased_messages"][0]["content"] = source["question"] + NONE_ANSWER_FORMAT_TERMINAL
    expected["biased_messages"][0]["content"] = (
        LEGACY_G4_WRONG_ARGUMENT_TEMPLATE.format(
            argument=source["biasing_text"],
            question=source["question"],
        )
        + NONE_ANSWER_FORMAT_TERMINAL
    )
    assert converted == expected
    assert "Don’t be biased" in converted["biased_messages"][0]["content"]
    assert "Don't be biased" not in converted["biased_messages"][0]["content"]
    assert LEGACY_COT_TERMINAL not in converted["unbiased_messages"][0]["content"]
    assert LEGACY_COT_TERMINAL not in converted["biased_messages"][0]["content"]


def test_recovery_writes_attested_native_rows_and_manifest(tmp_path):
    source = tmp_path / "legacy-g4-cot.jsonl"
    output = tmp_path / "recovered-none.jsonl"
    manifest_output = tmp_path / "recovered-none.manifest.json"
    rows = [_legacy_cot_row("logiqa", 0), _legacy_cot_row("hellaswag", 0)]
    _write_jsonl(source, rows)

    returned_manifest = recover_legacy_g4_cot_pairs_to_none(
        source,
        output=output,
        manifest_output=manifest_output,
    )

    expected_rows = [
        convert_legacy_g4_cot_row_to_none(row, path=source, line_number=index + 1) for index, row in enumerate(rows)
    ]
    assert load_paths([output], n_datapoints=2) == expected_rows

    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest == returned_manifest
    assert manifest["kind"] == ARTIFACT_KIND
    assert manifest["transform"]["version"] == TRANSFORM_VERSION
    assert manifest["transform"]["changed_row_fields"] == [
        "prompt_style",
        "unbiased_messages",
        "biased_messages",
    ]
    assert manifest["source"]["content_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["source"]["row_count"] == 2
    assert manifest["output"]["content_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["selection"] == {
        "row_count": 2,
        "ordered_question_ids_sha256": hashlib.sha256(b"logiqa-0\nhellaswag-0").hexdigest(),
        "source_dataset_counts": {"hellaswag": 1, "logiqa": 1},
    }
    assert not output.with_name(output.name + ".tmp").exists()
    assert not manifest_output.with_name(manifest_output.name + ".tmp").exists()


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        (
            lambda row: row["unbiased_messages"][0].__setitem__(
                "content", row["question"] + NONE_ANSWER_FORMAT_TERMINAL
            ),
            "unbiased_messages does not match",
        ),
        (
            lambda row: row["biased_messages"][0].__setitem__(
                "content",
                row["biased_messages"][0]["content"].replace("Don’t", "Don't"),
            ),
            "biased_messages does not match",
        ),
    ],
)
def test_recovery_fails_closed_on_missing_terminal_or_reconstructed_current_body(tmp_path, corruption, match):
    source = tmp_path / "legacy-g4-cot.jsonl"
    output = tmp_path / "recovered-none.jsonl"
    manifest_output = tmp_path / "recovered-none.manifest.json"
    row = _legacy_cot_row("logiqa", 0)
    corruption(row)
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match=match):
        recover_legacy_g4_cot_pairs_to_none(
            source,
            output=output,
            manifest_output=manifest_output,
        )

    assert not output.exists()
    assert not manifest_output.exists()


@pytest.mark.parametrize("existing_target", ["output", "manifest"])
def test_recovery_refuses_to_overwrite_either_target(tmp_path, existing_target):
    source = tmp_path / "legacy-g4-cot.jsonl"
    output = tmp_path / "recovered-none.jsonl"
    manifest_output = tmp_path / "recovered-none.manifest.json"
    _write_jsonl(source, [_legacy_cot_row("logiqa", 0)])
    target = output if existing_target == "output" else manifest_output
    target.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        recover_legacy_g4_cot_pairs_to_none(
            source,
            output=output,
            manifest_output=manifest_output,
        )

    assert target.read_text(encoding="utf-8") == "do not replace"
    other = manifest_output if existing_target == "output" else output
    assert not other.exists()
