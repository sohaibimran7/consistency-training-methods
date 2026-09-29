"""Offline coverage for the legacy paired-prompt import boundary."""

import hashlib
import json

import pytest

from ctm_data.adapters.mcq_bias.data import load_paths
from ctm_data.adapters.mcq_bias.import_legacy_pairs import (
    FALLBACK_BIASING_TEXT,
    import_legacy_pairs,
    interleave_sources,
)


def _legacy_row(source: str, index: int, *, biased: bool = True, tagged_argument: bool = True) -> dict:
    biased_content = f"legacy biased prompt for {source}-{index}"
    if tagged_argument:
        biased_content = f"prefix\n<argument>\nArgument for {source}-{index}.\n</argument>\nsuffix"
    return {
        "bias_name": "distractor_argument",
        "biased_option": "B",
        "biased_question": [{"role": "user", "content": biased_content}] if biased else [],
        "ground_truth": "A",
        "original_dataset": source,
        "original_question": f"Question {source}-{index}?\n\nAnswer choices:\n(A) yes\n(B) no",
        "original_question_hash": f"{source}-{index}",
        "unbiased_question": [{"role": "user", "content": f"unbiased prompt for {source}-{index}"}],
    }


@pytest.fixture
def legacy_sources(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_rows = [
        _legacy_row("first", 0),
        _legacy_row("first", 1, biased=False),
        _legacy_row("first", 2, tagged_argument=False),
        _legacy_row("first", 3),
    ]
    second_rows = [
        _legacy_row("second", 0),
        _legacy_row("second", 1, biased=False),
        _legacy_row("second", 2),
        _legacy_row("second", 3),
    ]
    first.write_text("".join(json.dumps(row) + "\n" for row in first_rows), encoding="utf-8")
    second.write_text("".join(json.dumps(row) + "\n" for row in second_rows), encoding="utf-8")
    return first, second


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_import_filters_empty_pairs_selects_exact_prefixes_and_validates_native_schema(
    tmp_path, legacy_sources
):
    first, second = legacy_sources
    output = tmp_path / "native.jsonl"
    manifest_output = tmp_path / "native.manifest.json"

    returned_manifest = import_legacy_pairs(
        [first, second],
        output=output,
        manifest_output=manifest_output,
        per_source_limit=2,
    )

    rows = _read_jsonl(output)
    assert [row["question_id"] for row in rows] == ["first-0", "second-0", "first-2", "second-2"]
    assert [row["source_dataset"] for row in rows] == ["first", "second", "first", "second"]
    assert all(row["bias_type"] == "wrong_argument" for row in rows)
    assert all(row["prompt_style"] == "encourage_cot" for row in rows)
    assert rows[0]["biasing_text"] == "Argument for first-0."
    assert rows[2]["biasing_text"] == FALLBACK_BIASING_TEXT
    assert set(rows[0]) == {
        "question",
        "question_id",
        "source_dataset",
        "prompt_style",
        "unbiased_messages",
        "biased_messages",
        "bias_type",
        "ground_truth",
        "biased_option",
        "biasing_text",
    }

    # Exercise the trainer-facing native validator against the emitted bytes.
    assert load_paths([output], n_datapoints=4) == rows

    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest == returned_manifest
    assert manifest["row_count"] == 4
    assert manifest["selection"] == {
        "merge": "round_robin in CLI input order",
        "per_source_limit": 2,
        "prompt_style": "encourage_cot",
        "shuffle": False,
        "within_source": "first valid rows in file order",
    }
    assert [source["valid_row_count"] for source in manifest["sources"]] == [3, 3]
    assert [source["invalid_row_count"] for source in manifest["sources"]] == [1, 1]
    assert manifest["sources"][0]["invalid_reasons"] == {"empty_biased_question": 1}
    assert manifest["sources"][0]["selected_biasing_text_modes"] == {
        "argument_block": 1,
        "fallback_marker": 1,
    }
    assert manifest["sources"][0]["content_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert manifest["sources"][1]["content_sha256"] == hashlib.sha256(second.read_bytes()).hexdigest()
    assert manifest["output"]["content_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert not output.with_name(output.name + ".tmp").exists()
    assert not manifest_output.with_name(manifest_output.name + ".tmp").exists()


def test_round_robin_makes_the_first_2048_exactly_balanced():
    first = [{"question_id": f"first-{index}"} for index in range(1500)]
    second = [{"question_id": f"second-{index}"} for index in range(1500)]

    rows = interleave_sources([first, second])

    assert len(rows) == 3000
    assert sum(row["question_id"].startswith("first-") for row in rows[:2048]) == 1024
    assert sum(row["question_id"].startswith("second-") for row in rows[:2048]) == 1024
    assert [row["question_id"] for row in rows[:6]] == [
        "first-0",
        "second-0",
        "first-1",
        "second-1",
        "first-2",
        "second-2",
    ]


@pytest.mark.parametrize("existing_target", ["output", "manifest"])
def test_import_refuses_to_overwrite_either_target(tmp_path, legacy_sources, existing_target):
    first, second = legacy_sources
    output = tmp_path / "native.jsonl"
    manifest_output = tmp_path / "native.manifest.json"
    target = output if existing_target == "output" else manifest_output
    target.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        import_legacy_pairs(
            [first, second],
            output=output,
            manifest_output=manifest_output,
            per_source_limit=2,
        )

    assert target.read_text(encoding="utf-8") == "do not replace"
    other = manifest_output if existing_target == "output" else output
    assert not other.exists()


def test_import_requires_the_exact_limit_from_every_source(tmp_path, legacy_sources):
    first, second = legacy_sources
    output = tmp_path / "native.jsonl"
    manifest_output = tmp_path / "native.manifest.json"

    with pytest.raises(ValueError, match=r"only 3/4 valid legacy prompt pairs"):
        import_legacy_pairs(
            [first, second],
            output=output,
            manifest_output=manifest_output,
            per_source_limit=4,
        )

    assert not output.exists()
    assert not manifest_output.exists()
