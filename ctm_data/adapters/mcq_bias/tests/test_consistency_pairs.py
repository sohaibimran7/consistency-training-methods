from __future__ import annotations

import json

import pytest

from ctm.artifacts import read_verified_artifact_manifest
from ctm_data.adapters.mcq_bias.consistency_pairs import (
    ARTIFACT_SCHEMA,
    TRANSFORM_VERSION,
    canonicalize_wrong_argument_row,
    materialize_consistency_pairs,
)


def _row() -> dict:
    clean = "What is two plus two?\n\nAnswer in the requested format."
    return {
        "question": "What is two plus two?",
        "question_id": "q1",
        "source_dataset": "toy",
        "prompt_style": "encourage_cot",
        "unbiased_messages": [{"role": "user", "content": clean}],
        "biased_messages": [
            {
                "role": "user",
                "content": (
                    "<argument>Five is correct.</argument>\n"
                    "<question>What is two plus two?</question>\n"
                    "Biased-only suffix.\n\nAnswer in the requested format."
                ),
            }
        ],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": "Five is correct.",
    }


def test_canonical_pair_is_argument_prefix_plus_exact_clean_suffix() -> None:
    source = _row()
    derived = canonicalize_wrong_argument_row(source)
    clean = source["unbiased_messages"][-1]["content"]
    biased = derived["biased_messages"][-1]["content"]

    # The legacy native wrapper leaves bias-only terminal text after the
    # question, so it cannot support complete-prompt consistency alignment.
    assert not source["biased_messages"][-1]["content"].endswith(clean)
    assert biased.endswith(clean)
    assert derived["unbiased_messages"] == source["unbiased_messages"]
    assert derived["biasing_text"] == source["biasing_text"]
    assert derived["consistency_pair_transform"] == TRANSFORM_VERSION
    assert "Biased-only suffix" not in biased
    assert "Five is correct." in biased[: -len(clean)]
    # Never mutate the frozen source object in-place.
    assert source == _row()


def test_canonical_pair_rejects_other_biases() -> None:
    source = _row()
    source["bias_type"] = "suggested_answer"
    with pytest.raises(ValueError, match="supports only wrong_argument"):
        canonicalize_wrong_argument_row(source)


def test_materialization_is_verified_and_idempotent(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "pairs.jsonl"
    manifest = tmp_path / "pairs.manifest.json"
    source.write_text(json.dumps(_row()) + "\n")

    assert materialize_consistency_pairs(source, output, manifest) == "written"
    assert materialize_consistency_pairs(source, output, manifest) == "resumed"
    document = read_verified_artifact_manifest(
        output,
        manifest_path=manifest,
        expected_schema=ARTIFACT_SCHEMA,
        expected_schema_version=1,
    )
    assert document["row_count"] == 1
    assert document["provenance"]["transform"]["name"] == TRANSFORM_VERSION
    row = json.loads(output.read_text())
    assert row["biased_messages"][-1]["content"].endswith(
        row["unbiased_messages"][-1]["content"]
    )


def test_materialization_refuses_differing_existing_output(tmp_path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "pairs.jsonl"
    manifest = tmp_path / "pairs.manifest.json"
    source.write_text(json.dumps(_row()) + "\n")
    output.write_text("different\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize_consistency_pairs(source, output, manifest)
