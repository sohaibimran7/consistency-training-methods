import json

import pytest

from ctm_data.adapters.mcq_bias.dataset_specs import (
    DatasetSpec,
    normalize_dataset_specs,
    parse_dataset_cli_token,
    parse_dataset_cli_tokens,
)


def test_pinned_upstream_exposes_the_adapter_contract():
    import inspect

    from mcq_bias.pipeline.records import PROMPT_FAMILIES
    from mcq_bias.tasks import mcq_bias, suite_tasks

    assert set(PROMPT_FAMILIES) == {"chua", "irpan"}
    assert {"revision", "prompt_family", "source_format", "wrong_option_seed"} <= set(
        inspect.signature(mcq_bias).parameters
    )
    assert "datasets" in inspect.signature(suite_tasks).parameters
    assert DatasetSpec.__module__ == "mcq_bias.dataset_specs"


def test_mixed_string_and_mapping_specs_preserve_source_fields():
    specs = normalize_dataset_specs(
        [
            "mmlu",
            {
                "dataset": "org/custom",
                "dataset_config": "challenge",
                "split": "validation",
                "revision": "abc123",
                "question_field": "prompt",
                "choices_field": "choices.text",
                "answer_field": "answerKey",
            },
        ]
    )

    assert specs[0] == DatasetSpec(dataset="mmlu")
    assert specs[1].as_dict() == {
        "dataset": "org/custom",
        "dataset_config": "challenge",
        "split": "validation",
        "revision": "abc123",
        "question_field": "prompt",
        "choices_field": "choices.text",
        "answer_field": "answerKey",
    }


def test_cli_mapping_tokens_are_json_objects_but_plain_names_stay_plain():
    assert parse_dataset_cli_token("mmlu") == "mmlu"
    assert parse_dataset_cli_tokens(['{"split":"test","dataset":"org/custom"}']) == (
        DatasetSpec(dataset="org/custom", split="test"),
    )


def test_bbh_source_format_is_configuration_not_a_paper_normalizer():
    token = json.dumps(
        {
            "dataset": "org/bbh",
            "dataset_config": "logical_deduction_three_objects",
            "split": "train",
            "revision": "abc123",
            "source_format": "bbh",
        }
    )
    (spec,) = parse_dataset_cli_tokens([token])

    assert isinstance(spec, DatasetSpec)
    assert spec.as_dict()["source_format"] == "bbh"
    with pytest.raises(ValueError, match="cannot be combined"):
        DatasetSpec(dataset="org/bbh", source_format="bbh", question_field="input")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "non-empty"),
        ([{"split": "test"}], "missing required field"),
        ([{"dataset": "mmlu", "unknown": "x"}], "unknown dataset spec"),
        ([{"dataset": ""}], "non-empty string"),
        (["mmlu", {"dataset": "mmlu"}], "duplicates"),
    ],
)
def test_invalid_or_duplicate_specs_fail_closed(values, message):
    with pytest.raises(ValueError, match=message):
        normalize_dataset_specs(values)


def test_invalid_json_object_token_is_rejected():
    with pytest.raises(ValueError, match="invalid dataset-spec JSON"):
        parse_dataset_cli_token('{"dataset":')
