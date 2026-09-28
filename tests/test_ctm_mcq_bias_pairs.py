from __future__ import annotations

import pytest

from ctm_data.adapters.mcq_bias.materialize import native_rows_to_prompt_pairs


def _native_row() -> dict:
    return {
        "question": "Which number is even?",
        "question_id": "question-1",
        "source_dataset": "unit",
        "prompt_style": "none",
        "prompt_family": "irpan",
        "unbiased_messages": [{"role": "user", "content": "clean"}],
        "biased_messages": [{"role": "user", "content": "biased"}],
        "bias_type": "suggested_answer",
        "ground_truth": "B",
        "biased_option": "C",
        "biasing_text": "I prefer C",
        "option_labels": ["A", "B", "C", "D"],
        "wrong_option_seed": "paper-seed",
    }


def test_native_mcq_rows_adapt_to_shared_prompt_pairs() -> None:
    pairs = native_rows_to_prompt_pairs([_native_row()])

    assert pairs == [
        {
            "pair_id": "mcq-bias:suggested_answer:question-1",
            "source_id": "question-1",
            "source": "unit",
            "reference_messages": [{"role": "user", "content": "clean"}],
            "variant_messages": [{"role": "user", "content": "biased"}],
            "alignment_text": "Which number is even?",
            "metadata": {
                "bias_type": "suggested_answer",
                "correct_label": "B",
                "biased_option": "C",
                "valid_labels": ["A", "B", "C", "D"],
                "prompt_family": "irpan",
                "prompt_style": "none",
                "wrong_option_seed": "paper-seed",
                "biasing_text": "I prefer C",
            },
        }
    ]


def test_native_adapter_requires_explicit_parser_labels() -> None:
    row = _native_row()
    row.pop("option_labels")

    with pytest.raises(ValueError, match="no option_labels"):
        native_rows_to_prompt_pairs([row])
