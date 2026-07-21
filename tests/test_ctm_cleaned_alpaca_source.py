import pytest

from ctm_data.sources.cleaned_alpaca import select_prompt_rows


def test_cleaned_alpaca_selection_is_deterministic_and_ignores_original_outputs():
    records = [
        {"instruction": f"Instruction {index}", "input": f"Context {index}", "output": f"Answer {index}"}
        for index in range(5)
    ]

    first = select_prompt_rows(records, count=3, seed="42")
    second = select_prompt_rows(records, count=3, seed="42")

    assert first == second
    assert len(first) == 3
    assert all("Answer" not in row["reference_messages"][0]["content"] for row in first)
    assert all(row["reference_messages"] == row["variant_messages"] for row in first)


def test_cleaned_alpaca_selection_enforces_exact_count_and_schema():
    with pytest.raises(ValueError, match="only 1/2"):
        select_prompt_rows([{"instruction": "Do this", "input": "", "output": "ignored"}], count=2, seed="42")
    with pytest.raises(ValueError, match="no instruction"):
        select_prompt_rows([{"instruction": "", "input": "", "output": "ignored"}], count=1, seed="42")
