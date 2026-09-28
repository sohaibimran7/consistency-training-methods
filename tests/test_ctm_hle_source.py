import pytest

from scripts.rmct_paper_vast_more_methods.hle_source import canonical_rows, parse_multiple_choice_question


def test_parse_hle_inline_choices():
    question, choices = parse_multiple_choice_question(
        "Which option is correct?\n\nAnswer Choices:\nA. First answer\nB. Second answer\ncontinued\nC. Third"
    )
    assert question == "Which option is correct?"
    assert choices == ["First answer", "Second answer\ncontinued", "Third"]


def test_canonical_hle_rows_filter_non_mc_and_images():
    rows = canonical_rows(
        [
            {
                "id": "kept",
                "question": "Question?\n\nAnswer Choices:\nA. Alpha\nB. Beta",
                "answer": "B",
                "answer_type": "multipleChoice",
                "image": None,
            },
            {
                "id": "short",
                "question": "Short answer",
                "answer": "x",
                "answer_type": "exactMatch",
                "image": None,
            },
            {
                "id": "image",
                "question": "Image?\n\nAnswer Choices:\nA. Alpha\nB. Beta",
                "answer": "A",
                "answer_type": "multipleChoice",
                "image": "image.png",
            },
        ]
    )
    assert rows == [{"source_id": "kept", "question": "Question?", "options": ["Alpha", "Beta"], "answer": "B"}]


def test_hle_parser_rejects_nonconsecutive_labels():
    with pytest.raises(ValueError, match="consecutive"):
        parse_multiple_choice_question("Question?\nAnswer Choices:\nA. Alpha\nC. Gamma")
