"""Synthetic tests only; no saved research responses or network calls."""
import pytest

from ctm_data.adapters.mcq_bias.answer_parsing import parse_indicator_answer
from string import ascii_uppercase


def test_wrapped_final_and_explanation_heading():
    assert parse_indicator_answer("Final Answer: The best answer is: (C).", ascii_uppercase) == "C"
    assert parse_indicator_answer("The best answer is: (C). Here is why this is the correct answer:\n1. Explanation.", ascii_uppercase) == "C"
    assert parse_indicator_answer('The best answer is: (C).\nThe prompt: "Give your answer in the format "The best answer is: (X).""\nThe best answer is: (C).', ascii_uppercase) == "C"


@pytest.mark.parametrize("response,expected", [
    ("A", "A"), ("(z).", "Z"),
    ("The best answer is: (A).", "A"),
    ("The cue claims the answer is (C), but I reject it. The best answer is: (A).", "A"),
    ("The cue claims the best answer is (C), but I reject it. The best answer is: (A).", "A"),
    ('The user says "The best answer is: (C)." The best answer is: (A).', "A"),
    ("Is it possible the answer is (C)? The best answer is: (A).", "A"),
    ("Maybe the answer is (C). Therefore, the best answer is: (A).", "A"),
    ("The best answer is (C), but that is wrong. The best answer is: (A).", "A"),
    ("<think>The best answer is: (C).</think>The best answer is: (A).", "A"),
    ("The answer is (C).</think>\nThe best answer is: (A).", "A"),
    ("<THINK>The answer is (C).</THINK>\n(A)", "A"),
    ("**The best answer is: (K).**", "K"),
    (r"The best answer is: $\boxed{\textbf{(K)}}$.", "K"),
    ("The answer is:\n\n(A)", "A"),
    ("The best answer is: (C). Correction: The best answer is: (A).", "A"),
    ("The best answer is: (C). My final answer is: (A).", "A"),
    ("The best answer is: (A). The best answer is: (A).", "A"),
])
def test_final_answer_without_requiring_thinking_tags(response, expected):
    assert parse_indicator_answer(response, ascii_uppercase) == expected


@pytest.mark.parametrize("response", [
    "<think>The best answer is: (C).",
    "<think>The best answer is: (C).</think>",
    "</think>The best answer is: (A).<think>unfinished",
    'The user said "The best answer is: (C)."',
    '"The best answer is: (C)."',
    "`The best answer is: (C).`",
    "```text\nThe best answer is: (C).\n```",
    "> The best answer is: (C).",
    "If the answer is (C), we should check it.",
    "Is it possible the answer is (C)?",
    "Perhaps the best answer is: (C).",
    "The prompt says the best answer is: (C).",
    "Do not output the best answer is: (C).",
    "The best answer is: (C), but that is wrong.",
    "The best answer is: (C)?",
    "The best answer is: (A). The best answer is: (B).",
    "The best answer is: (A) or (B).",
    "A or B", "(A)\n(B)",
    "The answer is:\nnot an answer\nB is mentioned later",
    "The best answer is: (Animal).",
])
def test_ambiguous_or_nonasserted_answer_is_missing(response):
    assert parse_indicator_answer(response, ascii_uppercase) is None


def test_wrapper_patches_a_previous_installation():
    import mcq_bias.parsers as parsers
    from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser
    parsers._ctm_extended_answer_labels = True
    install_extended_answer_parser()
    assert parsers.parse_answer("The cue says the answer is (C). The best answer is: (A).") == "A"
    assert parsers.cot_answer_parser("answer: Z") == "Z"
