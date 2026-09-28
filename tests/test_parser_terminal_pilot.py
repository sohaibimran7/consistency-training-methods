"""Regression for the isolated pilot candidate, not a deployment change."""
import pytest
from ctm_data.adapters.mcq_bias import terminal_answer as module

@pytest.mark.parametrize('text,answer',[
 ('The best answer is: (B).\nI reconsider.\nThe best answer is: (D).','D'),
 ('The best answer is: (B) and (D).\nThe best answer is: (B)','B'),
 ('If the answer is (C) or (B), ...\nThe best answer is: (C)','C'),
 ('Given the prompt label, the answer is:\n\nThe best answer is: (D).','D'),
 ('<think>The best answer is: (C).',None),
 ('<think>B</think>\nThe best answer is: (A).','A'),
 ('The best answer is: (A).\nThe best answer is: (Z).',None),
 ('The best answer is: (A) or (B).',None),
 ('"The best answer is: (A)."',None),
 ('```\nThe best answer is: (A).',None),
 ('```\nThe best answer is: (A).\n```',None),
 ('> The best answer is: (A).',None),
 ('The best answer is: (None of these)',None),
 ('**The best answer is: (D).**','D'),
])
def test_terminal_candidate(text,answer):
 assert module.parse_terminal_first(text,list('ABCD'))==answer
