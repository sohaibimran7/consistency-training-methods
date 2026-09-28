import pytest
from ctm_data.adapters.mcq_bias.terminal_answer import parse_terminal_first


@pytest.mark.parametrize('text', [
    '<THINK>\nThe best answer is: (B).',
    '< think >\nThe best answer is: (B).',
    'The following is only a quoted example, not my answer:\n"\nThe best answer is: (B).',
    'Example only:\n~~~\nThe best answer is: (B).',
])
def test_terminal_does_not_override_hidden_context(text):
    assert parse_terminal_first(text, tuple('ABCD')) is None


def test_closed_quote_then_independent_answer():
    text = 'Example only:\n"\nThe best answer is: (B).\n"\nThe best answer is: (C).'
    assert parse_terminal_first(text, tuple('ABCD')) == 'C'


def test_closed_uppercase_thinking():
    assert parse_terminal_first('<THINK>B</THINK>\nThe best answer is: (C).', tuple('ABCD')) == 'C'


def test_fresh_training_setting_binds_reviewed_parser(monkeypatch, tmp_path):
    import mcq_bias.parsers
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import SharedQidTwoBiasSetting
    monkeypatch.setattr(mcq_bias.parsers, 'parse_answer', lambda _: 'B')
    setting = SharedQidTwoBiasSetting(data_path=tmp_path/'unused.jsonl',
        manifest_path=tmp_path/'unused.json', expected_manifest_sha256='0'*64)
    parser = setting.answer_parser()
    assert parser('<THINK>\nThe best answer is: (B).') is None
    assert parser('The best answer is: (C).') == 'C'
    assert parser('The best answer is: (Z).') is None
