import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gemma_thinking', Path(__file__).parents[1] / 'ctm/backends/gemma_thinking.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Processor:
    chat_template = 'test-template'

    def apply_chat_template(self, messages, *, tokenize, enable_thinking, **kwargs):
        text = ('<|turn>system\n<|think|>\n' if enable_thinking else '') + '<|turn>user\nQuestion<|turn>model\n'
        if not enable_thinking:
            text += '<|channel>thought\n<channel|>'
        return self.encode(text) if tokenize else text

    def encode(self, text, **kwargs):
        return list(text.encode())


def test_attestation_records_distinct_native_tokens():
    record = module.attest_thinking(Processor())
    assert record['enable_thinking'] is True
    assert record['prompt_token_ids']['true'] != record['prompt_token_ids']['false']


def test_ignored_thinking_flag_fails_closed():
    class Ignored(Processor):
        def apply_chat_template(self, messages, **kwargs):
            kwargs['enable_thinking'] = False
            return super().apply_chat_template(messages, **kwargs)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'signature'):
        module.attest_thinking(Ignored())


def test_tokenization_mismatch_fails_closed():
    class Mismatch(Processor):
        def apply_chat_template(self, messages, **kwargs):
            value = super().apply_chat_template(messages, **kwargs)
            return value + [42] if kwargs['tokenize'] else value
    with unittest.TestCase().assertRaisesRegex(ValueError, 'tokenization'):
        module.attest_thinking(Mismatch())


class Tests(unittest.TestCase):
    test_native_tokens = staticmethod(test_attestation_records_distinct_native_tokens)
    test_ignored_flag = staticmethod(test_ignored_thinking_flag_fails_closed)
    test_token_mismatch = staticmethod(test_tokenization_mismatch_fails_closed)
