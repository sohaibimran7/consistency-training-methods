"""Offline regression tests for the restart CPU gate, without model access."""
from types import SimpleNamespace
import pytest
from experiments.rmct_restart_20260928.restart_preflight import identity, thinking_probe


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking, return_dict=False):
        if not tokenize:
            return 'prompt<think>' if enable_thinking else 'prompt<think></think>'
        return [1, 2] if enable_thinking else [1, 3]

    def encode(self, text, add_special_tokens):
        return [1, 2]


def renderer(tokens):
    return SimpleNamespace(build_generation_prompt=lambda messages:
        SimpleNamespace(to_ints=lambda: tokens))


def test_native_thinking_matches():
    assert thinking_probe(renderer([1, 2]), Tokenizer(), 'qwen')['enable_thinking']


def test_disabled_production_rejected():
    with pytest.raises(ValueError, match='Production renderer'):
        thinking_probe(renderer([1, 3]), Tokenizer(), 'qwen')


def test_tokenizer_mapping_is_normalized_without_changing_tokens():
    class MappingTokenizer(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            result = super().apply_chat_template(*args, **kwargs)
            return {'input_ids': result} if kwargs['tokenize'] else result
    assert thinking_probe(renderer([1, 2]), MappingTokenizer(), 'qwen')['enable_thinking']


def test_noop_toggle_rejected():
    class Noop(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2]
    with pytest.raises(ValueError, match='toggle'):
        thinking_probe(renderer([1, 2]), Noop(), 'qwen')


def test_hash_actual_bytes(tmp_path):
    path = tmp_path / 'weights'
    path.write_bytes(b'abc')
    result = identity(path)
    assert result['bytes'] == 3
    assert result['sha256'] == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
