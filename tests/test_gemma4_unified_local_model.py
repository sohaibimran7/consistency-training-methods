"""CPU-only contracts for the canonical Gemma 4 unified HF bridge."""

from types import SimpleNamespace

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from ctm.evals import local_model as module


class _Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 7
    pad_token = None
    pad_token_id = None
    padding_side = "right"
    name_or_path = "nested-tokenizer"

    def batch_decode(self, *args, **kwargs):
        return ("batch_decode", args, kwargs)

    def decode(self, *args, **kwargs):
        return ("decode", args, kwargs)

    def convert_ids_to_tokens(self, value):
        return f"tok-{value}"


class _Processor:
    chat_template = "canonical-processor-template"
    name_or_path = "canonical-processor"

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.calls = []

    def __call__(self, *, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def apply_chat_template(self, *args, **kwargs):
        return ("processor-template", args, kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


def _gemma_api():
    return SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified")),
        model_name="/pinned/gemma-snapshot",
        api_key=None,
        tokenizer_call_args={},
    )


def test_gemma4_unified_processor_adapter_preserves_official_text_route(monkeypatch):
    processor = _Processor()
    monkeypatch.setattr(module, "_load_gemma4_unified_processor", lambda source, *, token: processor)
    api = _gemma_api()

    adapter = module.configure_gemma4_unified_hf_api(api)
    output = adapter(["one canonical prompt"], return_tensors="pt", padding=True)

    assert api.processor is processor
    assert api.tokenizer is adapter
    assert adapter.chat_template == "canonical-processor-template"
    assert processor.calls == [
        (
            ["one canonical prompt"],
            {"return_mm_token_type_ids": False, "return_tensors": "pt", "padding": True},
        )
    ]
    assert output == {"input_ids": [1, 2], "attention_mask": [1, 1]}
    assert processor.tokenizer.pad_token == "<eos>"
    assert processor.tokenizer.padding_side == "left"
    assert api.tokenizer_call_args == {"return_mm_token_type_ids": False}
    assert adapter.apply_chat_template([{"role": "user", "content": "x"}]) == (
        "processor-template",
        ([{"role": "user", "content": "x"}],),
        {},
    )
    assert adapter.convert_ids_to_tokens(4) == "tok-4"


def test_gemma4_unified_chat_adapter_converts_inspect_message_models(monkeypatch):
    processor = _Processor()
    monkeypatch.setattr(module, "_load_gemma4_unified_processor", lambda source, *, token: processor)
    adapter = module.configure_gemma4_unified_hf_api(_gemma_api())
    inspect_messages = [
        ChatMessageSystem(content="Be precise."),
        ChatMessageUser(content="Choose one answer."),
    ]

    output = adapter.apply_chat_template(
        inspect_messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    assert output == (
        "processor-template",
        ([
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Choose one answer."},
        ],),
        {"add_generation_prompt": True, "tokenize": False},
    )


def test_gemma4_unified_chat_adapter_handles_batched_mapping_conversations(monkeypatch):
    processor = _Processor()
    monkeypatch.setattr(module, "_load_gemma4_unified_processor", lambda source, *, token: processor)
    adapter = module.configure_gemma4_unified_hf_api(_gemma_api())
    conversations = (
        ({"role": "user", "content": "first"},),
        ({"role": "user", "content": "second"},),
    )

    output = adapter.apply_chat_template(conversation=conversations, tokenize=False)

    assert output == (
        "processor-template",
        (),
        {
            "conversation": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            "tokenize": False,
        },
    )


def test_gemma4_unified_processor_rejects_multimodal_text_inputs(monkeypatch):
    processor = _Processor()
    monkeypatch.setattr(module, "_load_gemma4_unified_processor", lambda source, *, token: processor)
    adapter = module.configure_gemma4_unified_hf_api(_gemma_api())

    with pytest.raises(ValueError, match="return_mm_token_type_ids must be false"):
        adapter(["text only"], return_mm_token_type_ids=True)


def test_gemma4_unified_loader_forces_image_text_auto_class_and_has_no_cap(monkeypatch):
    processor = _Processor()
    resolved = SimpleNamespace(api=_gemma_api())
    captured = {}

    def fake_get_model(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return resolved

    monkeypatch.setattr("inspect_ai.model.get_model", fake_get_model)
    monkeypatch.setattr(module, "_load_gemma4_unified_processor", lambda source, *, token: processor)

    result = module.gemma4_unified_hf_model(
        "hf//pinned/gemma-snapshot",
        model_args={
            module.GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG: True,
            "device": "cuda:0",
            "dtype": "bfloat16",
            "do_sample": True,
        },
        generation_config={"temperature": 1.0, "top_p": 0.95},
    )

    assert result is resolved
    assert captured["model"] == "hf//pinned/gemma-snapshot"
    assert captured["kwargs"]["auto_model_class"] == module.GEMMA4_UNIFIED_AUTO_MODEL_CLASS
    assert captured["kwargs"]["device"] == "cuda:0"
    assert captured["kwargs"]["dtype"] == "bfloat16"
    assert captured["kwargs"]["do_sample"] is True
    assert captured["kwargs"]["config"].max_tokens is None
    assert resolved.api.tokenizer is not None


@pytest.mark.parametrize("field", ["max_tokens", "max_new_tokens", "output_token_cap"])
def test_gemma4_unified_loader_rejects_any_concrete_output_cap(field):
    with pytest.raises(ValueError, match="forbidden output-token cap"):
        module.gemma4_unified_hf_model(
            "hf/google/gemma-4-12B-it",
            model_args={module.GEMMA4_UNIFIED_PROCESSOR_MODEL_ARG: True},
            generation_config={field: 1},
        )
