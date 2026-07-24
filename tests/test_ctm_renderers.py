"""Backend-specific renderer resolution and generic HF chat-template tests."""

import pytest

from ctm.backends.renderers import HuggingFaceChatTemplateRenderer, get_renderer_and_tokenizer


class _HFTokenizer:
    chat_template = "unit-template"
    eos_token_id = 99
    name_or_path = "unit/hf-model"
    response_schema = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
        assert tokenize is True and return_dict is False
        tokens = [1]
        for message in messages:
            if message["role"] == "user":
                tokens.extend([10, len(message["content"]), 11])
            elif message["role"] == "assistant":
                tokens.extend([20, len(message["content"]), 99])
            else:
                raise ValueError("unsupported role")
        if add_generation_prompt:
            tokens.append(20)
        return tokens

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [len(text)]

    def decode(self, tokens, *, skip_special_tokens):
        if tokens == [43] and not skip_special_tokens:
            return (
                "<|channel|>analysis<|message|>hidden reasoning<|end|>"
                "<|start|>assistant<|channel|>final<|message|>visible answer<|return|>"
            )
        return "answer"


def test_hf_renderer_builds_generation_prompt_and_assistant_only_sft_mask():
    renderer = HuggingFaceChatTemplateRenderer(_HFTokenizer())
    prompt = [{"role": "user", "content": "question"}]
    assert renderer.build_generation_prompt(prompt).to_ints() == [1, 10, 8, 11, 20]

    model_input, weights = renderer.build_supervised_example([*prompt, {"role": "assistant", "content": "answer"}])
    assert model_input.to_ints() == [1, 10, 8, 11, 20, 6, 99]
    assert weights.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert renderer.get_stop_sequences() == [99]


def test_hf_renderer_extracts_final_channel_without_a_model_name_check():
    renderer = HuggingFaceChatTemplateRenderer(_HFTokenizer())
    parsed, _ = renderer.parse_response([43])
    assert parsed == {"role": "assistant", "content": "visible answer"}


def test_hf_renderer_fails_closed_when_template_cannot_prove_the_sft_boundary():
    class NonExtendingTokenizer(_HFTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
            tokens = super().apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                return_dict=return_dict,
            )
            return [77, *tokens] if messages and messages[-1]["role"] == "assistant" else tokens

    renderer = HuggingFaceChatTemplateRenderer(NonExtendingTokenizer())
    with pytest.raises(ValueError, match="refusing to guess the supervised loss mask"):
        renderer.build_supervised_example(
            [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
        )


def test_local_renderer_resolution_uses_hf_without_consulting_tinker(monkeypatch):
    tokenizer = _HFTokenizer()
    monkeypatch.setattr("ctm.backends.renderers.AutoTokenizer.from_pretrained", lambda model: tokenizer)
    monkeypatch.setattr(
        "ctm.backends.renderers.model_info.get_recommended_renderer_name",
        lambda model: pytest.fail("local renderer resolution consulted the Tinker registry"),
    )

    renderer, resolved_tokenizer = get_renderer_and_tokenizer("any/hf-model", source="hf")

    assert isinstance(renderer, HuggingFaceChatTemplateRenderer)
    assert resolved_tokenizer is tokenizer


def test_tinker_renderer_resolution_still_uses_service_recommendation(monkeypatch):
    tokenizer = object()
    renderer = object()
    monkeypatch.setattr("ctm.backends.renderers.get_tokenizer", lambda model: tokenizer)
    monkeypatch.setattr("ctm.backends.renderers.model_info.get_recommended_renderer_name", lambda model: "unit")
    monkeypatch.setattr("ctm.backends.renderers.renderers.get_renderer", lambda name, tok: renderer)

    assert get_renderer_and_tokenizer("managed/model", source="tinker") == (renderer, tokenizer)
