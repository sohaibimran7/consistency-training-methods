import re

import pytest
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.renderers.qwen3 import Qwen3Renderer

from ctm.backends.renderers import decode_response
from ctm.backends.renderers import HuggingFaceChatTemplateRenderer
from ctm.training.bct_response import decode_bct_response, verify_supervised_preservation


class Tokenizer:
    """Offline lossless tokenizer; actual cookbook renderer/parser/masks under test."""
    def __init__(self):
        self.special = {}

    def encode(self, text, add_special_tokens=False):
        result = []
        for part in re.split(r"(<\|[^>]+\|>|</?think>)", text):
            if re.fullmatch(r"<\|[^>]+\|>|</?think>", part):
                result.append(self.special.setdefault(part, 100000 + len(self.special)))
            else:
                result.extend(map(ord, part))
        return result

    def decode(self, tokens, **kwargs):
        reverse = {v: k for k, v in self.special.items()}
        return "".join(reverse[t] if t in reverse else chr(t) for t in tokens)


@pytest.mark.parametrize("kind", ["gptoss", "qwen"])
def test_real_cookbook_renderer_preserves_and_supervises_reasoning_and_final(kind):
    tokenizer = Tokenizer()
    renderer = GptOssRenderer(tokenizer) if kind == "gptoss" else Qwen3Renderer(tokenizer)
    prompt = [{"role": "user", "content": "PROMPT_SENTINEL"}]
    if kind == "gptoss":
        raw = "<|channel|>analysis<|message|>REASONING_SENTINEL<|end|><|start|>assistant<|channel|>final<|message|>FINAL_SENTINEL<|return|>"
    else:
        raw = "<think>REASONING_SENTINEL</think>FINAL_SENTINEL<|im_end|>"
    tokens = tokenizer.encode(raw)
    assistant = decode_bct_response(renderer, tokenizer, tokens, prompt=renderer.build_generation_prompt(prompt))
    verify_supervised_preservation(renderer, tokenizer, [*prompt, assistant])
    rendered, weights = renderer.build_supervised_example([*prompt, assistant])
    supervised = tokenizer.decode([t for t, w in zip(rendered.to_ints(), weights) if w > 0])
    assert "REASONING_SENTINEL" in supervised
    assert "FINAL_SENTINEL" in supervised
    assert "PROMPT_SENTINEL" not in supervised
    # The independent scoring decoder must continue to select visible final text.
    assert "REASONING_SENTINEL" not in decode_response(renderer, tokenizer, tokens)
    # Exercise the actual target writer path, not only the decoder helper.
    import asyncio
    from ctm.backends.base import SampledSequence
    from ctm.training.bct_targets import PairedPrompt, generate_bct_rows
    from ctm.training.bct_response import TARGET_POLICY

    class Sampler:
        async def sample(self, prompt, **kwargs):
            return [SampledSequence(tokens=tokens, logprobs=[])]
    paired = PairedPrompt("sentinel", prompt, prompt, prompt)
    main, control = asyncio.run(generate_bct_rows(
        [paired], sampler=Sampler(), renderer=renderer, tokenizer=tokenizer, max_tokens=None))
    assert main == control
    assert main[0]["messages"][-1] == assistant
    assert main[0]["target_policy"] == TARGET_POLICY
    assert main[0]["sampled_tokens_sha256"]


def test_unknown_content_part_is_rejected():
    class Renderer:
        def parse_response(self, tokens):
            return {"role": "assistant", "content": [{"type": "unknown", "text": "lost"}]}, None
    with pytest.raises(ValueError, match="unsupported"):
        decode_bct_response(Renderer(), Tokenizer(), [1], prompt=None)


def test_hf_harmony_uses_native_thinking_field_and_keeps_scoring_separate():
    class HFTokenizer(Tokenizer):
        chat_template = "native-thinking-field-contract"
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
            text = ""
            for message in messages:
                text += "<|start|>" + message["role"]
                if message["role"] == "assistant":
                    if message.get("thinking"):
                        text += "<|channel|>analysis<|message|>" + message["thinking"] + "<|end|><|start|>assistant"
                    text += "<|channel|>final<|message|>" + message["content"] + "<|return|>"
                else:
                    text += "<|message|>" + message["content"] + "<|end|>"
            if add_generation_prompt:
                text += "<|start|>assistant"
            return self.encode(text) if tokenize else text
    tokenizer = HFTokenizer()
    stop = tokenizer.encode("<|return|>")[0]
    renderer = HuggingFaceChatTemplateRenderer(tokenizer, stop_token_ids=[stop])
    prompt = [{"role": "user", "content": "PROMPT_SENTINEL"}]
    raw = "<|channel|>analysis<|message|>REASONING_SENTINEL<|end|><|start|>assistant<|channel|>final<|message|>FINAL_SENTINEL<|return|>"
    tokens = tokenizer.encode(raw)
    assistant = decode_bct_response(renderer, tokenizer, tokens, prompt=renderer.build_generation_prompt(prompt))
    assert assistant["thinking"] == "REASONING_SENTINEL"
    assert assistant["content"] == "FINAL_SENTINEL"
    verify_supervised_preservation(renderer, tokenizer, [*prompt, assistant])
    assert decode_response(renderer, tokenizer, tokens) == "FINAL_SENTINEL"


def test_hf_gemma_splits_native_channel_into_reasoning_field():
    class GemmaTokenizer(Tokenizer):
        chat_template = "native-gemma-reasoning-field-contract"
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):
            text = ""
            for message in messages:
                text += "<turn>" + message["role"] + "\n"
                if message.get("reasoning"):
                    text += "<|channel>thought\n" + message["reasoning"] + "<channel|>"
                text += message["content"] + "<turn|>"
            if add_generation_prompt:
                text += "<turn>assistant\n"
            return self.encode(text) if tokenize else text
    tokenizer = GemmaTokenizer()
    renderer = HuggingFaceChatTemplateRenderer(tokenizer, stop_token_ids=[])
    prompt = [{"role": "user", "content": "PROMPT_SENTINEL"}]
    raw = "<|channel>thought\nREASONING_SENTINEL\n<channel|>FINAL_SENTINEL"
    message = decode_bct_response(renderer, tokenizer, tokenizer.encode(raw),
                                  prompt=renderer.build_generation_prompt(prompt))
    assert message["reasoning"] == "REASONING_SENTINEL\n"
    assert message["content"] == "FINAL_SENTINEL"
    verify_supervised_preservation(renderer, tokenizer, [*prompt, message])
    with pytest.raises(ValueError, match="malformed"):
        decode_bct_response(renderer, tokenizer, tokenizer.encode("<|channel>thought\nunclosed"),
                            prompt=renderer.build_generation_prompt(prompt))


@pytest.mark.parametrize("drop_reasoning", [False, True])
def test_only_exact_prefilled_think_wrapper_is_exempt_from_loss_mask(drop_reasoning):
    import torch
    from tinker import types

    tokenizer = Tokenizer()
    class Renderer:
        def build_generation_prompt(self, messages):
            return types.ModelInput.from_ints(tokens=tokenizer.encode("assistant\n<think>\n"))
        def build_supervised_example(self, messages):
            text = "FINAL" if drop_reasoning else "REASONING\n</think>\nFINAL"
            ids = tokenizer.encode(text)
            return types.ModelInput.from_ints(tokens=ids), torch.ones(len(ids))
    messages = [{"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "<think>\nREASONING\n</think>\nFINAL"}]
    if drop_reasoning:
        with pytest.raises(ValueError, match="dropped or masked"):
            verify_supervised_preservation(Renderer(), tokenizer, messages)
    else:
        verify_supervised_preservation(Renderer(), tokenizer, messages)
