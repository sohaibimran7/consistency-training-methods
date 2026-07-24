"""Renderer/tokenizer helpers selected by the compute backend.

Tinker runs use the service's recommended cookbook renderer. Local runs use the
Hugging Face tokenizer's own chat template, whether sampling is performed by
``transformers.generate`` or vLLM. This keeps the managed service registry out
of the local model-compatibility path.

Canonical home of helpers previously in ``cot_transparency.apis.tinker.common``
(``get_renderer_and_tokenizer``) and ``...tinker.inference`` (``decode_response``,
``parse_response_text``); those modules now re-export from here.
"""

from typing import Any, Literal

import torch
from tinker import types
from tinker_cookbook.renderers.base import TrainOnWhat
from transformers import AutoTokenizer

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers.base import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _flat_token_ids(value: Any, *, operation: str) -> list[int]:
    """Normalize one unbatched HF chat-template result to integer token IDs."""

    if isinstance(value, dict):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) or isinstance(token, bool) for token in value):
        raise TypeError(f"Hugging Face chat template returned invalid token IDs while {operation}")
    return value


class HuggingFaceChatTemplateRenderer:
    """Minimal renderer adapter around a Hugging Face tokenizer chat template.

    CTM still uses ``tinker.ModelInput`` as its backend-neutral token container;
    no Tinker registry or service behavior is involved in constructing it.
    """

    def __init__(self, tokenizer):
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError(
                f"local model tokenizer {getattr(tokenizer, 'name_or_path', '<unknown>')!r} has no chat template"
            )
        self.tokenizer = tokenizer

    def _apply(self, messages: list[dict], *, add_generation_prompt: bool, operation: str) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
        return _flat_token_ids(rendered, operation=operation)

    def build_generation_prompt(
        self,
        messages: list[dict],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> types.ModelInput:
        if role != "assistant":
            raise NotImplementedError("Hugging Face chat-template rendering currently generates assistant turns only")
        tokens = self._apply(messages, add_generation_prompt=True, operation="building a generation prompt")
        if prefill:
            tokens.extend(self.tokenizer.encode(prefill, add_special_tokens=False))
        return types.ModelInput.from_ints(tokens=tokens)

    def build_supervised_example(
        self,
        messages: list[dict],
        train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    ) -> tuple[types.ModelInput, torch.Tensor]:
        if train_on_what != TrainOnWhat.LAST_ASSISTANT_MESSAGE:
            raise NotImplementedError(
                "the local Hugging Face renderer currently supports train_on_what=last_assistant_message only"
            )
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("supervised chat data must end with an assistant message")

        prefix = self._apply(
            messages[:-1],
            add_generation_prompt=True,
            operation="building the supervised prompt prefix",
        )
        full = self._apply(messages, add_generation_prompt=False, operation="building the supervised example")
        if len(full) <= len(prefix) or full[: len(prefix)] != prefix:
            raise ValueError(
                "the Hugging Face chat template does not preserve the generation-prompt prefix for a completed "
                "assistant message; refusing to guess the supervised loss mask"
            )
        weights = torch.tensor([0.0] * len(prefix) + [1.0] * (len(full) - len(prefix)))
        return types.ModelInput.from_ints(tokens=full), weights

    def get_stop_sequences(self) -> list[int]:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos, int) and not isinstance(eos, bool):
            return [eos]
        if isinstance(eos, (list, tuple)):
            return [token for token in eos if isinstance(token, int) and not isinstance(token, bool)]
        return []

    def parse_response(self, response: list[int]):
        # Prefer a tokenizer-provided response schema when available. This is a
        # generic Transformers capability for structured/reasoning chat formats.
        if getattr(self.tokenizer, "response_schema", None) is not None:
            parsed = self.tokenizer.parse_response(response)
            if isinstance(parsed, dict):
                return parsed, None

        # Some channel-based tokenizers predate response_schema. Detect the
        # protocol from the decoded tokens rather than the model name so local
        # support remains capability-based.
        raw = self.tokenizer.decode(response, skip_special_tokens=False)
        final_marker = "<|channel|>final<|message|>"
        if final_marker in raw:
            content = raw.rsplit(final_marker, 1)[1]
            for terminator in ("<|return|>", "<|end|>"):
                content = content.split(terminator, 1)[0]
            return {"role": "assistant", "content": content.strip()}, None

        content = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        return {"role": "assistant", "content": content}, None


def get_renderer_and_tokenizer(model: str, *, source: Literal["tinker", "hf"]):
    """
    Get the appropriate renderer and tokenizer for a model.

    The renderer handles chat template formatting and knows how to:
    - Build supervised examples (tokens + loss weights)
    - Build generation prompts
    - Parse responses
    """
    if source == "tinker":
        tokenizer = get_tokenizer(model)
        renderer_name = model_info.get_recommended_renderer_name(model)
        renderer = renderers.get_renderer(renderer_name, tokenizer)
        return renderer, tokenizer
    if source == "hf":
        tokenizer = AutoTokenizer.from_pretrained(model)
        return HuggingFaceChatTemplateRenderer(tokenizer), tokenizer
    raise ValueError(f"unknown renderer source {source!r}; expected 'tinker' or 'hf'")


def parse_response_text(parsed_msg, tokenizer, tokens) -> str:
    """Extract assistant text from a parsed response message.

    gpt-oss returns structured list content (channel-tagged Thinking/Text parts);
    Llama returns a plain string. ``get_text_content`` handles both (concatenating
    the Text parts, stripping thinking). A plain ``parsed_msg["content"]`` returns
    the raw list for gpt-oss, so any downstream string op (``.strip()``, regex,
    JSON) breaks. Falls back to decoding raw tokens when parsing fails.
    """
    if not parsed_msg:
        return tokenizer.decode(tokens)
    try:
        return get_text_content(parsed_msg) or ""
    except Exception:  # noqa: BLE001
        c = parsed_msg.get("content", "")
        return c if isinstance(c, str) else tokenizer.decode(tokens)


def decode_response(renderer, tokenizer, tokens) -> str:
    """Robustly turn sampled tokens into assistant text.

    Wraps BOTH ``renderer.parse_response`` (which raises RendererError for gpt-oss when a
    sequence carries >1 stop token) and ``get_text_content`` (which can KeyError on
    malformed structured content) so one bad sequence degrades to the decoded tokens
    instead of aborting the caller. Prefer this over a bare ``parse_response`` +
    ``parse_response_text`` pair, which leaves the parse call itself unguarded.
    """
    try:
        parsed_msg, _ = renderer.parse_response(tokens)
    except Exception:  # noqa: BLE001
        return tokenizer.decode(tokens)
    return parse_response_text(parsed_msg, tokenizer, tokens)
