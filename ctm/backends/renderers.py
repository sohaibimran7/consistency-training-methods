"""Renderer/tokenizer helpers shared by ALL backends.

tinker_cookbook's renderers are template+tokenizer logic with no service
dependency, so both the Tinker and local backends use them — which keeps prompt
token streams identical across backends (including gpt-oss channel tags).

Canonical home of helpers previously in ``cot_transparency.apis.tinker.common``
(``get_renderer_and_tokenizer``) and ``...tinker.inference`` (``decode_response``,
``parse_response_text``); those modules now re-export from here.
"""

from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers.base import get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


def get_renderer_and_tokenizer(model: str):
    """
    Get the appropriate renderer and tokenizer for a model.

    The renderer handles chat template formatting and knows how to:
    - Build supervised examples (tokens + loss weights)
    - Build generation prompts
    - Parse responses
    """
    tokenizer = get_tokenizer(model)
    renderer_name = model_info.get_recommended_renderer_name(model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return renderer, tokenizer


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
