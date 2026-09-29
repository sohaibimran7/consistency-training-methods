"""BCT-only response decoding. Evaluation's final-answer decoder is intentional.

Keep the assistant message, including structured thinking, rather than extracting
only its display text. Unknown or malformed structures fail closed.
"""

from copy import deepcopy

from ctm.backends.renderers import HuggingFaceChatTemplateRenderer


TARGET_POLICY = "complete-assistant-reasoning-and-final-v2"


def content_spans(message):
    content = message.get("content")
    if isinstance(content, str):
        spans = [content]
    elif isinstance(content, list) and content:
        spans = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"thinking", "text"}:
                raise ValueError("unsupported BCT assistant content part")
            value = part.get("thinking" if part["type"] == "thinking" else "text")
            if not isinstance(value, str):
                raise ValueError("invalid BCT assistant content part")
            spans.append(value)
    else:
        raise ValueError("BCT target requires non-empty assistant content")
    for field in ("thinking", "reasoning", "reasoning_content"):
        reasoning = message.get(field)
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ValueError(f"invalid BCT {field}")
            spans.insert(0, reasoning)
    if not any(span.strip() for span in spans):
        raise ValueError("BCT target requires non-empty assistant content")
    if message.get("role") != "assistant" or message.get("tool_calls"):
        raise ValueError("BCT target must be a complete assistant response, not a tool call")
    return spans


def decode_bct_response(renderer, tokenizer, tokens, *, prompt):
    if isinstance(renderer, HuggingFaceChatTemplateRenderer):
        # Do not call HF parse_response: it deliberately selects only final.
        body = list(tokens)
        stops = renderer.get_stop_sequences()
        while body and body[-1] in stops:
            body.pop()
        raw = tokenizer.decode(body, skip_special_tokens=False)
        if raw.startswith("<|channel>thought\n"):
            # Gemma's unified processor removes channel blocks embedded in
            # content. Its native assistant representation has separate fields.
            reasoning, boundary, final = raw[len("<|channel>thought\n"):].partition("<channel|>")
            if not boundary or "<|channel>" in final or "<channel|>" in final:
                raise ValueError("malformed BCT Gemma thought channel")
            message = {"role": "assistant", "content": final, "reasoning": reasoning}
        elif "<|channel|>" in raw:
            from tinker_cookbook.renderers.gpt_oss import GptOssRenderer

            parsed, termination = GptOssRenderer(tokenizer).parse_response(list(tokens))
            if not termination.is_clean:
                raise ValueError("malformed BCT Harmony response")
            content_spans(parsed)
            parts = parsed["content"]
            if not isinstance(parts, list):
                raise ValueError("BCT Harmony response must preserve channel structure")
            message = {
                "role": "assistant",
                "content": "".join(p["text"] for p in parts if p["type"] == "text"),
                # Native GPT-OSS HF template consumes `thinking`. Rendering
                # is checked below; unsupported templates fail closed.
                "thinking": "".join(p["thinking"] for p in parts if p["type"] == "thinking"),
            }
        else:
            # Qwen generation templates may prefill the opening think tag.
            prefix = tokenizer.decode(prompt.to_ints(), skip_special_tokens=False)
            if "</think>" in raw and "<think>" not in raw:
                if "<think>" not in prefix or prefix.rfind("<think>") < prefix.rfind("</think>"):
                    raise ValueError("BCT closing think tag has no provable opening")
                raw = "<think>" + prefix.rsplit("<think>", 1)[1] + raw
            message = {"role": "assistant", "content": raw}
    else:
        message, termination = renderer.parse_response(list(tokens))
        if termination is not None and hasattr(termination, "is_clean") and not termination.is_clean:
            raise ValueError("malformed BCT response")
    content_spans(message)
    return deepcopy(message)


def verify_supervised_preservation(renderer, tokenizer, messages):
    """Refuse templates that erase reasoning or give it zero training weight."""
    tokens, weights = renderer.build_supervised_example(messages)
    supervised = [token for token, weight in zip(tokens.to_ints(), weights.tolist(), strict=True) if weight > 0]
    text = tokenizer.decode(supervised)
    for span in content_spans(messages[-1]):
        # Qwen's opening marker can belong to the generation prompt, not the
        # generated response. Only exempt that exact, independently rendered
        # prefill; never exempt any reasoning text or the closing marker.
        if span.startswith("<think>"):
            prefix = tokenizer.decode(renderer.build_generation_prompt(messages[:-1]).to_ints())
            prefill = "<think>" + prefix.rsplit("<think>", 1)[-1]
            if "<think>" in prefix and not prefill[len("<think>"):].strip() and span.startswith(prefill):
                span = span[len(prefill):]
        if span.strip() and span.strip() not in text:
            raise ValueError("BCT renderer dropped or masked part of the complete assistant response")
