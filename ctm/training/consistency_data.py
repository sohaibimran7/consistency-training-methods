"""Paired-prompt datum construction for internal-consistency training (ACT/AttCT/MLPCT).

Boundary-finding helpers ported from https://github.com/c-wei/AttCT
``data/attct_datasets.py`` @ 79527cf (2026-07-10).

Consistency methods train on paired PROMPTS (no assistant response): a biased
prompt and its clean counterpart, formatted with the generation header so the
model is primed to generate. Each JSONL sample carries both sides — the shape
mcq-bias frozen rows already have:

    {"biased_messages": [...], "unbiased_messages": [...]}

The clean user content must appear verbatim inside the biased prompt (cue
wrapped around the question). Samples where it doesn't (e.g. answer-order
perturbations) raise ValueError — callers skip and count them.

Token alignment uses the HF tokenizer directly (chat template + offset mapping),
not the cookbook renderer: offsets are only defined against the formatted
string, and these datums are consumed exclusively by the HF-native LocalBackend,
so there is no cross-backend token-parity concern.
"""

from typing import Optional

import torch
from tinker import types


def longest_matching_suffix_len(seq_a: list, seq_b: list) -> int:
    """Length of the longest token suffix on which seq_a and seq_b agree.

    This is the "matching suffix" used by ACT (Irpan et al. 2025) — the natural
    training window for paired clean/wrapped prompts because activations at
    these positions are computed under different prefixes but must agree for
    the model to behave consistently.
    """
    n = min(len(seq_a), len(seq_b))
    match = 0
    for i in range(1, n + 1):
        if seq_a[-i] != seq_b[-i]:
            break
        match = i
    return match


def find_content_token_boundary(formatted_str: str, content_text: str, tokenizer) -> tuple[list[int], int, int]:
    """Find the token-level start index and length of content_text within
    the already-formatted (chat-template-applied) string.

    Uses offset_mapping so results are correct even when the tokenizer produces
    different token IDs for the same text depending on context (e.g. BPE merges
    differ after chat header tokens). Requires a fast tokenizer.

    Returns:
        (token_ids, start_index, content_len) where:
            token_ids   — full tokenized sequence as a list of ints
            start_index — index of first token that belongs to content_text
            content_len — number of tokens that span content_text
    """
    # Try exact match first, then stripped (chat templates sometimes trim whitespace)
    idx = formatted_str.find(content_text)
    if idx == -1:
        content_text = content_text.strip()
        idx = formatted_str.find(content_text)
        if idx == -1:
            raise ValueError("content_text not found in formatted_str")
    content_char_start = idx
    content_char_end = content_char_start + len(content_text)

    encoding = tokenizer(
        formatted_str,
        add_special_tokens=False,  # BOS already present in formatted_str
        return_offsets_mapping=True,
    )
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]  # (char_start, char_end) per token

    # First token that overlaps content_char_start (tok_e > content_char_start).
    # Overlap rather than tok_s >= content_char_start handles BPE merges where
    # the tokenizer fuses the last char(s) of the prefix with the first char(s)
    # of content_text into a single token — that merged token partially covers
    # the content region and must be included.
    start_index = next(i for i, (tok_s, tok_e) in enumerate(offsets) if tok_e > content_char_start)
    # First token whose start >= content_char_end (i.e. fully past the content)
    end_index = next((i for i, (tok_s, tok_e) in enumerate(offsets) if tok_s >= content_char_end), len(token_ids))
    return token_ids, start_index, end_index - start_index


def _prompt_messages(messages: list[dict]) -> list[dict]:
    """Messages up to (and including) the last user turn — consistency training is prompt-only."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return messages[: i + 1]
    raise ValueError("no user message in sample")


def _format_prompt(tokenizer, messages: list[dict]) -> str:
    """Chat-format a prompt with the generation header; plain-text join for template-less tokenizers."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(str(m["content"]) for m in messages)


def build_consistency_datum(tokenizer, sample: dict) -> types.Datum:
    """Turn one {biased_messages, unbiased_messages} sample into a paired Datum.

    The datum's ``model_input`` is the biased (wrapped) prompt; ``loss_fn_inputs``
    carries the clean prompt tokens and the alignment indices the consistency
    losses slice with (all as 1-element int tensors except ``clean_tokens``):

        clean_tokens, start_index, clean_start_index, clean_len, match_len

    Raises ValueError for samples that can't be aligned (missing keys, clean
    content not contained verbatim in the biased prompt).
    """
    try:
        biased_messages = sample["biased_messages"]
        unbiased_messages = sample["unbiased_messages"]
    except KeyError as e:
        raise ValueError(f"consistency sample missing {e.args[0]!r} key") from e

    clean_prompt = _prompt_messages(unbiased_messages)
    biased_prompt = _prompt_messages(biased_messages)
    content = str(clean_prompt[-1]["content"])

    clean_formatted = _format_prompt(tokenizer, clean_prompt)
    biased_formatted = _format_prompt(tokenizer, biased_prompt)

    clean_ids, clean_start_index, clean_len = find_content_token_boundary(clean_formatted, content, tokenizer)
    biased_ids, start_index, _ = find_content_token_boundary(biased_formatted, content, tokenizer)
    match_len = longest_matching_suffix_len(clean_ids, biased_ids)

    def _scalar(v: int) -> types.TensorData:
        return types.TensorData.from_torch(torch.tensor([v], dtype=torch.long))

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=biased_ids),
        loss_fn_inputs={
            "clean_tokens": types.TensorData.from_torch(torch.tensor(clean_ids, dtype=torch.long)),
            "start_index": _scalar(start_index),
            "clean_start_index": _scalar(clean_start_index),
            "clean_len": _scalar(clean_len),
            "match_len": _scalar(match_len),
        },
    )


def build_consistency_datums(tokenizer, samples: list[dict]) -> tuple[list[types.Datum], int]:
    """Build datums for all alignable samples. Returns (datums, n_skipped)."""
    datums: list[types.Datum] = []
    skipped = 0
    first_error: Optional[str] = None
    for sample in samples:
        try:
            datums.append(build_consistency_datum(tokenizer, sample))
        except ValueError as e:
            skipped += 1
            if first_error is None:
                first_error = str(e)
    if skipped and first_error:
        print(f"consistency data: skipped {skipped}/{len(samples)} unalignable samples (first: {first_error})")
    return datums, skipped
