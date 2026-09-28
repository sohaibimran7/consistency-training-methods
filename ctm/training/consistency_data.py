"""Paired-prompt datum construction for internal-consistency training (ACT/AttCT/MLPCT).

Boundary-finding helpers ported from https://github.com/c-wei/AttCT
``data/attct_datasets.py`` @ 79527cf (2026-07-10).

Consistency methods train on paired PROMPTS (no assistant response): a variant
prompt and its reference counterpart, formatted with the generation header so
the model is primed to generate. Each JSONL sample carries both sides:

    {"variant_messages": [...], "reference_messages": [...]}

The reference user content must appear verbatim inside the variant prompt (cue
wrapped around the question). Samples where it doesn't (e.g. answer-order
perturbations) raise ValueError — callers skip and count them.

Token alignment uses the HF tokenizer directly (chat template + offset mapping),
not the cookbook renderer: offsets are only defined against the formatted
string, and these datums are consumed exclusively by the HF-native LocalBackend,
so there is no cross-backend token-parity concern.
"""

from typing import Any, Optional

import torch
from tinker import types

DEFAULT_REFERENCE_FIELD = "reference_messages"
DEFAULT_VARIANT_FIELD = "variant_messages"


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
    if not isinstance(content_text, str) or not content_text.strip():
        raise ValueError("content_text must be a non-empty string")

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
    start_index = next(
        (i for i, (_tok_s, tok_e) in enumerate(offsets) if tok_e > content_char_start),
        None,
    )
    if start_index is None:
        raise ValueError("tokenizer produced no token overlapping content_text")
    # First token whose start >= content_char_end (i.e. fully past the content)
    end_index = next((i for i, (tok_s, tok_e) in enumerate(offsets) if tok_s >= content_char_end), len(token_ids))
    content_len = end_index - start_index
    if content_len <= 0:
        raise ValueError("content_text aligned to an empty token window")
    return token_ids, start_index, content_len


def _prompt_messages(messages: list[dict]) -> list[dict]:
    """Messages up to (and including) the last user turn — consistency training is prompt-only."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt messages must be a non-empty list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"prompt message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise ValueError(f"prompt message {index} must have a non-empty string role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"prompt message {index} must have non-empty string content")
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            return messages[: i + 1]
    raise ValueError("no user message in sample")


def _format_prompt(tokenizer, messages: list[dict]) -> str:
    """Chat-format a prompt with the generation header; plain-text join for template-less tokenizers."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(str(m["content"]) for m in messages)


def build_consistency_datum(
    tokenizer,
    sample: dict,
    *,
    reference_field: str = DEFAULT_REFERENCE_FIELD,
    variant_field: str = DEFAULT_VARIANT_FIELD,
    alignment_text_field: Optional[str] = None,
) -> types.Datum:
    """Turn one reference/variant prompt pair into a paired Datum.

    The datum's ``model_input`` is the variant prompt; ``loss_fn_inputs``
    carries the clean prompt tokens and the alignment indices the consistency
    losses slice with (all as 1-element int tensors except ``clean_tokens``):

        clean_tokens, start_index, clean_start_index, clean_len, match_len

    Raises ValueError for samples that can't be aligned (missing keys, reference
    content not contained verbatim in the variant prompt).
    """
    try:
        reference_messages = sample[reference_field]
        variant_messages = sample[variant_field]
    except KeyError as e:
        raise ValueError(f"consistency sample missing {e.args[0]!r} key") from e

    reference_prompt = _prompt_messages(reference_messages)
    variant_prompt = _prompt_messages(variant_messages)
    if alignment_text_field is None:
        content = reference_prompt[-1]["content"]
    else:
        content = sample.get(alignment_text_field)
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                f"consistency sample needs non-empty string {alignment_text_field!r} for explicit alignment"
            )

    clean_formatted = _format_prompt(tokenizer, reference_prompt)
    variant_formatted = _format_prompt(tokenizer, variant_prompt)

    clean_ids, clean_start_index, clean_len = find_content_token_boundary(clean_formatted, content, tokenizer)
    variant_ids, start_index, _ = find_content_token_boundary(variant_formatted, content, tokenizer)
    match_len = longest_matching_suffix_len(clean_ids, variant_ids)

    def _scalar(v: int) -> types.TensorData:
        return types.TensorData.from_torch(torch.tensor([v], dtype=torch.long))

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=variant_ids),
        loss_fn_inputs={
            "clean_tokens": types.TensorData.from_torch(torch.tensor(clean_ids, dtype=torch.long)),
            "start_index": _scalar(start_index),
            "clean_start_index": _scalar(clean_start_index),
            "clean_len": _scalar(clean_len),
            "match_len": _scalar(match_len),
        },
    )


def build_consistency_datums(
    tokenizer,
    samples: list[dict],
    *,
    reference_field: str = DEFAULT_REFERENCE_FIELD,
    variant_field: str = DEFAULT_VARIANT_FIELD,
    alignment_text_field: Optional[str] = None,
) -> tuple[list[types.Datum], int]:
    """Build datums for all alignable samples. Returns (datums, n_skipped)."""

    datums, audit = build_consistency_datums_with_audit(
        tokenizer,
        samples,
        reference_field=reference_field,
        variant_field=variant_field,
        alignment_text_field=alignment_text_field,
    )
    skipped = int(audit["skipped_rows"])
    if skipped:
        first_error = audit.get("first_error")
        if isinstance(first_error, dict) and isinstance(first_error.get("message"), str):
            print(
                f"consistency data: skipped {skipped}/{len(samples)} unalignable samples "
                f"(first: {first_error['message']})"
            )
    return datums, skipped


def _integer_summary(values: list[int]) -> dict[str, int] | None:
    """Return compact, JSON-safe descriptive statistics for a non-empty list."""

    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def build_consistency_datums_with_audit(
    tokenizer,
    samples: list[dict],
    *,
    reference_field: str = DEFAULT_REFERENCE_FIELD,
    variant_field: str = DEFAULT_VARIANT_FIELD,
    alignment_text_field: Optional[str] = None,
) -> tuple[list[types.Datum], dict[str, Any]]:
    """Build paired datums and emit a compact alignment audit.

    The ordinary datum builder deliberately skips malformed / unalignable rows
    so exploratory jobs can make progress.  Recovery runs need a stronger
    contract: their loss must cover the complete clean user prompt rather than
    merely a shared question substring.  This audit records the facts needed
    to enforce that contract without storing prompt text in logs.
    """

    datums: list[types.Datum] = []
    first_error: dict[str, Any] | None = None
    full_reference_suffixes: list[bool] = []
    match_covers_content: list[bool] = []
    variant_token_counts: list[int] = []
    clean_token_counts: list[int] = []
    starts: list[int] = []
    clean_starts: list[int] = []
    content_lengths: list[int] = []
    match_lengths: list[int] = []

    for row_index, sample in enumerate(samples, start=1):
        try:
            # Check the source-level invariant separately from token alignment.
            # An explicit question-only alignment can produce a technically
            # valid datum while still omitting the actual bias cue from the
            # consistency window; strict recovery intentionally rejects that.
            reference_prompt = _prompt_messages(sample[reference_field])
            variant_prompt = _prompt_messages(sample[variant_field])
            full_reference_suffixes.append(
                variant_prompt[-1]["content"].endswith(reference_prompt[-1]["content"])
            )
            datum = build_consistency_datum(
                tokenizer,
                sample,
                reference_field=reference_field,
                variant_field=variant_field,
                alignment_text_field=alignment_text_field,
            )
            scalar = lambda name: int(datum.loss_fn_inputs[name].to_torch()[0])  # noqa: E731
            clean_len = scalar("clean_len")
            match_len = scalar("match_len")
            starts.append(scalar("start_index"))
            clean_starts.append(scalar("clean_start_index"))
            content_lengths.append(clean_len)
            match_lengths.append(match_len)
            match_covers_content.append(match_len >= clean_len)
            variant_token_counts.append(len(datum.model_input.to_ints()))
            clean_token_counts.append(len(datum.loss_fn_inputs["clean_tokens"].to_torch()))
            datums.append(datum)
        except KeyError as exc:
            # Preserve the lenient builder's historical behavior for malformed
            # rows: record one skipped pair rather than leaking a raw KeyError.
            if first_error is None:
                first_error = {"row_index": row_index, "message": f"consistency sample missing {exc.args[0]!r} key"}
        except ValueError as e:
            if first_error is None:
                first_error = {"row_index": row_index, "message": str(e)}

    attempted = len(samples)
    aligned = len(datums)
    skipped = attempted - aligned
    all_full_suffixes = len(full_reference_suffixes) == attempted and all(full_reference_suffixes)
    all_match_covers_content = aligned == attempted and all(match_covers_content)
    return datums, {
        "attempted_rows": attempted,
        "aligned_rows": aligned,
        "skipped_rows": skipped,
        "first_error": first_error,
        "alignment_text": {
            "field": alignment_text_field,
            "uses_complete_reference_user_message": alignment_text_field is None,
        },
        "checks": {
            "all_pairs_aligned": skipped == 0,
            "all_variants_end_with_complete_reference_user_message": all_full_suffixes,
            "all_matching_suffixes_cover_content_window": all_match_covers_content,
        },
        "token_windows": {
            "variant_tokens": _integer_summary(variant_token_counts),
            "clean_prompt_tokens": _integer_summary(clean_token_counts),
            "variant_content_start_index": _integer_summary(starts),
            "clean_content_start_index": _integer_summary(clean_starts),
            "content_length": _integer_summary(content_lengths),
            "matching_suffix_length": _integer_summary(match_lengths),
        },
    }


def require_full_reference_suffix_alignment(audit: dict[str, Any]) -> None:
    """Fail closed unless a run aligns every complete clean user prompt.

    This is deliberately stronger than merely requiring non-empty datums.  It
    prevents the historical failure mode where a question-only alignment field
    allowed training to optimize terminal boilerplate while excluding the bias
    cue from the compared window.
    """

    alignment_text = audit.get("alignment_text")
    checks = audit.get("checks")
    if not isinstance(alignment_text, dict) or not isinstance(checks, dict):
        raise ValueError("invalid consistency alignment audit")
    if not alignment_text.get("uses_complete_reference_user_message"):
        raise ValueError(
            "strict consistency pairing requires the complete reference user message; "
            "remove alignment_text_field"
        )
    failures = [
        label
        for label in (
            "all_pairs_aligned",
            "all_variants_end_with_complete_reference_user_message",
            "all_matching_suffixes_cover_content_window",
        )
        if checks.get(label) is not True
    ]
    if failures:
        attempted = audit.get("attempted_rows")
        aligned = audit.get("aligned_rows")
        raise ValueError(
            "strict consistency pairing failed "
            f"({aligned}/{attempted} aligned; failed checks: {', '.join(failures)})"
        )
