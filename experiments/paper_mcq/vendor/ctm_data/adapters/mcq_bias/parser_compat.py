"""Compatibility fixes for pinned ``mcq-bias`` answer parsing.

The pinned parser only accepts labels A-J, while HLE multiple-choice records
can contain up to 20 options. Install an A-Z parser before importing upstream
tasks/scorers so high-option questions remain parseable. Explicit reasoning
blocks are excluded and terminal canonical answers take precedence over cues.
"""

from __future__ import annotations

import re
import sys
from string import ascii_uppercase

_MARKER = "_ctm_extended_answer_labels"


def install_extended_answer_parser() -> None:
    """Extend the pinned canonical answer parser from A-J to A-Z."""

    import mcq_bias.parsers as parsers

    if getattr(parsers, _MARKER, False):
        return

    def cot_answer_parser(model_answer: str) -> str | None:
        # Never score a quoted or provisional answer from a reasoning block.
        if "</think>" in model_answer:
            model_answer = model_answer.rsplit("</think>", 1)[1]
        elif "<think>" in model_answer:
            return None  # Unfinished reasoning is not a final answer.
        # The explicit terminal answer takes precedence over quoted answer
        # phrases in explanatory prose (including prose after </think>).
        terminal = re.search(
            r"(?:^|\n)\s*The best answer is:\s*\(([A-Z])\)\.?\s*$",
            model_answer,
            re.IGNORECASE,
        )
        if terminal:
            return terminal.group(1).upper()
        for break_word in parsers.BREAK_WORDS:
            if break_word not in model_answer:
                continue
            last_item = model_answer.split(break_word)[-1].lstrip()
            if not last_item:
                continue

            possible_indicators = list(ascii_uppercase)
            possible_indicators_lower = [indicator.lower() for indicator in possible_indicators]
            possible_indicators_re = "|".join(
                possible_indicators + possible_indicators_lower
            )
            pattern = (
                rf"^(?:[Oo]ption |[Ss]tatement )?\(?({possible_indicators_re})\)?"
                r"(\s|\)|\.|\,|$)+.*$"
            )
            match = re.search(pattern, last_item, re.MULTILINE)
            if match:
                candidate = match.group(1)
                if candidate in possible_indicators:
                    return candidate
                if candidate in possible_indicators_lower:
                    return candidate.upper()
            return None
        return None

    def parse_answer(model_answer: str) -> str | None:
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", model_answer)
        cleaned = re.sub(
            r"\$?\\boxed\{(?:\\text(?:bf)?\{)?([^}]*)\}?\}\$?",
            r"\1",
            cleaned,
        )
        return cot_answer_parser(cleaned)

    parsers.cot_answer_parser = cot_answer_parser
    parsers.parse_answer = parse_answer
    setattr(parsers, _MARKER, True)

    # These modules bind parse_answer with ``from ... import``. Patch them too
    # if another caller imported them before this compatibility layer.
    for module_name in (
        "mcq_bias.scorers",
        "mcq_bias.pipeline.wrong_arguments",
    ):
        module = sys.modules.get(module_name)
        if module is not None:
            module.parse_answer = parse_answer


__all__ = ["install_extended_answer_parser"]
