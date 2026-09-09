"""Compatibility fixes for pinned ``mcq-bias`` answer parsing.

The pinned parser only accepts labels A-J, while HLE multiple-choice records
can contain up to 20 options. Install an A-Z extension without replacing the
upstream ``prompt_family`` contract used by current native scorers.
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

    native_parse_answer = parsers.parse_answer

    def cot_answer_parser(model_answer: str) -> str | None:
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

    def _clean_answer(model_answer: str) -> str:
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", model_answer)
        return re.sub(
            r"\$?\\boxed\{(?:\\text(?:bf)?\{)?([^}]*)\}?\}\$?",
            r"\1",
            cleaned,
        )

    def _extended_irpan_final_answer(cleaned: str) -> str | None:
        """Extend Irpan's exact final-answer line while retaining its grammar."""

        nonempty_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not nonempty_lines:
            return None
        match = re.fullmatch(
            r"ANSWER\s*:\s*\(?([A-Z])\)?[.,]?",
            nonempty_lines[-1],
            flags=re.IGNORECASE,
        )
        return match.group(1).upper() if match else None

    def parse_answer(model_answer: str, prompt_family: str = "chua") -> str | None:
        """Preserve upstream families while extending unparsed labels to A-Z."""

        # Calling upstream first preserves its default behavior and validates
        # supported prompt families. Its global ``cot_answer_parser`` binding
        # is replaced below, so Chua's fallback gains A-Z labels too.
        parsed = native_parse_answer(model_answer, prompt_family=prompt_family)
        if parsed is not None or prompt_family != "irpan":
            return parsed

        # Native Irpan handling recognizes only an A-J label on its final
        # ``ANSWER: <label>`` line. Keep that exact form, extending its label
        # alphabet only after the upstream parser has declined it.
        return _extended_irpan_final_answer(_clean_answer(model_answer))

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
