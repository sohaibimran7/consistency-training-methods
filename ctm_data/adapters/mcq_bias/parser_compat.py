"""Compatibility fixes for pinned ``mcq-bias`` answer parsing.

The pinned parser only accepts labels A-J, while HLE multiple-choice records
can contain up to 20 options. Install an A-Z parser before importing upstream
tasks/scorers so high-option questions remain parseable. Explicit reasoning
blocks are excluded and terminal canonical answers take precedence over cues.
"""

from __future__ import annotations

import sys
from string import ascii_uppercase

from .terminal_answer import parse_terminal_first

_MARKER = "_ctm_extended_answer_labels"
_VERSION = 3


def install_extended_answer_parser() -> None:
    """Extend the pinned canonical answer parser from A-J to A-Z."""

    import mcq_bias.parsers as parsers

    if getattr(parsers, _MARKER, None) == _VERSION:
        return

    def cot_answer_parser(model_answer: str) -> str | None:
        return parse_terminal_first(model_answer, ascii_uppercase, parsers.BREAK_WORDS)

    def parse_answer(model_answer: str) -> str | None:
        return cot_answer_parser(model_answer)

    parsers.cot_answer_parser = cot_answer_parser
    parsers.parse_answer = parse_answer
    setattr(parsers, _MARKER, _VERSION)

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
