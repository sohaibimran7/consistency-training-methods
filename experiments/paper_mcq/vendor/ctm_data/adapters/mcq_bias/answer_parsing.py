"""Conservative, dependency-free answer extraction shared with the upstream patch.

This is a syntactic parser, not a judge of reasoning. Quoted, hypothetical and
conflicting declarations are missing answers, not opportunities to guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


def visible_answer_text(response: str) -> str | None:
    """Exclude explicit thinking, including providers that omit the opening tag."""
    depth = 0
    end = 0
    for tag in re.finditer(r"<\s*(/?)think\s*>", response, re.IGNORECASE):
        if tag.group(1):
            depth = max(0, depth - 1)
            end = tag.end()
        else:
            depth += 1
    if depth:
        return None
    text = response[end:]
    # Nested same-character quotes in a quoted formatting instruction are
    # common. Mask the whole instruction, not alternating quote fragments.
    text = re.sub(r'(?im)["“][^\n]*\bgive your answer in the format\b[^\n]*', '', text)
    # Mask quotations/code before removing formatting. Preserve offsets/newlines.
    quoted = r'```[\s\S]*?(?:```|$)|`[^`\n]*(?:`|$)|"[^"\n]*(?:"|$)|“[^”\n]*(?:”|$)|(?<!\w)\x27[^\x27\n]+\x27(?!\w)'
    def mask_quote(match: re.Match[str]) -> str:
        masked = re.sub(r"[^\n]", " ", match.group())
        # A subsequent independent declaration must not inherit attribution
        # from prose preceding a quotation ("the user says ...").
        return masked[:-1] + ";" if masked else masked
    text = re.sub(quoted, mask_quote, text)
    text = re.sub(r"(?m)^\s*>[^\n]*", "", text)
    text = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", lambda m: m.group(1) or m.group(2), text)
    # Leave closing braces in place; label boundaries accept them. This supports
    # nested boxed/text wrappers without consuming option text or later prose.
    text = re.sub(r"\\(?:boxed|textbf|text)\s*\{", "", text)
    text = text.replace(r"\[", "").replace(r"\]", "")
    return text


@dataclass(frozen=True)
class AnswerCandidate:
    start: int
    end: int
    tail: str
    rank: int


_DECLARATION = re.compile(
    r"\b(?:(?:the|my)\s+)?(?:(?:best|correct|final)\s+)?answer\s*(?:is\s*:?|:)\s*",
    re.IGNORECASE,
)
_UNSAFE_PREFIX = re.compile(
    r"\b(?:if|whether|maybe|perhaps|possibly|suppose|assuming|assume|hypothetical|"
    r"provisional|tentative|possible|possibility|chance|quote|format|template|"
    r"user|professor|cue|hint|prompt|example|dataset|benchmark|source|version|"
    r"says|said|claims|claimed|suggests|suggested|according|states|stated|told)\b"
    r"|\b(?:do not|don't|should not|shouldn't|would not|wouldn't)\s+(?:say|output|write)\b",
    re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"\b(?:correction|made a mistake|was mistaken|I change my answer|"
    r"on reconsideration|after reconsideration|instead I choose)\b",
    re.IGNORECASE,
)


def answer_candidates(response: str, break_words: Sequence[str] = ()) -> tuple[str, list[AnswerCandidate]]:
    text = visible_answer_text(response)
    if text is None:
        return "", []
    matches = [(m.start(), m.end(), m.group()) for m in _DECLARATION.finditer(text)]
    for word in break_words:
        for m in re.finditer(re.escape(word), text, re.IGNORECASE):
            if not any(a <= m.start() < b for a, b, _ in matches):
                matches.append((m.start(), m.end(), m.group()))
    matches.sort()
    # Treat "Final answer: The best answer is (A)" as one declaration,
    # carrying the outer final status to the nested commitment.
    collapsed = []
    for start, end, phrase in matches:
        if collapsed and not text[collapsed[-1][1]:start].strip():
            outer = collapsed.pop()
            start, phrase = outer[0], outer[2] + phrase
        collapsed.append((start, end, phrase))
    result = []
    for start, end, phrase in collapsed:
        boundary = max([m.end() for m in re.finditer(r"[.!?;\n]", text[:start])] or [0])
        prefix = text[boundary:start]
        if _UNSAFE_PREFIX.search(prefix):
            continue
        rank = 2 if re.search(r"\bfinal\b", phrase, re.I) else 1 if re.search(r"\b(?:best|correct)\b", phrase, re.I) else 0
        result.append(AnswerCandidate(start, end, text[end:], rank))
    return text, result


def _select(text: str, values: list[tuple[AnswerCandidate, str]]) -> str | None:
    if not values:
        return None
    rank = max(c.rank for c, _ in values)
    top = [(c, value) for c, value in values if c.rank == rank]
    if len({value for _, value in top}) == 1:
        return top[-1][1]
    # Unlike 'last answer wins', changing an explicit commitment requires an
    # explicit correction. Otherwise contradictory final declarations abstain.
    previous, last = top[-2][0], top[-1][0]
    if _CORRECTION.search(text[previous.end:last.start]):
        return top[-1][1]
    return None


def parse_indicator_answer(response: str, indicators: Sequence[str], break_words: Sequence[str] = ()) -> str | None:
    """Return the declared indicator, or None; never match at a later line."""
    text, candidates = answer_candidates(response, break_words)
    if not indicators:
        return None
    alternatives = "|".join(re.escape(s) for s in sorted(indicators, key=len, reverse=True))
    if not candidates:
        direct = re.fullmatch(rf"\s*\(?({alternatives})\)?[.]?\s*", text, re.I)
        if direct:
            return next(s for s in indicators if s.casefold() == direct.group(1).casefold())
    pattern = re.compile(rf"^[\s($]*(?:(?:option|statement)\s*)?\(*({alternatives})(?=$|[\s).,;}}$\\\]])", re.I)
    values = []
    for candidate in candidates:
        match = pattern.match(candidate.tail)
        if match is None:
            continue
        following = candidate.tail[match.end():]
        if re.match(r"^[\s)}$.,:;\]]*\?", following):
            continue
        if re.match(rf"^[\s)}}$.,\]]*(?:or|and|/)\s*\(*({alternatives})(?=$|[\s).,])", following, re.I):
            # Do not salvage the earlier candidate if the final choice is ambiguous.
            return None
        if re.match(r"^[\s)}$.,]*but\b[^.\n]*(?:wrong|incorrect|reject|not my answer)", following, re.I):
            continue
        label = next(s for s in indicators if s.casefold() == match.group(1).casefold())
        values.append((candidate, label))
    if candidates:
        top_rank = max(c.rank for c in candidates)
        last_top = [c for c in candidates if c.rank == top_rank][-1]
        attempted_label = re.match(r"^[\s($]*(?:(?:option|statement)\s*)?\(*[A-Z](?=$|[\s).,;}$\\\]])", last_top.tail)
        if not any(c == last_top for c, _ in values) and (attempted_label or not last_top.tail.strip() or last_top.rank == 2):
            return None  # An invalid final declaration must not revive an old answer.
    return _select(text, values)


def answer_text_tail(response: str, break_words: Sequence[str] = ()) -> str | None:
    """Safe tail for literal/fuzzy option extractors, without cue fallback."""
    text, candidates = answer_candidates(response, break_words)
    return _select(text, [(c, c.tail.strip()) for c in candidates])
