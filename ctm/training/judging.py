"""Bounded async judge attempts with explicit failure diagnostics.

Judge prompts and parsers remain setting-specific.  This module owns only the
provider/parse retry boundary so callers use one retry budget and can report
abstentions without treating malformed verdicts as negative judgments.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

VerdictT = TypeVar("VerdictT")


@dataclass(frozen=True, slots=True)
class JudgeAttemptResult(Generic[VerdictT]):
    """Outcome and diagnostics for one bounded judge operation."""

    verdict: VerdictT | None
    attempt_count: int
    parse_failure_count: int
    provider_failure_count: int
    last_error: str | None


async def run_judge_attempts(
    completion: Callable[[], Awaitable[str]],
    parser: Callable[[str], VerdictT],
    *,
    retries: int,
    retry_delay: float,
) -> JudgeAttemptResult[VerdictT]:
    """Call and parse a judge with bounded exponential-backoff retries.

    Completion exceptions, including a non-text completion, count as provider
    failures.  Parser exceptions count separately as parse failures.  Exhausted
    attempts return ``verdict=None`` so the caller can apply its own abstention
    or run-fatal policy.
    """

    if not callable(completion):
        raise TypeError("completion must be callable")
    if not callable(parser):
        raise TypeError("parser must be callable")
    if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
        raise ValueError("retries must be a non-negative integer")
    if (
        not isinstance(retry_delay, (int, float))
        or isinstance(retry_delay, bool)
        or not math.isfinite(float(retry_delay))
        or retry_delay < 0
    ):
        raise ValueError("retry_delay must be a finite non-negative number")

    parse_failures = 0
    provider_failures = 0
    last_error: str | None = None
    delay = float(retry_delay)
    for attempt in range(retries + 1):
        try:
            raw = await completion()
            if not isinstance(raw, str):
                raise TypeError(f"judge completion returned {type(raw).__name__}, expected str")
        except Exception as exc:  # noqa: BLE001 - explicit provider retry boundary
            provider_failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            try:
                verdict = parser(raw)
            except Exception as exc:  # noqa: BLE001 - parser failures are retryable judge failures
                parse_failures += 1
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                return JudgeAttemptResult(
                    verdict=verdict,
                    attempt_count=attempt + 1,
                    parse_failure_count=parse_failures,
                    provider_failure_count=provider_failures,
                    last_error=None,
                )
        if attempt < retries and delay:
            await asyncio.sleep(delay * (2**attempt))

    return JudgeAttemptResult(
        verdict=None,
        attempt_count=retries + 1,
        parse_failure_count=parse_failures,
        provider_failure_count=provider_failures,
        last_error=last_error,
    )


__all__ = ["JudgeAttemptResult", "run_judge_attempts"]
