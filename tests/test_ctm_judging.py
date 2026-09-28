"""Tests for the policy-neutral bounded judge-attempt helper."""

from __future__ import annotations

import asyncio

import pytest

from ctm.training import judging
from ctm.training.judging import JudgeAttemptResult, run_judge_attempts


def test_judge_attempts_separate_provider_and_parse_failures() -> None:
    attempts: list[object] = [RuntimeError("provider down"), "not-an-int", "7"]

    async def complete() -> str:
        value = attempts.pop(0)
        if isinstance(value, Exception):
            raise value
        assert isinstance(value, str)
        return value

    result = asyncio.run(run_judge_attempts(complete, int, retries=2, retry_delay=0))

    assert result == JudgeAttemptResult(
        verdict=7,
        attempt_count=3,
        parse_failure_count=1,
        provider_failure_count=1,
        last_error=None,
    )


def test_judge_attempts_abstain_after_exhaustion() -> None:
    async def complete() -> str:
        return "malformed"

    def parse(_raw: str) -> bool:
        raise ValueError("bad verdict")

    result = asyncio.run(run_judge_attempts(complete, parse, retries=1, retry_delay=0))

    assert result.verdict is None
    assert result.attempt_count == 2
    assert result.parse_failure_count == 2
    assert result.provider_failure_count == 0
    assert result.last_error == "ValueError: bad verdict"


def test_judge_attempts_use_exponential_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    attempts = iter(["bad", "bad", "ok"])

    async def complete() -> str:
        return next(attempts)

    async def sleep(delay: float) -> None:
        delays.append(delay)

    def parse(raw: str) -> bool:
        if raw != "ok":
            raise ValueError("bad")
        return True

    monkeypatch.setattr(judging.asyncio, "sleep", sleep)
    result = asyncio.run(
        run_judge_attempts(
            complete,
            parse,
            retries=2,
            retry_delay=0.25,
        )
    )

    assert result.verdict is True
    assert delays == [0.25, 0.5]


@pytest.mark.parametrize(
    ("retries", "retry_delay", "match"),
    [
        (True, 0, "non-negative integer"),
        (-1, 0, "non-negative integer"),
        (0, float("inf"), "finite non-negative"),
        (0, -0.1, "finite non-negative"),
    ],
)
def test_judge_attempts_validate_retry_budget(retries: object, retry_delay: object, match: str) -> None:
    async def complete() -> str:
        return "ok"

    with pytest.raises(ValueError, match=match):
        asyncio.run(run_judge_attempts(complete, lambda raw: raw, retries=retries, retry_delay=retry_delay))  # type: ignore[arg-type]
