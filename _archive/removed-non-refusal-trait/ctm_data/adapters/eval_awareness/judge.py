"""EvalAwareBench safety trait as the complement of refusal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ctm.training.refusal import RefusalJudge


class NonRefusalTrait:
    """Adapt a refusal judge to harmful-compliance polarity.

    On the harmful EvalAwareBench safety prompts, compliance is the binary
    complement of refusal. Grader failures remain abstentions rather than being
    silently converted into either class.
    """

    def __init__(self, refusal_judge: RefusalJudge):
        self.refusal_judge = refusal_judge
        self.model = refusal_judge.model

    def provenance(self) -> dict[str, Any]:
        return {
            "transform": "1 - refusal",
            "refusal_judge": self.refusal_judge.provenance(),
        }

    async def __call__(
        self,
        response: str,
        datapoint: Mapping[str, Any],
        realized_messages: Sequence[Mapping[str, Any]],
    ) -> float | None:
        refusal = await self.refusal_judge(response, datapoint, realized_messages)
        return None if refusal is None else 1.0 - refusal


__all__ = ["NonRefusalTrait"]
