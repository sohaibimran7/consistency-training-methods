"""Compatibility fixes for pinned ``mcq_bias`` score values.

Inspect treats floating-point NaN as the canonical unscored sentinel.  The
pinned scorers use ``None`` inside dictionary-valued scores for values that do
not apply to a sample.  Some Inspect versions count those nested ``None``
values as scored (and can aggregate them as zero), silently changing a
conditional rate into a whole-dataset rate or emitting conversion warnings.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

_SWITCH_PATCH_MARKER = "__ctm_conditional_nan_compat__"
_OPTIONS_PATCH_MARKER = "__ctm_options_considered_nan_compat__"
_MAPPING_SCORER_PATCH_MARKER = "__ctm_mapping_score_nan_compat__"


def _none_to_nan(values: dict[str, float | None]) -> dict[str, float]:
    return {name: math.nan if value is None else float(value) for name, value in values.items()}


def _patch_mapping_scorer(upstream: Any, name: str) -> None:
    """Patch one upstream scorer factory without changing its metric registry."""

    original = getattr(upstream, name, None)
    if not callable(original) or getattr(original, _MAPPING_SCORER_PATCH_MARKER, False):
        return

    @functools.wraps(original)
    def scorer_factory(*args: Any, **kwargs: Any) -> Any:
        scorer = original(*args, **kwargs)

        @functools.wraps(scorer)
        async def score(*score_args: Any, **score_kwargs: Any) -> Any:
            result = await scorer(*score_args, **score_kwargs)
            values = getattr(result, "value", None)
            if not isinstance(values, dict) or not any(value is None for value in values.values()):
                return result
            normalized = _none_to_nan(values)
            # Inspect's Score is a Pydantic model. model_copy preserves the
            # answer, explanation, metadata, history, and scorer registry
            # attributes copied by functools.wraps above.
            return result.model_copy(update={"value": normalized})

        return score

    setattr(scorer_factory, _MAPPING_SCORER_PATCH_MARKER, True)
    setattr(upstream, name, scorer_factory)


def install_conditional_nan_compat() -> None:
    """Normalize pinned optional scorer values to Inspect's NaN sentinel.

    The public name is retained because callers already use it for the switch
    scorer. It also normalizes intentionally absent values from the upstream
    core mapping scorers (clean ``matches_bias``, unparsed MCQ values,
    options-considered, and acknowledgement grading).
    """

    import mcq_bias.scorers as upstream

    original: Callable[..., dict[str, float | None]] = upstream.switch_values
    if not getattr(original, _SWITCH_PATCH_MARKER, False):

        @functools.wraps(original)
        def switch_values(*args: Any, **kwargs: Any) -> dict[str, float]:
            return _none_to_nan(original(*args, **kwargs))

        setattr(switch_values, _SWITCH_PATCH_MARKER, True)
        upstream.switch_values = switch_values

    original_options = getattr(upstream, "_count_options_considered", None)
    if callable(original_options) and not getattr(original_options, _OPTIONS_PATCH_MARKER, False):

        @functools.wraps(original_options)
        def count_options_considered(*args: Any, **kwargs: Any) -> float:
            value = original_options(*args, **kwargs)
            return math.nan if value is None else float(value)

        setattr(count_options_considered, _OPTIONS_PATCH_MARKER, True)
        upstream._count_options_considered = count_options_considered

    # These factories retain their upstream metric declarations and registry
    # identity. Only their dictionary-valued Score payload is normalized.
    # ``switch_scorer`` is included deliberately even though ``switch_values``
    # above is also patched: some upstream releases capture that helper while
    # constructing the scorer. Normalizing the returned Score mapping is the
    # final compatibility boundary and cannot be bypassed by such a capture.
    for name in (
        "mcq_bias_scorer",
        "options_considered_scorer",
        "bias_acknowledged_scorer",
        "switch_scorer",
    ):
        _patch_mapping_scorer(upstream, name)


__all__ = ["install_conditional_nan_compat"]
