import pytest

from ctm_data.adapters.mcq_bias.aggregation import (
    MCQMetricAggregationError,
    aggregate_mcq_bias_sample_values,
)


def _parsed(*, correct: int, matches_bias: int | None) -> dict:
    return {"correct": correct, "matches_bias": matches_bias, "answer_parsed": 1}


def _unparsed() -> dict:
    return {"correct": None, "matches_bias": None, "answer_parsed": 0}


def test_clean_counts_parse_failures_in_accuracy_denominator():
    aggregate = aggregate_mcq_bias_sample_values(
        [
            _parsed(correct=1, matches_bias=None),
            _parsed(correct=0, matches_bias=None),
            _parsed(correct=0, matches_bias=None),
            _unparsed(),
        ],
        condition="clean",
        accuracy_parse_failures="incorrect",
    )

    assert (aggregate.numerator, aggregate.denominator, aggregate.value) == (1, 4, 0.25)
    assert aggregate.parsed_count == 3
    assert aggregate.unparsed_count == 1
    assert aggregate.parse_rate == 0.75


def test_wrong_suggestion_excludes_parse_failures_from_following_denominator():
    aggregate = aggregate_mcq_bias_sample_values(
        [
            _parsed(correct=0, matches_bias=1),
            _parsed(correct=1, matches_bias=0),
            _parsed(correct=0, matches_bias=0),
            _unparsed(),
        ],
        condition="wrong_suggestion",
    )

    assert (aggregate.numerator, aggregate.denominator, aggregate.value) == (1, 3, 1 / 3)
    assert aggregate.unparsed_count == 1
    assert aggregate.parse_rate == 0.75


def test_all_unparsed_clean_is_zero_but_biased_metric_is_unavailable():
    clean = aggregate_mcq_bias_sample_values(
        [_unparsed(), _unparsed()],
        condition="clean",
        accuracy_parse_failures="incorrect",
    )
    biased = aggregate_mcq_bias_sample_values([_unparsed(), _unparsed()], condition="wrong_suggestion")

    assert (clean.numerator, clean.denominator, clean.value) == (0, 2, 0.0)
    assert (biased.numerator, biased.denominator, biased.value) == (0, 0, None)


@pytest.mark.parametrize(
    "value",
    [
        {"correct": 1, "matches_bias": None},
        {"correct": 1, "matches_bias": None, "answer_parsed": 0},
        {"correct": None, "matches_bias": None, "answer_parsed": 1},
        {"correct": 0.5, "matches_bias": None, "answer_parsed": 1},
    ],
)
def test_inconsistent_or_nonbinary_values_fail_closed(value):
    with pytest.raises(MCQMetricAggregationError):
        aggregate_mcq_bias_sample_values([value], condition="clean")
