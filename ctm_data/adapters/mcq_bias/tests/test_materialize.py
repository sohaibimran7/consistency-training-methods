import pytest

from ctm_data.adapters.mcq_bias.materialize import interleave_rows, retry_wrong_argument_materialization


def test_interleave_rows_keeps_global_prefix_balanced():
    assert interleave_rows([[{"id": "a1"}, {"id": "a2"}], [{"id": "b1"}, {"id": "b2"}, {"id": "b3"}]]) == [
        {"id": "a1"},
        {"id": "b1"},
        {"id": "a2"},
        {"id": "b2"},
        {"id": "b3"},
    ]


def test_wrong_argument_materialization_retries_only_remaining_misses():
    calls = 0

    def materialize():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ValueError("Only 6/8 matched questions for 'wrong_argument' on 'logiqa' (floor: 8).")
        return "complete"

    assert retry_wrong_argument_materialization(
        materialize,
        enabled=True,
        max_rounds=5,
        dataset="logiqa",
    ) == "complete"
    assert calls == 3


def test_wrong_argument_materialization_does_not_hide_other_errors():
    with pytest.raises(ValueError, match="different failure"):
        retry_wrong_argument_materialization(
            lambda: (_ for _ in ()).throw(ValueError("different failure")),
            enabled=True,
            max_rounds=5,
            dataset="logiqa",
        )
