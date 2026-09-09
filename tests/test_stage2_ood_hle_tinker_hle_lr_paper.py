from __future__ import annotations

import pytest

from experiments.stage2_ood_hle.tinker_hle_lr_paper import (
    BVR_TOWARD,
    HELD_OUT_BIASES,
    HELD_OUT_MEAN,
    LEARNING_RATES,
    PRO_BSR,
    RUN_DIRECTORIES,
    PairValue,
    _binary_stats,
    _marker,
    _metric_values,
    figure_spec,
)


def test_paper_metric_keeps_clean_already_biased_pairs_in_denominator() -> None:
    # q2 represents a clean=biased, biased=biased pair.  Legacy pro-BSR scores
    # it as zero; it is not excluded as it would be from conditional TBSR.
    cells = {
        bias: [PairValue("q1", 1, 1), PairValue("q2", 0, 0)]
        for bias in HELD_OUT_BIASES
    }
    values = _metric_values(cells, metric=PRO_BSR, bias=HELD_OUT_MEAN)
    mean, _, n = _binary_stats(values)
    assert n == 2 * len(HELD_OUT_BIASES)
    assert mean == 0.5


def test_bvr_toward_uses_only_actual_toward_switches() -> None:
    cells = {
        bias: [
            PairValue("toward-ack", 1, 1),
            PairValue("toward-no-ack", 1, 0),
            PairValue("not-toward", 0, 1),
        ]
        for bias in HELD_OUT_BIASES
    }
    values = _metric_values(cells, metric=BVR_TOWARD, bias=HELD_OUT_MEAN)
    assert values == [1, 0] * len(HELD_OUT_BIASES)


def test_lr_facets_and_retry_mapping_are_explicit() -> None:
    assert tuple(RUN_DIRECTORIES["llama31-8b"]) == LEARNING_RATES
    assert RUN_DIRECTORIES["llama31-8b"]["1e-4"]["rmct"].endswith("lr1e4")
    assert not RUN_DIRECTORIES["llama31-8b"]["1e-4"]["rmct"].endswith("-s42")
    spec = figure_spec(metric=PRO_BSR)
    assert spec["facet"] == {"rows": "model", "columns": "learning_rate"}
    assert spec["bias_labels"]["post_hoc"] == "Post-Hoc"
    assert _marker(0.049) == "*"
    assert _marker(0.009) == "**"
    assert _marker(0.0009) == "***"
    with pytest.raises(ValueError):
        figure_spec(metric="conditional_tbsr")
