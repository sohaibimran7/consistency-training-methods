from types import SimpleNamespace

import pytest

from ctm_data.adapters.eval_awareness.analysis import aggregate_logs as aggregate_evalaware
from ctm_data.adapters.wildjailbreak.analysis import aggregate_logs as aggregate_wildjailbreak


def _log(*, benchmark, task, created, model, prompt_type, valence, factors=None, rows, sample_metadata=None):
    metadata_rows = sample_metadata or [{} for _ in rows]
    samples = [
        SimpleNamespace(scores={"scores": SimpleNamespace(value=row)}, metadata=metadata)
        for row, metadata in zip(rows, metadata_rows, strict=True)
    ]
    return SimpleNamespace(
        status="success",
        location=f"{task}-{created}.eval",
        eval=SimpleNamespace(
            task=task,
            created=created,
            model=model,
            metadata={
                "benchmark": benchmark,
                "prompt_type": prompt_type,
                "valence": valence,
                "factors": factors,
            },
        ),
        samples=samples,
    )


def _evalaware(*, valence, created="1", values=(0.0, 1.0)):
    return _log(
        benchmark="evalawarebench",
        task="evalaware",
        created=created,
        model="hf/model",
        prompt_type="factor",
        valence=valence,
        factors=["F6"],
        rows=[{"net_refusal_switch": value} for value in values],
    )


def _wild(*, valence, created="1", values=(0.0, 1.0)):
    return _log(
        benchmark="wildjailbreak",
        task="wildjailbreak",
        created=created,
        model="hf/model",
        prompt_type="adversarial",
        valence=valence,
        rows=[{"net_refusal_switch": value} for value in values],
    )


def test_evalaware_analysis_keeps_factor_set_and_valence_explicit():
    rows = aggregate_evalaware(
        {
            "base": [[_evalaware(valence="safety"), _evalaware(valence="capability")]],
            "trained": [[_evalaware(valence="safety"), _evalaware(valence="capability")]],
        },
        condition_metadata={"trained": {"method": "rate_matching"}},
    )

    assert len(rows) == 4
    assert {row["factor_set"] for row in rows} == {"F6"}
    assert {row["valence"] for row in rows} == {"safety", "capability"}
    assert all(row["prompt_type"] == "factor" for row in rows)
    assert all(row["n_scored"] == 2 and row["n_total"] == 2 for row in rows)
    assert all(row["n_source_logs"] == 1 and len(row["source_logs"]) == 1 for row in rows)
    assert next(row for row in rows if row["condition"] == "trained")["method"] == "rate_matching"


def test_evalaware_analysis_pools_independent_replicates():
    rows = aggregate_evalaware(
        {
            "trained": [
                [_evalaware(valence="safety", values=(0.0, 0.0))],
                [_evalaware(valence="safety", values=(1.0, 1.0))],
            ]
        },
        valences=["safety"],
    )

    assert rows[0]["mean"] == 0.5
    assert rows[0]["n_replicates"] == 2
    assert rows[0]["n_scored"] == 4
    assert rows[0]["stderr"] == pytest.approx((1 / 3 / 4) ** 0.5)


def test_evalaware_analysis_uses_only_the_latest_successful_attempt_for_one_task():
    rows = aggregate_evalaware(
        {
            "trained": [
                [
                    _evalaware(valence="safety", created="1", values=(0.0, 0.0)),
                    _evalaware(valence="safety", created="2", values=(1.0, 1.0)),
                ]
            ]
        },
        valences=["safety"],
    )

    assert rows[0]["mean"] == 1.0
    assert rows[0]["n_scored"] == 2
    assert rows[0]["n_source_logs"] == 1
    assert rows[0]["source_logs"] == ["evalaware-2.eval"]


def test_evalaware_analysis_rejects_incomplete_condition_matrix_by_default():
    with pytest.raises(ValueError, match="different chart-cell matrix"):
        aggregate_evalaware(
            {
                "base": [[_evalaware(valence="safety"), _evalaware(valence="capability")]],
                "trained": [[_evalaware(valence="safety")]],
            }
        )


def test_wildjailbreak_analysis_keeps_harmful_and_benign_valence_explicit():
    rows = aggregate_wildjailbreak(
        {
            "base": [[_wild(valence="harmful"), _wild(valence="benign")]],
            "trained": [[_wild(valence="harmful"), _wild(valence="benign")]],
        }
    )

    assert len(rows) == 4
    assert {row["valence"] for row in rows} == {"harmful", "benign"}
    assert {row["valence_label"] for row in rows} == {"Harmful", "Benign"}
    assert all(row["metric"] == "net_refusal_switch" for row in rows)


def test_wildjailbreak_analysis_can_select_official_metric_without_faking_other_valence():
    harmful = _log(
        benchmark="wildjailbreak",
        task="wildjailbreak",
        created="1",
        model="hf/model",
        prompt_type="adversarial",
        valence="harmful",
        rows=[{"attack_success": 1.0}, {"attack_success": 0.0}],
    )

    rows = aggregate_wildjailbreak(
        {"trained": [[harmful]]},
        metric="attack_success",
        valences=["harmful"],
    )

    assert rows[0]["valence"] == "harmful"
    assert rows[0]["metric"] == "attack_success"
    assert rows[0]["mean"] == 0.5


def test_wildjailbreak_tactic_analysis_is_explicitly_multi_label_marginal():
    harmful = _log(
        benchmark="wildjailbreak",
        task="wildjailbreak",
        created="1",
        model="hf/model",
        prompt_type="adversarial",
        valence="harmful",
        rows=[
            {"net_refusal_switch": 1.0},
            {"net_refusal_switch": 0.0},
            {"net_refusal_switch": 1.0},
        ],
        sample_metadata=[
            {"tactics": ["roleplay", "encoding", "roleplay"]},
            {"tactics": ["roleplay"]},
            {"tactics": []},
        ],
    )

    rows = aggregate_wildjailbreak(
        {"trained": [[harmful]]},
        group_by="tactic",
        valences=["harmful"],
    )

    by_tactic = {row["tactic"]: row for row in rows}
    assert set(by_tactic) == {"encoding", "roleplay", "untagged"}
    assert by_tactic["roleplay"]["mean"] == 0.5
    assert by_tactic["roleplay"]["n_scored"] == 2
    assert by_tactic["encoding"]["mean"] == 1.0
    assert by_tactic["encoding"]["n_total"] == 1
    assert by_tactic["untagged"]["mean"] == 1.0
    assert all(row["tactic_grouping"] == "multi_label_marginal" for row in rows)


def test_wildjailbreak_tactic_analysis_can_explicitly_exclude_untagged_samples():
    harmful = _log(
        benchmark="wildjailbreak",
        task="wildjailbreak",
        created="1",
        model="hf/model",
        prompt_type="adversarial",
        valence="harmful",
        rows=[{"net_refusal_switch": 1.0}, {"net_refusal_switch": 0.0}],
        sample_metadata=[{"tactics": ["roleplay"]}, {"tactics": []}],
    )

    rows = aggregate_wildjailbreak(
        {"trained": [[harmful]]},
        group_by="tactic",
        valences=["harmful"],
        include_untagged=False,
    )

    assert [row["tactic"] for row in rows] == ["roleplay"]
    assert rows[0]["n_total"] == 1
