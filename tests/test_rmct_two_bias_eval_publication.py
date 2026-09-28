from types import SimpleNamespace

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.rmct_two_bias_eval.contract import ALL_BIASES, HELD_OUT_BIASES, SEEN_BIASES
from experiments.rmct_two_bias_eval.publication import (
    BIAS_GROUPS,
    CONDITIONS,
    DATASETS,
    POPULATION_DATASETS,
    chart_rows,
    publication_spec,
)


def _log(*, bias: str, dataset: str, values: list[float | None], created: str = "1"):
    samples = [
        SimpleNamespace(
            id=f"q-{index}",
            scores={
                "scores": SimpleNamespace(
                    value={
                        "bias_acknowledged": value,
                        "towards_bias_switch": value,
                    }
                )
            }
        )
        for index, value in enumerate(values)
    ]
    return SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            created=created,
            task_args={
                "bias_type": bias,
                "dataset": dataset,
                "prompt_style": "none",
                "seed": "42",
                "n_questions": len(values),
            },
        ),
        samples=samples,
    )


def _matrix(*, positive: bool):
    values = [1.0, 1.0] if positive else [1.0, 0.0]
    return [
        _log(bias=bias, dataset=dataset, values=values)
        for bias in ALL_BIASES
        for dataset in DATASETS
    ]


def test_chart_rows_keep_bias_groups_and_provenance_data_driven():
    rows = chart_rows(
        {
            "untrained": _matrix(positive=False),
            "rate-matching": _matrix(positive=True),
        },
        significance_permutations=100,
    )

    assert len(rows) == len(POPULATION_DATASETS) * len(CONDITIONS) * (len(ALL_BIASES) + len(BIAS_GROUPS))
    seen = next(
        row
        for row in rows
        if row["population"] == "held_in_datasets"
        and row["condition"] == "rate-matching"
        and row["bias_type"] == "seen_mean"
    )
    heldout = next(
        row
        for row in rows
        if row["population"] == "held_out_dataset"
        and row["condition"] == "untrained"
        and row["bias_type"] == "held_out_mean"
    )
    suggested = next(
        row
        for row in rows
        if row["population"] == "held_out_dataset"
        and row["condition"] == "rate-matching"
        and row["bias_type"] == "suggested_answer"
    )

    assert seen["component_biases"] == list(SEEN_BIASES)
    assert seen["bias_group"] == "seen"
    assert seen["mean"] == 1.0
    assert seen["n_scored"] == 8
    assert seen["n_total"] == 8
    assert seen["success_count"] == 8
    assert seen["population_datasets"] == ["logiqa", "hellaswag"]
    assert seen["ci_method"] == "wilson"
    assert heldout["component_biases"] == list(HELD_OUT_BIASES)
    assert heldout["bias_group"] == "held_out"
    assert heldout["mean"] == 0.5
    assert heldout["n_scored"] == 8
    assert heldout["population_datasets"] == ["hle-text-mc"]
    assert suggested["bias_status"] == "seen"
    assert suggested["training_biases"] == list(SEEN_BIASES)
    assert suggested["provenance_class"] == "sealed_r005_step176"


def test_standard_renderer_accepts_r005_chart_contract(tmp_path):
    rows = chart_rows(
        {
            "untrained": _matrix(positive=False),
            "rate-matching": _matrix(positive=True),
        },
        significance_permutations=100,
    )
    spec = publication_spec()
    output = tmp_path / "standard.svg"

    render_publication_plot(rows, spec, output)

    svg = output.read_text()
    assert "Bias verbalised" in svg
    assert "Archived base" in svg
    assert "descriptive" in svg
    assert "Held-in datasets" in svg
    assert "Held-out dataset" in svg
    assert "Seen avg." in svg
    assert "Held-out avg." in svg


def test_switch_rate_uses_the_same_dataset_split_and_standard_renderer(tmp_path):
    rows = chart_rows(
        {
            "untrained": _matrix(positive=False),
            "rate-matching": _matrix(positive=True),
        },
        metric="towards_bias_switch",
        significance_permutations=100,
    )
    spec = publication_spec(metric="towards_bias_switch")
    output = tmp_path / "switch.svg"

    render_publication_plot(rows, spec, output)

    assert len(rows) == len(POPULATION_DATASETS) * len(CONDITIONS) * (len(ALL_BIASES) + len(BIAS_GROUPS))
    assert {row["population"] for row in rows} == set(POPULATION_DATASETS)
    assert {row["metric"] for row in rows} == {"towards_bias_switch"}
    assert "* adjusted p<0.05; ** p<0.01; *** p<0.001" in spec["significance_note"]
    treatment = [row for row in rows if row["condition"] == "rate-matching"]
    assert all(row["significance_method"] == "paired_question_cluster_label_swap_randomization" for row in treatment)
    assert all(row["holm_family_size"] == 18 for row in treatment)
    assert all(row["p_value"] == row["p_value_holm"] for row in treatment)
    svg = output.read_text()
    assert "Towards-bias switch rate" in svg
    assert "Held-in datasets" in svg
    assert "Held-out dataset" in svg
    assert "clean-not-bias" in svg
