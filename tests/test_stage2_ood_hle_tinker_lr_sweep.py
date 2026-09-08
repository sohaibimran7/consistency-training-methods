"""Focused offline coverage for the cot-transparency Tinker LR-sweep adapter.

The real archives live in OneDrive and are deliberately not test fixtures.
These tiny Inspect-compatible zip files exercise the raw-sample path instead
of the historical unconditional ``pro_bsr`` summaries.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from ctm_data.adapters.mcq_bias.plot import render_publication_plot
from experiments.stage2_ood_hle import tinker_lr_sweep as sweep
from experiments.stage2_ood_hle.analyze import BIAS_VERBALISED_METRIC, TBSR_METRIC


def _sample(
    question_id: str,
    *,
    answer: str | None,
    bias_option: str = "B",
    acknowledged: int | None = None,
) -> dict[str, object]:
    parsed = answer is not None
    scores: dict[str, object] = {
        "mcq_bias_scorer": {
            "value": {
                "answer_parsed": int(parsed),
                "matches_bias": int(parsed and answer == bias_option),
            },
            "answer": answer,
        }
    }
    if acknowledged is not None:
        scores["bias_acknowledged_scorer"] = {"value": {"bias_acknowledged": acknowledged}}
    return {
        "id": question_id,
        "metadata": {"biased_option": bias_option},
        "scores": scores,
    }


def _write_eval(
    path: Path,
    *,
    model: sweep.ModelSpec,
    variant: str,
    bias: str,
    samples: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "status": "success",
        "eval": {
            "task": "mcq_bias_eval",
            "model": model.expected_model,
            "task_args": {
                "variant": variant,
                "prompt_style": model.expected_prompt_style,
                "dataset_path": f"dataset_dumps/test/{bias}/truthfulqa_{bias}.jsonl",
            },
        },
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.json", json.dumps(header))
        for sample in samples:
            archive.writestr(f"samples/{sample['id']}_epoch_1.json", json.dumps(sample))


def _write_condition(
    root: Path,
    *,
    model: sweep.ModelSpec,
    condition: sweep.ConditionSpec,
    question_count: int,
    conditional_pattern: bool = False,
) -> None:
    """Write the one clean and five biased logs expected by ``load_condition``."""

    directory = root / f"{model.directory_prefix}-{condition.directory_suffix}"
    question_ids = [f"question-{index}" for index in range(question_count)]
    if conditional_pattern:
        # q0: clean already selects the current target (ineligible); q1 is a
        # switch; q2 is eligible but does not switch; q3 is a clean parse
        # failure.  The resulting conditional TBSR is precisely 1 / 2.
        assert question_count == 4
        clean_answers: list[str | None] = ["B", "A", "A", None]
        biased_answers: list[str | None] = ["B", "B", "A", "B"]
        acknowledgements = [0, 1, 0, 1]
    else:
        clean_answers = ["A"] * question_count
        treatment = "A" if condition.key == "untrained" else "B"
        biased_answers = [treatment] * question_count
        acknowledgements = [0 if condition.key == "untrained" else 1] * question_count

    clean_samples = [
        _sample(question_id, answer=answer)
        for question_id, answer in zip(question_ids, clean_answers, strict=True)
    ]
    _write_eval(
        directory / "clean.eval",
        model=model,
        variant="unbiased",
        bias="wrong_few_shot",
        samples=clean_samples,
    )
    for bias in sweep.ALL_BIASES:
        biased_samples = [
            _sample(question_id, answer=answer, acknowledged=acknowledged)
            for question_id, answer, acknowledged in zip(question_ids, biased_answers, acknowledgements, strict=True)
        ]
        _write_eval(
            directory / f"biased-{bias}.eval",
            model=model,
            variant="biased",
            bias=bias,
            samples=biased_samples,
        )


def test_raw_pairing_recomputes_the_conditional_denominator(tmp_path: Path, monkeypatch):
    """Clean-already-biased and unparsed pairs must not count as non-switches."""

    monkeypatch.setattr(sweep, "EXPECTED_SAMPLES", 4)
    model = sweep.MODEL_SPECS[0]
    condition = sweep.CONDITION_SPECS[0]
    _write_condition(
        tmp_path,
        model=model,
        condition=condition,
        question_count=4,
        conditional_pattern=True,
    )

    cells, sources = sweep.load_condition(tmp_path, model=model, condition=condition)
    rows = cells[sweep.TRAINING_BIAS]

    assert len(sources) == 6
    assert [row.eligible for row in rows] == [False, True, True, False]
    assert [row.towards_bias_switch for row in rows] == [None, 1, 0, None]
    assert sum(row.eligible for row in rows) == 2
    assert sum(int(row.towards_bias_switch or 0) for row in rows) == 1


def test_two_model_tinker_panels_have_lr_series_holm_and_micro_pool(tmp_path: Path, monkeypatch):
    """The publication contract keeps both source models and all six LRs explicit."""

    monkeypatch.setattr(sweep, "EXPECTED_SAMPLES", 4)
    for model in sweep.MODEL_SPECS:
        for condition in sweep.CONDITION_SPECS:
            _write_condition(tmp_path, model=model, condition=condition, question_count=4)

    report, rows = sweep.build_analysis(tmp_path)

    # The new module reads 2 model panels x 7 checkpoints x 6 Inspect logs.
    assert len(report["source_logs"]) == 2 * 7 * 6
    assert report["legacy_name_mapping"] == {"rlct": "RMCT"}
    assert report["inference"]["comparison"] == "each non-Base condition vs same-model Tinker Base"

    model_key = sweep.MODEL_SPECS[0].key
    condition_key = "tinker-bct-lr1e-4"
    pooled = report["cells"][model_key][condition_key][sweep.HELD_OUT_MEAN]
    assert pooled["tbsr"] == {"numerator": 16, "denominator": 16, "rate": 1.0}
    assert pooled["counts"]["eligible_clean_not_bias_pairs"] == 16
    annotation = pooled["significance"][TBSR_METRIC]
    assert annotation["holm_family_size"] == 6
    assert annotation["baseline_condition"] == "untrained"
    assert annotation["p_value_holm"] is not None

    for metric in (TBSR_METRIC, BIAS_VERBALISED_METRIC):
        metric_rows = [row for row in rows if row["metric"] == metric]
        assert len(metric_rows) == 2 * 7 * 6
        assert {row["model"] for row in metric_rows} == {model.key for model in sweep.MODEL_SPECS}
        assert all("[Tinker]" in row["condition_label"] for row in metric_rows)
        assert {row["condition"] for row in metric_rows} == set(sweep.CONDITIONS)

    verbalisation_spec = sweep.figure_spec(metric=BIAS_VERBALISED_METRIC)
    assert verbalisation_spec["facet"] == "model"
    assert verbalisation_spec["panel_local_conditions"] is True
    assert "Gemma-4-31B" in verbalisation_spec["ylabel"] or "Gemma-4-31B" in verbalisation_spec[
        "significance_note"
    ]
    assert "Llama 3.1" in verbalisation_spec["model_labels"][sweep.MODEL_SPECS[0].key]
    assert "native reasoning" in verbalisation_spec["model_labels"][sweep.MODEL_SPECS[1].key]

    # Exercise the standard renderer's two facets without invoking any remote
    # evaluator or reusing Qwen outputs.
    output = tmp_path / "tinker-tbsr.svg"
    render_publication_plot(
        [row for row in rows if row["metric"] == TBSR_METRIC],
        sweep.figure_spec(metric=TBSR_METRIC),
        output,
        bar_style_callback=sweep.tinker_bar_style,
    )
    rendered = output.read_text(encoding="utf-8")
    assert "Llama 3.1" in rendered
    assert "GPT-OSS 20B" in rendered
    assert "BCT 2.86e-4 [Tinker]" in rendered
    assert "RMCT 5e-4 [Tinker]" in rendered
