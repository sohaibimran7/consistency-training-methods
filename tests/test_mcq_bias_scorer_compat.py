from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat


def test_ineligible_switch_values_use_inspect_unscored_nan(monkeypatch):
    import mcq_bias.scorers as scorers

    original = scorers.switch_values
    monkeypatch.setattr(scorers, "switch_values", original)
    install_conditional_nan_compat()

    values = scorers.switch_values("B", "B", "B")

    assert math.isnan(values["towards_bias_switch"])
    assert values["away_from_bias_switch"] == 0.0
    assert values["unbiased_matches_bias"] == 1.0


def test_unparsed_pair_is_unscored_for_every_switch_metric(monkeypatch):
    import mcq_bias.scorers as scorers

    original = scorers.switch_values
    monkeypatch.setattr(scorers, "switch_values", original)
    install_conditional_nan_compat()

    values = scorers.switch_values(None, "A", "B")

    assert values
    assert all(math.isnan(value) for value in values.values())


def test_missing_options_considered_uses_inspect_unscored_nan(monkeypatch):
    import mcq_bias.scorers as scorers

    original = scorers._count_options_considered
    monkeypatch.setattr(scorers, "_count_options_considered", original)
    install_conditional_nan_compat()

    assert math.isnan(scorers._count_options_considered("no labelled options here", ["A", "B"]))


def test_core_scorers_emit_nan_for_clean_or_missing_numeric_values():
    import mcq_bias.scorers as scorers

    install_conditional_nan_compat()

    clean_state = SimpleNamespace(
        output=SimpleNamespace(completion="Final answer: A"),
        metadata={"prompt_family": "chua", "biased_option": ""},
    )
    clean_result = asyncio.run(scorers.mcq_bias_scorer()(clean_state, SimpleNamespace(text="A")))
    assert clean_result.value["correct"] == 1.0
    assert math.isnan(clean_result.value["matches_bias"])

    options_state = SimpleNamespace(
        output=SimpleNamespace(completion="No option labels occur here.", message=None),
        input_text="This prompt has no multiple-choice options.",
    )
    options_result = asyncio.run(scorers.options_considered_scorer()(options_state, None))
    assert math.isnan(options_result.value["options_considered"])

    acknowledgement_state = SimpleNamespace(
        output=SimpleNamespace(completion="", message=None),
        metadata={"variant": "unbiased", "bias_type": "", "biasing_text": ""},
    )
    acknowledgement_result = asyncio.run(scorers.bias_acknowledged_scorer()(acknowledgement_state, None))
    assert math.isnan(acknowledgement_result.value["bias_acknowledged"])


def test_real_switch_scorer_factory_normalizes_every_nested_none(monkeypatch):
    import mcq_bias.scorers as scorers
    import mcq_bias.unbiased_log as unbiased_log

    async def wait_for_unbiased_log(*_args, **_kwargs):
        return "/tmp/clean.eval"

    monkeypatch.setattr(unbiased_log, "wait_for_unbiased_log", wait_for_unbiased_log)
    monkeypatch.setattr(unbiased_log, "unbiased_answers", lambda _path: {"sample-1": "B"})
    monkeypatch.setattr(scorers, "parse_answer", lambda *_args, **_kwargs: "B")
    install_conditional_nan_compat()

    state = SimpleNamespace(
        model="vllm/test-model",
        sample_id="sample-1",
        output=SimpleNamespace(completion="ignored"),
        metadata={
            "source_dataset": "logiqa",
            "prompt_style": "encourage_cot",
            "prompt_family": "chua",
            "biased_option": "B",
        },
    )
    result = asyncio.run(scorers.switch_scorer("/tmp/clean.eval")(state, None))

    assert result.value["unbiased_matches_bias"] == 1.0
    assert result.value["away_from_bias_switch"] == 0.0
    assert all(value is not None for value in result.value.values())
    assert math.isnan(result.value["towards_bias_switch"])


def test_core_mapping_scorers_normalize_missing_values_and_keep_registry(monkeypatch):
    from inspect_ai.scorer import Score
    from inspect_ai.scorer._scorer import as_scorer_spec
    import mcq_bias.scorers as scorers

    async def raw_score(*_args, **_kwargs):
        return Score(value={"present": 1.0, "missing": None})

    def raw_factory():
        return raw_score

    monkeypatch.setattr(scorers, "mcq_bias_scorer", raw_factory)
    install_conditional_nan_compat()

    scorer = scorers.mcq_bias_scorer()
    result = asyncio.run(scorer(None, None))

    assert result.value["present"] == 1.0
    assert math.isnan(result.value["missing"])

    # The real factory's Inspect registry contract is preserved by the same
    # wrapper used above, so task construction still records its metrics.
    monkeypatch.undo()
    install_conditional_nan_compat()
    spec = as_scorer_spec(scorers.mcq_bias_scorer())
    assert set(spec.metrics) == {"correct", "matches_bias", "answer_parsed"}


def test_install_is_idempotent(monkeypatch):
    import mcq_bias.scorers as scorers

    original = scorers.switch_values
    monkeypatch.setattr(scorers, "switch_values", original)
    install_conditional_nan_compat()
    first = scorers.switch_values
    first_options = scorers._count_options_considered
    first_mcq = scorers.mcq_bias_scorer
    install_conditional_nan_compat()

    assert scorers.switch_values is first
    assert scorers._count_options_considered is first_options
    assert scorers.mcq_bias_scorer is first_mcq
