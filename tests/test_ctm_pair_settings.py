from __future__ import annotations

import asyncio

import pytest

from ctm.artifacts import write_verified_jsonl_artifact
from ctm.settings.pairs import PairSetting, RefusalPairSetting
from ctm.settings.runtime import prepare_setting_instance, setting_run_metadata
from ctm_data.adapters.mcq_bias.setting import MCQCorrectnessPairSetting
from ctm_data.pairs import canonical_pair_row


def _pair(pair_id: str, *, labels=("A", "B", "C"), correct="B", biased="C") -> dict:
    return canonical_pair_row(
        {
            "pair_id": pair_id,
            "source_id": pair_id,
            "source": "unit",
            "reference_messages": [{"role": "user", "content": f"clean {pair_id}"}],
            "variant_messages": [{"role": "user", "content": f"biased {pair_id}"}],
            "metadata": {
                "valid_labels": list(labels),
                "correct_label": correct,
                "biased_option": biased,
                "prompt_family": "irpan",
            },
        }
    )


def _write_pairs(path, rows):
    return write_verified_jsonl_artifact(
        path,
        rows,
        artifact_schema="ctm.prompt_pairs",
        schema_version=1,
        provenance={"source": "unit"},
        row_validator=canonical_pair_row,
        nonempty=True,
    )


def test_pair_setting_loads_exact_prefix_and_records_selection(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [_pair("one"), _pair("two")])

    setting = PairSetting(path)
    datapoints = setting.load_datapoints(n_datapoints=1)
    perturbations = setting.perturbations()

    assert [row["pair_id"] for row in datapoints] == ["one"]
    assert perturbations[0](datapoints[0])["messages"][0]["content"] == "clean one"
    assert perturbations[1](datapoints[0])["messages"][0]["content"] == "biased one"
    identity = setting.training_artifact_identity()
    assert identity["selection"]["selected_source_ids"] == ["one"]


def test_mcq_correctness_setting_supports_irpan_parser_and_abstains_outside_row_labels(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [_pair("three-choice"), _pair("two-choice", labels=("A", "B"), biased="A")])
    setting = MCQCorrectnessPairSetting(path, prompt_family="irpan")
    rows = setting.load_datapoints()
    classify = setting.trait_classifier()

    assert classify("reasoning\nANSWER: B", rows[0], []) == 1.0
    assert classify("ANSWER: C", rows[0], []) == 0.0
    assert classify("ANSWER: C", rows[1], []) is None
    assert setting.answer_parser()("bare (B)") is None


def test_mcq_correctness_setting_rejects_parser_contract_mismatch(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [_pair("one")])

    with pytest.raises(ValueError, match="prompt_family='irpan'.*expected 'chua'"):
        MCQCorrectnessPairSetting(path, prompt_family="chua").load_datapoints()


class _Judge:
    model = "unit/judge"

    async def __call__(self, response, datapoint, realized_messages):
        del datapoint, realized_messages
        return float(response.startswith("I cannot"))

    def provenance(self):
        return {"model": self.model, "kind": "injected"}


def test_refusal_pair_setting_accepts_injected_shared_judge(tmp_path) -> None:
    path = tmp_path / "pairs.jsonl"
    _write_pairs(path, [_pair("one")])
    setting = RefusalPairSetting(path, judge=_Judge())
    prepared = prepare_setting_instance(setting)

    assert asyncio.run(prepared.trait_classifier("I cannot help.", prepared.datapoints[0], [])) == 1.0
    metadata = setting_run_metadata(setting)
    assert metadata["setting_metadata"]["grader_provenance"] == {
        "kind": "injected",
        "model": "unit/judge",
    }
