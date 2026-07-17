"""Offline tests for WildJailbreak training families and refusal traits."""

from __future__ import annotations

import asyncio
import json

import pytest

from ctm.settings.families import FAMILY_SCHEMA_VERSION, write_frozen_artifact
from ctm_data.adapters.wildjailbreak import JailbreakSetting
from ctm.training.refusal import RefusalJudge, RefusalJudgeError
from ctm.training.refusal.parse import RefusalParseError, parse_refusal_verdict
from ctm_data.adapters.wildjailbreak.builder import (
    WILDJAILBREAK_DATASET,
    WILDJAILBREAK_REVISION,
    build_wildjailbreak_families,
    main as build_main,
    materialize_wildjailbreak,
)


def _completion(values):
    outputs = iter(values)

    async def complete(*, system: str, user: str, model: str) -> str:
        assert "strict evaluator" in system
        assert "<request>" in user and "<response>" in user
        assert model
        return next(outputs)

    return complete


def _source_row(
    base: str = "base-a",
    adversarial: str = "adversarial-a",
    *,
    data_type: str = "adversarial_harmful",
    tactics=None,
) -> dict:
    return {
        "vanilla": base,
        "adversarial": adversarial,
        "data_type": data_type,
        "tactics": tactics or ["roleplay"],
        "completion": "ignored source completion",
    }


def _source_rows():
    return [
        _source_row(adversarial="a-1", tactics=["roleplay"]),
        _source_row(adversarial="a-2", tactics=["encoding"]),
        _source_row("base-b", "b-1", data_type="adversarial_benign", tactics=["roleplay"]),
        _source_row("base-b", "b-2", data_type="adversarial_benign", tactics=["encoding"]),
    ]


def test_refusal_parser_is_strict():
    assert parse_refusal_verdict(" <refusal>YES</refusal>\n") is True
    assert parse_refusal_verdict("<refusal>no</refusal>") is False
    assert parse_refusal_verdict("Reasoning complete. <refusal>yes</refusal>") is True
    for bad in (
        "",
        "yes",
        "<refusal>maybe</refusal>",
        "<refusal>yes</refusal><refusal>no</refusal>",
    ):
        with pytest.raises(RefusalParseError):
            parse_refusal_verdict(bad)


def test_refusal_judge_retries_and_caches_success():
    judge = RefusalJudge(
        model="unit/grader",
        completion=_completion(["not parseable", "<refusal>yes</refusal>"]),
        retries=1,
        retry_delay=0,
    )
    datapoint = {"reference_messages": [{"role": "user", "content": "neutral request"}]}
    messages = [{"role": "user", "content": "realized cued request"}]
    assert asyncio.run(judge("response", datapoint, messages)) == 1.0
    assert asyncio.run(judge("response", datapoint, messages)) == 1.0
    assert judge.provenance()["model"] == "unit/grader"


def test_refusal_judge_uses_realized_cued_request():
    observed = {}

    async def complete(*, system: str, user: str, model: str) -> str:
        observed["user"] = user
        return "<refusal>no</refusal>"

    judge = RefusalJudge(completion=complete, retries=0)
    datapoint = {"reference_messages": [{"role": "user", "content": "neutral request"}]}
    realized = [{"role": "user", "content": "adversarial realized request"}]

    assert asyncio.run(judge("response", datapoint, realized)) == 0.0
    assert "adversarial realized request" in observed["user"]
    assert "neutral request" not in observed["user"]


def test_refusal_judge_abstains_after_exhausted_failure():
    judge = RefusalJudge(completion=_completion(["bad"]), retries=0)
    assert asyncio.run(judge.judge("request", "response")) is None


def test_refusal_judge_can_raise_for_debugging():
    judge = RefusalJudge(completion=_completion(["bad"]), retries=0, failure_policy="raise")
    with pytest.raises(RefusalJudgeError, match="failed after 1 attempt"):
        asyncio.run(judge.judge("request", "response"))


@pytest.mark.parametrize("retries", [1.5, True, "2"])
def test_refusal_judge_rejects_non_integer_retries(retries):
    with pytest.raises(ValueError, match="non-negative integer"):
        RefusalJudge(retries=retries)


def test_family_builder_uses_exactly_the_supplied_rows_without_a_holdout():
    families = build_wildjailbreak_families(_source_rows(), n_variants=2, seed="unit")

    assert len(families) == 2
    assert {family["metadata"]["valence"] for family in families} == {"harmful", "benign"}
    assert all(len(family["variants"]) == 2 for family in families)
    assert all("completion" not in family for family in families)


def test_family_builder_is_input_order_independent():
    rows = _source_rows()
    assert build_wildjailbreak_families(rows, n_variants=1, seed="unit") == build_wildjailbreak_families(
        list(reversed(rows)), n_variants=1, seed="unit"
    )


def test_family_builder_rejects_unselected_types_and_incomplete_groups():
    rows = [*_source_rows(), _source_row("unused", "unused", data_type="vanilla_harmful")]
    with pytest.raises(ValueError, match="unsupported WildJailbreak data_type"):
        build_wildjailbreak_families(rows, n_variants=2)
    with pytest.raises(ValueError, match="incomplete prompt family"):
        build_wildjailbreak_families([_source_row()], n_variants=2)


def test_materializer_records_source_identity_and_never_overwrites(tmp_path):
    path = tmp_path / "families.jsonl"
    manifest = materialize_wildjailbreak(_source_rows(), path, n_variants=2, seed="unit")

    assert manifest["artifact_schema"] == "ctm.prompt_families"
    assert manifest["provenance"]["dataset"] == WILDJAILBREAK_DATASET
    assert manifest["provenance"]["revision"] == WILDJAILBREAK_REVISION
    assert manifest["provenance"]["completion_fields_used"] is False
    assert len(manifest["provenance"]["source_rows_sha256"]) == 64
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_wildjailbreak(_source_rows(), path, n_variants=2)


def test_build_cli_reads_explicit_local_files(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _source_rows()))
    output = tmp_path / "families.jsonl"

    assert build_main(["--input-jsonl", str(source), "--output", str(output), "--n-variants", "2"]) == 0
    assert output.exists()


def test_setting_loads_selected_family_artifact(tmp_path):
    path = tmp_path / "families.jsonl"
    materialize_wildjailbreak(_source_rows(), path, n_variants=2)
    judge = RefusalJudge(completion=_completion(["<refusal>no</refusal>"]), retries=0)
    setting = JailbreakSetting(family_path=path, n_variants=2, judge=judge)

    datapoints = setting.load_datapoints(n_datapoints=1)

    assert len(datapoints) == 1
    assert len(setting.perturbations()) == 3
    assert setting.training_perturbation_indices() == [1, 2]
    assert setting.answer_parser() is None
    assert setting.training_artifact_identity()["selection"]["selected_row_count"] == 1


def test_setting_control_reuses_reference_and_rejects_wrong_k(tmp_path):
    path = tmp_path / "families.jsonl"
    materialize_wildjailbreak(_source_rows(), path, n_variants=2)
    setting = JailbreakSetting(family_path=path, n_variants=2, control=True)
    datapoint = setting.load_datapoints(n_datapoints=1)[0]
    perturbations = setting.perturbations()
    assert all(perturbation(datapoint)["messages"] == datapoint["reference_messages"] for perturbation in perturbations)

    with pytest.raises(ValueError, match="manifest records 2"):
        JailbreakSetting(family_path=path, n_variants=1).load_datapoints()


def test_setting_rejects_family_without_valence(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = {
        "schema_version": FAMILY_SCHEMA_VERSION,
        "source_id": "x",
        "source": "unit",
        "reference_messages": [{"role": "user", "content": "clean"}],
        "variants": [
            {
                "variant_id": "x:1",
                "messages": [{"role": "user", "content": "variant"}],
            }
        ],
        "metadata": {},
    }
    write_frozen_artifact(path, [row], provenance={"n_variants": 1})
    with pytest.raises(ValueError, match="metadata.valence"):
        JailbreakSetting(family_path=path, n_variants=1).load_datapoints()
