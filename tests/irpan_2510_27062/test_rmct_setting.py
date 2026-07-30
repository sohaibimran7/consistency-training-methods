"""Focused tests for the Irpan pair-view RMCT Setting bridges."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import pytest

from ctm.artifacts import ArtifactManifestError, artifact_manifest_path
from ctm.settings import Setting, create_setting
from ctm.settings.runtime import prepare_setting_instance, setting_run_metadata
from scripts.irpan_2510_27062 import rmct_setting as setting_module
from scripts.irpan_2510_27062.sycophancy import build_sycophancy_pairs, normalize_arc_rows
from scripts.irpan_2510_27062.training_views import materialize_training_view

MODULE = "scripts.irpan_2510_27062.rmct_setting"


def _manifest(domain: str, rows: list[dict]) -> dict:
    return {
        "artifact_schema": "ctm_data.irpan_2510_27062.training_view",
        "schema_version": 1,
        "row_count": len(rows),
        "content_sha256": "a" * 64,
        "role": "training",
        "domain": domain,
        "provenance": {"source_artifact": "fixture"},
    }


def _sycophancy_row(index: int = 1) -> dict:
    return {
        "pair_id": f"syc-pair-{index}",
        "example_id": f"base-example-{index}",
        "source": "mmlu",
        "domain": "sycophancy",
        "reference_messages": [{"role": "user", "content": f"clean question {index}"}],
        "variant_messages": [
            {"role": "system", "content": "The user prefers option A."},
            {"role": "user", "content": f"wrapped question {index}"},
        ],
        "choices": [
            {"label": "A", "text": "wrong"},
            {"label": "B", "text": "correct"},
            {"label": "C", "text": "also wrong"},
        ],
        "correct_label": "B",
        "suggested_wrong_label": "A",
    }


def _jailbreak_row(index: int = 1) -> dict:
    return {
        "pair_id": f"jailbreak-pair-{index}",
        "example_id": f"harmbench-example-{index}",
        "source": "harmbench",
        "domain": "jailbreak",
        "reference_messages": [{"role": "user", "content": f"clean harmful request {index}"}],
        "variant_messages": [{"role": "user", "content": f"wrapped harmful request {index}"}],
    }


def _serve_view(monkeypatch: pytest.MonkeyPatch, rows: list[dict], manifest: dict) -> None:
    def read(path: Path, *, expected_domain: str):
        assert path.is_absolute()
        assert expected_domain in {"sycophancy", "jailbreak"}
        return copy.deepcopy(rows), copy.deepcopy(manifest)

    monkeypatch.setattr(setting_module, "_read_training_view", read)


def _materialize_sycophancy_view(tmp_path: Path, *, name: str = "view") -> Path:
    normalized_path = tmp_path / f"{name}.normalized.jsonl"
    pair_path = tmp_path / f"{name}.pairs.jsonl"
    view_path = tmp_path / f"{name}.training.jsonl"
    normalize_arc_rows(
        [
            {
                "id": "arc-1",
                "question": "Which choice is correct?",
                "choices": {
                    "label": ["A", "B", "C"],
                    "text": ["first", "second", "third"],
                },
                "answerKey": "B",
            }
        ],
        subset="ARC-Easy",
        split="train",
        revision="fixture-r1",
        output_path=normalized_path,
    )
    build_sycophancy_pairs(normalized_path, output_path=pair_path)
    materialize_training_view(pair_path, view_path, domain="sycophancy")
    return view_path


def test_sycophancy_factory_prepares_reference_then_variant_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_sycophancy_row(2), _sycophancy_row(1)]
    _serve_view(monkeypatch, rows, _manifest("sycophancy", rows))

    setting = create_setting(
        f"{MODULE}:sycophancy_rmct_setting",
        training_view_path=tmp_path / "sycophancy.training.jsonl",
    )
    assert isinstance(setting, Setting)
    prepared = prepare_setting_instance(setting, load_config={"n_datapoints": 1})

    assert [row["pair_id"] for row in prepared.datapoints] == ["syc-pair-2"]
    assert prepared.training_indices == [1]
    datapoint = prepared.datapoints[0]
    original = copy.deepcopy(datapoint)
    reference = prepared.perturbations[0](datapoint)
    variant = prepared.perturbations[1](datapoint)
    assert reference["messages"] == datapoint["reference_messages"]
    assert variant["messages"] == datapoint["variant_messages"]
    reference["messages"][0]["content"] = "mutated copy"
    variant["messages"][0]["content"] = "mutated copy"
    assert datapoint == original


def test_setting_uses_the_real_verified_training_view_reader_and_rejects_role_tamper_and_missing(
    tmp_path: Path,
) -> None:
    view_path = _materialize_sycophancy_view(tmp_path)
    setting = setting_module.SycophancyRMCTSetting(view_path)
    prepared = prepare_setting_instance(setting)
    assert len(prepared.datapoints) == 1
    assert prepared.datapoints[0]["domain"] == "sycophancy"
    actual_manifest = json.loads(artifact_manifest_path(view_path).read_text(encoding="utf-8"))
    actual_identity = setting.training_artifact_identity()
    assert actual_identity["content_sha256"]
    assert actual_identity["manifest_sha256"] == actual_manifest["manifest_sha256"]

    for role in ("validation", "final"):
        role_path = tmp_path / f"{role}-view.jsonl"
        role_path.write_bytes(view_path.read_bytes())
        role_manifest_path = artifact_manifest_path(role_path)
        role_manifest = json.loads(artifact_manifest_path(view_path).read_text(encoding="utf-8"))
        role_manifest["provenance"]["role"] = role
        role_manifest_path.write_text(json.dumps(role_manifest), encoding="utf-8")
        with pytest.raises(ArtifactManifestError, match=rf"role is '{role}'.*expected 'training'"):
            setting_module.SycophancyRMCTSetting(role_path).load_datapoints()

    tampered_path = tmp_path / "tampered-view.jsonl"
    tampered_path.write_bytes(view_path.read_bytes() + b" ")
    artifact_manifest_path(tampered_path).write_bytes(artifact_manifest_path(view_path).read_bytes())
    with pytest.raises(ArtifactManifestError, match="digest mismatch"):
        setting_module.SycophancyRMCTSetting(tampered_path).load_datapoints()

    with pytest.raises(ArtifactManifestError, match="missing frozen artifact"):
        setting_module.SycophancyRMCTSetting(tmp_path / "missing.jsonl").load_datapoints()


def test_sycophancy_uses_strict_committed_parser_and_scores_correctness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_sycophancy_row()]
    _serve_view(monkeypatch, rows, _manifest("sycophancy", rows))
    setting = setting_module.sycophancy_rmct_setting(tmp_path / "view.jsonl")
    datapoint = setting.load_datapoints()[0]
    parser = setting.answer_parser()
    classify = setting.trait_classifier()

    assert parser("Reasoning.\nANSWER: B") == "B"
    assert parser("Reasoning.\n(B)") == "B"
    assert parser("I considered options A, B, and C.") is None
    assert parser("ANSWER: A\nCorrection.\nANSWER: B") is None
    assert classify("ANSWER: B", datapoint, datapoint["variant_messages"]) == 1.0
    assert classify("ANSWER: A", datapoint, datapoint["variant_messages"]) == 0.0
    assert classify("The user suggested A, but I am unsure.", datapoint, datapoint["variant_messages"]) is None
    metadata = setting.run_metadata()
    assert metadata["evaluation_metrics"] == ["mmlu_accuracy", "followed_wrong_suggestion"]
    assert metadata["unparsed_trait_value"] is None


def test_sycophancy_accepts_choice_and_gold_metadata_without_loose_matching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _sycophancy_row()
    row.pop("choices")
    row.pop("correct_label")
    row["metadata"] = {
        "gold": {
            "valid_labels": ["A", "B", "C"],
            "correct_label": "C",
        }
    }
    rows = [row]
    _serve_view(monkeypatch, rows, _manifest("sycophancy", rows))
    setting = setting_module.SycophancyRMCTSetting(tmp_path / "view.jsonl")
    datapoint = setting.load_datapoints()[0]

    assert setting.trait_classifier()("final answer is (C).", datapoint, []) == 1.0
    assert setting.trait_classifier()("C appears in my explanation.", datapoint, []) is None


def test_selection_and_manifest_identity_are_exact_and_serializable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_sycophancy_row(2), _sycophancy_row(1)]
    manifest = _manifest("sycophancy", rows)
    _serve_view(monkeypatch, rows, manifest)
    path = tmp_path / "view.jsonl"
    setting = setting_module.SycophancyRMCTSetting(path)
    loaded = setting.load_datapoints(n_datapoints=1)

    assert loaded[0]["example_id"] == "base-example-2"
    assert loaded[0]["source"] == "mmlu"
    identity = setting.training_artifact_identity()
    assert identity["path"] == str(path.resolve())
    assert identity["artifact_schema"] == manifest["artifact_schema"]
    assert identity["content_sha256"] == manifest["content_sha256"]
    assert identity["role"] == "training"
    assert identity["domain"] == "sycophancy"
    assert len(identity["manifest_sha256"]) == 64
    assert identity["selection"]["selected_pair_ids"] == ["syc-pair-2"]
    assert identity["selection"]["selected_pair_ids_sha256"] == hashlib.sha256(b"syc-pair-2").hexdigest()
    metadata = setting_run_metadata(setting, load_config={"n_datapoints": 1})
    assert metadata["training_artifacts"] == identity
    json.dumps(metadata)


def test_answer_parser_labels_follow_only_the_selected_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _sycophancy_row(1)
    first["choices"] = first["choices"][:2]
    second = _sycophancy_row(2)
    second["choices"].append({"label": "D", "text": "fourth"})
    rows = [first, second]
    _serve_view(monkeypatch, rows, _manifest("sycophancy", rows))
    setting = setting_module.SycophancyRMCTSetting(tmp_path / "view.jsonl")
    setting.load_datapoints(n_datapoints=1)

    assert setting.answer_parser()("ANSWER: B") == "B"
    assert setting.answer_parser()("ANSWER: D") is None


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows, manifest: manifest.update(role="validation"), "role.*validation"),
        (lambda rows, manifest: manifest.update(domain="jailbreak"), "manifest domain"),
        (lambda rows, manifest: rows[0].update(domain="jailbreak"), "has domain"),
        (lambda rows, manifest: rows[0].pop("suggested_wrong_label"), "no suggested_wrong_label"),
        (
            lambda rows, manifest: rows[0].update(suggested_wrong_label="B"),
            "must differ from correct label",
        ),
        (lambda rows, manifest: rows.append(copy.deepcopy(rows[0])), "duplicate pair_id"),
    ],
)
def test_sycophancy_fails_closed_on_role_domain_and_duplicate_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    rows = [_sycophancy_row()]
    manifest = _manifest("sycophancy", rows)
    mutation(rows, manifest)
    manifest["row_count"] = len(rows)
    _serve_view(monkeypatch, rows, manifest)

    with pytest.raises(ValueError, match=match):
        setting_module.SycophancyRMCTSetting(tmp_path / "bad.jsonl").load_datapoints()


def test_setting_rejects_empty_view_and_disagreeing_canonical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_view(monkeypatch, [], _manifest("sycophancy", []))
    setting = setting_module.SycophancyRMCTSetting(tmp_path / "first.jsonl")
    with pytest.raises(ValueError, match="contains no pairs"):
        setting.load_datapoints()
    with pytest.raises(ValueError, match="exactly one canonical"):
        setting.load_datapoints(training_view_path=tmp_path / "second.jsonl")


@pytest.mark.parametrize(
    "reader_error",
    [FileNotFoundError("missing frozen training view"), ValueError("training-view digest mismatch")],
)
def test_reader_missing_and_tamper_failures_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader_error: Exception,
) -> None:
    def fail(_path: Path, *, expected_domain: str):
        assert expected_domain == "jailbreak"
        raise reader_error

    monkeypatch.setattr(setting_module, "_read_training_view", fail)
    with pytest.raises(type(reader_error), match=str(reader_error)):
        setting_module.JailbreakRMCTSetting(tmp_path / "bad.jsonl").load_datapoints()


def test_jailbreak_prepare_is_lazy_and_first_classification_constructs_judge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_jailbreak_row()]
    _serve_view(monkeypatch, rows, _manifest("jailbreak", rows))
    events = {"constructed": 0, "called": 0, "messages": None}

    class FakeJudge:
        def __init__(self, *, model, completion=None, **config):
            events["constructed"] += 1
            self.model = model
            self.config = config
            assert completion is None

        async def __call__(self, response, datapoint, realized_messages):
            events["called"] += 1
            events["messages"] = realized_messages
            return 1.0 if response == "I refuse." else 0.0

        def provenance(self):
            return {"model": self.model, "config": self.config}

    monkeypatch.setattr(setting_module, "RefusalJudge", FakeJudge)
    setting = create_setting(
        f"{MODULE}:jailbreak_rmct_setting",
        training_view_path=tmp_path / "jailbreak.training.jsonl",
        grader_model="unit/refusal-grader",
        grader_config={"retries": 0},
    )
    assert isinstance(setting, Setting)
    prepared = prepare_setting_instance(setting)
    assert events == {"constructed": 0, "called": 0, "messages": None}
    assert prepared.answer_parser is None
    assert prepared.training_indices == [1]
    setting_run_metadata(setting)
    assert events["constructed"] == 0

    realized = prepared.perturbations[1](prepared.datapoints[0])["messages"]
    assert asyncio.run(prepared.trait_classifier("I refuse.", prepared.datapoints[0], realized)) == 1.0
    assert events["constructed"] == events["called"] == 1
    assert events["messages"] == realized


def test_jailbreak_injected_completion_is_deterministic_and_records_grader_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_jailbreak_row()]
    _serve_view(monkeypatch, rows, _manifest("jailbreak", rows))
    observed: dict[str, str] = {}

    async def complete(*, system: str, user: str, model: str) -> str:
        observed.update(system=system, user=user, model=model)
        return "<refusal>yes</refusal>"

    setting = setting_module.jailbreak_rmct_setting(
        tmp_path / "view.jsonl",
        grader_model="unit/refusal-grader",
        grader_config={"retries": 0, "retry_delay": 0},
        completion=complete,
    )
    prepared = prepare_setting_instance(setting)
    realized = prepared.perturbations[1](prepared.datapoints[0])["messages"]
    score = asyncio.run(prepared.trait_classifier("I cannot help with that.", prepared.datapoints[0], realized))

    assert score == 1.0
    assert observed["model"] == "unit/refusal-grader"
    assert "wrapped harmful request 1" in observed["user"]
    assert "clean harmful request 1" not in observed["user"]
    metadata = setting_run_metadata(setting)
    setting_metadata = metadata["setting_metadata"]
    assert setting_metadata["grader_model"] == "unit/refusal-grader"
    assert setting_metadata["grader_config"] == {"retries": 0, "retry_delay": 0}
    assert setting_metadata["pair_direction"] == "clean_reference_to_wrapped_variant"
    assert metadata["trait_classifier_identity"]["provenance"]["model"] == "unit/refusal-grader"


def test_jailbreak_accepts_an_injected_judge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [_jailbreak_row()]
    _serve_view(monkeypatch, rows, _manifest("jailbreak", rows))

    class InjectedJudge:
        model = "unit/injected"

        async def __call__(self, response, datapoint, realized_messages):
            del datapoint, realized_messages
            return 0.0 if response == "substantive answer" else 1.0

        def provenance(self):
            return {"model": self.model, "kind": "injected"}

    setting = setting_module.JailbreakRMCTSetting(tmp_path / "view.jsonl", judge=InjectedJudge())
    datapoint = setting.load_datapoints()[0]
    realized = setting.perturbations()[0](datapoint)["messages"]

    assert asyncio.run(setting.trait_classifier()("substantive answer", datapoint, realized)) == 0.0
    assert setting.run_metadata()["grader_provenance"] == {"model": "unit/injected", "kind": "injected"}
