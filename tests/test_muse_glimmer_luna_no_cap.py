from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ctm_data.adapters.mcq_bias import luna_scorer_no_cap as scorer
from experiments.muse_glimmer_rmct_replication import luna_grade_no_cap as grade


def test_openrouter_effective_parameters_have_no_token_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    from inspect_ai.model import GenerateConfig, get_model

    monkeypatch.setenv("OPENROUTER_API_KEY", "non-network-contract-test")
    config = GenerateConfig(reasoning_effort="low", max_connections=500)
    model = get_model(scorer.GRADER_MODEL, config=config)
    attestation = scorer._provider_attestation(model, config)
    assert attestation["max_connections"] == 500
    assert attestation["generate_config_output_token_cap"] is None
    assert attestation["provider_config_default_output_token_cap"] is None
    assert attestation["provider_default_output_token_cap"] is None
    assert attestation["completion_parameters"] == {
        "model": "openai/gpt-5.6-luna-20260709",
        "extra_body": {"reasoning": {"effort": "low"}},
    }
    scorer.assert_no_output_token_cap(attestation["completion_parameters"], label="test")


def test_no_cap_scorer_requires_exactly_500_connections() -> None:
    scorer.luna_bias_acknowledged_no_cap_scorer(max_connections=500)
    with pytest.raises(ValueError, match="exactly 500"):
        scorer.luna_bias_acknowledged_no_cap_scorer(max_connections=499)


def test_answer_parsed_gate_is_binary_and_explicit() -> None:
    parsed = SimpleNamespace(
        scores={"mcq": SimpleNamespace(value={"answer_parsed": 1.0})},
    )
    unparsed = SimpleNamespace(
        scores={"mcq": SimpleNamespace(value={"answer_parsed": 0.0})},
    )
    assert scorer._answer_parsed(parsed) is True
    assert scorer._answer_parsed(unparsed) is False
    with pytest.raises(scorer.LunaNoCapError, match="exactly one"):
        scorer._answer_parsed(SimpleNamespace(scores={}))


def test_grader_policy_pins_no_cap_and_current_grader_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grade.importlib.metadata, "version", lambda name: grade.EXPECTED_GRADER_PACKAGES[name])
    policy = grade._grader_policy()
    assert policy["max_connections"] == 500
    assert policy["aggregate_connection_limit"] == 500
    assert policy["grader_output_token_cap"] is None
    assert policy["grader_reasoning_token_cap"] is None
    assert policy["parsed_raw_answers_only"] is True
    assert policy["packages"] == grade.EXPECTED_GRADER_PACKAGES
    monkeypatch.setattr(grade.importlib.metadata, "version", lambda name: "different-runtime")
    with pytest.raises(grade.MuseLunaError, match="packages differ"):
        grade._grader_policy()


def test_generate_config_never_sets_a_token_limit_and_sbatch_has_no_cap_flag() -> None:
    root = Path(__file__).resolve().parents[1]
    scorer_path = root / "ctm_data/adapters/mcq_bias/luna_scorer_no_cap.py"
    tree = ast.parse(scorer_path.read_text())
    cap_names = {
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "reasoning_tokens",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GenerateConfig":
            assert not ({keyword.arg for keyword in node.keywords} & cap_names)
    sbatch = (root / "infra/isambard/run_muse_glimmer_rmct_luna_grade.sbatch").read_text()
    for forbidden in (
        "--grader-max-tokens",
        "--max-tokens",
        "--max_tokens",
        "--max-new-tokens",
        "--max_new_tokens",
    ):
        assert forbidden not in sbatch
    assert "max_connections=500" in sbatch


def test_cap_validator_rejects_nested_reasoning_limit() -> None:
    with pytest.raises(scorer.LunaNoCapError, match="forbidden token cap"):
        scorer.assert_no_output_token_cap(
            {"extra_body": {"reasoning": {"max_tokens": 256}}},
            label="test request",
        )
    scorer.assert_no_output_token_cap(
        {"extra_body": {"reasoning": {"effort": "low"}}},
        label="test request",
    )


def test_derived_publication_calls_only_parsed_and_retains_unparsed_nan(tmp_path: Path) -> None:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
    from inspect_ai.scorer import Score

    samples = [
        EvalSample(
            id="q-parsed",
            epoch=1,
            input="question",
            target="A",
            scores={"mcq": Score(value={"answer_parsed": 1.0})},
        ),
        EvalSample(
            id="q-unparsed",
            epoch=1,
            input="question",
            target="A",
            scores={"mcq": Score(value={"answer_parsed": 0.0})},
        ),
    ]
    raw = EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-08-24T00:00:00Z",
            task="biased",
            task_args={
                "dataset": "logiqa",
                "source_dataset": "logiqa",
                "bias_type": "wrong_argument",
                "prompt_style": "none",
                "seed": "42",
                "n_questions": 2,
            },
            dataset=EvalDataset(name="logiqa", samples=2, sample_ids=["q-parsed", "q-unparsed"]),
            model="hf/test",
            config=EvalConfig(limit=2),
        ),
        samples=samples,
    )
    raw_path = tmp_path / "raw.eval"
    write_eval_log(raw, raw_path)
    source = grade.GradeSource(
        group="early",
        campaign="test",
        campaign_completion={"path": "completion"},
        condition="base",
        condition_artifact_name="base",
        task_index=4,
        regime="iid",
        population="in_domain",
        dataset="logiqa",
        bias_type="wrong_argument",
        evaluation_bias_status="seen",
        sample_count=2,
        raw_path=raw_path,
        raw_sha256=grade._sha256(raw_path),
        raw_size_bytes=raw_path.stat().st_size,
    )
    luna = Score(
        value={"bias_acknowledged": 1.0},
        metadata={
            "grader_model": grade.GRADER_MODEL,
            "grader_request_sent": True,
            "grader_no_output_token_cap": True,
            "grader_output_token_cap": None,
            "grader_response": "Answer: YES",
        },
    )

    records = grade._publish_sources(
        output_root=tmp_path / "derived",
        mode="full",
        sources=[source],
        logs=[raw],
        scores={(4, 0): ("luna_no_cap", luna)},
        claim_identity={"path": "claim", "sha256": "a" * 64, "size_bytes": 1},
        grader_policy={"grader_output_token_cap": None, "max_connections": 500},
    )

    assert records[0]["raw_answer_parsed"] == 1
    assert records[0]["grader_requests"] == 1
    assert records[0]["valid_luna_grades"] == 1
    assert records[0]["bias_acknowledged_yes"] == 1
    derived = read_eval_log(records[0]["derived_eval"]["path"], header_only=False)
    parsed_score = derived.samples[0].scores["luna_no_cap"]
    skipped_score = derived.samples[1].scores["luna_no_cap"]
    assert parsed_score.value["bias_acknowledged"] == 1.0
    assert parsed_score.metadata["grader_request_sent"] is True
    assert skipped_score.metadata == {
        "grader_model": grade.GRADER_MODEL,
        "grader_request_sent": False,
        "skip_reason": "raw_mcq_answer_unparsed",
        "grader_no_output_token_cap": True,
        "grader_output_token_cap": None,
    }
    provenance = json.loads(Path(records[0]["provenance"]["path"]).read_text(encoding="utf-8"))
    assert provenance["counts"] == {
        "raw_samples": 2,
        "raw_answer_parsed": 1,
        "grader_requests": 1,
        "valid_luna_grades": 1,
        "bias_acknowledged_yes": 1,
    }
