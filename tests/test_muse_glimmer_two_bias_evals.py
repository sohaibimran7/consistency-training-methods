from __future__ import annotations

from pathlib import Path

import pytest

from infra.isambard import run_muse_glimmer_rmct_two_bias_evals_16gpu as module


def test_two_condition_topology_has_three_exact_fourteen_worker_phases() -> None:
    assignments = module.PHASE_ASSIGNMENTS
    assert sorted(assignments) == [1, 2, 3]
    assert [len(assignments[phase]) for phase in (1, 2, 3)] == [14, 14, 14]
    observed = [cell for phase in (1, 2, 3) for cell in assignments[phase]]
    expected = [(condition, task) for condition in (0, 1) for task in range(1, 22)]
    assert sorted(observed) == sorted(expected)
    assert len(observed) == len(set(observed)) == 42


def test_generation_contract_has_no_output_token_cap() -> None:
    assert module.GENERATION_CONFIG == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_connections": 8,
    }
    module._assert_no_token_cap(module.GENERATION_CONFIG, label="test")
    with pytest.raises(module.audited.EvaluationError, match="forbidden output-token cap"):
        module._assert_no_token_cap({"max_tokens": 500}, label="test")
    # A null field is an explicit no-cap attestation, not a cap.
    module._assert_no_token_cap({"max_tokens": None}, label="test")


def test_checkpoint_loader_requires_amendment_and_records_both_step_axes() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "training_amendment.validate_attestation(repository)" in source
    assert "training_amendment.install()" in source
    assert '"checkpoint_axis": "global_training_batch_and_frozen_data_window"' in source
    assert '"global_step": expected_step' in source
    assert '"optimizer_step": loop["optimizer_step"]' in source
    assert '"boundary_amendment":' in source


def test_condition_labels_do_not_claim_global_boundaries_are_optimizer_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "_snapshot_from_receipt", lambda: Path("/tmp/muse-snapshot"))
    module._configure("early")
    early = {condition.name: condition for condition in module.audited.CONDITIONS}
    assert early["step016"].label == "global/data-step-016"
    assert early["step016"].optimizer_step is None
    module._configure("late")
    late = {condition.name: condition for condition in module.audited.CONDITIONS}
    assert late["step064"].label == "global/data-step-064"
    assert late["step064"].optimizer_step is None


def test_model_arguments_require_stochastic_text_only_hf() -> None:
    assert module.HF_MODEL_ARGS == {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "do_sample": True,
        "hf_language_model_only": True,
    }
    assert module.HF_LOCAL_MODEL_ARGS == {"provider": "hf", **module.HF_MODEL_ARGS}
