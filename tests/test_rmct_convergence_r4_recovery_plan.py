"""Focused contract tests for the fresh 40,960-token r4 recovery."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.rmct_convergence_accelerated import plan as r3
from experiments.rmct_convergence_r4_recovery import plan as r4
from scripts import run_experiment


ROOT = Path(__file__).parent.parent
R3_PLAN = (
    ROOT
    / "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct_convergence_gcall_r2_mb49152_r3_isambard_20260814.yaml"
)
R4_PLAN = ROOT / "experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_convergence_gcall_r2_mb40960_r4_isambard_20260818.yaml"
FAILURE_PARENT = ROOT / "tests/fixtures/rmct_history/failure-parent.json"


def _args_by_index(compiled: dict[str, object]) -> dict[int, dict[str, object]]:
    entries = compiled["training"]
    assert isinstance(entries, list)
    result: dict[int, dict[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, dict)
        args = entry["args"]
        assert isinstance(args, dict)
        load_config = args["load_config"]
        assert isinstance(load_config, dict)
        index = load_config["segment_index"]
        assert isinstance(index, int)
        result[index] = args
    return result


def test_r4_uses_only_the_sealed_r3_step64_parent_and_a_distinct_namespace():
    source = run_experiment.load_experiment_source(R4_PLAN)
    r4_compiled = run_experiment.load_experiment(R4_PLAN, topology_profile=r4.TOPOLOGY_PROFILE)
    custody = r4_compiled["rmct_convergence"]

    assert source["experiment_factory"] == "experiments.rmct_convergence_r4_recovery.plan:compile_experiment"
    assert source["spec"] == r4._frozen_spec()
    assert custody["schema"] == r4.CONTINUATION_SCHEMA
    assert custody["run_prefix"] == r4.RUN_PREFIX == "rmct-convergence-gcall-r2-mb40960-r4"
    assert custody["logical_start_segment_index"] == r4.START_SEGMENT_INDEX == 4
    assert custody["continuation"] == r4.CONTINUATION_METADATA
    assert custody["continuation"]["failed_attempt_not_used_as_state"] == {
        "scheduler": "slurm",
        "job_id": "6031786",
        "outcome": "failed",
        "attempted_run_prefix": r3.RUN_PREFIX,
        "attempted_run_name": "rmct-convergence-gcall-r2-mb49152-r3-s005",
        "attempted_logical_segment_index": 4,
        "recovery_parent_optimizer_step": 64,
        "policy": "fresh_r4_namespace_without_checkpoint_or_receipt_from_failed_attempt",
    }
    assert custody["continuation"]["execution_delta"] == {
        "local_forward_microbatch_max_tokens": {"from": 49152, "to": 40960}
    }
    assert custody["continuation"]["failed_attempt_evidence"] == r4.FAILED_ATTEMPT_EVIDENCE
    assert custody["continuation"]["scientific_contract_unchanged"] is True

    entries = r4_compiled["training"]
    segments = custody["segments"]
    assert isinstance(entries, list) and isinstance(segments, list)
    assert len(entries) == len(segments) == 28
    assert [segment["segment_index"] for segment in segments] == list(range(4, 32))
    assert entries[0]["args"]["run_name"] == "rmct-convergence-gcall-r2-mb40960-r4-s005"
    assert entries[0]["args"]["load_config"] == {"n_datapoints": 32, "segment_index": 4}
    assert entries[0]["args"]["resume_from"] == f"file://{r4.parent_checkpoint_path(ROOT)}"
    assert entries[0]["args"]["resume_with_optimizer"] is True
    assert entries[0]["args"]["resume_state_required"] is True
    assert entries[0]["args"]["local_forward_microbatch_max_tokens"] == 40960
    assert "local_gradient_checkpointing_layers" not in entries[0]["args"]
    assert segments[0]["parent"] == {
        "kind": "sealed_external_final_checkpoint",
        "resume": True,
        "condition": "rmct-convergence",
        "run_prefix": r3.RUN_PREFIX,
        "segment_index": 3,
        "target": "rmct-convergence-gcall-r2-mb49152-r3-s004",
        "run_name": "rmct-convergence-gcall-r2-mb49152-r3-s004",
        "uri": f"file://{r4.parent_checkpoint_path(ROOT)}",
        "optimizer_step": 64,
        "expected_kind": "both",
        "expected_final": True,
        "resume_with_optimizer": True,
        "resume_state_required": True,
    }


def test_r4_segment_four_argv_changes_only_namespace_and_physical_token_cap_from_r3():
    r3_compiled = run_experiment.load_experiment(R3_PLAN, topology_profile=r3.TOPOLOGY_PROFILE)
    r4_compiled = run_experiment.load_experiment(R4_PLAN, topology_profile=r4.TOPOLOGY_PROFILE)
    r3_args = _args_by_index(r3_compiled)[4]
    r4_args = _args_by_index(r4_compiled)[4]

    assert set(r3_args) == set(r4_args)
    delta = {key: (r3_args[key], r4_args[key]) for key in r3_args if r3_args[key] != r4_args[key]}
    assert delta == {
        "run_name": ("rmct-convergence-gcall-r2-mb49152-r3-s005", "rmct-convergence-gcall-r2-mb40960-r4-s005"),
        "local_forward_microbatch_max_tokens": (49152, 40960),
    }


def test_r4_failure_parent_artifact_is_immutable_and_binds_job_and_step64_parent():
    assert FAILURE_PARENT.is_file() and not FAILURE_PARENT.is_symlink()
    assert sha256(FAILURE_PARENT.read_bytes()).hexdigest() == r4.FAILURE_PARENT_ARTIFACT_SHA256
    document = json.loads(FAILURE_PARENT.read_text(encoding="utf-8"))
    assert document["schema"] == "rmct-convergence-r4-recovery-failure-parent-v2"
    assert document["failed_attempt_not_used_as_state"]["job_id"] == "6031786"
    assert document["failed_attempt_not_used_as_state"]["outcome"] == "failed"
    assert document["failed_attempt_not_used_as_state"]["policy"] == "fresh_r4_namespace_without_checkpoint_or_receipt_from_failed_attempt"
    assert document["parent"]["run_prefix"] == r3.RUN_PREFIX
    assert document["parent"]["run_name"] == r4.parent_run_name()
    assert document["parent"]["optimizer_step"] == 64
    assert document["failed_attempt_evidence"] == r4.FAILED_ATTEMPT_EVIDENCE
    assert set(document["failed_attempt_evidence"]) == {
        "slurm_output",
        "training_command",
        "training_started",
        "metrics",
        "logs",
        "config",
        "manifest",
    }
    assert document["execution_delta"] == {
        "local_forward_microbatch_max_tokens": {"from": 49152, "to": 40960},
        "scientific_contract_unchanged": True,
    }


def test_r4_rejects_any_earlier_segment_or_frozen_contract_drift():
    source = run_experiment.load_experiment_source(R4_PLAN)
    with pytest.raises(r4.PlanError, match=r"\[4, 31\]"):
        r4.segment_args(ROOT, 3)
    with pytest.raises(r4.PlanError, match="explicit topology profile"):
        r4.compile_experiment(name=r4.CONDITION_NAME, spec=source["spec"], topology_profile=None)

    drift = deepcopy(source["spec"])
    drift["local"]["forward_microbatch_max_tokens"] = 49152
    with pytest.raises(r4.PlanError, match="frozen production contract"):
        r4.compile_experiment(name=r4.CONDITION_NAME, spec=drift, topology_profile=r4.TOPOLOGY_PROFILE)

    science_drift = deepcopy(source["spec"])
    science_drift["optimizer"]["learning_rate"] = 0.0002
    with pytest.raises(r4.PlanError, match="frozen production contract"):
        r4.compile_experiment(name=r4.CONDITION_NAME, spec=science_drift, topology_profile=r4.TOPOLOGY_PROFILE)
