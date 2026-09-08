"""Focused contract tests for the r3 49,152-token RMCT continuation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.rmct_convergence import plan as gcall
from experiments.rmct_convergence_accelerated import plan as accelerated
from scripts import run_experiment


ROOT = Path(__file__).parent.parent
GCALL_PLAN = (
    ROOT
    / "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct_convergence_gcall_r2_isambard_20260814.yaml"
)
ACCELERATED_PLAN = (
    ROOT
    / "experiments/rmct_paper_vast_dense_models/stage1/"
    "qwen3_5_9b_rmct_convergence_gcall_r2_mb49152_r3_isambard_20260814.yaml"
)


def _args_by_logical_index(compiled: dict[str, object]) -> dict[int, dict[str, object]]:
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


def test_r3_starts_at_segment_one_and_is_scientifically_identical_to_gcall_r2():
    gcall_compiled = run_experiment.load_experiment(GCALL_PLAN, topology_profile=gcall.TOPOLOGY_PROFILE)
    source = run_experiment.load_experiment_source(ACCELERATED_PLAN)
    r3 = run_experiment.load_experiment(ACCELERATED_PLAN, topology_profile=accelerated.TOPOLOGY_PROFILE)
    custody = r3["rmct_convergence"]

    assert source["name"] == accelerated.CONDITION_NAME == "rmct-convergence"
    assert source["experiment_factory"] == "experiments.rmct_convergence_accelerated.plan:compile_experiment"
    assert source["spec"] == accelerated._frozen_spec()
    assert custody["schema"] == accelerated.CONTINUATION_SCHEMA
    assert custody["run_prefix"] == accelerated.RUN_PREFIX == "rmct-convergence-gcall-r2-mb49152-r3"
    assert custody["logical_start_segment_index"] == accelerated.START_SEGMENT_INDEX == 1
    assert custody["continuation"] == accelerated.CONTINUATION_METADATA
    assert custody["continuation"]["scientific_contract_unchanged"] is True
    assert custody["continuation"]["execution_delta"] == {
        "local_forward_microbatch_max_datums": 8,
        "local_forward_microbatch_max_tokens": {"from": 20480, "to": 49152},
        "local_target_logprob_chunk_size": 2048,
        "gradient_checkpointing_layers": "all",
    }
    assert custody["continuation"]["same_hardware_all_gradient_checkpointing_preflight_evidence"] == {
        "hardware_scope": "same-GH200",
        "highest_validated_local_forward_microbatch_max_tokens": 49152,
        "all_gradient_checkpointing_layers": "all",
        "result": {
            "path": accelerated.PREFLIGHT_RESULT_ARTIFACT,
            "sha256": accelerated.PREFLIGHT_RESULT_ARTIFACT_SHA256,
        },
        "contract": {
            "path": accelerated.PREFLIGHT_CONTRACT_ARTIFACT,
            "sha256": accelerated.PREFLIGHT_CONTRACT_ARTIFACT_SHA256,
        },
        "result_sha256": "da5783f1aca63cd7f7df500cd85e3460f204ddf2398417c8f44a5c9fadcdd0e9",
        "contract_sha256": "a71c0d74cb65b3b3aba56eb61f03e5ab25fe5cf28e909448b60fe62e051c6394",
    }
    assert custody["continuation"]["continuation_parent_artifact"] == {
        "path": accelerated.CONTINUATION_PARENT_ARTIFACT,
        "sha256": accelerated.CONTINUATION_PARENT_ARTIFACT_SHA256,
    }

    # The r3 compiler inherits every scientific, data, optimizer, sampling,
    # and convergence field from the all-GC r2 contract. The token cap and
    # lineage namespace are deliberately recorded outside this comparison.
    for field in ("base_snapshot", "data", "optimizer", "method", "sampling", "loop", "convergence", "topology"):
        assert custody[field] == gcall_compiled["rmct_convergence"][field]

    entries = r3["training"]
    segments = custody["segments"]
    assert isinstance(entries, list) and isinstance(segments, list)
    assert len(entries) == len(segments) == 31
    assert [segment["segment_index"] for segment in segments] == list(range(1, 32))
    assert [entry["target"] for entry in entries] == [accelerated.target_name(index) for index in range(1, 32)]

    first = entries[0]["args"]
    assert first["run_name"] == "rmct-convergence-gcall-r2-mb49152-r3-s002"
    assert first["load_config"] == {"n_datapoints": 32, "segment_index": 1}
    assert first["resume_from"] == f"file://{accelerated.parent_checkpoint_path(ROOT)}"
    assert first["resume_with_optimizer"] is True
    assert first["resume_state_required"] is True
    assert first["local_gradient_checkpointing"] is True
    assert "local_gradient_checkpointing_layers" not in first
    assert first["local_forward_microbatch_max_datums"] == 8
    assert first["local_forward_microbatch_max_tokens"] == 49152
    assert first["local_target_logprob_chunk_size"] == 2048
    assert first["batch_size"] == 2
    assert first["gradient_accumulation_steps"] == 1
    assert first["n_ref_rollouts"] == first["n_train_rollouts"] == 96
    assert first["n_consistency_rollouts"] == first["n_anchor_rollouts"] == 96
    assert first["max_new_tokens"] == 20480

    parent = segments[0]["parent"]
    assert parent == {
        "kind": "sealed_external_final_checkpoint",
        "resume": True,
        "condition": "rmct-convergence",
        "run_prefix": "rmct-convergence-gcall-r2",
        "segment_index": 0,
        "target": "rmct-convergence-gcall-r2-s001",
        "run_name": "rmct-convergence-gcall-r2-s001",
        "uri": f"file://{accelerated.parent_checkpoint_path(ROOT)}",
        "optimizer_step": 16,
        "expected_kind": "both",
        "expected_final": True,
        "resume_with_optimizer": True,
        "resume_state_required": True,
    }
    assert segments[1]["parent"]["run_prefix"] == accelerated.RUN_PREFIX
    assert segments[1]["parent"]["segment_index"] == 1


def test_r3_segment_one_argv_changes_only_lineage_and_the_validated_token_cap():
    gcall_compiled = run_experiment.load_experiment(GCALL_PLAN, topology_profile=gcall.TOPOLOGY_PROFILE)
    r3 = run_experiment.load_experiment(ACCELERATED_PLAN, topology_profile=accelerated.TOPOLOGY_PROFILE)
    gcall_args = _args_by_logical_index(gcall_compiled)[1]
    r3_args = _args_by_logical_index(r3)[1]

    assert gcall_args["resume_from"] == r3_args["resume_from"] == f"file://{accelerated.parent_checkpoint_path(ROOT)}"
    assert set(gcall_args) == set(r3_args)
    delta = {key: (gcall_args[key], r3_args[key]) for key in gcall_args if gcall_args[key] != r3_args[key]}
    assert delta == {
        "run_name": ("rmct-convergence-gcall-r2-s002", "rmct-convergence-gcall-r2-mb49152-r3-s002"),
        "local_forward_microbatch_max_tokens": (20480, 49152),
    }


def test_r3_rejects_segment_zero_or_any_frozen_contract_drift():
    source = run_experiment.load_experiment_source(ACCELERATED_PLAN)
    with pytest.raises(accelerated.PlanError, match=r"\[1, 31\]"):
        accelerated.segment_args(ROOT, 0)
    with pytest.raises(accelerated.PlanError, match="explicit topology profile"):
        accelerated.compile_experiment(
            name=accelerated.CONDITION_NAME,
            spec=source["spec"],
            topology_profile=None,
        )

    drift = deepcopy(source["spec"])
    drift["local"]["forward_microbatch_max_tokens"] = 40960
    with pytest.raises(accelerated.PlanError, match="frozen production contract"):
        accelerated.compile_experiment(
            name=accelerated.CONDITION_NAME,
            spec=drift,
            topology_profile=accelerated.TOPOLOGY_PROFILE,
        )
