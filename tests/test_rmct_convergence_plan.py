"""Focused contracts for the sealed B=2 RMCT convergence production plan."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from experiments.rmct_convergence import plan as convergence
from scripts import run_experiment


ROOT = Path(__file__).parent.parent
PLAN = ROOT / "experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_convergence_isambard_20260813.yaml"
RECOVERY_PLAN = (
    ROOT
    / "experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_rmct_convergence_gcall_r2_isambard_20260814.yaml"
)


def test_authored_plan_is_the_frozen_deadline_four_lane_configuration():
    source = run_experiment.load_experiment_source(PLAN)
    spec = source["spec"]

    assert source["name"] == convergence.CONDITION_NAME == "rmct-convergence"
    assert source["experiment_factory"] == "experiments.rmct_convergence.plan:compile_experiment"
    assert spec["topology_profiles"] == {
        convergence.TOPOLOGY_PROFILE: {
            "gpu_count": 4,
            "training_gpus": "all",
            "rollout_gpus": "all",
            "phase_shared": True,
            "deadline_execution_choice": True,
            "comparative_optimality_validated": False,
            "gradient_checkpointing_layers": 16,
        }
    }
    assert spec["setting"]["factory"] == convergence.SETTING_FACTORY
    assert spec["setting"]["setting_config"] == {
        "data_path": convergence.DATA_PATH,
        "manifest_path": convergence.MANIFEST_PATH,
        "expected_manifest_sha256": convergence.MANIFEST_SHA256,
    }
    assert spec["setting"]["data_content_sha256"] == convergence.DATA_SHA256
    assert spec["setting"]["ordering"] == "setting_interleaved_logiqa_hellaswag_and_trainer_no_shuffle"
    assert spec["local"]["worker_parity_attestation"] == convergence.WORKER_PARITY_ATTESTATION
    assert spec["optimizer"] == {
        "learning_rate": 1e-4,
        "learning_rate_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    assert spec["method"] == {
        "loss": "ppo",
        "advantage_estimator": "grpo_normalized",
        "normalization": "pooled",
        "kl_coefficient": 0.05,
        "kl_discount_factor": 0.0,
        "ppo_clip_epsilon": 0.2,
        "anchor_weight": 0.0,
        "anchor_model": "base",
    }
    assert spec["sampling"]["reference_rollouts"] == spec["sampling"]["training_rollouts"] == 96
    assert spec["sampling"]["consistency_rollouts"] == spec["sampling"]["anchor_rollouts"] == 96
    assert spec["loop"] == {
        "batch_size": 2,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "refresh_every": 1,
        "shuffle_datapoints": False,
        "checkpoint_every_optimizer_steps": 16,
        "save_state": True,
    }
    assert spec["convergence"]["hard_cap_optimizer_steps"] == 512
    assert spec["convergence"]["cap_label"] == "capped_not_converged"
    assert spec["convergence"]["first_eligible_optimizer_step"] == 32


def test_compiler_emits_32_exact_sealed_segments_and_strict_parents():
    compiled = run_experiment.load_experiment(PLAN, topology_profile=convergence.TOPOLOGY_PROFILE)
    entries = compiled["training"]
    segments = compiled["rmct_convergence"]["segments"]

    assert compiled["onpolicy_topology"]["execution_choice"] == "deadline_driven_not_comparative_optimality_validated"
    assert len(entries) == len(segments) == convergence.TOTAL_SEGMENTS == 32
    assert compiled["rmct_convergence"]["convergence"]["hard_cap_optimizer_steps"] == 512
    for index, (entry, segment) in enumerate(zip(entries, segments, strict=True)):
        args = entry["args"]
        assert entry["gpu_count"] == 4
        assert entry["target"] == convergence.target_name(index)
        assert args["run_name"] == convergence.run_name(index)
        assert args["load_config"] == {"n_datapoints": 32, "segment_index": index}
        assert args["batch_size"] == 2
        assert args["gradient_accumulation_steps"] == 1
        assert args["no_shuffle_datapoints"] is True
        assert args["checkpoint_every"] == 16
        assert args["save_state"] is True
        assert args["local_phase_shared"] is True
        assert args["local_training_gpus"] == args["local_rollout_gpus"] == "all"
        assert args["local_rollout_seed_base"] == 42
        assert args["local_qwen35_rollout_parity_attestation"].endswith(convergence.WORKER_PARITY_ATTESTATION)
        assert args["local_gradient_checkpointing_layers"] == 16
        assert args["local_forward_microbatch_max_datums"] == 8
        assert args["local_forward_microbatch_max_tokens"] == 20480
        assert args["n_ref_rollouts"] == args["n_train_rollouts"] == 96
        assert args["n_consistency_rollouts"] == args["n_anchor_rollouts"] == 96
        assert args["anchor_weight"] == 0.0
        assert args["local_ppo_clip_epsilon"] == 0.2
        assert segment["optimizer_step_start"] == index * 16 + 1
        assert segment["optimizer_step_end"] == (index + 1) * 16
        assert segment["expected_batches"] == 16
        assert segment["dataset_balance"] == "one_logiqa_plus_one_hellaswag_per_optimizer_update"
        if index == 0:
            assert "resume_from" not in args
            assert segment["parent"]["kind"] == "pinned_base_snapshot"
        else:
            assert args["resume_with_optimizer"] is True
            assert args["resume_state_required"] is True
            assert args["resume_from"].endswith(
                f"/{convergence.CONDITION_NAME}_{convergence.run_name(index - 1)}"
            )


def test_compiler_rejects_profile_or_hyperparameter_drift():
    source = run_experiment.load_experiment_source(PLAN)
    with pytest.raises(convergence.PlanError, match="explicit topology profile"):
        convergence.compile_experiment(name=convergence.CONDITION_NAME, spec=source["spec"], topology_profile=None)

    drift = deepcopy(source["spec"])
    drift["loop"]["shuffle_datapoints"] = True
    with pytest.raises(convergence.PlanError, match="frozen production contract"):
        convergence.compile_experiment(
            name=convergence.CONDITION_NAME,
            spec=drift,
            topology_profile=convergence.TOPOLOGY_PROFILE,
        )


def test_gcall_r2_clean_recovery_preserves_the_science_but_uses_a_new_all_gc_namespace():
    original = run_experiment.load_experiment(PLAN, topology_profile=convergence.TOPOLOGY_PROFILE)
    source = run_experiment.load_experiment_source(RECOVERY_PLAN)
    recovery = run_experiment.load_experiment(RECOVERY_PLAN, topology_profile=convergence.TOPOLOGY_PROFILE)
    custody = recovery["rmct_convergence"]

    assert source["spec"]["recovery"] == convergence.RECOVERY_METADATA
    assert custody["condition"] == convergence.CONDITION_NAME
    assert custody["run_prefix"] == convergence.RUN_PREFIX == "rmct-convergence-gcall-r2"
    assert custody["recovery"] == convergence.RECOVERY_METADATA
    assert custody["recovery"]["recovery_of"]["prior_job_id"] == "6011179"
    assert custody["recovery"]["recovery_of"]["successful_optimizer_steps"] == 0
    assert custody["recovery"]["recovery_of"]["checkpoint_written"] is False
    assert custody["recovery"]["execution_delta"]["activation_checkpointing_layers"] == {
        "from": "first16",
        "to": "all32",
        "local_gradient_checkpointing_layers_cli": "omitted",
    }
    assert custody["recovery"]["scientific_contract_unchanged"] is True
    assert custody["recovery"]["recovery_parent_artifact"] == {
        "path": convergence.RECOVERY_PARENT_ARTIFACT,
        "sha256": convergence.RECOVERY_PARENT_ARTIFACT_SHA256,
    }
    assert custody["topology"]["gradient_checkpointing_layers"] == "all"

    # Every rate/data/optimizer field stays byte-for-byte represented by the
    # same frozen mappings; only execution namespace/checkpointing changes.
    for field in ("base_snapshot", "data", "optimizer", "method", "sampling", "loop", "convergence"):
        assert custody[field] == original["rmct_convergence"][field]
    assert len(recovery["training"]) == 32
    first = recovery["training"][0]["args"]
    second = recovery["training"][1]["args"]
    assert first["local_gradient_checkpointing"] is True
    assert "local_gradient_checkpointing_layers" not in first
    assert first["run_name"] == "rmct-convergence-gcall-r2-s001"
    assert first["run_name"] != original["training"][0]["args"]["run_name"]
    assert second["resume_from"].endswith(
        "/rmct-convergence_rmct-convergence-gcall-r2-s001"
    )
