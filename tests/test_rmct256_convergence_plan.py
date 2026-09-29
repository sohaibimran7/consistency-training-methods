"""Static contracts for the fixed RMCT-256 Isambard convergence plan."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.rmct_256_convergence import plan as convergence
from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_rmct256_convergence_isambard_4x64_20260811.yaml"
)


def _source() -> dict:
    return experiment.load_experiment_source(PLAN)


def _parent_uri(run_name: str) -> str:
    return (
        f"file://${{project_root}}/logs/{convergence.CONDITION_NAME}/{run_name}/checkpoints/"
        f"{convergence.CONDITION_NAME}_{run_name}"
    )


def test_authored_condition_is_a_fixed_four_pass_isambard_trajectory():
    source = _source()
    spec = source["spec"]

    assert source["name"] == convergence.CONDITION_NAME
    assert source["experiment_factory"] == "experiments.rmct_256_convergence.plan:compile_experiment"
    assert spec["training_only"] is True
    assert spec["onpolicy_target_attestation"] is True
    assert spec["topology_profiles"] == {
        "four-gpu": {
            "target_prefix": "rmct256-convergence",
            "run_name_prefix": convergence.CONDITION_NAME,
            "gpu_count": 4,
            "coordinator_device": "cuda:0",
            "rollout_gpus": [1, 2, 3],
            "gradient_checkpointing_layers": "all",
        }
    }
    assert spec["base_snapshot"] == {
        "repo_id": "Qwen/Qwen3.5-9B",
        "revision": convergence.BASE_SNAPSHOT,
        "offline_only": True,
    }
    assert spec["isambard_runtime"] == {
        "platform": "isambard-ai-gh200",
        "architecture": "aarch64",
        "vllm_version": "0.21.0",
        "vllm_cuda": "12.9",
        "transformers_version": "5.5.4",
        "gdn_prefill_backend": "triton",
    }
    assert spec["selection"] == {
        "pairs_path": convergence.SELECTION_PATH,
        "selection_manifest": convergence.SELECTION_MANIFEST_PATH,
        "selection_content_sha256": convergence.SELECTION_CONTENT_SHA256,
        "selection_manifest_sha256": convergence.SELECTION_MANIFEST_SHA256,
        "segment_manifest": convergence.SEGMENT_MANIFEST_PATH,
        "segment_manifest_sha256": convergence.SEGMENT_MANIFEST_SHA256,
    }
    assert spec["setting_config"] == {"control": False}
    assert spec["convergence"] == {
        "mode": "optimizer_data_segment",
        "hard_cap_passes": 4,
        "segments_per_pass": 4,
        "rows_per_segment": 64,
        "updates_per_segment": 16,
        "row_offsets": [0, 64, 128, 192],
        "minimum_completed_passes": 1,
        "full_pass_min_delta": 0.01,
        "non_improving_pass_patience": 2,
        "retain": "retain_all_16_step_checkpoints",
        "cap_outcome": "capped_while_improving",
    }
    assert spec["optimizer"] == {
        "learning_rate": 0.0001,
        "learning_rate_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    assert spec["rate_matching"]["kl_discount_factor"] == 0.0
    assert spec["rate_matching"]["ppo_clip_epsilon"] == 0.2


def test_compiler_emits_all_static_segments_with_unique_namespaces_and_parents():
    compiled = experiment.load_experiment(PLAN, topology_profile="four-gpu")
    entries = compiled["training"]
    segments = compiled["rmct256_convergence"]["segments"]

    assert compiled["training_only"] is True
    assert compiled["onpolicy_target_attestation"] is True
    assert compiled["onpolicy_topology_profile"] == "four-gpu"
    assert compiled["onpolicy_topology"] == {
        "gpu_count": 4,
        "coordinator_device": "cuda:0",
        "rollout_gpus": [1, 2, 3],
    }
    assert compiled["rmct256_convergence"]["base_snapshot_commit"] == convergence.BASE_SNAPSHOT
    assert compiled["rmct256_convergence"]["optimizer"] == {
        "learning_rate": 0.0001,
        "learning_rate_schedule": "constant",
        "beta1": 0.9,
        "beta2": 0.95,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
    }
    assert compiled["rmct256_convergence"]["method"] == {
        "loss_fn": "ppo",
        "kl_coefficient": 0.05,
        "kl_discount_factor": 0.0,
        "ppo_clip_epsilon": 0.2,
    }
    assert compiled["rmct256_convergence"]["setting_config"] == {"control": False}
    assert compiled["rmct256_convergence"]["hard_cap"] == {
        "passes": 4,
        "segments": 16,
        "updates": 256,
        "extension_policy": "new_explicit_plan_required",
    }
    assert compiled["rmct256_convergence"]["plateau"] == {
        "minimum_completed_passes": 1,
        "full_pass_min_delta": 0.01,
        "non_improving_pass_patience": 2,
        "decision_boundary": "completed_full_pass_only",
        "retain": "retain_all_16_step_checkpoints",
        "cap_outcome": "capped_while_improving",
    }

    assert len(entries) == len(segments) == convergence.TOTAL_SEGMENTS == 16
    assert len({entry["target"] for entry in entries}) == 16
    assert len({entry["args"]["run_name"] for entry in entries}) == 16
    assert len({entry["args"]["load_config"]["rmct256_convergence_metadata"]["segment_namespace"] for entry in entries}) == 16

    for global_index, (entry, segment) in enumerate(zip(entries, segments, strict=True)):
        pass_index = global_index // 4 + 1
        segment_index = global_index % 4
        row_offset = segment_index * 64
        run_name = f"{convergence.CONDITION_NAME}-p{pass_index:02d}-s{segment_index + 1:02d}"
        target = f"rmct256-convergence-p{pass_index:02d}-s{segment_index + 1:02d}"
        args = entry["args"]
        load_config = args["load_config"]
        metadata = load_config["rmct256_convergence_metadata"]

        assert entry["name"] == f"rmct256_convergence_p{pass_index:02d}_s{segment_index + 1:02d}"
        assert entry["target"] == target
        assert entry["gpu_count"] == 4
        assert entry["command"] == ["${python}", "scripts/train_rlct.py"]
        assert args["run_name"] == run_name
        assert args["local_device"] == "cuda:0"
        assert args["local_rollout_gpus"] == "1,2,3"
        assert args["local_rollout_seed_base"] == 42 + 3 * global_index
        assert args["seed"] == args["lora_config"]["seed"] == 42
        assert args["local_gradient_checkpointing"] is True
        assert "local_gradient_checkpointing_layers" not in args
        assert args["local_vllm_language_model_only"] is True
        assert args["local_vllm_gdn_prefill_backend"] == "triton"
        assert args["local_vllm_max_model_len"] == 32768
        assert args["local_vllm_max_num_seqs"] == 256
        assert args["local_vllm_max_num_batched_tokens"] == 8192
        assert args["local_target_logprob_chunk_size"] == 2048

        assert args["setting_config"] == {
            "data_paths": [convergence.SELECTION_PATH],
            "control": False,
        }
        assert load_config["n_datapoints"] == 64
        assert load_config["row_offset"] == row_offset
        assert load_config["rmct256_segment_index"] == segment_index
        assert load_config["selection_manifest"] == convergence.SELECTION_MANIFEST_PATH
        assert load_config["rmct256_convergence_manifest"] == convergence.SEGMENT_MANIFEST_PATH
        assert load_config["rmct256_convergence_manifest_sha256"] == convergence.SEGMENT_MANIFEST_SHA256

        assert metadata["schema"] == convergence.SEGMENT_METADATA_SCHEMA
        assert metadata["base_snapshot_commit"] == convergence.BASE_SNAPSHOT
        assert metadata["global_segment_index"] == global_index
        assert metadata["pass_index"] == pass_index
        assert metadata["segment_index"] == segment_index
        assert metadata["checkpoint_step"] == (global_index + 1) * 16
        assert metadata["updates"] == 16
        assert metadata["questions"] == 64
        assert metadata["row_offset"] == row_offset
        assert metadata["segment_manifest"] == convergence.SEGMENT_MANIFEST_PATH
        assert metadata["segment_manifest_sha256"] == convergence.SEGMENT_MANIFEST_SHA256
        assert metadata["setting_config"] == {"control": False}
        assert metadata["optimizer"] == compiled["rmct256_convergence"]["optimizer"]
        assert metadata["method"] == compiled["rmct256_convergence"]["method"]
        assert segment == {
            "name": entry["name"],
            "target": target,
            "run_name": run_name,
            "metadata": metadata,
            "resume_from": args.get("resume_from"),
        }

        scientific = {
            "lr": 0.0001,
            "lr_schedule": "constant",
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "kl_coef": 0.05,
            "kl_discount_factor": 0.0,
            "local_ppo_clip_epsilon": 0.2,
            "anchor_weight": 0.0,
            "anchor_model": "base",
            "loss_fn": "ppo",
            "advantage_estimator": "grpo_normalized",
            "normalization": "pooled",
            "n_ref_rollouts": 96,
            "n_train_rollouts": 96,
            "n_consistency_rollouts": 96,
            "n_anchor_rollouts": 96,
            "temperature": 1.0,
            "max_new_tokens": 20480,
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "refresh_every": 1,
            "n_epochs": 1,
            "checkpoint_every": 16,
            "save_state": True,
        }
        for key, expected in scientific.items():
            assert args[key] == expected

        if global_index == 0:
            assert "resume_from" not in args
            assert "resume_with_optimizer" not in args
            assert "resume_state_required" not in args
            assert metadata["parent"] == {
                "kind": "pinned_base_snapshot",
                "resume": False,
                "base_snapshot": {
                    "repo_id": "Qwen/Qwen3.5-9B",
                    "revision": convergence.BASE_SNAPSHOT,
                    "offline_only": True,
                },
            }
        else:
            parent_global = global_index - 1
            parent_pass = parent_global // 4 + 1
            parent_segment = parent_global % 4
            parent_run = f"{convergence.CONDITION_NAME}-p{parent_pass:02d}-s{parent_segment + 1:02d}"
            parent_target = f"rmct256-convergence-p{parent_pass:02d}-s{parent_segment + 1:02d}"
            uri = _parent_uri(parent_run)
            assert args["resume_from"] == uri
            assert args["resume_with_optimizer"] is True
            assert args["resume_state_required"] is True
            assert metadata["parent"] == {
                "kind": "strict_final_checkpoint",
                "resume": True,
                "global_segment_index": parent_global,
                "target": parent_target,
                "run_name": parent_run,
                "uri": uri,
                "checkpoint_step": (parent_global + 1) * 16,
                "expected_kind": "both",
                "expected_final": True,
                "resume_with_optimizer": True,
                "resume_state_required": True,
            }


def test_static_parent_uris_render_without_an_ambient_checkpoint_context():
    compiled = experiment.load_experiment(PLAN, topology_profile="four-gpu")
    context = experiment.initial_context(compiled)
    commands = experiment.planned_commands(compiled, ["training"], context, strict=True)

    assert len(commands) == 16
    assert all("${checkpoint}" not in str(entry) for entry in compiled["training"])
    for global_index, (_stage, _name, argv) in enumerate(commands):
        if global_index == 0:
            assert "--resume-from" not in argv
            continue
        parent_global = global_index - 1
        parent_pass = parent_global // 4 + 1
        parent_segment = parent_global % 4
        parent_run = f"{convergence.CONDITION_NAME}-p{parent_pass:02d}-s{parent_segment + 1:02d}"
        expected_uri = (
            f"file://{ROOT}/logs/{convergence.CONDITION_NAME}/{parent_run}/checkpoints/"
            f"{convergence.CONDITION_NAME}_{parent_run}"
        )
        assert argv[argv.index("--resume-from") + 1] == expected_uri
        assert "--resume-with-optimizer" in argv
        assert "--resume-state-required" in argv

    for _stage, _name, argv in commands:
        assert json.loads(argv[argv.index("--setting-config") + 1]) == {
            "data_paths": [convergence.SELECTION_PATH],
            "control": False,
        }
        for flag, expected in {
            "--beta1": "0.9",
            "--beta2": "0.95",
            "--eps": "1e-08",
            "--weight-decay": "0.0",
            "--grad-clip-norm": "1.0",
            "--kl-discount-factor": "0.0",
            "--local-ppo-clip-epsilon": "0.2",
        }.items():
            assert argv[argv.index(flag) + 1] == expected


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("convergence", "hard_cap_passes"), 5, "convergence.hard_cap_passes"),
        (("convergence", "row_offsets"), [0, 32, 64, 96], "convergence.row_offsets"),
        (("base_snapshot", "revision"), "0" * 40, "base_snapshot.revision"),
        (("local", "gradient_checkpointing_layers"), 16, "local.gradient_checkpointing_layers"),
        (("selection", "segment_manifest_sha256"), "0" * 64, "selection.segment_manifest_sha256"),
        (("setting_config", "control"), True, "setting_config.control"),
        (("rate_matching", "kl_discount_factor"), 0.1, "rate_matching.kl_discount_factor"),
        (("rate_matching", "ppo_clip_epsilon"), 0.1, "rate_matching.ppo_clip_epsilon"),
        (("rate_matching", "ppo_clip_epsilon"), True, "rate_matching.ppo_clip_epsilon"),
    ],
)
def test_compiler_rejects_drift_from_the_fixed_convergence_contract(path, value, match):
    source = deepcopy(_source())
    section, field = path
    source["spec"][section][field] = value

    with pytest.raises(experiment.ExperimentConfigError, match=match):
        experiment.compile_experiment(source, topology_profile="four-gpu")


def test_compiler_requires_the_single_explicit_isambard_topology_profile():
    with pytest.raises(experiment.ExperimentConfigError, match="requires an explicit topology profile"):
        experiment.load_experiment(PLAN)
    with pytest.raises(experiment.ExperimentConfigError, match="unknown RMCT-256 convergence topology profile"):
        experiment.load_experiment(PLAN, topology_profile="eight-gpu")


def test_compiler_rejects_an_omitted_biased_prompt_control_freeze():
    source = deepcopy(_source())
    del source["spec"]["setting_config"]["control"]

    with pytest.raises(experiment.ExperimentConfigError, match="setting_config: missing"):
        experiment.compile_experiment(source, topology_profile="four-gpu")
