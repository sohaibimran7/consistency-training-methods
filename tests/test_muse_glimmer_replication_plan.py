"""Contract tests for the uncapped Muse Glimmer RMCT replication."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from experiments.muse_glimmer_rmct_replication import plan
from scripts.run_experiment import _argument_tokens, load_experiment


PLAN_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "muse_glimmer_rmct_replication"
    / "muse_glimmer_rmct_replication.yaml"
)


def test_authored_plan_compiles_to_32_four_gpu_uncapped_segments():
    compiled = load_experiment(PLAN_PATH, topology_profile=plan.TOPOLOGY_PROFILE)

    assert compiled["muse_replication"]["frozen_spec_sha256"] == plan.FROZEN_SPEC_SHA256
    assert len(compiled["training"]) == 32
    assert {entry["gpu_count"] for entry in compiled["training"]} == {4}
    for entry in compiled["training"]:
        args = entry["args"]
        tokens = _argument_tokens(args)
        assert args["local_device"] == "cuda:0"
        assert args["local_rollout_gpus"] == "1,2,3"
        assert args["local_hf_language_model_only"] is True
        assert args["local_vllm_language_model_only"] is True
        assert args["no_max_new_tokens"] is True
        assert "max_new_tokens" not in args
        assert "--no-max-new-tokens" in tokens
        assert "--max-new-tokens" not in tokens


def test_evaluation_pool_is_exactly_50_50_100_for_every_condition():
    evaluation = plan.FROZEN_SPEC["evaluation"]

    assert evaluation["conditions"] == ["base", "step16", "step64", "final"]
    assert evaluation["datasets"] == {"logiqa": 50, "hellaswag": 50, "hle": 100}
    assert evaluation["tasks_per_condition"] == 21
    assert evaluation["generations_per_condition"] == 1400
    assert evaluation["pool_policy"] == "same_question_ids_for_every_condition_and_clean_biased_pair"
    assert evaluation["model_args"] == {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "hf_language_model_only": True,
        "do_sample": True,
    }
    assert evaluation["verbalisation"]["max_connections"] == 500
    assert evaluation["aita_nta_flip"]["parser_scope"] == "final_output_after_cot_only"


def test_all_generation_contracts_are_null_and_eos_only():
    assert plan.FROZEN_SPEC["sampling"]["max_tokens"] is None
    assert plan.FROZEN_SPEC["sampling"]["termination"] == "eos_only"
    assert plan.FROZEN_SPEC["evaluation"]["generation"]["max_tokens"] is None
    assert plan.FROZEN_SPEC["evaluation"]["aita_nta_flip"]["max_tokens"] is None


@pytest.mark.parametrize("path", [("sampling", "max_tokens"), ("evaluation", "generation", "max_tokens")])
def test_compiler_rejects_any_non_null_generation_cap(path):
    mutated = copy.deepcopy(plan.FROZEN_SPEC)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = 500

    with pytest.raises(plan.PlanError, match="frozen scientific/runtime contract"):
        plan.compile_experiment(
            name=plan.CONDITION_NAME,
            spec=mutated,
            topology_profile=plan.TOPOLOGY_PROFILE,
        )


def test_segment_boundaries_resume_strictly_and_preserve_optimizer_steps(tmp_path):
    first = plan.segment_record(tmp_path, 0)
    fourth = plan.segment_record(tmp_path, 3)

    assert first["parent"] == {
        "kind": "pinned_base_snapshot",
        "resume": False,
        "repo_id": plan.MODEL_ID,
        "revision": plan.MODEL_REVISION,
    }
    assert fourth["optimizer_step_start"] == 49
    assert fourth["optimizer_step_end"] == 64
    assert fourth["parent"]["optimizer_step"] == 48
    assert fourth["parent"]["resume_with_optimizer"] is True
    assert fourth["parent"]["resume_state_required"] is True
    assert plan.MINIMUM_COMPARISON_OPTIMIZER_STEP == 64


def test_model_runtime_and_data_are_pinned():
    assert plan.MODEL_REVISION == "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
    assert plan.FROZEN_SPEC["runtime"] == {
        "transformers_version": "5.15.1",
        "vllm_git_commit": "8c2bbe00d58a930c6c09a80495728b26b79d9200",
        "vllm_released_wheel_acceptable": False,
        "vllm_installation": "clean_pinned_source_checkout",
        "cuda_toolkit": {
            "version": "12.9.1",
            "nvcc": "12.9.86",
            "sbsa_installer_sha256": "64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6",
        },
        "receipt_dir": "artifacts/muse-glimmer-runtime-cu129-20260824",
    }
    assert plan.DATA_SHA256 == "cc7842566093e10d3867514e40b55d62ba56521fee24e2ba3695036295199075"
    assert plan.MANIFEST_SHA256 == "eac0682fe0286126cc5928253e2e2ef4968eee50775a65feb21d061eec6853cc"
