"""Static contracts for the profile-bound native ACT-Max plan and launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
PLAN = ROOT / "experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_act_max_from_base_20260804.yaml"
LAUNCHER = ROOT / "infra/vastai/run_qwen35_act_max.sh"
SELECTION = (
    "artifacts/act-max-training-20260804/"
    "act-max-training-cec56e4d33531a9f997740850a654e7ceaf16b6c2d4830108861df903660cbd5.jsonl"
)
MANIFEST = (
    "artifacts/act-max-training-20260804/"
    "act-max-training-manifest-18b8f70b8260c3d2deaf3345efcfddd41da3e9b7c61ed50d52664bbca56bd04c.json"
)


def test_act_max_plan_is_explicitly_profile_bound_and_uses_native_repaired_act_inputs():
    source = experiment.load_experiment_source(PLAN)
    spec = source["spec"]
    assert source["name"] == "act-max-from-base-20260804"
    assert source["experiment_factory"] == "experiments.act_max.plan:compile_experiment"
    assert spec["topology_profiles"] == {
        "single-gpu": {
            "target": "act-max-1gpu",
            "run_name": "act-max-none-n2800-5600steps-lr-1e-4",
            "gpu_count": 1,
        }
    }
    assert spec["model"] == "Qwen/Qwen3.5-9B"
    assert spec["selection"] == SELECTION
    assert spec["selection_manifest"] == MANIFEST

    compiled = experiment.load_experiment(PLAN, topology_profile="single-gpu")
    assert compiled["supervised_topology_profile"] == "single-gpu"
    assert compiled["supervised_topology"] == {"gpu_count": 1, "device": "cuda:0"}
    assert compiled["data_preparation"][0]["target"] == "act-max-1gpu"
    args = compiled["training"][0]["args"]
    assert args["method"] == "act"
    assert args["data"] == ["${selection}:2800"]
    assert args["data_manifest"] == ["${selection_manifest}"]
    assert args["reference_messages_field"] == "unbiased_messages"
    assert args["variant_messages_field"] == "biased_messages"
    assert args["require_full_reference_suffix_alignment"] is True
    assert args["qwen35_consistency_preflight"] is True
    assert args["optimizer_config"]["learning_rate"] == 0.0001
    assert args["epochs"] == 2
    assert args["minimum_optimizer_steps"] == 5600
    assert args["save_every"] == 700
    assert args["save_state"] is True
    assert args["lora_config"]["target_modules"] == [
        "q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"
    ]

    with pytest.raises(experiment.ExperimentConfigError, match="requires an explicit topology profile"):
        experiment.load_experiment(PLAN)


def test_act_max_launcher_is_syntax_valid_and_hard_binds_the_single_gpu_profile():
    syntax = subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    help_result = subprocess.run(["bash", str(LAUNCHER), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert help_result.returncode == 0, help_result.stderr
    assert "--topology-profile single-gpu" in help_result.stdout
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'target="act-max-1gpu"' in text
    assert 'run_name="act-max-none-n2800-5600steps-lr-1e-4"' in text
    assert "validate_visible_gpu" in text
    assert "verify_selection_contract" in text
    assert "--topology-profile \"$topology_profile\"" in text
    assert "--stages data_preparation,training" in text
    assert "--gpus" not in text
