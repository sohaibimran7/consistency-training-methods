"""Static contract for the Qwen3.5 on-policy recovery configuration."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).parent.parent
PLAN = ROOT / "experiments" / "rmct_paper_vast_dense_models" / "stage1" / "qwen3_5_9b_onpolicy_recovery_20260801.yaml"


def _source() -> dict:
    value = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_recovery_plan_requires_a_distinct_frozen_none_style_pair_store():
    source = _source()
    spec = source["spec"]
    data = spec["data"]
    training = data["training"]

    assert source["name"] == "rmct_paper_vast_dense_qwen3_5_9b_stage1_recovery_20260801"
    assert spec["training_only"] is True
    assert data["prepare_shared"] is False
    assert data["shared_root"] == "artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801"
    assert data["shared_root"] != "artifacts/rmct-hle-dense-models-shared"
    assert training["prompt_style"] == "none"
    assert training["examples"] == 2048
    assert training["pool_per_dataset"] == training["minimum_per_dataset"] == 1024


def test_recovery_plan_has_only_the_three_fresh_on_policy_training_targets():
    source = _source()
    spec = source["spec"]
    execution = spec["execution"]

    assert spec["conditions"] == [
        {"name": "untrained", "method": "none"},
        {"name": "rate-matching", "method": "rate_matching"},
        {"name": "rate-matching-control", "method": "rate_matching", "control": True},
        {"name": "opct", "method": "opct"},
    ]
    assert spec["artifact_root"] == "artifacts/rmct-hle-qwen3.5-9b-dense-stage1-recovery-20260801"
    assert spec["figure_root"] == "figures/rmct-hle-qwen3.5-9b-dense-stage1-recovery-20260801"
    assert [item["target"] for item in execution["allocations"]] == ["rmct-main", "rmct-control", "opct"]
    assert [item["commands"] for item in execution["allocations"]] == [
        ["rate_matching_lr1"],
        ["rate_matching_control_lr1"],
        ["opct_lr1"],
    ]
    for item in execution["allocations"]:
        assert item["stage"] == "training"
        assert item["gpu_count"] == 8
        assert item["local_device"] == "cuda:0"
        assert item["rollout_gpus"] == [1, 2, 3, 4, 5, 6, 7]
        assert item["gradient_checkpointing_layers"] == "all"
