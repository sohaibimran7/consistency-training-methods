"""Static contract for the fresh Qwen3.5 RMCT paper-fidelity run."""

from __future__ import annotations

from pathlib import Path

from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
PLAN = ROOT / "experiments" / "rmct_paper_vast_dense_models" / "stage1" / "qwen3_5_9b_rmct_paper_fidelity_20260803.yaml"


def test_rmct_paper_fidelity_plan_uses_real_pooled_four_datapoint_batches():
    source = experiment.load_experiment_source(PLAN)
    spec = source["spec"]
    rate_matching = spec["rate_matching"]

    assert source["name"] == "rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803"
    assert spec["training_only"] is True
    assert spec["data"]["prepare_shared"] is False
    assert spec["data"]["shared_root"] == "artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801"
    assert spec["data"]["training"]["prompt_style"] == "none"
    assert spec["conditions"] == [
        {"name": "untrained", "method": "none"},
        {"name": "rate-matching", "method": "rate_matching"},
        {"name": "rate-matching-control", "method": "rate_matching", "control": True},
    ]

    # Appendix B's b4 behavior: four whole datapoints are standardized together
    # in one optimizer submission, not four per-item normalized accumulations.
    assert rate_matching["datapoints"] == 64
    assert rate_matching["batch_size"] == 4
    assert rate_matching["gradient_accumulation_steps"] == 1
    assert rate_matching["normalization"] == "pooled"
    assert rate_matching["rollouts"] == {
        "reference": 96,
        "training": 96,
        "consistency": 96,
        "anchor": 96,
    }
    assert rate_matching["max_new_tokens"] == 20480
    assert rate_matching["kl_coefficient"] == 0.05
    assert rate_matching["anchor_weight"] == 0.0
    assert rate_matching["loss"] == "ppo"


def test_compiled_paper_fidelity_plan_emits_only_two_attested_training_targets():
    compiled = experiment.load_experiment(PLAN)
    training = {entry["name"]: entry for entry in compiled["training"]}

    assert set(training) == {"rate_matching_lr1", "rate_matching_control_lr1"}
    for entry in training.values():
        args = entry["args"]
        assert entry["command"] == ["${python}", "scripts/train_rlct.py"]
        assert args["batch_size"] == 4
        assert args["gradient_accumulation_steps"] == 1
        assert args["normalization"] == "pooled"
        assert args["n_ref_rollouts"] == args["n_train_rollouts"] == args["n_consistency_rollouts"] == 96
        assert args["max_new_tokens"] == 20480
        assert args["local_device"] == "cuda:0"
        assert args["local_rollout_gpus"] == "1,2,3,4,5,6,7"
    assert "evaluation" not in compiled
