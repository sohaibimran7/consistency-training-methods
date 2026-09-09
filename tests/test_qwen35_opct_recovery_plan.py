"""Static contract for the fresh Qwen3.5 OPCT-only recovery plan."""

from __future__ import annotations

from pathlib import Path

from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
PLAN = ROOT / "experiments" / "rmct_paper_vast_dense_models" / "stage1" / "qwen3_5_9b_opct_recovery_20260803.yaml"


def test_opct_recovery_has_a_fresh_namespace_and_no_invalid_rmct_branch():
    source = experiment.load_experiment_source(PLAN)
    spec = source["spec"]

    assert source["name"] == "rmct_paper_vast_dense_qwen3_5_9b_opct_recovery_20260803"
    assert "20260801" not in source["name"]
    assert spec["artifact_root"] == "artifacts/rmct-hle-qwen3.5-9b-dense-opct-recovery-20260803"
    assert spec["figure_root"] == "figures/rmct-hle-qwen3.5-9b-dense-opct-recovery-20260803"
    assert spec["training_only"] is True
    assert spec["data"]["prepare_shared"] is False
    assert spec["data"]["shared_root"] == "artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801"
    assert spec["data"]["training"]["prompt_style"] == "none"
    assert spec["conditions"] == [
        {"name": "untrained", "method": "none"},
        {"name": "opct", "method": "opct"},
    ]
    assert [item["target"] for item in spec["execution"]["allocations"]] == ["opct"]
    assert spec["rate_matching"]["batch_size"] == 4
    assert spec["rate_matching"]["gradient_accumulation_steps"] == 1
    assert spec["rate_matching"]["normalization"] == "pooled"


def test_compiled_opct_recovery_emits_one_eight_gpu_onpolicy_command_with_exact_frozen_pair_input():
    compiled = experiment.load_experiment(PLAN)
    training = {entry["name"]: entry for entry in compiled["training"]}

    assert set(training) == {"opct_lr1"}
    entry = training["opct_lr1"]
    args = entry["args"]
    assert entry["target"] == "opct"
    assert entry["gpu_count"] == 8
    assert entry["command"] == ["${python}", "scripts/train_opct.py"]
    assert args["experiment_name"] == "${experiment}"
    assert args["run_name"] == "opct-lr-1e-4"
    assert args["data"] == [
        "artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl:2048"
    ]
    assert args["data_manifest"] == [
        "artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json"
    ]
    assert args["reference_messages_field"] == "unbiased_messages"
    assert args["variant_messages_field"] == "biased_messages"
    assert args["local_device"] == "cuda:0"
    assert args["local_rollout_gpus"] == "1,2,3,4,5,6,7"
    assert args["local_rollout_gpu_mem_util"] == 0.75
    assert args["local_vllm_max_model_len"] == 32768
    assert args["local_vllm_max_num_seqs"] == 256
    assert args["local_vllm_max_num_batched_tokens"] == 8192
    assert args["local_target_logprob_chunk_size"] == 2048
    assert args["local_forward_microbatch_max_tokens"] == 49152
    assert args["lr"] == 0.0001
    assert args["max_new_tokens"] == 20480
