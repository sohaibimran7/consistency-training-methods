"""Offline contract tests for the two dense-Qwen Vast experiment plans."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from inspect_ai.model import GenerateConfig

from ctm_data.adapters.mcq_bias.experiment_factory import compile_experiment
from scripts import run_experiment as experiment

ROOT = Path(__file__).parent.parent
EXPERIMENT_ROOT = ROOT / "experiments" / "rmct_paper_vast_dense_models"
FULL_PLANS = {
    "Qwen/Qwen3.5-9B": EXPERIMENT_ROOT / "qwen3_5_9b.yaml",
    "Qwen/Qwen3-8B": EXPERIMENT_ROOT / "qwen3_8b.yaml",
}
SMOKE_PLANS = {
    "Qwen/Qwen3.5-9B": EXPERIMENT_ROOT / "debug" / "qwen3_5_9b_smoke.yaml",
    "Qwen/Qwen3-8B": EXPERIMENT_ROOT / "debug" / "qwen3_8b_smoke.yaml",
}
SHARED_PLAN = EXPERIMENT_ROOT / "shared_data.yaml"
SHARED_SMOKE_PLAN = EXPERIMENT_ROOT / "debug" / "shared_data_smoke.yaml"
EXPEDITED_12H_PLAN = EXPERIMENT_ROOT / "expedited" / "qwen3_5_9b_12h_screen.yaml"
STAGE1_QWEN35_PLAN = EXPERIMENT_ROOT / "stage1" / "qwen3_5_9b.yaml"
EXISTING_PLAN = ROOT / "experiments" / "rmct_paper_vast_more_methods" / "experiment.yaml"

CONDITIONS = [
    {"name": "untrained", "method": "none"},
    {"name": "rate-matching", "method": "rate_matching"},
    {"name": "rate-matching-control", "method": "rate_matching", "control": True},
    {"name": "bias-augmented-consistency", "method": "bias_augmented_consistency"},
    {"name": "bias-augmented-consistency-control", "method": "bias_augmented_consistency", "control": True},
    {"name": "act", "method": "act"},
    {"name": "attct", "method": "attct"},
    {"name": "mlpct", "method": "mlpct"},
    {"name": "opct", "method": "opct"},
]
COMMON_LRS = [0.0001]
DECODING = {
    "Qwen/Qwen3.5-9B": {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "extra_body": {"top_k": 20},
    },
    "Qwen/Qwen3-8B": {"max_tokens": 20480, "temperature": 0.6, "top_p": 0.95, "top_k": 20},
}


def _source(path: Path) -> dict:
    return experiment.load_experiment_source(path)


def _compiled(path: Path) -> dict:
    return experiment.load_experiment(path)


def _training_rate(entry: dict) -> float:
    if entry["command"][-1] == "scripts/train_bct.py":
        return entry["args"]["optimizer_config"]["learning_rate"]
    return entry["args"]["lr"]


@pytest.mark.parametrize(("model", "path"), FULL_PLANS.items())
def test_full_dense_qwen_plan_has_exact_scientific_matrix(model: str, path: Path):
    source = _source(path)
    spec = source["spec"]
    config = _compiled(path)

    assert source["experiment_factory"] == "ctm_data.adapters.mcq_bias.experiment_factory:compile_experiment"
    assert spec["model"] == model
    assert spec["backend"] == "local"
    assert spec["conditions"] == CONDITIONS
    assert [item["value"] for item in spec["learning_rates"]] == COMMON_LRS
    assert [item["value"] for item in spec["opct"]["learning_rates"]] == COMMON_LRS
    assert spec["data"]["prepare_shared"] is False
    assert spec["data"]["training"]["examples"] == 2048
    assert spec["data"]["instruction"]["examples"] == 2048
    assert spec["data"]["evaluation"]["questions"] == 100
    assert len(spec["data"]["evaluation"]["biases"]) == 6
    assert spec["lora"]["rank"] == 8
    assert spec["lora"]["alpha"] == 16
    assert spec["local"].get("vllm_language_model_only", False) is (model == "Qwen/Qwen3.5-9B")
    assert spec["local"]["vllm_max_num_seqs"] == 256
    assert spec["local"]["gpu_memory_utilization"] == (0.34 if model == "Qwen/Qwen3.5-9B" else 0.45)
    assert spec["local"].get("vllm_max_model_len") == (32768 if model == "Qwen/Qwen3.5-9B" else None)
    assert spec["supervised_consistency"]["method_config"]["mlpct"]["variant"] == "hidden"

    assert "data_generation" not in config
    assert len(config["data_preparation"]) == 3
    assert len(config["training"]) == 8
    assert len(config["evaluation"]) == 9

    training = {entry["name"]: entry for entry in config["training"]}
    prefixes = (
        "rate_matching",
        "rate_matching_control",
        "bias_augmented_consistency",
        "bias_augmented_consistency_control",
        "act",
        "attct",
        "mlpct",
    )
    for prefix in prefixes:
        entries = [training[f"{prefix}_lr1"]]
        assert [_training_rate(entry) for entry in entries] == COMMON_LRS
    opct_entries = [training["opct_lr1"]]
    assert [_training_rate(entry) for entry in opct_entries] == COMMON_LRS
    assert not any(name.startswith("opct_control") for name in training)

    for entry in config["training"]:
        assert entry["args"]["backend"] == "local"
        assert entry["args"]["model"] == model
        assert entry["args"]["lora_config"]["rank"] == 8
        assert entry["args"]["lora_config"]["alpha"] == 16
        assert entry["args"].get("local_vllm_language_model_only", False) is (model == "Qwen/Qwen3.5-9B")
        assert entry["args"]["local_vllm_max_num_seqs"] == 256
        assert entry["args"].get("local_vllm_max_model_len") == (32768 if model == "Qwen/Qwen3.5-9B" else None)

    if model == "Qwen/Qwen3.5-9B":
        assert training["rate_matching_lr1"]["args"]["batch_size"] == 1
        assert training["rate_matching_lr1"]["args"]["gradient_accumulation_steps"] == 4

    for entry in opct_entries:
        assert entry["command"] == ["${python}", "scripts/train_opct.py"]
        assert entry["args"]["reference_messages_field"] == "unbiased_messages"
        assert entry["args"]["variant_messages_field"] == "biased_messages"
        assert entry["args"]["kl_coef"] == 2.0
        assert entry["args"]["kl_discount_factor"] == 0.9
        assert entry["args"]["loss_fn"] == "importance_sampling"

    for method in ("act", "attct", "mlpct"):
        args = training[f"{method}_lr1"]["args"]
        assert "alignment_text_field" not in args
        assert args["data"][0].endswith("/canonical-consistency-pairs.jsonl:2048")
        assert args["data_manifest"][0].endswith("/canonical-consistency-pairs.manifest.json")

    expected_eval_provider = "vllm" if model == "Qwen/Qwen3.5-9B" else "hf"
    assert config["evaluation"][0]["args"]["model"] == f"{expected_eval_provider}/{model}"
    assert "local_checkpoint" not in config["evaluation"][0]["args"]
    for entry in config["evaluation"]:
        assert entry["args"]["generation_config"] == DECODING[model]
        assert entry["args"].get("max_tasks") == (1 if model == "Qwen/Qwen3.5-9B" else None)
        assert entry["args"].get("isolate_tasks", False) is (model == "Qwen/Qwen3.5-9B")
        if model == "Qwen/Qwen3.5-9B":
            expected_args = {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 32768,
                "language_model_only": True,
                "max_num_seqs": 256,
            }
            if "local_checkpoint" in entry["args"]:
                expected_args = {"provider": "vllm", **expected_args}
            assert entry["args"]["model_args"] == expected_args
    for entry in config["evaluation"][1:]:
        assert entry["args"]["base_model"] == model
        assert entry["args"]["local_checkpoint"].startswith("${training.")

    for chart_path in spec["reports"]["charts"].values():
        assert chart_path.startswith("experiments/rmct_paper_vast_more_methods/")
        assert (ROOT / chart_path).is_file()


def test_full_models_share_frozen_inputs_and_keep_outputs_model_specific():
    configs = [_compiled(path) for path in FULL_PLANS.values()]
    first_training = {entry["name"]: entry for entry in configs[0]["training"]}
    second_training = {entry["name"]: entry for entry in configs[1]["training"]}

    assert first_training["opct_lr1"]["args"]["data"] == second_training["opct_lr1"]["args"]["data"]
    assert first_training["act_lr1"]["args"]["data"] != second_training["act_lr1"]["args"]["data"]
    assert first_training["opct_lr1"]["args"]["data_manifest"] == second_training["opct_lr1"]["args"]["data_manifest"]
    for first_eval, second_eval in zip(configs[0]["evaluation"], configs[1]["evaluation"], strict=True):
        assert first_eval["args"]["task_args"]["datasets"] == second_eval["args"]["task_args"]["datasets"]
        assert first_eval["args"]["task_args"]["dataset_dir"] == second_eval["args"]["task_args"]["dataset_dir"]

    first_prep = {entry["name"]: entry for entry in configs[0]["data_preparation"]}
    second_prep = {entry["name"]: entry for entry in configs[1]["data_preparation"]}
    for name in ("bias-augmented-consistency-targets", "instruction-targets"):
        assert first_prep[name]["args"]["data"] == second_prep[name]["args"]["data"]
        assert first_prep[name]["args"]["main_output"] != second_prep[name]["args"]["main_output"]
        assert first_prep[name]["args"]["manifest_output"] != second_prep[name]["args"]["manifest_output"]
    canonical_name = "canonical-consistency-pairs"
    assert first_prep[canonical_name]["args"]["source"] == second_prep[canonical_name]["args"]["source"]
    assert first_prep[canonical_name]["args"]["output"] != second_prep[canonical_name]["args"]["output"]

    assert configs[0]["name"] != configs[1]["name"]
    assert experiment.output_state_path(configs[0]) != experiment.output_state_path(configs[1])
    contexts = [experiment.initial_context(config) for config in configs]
    for config, context in zip(configs, contexts, strict=True):
        training_command = experiment.planned_commands(config, ["training"], context, strict=False)[0][2]
        assert training_command[training_command.index("--experiment-name") + 1] == config["name"]
        eval_command = experiment.planned_commands(config, ["evaluation"], context, strict=False)[0][2]
        assert config["name"] in eval_command[eval_command.index("--log-dir") + 1]


def test_production_qwen35_stage1_has_race_free_node_allocations_and_rollout_workers():
    source = _source(STAGE1_QWEN35_PLAN)
    spec = source["spec"]
    config = _compiled(STAGE1_QWEN35_PLAN)

    assert spec["model"] == "Qwen/Qwen3.5-9B"
    assert spec["conditions"] == CONDITIONS
    assert [rate["value"] for rate in spec["learning_rates"]] == COMMON_LRS
    assert [rate["value"] for rate in spec["opct"]["learning_rates"]] == COMMON_LRS
    assert spec["data"]["training"]["examples"] == 2048
    assert spec["data"]["instruction"]["examples"] == 2048
    assert spec["data"]["evaluation"]["questions"] == 100
    assert spec["data"]["evaluation"]["minimum_questions_by_bias"] == {"wrong_argument": 92}
    assert spec["local"]["gpu_memory_utilization"] == 0.34
    assert spec["local"]["rollout_gpu_memory_utilization"] == 0.75
    assert spec["local"]["vllm_max_num_batched_tokens"] == 8192
    assert spec["local"]["gradient_checkpointing"] is True
    assert spec["local"]["gradient_checkpointing_layers"] == 16
    assert spec["local"]["forward_microbatch_max_datums"] == 8
    assert spec["local"]["forward_microbatch_max_tokens"] == 20480
    assert spec["local"]["target_logprob_chunk_size"] == 2048
    assert config["training_output_publication"] == {
        "owner": "stage1-coordinator",
        "targets": [
            "rmct-main",
            "rmct-control",
            "bct-main",
            "bct-control",
            "act",
            "attct",
            "mlpct",
            "opct",
        ],
    }

    preparation = {entry["name"]: entry for entry in config["data_preparation"]}
    assert set(preparation) == {
        "canonical-consistency-pairs",
        "bias-augmented-consistency-targets",
        "instruction-targets",
    }
    canonical = preparation["canonical-consistency-pairs"]
    assert canonical["target"] == "data-preparation"
    assert canonical["resource"] == "cpu"
    assert "gpu_count" not in canonical
    assert "local_rollout_gpus" not in canonical["args"]
    target_generation = [
        preparation["bias-augmented-consistency-targets"],
        preparation["instruction-targets"],
    ]
    assert all(entry["target"] == "data-preparation" and entry["gpu_count"] == 4 for entry in target_generation)
    assert all(entry["args"]["local_rollout_gpus"] == "0,1,2,3" for entry in target_generation)
    assert all(entry["args"]["local_rollout_gpu_mem_util"] == 0.75 for entry in target_generation)
    assert all(entry["args"]["local_gradient_checkpointing_layers"] == 16 for entry in target_generation)

    training = {entry["name"]: entry for entry in config["training"]}
    assert all(
        entry["args"]["local_gradient_checkpointing_layers"] == 16
        for name, entry in training.items()
        if name
        not in {
            "rate_matching_lr1",
            "rate_matching_control_lr1",
            "bias_augmented_consistency_lr1",
            "bias_augmented_consistency_control_lr1",
            "opct_lr1",
        }
    )
    for name, target in (
        ("rate_matching_lr1", "rmct-main"),
        ("rate_matching_control_lr1", "rmct-control"),
    ):
        entry = training[name]
        assert entry["target"] == target
        assert entry["gpu_count"] == 8
        assert entry["args"]["local_device"] == "cuda:0"
        assert entry["args"]["local_rollout_gpus"] == "1,2,3,4,5,6,7"
        assert entry["args"]["local_gpu_mem_util"] == 0.34
        assert entry["args"]["local_rollout_gpu_mem_util"] == 0.75
        assert entry["args"]["local_gradient_checkpointing"] is True
        assert "local_gradient_checkpointing_layers" not in entry["args"]
        assert entry["args"]["local_forward_microbatch_max_datums"] == 8
        assert entry["args"]["local_forward_microbatch_max_tokens"] == 20480
        assert entry["args"]["local_target_logprob_chunk_size"] == 2048

    short_targets = {
        "bias_augmented_consistency_lr1": "bct-main",
        "bias_augmented_consistency_control_lr1": "bct-control",
        "act_lr1": "act",
        "attct_lr1": "attct",
        "mlpct_lr1": "mlpct",
    }
    assert {name: training[name]["target"] for name in short_targets} == short_targets
    assert all(training[name]["gpu_count"] == 1 for name in short_targets)
    assert all("local_rollout_gpus" not in training[name]["args"] for name in short_targets)
    assert "local_gradient_checkpointing_layers" not in training["bias_augmented_consistency_lr1"]["args"]
    assert "local_gradient_checkpointing_layers" not in training["bias_augmented_consistency_control_lr1"]["args"]
    assert all(training[name]["args"]["local_gradient_checkpointing_layers"] == 16 for name in ("act_lr1", "attct_lr1", "mlpct_lr1"))

    opct = training["opct_lr1"]
    assert opct["target"] == "opct"
    assert opct["gpu_count"] == 8
    assert opct["args"]["local_device"] == "cuda:0"
    assert opct["args"]["local_rollout_gpus"] == "1,2,3,4,5,6,7"
    assert opct["args"]["local_gpu_mem_util"] == 0.34
    assert opct["args"]["local_rollout_gpu_mem_util"] == 0.75
    assert opct["args"]["local_gradient_checkpointing"] is True
    assert "local_gradient_checkpointing_layers" not in opct["args"]
    assert opct["args"]["local_target_logprob_chunk_size"] == 2048
    assert len(config["training"]) == 8

    assert len(config["evaluation"]) == 9
    assert all(entry["target"] == "evaluation" and entry["gpu_count"] == 1 for entry in config["evaluation"])
    assert all(entry["args"]["persistent_vllm_server"] is True for entry in config["evaluation"])
    assert all(entry["args"]["task_args"]["n_questions"] == 100 for entry in config["evaluation"])
    assert all(entry["args"]["task_args"]["min_n_questions"] == 92 for entry in config["evaluation"])
    rmct_command = experiment.command_argv(training["rate_matching_lr1"], experiment.initial_context(config))
    assert rmct_command[rmct_command.index("--local-device") + 1] == "cuda:0"
    assert rmct_command[rmct_command.index("--local-rollout-gpus") + 1] == "1,2,3,4,5,6,7"
    assert "--local-gradient-checkpointing-layers" not in rmct_command
    assert "--gpu-count" not in rmct_command
    target_command = experiment.command_argv(
        preparation["bias-augmented-consistency-targets"],
        experiment.initial_context(config),
    )
    assert target_command[target_command.index("--local-rollout-gpus") + 1] == "0,1,2,3"
    assert target_command[target_command.index("--local-rollout-gpu-mem-util") + 1] == "0.75"
    assert target_command[target_command.index("--local-gradient-checkpointing-layers") + 1] == "16"
    assert "--gpu-count" not in target_command

    # The CPU-only canonicalization step must remain plannable alongside the
    # four-GPU target-generation commands.  This catches accidental GPU
    # reservation for a JSONL-only transform before a paid deployment.
    planned_preparation = experiment.planned_commands(
        config,
        ["data_preparation"],
        experiment.initial_context(config),
        strict=False,
        target="data-preparation",
    )
    assert {name for _stage, name, _argv in planned_preparation} == set(preparation)


def test_production_qwen35_stage1_forwards_top_k_in_native_vllm_generate_config():
    config = _compiled(STAGE1_QWEN35_PLAN)
    expected = DECODING["Qwen/Qwen3.5-9B"]

    for entry in config["evaluation"]:
        assert entry["args"]["task_factory"] == (
            "ctm_data.adapters.mcq_bias.tasks:suite_tasks"
        )
        generation_config = entry["args"]["generation_config"]
        assert generation_config == expected

        native_config = GenerateConfig(**generation_config)
        assert native_config.max_tokens == 20480
        assert native_config.temperature == 1.0
        assert native_config.top_p == 0.95
        assert native_config.top_k == 20
        assert native_config.extra_body == {"top_k": 20}

    from inspect_ai.model._providers.vllm import VLLMAPI

    api = VLLMAPI("unit/stage1-top-k-contract", base_url="http://127.0.0.1:1/v1")
    try:
        api._resolve_server()
        params = api.completion_params(GenerateConfig(**expected), tools=False)
        assert params["temperature"] == 1.0
        assert params["top_p"] == 0.95
        assert params["extra_body"] == {"top_k": 20}
    finally:
        api.close()

    hf_config = _compiled(FULL_PLANS["Qwen/Qwen3-8B"])
    assert all("extra_body" not in entry["args"]["generation_config"] for entry in hf_config["evaluation"])


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "16"])
def test_dense_model_factory_rejects_invalid_selective_checkpoint_layer_counts(value):
    spec = deepcopy(_source(STAGE1_QWEN35_PLAN)["spec"])
    spec["local"]["gradient_checkpointing_layers"] = value

    with pytest.raises(TypeError, match="local.gradient_checkpointing_layers must be a positive integer"):
        compile_experiment(name="invalid-selective-checkpointing", spec=spec)


def test_dense_model_factory_requires_checkpointing_for_selective_layer_count():
    spec = deepcopy(_source(STAGE1_QWEN35_PLAN)["spec"])
    spec["local"]["gradient_checkpointing"] = False

    with pytest.raises(ValueError, match="requires local.gradient_checkpointing=true"):
        compile_experiment(name="invalid-selective-checkpointing", spec=spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gpu_memory_utilization", 0, r"\(0, 1\]"),
        ("rollout_gpu_memory_utilization", 1.1, r"\(0, 1\]"),
        ("vllm_max_num_batched_tokens", False, "positive integer"),
        ("forward_microbatch_max_datums", 0, "positive integer"),
        ("forward_microbatch_max_tokens", False, "positive integer"),
        ("target_logprob_chunk_size", 0, "positive integer"),
    ],
)
def test_factory_rejects_invalid_local_execution_capacity(field, value, message):
    source = _source(STAGE1_QWEN35_PLAN)
    spec = deepcopy(source["spec"])
    spec["local"][field] = value

    with pytest.raises((TypeError, ValueError), match=message):
        compile_experiment(name=source["name"], spec=spec)


def test_factory_maps_vllm_max_num_batched_tokens_to_local_cli_args():
    source = _source(STAGE1_QWEN35_PLAN)
    spec = deepcopy(source["spec"])
    spec["local"]["vllm_max_num_batched_tokens"] = 8192

    compiled = compile_experiment(name=source["name"], spec=spec)

    target_generation = next(entry for entry in compiled["data_preparation"] if entry["name"] == "bias-augmented-consistency-targets")
    assert target_generation["args"]["local_vllm_max_num_batched_tokens"] == 8192
    assert all(entry["args"]["local_vllm_max_num_batched_tokens"] == 8192 for entry in compiled["training"])


def test_factory_rejects_target_bundle_that_wastes_a_gpu_on_a_nonexistent_coordinator():
    source = _source(STAGE1_QWEN35_PLAN)
    spec = deepcopy(source["spec"])
    # The CPU-only canonical-pair transform intentionally precedes the
    # GPU target-generation allocation.  Mutate the latter: it is the one
    # whose workers must cover every GPU in its coordinator-free bundle.
    allocation = next(
        item
        for item in spec["execution"]["allocations"]
        if "bias-augmented-consistency-targets" in item["commands"]
    )
    allocation["rollout_gpus"] = [1, 2, 3]

    with pytest.raises(ValueError, match="every logical GPU"):
        compile_experiment(name=source["name"], spec=spec)


@pytest.mark.parametrize("value", [0, False, "half"])
def test_factory_rejects_invalid_allocation_checkpoint_layer_override(value):
    source = _source(STAGE1_QWEN35_PLAN)
    spec = deepcopy(source["spec"])
    allocation = next(item for item in spec["execution"]["allocations"] if item["target"] == "bct-main")
    allocation["gradient_checkpointing_layers"] = value

    with pytest.raises(ValueError, match="positive integer or 'all'"):
        compile_experiment(name=source["name"], spec=spec)


def test_shared_data_plans_are_the_only_shared_writers():
    full = _compiled(SHARED_PLAN)
    smoke = _compiled(SHARED_SMOKE_PLAN)

    assert len(full["data_generation"]) == 3
    assert len(full["data_preparation"]) == 1
    full_generation = {entry["name"]: entry for entry in full["data_generation"]}
    pair_import = full_generation["distractor-argument-pairs"]
    assert pair_import["command"][-1] == "ctm_data.adapters.mcq_bias.import_legacy_pairs"
    assert pair_import["args"]["per_source_limit"] == 1500
    assert pair_import["args"]["prompt_style"] == "encourage_cot"
    assert len(pair_import["args"]["inputs"]) == 2
    assert full_generation["cleaned-alpaca-prompts"]["args"]["count"] == 2048
    assert full["data_preparation"][0]["args"]["n_questions"] == 100
    assert full["data_preparation"][0]["args"]["min_n_questions"] == 92
    assert len(full["data_preparation"][0]["args"]["bias_types"]) == 6

    assert len(smoke["data_generation"]) == 3
    assert len(smoke["data_preparation"]) == 1
    smoke_generation = {entry["name"]: entry for entry in smoke["data_generation"]}
    smoke_pair_import = smoke_generation["distractor-argument-pairs"]
    assert smoke_pair_import["command"][-1] == "ctm_data.adapters.mcq_bias.import_legacy_pairs"
    assert smoke_pair_import["args"]["per_source_limit"] == 8
    assert smoke_pair_import["args"]["prompt_style"] == "encourage_cot"
    assert smoke_generation["cleaned-alpaca-prompts"]["args"]["count"] == 16
    assert smoke["data_preparation"][0]["args"]["n_questions"] == 2


@pytest.mark.parametrize("model", FULL_PLANS)
def test_smoke_plan_has_condition_parity_and_one_rate_per_family(model: str):
    full_source = _source(FULL_PLANS[model])["spec"]
    smoke_source = _source(SMOKE_PLANS[model])["spec"]
    smoke = _compiled(SMOKE_PLANS[model])

    assert smoke_source["conditions"] == full_source["conditions"] == CONDITIONS
    assert smoke_source["lora"] == full_source["lora"]
    assert smoke_source["supervised_consistency"]["method_config"] == full_source["supervised_consistency"]["method_config"]
    assert [item["value"] for item in smoke_source["learning_rates"]] == COMMON_LRS
    assert [item["value"] for item in smoke_source["opct"]["learning_rates"]] == COMMON_LRS
    assert smoke_source["data"]["training"]["examples"] == 16
    assert smoke_source["data"]["instruction"]["examples"] == 16
    assert smoke_source["data"]["evaluation"]["questions"] == 2
    assert smoke_source["rate_matching"]["datapoints"] == 2
    assert set(smoke_source["rate_matching"]["rollouts"].values()) == {2}
    assert len(smoke["training"]) == 8
    assert len(smoke["evaluation"]) == 9
    assert all(entry["args"]["generation_config"] == DECODING[model] for entry in smoke["evaluation"])


def test_smoke_models_share_inputs_but_not_model_targets():
    configs = [_compiled(path) for path in SMOKE_PLANS.values()]
    first_opct = next(entry for entry in configs[0]["training"] if entry["name"] == "opct_lr1")
    second_opct = next(entry for entry in configs[1]["training"] if entry["name"] == "opct_lr1")
    assert first_opct["args"]["data"] == second_opct["args"]["data"]
    first_prep = {entry["name"]: entry for entry in configs[0]["data_preparation"]}
    second_prep = {entry["name"]: entry for entry in configs[1]["data_preparation"]}
    first_target = first_prep["bias-augmented-consistency-targets"]["args"]["main_output"]
    second_target = second_prep["bias-augmented-consistency-targets"]["args"]["main_output"]
    assert first_target != second_target


def test_expedited_12h_plan_is_separate_and_budget_limited():
    source = _source(EXPEDITED_12H_PLAN)
    spec = source["spec"]
    compiled = _compiled(EXPEDITED_12H_PLAN)

    assert "12h_screen" in source["name"]
    assert "expedited-12h" in spec["artifact_root"]
    assert spec["data"]["shared_root"].endswith("shared-expedited-12h")
    assert spec["data"]["training"]["examples"] == 512
    assert spec["data"]["instruction"]["examples"] == 512
    assert spec["opct"]["examples"] == 256
    assert spec["rate_matching"]["datapoints"] == 32
    assert set(spec["rate_matching"]["rollouts"].values()) == {32}
    assert spec["rate_matching"]["max_new_tokens"] == 4096
    assert spec["supervised_consistency"]["bias_augmented_targets"]["max_tokens"] == 4096
    assert spec["evaluation"]["max_tokens"] == 4096
    assert spec["data"]["evaluation"]["questions"] == 100
    assert len(spec["data"]["evaluation"]["biases"]) == 6
    assert len(compiled["training"]) == 8
    assert len(compiled["evaluation"]) == 9

    training = {entry["name"]: entry for entry in compiled["training"]}
    assert training["opct_lr1"]["args"]["data"][0].endswith(":256")
    assert training["act_lr1"]["args"]["data"][0].endswith(":512")


def test_opct_config_is_required_and_validated_and_control_is_only_authored():
    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    del spec["opct"]
    with pytest.raises(ValueError, match="requires an opct configuration"):
        compile_experiment(name="missing-opct", spec=spec)

    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    spec["opct"]["learning_rates"][0]["value"] = 0
    with pytest.raises(ValueError, match="finite positive"):
        compile_experiment(name="invalid-opct", spec=spec)

    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    spec["conditions"].append({"name": "opct-control", "method": "opct", "control": True})
    compiled = compile_experiment(name="authored-opct-control", spec=spec)
    control = next(entry for entry in compiled["training"] if entry["name"] == "opct_control_lr1")
    assert control["args"]["reference_messages_field"] == "unbiased_messages"
    assert control["args"]["variant_messages_field"] == "unbiased_messages"

    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    spec["opct"]["examples"] = 256
    compiled = compile_experiment(name="opct-specific-example-budget", spec=spec)
    opct_entry = next(entry for entry in compiled["training"] if entry["name"] == "opct_lr1")
    assert opct_entry["args"]["data"][0].endswith(":256")
    act_entry = next(entry for entry in compiled["training"] if entry["name"] == "act_lr1")
    assert act_entry["args"]["data"][0].endswith(":2048")

    spec["opct"]["examples"] = 0
    with pytest.raises(ValueError, match="opct.examples must be a positive integer"):
        compile_experiment(name="invalid-opct-example-budget", spec=spec)

    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    spec["evaluation"]["isolate_tasks"] = "yes"
    with pytest.raises(TypeError, match="evaluation.isolate_tasks must be a boolean"):
        compile_experiment(name="invalid-eval-isolation", spec=spec)

    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    spec["evaluation"]["persistent_vllm_server"] = True
    with pytest.raises(ValueError, match="requires evaluation.isolate_tasks"):
        compile_experiment(name="persistent-eval-needs-isolation", spec=spec)


def test_dense_plan_rejects_tinker_and_existing_five_method_plan_still_loads():
    spec = deepcopy(_source(FULL_PLANS["Qwen/Qwen3-8B"])["spec"])
    spec["backend"] = "tinker"
    with pytest.raises(ValueError, match="requires backend: local"):
        compile_experiment(name="no-tinker-internal-methods", spec=spec)

    existing = _compiled(EXISTING_PLAN)
    assert len(existing["training"]) == 21
    assert len(existing["evaluation"]) == 22
    assert not any(entry["command"][-1] == "scripts/train_opct.py" for entry in existing["training"])
