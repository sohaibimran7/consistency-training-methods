"""Static contract for the isolated Qwen3.5 no-CoT supervised recovery."""

from __future__ import annotations

from pathlib import Path

from scripts import run_experiment as experiment

ROOT = Path(__file__).parent.parent
PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_supervised_recovery_none_20260801.yaml"
)
MODEL = "Qwen/Qwen3.5-9B"
RECOVERED_ROOT = "artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801"
ARTIFACT_ROOT = "artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-calibration-20260803"
STRICT_QV_LORA = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "train_mlp": False,
    "train_attn": False,
    "train_unembed": False,
    "seed": 42,
}
UPSTREAM_ADAMW_WITH_SHARED_LR = {
    "learning_rate": 0.0001,
    "lr_schedule": "constant",
    "beta1": 0.9,
    "beta2": 0.999,
    "eps": 0.00000001,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


def _by_name(entries: list[dict]) -> dict[str, dict]:
    return {entry["name"]: entry for entry in entries}


def test_supervised_recovery_is_a_direct_fresh_none_plan_with_no_evaluation_stage():
    source = experiment.load_experiment_source(PLAN)
    config = experiment.load_experiment(PLAN)

    assert "experiment_factory" not in source
    assert config["name"] == "rmct_paper_vast_dense_qwen3_5_9b_stage1_supervised_recovery_none_calibration_20260803"
    assert "evaluation" not in config
    assert "analysis" not in config
    assert "rendering" not in config
    assert config["variables"]["model"] == MODEL
    assert config["variables"]["artifact_root"] == ARTIFACT_ROOT
    assert config["variables"]["recovered_none_pairs"] == f"{RECOVERED_ROOT}/data/distractor-argument-pairs.jsonl"
    assert (
        config["variables"]["recovered_none_manifest"]
        == f"{RECOVERED_ROOT}/data/distractor-argument-pairs.manifest.json"
    )
    assert "stage1-recovery-20260801" not in str(config)
    assert "rmct" not in {entry["target"] for entry in config["training"]}
    assert "opct" not in {entry["target"] for entry in config["training"]}


def test_cpu_prep_rebuilds_every_prompt_dependent_input_from_the_none_source():
    config = experiment.load_experiment(PLAN)
    entries = _by_name(config["data_preparation"])

    common = entries["prepare_none_supervised_inputs"]
    assert common["target"] == "inputs"
    assert common["resource"] == "cpu"
    assert common["command"][-1] == "prepare"
    assert common["args"]["source"] == "${recovered_none_pairs}"
    assert common["args"]["source_manifest"] == "${recovered_none_manifest}"
    assert common["args"]["output_dir"] == "${artifact_root}/data"
    assert common["args"]["instruction_main"].endswith("/instruction-targets.jsonl")
    assert common["args"]["instruction_control"].endswith("/instruction-targets-control.jsonl")

    bct_source = entries["verify_bct_target_source"]
    assert bct_source["target"] == "bct-targets"
    assert bct_source["resource"] == "cpu"
    assert bct_source["command"][-1] == "prepare"
    assert bct_source["args"] == common["args"]

    targets = entries["generate_none_bct_targets"]
    assert targets["target"] == "bct-targets"
    assert targets["resource"] == "gpu"
    assert targets["gpu_count"] == 4
    assert targets["args"]["data"] == ["${recovered_none_pairs}"]
    assert targets["args"]["limit"] == 2048
    assert targets["args"]["main_messages_field"] == "biased_messages"
    assert targets["args"]["control_messages_field"] == "unbiased_messages"
    assert targets["args"]["main_output"] == "${artifact_root}/data/bias-augmented-consistency-none.jsonl"
    assert targets["args"]["control_output"] == "${artifact_root}/data/bias-augmented-consistency-control-none.jsonl"
    assert targets["args"]["manifest_output"] == "${artifact_root}/data/bias-augmented-consistency-none.manifest.json"
    assert targets["args"]["max_tokens"] == 20480
    assert targets["args"]["temperature"] == 1.0
    assert targets["args"]["max_concurrency"] == 96
    assert targets["args"]["local_rollout_gpus"] == "0,1,2,3"

    for name, target in (
        ("verify_repaired_act_inputs", "repaired-act"),
        ("verify_attct_inputs", "attct"),
        ("verify_mlpct_inputs", "mlpct"),
        ("verify_bct_main_inputs", "bct-main"),
        ("verify_bct_control_inputs", "bct-control"),
    ):
        entry = entries[name]
        assert entry["target"] == target
        assert entry["resource"] == "cpu"
        assert entry["command"][-1] == "prepare"
        assert entry["args"] == common["args"]

    for name, target in (("verify_bct_main_targets", "bct-main"), ("verify_bct_control_targets", "bct-control")):
        entry = entries[name]
        assert entry["target"] == target
        assert entry["resource"] == "cpu"
        assert entry["command"][-1] == "verify-bct-targets"
        assert entry["args"]["source"] == "${recovered_none_pairs}"
        assert entry["args"]["source_manifest"] == "${recovered_none_manifest}"
        assert entry["args"]["main"] == "${artifact_root}/data/bias-augmented-consistency-none.jsonl"
        assert entry["args"]["control"] == "${artifact_root}/data/bias-augmented-consistency-control-none.jsonl"


def test_five_supervised_targets_use_fresh_none_artifacts_and_publish_atomically():
    config = experiment.load_experiment(PLAN)
    training = _by_name(config["training"])

    assert config["training_output_publication"] == {
        "owner": "supervised-recovery-none-coordinator",
        "targets": ["repaired-act", "attct", "mlpct", "bct-main", "bct-control"],
    }
    assert set(training) == {"repaired_act_none", "attct_none", "mlpct_none", "bct_none", "bct_control_none"}
    assert [(entry["name"], entry["target"], entry["gpu_count"]) for entry in config["training"]] == [
        ("repaired_act_none", "repaired-act", 1),
        ("attct_none", "attct", 1),
        ("mlpct_none", "mlpct", 1),
        ("bct_none", "bct-main", 1),
        ("bct_control_none", "bct-control", 1),
    ]

    repaired = training["repaired_act_none"]["args"]
    assert repaired["model"] == "${model}"
    assert repaired["method"] == "act"
    assert repaired["data"] == ["${artifact_root}/data/canonical-repaired-act-train-n200.jsonl:200"]
    assert repaired["data_manifest"] == ["${artifact_root}/data/canonical-repaired-act-train-n200.manifest.json"]
    assert "alignment_text_field" not in repaired
    assert repaired["require_full_reference_suffix_alignment"] is True
    assert repaired["qwen35_consistency_preflight"] is True
    assert repaired["minimum_optimizer_steps"] == 4000
    assert repaired["epochs"] == 20
    assert repaired["gradient_accumulation_steps"] == 1
    assert repaired["lora_config"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    ]

    for name, method in (("attct_none", "attct"), ("mlpct_none", "mlpct")):
        args = training[name]["args"]
        assert args["method"] == method
        assert args["data"] == ["${artifact_root}/data/canonical-consistency-pairs-n2048.jsonl:2048"]
        assert args["data_manifest"] == ["${artifact_root}/data/canonical-consistency-pairs-n2048.manifest.json"]
        assert "alignment_text_field" not in args
        assert args["require_full_reference_suffix_alignment"] is True
        assert args["qwen35_consistency_preflight"] is True
        # The upstream default is Q/V-only LoRA. Qwen3.5's linear-attention
        # Q/K/V projection is fused, so keep the primary protocol strictly on
        # its conventional attention blocks rather than silently broadening
        # the method to MLP or fused-QKV adaptation.
        assert args["lora_config"] == STRICT_QV_LORA
        assert args["optimizer_config"] == UPSTREAM_ADAMW_WITH_SHARED_LR

    # The paper claims both 4,000 AttCT optimizer steps and accumulation 8,
    # but the released Qwen AttCT script actually invokes 4,000 steps against
    # a config with no accumulation.  Preserve that executable protocol.
    attct = training["attct_none"]["args"]
    assert attct["gradient_accumulation_steps"] == 1
    assert attct["minimum_optimizer_steps"] == 4096
    assert attct["epochs"] == 2
    assert attct["run_name"] == "attct-none-qv-4096steps-lr-1e-4"

    # Qwen MLPCT, in contrast, explicitly configures accumulation 8.  Two
    # 2,048-pair epochs reproduce the released one-epoch 4K-presentation
    # budget (512 local updates versus the released 500).
    mlpct = training["mlpct_none"]["args"]
    assert mlpct["gradient_accumulation_steps"] == 8
    assert mlpct["minimum_optimizer_steps"] == 512
    assert mlpct["epochs"] == 2
    assert mlpct["run_name"] == "mlpct-none-qv-512steps-lr-1e-4"

    main = training["bct_none"]["args"]
    control = training["bct_control_none"]["args"]
    assert main["method"] == control["method"] == "bct"
    assert main["data"][0] == "${artifact_root}/data/bias-augmented-consistency-none.jsonl:2048"
    assert control["data"][0] == "${artifact_root}/data/bias-augmented-consistency-control-none.jsonl:2048"
    assert main["data"][1].endswith("/instruction-targets.jsonl:2048")
    assert control["data"][1].endswith("/instruction-targets-control.jsonl:2048")
    assert (
        main["data_manifest"][0]
        == control["data_manifest"][0]
        == "${artifact_root}/data/bias-augmented-consistency-none.manifest.json"
    )
    assert main["data_manifest"][1] == control["data_manifest"][1]
    assert main["interleave"] is control["interleave"] is True
    assert main["gradient_accumulation_steps"] == control["gradient_accumulation_steps"] == 128
    assert "local_gradient_checkpointing_layers" not in main
    assert "local_gradient_checkpointing_layers" not in control
    assert main["epochs"] == control["epochs"] == 1
    # This is a causal control: after selecting the two complementary prompt
    # views, no optimization or adapter setting may differ between BCT main
    # and BCT-control.
    main_shared = {key: value for key, value in main.items() if key not in {"data", "run_name"}}
    control_shared = {key: value for key, value in control.items() if key not in {"data", "run_name"}}
    assert main_shared == control_shared

    for entry in config["training"]:
        args = entry["args"]
        assert entry["command"] == ["${python}", "scripts/train_bct.py"]
        assert args["backend"] == "local"
        assert args["local_sampler"] == "hf"
        assert args["model"] == "${model}"
        assert args["wandb_project"] == "rmct_paper_vast_dense_models_stage1_supervised_recovery_none_20260801"
        assert args["yes"] is True


def test_target_dry_runs_are_resolvable_without_an_implicit_checkpoint_owner():
    config = experiment.load_experiment(PLAN)
    context = experiment.initial_context(config)

    for target, expected_training in (
        ("repaired-act", "repaired_act_none"),
        ("attct", "attct_none"),
        ("mlpct", "mlpct_none"),
        ("bct-main", "bct_none"),
        ("bct-control", "bct_control_none"),
    ):
        preview = experiment.planned_commands(
            config, ["data_preparation", "training"], context, strict=True, target=target
        )
        assert preview[-1][0:2] == ("training", expected_training)
        assert all("${checkpoint}" not in " ".join(argv) for _, _, argv in preview)

    publication_owner, publication_targets = experiment._publication_manifest(config)
    assert publication_owner == "supervised-recovery-none-coordinator"
    assert publication_targets == {
        "repaired-act": ["repaired_act_none"],
        "attct": ["attct_none"],
        "mlpct": ["mlpct_none"],
        "bct-main": ["bct_none"],
        "bct-control": ["bct_control_none"],
    }
