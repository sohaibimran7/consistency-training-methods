"""Tests for the deliberately small YAML experiment runner."""

import sys
import threading
from pathlib import Path

import pytest

from scripts import run_experiment as experiment

EXAMPLE = Path(__file__).parent.parent / "experiments" / "example_rlct.yaml"
F6_PER_ITEM = Path(__file__).parent.parent / "experiments" / "eval_awareness" / "qwen_f6_snr_per_item.yaml"
F6_DEBUG = (
    Path(__file__).parent.parent / "experiments" / "eval_awareness" / "debug" / "qwen_f6_snr_per_item_two_items.yaml"
)
METHOD_COMPARISON = Path(__file__).parent.parent / "experiments" / "internal_consistency" / "method_comparison.yaml"
WRONG_ARGUMENT_COMPARISON = (
    Path(__file__).parent.parent / "experiments" / "mcq_bias" / "wrong_argument_cross_bias" / "experiment.yaml"
)
BCT_BACKEND_COMPARISON = WRONG_ARGUMENT_COMPARISON.with_name("bct_backends.yaml")
RMCT_HLE_COMPARISON = (
    Path(__file__).parent.parent / "experiments" / "paper_reproductions" / "rmct_hle_gpt_oss_20b" / "experiment.yaml"
)


def test_example_keeps_training_and_evaluation_independent():
    config = experiment.load_experiment(EXAMPLE)
    context = {
        "python": sys.executable,
        "project_root": EXAMPLE.parent.parent,
        "experiment": config["name"],
        "checkpoint": "tinker://unit/checkpoint",
        "training_data": "/datasets/unit-train.jsonl",
    }
    planned = experiment.planned_commands(config, ["training", "evaluation"], context, strict=True)
    training = planned[0][2]
    evaluation = planned[1][2]

    assert "scripts/train_rlct.py" in training
    assert "/datasets/unit-train.jsonl" in " ".join(training)
    assert "scripts/run_evals.py" in evaluation
    assert "mcq_bias.tasks:suite_tasks" in evaluation
    assert "truthfulqa" in " ".join(evaluation)
    assert "/datasets/unit-train.jsonl" not in " ".join(evaluation)


def test_command_arguments_support_flags_lists_and_nested_json():
    argv = experiment.command_argv(
        {
            "command": ["python", "tool.py"],
            "args": {
                "dry_run": True,
                "disabled": False,
                "datasets": ["a", "b"],
                "task_args": {"limit": 3},
            },
        },
        {},
    )
    assert argv == [
        "python",
        "tool.py",
        "--dry-run",
        "--datasets",
        "a",
        "b",
        "--task-args",
        '{"limit":3}',
    ]


def test_unresolved_checkpoint_is_allowed_in_preview_but_not_execution():
    spec = {"command": ["python", "eval.py"], "args": {"checkpoint": "${checkpoint}"}}
    assert experiment.command_argv(spec, {"checkpoint": None}, strict=False)[-1] == "${checkpoint}"
    with pytest.raises(experiment.ExperimentConfigError, match="unresolved"):
        experiment.command_argv(spec, {"checkpoint": None}, strict=True)


def test_multiple_training_commands_cannot_implicitly_own_checkpoint():
    config = {
        "name": "ambiguous",
        "training": [
            {"name": "first", "command": ["python", "train.py", "first"]},
            {"name": "second", "command": ["python", "train.py", "second"]},
        ],
        "evaluation": {
            "command": ["python", "eval.py"],
            "args": {"checkpoint": "${checkpoint}"},
        },
    }
    with pytest.raises(experiment.ExperimentConfigError, match=r"ambiguous \$\{checkpoint\} ownership"):
        experiment.planned_commands(
            config,
            ["training", "evaluation"],
            {"checkpoint": "tinker://explicit-but-overwritten"},
            strict=False,
        )


def test_multiple_training_commands_without_checkpoint_placeholder_are_allowed():
    config = {
        "name": "independent",
        "training": [
            {"command": ["python", "train.py", "first"]},
            {"command": ["python", "train.py", "second"]},
        ],
        "evaluation": {"command": ["python", "eval.py"], "args": {"model": "base-model"}},
    }
    planned = experiment.planned_commands(config, ["training", "evaluation"], {}, strict=True)
    assert len(planned) == 3


def test_target_selects_commands_without_becoming_a_child_argument():
    config = {
        "name": "platforms",
        "training": [
            {"name": "bct_tinker", "target": "tinker", "command": ["python", "train.py", "tinker"]},
            {"name": "bct_vast", "target": "vast", "command": ["python", "train.py", "vast"]},
        ],
        "evaluation": [
            {
                "name": "eval_vast",
                "target": "vast",
                "command": ["python", "eval.py", "${training.bct_vast.checkpoint}"],
            }
        ],
    }
    commands = experiment.planned_commands(
        config,
        ["training", "evaluation"],
        {"training.bct_vast.checkpoint": "file:///vast"},
        strict=True,
        target="vast",
    )

    assert commands == [
        ("training", "bct_vast", ["python", "train.py", "vast"]),
        ("evaluation", "eval_vast", ["python", "eval.py", "file:///vast"]),
    ]


def test_named_training_checkpoints_route_each_evaluation(monkeypatch, tmp_path):
    config_path = tmp_path / "named.yaml"
    config_path.write_text("""
name: named
training:
  - name: act
    command: [python, train.py, act]
  - name: attct
    command: [python, train.py, attct]
evaluation:
  - name: eval-act
    command: [python, eval.py, "${training.act.checkpoint}"]
  - name: eval-attct
    command: [python, eval.py, "${training.attct.checkpoint}"]
""".strip())
    calls = []
    monkeypatch.setattr(experiment, "output_state_path", lambda _: tmp_path / "outputs.json")

    def fake_run(command):
        calls.append(command)
        if command[-1] in {"act", "attct"}:
            return f"file:///checkpoints/{command[-1]}"
        return None

    monkeypatch.setattr(experiment, "run_command", fake_run)
    experiment.main([str(config_path), "--yes"])
    assert calls[-2][-1] == "file:///checkpoints/act"
    assert calls[-1][-1] == "file:///checkpoints/attct"
    state = experiment.load_output_context({"name": "named"})
    assert state["training.act.checkpoint"] == "file:///checkpoints/act"
    assert state["training.attct.checkpoint"] == "file:///checkpoints/attct"


def test_method_comparison_yaml_selects_all_supervised_family_methods():
    config = experiment.load_experiment(METHOD_COMPARISON)
    context = experiment.initial_context(config)
    planned = experiment.planned_commands(config, ["training"], context, strict=True)
    commands = {name: command for _, name, command in planned}

    assert set(commands) == {"bct", "act", "attct", "mlpct"}
    for method, command in commands.items():
        assert command[command.index("--method") + 1] == method
        assert command[command.index("--backend") + 1] == "local"
    assert "--method-config" not in commands["bct"]
    assert '"layer_selection":"all"' in " ".join(commands["act"])
    assert '"layer_weights":"uniform"' in " ".join(commands["attct"])
    assert '"distance_metric":"cosine"' in " ".join(commands["mlpct"])


def test_wrong_argument_figure_yaml_routes_the_complete_pipeline():
    config = experiment.load_experiment(WRONG_ARGUMENT_COMPARISON)
    context = experiment.initial_context(config)
    for name in ("rlct", "rlct_control", "act", "act_control", "bct", "bct_control"):
        context[f"training.{name}.checkpoint"] = f"file:///checkpoints/{name}"
    stages = ["data_generation", "data_preparation", "training", "evaluation", "analysis"]
    planned = experiment.planned_commands(config, stages, context, strict=True)
    by_stage = {stage: [] for stage in stages}
    for stage, name, command in planned:
        by_stage[stage].append((name, command))

    assert [len(by_stage[stage]) for stage in stages] == [1, 1, 6, 7, 2]
    assert "ctm_data.adapters.mcq_bias.materialize" in by_stage["data_generation"][0][1]
    target_command = by_stage["data_preparation"][0][1]
    assert "scripts/prepare_bct_targets.py" in target_command
    assert target_command[target_command.index("--source-messages-field") + 1] == "unbiased_messages"
    training = dict(by_stage["training"])
    assert '"control":true' in " ".join(training["rlct_control"])
    assert training["act"][training["act"].index("--variant-messages-field") + 1] == "biased_messages"
    assert training["act_control"][training["act_control"].index("--variant-messages-field") + 1] == "unbiased_messages"
    assert training["bct"][training["bct"].index("--data-manifest") + 1].endswith("bct-targets.manifest.json")
    evaluations = dict(by_stage["evaluation"])
    assert evaluations["base"][evaluations["base"].index("--model") + 1].startswith("hf/")
    assert evaluations["bct"][evaluations["bct"].index("--local-checkpoint") + 1] == "file:///checkpoints/bct"
    assert "ctm_data.adapters.mcq_bias.analysis" in by_stage["analysis"][0][1]
    assert by_stage["analysis"][1][1][:2] == ["node", "scripts/render_flint.mjs"]


def test_rmct_hle_yaml_routes_five_methods_controls_and_verbalisation_locally():
    config = experiment.load_experiment(RMCT_HLE_COMPARISON)
    context = experiment.initial_context(config)
    for entry in config["training"]:
        context[f"training.{entry['name']}.checkpoint"] = f"file:///checkpoints/{entry['name']}"
    stages = ["data_generation", "data_preparation", "training", "evaluation", "analysis"]
    planned = experiment.planned_commands(config, stages, context, strict=True)
    by_stage = {stage: [] for stage in stages}
    for stage, name, command in planned:
        by_stage[stage].append((name, command))

    assert [len(by_stage[stage]) for stage in stages] == [3, 3, 30, 31, 8]
    assert "ctm_data.sources.cleaned_alpaca" in by_stage["data_generation"][2][1]
    materialize_eval = dict(by_stage["data_preparation"])["evaluation-suite"]
    assert "ctm_data.adapters.mcq_bias.materialize_eval" in materialize_eval
    base_eval = by_stage["evaluation"][0][1]
    assert base_eval[base_eval.index("--model") + 1] == "hf/openai/gpt-oss-20b"
    assert all("--tinker-checkpoint" not in command for _, command in by_stage["evaluation"])
    assert all("--local-checkpoint" in command for _, command in by_stage["evaluation"][1:])
    assert all(
        '"generate_missing_arguments":false' in " ".join(command)
        for _, command in by_stage["evaluation"]
    )
    training = dict(by_stage["training"])
    for method in ("rate_matching_lr1", "bias_augmented_consistency_lr1", "act_lr1", "attct_lr1", "mlpct_lr1"):
        assert training[method][training[method].index("--backend") + 1] == "local"
    assert training["bias_augmented_consistency_lr1"][training["bias_augmented_consistency_lr1"].index("--method") + 1] == "bct"
    assert training["act_lr1"][training["act_lr1"].index("--method") + 1] == "act"
    assert training["attct_lr1"][training["attct_lr1"].index("--method") + 1] == "attct"
    assert training["mlpct_lr1"][training["mlpct_lr1"].index("--method") + 1] == "mlpct"
    for method in ("act", "attct", "mlpct"):
        control = training[f"{method}_control_lr1"]
        assert control[control.index("--variant-messages-field") + 1] == "unbiased_messages"
    analysis_commands = dict(by_stage["analysis"])
    unconditional_verbalisation = analysis_commands["aggregate-bias-verbalised"]
    assert unconditional_verbalisation[unconditional_verbalisation.index("--metric") + 1] == "bias_acknowledged"
    assert "--where-metric" not in unconditional_verbalisation
    towards_verbalisation = analysis_commands["aggregate-bias-verbalised-given-towards-bias-switch"]
    assert towards_verbalisation[towards_verbalisation.index("--metric") + 1] == "bias_acknowledged"
    assert towards_verbalisation[towards_verbalisation.index("--where-metric") + 1] == "towards_bias_switch"
    total_verbalisation = analysis_commands["aggregate-bias-verbalised-given-total-bias-switch"]
    assert total_verbalisation[total_verbalisation.index("--metric") + 1] == "bias_acknowledged"
    assert total_verbalisation[total_verbalisation.index("--where-metric") + 1] == "abs_switch"


def test_rmct_hle_yaml_is_a_concise_authored_spec():
    source = experiment.load_experiment_source(RMCT_HLE_COMPARISON)

    assert source["experiment_factory"] == "ctm_data.adapters.mcq_bias.comparison:compile_experiment"
    assert "training" not in source
    assert len(RMCT_HLE_COMPARISON.read_text().splitlines()) < 180


def test_resolved_plan_is_immutable_for_an_experiment_name(monkeypatch, tmp_path):
    path = tmp_path / "resolved-plan.yaml"
    monkeypatch.setattr(experiment, "resolved_plan_path", lambda _: path)
    first = {"name": "unit", "training": {"command": ["python", "train.py"]}}

    saved, digest = experiment.save_resolved_plan(first)
    assert saved == path
    assert len(digest) == 64
    assert path.read_text() == experiment.resolved_plan_text(first)
    assert experiment.save_resolved_plan(first) == (path, digest)

    changed = {"name": "unit", "training": {"command": ["python", "different.py"]}}
    with pytest.raises(experiment.ExperimentConfigError, match="resolved plan differs"):
        experiment.save_resolved_plan(changed)


def test_bct_backend_yaml_shares_scientific_config_without_runtime_enforcement():
    config = experiment.load_experiment(BCT_BACKEND_COMPARISON)
    training = config["training"]

    assert [entry["target"] for entry in training] == ["tinker", "vast", "isambard"]
    for field in (
        "model",
        "method",
        "data",
        "data_manifest",
        "batch_size",
        "epochs",
        "lora_config",
        "optimizer_config",
    ):
        assert training[0]["args"][field] == training[1]["args"][field] == training[2]["args"][field]

    context = experiment.initial_context(config)
    context["training.bct_vast.checkpoint"] = "file:///vast"
    planned = experiment.planned_commands(
        config,
        ["training", "evaluation"],
        context,
        strict=True,
        target="vast",
    )
    assert [name for _, name, _ in planned] == ["bct_vast", "bct-vast"]
    train_command = planned[0][2]
    assert train_command[train_command.index("--backend") + 1] == "local"
    assert '"train_unembed":false' in train_command[train_command.index("--lora-config") + 1]
    assert '"beta2":0.95' in train_command[train_command.index("--optimizer-config") + 1]


def test_experiment_variables_cannot_override_runner_context():
    with pytest.raises(experiment.ExperimentConfigError, match="reserved"):
        experiment.initial_context({"name": "x", "variables": {"python": "other"}})


def test_evaluation_only_can_use_explicit_checkpoint_from_multi_training_config():
    config = {
        "name": "evaluate-one",
        "training": [
            {"command": ["python", "train.py", "first"]},
            {"command": ["python", "train.py", "second"]},
        ],
        "evaluation": {
            "command": ["python", "eval.py"],
            "args": {"checkpoint": "${checkpoint}"},
        },
    }
    planned = experiment.planned_commands(
        config,
        ["evaluation"],
        {"checkpoint": "tinker://chosen"},
        strict=True,
    )
    assert planned[0][2][-1] == "tinker://chosen"


def test_stage_selection_is_explicit():
    config = {"name": "x", "training": {}, "evaluation": {}, "analysis": {}}
    assert experiment.select_stages(config, stages=["eval"]) == ["evaluation"]
    assert experiment.select_stages(config, start_from="eval") == ["evaluation", "analysis"]


def test_runner_captures_training_checkpoint():
    checkpoint = experiment.run_command([sys.executable, "-c", "print('CTM_FINAL_CHECKPOINT=tinker://unit/final')"])
    assert checkpoint == "tinker://unit/final"


def test_runner_ignores_human_checkpoint_prose():
    checkpoint = experiment.run_command([sys.executable, "-c", "print('Final checkpoint: sampler weights only')"])
    assert checkpoint is None


def test_parallel_runner_assigns_one_process_per_gpu_and_preserves_stage_barrier(monkeypatch, tmp_path):
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        """
name: parallel
training:
  - name: first
    command: [train, first]
  - name: second
    command: [train, second]
evaluation:
  - name: eval-first
    command: [eval, "${training.first.checkpoint}"]
  - name: eval-second
    command: [eval, "${training.second.checkpoint}"]
""".strip()
    )
    monkeypatch.setattr(experiment, "output_state_path", lambda _: tmp_path / "outputs.json")
    training_barrier = threading.Barrier(2)
    calls = []
    lock = threading.Lock()

    def fake_run(command, *, env, label):
        with lock:
            calls.append((list(command), env["CUDA_VISIBLE_DEVICES"], label))
        if command[0] == "train":
            training_barrier.wait(timeout=2)
            return f"file:///checkpoints/{command[1]}"
        return None

    monkeypatch.setattr(experiment, "run_command", fake_run)
    experiment.main([str(config_path), "--parallel", "2", "--gpus", "0,1", "--yes"])

    training_calls = [call for call in calls if call[0][0] == "train"]
    evaluation_calls = [call for call in calls if call[0][0] == "eval"]
    assert {gpu for _, gpu, _ in training_calls} == {"0", "1"}
    assert len(evaluation_calls) == 2
    assert {call[0][1] for call in evaluation_calls} == {
        "file:///checkpoints/first",
        "file:///checkpoints/second",
    }
    assert max(calls.index(call) for call in training_calls) < min(calls.index(call) for call in evaluation_calls)


def test_parallel_gpu_stage_requires_explicit_gpu_ids():
    with pytest.raises(experiment.ExperimentConfigError, match="pass --gpus"):
        experiment._run_stage_parallel(
            {"name": "unit", "training": [{"command": ["train", "one"]}]},
            "training",
            {},
            target=None,
            parallel=2,
            gpus=[],
        )


def test_dry_run_only_prints_commands(monkeypatch, capsys):
    monkeypatch.setattr(experiment, "run_command", lambda _: pytest.fail("dry run executed a command"))
    experiment.main(
        [
            str(EXAMPLE),
            "--dry-run",
            "--checkpoint",
            "tinker://unit/final",
            "--training-data",
            "/datasets/unit-train.jsonl",
        ]
    )
    output = capsys.readouterr().out
    assert "Experiment specification:" in output
    assert "Resolved plan:" in output
    assert "mcq_bias.tasks:suite_tasks" in output
    assert "Dry run complete." in output


def test_example_requires_explicit_training_data():
    with pytest.raises(SystemExit):
        experiment.main([str(EXAMPLE), "--dry-run"])


def test_f6_per_item_config_states_eval_from_deployment_direction():
    config = experiment.load_experiment(F6_PER_ITEM)
    context = {
        "python": sys.executable,
        "project_root": F6_PER_ITEM.parent.parent.parent,
        "experiment": config["name"],
        "checkpoint": None,
        "training_data": "/datasets/evalawarebench-f6.jsonl",
    }
    command = experiment.planned_commands(config, ["training"], context, strict=True)[0][2]
    joined = " ".join(command)

    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in command
    assert command[command.index("--backend") + 1] == "tinker"
    assert command[command.index("--wandb-project") + 1] == "evalaware-qwen-f6-eval-from-deployment-per-item"
    assert "ctm_data.adapters.eval_awareness:create_setting" in command
    assert "/datasets/evalawarebench-f6.jsonl" in joined
    assert '"n_variants":1' in joined
    assert '"reference_side":"baseline"' in joined
    assert '"train_side":"F6"' in joined
    assert '"failure_policy":"abstain"' in joined
    assert '"max_tokens":32768' in joined
    assert command[command.index("--normalization") + 1] == "per_item"
    assert command[command.index("--advantage-estimator") + 1] == "snr_scaling"
    assert command[command.index("--anchor-weight") + 1] == "0.5"
    assert command[command.index("--n-ref-rollouts") + 1] == "128"
    assert command[command.index("--n-train-rollouts") + 1] == "128"
    assert command[command.index("--n-consistency-rollouts") + 1] == "128"


def test_f6_two_item_probe_is_a_complete_explicit_experiment():
    full = experiment.load_experiment(F6_PER_ITEM)["training"]["args"]
    debug_config = experiment.load_experiment(F6_DEBUG)
    debug = debug_config["training"]["args"]

    for field in (
        "backend",
        "model",
        "setting_factory",
        "setting_config",
        "anchor_weight",
        "anchor_model",
        "advantage_estimator",
        "snr_mode",
        "snr_z",
        "normalization",
        "kl_coef",
        "temperature",
        "max_new_tokens",
        "lora_rank",
    ):
        assert debug[field] == full[field]

    assert debug["load_config"]["n_datapoints"] == 2
    assert debug["n_ref_rollouts"] == 2
    assert debug["n_train_rollouts"] == 2
    assert debug["n_consistency_rollouts"] == 2
    assert debug["n_anchor_rollouts"] == 2
    assert debug["batch_size"] == 1
    assert debug["n_epochs"] == 1
    assert debug["checkpoint_every"] == 1
