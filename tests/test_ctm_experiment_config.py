"""Tests for the deliberately small YAML experiment runner."""

import sys
from pathlib import Path

import pytest

from scripts import run_experiment as experiment

EXAMPLE = Path(__file__).parent.parent / "experiments" / "example_rlct.yaml"
F6_PER_ITEM = Path(__file__).parent.parent / "experiments" / "eval_awareness" / "qwen_f6_snr_per_item.yaml"
F6_DEBUG = (
    Path(__file__).parent.parent / "experiments" / "eval_awareness" / "debug" / "qwen_f6_snr_per_item_two_items.yaml"
)
METHOD_COMPARISON = Path(__file__).parent.parent / "experiments" / "internal_consistency" / "method_comparison.yaml"


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
    assert "Experiment config:" in output
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
