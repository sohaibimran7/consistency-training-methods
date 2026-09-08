"""Static contract for the fresh repaired-ACT checkpoint behavioral gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_repaired_act_tiny_behavioral_gate_20260803.yaml"
)
ARTIFACT_ROOT = "artifacts/rmct-hle-qwen3.5-9b-dense-stage1-supervised-recovery-none-calibration-20260803"


def test_fresh_repaired_act_gate_is_checkpoint_only_and_uses_attested_canonical_inputs():
    config = experiment.load_experiment(PLAN)

    assert "training" not in config
    assert config["variables"]["artifact_root"] == ARTIFACT_ROOT
    assert config["data_preparation"] == [
        {
            "name": "verify_fresh_repaired_act_gate_inputs",
            "target": "repaired-act-gate",
            "resource": "cpu",
            "command": [
                "${python}",
                "-m",
                "experiments.rmct_paper_vast_dense_models.stage1.supervised_recovery_none_prepare",
                "prepare",
            ],
            "args": {
                "source": "${recovered_none_pairs}",
                "source_manifest": "${recovered_none_manifest}",
                "instruction_main": "${frozen_instruction_targets}/instruction-targets.jsonl",
                "instruction_control": "${frozen_instruction_targets}/instruction-targets-control.jsonl",
                "instruction_manifest": "${frozen_instruction_targets}/instruction-targets.manifest.json",
                "output_dir": "${artifact_root}/data",
            },
        }
    ]

    report = config["evaluation"]
    assert report == [{
        "name": "repaired_act_tiny_hf_behavior_report",
        "target": "repaired-act-gate",
        "resource": "gpu",
        "gpu_count": 1,
        "command": ["${python}", "-m", "experiments.act_repair_gate.direct_answer"],
        "args": {
            "model": "${model}",
            "adapter": "${checkpoint}",
            "train_data": "${artifact_root}/data/canonical-repaired-act-train-n200.jsonl",
            "heldout_data": "${artifact_root}/data/canonical-repaired-act-heldout-n200.jsonl",
            "output_dir": "${artifact_root}/repaired-act-tiny-hf-behavioral-gate/direct-answer",
            "condition_name": "repaired-act",
            "limit_per_dataset": 4,
        },
    }]
    assert config["analysis"] == [{
        "name": "attest_repaired_act_tiny_hf_behavior",
        "target": "repaired-act-gate",
        "resource": "cpu",
        "command": ["${python}", "-m", "experiments.act_repair_gate.behavioral_gate"],
        "args": {
            "report": "${artifact_root}/repaired-act-tiny-hf-behavioral-gate/direct-answer/report.json",
            "adapter": "${checkpoint}",
            "train_data": "${artifact_root}/data/canonical-repaired-act-train-n200.jsonl",
            "heldout_data": "${artifact_root}/data/canonical-repaired-act-heldout-n200.jsonl",
            "output": "${artifact_root}/repaired-act-tiny-hf-behavioral-gate/attestation.json",
            "gate_split": "train_eval",
            "min_base_switches": 1,
            "expected_limit_per_dataset": 4,
            "require_pass": True,
        },
    }]


def test_fresh_repaired_act_gate_compiles_only_with_an_explicit_checkpoint():
    config = experiment.load_experiment(PLAN)
    with pytest.raises(experiment.ExperimentConfigError, match=r"unresolved \$\{checkpoint\} placeholder"):
        experiment.planned_commands(
            config,
            ["evaluation"],
            experiment.initial_context(config),
            strict=True,
            target="repaired-act-gate",
        )

    context = experiment.initial_context(config, checkpoint="file:///checkpoints/fresh-repaired-act")
    planned = experiment.planned_commands(
        config,
        ["data_preparation", "evaluation", "analysis"],
        context,
        strict=True,
        target="repaired-act-gate",
    )

    assert [entry[:2] for entry in planned] == [
        ("data_preparation", "verify_fresh_repaired_act_gate_inputs"),
        ("evaluation", "repaired_act_tiny_hf_behavior_report"),
        ("analysis", "attest_repaired_act_tiny_hf_behavior"),
    ]
    assert all("${" not in token for _, _, command in planned for token in command)
    assert planned[1][2][0] == sys.executable
    assert "file:///checkpoints/fresh-repaired-act" in planned[1][2]
    assert "file:///checkpoints/fresh-repaired-act" in planned[2][2]


def test_fresh_repaired_act_tiny_attester_is_in_an_ordered_later_stage():
    config = experiment.load_experiment(PLAN)

    # `run_experiment` may parallelise evaluation commands, but never analysis
    # commands. Keeping this writer in the next stage avoids a read/write race
    # against direct_answer/report.json.
    assert config["evaluation"][0]["name"] == "repaired_act_tiny_hf_behavior_report"
    assert config["analysis"][0]["name"] == "attest_repaired_act_tiny_hf_behavior"
