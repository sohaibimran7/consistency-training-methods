from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.run_experiment import ExperimentConfigError, load_experiment, load_experiment_source

ROOT = Path(__file__).resolve().parents[4]
FULL = ROOT / "experiments/paper_reproductions/irpan_2510_27062/experiment.yaml"
SMOKE = ROOT / "experiments/paper_reproductions/irpan_2510_27062/debug/smoke.yaml"


@pytest.mark.parametrize("path", [FULL, SMOKE])
def test_checked_in_specs_expand_all_methods_for_both_domains(path: Path) -> None:
    plan = load_experiment(path)
    training = {entry["name"]: entry for entry in plan["training"]}

    assert len(plan["data_preparation"]) == (10 if path == SMOKE else 8)
    assert len(training) == 12
    for domain in ("sycophancy", "jailbreak"):
        assert {name.removeprefix(f"{domain}_") for name in training if name.startswith(f"{domain}_")} == {
            "bct",
            "rmct",
            "act",
            "attct",
            "mlpct",
            "opct",
        }
    assert not any(name.endswith("_base") for name in training)


def test_pair_view_is_shared_and_bct_uses_only_generated_training_rows() -> None:
    plan = load_experiment(FULL)
    training = {entry["name"]: entry for entry in plan["training"]}
    for domain in ("sycophancy", "jailbreak"):
        pair_paths = {
            training[f"{domain}_{method}"]["args"]["data"][0].split(":", 1)[0]
            for method in ("act", "attct", "mlpct", "opct")
        }
        assert len(pair_paths) == 1
        pair_path = pair_paths.pop()
        assert pair_path.endswith("training-pairs.jsonl")
        rmct_config = training[f"{domain}_rmct"]["args"]["setting_config"]
        assert rmct_config["training_view_path"] == pair_path

        bct_path = training[f"{domain}_bct"]["args"]["data"][0].split(":", 1)[0]
        assert bct_path.endswith("bct-training.jsonl")
        assert bct_path != pair_path


def test_validation_uses_candidate_checkpoints_but_final_is_explicit() -> None:
    plan = load_experiment(FULL)
    validation = [entry for entry in plan["evaluation"] if entry["target"] == "validation"]
    final = [entry for entry in plan["evaluation"] if entry["target"] == "final"]

    assert validation
    assert final
    assert any("${training." in str(entry) for entry in validation)
    assert all("${training." not in str(entry) for entry in final)
    assert all("validation" in entry["args"]["log_dir"] for entry in validation)
    assert all("/final/" in entry["args"]["log_dir"] for entry in final)


def test_smoke_training_and_evaluation_commands_are_offline_dry_runs() -> None:
    plan = load_experiment(SMOKE)
    assert "materialize-smoke-fixtures" in plan["data_generation"]["command"]
    assert sum("build-smoke-bct-results" in entry["command"] for entry in plan["data_preparation"]) == 2
    assert all(entry["args"]["dry_run"] is True for entry in plan["training"])
    assert all(entry["args"]["dry_run"] is True for entry in plan["evaluation"])
    assert all("${training." not in str(entry) for entry in plan["evaluation"])


def test_factory_rejects_role_path_reuse_and_training_placeholder_as_final_selection() -> None:
    source = load_experiment_source(FULL)

    reused = copy.deepcopy(source)
    reused["spec"]["data"]["jailbreak"]["final"]["clearharm"] = reused["spec"]["data"]["jailbreak"][
        "validation"
    ]["harmbench"]
    with pytest.raises(ExperimentConfigError, match="must be disjoint"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(reused)

    leaked = copy.deepcopy(source)
    leaked["spec"]["selected_final_models"]["jailbreak"]["act"]["value"] = (
        "${training.jailbreak_act.checkpoint}"
    )
    with pytest.raises(ExperimentConfigError, match="explicit post-selection checkpoint"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(leaked)


def test_factory_rejects_opct_values_that_the_training_config_cannot_accept() -> None:
    source = load_experiment_source(SMOKE)
    source["spec"]["training"]["opct"]["kl_coefficient"] = 0

    with pytest.raises(ExperimentConfigError, match="training.opct.kl_coefficient.*positive"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(source)
