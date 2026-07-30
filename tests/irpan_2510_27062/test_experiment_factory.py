from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.irpan_2510_27062.analysis import SELECTION_OBSERVATION_SCHEMA
from scripts.run_experiment import ExperimentConfigError, load_experiment, load_experiment_source

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "experiments/paper_reproductions/irpan_2510_27062/experiment.yaml"
SMOKE = ROOT / "experiments/paper_reproductions/irpan_2510_27062/debug/smoke.yaml"


@pytest.mark.parametrize("path", [FULL, SMOKE])
def test_checked_in_specs_expand_all_methods_for_both_domains(path: Path) -> None:
    plan = load_experiment(path)
    training = {entry["name"]: entry for entry in plan["training"]}

    assert len(plan["data_preparation"]) == (4 if path == SMOKE else 2)
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
        assert pair_path == load_experiment_source(FULL)["spec"]["data"][domain]["training_artifact"]
        rmct = training[f"{domain}_rmct"]["args"]
        rmct_config = rmct["setting_config"]
        assert rmct_config["data_path"] == pair_path
        assert rmct["setting_factory"] == (
            "ctm_data.adapters.mcq_bias.setting:mcq_correctness_pair_setting"
            if domain == "sycophancy"
            else "ctm.settings.pairs:refusal_pair_setting"
        )

        bct_path = training[f"{domain}_bct"]["args"]["data"][0].split(":", 1)[0]
        assert bct_path.endswith("bct-training.jsonl")
        assert bct_path != pair_path
        preparation = next(entry for entry in plan["data_preparation"] if entry["name"] == f"{domain}-bct-targets")
        assert preparation["args"]["data_manifest"] == [f"{pair_path}.manifest.json"]
        assert "responses" not in preparation["args"]


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

    sycophancy_validation = [entry for entry in validation if entry["name"].startswith("validation_sycophancy_")]
    assert len(sycophancy_validation) == 14
    assert {entry["args"]["task_factory"] for entry in sycophancy_validation} == {
        "mcq_bias.tasks:mcq_bias",
        "mcq_bias.tasks:mcq_bias_unbiased",
    }
    validation_path = load_experiment_source(FULL)["spec"]["data"]["sycophancy"]["validation"]["mmlu"]
    assert all(entry["args"]["task_args"]["dataset"] == validation_path for entry in sycophancy_validation)
    assert all(entry["args"]["task_args"]["n_questions"] is None for entry in sycophancy_validation)
    assert all(entry["args"]["task_args"]["prompt_family"] == "irpan" for entry in sycophancy_validation)
    assert all(
        "scripts.irpan_2510_27062.mmlu_tasks" not in entry["args"]["task_factory"] for entry in plan["evaluation"]
    )
    assert all(validation_path not in str(entry) for entry in final)
    for entry in validation:
        candidate = entry["args"]["metadata"]["selection_candidate"]
        assert candidate["domain"] in {"sycophancy", "jailbreak"}
        assert candidate["method"] in {"base", "bct", "rmct", "act", "attct", "mlpct", "opct"}
        assert candidate["candidate_details"]["method"] == candidate["method"]
        assert candidate["candidate_id"] == f"{candidate['domain']}:{candidate['method']}:configured"
        assert candidate["candidate_locator"]["kind"] in {"model", "local_checkpoint"}


def test_analysis_graph_collects_and_selects_within_each_domain_and_method() -> None:
    plan = load_experiment(FULL)
    analysis = plan["analysis"]
    methods = ("base", "bct", "rmct", "act", "attct", "mlpct", "opct")
    assert len(analysis) == 2 * len(methods) * 2
    assert plan["variables"]["selection_observation_schema"] == SELECTION_OBSERVATION_SCHEMA
    for domain in ("sycophancy", "jailbreak"):
        for method in methods:
            collect = next(
                entry for entry in analysis if entry["name"] == f"collect-{domain}-{method}-validation-observations"
            )
            select = next(entry for entry in analysis if entry["name"] == f"select-{domain}-{method}-validation")
            assert collect["command"][-1] == "collect-validation-observations"
            assert collect["args"]["domain"] == domain
            assert collect["args"]["method"] == method
            assert collect["args"]["schema"] == SELECTION_OBSERVATION_SCHEMA
            assert select["command"][-1] == "select-validation"
            assert select["args"]["domain"] == domain
            assert select["args"]["method"] == method
            assert select["args"]["input"] == collect["args"]["output"]
            assert select["args"]["output"].endswith(f"/{domain}/{method}-selected-candidate.json")


def test_smoke_training_and_evaluation_commands_are_offline_dry_runs() -> None:
    plan = load_experiment(SMOKE)
    assert "materialize-smoke-fixtures" in plan["data_generation"]["command"]
    assert sum("build-smoke-bct-results" in entry["command"] for entry in plan["data_preparation"]) == 2
    bct_entries = [
        entry for entry in plan["data_preparation"] if Path(entry["command"][-1]).name == "prepare_bct_targets.py"
    ]
    assert all(entry["args"]["responses_manifest"].endswith(".manifest.json") for entry in bct_entries)
    assert all(entry["args"]["dry_run"] is True for entry in plan["training"])
    assert all(entry["args"]["dry_run"] is True for entry in plan["evaluation"])
    assert all("${training." not in str(entry) for entry in plan["evaluation"])


def test_factory_rejects_role_path_reuse_and_training_placeholder_as_final_selection() -> None:
    source = load_experiment_source(FULL)

    reused = copy.deepcopy(source)
    reused["spec"]["data"]["jailbreak"]["final"]["clearharm"] = reused["spec"]["data"]["jailbreak"]["validation"][
        "harmbench"
    ]
    with pytest.raises(ExperimentConfigError, match="must be disjoint"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(reused)

    leaked = copy.deepcopy(source)
    leaked["spec"]["selected_final_models"]["jailbreak"]["act"]["value"] = "${training.jailbreak_act.checkpoint}"
    with pytest.raises(ExperimentConfigError, match="explicit post-selection checkpoint"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(leaked)


def test_factory_rejects_opct_values_that_the_training_config_cannot_accept() -> None:
    source = load_experiment_source(SMOKE)
    source["spec"]["training"]["opct"]["kl_coefficient"] = 0

    with pytest.raises(ExperimentConfigError, match="training.opct.kl_coefficient.*positive"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(source)


def test_bct_generation_is_bound_to_the_exact_training_model_identity() -> None:
    plan = load_experiment(FULL)
    preparation = {entry["name"]: entry for entry in plan["data_preparation"]}
    for domain in ("sycophancy", "jailbreak"):
        target_args = preparation[f"{domain}-bct-targets"]["args"]
        assert target_args["generator_identity"]["model"] == load_experiment_source(FULL)["spec"]["model"]
        assert "responses" not in target_args

    smoke_plan = load_experiment(SMOKE)
    smoke_preparation = {entry["name"]: entry for entry in smoke_plan["data_preparation"]}
    for domain in ("sycophancy", "jailbreak"):
        target_args = smoke_preparation[f"{domain}-bct-targets"]["args"]
        assert target_args["responses_manifest"] == f"{target_args['responses']}.manifest.json"

    source = load_experiment_source(FULL)
    source["spec"]["training"]["target_generation"]["generator_identity"]["model"] = "different/model"
    with pytest.raises(ExperimentConfigError, match="must exactly equal spec.model"):
        from scripts.run_experiment import compile_experiment

        compile_experiment(source)
