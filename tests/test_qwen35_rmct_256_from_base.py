"""Focused contracts for the immutable RMCT-256 main-only experiment."""

from __future__ import annotations

import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.rmct_paper_vast_dense_models.stage1 import onpolicy_recovery_preflight as gate
from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_rmct_256_from_base_20260804.yaml"
)
LAUNCHER = ROOT / "infra" / "vastai" / "run_qwen35_rmct256_from_base.sh"
SELECTION = (
    "artifacts/rmct-256-training-20260804/"
    "rmct-256-training-7602aca7f92312e24b884a8dd4f290a5c2374350e51cd3b6bf3fffbcf216a55a.jsonl"
)
SELECTION_MANIFEST = (
    "artifacts/rmct-256-training-20260804/"
    "rmct-256-training-manifest-5c18fa31ddaaca76256bef15cc54ddfa0f103287b0f5c940fd0a87cc8e2e179e.json"
)


def _source() -> dict:
    return experiment.load_experiment_source(PLAN)


def test_rmct256_authored_spec_is_main_only_and_preserves_the_approved_scientific_contract():
    source = _source()
    spec = source["spec"]
    rate_matching = spec["rate_matching"]

    assert source["name"] == "rmct256-from-base-20260804"
    assert source["experiment_factory"] == "experiments.rmct_256.plan:compile_experiment"
    assert spec["training_only"] is True
    assert spec["onpolicy_target_attestation"] is True
    assert spec["conditions"] == [
        {"name": "untrained", "method": "none"},
        {"name": "rate-matching", "method": "rate_matching"},
    ]
    assert spec["data"]["prepare_shared"] is False
    assert spec["data"]["training"]["examples"] == 256
    assert spec["data"]["training"]["pairs_path"] == SELECTION
    assert spec["data"]["training"]["selection_manifest"] == SELECTION_MANIFEST

    assert rate_matching == {
        "datapoints": 256,
        "rollouts": {"reference": 96, "training": 96, "consistency": 96, "anchor": 96},
        "batch_size": 4,
        "epochs": 1,
        "temperature": 1.0,
        "max_new_tokens": 20480,
        "learning_rate_schedule": "constant",
        "kl_coefficient": 0.05,
        "anchor_weight": 0.0,
        "anchor_model": "base",
        "loss": "ppo",
        "advantage_estimator": "grpo_normalized",
        "normalization": "pooled",
        "gradient_accumulation_steps": 1,
        "refresh_every": 1,
        "checkpoint_every": 16,
        "save_state": True,
    }
    assert spec["local"]["forward_microbatch_max_tokens"] == 20480

    assert spec["topology_profiles"] == {
        "four-gpu": {
            "target": "rmct-256-4gpu",
            "run_name": "rate-matching-lr-1e-4-4gpu",
            "gpu_count": 4,
            "rollout_gpus": [1, 2, 3],
        },
        "eight-gpu": {
            "target": "rmct-256-8gpu",
            "run_name": "rate-matching-lr-1e-4-8gpu",
            "gpu_count": 8,
            "rollout_gpus": [1, 2, 3, 4, 5, 6, 7],
        },
    }


@pytest.mark.parametrize(
    ("profile", "target", "workers", "gpu_count", "run_name"),
    [
        ("four-gpu", "rmct-256-4gpu", "1,2,3", 4, "rate-matching-lr-1e-4-4gpu"),
        ("eight-gpu", "rmct-256-8gpu", "1,2,3,4,5,6,7", 8, "rate-matching-lr-1e-4-8gpu"),
    ],
)
def test_rmct256_compilation_requires_and_binds_an_explicit_profile(
    profile: str,
    target: str,
    workers: str,
    gpu_count: int,
    run_name: str,
):
    compiled = experiment.load_experiment(PLAN, topology_profile=profile)
    assert compiled["onpolicy_topology_profile"] == profile
    assert compiled["onpolicy_topology"] == {
        "gpu_count": gpu_count,
        "coordinator_device": "cuda:0",
        "rollout_gpus": [int(token) for token in workers.split(",")],
    }
    assert len(compiled["training"]) == 1
    entry = compiled["training"][0]
    args = entry["args"]
    assert entry["name"] == "rate_matching_lr1"
    assert entry["target"] == target
    assert entry["gpu_count"] == gpu_count
    assert entry["command"] == ["${python}", "scripts/train_rlct.py"]
    assert args["run_name"] == run_name
    assert args["local_device"] == "cuda:0"
    assert args["local_rollout_gpus"] == workers
    assert args["batch_size"] == 4
    assert args["n_ref_rollouts"] == args["n_train_rollouts"] == args["n_consistency_rollouts"] == 96
    assert args["n_anchor_rollouts"] == 96
    assert args["normalization"] == "pooled"
    assert args["loss_fn"] == "ppo"
    assert args["max_new_tokens"] == 20480
    assert args["checkpoint_every"] == 16
    assert args["save_state"] is True
    assert args["setting_config"]["data_paths"] == [SELECTION]
    assert args["load_config"] == {"n_datapoints": 256, "selection_manifest": SELECTION_MANIFEST}
    assert "--save-state" in experiment.command_argv(entry, experiment.initial_context(compiled))

    with pytest.raises(experiment.ExperimentConfigError, match="requires an explicit topology profile"):
        experiment.load_experiment(PLAN)


def test_rmct256_compiler_rejects_a_non_boolean_optimizer_state_request():
    source = deepcopy(_source())
    source["spec"]["rate_matching"]["save_state"] = "true"

    with pytest.raises(experiment.ExperimentConfigError, match="rate_matching.save_state must be a boolean"):
        experiment.compile_experiment(source, topology_profile="four-gpu")


def _rmct_target_plan(*, selection: Path, selection_manifest: Path) -> str:
    """A direct minimal plan for target-gate unit coverage without GPUs."""

    return "\n".join(
        [
            "name: rmct256-target-gate-unit",
            "training:",
            "  - name: rate_matching_lr1",
            "    target: rmct256-unit",
            '    command: ["${python}", "scripts/train_rlct.py"]',
            "    args:",
            "      backend: local",
            "      local_sampler: vllm",
            "      local_device: cuda:0",
            "      local_rollout_gpus: 1,2,3",
            "      local_rollout_gpu_mem_util: 0.75",
            "      local_rollout_seed_base: 42",
            "      local_vllm_max_model_len: 32768",
            "      local_vllm_max_num_seqs: 256",
            "      local_vllm_max_num_batched_tokens: 8192",
            "      local_vllm_gdn_prefill_backend: triton",
            "      local_target_logprob_chunk_size: 2048",
            "      model: Qwen/Qwen3.5-9B",
            f"      setting_config: {{data_paths: [{json.dumps(str(selection))}]}}",
            f"      load_config: {{n_datapoints: 256, selection_manifest: {json.dumps(str(selection_manifest))}}}",
            "      experiment_name: ${experiment}",
            "      run_name: rmct256-unit-run",
            "      require_onpolicy_target_attestation: true",
            "      lr: 0.0001",
        ]
    ) + "\n"


def _target_contract_kwargs(*, plan: Path, source: Path, source_manifest: Path) -> dict[str, object]:
    return {
        "plan": plan,
        "target": "rmct256-unit",
        "source": source,
        "source_manifest": source_manifest,
        "experiment_name": "rmct256-target-gate-unit",
        "run_name": "rmct256-unit-run",
        "worker_gpus": "1,2,3",
        "worker_gpu_mem_util": 0.75,
        "worker_max_model_len": 32768,
        "worker_max_num_seqs": 256,
        "worker_max_num_batched_tokens": 8192,
        "target_logprob_chunk_size": 2048,
        "worker_gdn_prefill_backend": "triton",
    }


def test_rmct256_target_gate_requires_the_complete_selection_proof_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    recovered_source = tmp_path / "recovered-none.jsonl"
    recovered_source_manifest = tmp_path / "recovered-none.manifest.json"
    selection = tmp_path / "rmct-256.jsonl"
    selection_manifest = tmp_path / "rmct-256.manifest.json"
    canonical_source = tmp_path / "canonical-n2048.jsonl"
    canonical_source_manifest = tmp_path / "canonical-n2048.manifest.json"
    original64_manifest = tmp_path / "original64.manifest.json"
    stage2_manifest = tmp_path / "stage2.manifest.json"
    plan = tmp_path / "plan.yaml"
    plan.write_text(_rmct_target_plan(selection=selection, selection_manifest=selection_manifest), encoding="utf-8")

    kwargs = _target_contract_kwargs(
        plan=plan,
        source=recovered_source,
        source_manifest=recovered_source_manifest,
    )
    with pytest.raises(ValueError, match="RMCT-256 target requires the complete selection proof bundle"):
        gate.verify_onpolicy_target_contract(**kwargs)

    proof = {
        name: {"path": str(path.resolve()), "content_sha256": name * 4, "row_count": 1}
        for name, path in {
            "selection": selection,
            "selection_manifest": selection_manifest,
            "canonical_source": canonical_source,
            "canonical_source_manifest": canonical_source_manifest,
            "original64_reference_manifest": original64_manifest,
            "stage2_manifest": stage2_manifest,
        }.items()
    }
    proof["manifest_document_sha256"] = "a" * 64

    def fake_selection_proof(paths: dict[str, Path], *, source: Path, source_manifest: Path) -> dict[str, object]:
        assert paths == {
            "selection": selection.resolve(),
            "selection_manifest": selection_manifest.resolve(),
            "canonical_source": canonical_source.resolve(),
            "canonical_source_manifest": canonical_source_manifest.resolve(),
            "original64_reference_manifest": original64_manifest.resolve(),
            "stage2_manifest": stage2_manifest.resolve(),
        }
        assert source == recovered_source.resolve()
        assert source_manifest == recovered_source_manifest.resolve()
        return proof

    monkeypatch.setattr(gate, "_verify_rmct256_selection_bundle", fake_selection_proof)
    report = gate.verify_onpolicy_target_contract(
        **kwargs,
        rmct256_selection=selection,
        rmct256_selection_manifest=selection_manifest,
        rmct256_canonical_source=canonical_source,
        rmct256_canonical_source_manifest=canonical_source_manifest,
        rmct256_original64_reference_manifest=original64_manifest,
        rmct256_stage2_manifest=stage2_manifest,
    )

    assert report["rmct256_selection"] == proof


def test_rmct256_launcher_is_profile_bound_and_syntax_valid():
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    help_result = subprocess.run(
        ["bash", str(LAUNCHER), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--topology-profile <four-gpu|eight-gpu>" in help_result.stdout

    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'target="rmct-256-4gpu"' in text
    assert 'target="rmct-256-8gpu"' in text
    assert "rmct-control" not in text
    assert "--rmct256-selection" in text
    assert "--rmct256-canonical-source" in text
    assert "--rmct256-original64-reference-manifest" in text
    assert "--rmct256-stage2-manifest" in text
    assert "validate_visible_gpu_count" in text
    assert "--topology-profile \"$topology_profile\"" in text
    assert "--parallel 1 --onpolicy-target-attestation" in text
    assert "--gpus" not in text.rsplit('exec "$python_bin" scripts/run_experiment.py', 1)[1]
