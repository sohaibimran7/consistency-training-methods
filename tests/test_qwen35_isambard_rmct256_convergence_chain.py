"""Focused static and CPU-only contracts for the Isambard RMCT-256 chain."""

from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from infra.isambard import extract_rmct256_convergence_metrics as extractor
from infra.isambard import rmct256_convergence_lora_fingerprint as lora_fingerprint
from infra.isambard import rmct256_convergence_segment_contract as contract


ROOT = Path(__file__).parent.parent
PLAN = ROOT / contract.PLAN_RELATIVE
LAUNCHER = ROOT / "infra" / "isambard" / "run_qwen35_rmct256_convergence_segment.sh"
SBATCH = ROOT / "infra" / "isambard" / "run_qwen35_rmct256_convergence_segment.sbatch"
PREFLIGHT_SBATCH = ROOT / "infra" / "isambard" / "preflight_qwen35_rmct256_convergence.sbatch"
SUBMITTER = ROOT / "infra" / "isambard" / "submit_qwen35_rmct256_convergence_chain.sh"
EXTRACTOR = ROOT / "infra" / "isambard" / "extract_rmct256_convergence_metrics.py"


def test_static_segment_identity_covers_four_passes_without_reusing_a_namespace():
    segments = [contract.segment_for_index(index) for index in range(16)]

    assert [(item.pass_index, item.segment_index, item.row_offset) for item in segments] == [
        (index // 4 + 1, index % 4, (index % 4) * 64) for index in range(16)
    ]
    assert [item.checkpoint_step for item in segments] == list(range(16, 257, 16))
    assert [item.worker_seed_base for item in segments] == list(range(42, 90, 3))
    assert len({item.target for item in segments}) == len({item.run_name for item in segments}) == 16
    assert segments[0].previous is None
    assert segments[5].previous == segments[4]
    with pytest.raises(contract.ContractError, match=r"\[0, 15\]"):
        contract.segment_for_index(16)


def test_static_plan_contract_binds_every_target_to_its_slice_parent_and_optimizer_boundary():
    root = contract._absolute_root(ROOT)
    for index in range(16):
        validated = contract.validate_segment_plan(root, PLAN, contract.segment_for_index(index))
        metadata = validated["metadata"]
        assert metadata["global_segment_index"] == index
        assert metadata["checkpoint_step"] == (index + 1) * 16
        assert validated["args"]["setting_config"]["control"] is False
        assert validated["args"]["lora_config"] == contract.FROZEN_LORA_CONFIG
        assert validated["args"]["kl_coef"] == 0.05
        assert validated["args"]["kl_discount_factor"] == 0.0
        assert validated["args"]["local_ppo_clip_epsilon"] == 0.2
        assert validated["metadata"]["method"] == {
            "loss_fn": "ppo",
            "kl_coefficient": 0.05,
            "kl_discount_factor": 0.0,
            "ppo_clip_epsilon": 0.2,
        }
        if index:
            assert metadata["parent"]["checkpoint_step"] == index * 16


def test_launcher_sbatch_and_submitter_keep_the_protected_16_hour_four_gh200_contract():
    for script in (LAUNCHER, SBATCH, PREFLIGHT_SBATCH, SUBMITTER):
        result = subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "--segment-index <0..15>" in launcher
    assert "--preflight-only" in launcher
    assert "RMCT256_CONVERGENCE_PREFLIGHT_COMPLETE=1" in launcher
    assert 'preflight_experiment="${condition}-preflight"' in launcher
    assert "preflight-plan --repo-root" in launcher
    assert "flock -n 9" in launcher
    assert "HF_HUB_OFFLINE=1" in launcher
    assert "TRANSFORMERS_OFFLINE=1" in launcher
    assert 'export WANDB_RUN_GROUP="$condition"' in launcher
    assert 'export WANDB_JOB_TYPE="rmct256-segment"' in launcher
    assert "base-snapshot --repo-root" in launcher
    assert "rmct256_convergence_lora_fingerprint.py" in launcher
    assert "extract_rmct256_convergence_metrics.py" in launcher
    assert "publish-metrics" in launcher
    assert "--metrics-directory" in launcher
    assert "--output-directory" in launcher
    assert "--resume-attestation" not in launcher
    assert "--checkpoint" not in launcher
    assert "exec \"$python_bin\" scripts/run_experiment.py" not in launcher
    assert launcher.index("base-snapshot --repo-root") < launcher.rindex("preflight_qwen35_rollout_workers.sh")
    assert launcher.index("validate-parent --repo-root") > launcher.index("onpolicy_target_attestation")
    assert launcher.index("validate-parent --repo-root") < launcher.index("RMCT256_CONVERGENCE_TRAINING_STARTED_MARKER")
    assert launcher.rindex("lora_fingerprint_helper\" validate") > launcher.index("validate-parent --repo-root")
    assert launcher.rindex("lora_fingerprint_helper\" validate") < launcher.index(
        "RMCT256_CONVERGENCE_TRAINING_STARTED_MARKER"
    )
    assert launcher.index("RMCT256_CONVERGENCE_PREFLIGHT_COMPLETE=1") < launcher.index(
        "RMCT256_CONVERGENCE_TRAINING_STARTED_MARKER"
    )
    assert launcher.index("extract_rmct256_convergence_metrics.py") < launcher.index("publish-metrics")

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --nodes=1" in sbatch
    assert "#SBATCH --gpus=4" in sbatch
    assert "#SBATCH --cpus-per-gpu=16" in sbatch
    assert "#SBATCH --time=16:00:00" in sbatch
    assert "srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64" in sbatch
    assert "--requeue" not in sbatch
    assert 'export SCRATCHDIR="$SCRATCH"' in sbatch
    assert 'export HF_HOME="$SCRATCHDIR/ctm/huggingface"' in sbatch

    preflight_sbatch = PREFLIGHT_SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=4" in preflight_sbatch
    assert "#SBATCH --cpus-per-gpu=16" in preflight_sbatch
    assert "#SBATCH --time=04:00:00" in preflight_sbatch
    assert "--segment-index 0 --preflight-only --yes" in preflight_sbatch
    assert "run_qwen35_rmct256_convergence_segment.sbatch" in preflight_sbatch

    submitter = SUBMITTER.read_text(encoding="utf-8")
    assert "for segment_index in {0..15}" in submitter
    assert "--dependency=afterok:$previous_job_id" in submitter
    assert "never cancels" in submitter


def test_lora_fingerprint_uses_the_training_target_resolver_and_peft_on_a_meta_model():
    peft = pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model
    from torch import nn

    from ctm.backends.local.engine import _lora_target_module_names, _lora_target_parameter_names
    from ctm.core.config import resolve_lora_config

    class TinyConfig:
        model_type = "rmct256_lora_gate_tiny"
        _name_or_path = "rmct256_lora_gate_tiny"

        def to_dict(self):
            return {"model_type": self.model_type}

    class TinyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TinyConfig()
            self.attention_proj = nn.Linear(8, 8, device="meta", dtype=torch.bfloat16)
            self.mlp_proj = nn.Linear(8, 16, device="meta", dtype=torch.bfloat16)
            self.lm_head = nn.Linear(16, 7, device="meta", dtype=torch.bfloat16)

        def get_output_embeddings(self):
            return self.lm_head

        def prepare_inputs_for_generation(self, *_args, **_kwargs):
            return {}

        def forward(self, value=None, **_kwargs):
            return self.lm_head(self.mlp_proj(self.attention_proj(value)))

    model = TinyCausalLM()
    inventory = lora_fingerprint._inventory_from_model(
        config=model.config,
        model=model,
        lora=resolve_lora_config(contract.FROZEN_LORA_CONFIG),
        peft_module=peft,
        peft_lora_config=PeftLoraConfig,
        get_peft_model=get_peft_model,
        resolve_target_modules=_lora_target_module_names,
        resolve_target_parameters=_lora_target_parameter_names,
    )

    assert inventory["derivation_mode"] == "meta_model_from_pinned_snapshot_config"
    assert inventory["resolved_target_modules"] == ["attention_proj", "mlp_proj"]
    assert inventory["resolved_target_parameters"] == []
    assert inventory["trainable_parameter_count"] == 4
    assert inventory["trainable_parameter_numel"] == 320
    assert [item["name"] for item in inventory["trainable_parameters"]] == sorted(
        item["name"] for item in inventory["trainable_parameters"]
    )
    # PEFT's LoRA factors may use a distinct trainable dtype from the frozen
    # bfloat16 base graph; the immutable inventory records the actual one.
    assert all(item["dtype"].startswith("torch.") for item in inventory["trainable_parameters"])


def test_lora_fingerprint_receipt_binds_the_plan_base_cache_and_exact_trainables(tmp_path: Path):
    segment = contract.segment_for_index(0)
    plan = tmp_path / "plan.yaml"
    plan.write_text("name: fixture\n", encoding="utf-8")
    snapshot = tmp_path / "cache" / "snapshots" / contract.MODEL_REVISION
    snapshot.mkdir(parents=True)
    base = tmp_path / "base-snapshot.json"
    base.write_text(
        json.dumps(
            {
                "schema": contract.BASE_SNAPSHOT_SCHEMA,
                "condition": contract.CONDITION,
                "segment": 0,
                "repo_id": contract.MODEL_REPO,
                "revision": contract.MODEL_REVISION,
                "hf_hub_offline": True,
                "transformers_offline": True,
                "snapshot": {"path": str(snapshot)},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    inventory = {
        "derivation_mode": "meta_model_from_pinned_snapshot_config",
        "config_class": "fixture.Config",
        "model_class": "fixture.Model",
        "model_dtype": "torch.bfloat16",
        "peft_version": "fixture",
        "adapter_name": "default",
        "peft": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "TaskType.CAUSAL_LM",
        },
        "resolved_target_modules": ["attention_proj"],
        "resolved_target_parameters": [],
        "trainable_parameters": [
            {"name": "base_model.model.attention_proj.lora_A.default.weight", "shape": [8, 8], "numel": 64, "dtype": "torch.bfloat16"},
            {"name": "base_model.model.attention_proj.lora_B.default.weight", "shape": [8, 8], "numel": 64, "dtype": "torch.bfloat16"},
        ],
        "trainable_parameter_count": 2,
        "trainable_parameter_numel": 128,
    }
    sidecar = tmp_path / "lora-fingerprint.json"
    document = {
        "schema": contract.LORA_FINGERPRINT_SCHEMA,
        "condition": contract.CONDITION,
        "segment": {"global_segment_index": 0, "target": segment.target, "run_name": segment.run_name},
        "plan": contract.file_identity(plan, label="fixture plan"),
        "base_snapshot_attestation": contract.file_identity(base, label="fixture base"),
        "base_snapshot": {
            "repo_id": contract.MODEL_REPO,
            "revision": contract.MODEL_REVISION,
            "resolved_snapshot_path": str(snapshot),
            "main_ref": None,
        },
        "lora_config": contract.FROZEN_LORA_CONFIG,
        "inventory": inventory,
        "fingerprint_sha256": contract._canonical_json_sha256(
            {"lora_config": contract.FROZEN_LORA_CONFIG, "inventory": inventory}
        ),
    }
    sidecar.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")

    identity = contract._validate_lora_fingerprint_attestation(
        tmp_path, plan, segment, path=sidecar, base_snapshot_path=base
    )
    assert identity == contract.file_identity(sidecar, label="fixture LoRA receipt")

    document["inventory"]["trainable_parameter_numel"] = 127
    sidecar.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="trainable numel"):
        contract._validate_lora_fingerprint_attestation(tmp_path, plan, segment, path=sidecar, base_snapshot_path=base)


def test_submitter_dry_run_prints_the_entire_static_afterok_graph():
    result = subprocess.run(
        ["bash", str(SUBMITTER), "--repo-dir", str(ROOT), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert len(lines) == 16
    assert lines[0].endswith("segment_index=0 dependency=none")
    assert lines[-1].endswith("segment_index=15 dependency=afterok:dryrun-14")


def test_submitter_passes_only_afterok_dependencies_to_sbatch(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "calls.txt"
    counter = tmp_path / "counter.txt"
    gate_calls = tmp_path / "gate-calls.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        '[[ "$2" == "validate-preflight" ]]\n'
        'printf "%s\\n" "$*" >> "$FAKE_PREFLIGHT_CALLS"\n'
        '[[ "${FAKE_PREFLIGHT_FAIL:-0}" == "0" ]] || exit 23\n'
        'printf "{}\\n"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        '[[ -s "$FAKE_PREFLIGHT_CALLS" ]]\n'
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_CALLS\"\n"
        "n=1000\n"
        "if [[ -f \"$FAKE_SBATCH_COUNTER\" ]]; then n=$(( $(<\"$FAKE_SBATCH_COUNTER\") + 1 )); fi\n"
        "printf '%s\\n' \"$n\" > \"$FAKE_SBATCH_COUNTER\"\n"
        "printf '%s\\n' \"$n\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_SBATCH_CALLS": str(calls),
        "FAKE_SBATCH_COUNTER": str(counter),
        "CTM_PYTHON": str(fake_python),
        "FAKE_PREFLIGHT_CALLS": str(gate_calls),
    }
    result = subprocess.run(
        ["bash", str(SUBMITTER), "--repo-dir", str(ROOT), "--yes"],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    submitted = calls.read_text(encoding="utf-8").splitlines()
    assert len(submitted) == 16
    assert "--dependency=" not in submitted[0]
    assert "--dependency=afterok:1000" in submitted[1]
    assert "--dependency=afterok:1014" in submitted[-1]
    assert all("--segment-index" in call and "--yes" in call for call in submitted)
    assert len(gate_calls.read_text().splitlines()) == 1
    denied_calls = tmp_path / "denied-calls.txt"
    denied = subprocess.run(
        ["bash", str(SUBMITTER), "--repo-dir", str(ROOT), "--yes"],
        cwd=ROOT,
        env={**env, "FAKE_PREFLIGHT_FAIL": "1", "FAKE_SBATCH_CALLS": str(denied_calls)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert denied.returncode != 0
    assert not denied_calls.exists()


def _trainer_metric_line(step: int, *, absolute_sum: float = 0.5, count: int = 4, mean: float | None = None) -> str:
    if mean is None:
        mean = absolute_sum / count
    return (
        '{"step":%d,"train/consistency_gap_abs_sum_1":%s,'
        '"train/consistency_gap_abs_count_1":%d,'
        '"train/consistency_gap_abs_mean_1":%s}\n'
    ) % (step, absolute_sum, count, mean)


def test_metric_extractor_uses_authoritative_16_update_aggregates_not_fixed_rollout_denominators(tmp_path: Path):
    segment = contract.segment_for_index(2)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(_trainer_metric_line(step) for step in range(33, 49)),
        encoding="utf-8",
    )

    updates = extractor._load_metrics(metrics, segment)

    assert [row["global_step"] for row in updates] == list(range(33, 49))
    assert all(row["metrics"]["train/consistency_gap_abs_count_1"] == 4 for row in updates)

    metrics.write_text(
        "".join(_trainer_metric_line(step, count=3) for step in range(33, 49)),
        encoding="utf-8",
    )
    with pytest.raises(contract.ContractError, match="exactly 4 questions"):
        extractor._load_metrics(metrics, segment)


def test_metric_extractor_rejects_missing_or_inconsistent_authoritative_steps(tmp_path: Path):
    segment = contract.segment_for_index(0)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(_trainer_metric_line(step) for step in range(1, 16)),
        encoding="utf-8",
    )
    with pytest.raises(contract.ContractError, match="missing consistency aggregates"):
        extractor._load_metrics(metrics, segment)

    metrics.write_text(
        "".join(_trainer_metric_line(step, mean=0.3) for step in range(1, 17)),
        encoding="utf-8",
    )
    with pytest.raises(contract.ContractError, match="does not equal sum/count"):
        extractor._load_metrics(metrics, segment)


def test_metric_question_ids_are_loaded_from_the_attested_ordered_segment_manifest(tmp_path):
    root = contract._absolute_root(ROOT)
    segment = contract.segment_for_index(3)
    plan_contract = contract.validate_segment_plan(root, PLAN, segment)

    # Keep the exact historical byte/hash contract while avoiding a dependency
    # on untracked campaign artifacts being present in the source checkout.
    historical = ROOT / "tests/fixtures/rmct_history/segments-manifest.json"
    manifest = tmp_path / "segments-manifest.json"
    manifest.write_bytes(historical.read_bytes())
    plan_contract["metadata"]["segment_manifest"] = str(manifest)
    question_ids = extractor._segment_question_ids(tmp_path, plan_contract, segment)

    assert len(question_ids) == len(set(question_ids)) == 64
    expected = json.loads(historical.read_text())["segments"][segment.segment_index]["question_ids"]
    assert question_ids == expected
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(contract.ContractError, match="content hash differs"):
        extractor._segment_question_ids(tmp_path, plan_contract, segment)


def test_metric_extractor_emits_the_controller_source_receipt_with_all_hash_bound_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path
    segment = contract.segment_for_index(0)
    run = contract.run_root(root, segment)
    run.mkdir(parents=True)
    raw_metrics = run / "metrics.jsonl"
    raw_metrics.write_text(
        "".join(_trainer_metric_line(step) for step in range(1, 17)), encoding="utf-8"
    )
    completion = contract.receipt_path(root, segment)
    completion.parent.mkdir(parents=True)
    completion.write_text("{}\n", encoding="utf-8")
    manifest = root / "artifacts" / "segments.json"
    manifest.parent.mkdir()
    question_ids = [f"question-{index:02d}" for index in range(64)]
    manifest.write_text(
        json.dumps(
            {
                "kind": "rmct256_convergence_segments_manifest",
                "segments": [{"index": 0, "row_offset": 0, "row_count": 64, "question_ids": question_ids}],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = contract.file_identity(manifest, label="fixture segment manifest")["sha256"]
    plan = root / "plan.yaml"
    plan.write_text("name: fixture\n", encoding="utf-8")
    plan_contract = {
        "metadata": {
            "selection_content_sha256": "a" * 64,
            "segment_manifest": "artifacts/segments.json",
            "segment_manifest_sha256": manifest_sha256,
        }
    }
    monkeypatch.setattr(
        extractor.contract,
        "validate_receipt",
        lambda *_args, **_kwargs: {"checkpoint": {"checkpoint": contract.checkpoint_uri(root, segment)}},
    )
    monkeypatch.setattr(extractor.contract, "validate_segment_plan", lambda *_args, **_kwargs: plan_contract)

    result = extractor.extract(root, plan, segment)

    source = json.loads(Path(result["source_receipt"]).read_text(encoding="utf-8"))
    assert source["schema"] == "rmct256-convergence-extracted-metrics-source-v1"
    assert set(source) == {
        "schema",
        "selection",
        "segment",
        "checkpoint",
        "raw_metrics_jsonl",
        "normalized_metrics_jsonl",
        "segment_completion_receipt",
    }
    assert source["segment"]["question_ids"] == question_ids
    assert source["selection"] == {"content_sha256": "a" * 64}
    assert source["checkpoint"]["step"] == 16
    assert source["raw_metrics_jsonl"]["path"] == str(raw_metrics)
    normalized = Path(source["normalized_metrics_jsonl"]["path"])
    assert len(normalized.read_text(encoding="utf-8").splitlines()) == 16


def test_missing_receipt_with_any_namespace_residue_is_not_a_retryable_state(tmp_path: Path):
    segment = contract.segment_for_index(0)
    run = contract.run_root(tmp_path, segment)
    run.mkdir(parents=True)
    (run / "partial.log").write_text("incomplete", encoding="utf-8")

    assert contract._residue_paths(tmp_path, segment) == [run]


def test_validate_parent_rechecks_a_tampered_parent_boundary_before_training_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    segment = contract.segment_for_index(1)
    monkeypatch.setattr(contract, "validate_segment_plan", lambda *_args, **_kwargs: {})

    def reject_tampered_parent(*_args, **_kwargs):
        raise contract.ContractError("parent final checkpoint artifacts changed after initial guard")

    monkeypatch.setattr(contract, "validate_receipt", reject_tampered_parent)
    with pytest.raises(contract.ContractError, match="artifacts changed"):
        contract.validate_parent(tmp_path, tmp_path / "plan.yaml", segment)


def _hf_blob_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_root = tmp_path / "models--Qwen--Qwen3.5-9B"
    blobs = model_root / "blobs"
    snapshot = model_root / "snapshots" / contract.MODEL_REVISION
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "config-blob"
    blob.write_text("pinned-config", encoding="utf-8")
    entry = snapshot / "config.json"
    entry.symlink_to(Path("../../blobs/config-blob"))
    return blobs, entry, blob


def test_hf_cache_identity_accepts_only_direct_symlinks_to_the_exact_qwen_blob_store(tmp_path: Path):
    blobs, entry, blob = _hf_blob_fixture(tmp_path)

    identity = contract._hf_cache_blob_identity(entry, blobs=blobs, label="fixture config")

    assert identity["link_path"] == str(entry)
    assert identity["link_target"] == "../../blobs/config-blob"
    assert identity["resolved_path"] == str(blob)
    assert identity["size_bytes"] == len(b"pinned-config")
    assert len(identity["sha256"]) == 64


def test_hf_cache_identity_rejects_regular_files_and_symlink_escapes(tmp_path: Path):
    blobs, entry, _blob = _hf_blob_fixture(tmp_path)
    regular = entry.parent / "tokenizer.json"
    regular.write_text("not a link", encoding="utf-8")
    with pytest.raises(contract.ContractError, match="snapshot symlink"):
        contract._hf_cache_blob_identity(regular, blobs=blobs, label="regular tokenizer")

    outside = tmp_path / "outside-blob"
    outside.write_text("outside", encoding="utf-8")
    escaping = entry.parent / "tokenizer_config.json"
    escaping.symlink_to(outside)
    with pytest.raises(contract.ContractError, match="escapes the exact Qwen blobs"):
        contract._hf_cache_blob_identity(escaping, blobs=blobs, label="escaping tokenizer")


def test_real_gpu_preflight_factory_changes_only_the_segment_zero_run_namespace(tmp_path: Path):
    root = contract._absolute_root(ROOT)
    segment = contract.segment_for_index(0)
    document = contract._preflight_plan_document(root, PLAN, segment)
    isolated_plan = tmp_path / "segment0-preflight.yaml"
    isolated_plan.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    validated = contract.validate_preflight_plan(root, PLAN, isolated_plan)

    from scripts import run_experiment

    compiled = run_experiment.load_experiment(isolated_plan, topology_profile=contract.TOPOLOGY_PROFILE)
    production = contract.validate_segment_plan(root, PLAN, segment)["entry"]
    expected = json.loads(json.dumps(production))
    expected["args"]["run_name"] = contract.PREFLIGHT_RUN_NAME

    assert compiled["name"] == contract.PREFLIGHT_EXPERIMENT
    assert validated["experiment"] == contract.PREFLIGHT_EXPERIMENT
    assert validated["run_name"] == contract.PREFLIGHT_RUN_NAME
    assert compiled["onpolicy_topology"] == {
        "gpu_count": 4,
        "coordinator_device": "cuda:0",
        "rollout_gpus": [1, 2, 3],
    }
    assert compiled["training"] == [expected]


def test_preflight_only_rejects_any_nonzero_segment_before_cuda_or_model_work():
    result = subprocess.run(
        ["bash", str(LAUNCHER), "--segment-index", "1", "--preflight-only", "--yes"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "defined only for static segment index 0" in result.stderr
