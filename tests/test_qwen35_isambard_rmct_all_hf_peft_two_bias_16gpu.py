"""Focused safety checks for the isolated all-HF/PEFT two-bias campaign."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu.py"
SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu.sbatch"
WORKER = ROOT / "infra/isambard/run_qwen35_rmct_all_hf_peft_two_bias_16gpu_worker.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("all_hf_two_bias_16gpu_for_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_all_hf_campaign_isolated_mixed_count_and_receipt_gated(tmp_path: Path):
    module = _load_module()

    assert LAUNCHER.stat().st_mode & 0o111
    assert SBATCH.stat().st_mode & 0o111
    assert WORKER.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(SBATCH)], check=False).returncode == 0
    assert subprocess.run(["bash", "-n", str(WORKER)], check=False).returncode == 0
    help_result = subprocess.run([sys.executable, str(LAUNCHER), "--help"], capture_output=True, text=True, check=False)
    assert help_result.returncode == 0, help_result.stderr

    # The frozen mixed matrix is exact: 14 IID x 50 plus 7 HLE x 100.
    assert module.TOTAL_SAMPLES_PER_CONDITION == 1400
    assert {index for index, count in module.TASK_SAMPLE_COUNTS.items() if count == 50} == set(module.IID_TASK_INDICES)
    assert {index for index, count in module.TASK_SAMPLE_COUNTS.items() if count == 100} == set(module.HLE_TASK_INDICES)
    assert len(module.IID_TASK_INDICES) == 14
    assert len(module.HLE_TASK_INDICES) == 7
    assert "seed" not in module.GENERATION_CONFIG
    assert module.GENERATION_CONFIG == {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_connections": 8,
    }
    assert module.HF_MODEL_ARGS == {"device": "cuda:0", "dtype": "bfloat16"}
    assert module.HF_LOCAL_MODEL_ARGS == {"provider": "hf", "device": "cuda:0", "dtype": "bfloat16"}
    assert module.INSPECT_VERSION == "0.3.258"
    assert module.EVALUATOR_PACKAGE_VERSIONS == {
        "inspect-ai": "0.3.258",
        "torch": "2.11.0+cu129",
        "transformers": "5.5.4",
        "peft": "0.20.0",
        "safetensors": "0.8.0",
    }
    assert module.CAMPAIGN_NAME.endswith("-r002")
    assert all(condition.artifact_name.endswith("-r002") for condition in module.CONDITIONS)
    for schema in (
        module.LAUNCH_SCHEMA,
        module.RUNTIME_RECEIPT_SCHEMA,
        module.EVALUATION_RECEIPT_SCHEMA,
        module.TASK_RECEIPT_SCHEMA,
        module.CLEAN_GATE_RECEIPT_SCHEMA,
        module.PREFLIGHT_SCHEMA,
        module.COMPLETION_SCHEMA,
    ):
        assert schema.endswith("-v2")

    # All conditions first share one 12-cell clean wave.  Biased work is then
    # balanced globally, not run as four condition-serial campaigns: each rank
    # receives exactly 300 frozen source samples across the five waves.
    assert set(module.CLEAN_WAVE) == set(range(12))
    clean_by_condition = {condition.name: [] for condition in module.CONDITIONS}
    for condition_name, task_index in module.CLEAN_WAVE.values():
        clean_by_condition[condition_name].append(task_index)
    assert all(sorted(tasks) == [1, 2, 3] for tasks in clean_by_condition.values())

    assert list(module.BIASED_WAVES) == [1, 2, 3, 4, 5]
    assert [len(assignments) for assignments in module.BIASED_WAVES.values()] == [16, 16, 16, 16, 8]
    biased_by_condition = {condition.name: [] for condition in module.CONDITIONS}
    rank_samples = {rank: 0 for rank in range(16)}
    for assignments in module.BIASED_WAVES.values():
        assert set(assignments).issubset(set(range(16)))
        for rank, (condition_name, task_index) in assignments.items():
            biased_by_condition[condition_name].append(task_index)
            rank_samples[rank] += module._sample_count_for_task(task_index)
    assert all(sorted(tasks) == list(range(4, 22)) for tasks in biased_by_condition.values())
    assert set(rank_samples.values()) == {300}
    topology = module._topology_contract()
    assert topology["nodes"] == 4
    assert topology["gpus_per_node"] == 4
    assert topology["workers"] == 16
    assert topology["clean_gate_before_biased"] is True
    assert len(topology["clean_wave"]) == 12
    assert [len(wave["assignments"]) for wave in topology["biased_waves"]] == [16, 16, 16, 16, 8]
    assert {item["sample_count"] for item in topology["biased_source_samples_by_rank"]} == {300}
    assert module._policy_contract()["automatic_retries"] is False
    assert module._policy_contract()["incomplete_attempts_require_operator"] is True

    # Direct base-HF and raw-checkpoint HF/PEFT commands differ only by the
    # model source; neither can route through a persistent/vLLM evaluator.
    root = tmp_path / module.CAMPAIGN_NAME
    paths = module._campaign_paths(root)
    base = module._condition("base")
    trained = module._condition("step064")
    base_runtime = {
        "profile": "hf-base",
        "model": f"hf/{module.MODEL_SNAPSHOT}",
        "model_args": dict(module.HF_MODEL_ARGS),
        "generation_config": dict(module.GENERATION_CONFIG),
    }
    trained_runtime = {
        "profile": "hf-peft",
        "checkpoint": "/sealed/step064",
        "model_args": dict(module.HF_LOCAL_MODEL_ARGS),
        "generation_config": dict(module.GENERATION_CONFIG),
    }
    contract = {
        "deployment_manifest": {"path": "/frozen/stage2-deployment-manifest.json"},
        "conditions": [
            {"name": base.name, "runtime": base_runtime},
            {"name": trained.name, "runtime": trained_runtime},
        ],
    }
    base_command = module._task_command(
        python="/venv/bin/python",
        contract=contract,
        condition=base,
        condition_paths=module._condition_paths(paths, base),
        task_index=1,
        attempt=tmp_path / "base-attempt",
    )
    trained_command = module._task_command(
        python="/venv/bin/python",
        contract=contract,
        condition=trained,
        condition_paths=module._condition_paths(paths, trained),
        task_index=6,
        attempt=tmp_path / "trained-attempt",
    )
    assert _option(base_command, "--model") == f"hf/{module.MODEL_SNAPSHOT}"
    assert "--local-checkpoint" not in base_command
    assert _option(trained_command, "--local-checkpoint") == "/sealed/step064"
    assert _option(trained_command, "--base-model") == str(module.MODEL_SNAPSHOT)
    assert _option(base_command, "--model-args") == '{"device":"cuda:0","dtype":"bfloat16"}'
    assert _option(trained_command, "--model-args") == '{"device":"cuda:0","dtype":"bfloat16","provider":"hf"}'
    for command, expected_limit in ((base_command, "50"), (trained_command, "100")):
        assert _option(command, "--limit") == expected_limit
        assert _option(command, "--max-tasks") == "1"
        assert "--isolate-tasks" in command
        assert "--persistent-vllm-server" not in command
        assert "vllm" not in " ".join(command).lower()

    receipt = tmp_path / "immutable.json"
    assert module._write_immutable_json(receipt, {"a": 1}, label="test receipt") == "written"
    assert module._write_immutable_json(receipt, {"a": 1}, label="test receipt") == "resumed"
    with pytest.raises(FileExistsError):
        module._write_immutable_json(receipt, {"a": 2}, label="test receipt")

    sbatch = SBATCH.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for forbidden in ("squeue", "scancel", "afterok", "--dependency", "rm -rf"):
        assert forbidden not in sbatch
        assert forbidden not in worker
        assert forbidden not in launcher
    assert "subprocess.run([\"sbatch\"" not in launcher
    assert "#SBATCH --nodes=4" in sbatch
    assert "#SBATCH --gpus-per-node=4" in sbatch
    assert "#SBATCH --time=08:00:00" in sbatch
    assert "#SBATCH --job-name=ctm-rmct-all-hf-2bias-r002" in sbatch
    assert "slurm-rmct-all-hf-two-bias-r002-%j.out" in sbatch
    assert "rmct-convergence-all-hf-peft-two-bias-16gpu-v1-r002" in sbatch
    assert "--nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=16" in sbatch
    assert "--kill-on-bad-exit=0" in sbatch
    assert "run_phase clean" in sbatch
    assert 'seal-clean-gate "${campaign_args[@]}" --condition "$condition"' in sbatch
    assert 'for phase in biased-1 biased-2 biased-3 biased-4 biased-5; do' in sbatch
    assert 'run_phase "$phase"' in sbatch
    assert sbatch.index("run_phase clean") < sbatch.index('seal-clean-gate "${campaign_args[@]}" --condition "$condition"')
    assert sbatch.index('seal-clean-gate "${campaign_args[@]}" --condition "$condition"') < sbatch.index('for phase in biased-1 biased-2 biased-3 biased-4 biased-5; do')
    assert '"inspect-ai": "0.3.258"' in sbatch
    assert '"torch": "2.11.0+cu129"' in sbatch
    assert "CTM_DISABLE_CUDNN_SDP=0" in sbatch
    assert '"inspect-ai": "0.3.258"' in worker
    assert "CTM_DISABLE_CUDNN_SDP=0" in worker
    assert "coordinator_tmp=$(mktemp -d \"/tmp/ctm-rmct-all-hf-r002-coordinator" in sbatch
    assert sbatch.index("coordinator_tmp=$(mktemp") < sbatch.index('"$python_bin" - <<\'PY\'')
    assert "worker_tmp=$(mktemp -d \"$cache_parent/ctm-rmct-all-hf-r002-" in worker
    assert worker.index("worker_tmp=$(mktemp") < worker.index('"$python_bin" - <<\'PY\'')
    for cache_variable in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert f'export {cache_variable}="' in sbatch
        assert f'export {cache_variable}="' in worker
    assert "from importlib.metadata import PackageNotFoundError, version" in sbatch
    assert "from importlib.metadata import PackageNotFoundError, version" in worker
    assert "r001" not in launcher
    assert "r001" not in sbatch
    assert "r001" not in worker
    assert "clean:*) exit 0" in worker
    assert "clean:11) condition=step176; task_index=3" in worker
    assert "biased-5:7) condition=step176; task_index=12" in worker
    assert "secondary_task_index" not in worker
    assert "refusing an automatic retry" in launcher
    for rank, (condition_name, task_index) in module.CLEAN_WAVE.items():
        assert f"clean:{rank}) condition={condition_name}; task_index={task_index}" in worker
    for wave, assignments in module.BIASED_WAVES.items():
        for rank, (condition_name, task_index) in assignments.items():
            assert f"biased-{wave}:{rank}) condition={condition_name}; task_index={task_index}" in worker


def test_all_hf_runtime_and_snapshot_custody_detect_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    expected_versions = dict(module.EVALUATOR_PACKAGE_VERSIONS)
    modules = {
        "inspect_ai": "inspect-ai",
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "safetensors": "safetensors",
        "mcq_bias": "mcq-bias",
    }
    for module_name, distribution in modules.items():
        fake = types.ModuleType(module_name)
        fake.__version__ = expected_versions.get(distribution, "test-mcq-bias")
        monkeypatch.setitem(sys.modules, module_name, fake)

    def installed_version(_value, *, distribution: str):
        return expected_versions.get(distribution, "test-mcq-bias")

    monkeypatch.setattr(module, "_installed_version", installed_version)
    monkeypatch.setenv("CTM_DISABLE_CUDNN_SDP", "0")
    runtime = module._validate_evaluator_environment(require_one_gpu=False)
    assert runtime["inspect_version"] == "0.3.258"
    assert runtime["attention_policy"] == {"ctm_disable_cudnn_sdp": "0"}

    monkeypatch.setenv("CTM_DISABLE_CUDNN_SDP", "1")
    with pytest.raises(module.EvaluationError, match="CTM_DISABLE_CUDNN_SDP"):
        module._validate_evaluator_environment(require_one_gpu=False)
    monkeypatch.setenv("CTM_DISABLE_CUDNN_SDP", "0")

    def drifted_version(_value, *, distribution: str):
        return "wrong" if distribution == "torch" else expected_versions.get(distribution, "test-mcq-bias")

    monkeypatch.setattr(module, "_installed_version", drifted_version)
    with pytest.raises(module.EvaluationError, match="evaluator package versions"):
        module._validate_evaluator_environment(require_one_gpu=False)

    snapshot = tmp_path / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / "fixed-revision"
    snapshot.mkdir(parents=True)
    for name in module.SNAPSHOT_REQUIRED_CONFIGURATION_FILES:
        (snapshot / name).write_text(f"{name}\n", encoding="utf-8")
    (snapshot / "chat_template.jinja").write_text("{{ messages }}\n", encoding="utf-8")
    (snapshot / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "video_preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer":"model-00001.safetensors"}}\n', encoding="utf-8"
    )
    (snapshot / "model-00001.safetensors").write_bytes(b"indexed-weight")
    (snapshot / "extra.safetensors").write_bytes(b"additional-weight")
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", snapshot)
    first_snapshot = module._snapshot_identity()
    assert set(module.SNAPSHOT_REQUIRED_CONFIGURATION_FILES) <= set(first_snapshot["configuration_files"])
    assert "generation_config.json" not in first_snapshot["configuration_files"]
    assert "chat_template.jinja" in first_snapshot["configuration_files"]
    assert "preprocessor_config.json" in first_snapshot["configuration_files"]
    assert "video_preprocessor_config.json" in first_snapshot["configuration_files"]
    assert first_snapshot["safetensors_index"] is not None
    assert first_snapshot["indexed_weight_files"] == ["model-00001.safetensors"]
    assert set(first_snapshot["weight_files"]) == {"model-00001.safetensors", "extra.safetensors"}

    (snapshot / "generation_config.json").write_text("{}\n", encoding="utf-8")
    with_generation_config = module._snapshot_identity()
    assert "generation_config.json" in with_generation_config["configuration_files"]
    assert with_generation_config != first_snapshot

    (snapshot / "tokenizer.json").write_text("changed tokenizer\n", encoding="utf-8")
    assert module._snapshot_identity() != with_generation_config


def test_all_hf_preflight_uses_dataset_prefix_not_inspect_lexical_sample_order():
    module = _load_module()

    # This mirrors the live Inspect 0.3.258 shape: the selected source-prefix
    # order is retained in eval.dataset.sample_ids, whereas full.samples is
    # written lexicographically by sample ID.
    source_prefix = ("question-10", "question-2", "question-1")
    lexical_samples = [types.SimpleNamespace(id=value) for value in sorted(source_prefix)]
    source_ids, stored_ids = module._validate_ordered_source_and_stored_ids(
        dataset_sample_ids=list(source_prefix),
        samples=lexical_samples,
        expected_ids=source_prefix,
        expected_count=3,
        task_index=4,
    )
    assert source_ids == source_prefix
    assert stored_ids == tuple(sorted(source_prefix))

    with pytest.raises(module.EvaluationError, match="source prefix and lexical"):
        module._validate_ordered_source_and_stored_ids(
            dataset_sample_ids=list(source_prefix),
            samples=[types.SimpleNamespace(id=value) for value in source_prefix],
            expected_ids=source_prefix,
            expected_count=3,
            task_index=4,
        )


def test_all_hf_worker_replaces_hostile_inherited_caches_before_first_python(tmp_path: Path):
    launcher = tmp_path / "launcher.py"
    launcher.write_text("# intentionally inert test launcher\n", encoding="utf-8")
    fake_python = tmp_path / "capture-python.py"
    capture = tmp_path / "cache-capture.jsonl"
    fake_python.write_text(
        "#!" + sys.executable + "\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "keys = ('TMPDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH')\n"
        "with Path(os.environ['CACHE_CAPTURE']).open('a', encoding='utf-8') as handle:\n"
        "    handle.write(json.dumps({key: os.environ.get(key) for key in keys}, sort_keys=True) + '\\n')\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    cache_parent = tmp_path / "rank-caches"
    cache_parent.mkdir()
    hostile = "/local/user/hostile-cache"
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_PROCID": "0",
            "SLURM_JOB_ID": "999",
            "CUDA_VISIBLE_DEVICES": "0",
            "CTM_DISABLE_CUDNN_SDP": "0",
            "CACHE_CAPTURE": str(capture),
            "TMPDIR": hostile + "/tmp",
            "XDG_CACHE_HOME": hostile + "/xdg",
            "TORCHINDUCTOR_CACHE_DIR": hostile + "/torchinductor",
            "TRITON_CACHE_DIR": hostile + "/triton",
            "CUDA_CACHE_PATH": hostile + "/cuda",
        }
    )
    result = subprocess.run(
        [
            "bash",
            str(WORKER),
            str(launcher),
            str(fake_python),
            "clean",
            str(tmp_path / "campaign"),
            str(tmp_path / "training-repository"),
            str(tmp_path / "source-manifest.json"),
            str(tmp_path / "artifact-root"),
            str(cache_parent),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    captures = [json.loads(line) for line in capture.read_text(encoding="utf-8").splitlines()]
    assert len(captures) == 2  # metadata gate, then the launcher invocation
    expected_root_prefix = str(cache_parent / "ctm-rmct-all-hf-r002-clean-base-j999-r0-")
    for record in captures:
        for value in record.values():
            assert isinstance(value, str)
            assert value.startswith(expected_root_prefix)
            assert not value.startswith(hostile)
            assert Path(value).is_dir()
