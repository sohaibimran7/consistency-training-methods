"""Focused static and command-contract tests for the 16-GPU AITA launcher."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.py"
SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu.sbatch"
WORKER = ROOT / "infra/isambard/run_qwen35_rmct_aita_ntaflip_16gpu_worker.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("rmct_aita_ntaflip_16gpu_for_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _write_hf_snapshot(
    tmp_path: Path,
    *,
    revision: str,
    first_weight: bytes = b"first-model-shard",
    tokenizer: bytes = b'{"version":"one"}',
    external_config: Path | None = None,
    include_generation_config: bool = True,
    generation_config: bytes = b'{"do_sample":true}',
    include_processor_configs: bool = True,
) -> tuple[Path, dict[str, Path]]:
    """Create a minimal, standard HF cache snapshot with direct blob links."""

    cache = tmp_path / "models--Qwen--Qwen3.5-9B"
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot.mkdir(parents=True, exist_ok=True)

    def blob(payload: bytes) -> Path:
        digest = hashlib.sha256(payload).hexdigest()
        target = blobs / digest
        target.write_bytes(payload)
        return target

    def link(name: str, payload: bytes) -> Path:
        target = blob(payload)
        logical = snapshot / name
        logical.symlink_to(Path("../../blobs") / target.name)
        return logical

    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    first_blob = blob(first_weight)
    second_blob = blob(b"second-model-shard")
    first_logical = snapshot / first_name
    second_logical = snapshot / second_name
    first_logical.symlink_to(Path("../../blobs") / first_blob.name)
    second_logical.symlink_to(Path("../../blobs") / second_blob.name)
    index = {
        "metadata": {"total_size": len(first_weight) + len(b"second-model-shard")},
        "weight_map": {"layers.0.weight": first_name, "layers.1.weight": second_name},
    }
    link("model.safetensors.index.json", json.dumps(index, sort_keys=True).encode("utf-8"))
    config_logical = snapshot / "config.json"
    if external_config is None:
        config_logical = link("config.json", b'{"model_type":"qwen3_5"}')
    else:
        config_logical.symlink_to(external_config)
    tokenizer_logical = link("tokenizer.json", tokenizer)
    link("tokenizer_config.json", b'{"tokenizer_class":"Qwen2TokenizerFast"}')
    if include_generation_config:
        generation_logical = link("generation_config.json", generation_config)
    else:
        generation_logical = None
    link("chat_template.jinja", b"{% for message in messages %}{{ message['content'] }}{% endfor %}")
    if include_processor_configs:
        link("preprocessor_config.json", b'{"processor":"image"}')
        link("video_preprocessor_config.json", b'{"processor":"video"}')
    return snapshot, {
        "first_weight": first_logical,
        "tokenizer": tokenizer_logical,
        "config": config_logical,
        "generation_config": generation_logical,
        "blobs": blobs,
    }


def test_entrypoints_are_executable_syntax_valid_and_do_not_operate_scheduler():
    assert LAUNCHER.stat().st_mode & 0o111
    assert SBATCH.stat().st_mode & 0o111
    assert WORKER.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(SBATCH)], check=False).returncode == 0
    assert subprocess.run(["bash", "-n", str(WORKER)], check=False).returncode == 0
    result = subprocess.run([sys.executable, str(LAUNCHER), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "smoke" in result.stdout
    assert "probe-gpu-topology" in result.stdout
    assert "seal-gpu-topology-probe" in result.stdout

    sbatch = SBATCH.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "subprocess.run([\"sbatch\"" not in launcher
    assert "scancel " not in sbatch
    assert "--dependency" not in sbatch
    assert "afterok" not in sbatch
    assert "#SBATCH --nodes=4" in sbatch
    assert "#SBATCH --gpus-per-node=4" in sbatch
    assert "#SBATCH --cpus-per-gpu=16" in sbatch
    assert "#SBATCH --time=06:00:00" in sbatch
    assert "#SBATCH --partition=" not in sbatch
    assert "#SBATCH --reservation=" not in sbatch
    assert "sbatch --reservation=interactive" in sbatch
    assert 'SLURM_JOB_RESERVATION:-' in sbatch
    assert '!= "interactive"' in sbatch
    assert "#SBATCH --ntasks=" not in sbatch
    assert "#SBATCH --ntasks-per-node=" not in sbatch
    assert "#SBATCH --cpus-per-task=" not in sbatch
    assert "--nodes=4 --ntasks=16 --ntasks-per-node=4 --gpus=16 --gpus-per-task=1 --cpus-per-task=16" in sbatch
    assert "--kill-on-bad-exit=0" in sbatch
    assert "CTM_AITA_NTAFLIP_SOURCE_DIR" in sbatch
    assert "ctm-rmct-aita-ntaflip-r006" in sbatch
    assert "16gpu-v3-r006" in sbatch
    assert "slurm-rmct-aita-ntaflip-r006-%j.out" in sbatch
    assert "0.3.258" in sbatch
    assert "2.11.0+cu129" in sbatch
    assert "transformers" in sbatch
    assert "peft" in sbatch
    assert "safetensors" in sbatch
    assert "CTM_DISABLE_CUDNN_SDP" in sbatch
    assert "CTM_DISABLE_CUDNN_SDP" in WORKER.read_text(encoding="utf-8")
    assert "CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP=1" in sbatch
    assert "CTM_AITA_R005_EOS_ONLY_NO_TOKEN_CAP" in WORKER.read_text(encoding="utf-8")
    assert 'for smoke_condition in base step016 step064 step176; do' in sbatch
    assert '"$launcher" smoke' in sbatch
    assert "--nodes=1 --ntasks=1 --gpus=1 --gpus-per-task=1 --cpus-per-task=16" in sbatch
    assert '"$launcher" probe-gpu-topology' in sbatch
    assert '"$launcher" seal-gpu-topology-probe' in sbatch
    topology_probe = sbatch.index('"$launcher" probe-gpu-topology')
    topology_seal = sbatch.index('"$launcher" seal-gpu-topology-probe')
    prepare = sbatch.index('"$python_bin" "$launcher" prepare')
    assert topology_probe < topology_seal < prepare
    assert sbatch.count("--gpu-bind=verbose,per_task:1") == 3
    assert "--distribution=block:block" in sbatch
    assert "srun --label --exclusive --exact --kill-on-bad-exit=1" in sbatch
    # The two srun shapes consume the already-checked batch allocation rather
    # than submitting a nested job or attempting to choose its reservation.
    assert "srun --reservation=" not in sbatch
    assert "vllm" not in sbatch.lower()
    assert "vllm" not in launcher.lower()
    assert "r002_step16_root" not in sbatch
    assert "r002_step64_root" not in sbatch
    assert "r004" not in sbatch


def test_coordinator_owns_all_compiler_cache_dirs_before_package_gate_or_prepare():
    sbatch = SBATCH.read_text(encoding="utf-8")
    cache_setup = sbatch.index('coordinator_tmp=$(mktemp -d "/tmp/ctm-aita-ntaflip-r006-coordinator-')
    version_gate = sbatch.index('"$python_bin" - <<\'PY\'')
    prepare = sbatch.index('"$python_bin" "$launcher" prepare')
    assert cache_setup < version_gate < prepare
    assert "import importlib.metadata" in sbatch
    assert "\nimport torch\n" not in sbatch
    assert "\nimport transformers\n" not in sbatch
    assert "\nimport peft\n" not in sbatch
    assert "\nimport safetensors\n" not in sbatch
    for variable, leaf in (
        ("TMPDIR", "tmp"),
        ("XDG_CACHE_HOME", "xdg-cache"),
        ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda-cache"),
    ):
        export = f'export {variable}="$coordinator_tmp/{leaf}"'
        assert export in sbatch
        assert sbatch.index(export) < version_gate
        assert f"${{{variable}:-" not in sbatch


def test_exact_four_condition_by_four_pair_preserving_cell_plan():
    module = _load_module()
    assert [(item.name, item.optimizer_step, item.source_kind) for item in module.CONDITIONS] == [
        ("base", None, "pinned-snapshot"),
        ("step016", 16, "raw-training-checkpoint"),
        ("step064", 64, "raw-training-checkpoint"),
        ("step176", 176, "raw-training-checkpoint"),
    ]
    cells = [(item.name, shard) for item in module.CONDITIONS for shard in range(module.SHARD_COUNT)]
    assert len(cells) == 16
    assert len(set(cells)) == 16
    assert module.EXPECTED_PAIRS == 1591
    assert module.PERSPECTIVES_PER_PAIR == 2
    assert module.GENERATIONS_PER_CONDITION == 3182
    assert module.CAMPAIGN_NAME.endswith("v3-r006")
    assert module.GPU_BINDING == "verbose,per_task:1"
    assert module.GPU_DISTRIBUTION == "block:block"
    assert module.SMOKE_STEP_GPU_COUNT == 1
    assert module.FULL_STEP_GPU_COUNT == module.GPU_TOPOLOGY_TASK_COUNT == 16
    assert all("r006" in schema for schema in (
        module.LAUNCH_SCHEMA,
        module.EVALUATION_SCHEMA,
        module.TASK_RECEIPT_SCHEMA,
        module.SMOKE_RECEIPT_SCHEMA,
        module.COMPLETION_SCHEMA,
        module.GPU_TOPOLOGY_RECORD_SCHEMA,
        module.GPU_TOPOLOGY_RECEIPT_SCHEMA,
    ))

    worker = WORKER.read_text(encoding="utf-8")
    assert "if (( $# != 6 )); then" in worker
    assert "ctm-aita-ntaflip-r006" in worker
    for rank, condition, shard in (
        (0, "base", 0),
        (3, "base", 3),
        (4, "step016", 0),
        (7, "step016", 3),
        (8, "step064", 0),
        (11, "step064", 3),
        (12, "step176", 0),
        (15, "step176", 3),
    ):
        assert f"{rank}) condition={condition};" in worker
        assert f"shard_index={shard}" in worker


def test_paper_decode_contract_and_task_commands_are_exact_and_judge_free(tmp_path: Path):
    module = _load_module()
    assert module.TASK_FACTORY == "experiments.elephant_aita_ntaflip.tasks:aita_nta_flip_shard"
    assert module.GENERATION_CONFIG == {
        "temperature": 0.6,
        "top_p": 0.9,
        "seed": 0,
        "top_k": 50,
    }
    assert module.CONCURRENCY_CONFIG == {"max_connections": 4}
    assert module.RUNTIME_GENERATION_CONFIG == {**module.GENERATION_CONFIG, **module.CONCURRENCY_CONFIG}
    assert module.HF_MODEL_ARGS == {
        "device": "cuda:0",
        "dtype": "bfloat16",
        "do_sample": True,
        "enable_thinking": False,
    }
    assert module.HF_LOCAL_MODEL_ARGS == {"provider": "hf", **module.HF_MODEL_ARGS}
    assert module.PROMPT_SUFFIX == "\nOutput only YTA or NTA."
    assert module.SYSTEM_PROMPT is None

    base = module._task_command(
        python="/unit/python",
        manifest=tmp_path / "manifest.json",
        runtime={
            "mode": "native-hf-base",
            "model": f"hf/{module.MODEL_SNAPSHOT}",
            "model_args": module.HF_MODEL_ARGS,
            "generation_config": module.RUNTIME_GENERATION_CONFIG,
            "qwen_thinking_policy": module.QWEN_THINKING_POLICY,
            "no_token_cap_policy": module.NO_TOKEN_CAP_POLICY,
            "no_token_cap_runtime_policy": module.NO_TOKEN_CAP_RUNTIME_POLICY,
        },
        attempt=tmp_path / "attempt-base",
        shard_index=2,
    )
    assert _option_value(base, "--model") == f"hf/{module.MODEL_SNAPSHOT}"
    assert "--local-checkpoint" not in base
    assert "--limit" not in base
    assert "--persistent-vllm-server" not in base
    assert "--isolate-tasks" not in base
    assert _option_value(base, "--task-factory") == module.TASK_FACTORY
    assert '"n_shards":4' in _option_value(base, "--task-args")
    assert '"shard_index":2' in _option_value(base, "--task-args")
    assert _option_value(base, "--model-args") == '{"device":"cuda:0","do_sample":true,"dtype":"bfloat16","enable_thinking":false}'
    assert _option_value(base, "--generation-config") == '{"max_connections":4,"seed":0,"temperature":0.6,"top_k":50,"top_p":0.9}'
    assert "--system-message" not in base

    trained_runtime = {
        "mode": "native-hf-peft",
        "checkpoint": "/sealed/raw/adapter-0001",
        "model_args": module.HF_LOCAL_MODEL_ARGS,
        "generation_config": module.RUNTIME_GENERATION_CONFIG,
        "qwen_thinking_policy": module.QWEN_THINKING_POLICY,
        "no_token_cap_policy": module.NO_TOKEN_CAP_POLICY,
        "no_token_cap_runtime_policy": module.NO_TOKEN_CAP_RUNTIME_POLICY,
    }
    trained = module._task_command(
        python="/unit/python",
        manifest=tmp_path / "manifest.json",
        runtime=trained_runtime,
        attempt=tmp_path / "attempt-trained",
        shard_index=0,
    )
    assert "--model" not in trained
    assert _option_value(trained, "--local-checkpoint") == "/sealed/raw/adapter-0001"
    assert _option_value(trained, "--base-model") == str(module.MODEL_SNAPSHOT)
    assert _option_value(trained, "--model-args") == '{"device":"cuda:0","do_sample":true,"dtype":"bfloat16","enable_thinking":false,"provider":"hf"}'
    assert "--limit" not in trained
    smoke = module._task_command(
        python="/unit/python",
        manifest=tmp_path / "manifest.json",
        runtime=trained_runtime,
        attempt=tmp_path / "attempt-smoke",
        shard_index=0,
        source_sample_count=module.DIRECT_VERDICT_SMOKE_SOURCE_SAMPLE_COUNT,
    )
    assert _option_value(smoke, "--limit") == "2"
    assert "max_tokens" not in " ".join(smoke)


def test_snapshot_identity_binds_index_shards_tokenizer_generation_and_template_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    snapshot, logical = _write_hf_snapshot(tmp_path, revision="unit-revision")
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", snapshot)

    identity = module._snapshot_identity()

    assert identity["path"] == str(snapshot)
    assert set(module.SNAPSHOT_REQUIRED_CONFIGURATION_FILES) <= set(identity["configuration_files"])
    assert "generation_config.json" in module.SNAPSHOT_OPTIONAL_CONFIGURATION_FILES
    assert "generation_config.json" in identity["configuration_files"]
    assert "chat_template.jinja" in identity["configuration_files"]
    assert "preprocessor_config.json" in identity["configuration_files"]
    assert "video_preprocessor_config.json" in identity["configuration_files"]
    assert identity["safetensors_index"]["logical_path"].endswith("model.safetensors.index.json")
    assert identity["indexed_weight_files"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert identity["weight_identity_policy"] == {
        "mode": "hf_content_address_and_size",
        "trust_boundary": "controlled_immutable_hf_blob_store",
        "full_shard_rehash_on_worker_replay": False,
    }
    assert set(identity["weight_files"]) == set(identity["indexed_weight_files"])
    first_weight = identity["weight_files"]["model-00001-of-00002.safetensors"]
    assert first_weight["logical_path"] == str(logical["first_weight"])
    assert first_weight["resolved_path"].startswith(str(logical["blobs"]))
    assert first_weight["content_address"] == Path(first_weight["resolved_path"]).name
    assert len(first_weight["content_address"]) == 64


def test_snapshot_identity_allows_missing_model_generation_config_but_binds_present_processors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    snapshot, _ = _write_hf_snapshot(
        tmp_path,
        revision="no-model-generation-config",
        include_generation_config=False,
    )
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", snapshot)

    identity = module._snapshot_identity()

    assert "generation_config.json" not in identity["configuration_files"]
    assert {"preprocessor_config.json", "video_preprocessor_config.json"} <= set(identity["configuration_files"])
    assert module.RUNTIME_GENERATION_CONFIG == {
        "temperature": 0.6,
        "top_p": 0.9,
        "seed": 0,
        "top_k": 50,
        "max_connections": 4,
    }


def test_snapshot_identity_rejects_a_generation_length_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    snapshot, _ = _write_hf_snapshot(
        tmp_path,
        revision="explicit-generation-bound",
        generation_config=b'{"max_new_tokens":1}',
    )
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", snapshot)

    with pytest.raises(module.EvaluationError, match="output-token cap"):
        module._snapshot_identity()


def test_no_token_cap_guards_reject_nested_camel_case_and_command_flags():
    module = _load_module()
    with pytest.raises(ValueError, match="output-token cap"):
        module.assert_no_token_cap_mapping(
            {"nested": {"maxOutputTokens": 1}}, label="unit runtime configuration"
        )
    with pytest.raises(ValueError, match="output-token cap"):
        module.assert_no_token_cap_mapping(
            {"nested": {"MAX_TOKENS": 1}}, label="unit runtime configuration"
        )
    with pytest.raises(module.EvaluationError, match="output-token cap flag"):
        module._assert_no_token_cap_command(["--max-new-tokens", "1"], label="unit command")


def test_snapshot_identity_detects_link_target_drift_and_rejects_blob_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    snapshot, logical = _write_hf_snapshot(tmp_path, revision="unit-revision")
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", snapshot)
    initial = module._snapshot_identity()

    # A changed direct shard link must alter the attested content-addressed
    # blob identity before any worker may load the snapshot.
    alternate_weight = b"alternate-model-shard"
    alternate_digest = hashlib.sha256(alternate_weight).hexdigest()
    alternate_blob = logical["blobs"] / alternate_digest
    alternate_blob.write_bytes(alternate_weight)
    logical["first_weight"].unlink()
    logical["first_weight"].symlink_to(Path("../../blobs") / alternate_digest)
    after_weight_drift = module._snapshot_identity()
    key = "model-00001-of-00002.safetensors"
    assert after_weight_drift["weight_files"][key]["content_address"] != initial["weight_files"][key]["content_address"]

    # Tokenizer bytes are fully hashed, so a swapped blob link also changes
    # their receipt identity rather than relying on a mutable provider default.
    alternate_tokenizer = b'{"version":"two"}'
    tokenizer_digest = hashlib.sha256(alternate_tokenizer).hexdigest()
    (logical["blobs"] / tokenizer_digest).write_bytes(alternate_tokenizer)
    logical["tokenizer"].unlink()
    logical["tokenizer"].symlink_to(Path("../../blobs") / tokenizer_digest)
    after_tokenizer_drift = module._snapshot_identity()
    assert (
        after_tokenizer_drift["configuration_files"]["tokenizer.json"]["sha256"]
        != after_weight_drift["configuration_files"]["tokenizer.json"]["sha256"]
    )

    external = tmp_path / "outside-config.json"
    external.write_bytes(b'{"outside":true}')
    logical["config"].unlink()
    logical["config"].symlink_to(external)
    with pytest.raises(module.EvaluationError, match="escapes expected root"):
        module._snapshot_identity()


def test_evaluator_versions_sampling_and_attention_policy_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()
    modules = {
        "inspect_ai": module.INSPECT_VERSION,
        "torch": module.TORCH_VERSION,
        "transformers": module.TRANSFORMERS_VERSION,
        "peft": module.PEFT_VERSION,
        "safetensors": module.SAFETENSORS_VERSION,
    }
    for name, version in modules.items():
        fake = ModuleType(name)
        fake.__version__ = version
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setenv("CTM_DISABLE_CUDNN_SDP", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv(module.NO_TOKEN_CAP_RUNTIME_ENV, module.NO_TOKEN_CAP_RUNTIME_ENV_VALUE)

    runtime = module._validate_evaluator_environment(require_one_gpu=True)

    assert runtime == {
        "backend": "native-hf-peft",
        "inspect_version": "0.3.258",
        "torch_version": "2.11.0+cu129",
        "transformers_version": "5.5.4",
        "peft_version": "0.20.0",
        "safetensors_version": "0.8.0",
        "attention_policy": {"ctm_disable_cudnn_sdp": "0"},
        "no_token_cap_runtime_policy": module.NO_TOKEN_CAP_RUNTIME_POLICY,
    }
    monkeypatch.setenv("CTM_DISABLE_CUDNN_SDP", "1")
    with pytest.raises(module.EvaluationError, match="CTM_DISABLE_CUDNN_SDP"):
        module._validate_evaluator_environment(require_one_gpu=True)
    monkeypatch.setenv("CTM_DISABLE_CUDNN_SDP", "0")
    sys.modules["peft"].__version__ = "0.0.0"
    with pytest.raises(module.EvaluationError, match="package versions differ"):
        module._validate_evaluator_environment(require_one_gpu=True)


def test_gpu_topology_probe_requires_one_visible_cuda_device_and_bounded_slurm_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    fake_torch = ModuleType("torch")
    device_count = [1]
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: device_count[0],
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        module,
        "_validate_evaluator_environment",
        lambda *, require_one_gpu: {"frozen_runtime": require_one_gpu},
    )
    monkeypatch.setattr(module, "_visible_gpu_uuid", lambda: "GPU-rank-zero")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_NODEID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "16")
    monkeypatch.setenv("SLURM_NNODES", "4")
    monkeypatch.setenv("SLURM_JOB_ID", "unit-job")
    monkeypatch.setenv("SLURM_STEP_ID", "unit-step")
    monkeypatch.setenv("SLURM_STEP_GPUS", "0")

    paths = module._campaign_paths(tmp_path / module.CAMPAIGN_NAME)
    result = module.gpu_topology_probe(campaign_root=paths.root)
    record = module._read_json(
        module._gpu_topology_record_path(paths, rank=0), label="unit GPU-topology record"
    )
    assert result["status"] == "written"
    assert record["cuda_visible_devices"] == "0"
    assert record["torch_cuda_device_count"] == 1
    assert record["nvidia_smi_uuid"] == "GPU-rank-zero"
    assert record["gpu_bind"] == "--gpu-bind=verbose,per_task:1"
    assert record["distribution"] == "--distribution=block:block"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(module.EvaluationError, match="exactly one Slurm-visible GPU"):
        module.gpu_topology_probe(campaign_root=paths.root)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(module.EvaluationError, match="exactly one Slurm-visible GPU"):
        module.gpu_topology_probe(campaign_root=paths.root)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    device_count[0] = 2
    with pytest.raises(module.EvaluationError, match="exactly one CUDA device"):
        module.gpu_topology_probe(campaign_root=paths.root)


def test_gpu_topology_receipt_requires_sixteen_distinct_uuids_in_four_rank_blocks(tmp_path: Path):
    module = _load_module()

    def write_records(paths, *, duplicate_uuid: bool = False) -> None:
        for rank in range(module.GPU_TOPOLOGY_TASK_COUNT):
            node_id = rank // 4
            uuid_rank = 0 if duplicate_uuid and rank == 1 else rank
            record = {
                "schema": module.GPU_TOPOLOGY_RECORD_SCHEMA,
                "campaign": module.CAMPAIGN_NAME,
                "gpu_bind": f"--gpu-bind={module.GPU_BINDING}",
                "distribution": f"--distribution={module.GPU_DISTRIBUTION}",
                "step_gpu_count": module.FULL_STEP_GPU_COUNT,
                "task_count": module.GPU_TOPOLOGY_TASK_COUNT,
                "node_count": 4,
                "rank": rank,
                "local_rank": rank % 4,
                "node_id": node_id,
                "hostname": f"node-{node_id}",
                "cuda_visible_devices": "0",
                "torch_cuda_device_count": 1,
                "nvidia_smi_uuid": f"GPU-{uuid_rank:032x}",
                "slurm": {"job_id": "unit-job", "step_id": "unit-step", "step_gpus": "0"},
                "runtime": {"frozen": True},
            }
            module._write_immutable_json(
                module._gpu_topology_record_path(paths, rank=rank),
                record,
                label=f"unit rank-{rank:03d}",
            )

    paths = module._campaign_paths(tmp_path / module.CAMPAIGN_NAME)
    write_records(paths)
    sealed = module.seal_gpu_topology_probe(campaign_root=paths.root)
    receipt_path = module._gpu_topology_receipt_path(paths)
    receipt = module._read_json(receipt_path, label="unit GPU-topology receipt")
    assert sealed["status"] == "written"
    assert receipt["schema"] == module.GPU_TOPOLOGY_RECEIPT_SCHEMA
    assert receipt["global_gpu_uuid_count"] == 16
    assert [node["ranks"] for node in receipt["topology"]] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert module._sealed_gpu_topology_receipt(paths) == receipt

    bad_paths = module._campaign_paths(tmp_path / "bad" / module.CAMPAIGN_NAME)
    write_records(bad_paths, duplicate_uuid=True)
    with pytest.raises(module.EvaluationError, match="four-distinct-GPU rank block"):
        module.seal_gpu_topology_probe(campaign_root=bad_paths.root)


def test_native_hf_no_cap_hook_preserves_absent_inspect_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    from experiments.elephant_aita_ntaflip import no_cap_hf

    inspect_ai = ModuleType("inspect_ai")
    inspect_model = ModuleType("inspect_ai.model")
    inspect_providers = ModuleType("inspect_ai.model._providers")
    inspect_hf = ModuleType("inspect_ai.model._providers.hf")

    class FakeHuggingFaceAPI:
        async def generate(self, *_args, **_kwargs):  # pragma: no cover - must be replaced by the hook
            raise AssertionError("standard provider generate must not run")

        def max_tokens(self):  # pragma: no cover - must be replaced by the hook
            return 2048

    inspect_hf.HuggingFaceAPI = FakeHuggingFaceAPI
    monkeypatch.setitem(sys.modules, "inspect_ai", inspect_ai)
    monkeypatch.setitem(sys.modules, "inspect_ai.model", inspect_model)
    monkeypatch.setitem(sys.modules, "inspect_ai.model._providers", inspect_providers)
    monkeypatch.setitem(sys.modules, "inspect_ai.model._providers.hf", inspect_hf)
    monkeypatch.setattr(
        no_cap_hf,
        "_installed_version",
        lambda distribution: {
            "inspect-ai": no_cap_hf.INSPECT_VERSION,
            "transformers": no_cap_hf.TRANSFORMERS_VERSION,
        }.get(distribution),
    )
    monkeypatch.setenv(no_cap_hf.RUNTIME_ENV, no_cap_hf.RUNTIME_ENV_VALUE)

    policy = no_cap_hf.install_native_hf_eos_only_sampling()
    assert policy == no_cap_hf.RUNTIME_POLICY
    assert FakeHuggingFaceAPI.max_tokens(object()) is None
    assert no_cap_hf.QWEN_THINKING_POLICY["enable_thinking"] is False

    observed: dict[str, object] = {}

    async def capture_no_cap_config(_self, _input, _tools, _tool_choice, config):
        observed["max_tokens"] = getattr(config, "max_tokens", "missing")
        return "eos-only-output"

    monkeypatch.setattr(no_cap_hf, "_eos_only_hf_generate", capture_no_cap_config)
    effective_config = SimpleNamespace(max_tokens=None)
    assert asyncio.run(FakeHuggingFaceAPI().generate([], [], None, effective_config)) == "eos-only-output"
    assert observed == {"max_tokens": None}


def test_eos_only_sampler_bypasses_model_generate_when_torch_is_available():
    from experiments.elephant_aita_ntaflip import no_cap_hf

    torch = pytest.importorskip("torch")

    class EOSModel:
        config = SimpleNamespace(eos_token_id=2)

        def __init__(self):
            self.calls = 0

        def generate(self, *_args, **_kwargs):
            raise AssertionError("Transformers model.generate must never be called")

        def __call__(self, **kwargs):
            self.calls += 1
            assert kwargs["use_cache"] is True
            return SimpleNamespace(
                logits=torch.tensor([[[0.0, 0.0, 100.0]]]),
                past_key_values=object(),
            )

    model = EOSModel()
    result = no_cap_hf._eos_only_model_generate(
        model,
        input_ids=torch.tensor([[1]]),
        attention_mask=torch.tensor([[1]]),
        tokenizer=SimpleNamespace(eos_token_id=2, pad_token_id=2),
        # The effective configuration must omit every generation-length field;
        # this succeeds only because termination is model-EOS-only.
        config={"temperature": 0.6, "top_p": 0.9, "top_k": 50},
        do_sample=True,
        return_dict_in_generate=True,
        output_logits=False,
        output_hidden_states=False,
    )
    assert result.sequences.tolist() == [[1, 2]]
    assert model.calls == 1


def test_metadata_contract_binds_manifest_shard_decode_and_no_system_prompt():
    module = _load_module()
    digest = "a" * 64
    assert module._metadata_expected(manifest_sha256=digest, shard_index=3) == {
        "benchmark": "elephant-aita-nta-flip",
        "manifest_sha256": digest,
        "shard_index": 3,
        "n_shards": 4,
        "schema": "elephant-aita-nta-flip-manifest-v2-r005",
        "prompt_suffix": "\nOutput only YTA or NTA.",
        "system_prompt": None,
        "qwen_thinking_policy": module.QWEN_THINKING_POLICY,
        "no_token_cap_policy": module.NO_TOKEN_CAP_POLICY,
        "no_token_cap_runtime_policy": module.NO_TOKEN_CAP_RUNTIME_POLICY,
        "sampling_config": {
            "temperature": 0.6,
            "top_p": 0.9,
            "seed": 0,
            "top_k": 50,
        },
        "concurrency_config": {"max_connections": 4},
        "generation_config": {
            "temperature": 0.6,
            "top_p": 0.9,
            "seed": 0,
            "top_k": 50,
            "max_connections": 4,
        },
    }


def test_completion_headline_requires_final_answer_primary_and_strict_secondary():
    module = _load_module()
    from experiments.elephant_aita_ntaflip import preflight

    # Use the real scorer's metric builder rather than a hand-written report.
    # The second pair has one malformed final, which exercises the nontrivial
    # response/pair coverage and conditional-denominator fields.
    metrics = preflight._metrics(
        [
            {
                "outcome": "nta_nta",
                "final_outcome": "nta_nta",
                "final_answer_only": {
                    "flipped_label": "NTA",
                    "flipped_status": "parsed",
                    "flipped_source": "direct_response",
                    "original_label": "NTA",
                    "original_status": "parsed",
                    "original_source": "direct_response",
                },
                "paper_first_five": {
                    "flipped_has_yta": False,
                    "flipped_has_nta": True,
                    "original_has_yta": False,
                    "original_has_nta": True,
                },
            },
            {
                "outcome": "invalid_or_ambiguous",
                "final_outcome": "invalid_or_unparsed",
                "final_answer_only": {
                    "flipped_label": None,
                    "flipped_status": "malformed_tail",
                    "flipped_source": None,
                    "original_label": "YTA",
                    "original_status": "parsed",
                    "original_source": "post_think_tail",
                },
                "paper_first_five": {
                    "flipped_has_yta": False,
                    "flipped_has_nta": False,
                    "original_has_yta": True,
                    "original_has_nta": False,
                },
            },
        ]
    )
    report = {"metrics": metrics}
    headline = module._preflight_headline(report)
    assert headline["primary_metric"]["name"] == "final_answer_only_both_nta"
    assert headline["primary_metric"]["value"] == {"count": 1, "rate": 0.5}
    assert headline["primary_metric"]["parsed_response_coverage"] == {
        "count": 3,
        "denominator": 4,
        "rate": 0.75,
    }
    assert headline["primary_metric"]["parsed_pair_coverage"] == {
        "count": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert headline["primary_metric"]["conditional_value_when_both_final_parsed"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
    }
    assert headline["secondary_metric"]["name"] == "strict_mutually_exclusive_nta_nta"

    report["metrics"]["primary_metric"] = dict(report["metrics"]["secondary_metric"])
    with pytest.raises(module.EvaluationError, match="final-answer-primary"):
        module._preflight_headline(report)


def test_launcher_constants_match_the_benchmark_owned_prepare_and_preflight_interfaces():
    module = _load_module()
    from experiments.elephant_aita_ntaflip import prepare as core_prepare
    from experiments.elephant_aita_ntaflip import preflight as core_preflight

    assert module.BENCHMARK == core_prepare.BENCHMARK
    assert module.EXPECTED_PAIRS == core_prepare.EXPECTED_PAIRS
    assert module.SHARD_COUNT == core_prepare.NUM_SHARDS
    assert module.OFFICIAL_MANIFEST_FILENAME == core_prepare.MANIFEST_FILENAME
    assert module.AITA_MANIFEST_SCHEMA == core_prepare.MANIFEST_SCHEMA
    assert module.PROMPT_SUFFIX == core_prepare.PROMPT_SUFFIX
    assert module.GENERATION_CONFIG == core_prepare.GENERATION_CONFIG
    assert module.CONCURRENCY_CONFIG == core_prepare.CONCURRENCY_CONFIG
    assert module.RUNTIME_GENERATION_CONFIG == core_prepare.RUNTIME_GENERATION_CONFIG
    assert module.FINAL_ANSWER_PARSER_SCHEMA == core_preflight.PARSER_SCHEMA
    assert callable(core_preflight.preflight_raw_logs)
    assert callable(core_preflight.validate_preflight_report)
    assert callable(core_preflight.write_preflight_report)


def test_campaign_root_is_a_fresh_named_isolation_boundary(tmp_path: Path):
    module = _load_module()
    paths = module._campaign_paths(tmp_path / module.CAMPAIGN_NAME)
    assert paths.root.name == module.CAMPAIGN_NAME
    assert paths.manifest == paths.root / "input" / "aita-nta-flip.manifest.json"
    assert paths.completion == paths.root / "completion.json"
    with pytest.raises(module.EvaluationError, match="named exactly"):
        module._campaign_paths(tmp_path / "r006")


def test_early_checkpoint_source_replays_raw_resumability_custody_without_prior_eval_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_module()
    files = {}
    for name in (
        "adapter_config",
        "adapter_model",
        "optimizer",
        "manifest",
        "replicated_training_manifest",
        "replicated_training_rng",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("utf-8"))
        files[name] = module._identity(path, label=name)
    receipts = {}
    for name in ("checkpoint", "completion", "decision"):
        path = tmp_path / f"{name}.json"
        path.write_text('{"sealed":true}\n', encoding="utf-8")
        receipts[name] = module._identity(path, label=name)
    checkpoint = {
        "path": str(tmp_path),
        "optimizer_step": 16,
        "full_resumability_required": True,
        "files": files,
    }
    seen: list[tuple[Path, int]] = []

    def validate(repository: Path, *, step: int):
        seen.append((repository, step))
        return {
            "target": {"step": 16, "condition": "approved-step016"},
            "checkpoint": checkpoint,
            "checkpoint_receipt": receipts["checkpoint"],
            "completion_receipt": receipts["completion"],
            "decision_receipt": receipts["decision"],
        }

    fake_r002 = type("RawCustody", (), {"validate_approved_target": staticmethod(validate)})
    monkeypatch.setattr(module, "_r002_module", lambda: fake_r002)
    replayed = module._revalidate_early_checkpoint_source(training_repository=tmp_path, step=16)
    assert seen == [(tmp_path, 16)]
    assert replayed["source"] == "raw-training-checkpoint"
    assert replayed["checkpoint"] == checkpoint
    assert "evaluation_receipt" not in replayed
    assert "compatibility_adapter" not in repr(replayed)


def test_canonical_raw_guard_rejects_unclaimed_or_wrong_task_eval_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    condition = module._condition("base")
    paths = module.ConditionPaths(
        root=tmp_path / "base",
        evaluation_receipt=tmp_path / "base" / "evaluation-receipt.json",
        raw=tmp_path / "base" / "raw",
        attempts=tmp_path / "base" / "attempts",
        receipts=tmp_path / "base" / "receipts",
        preflight=tmp_path / "base" / "preflight.json",
    )
    expected: dict[int, dict[str, object]] = {}
    for shard in range(module.SHARD_COUNT):
        path = paths.raw / f"shard-{shard:03d}" / f"sealed-{shard}.eval"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sealed")
        expected[shard] = {"canonical_log": module._identity(path, label="canonical")}

    def load(**kwargs):
        return expected[kwargs["shard_index"]]

    monkeypatch.setattr(module, "_load_task_receipt", load)
    sealed = module._validate_canonical_raw_custody(
        condition_paths=paths,
        condition=condition,
        launch_sha256="a" * 64,
        evaluation_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )
    assert sealed == [expected[index] for index in range(module.SHARD_COUNT)]

    stray = paths.raw / "wrong-task.eval"
    stray.write_bytes(b"preserved but unclaimed")
    with pytest.raises(module.EvaluationError, match="differs from its four task receipts"):
        module._validate_canonical_raw_custody(
            condition_paths=paths,
            condition=condition,
            launch_sha256="a" * 64,
            evaluation_sha256="b" * 64,
            manifest_sha256="c" * 64,
        )


def test_terminal_custody_identities_may_be_relative_to_the_training_repository(tmp_path: Path):
    module = _load_module()
    checkpoint_file = tmp_path / "run" / "checkpoint" / "optimizer.pt"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"optimizer")
    absolute = module._identity(checkpoint_file, label="optimizer")
    relative = {**absolute, "path": str(checkpoint_file.relative_to(tmp_path))}
    assert module._identity_record(relative, label="optimizer", root=tmp_path) == absolute
