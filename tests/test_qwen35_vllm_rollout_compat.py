"""CPU contracts for dynamic Qwen3.5 on-policy vLLM adapter publication."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from ctm.backends.local.qwen35_vllm_compat import (
    DESTINATION_PREFIX,
    MANIFEST_NAME,
    SOURCE_PREFIX,
    WORKER_PARITY_ATTESTATION_NAME,
    file_sha256,
    is_qwen35_model_name,
    materialize_qwen35_vllm_rollout_compat_adapter,
    qwen35_rollout_adapter_has_nonzero_lora_effect,
    validate_qwen35_rollout_worker_parity_attestation,
    validate_qwen35_vllm_rollout_compat_adapter,
    write_qwen35_rollout_worker_parity_attestation,
)
from ctm.backends.local.rollout_workers import RolloutGPU, RolloutParallelBackend
from ctm.backends.local.vllm_sampler import VLLMSampler

MODEL = "Qwen/Qwen3.5-9B"


@pytest.mark.parametrize("model_type", ("qwen3_5", "qwen3_5_text", "qwen3_5_moe"))
def test_is_qwen35_model_name_detects_relocated_local_snapshot(tmp_path, model_type):
    snapshot = tmp_path / "models" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")

    assert is_qwen35_model_name(str(snapshot)) is True


def test_is_qwen35_model_name_preserves_name_detection_and_ignores_non_qwen_local_config(tmp_path):
    snapshot = tmp_path / "models" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

    assert is_qwen35_model_name("Qwen/Qwen3_5-9B") is True
    assert is_qwen35_model_name(str(snapshot)) is False


def _write_raw_adapter(path: Path, *, b_value: float = 0.25) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": MODEL}), encoding="utf-8")
    save_file(
        {
            SOURCE_PREFIX + "0.self_attn.q_proj.lora_A.weight": torch.tensor([[1.0, -2.0]]),
            SOURCE_PREFIX + "0.self_attn.q_proj.lora_B.weight": torch.full((2, 1), b_value),
            SOURCE_PREFIX + "1.linear_attn.in_proj_qkv.lora_A.weight": torch.tensor([[0.5, 3.0]]),
            SOURCE_PREFIX + "1.linear_attn.in_proj_qkv.lora_B.weight": torch.full((2, 1), -b_value),
        },
        str(path / "adapter_model.safetensors"),
    )


def _compat_pair(tmp_path: Path, *, version: int = 1, b_value: float = 0.25) -> tuple[Path, Path]:
    root = tmp_path / f"v{version:08d}"
    raw = root / "raw"
    compat = root / "vllm_compat"
    _write_raw_adapter(raw, b_value=b_value)
    materialize_qwen35_vllm_rollout_compat_adapter(raw, compat, model=MODEL, adapter_version=version)
    return raw, compat


def test_compatibility_snapshot_preserves_raw_bytes_and_translates_every_text_key(tmp_path):
    raw, compat = _compat_pair(tmp_path)

    raw_hash = file_sha256(raw / "adapter_model.safetensors")
    manifest = validate_qwen35_vllm_rollout_compat_adapter(compat, expected_version=1, strict_tensor_content=True)
    raw_tensors = load_file(str(raw / "adapter_model.safetensors"))
    compat_tensors = load_file(str(compat / "adapter_model.safetensors"))

    assert file_sha256(raw / "adapter_model.safetensors") == raw_hash
    assert set(compat_tensors) == {DESTINATION_PREFIX + key.removeprefix(SOURCE_PREFIX) for key in raw_tensors}
    assert manifest["source_adapter"]["relative_path"] == "../raw"
    assert manifest["destination_adapter"]["relative_path"] == "."
    assert (compat / MANIFEST_NAME).is_file()
    assert qwen35_rollout_adapter_has_nonzero_lora_effect(compat) is True


def test_compatibility_snapshot_rejects_post_publication_weight_mutation(tmp_path):
    _raw, compat = _compat_pair(tmp_path)
    tensors = load_file(str(compat / "adapter_model.safetensors"))
    first = next(iter(tensors))
    tensors[first] = tensors[first] + 1
    save_file(tensors, str(compat / "adapter_model.safetensors"))

    with pytest.raises(ValueError, match="weight hash mismatch"):
        validate_qwen35_vllm_rollout_compat_adapter(compat)


class _FakeAPI:
    class LoRARequest:
        def __init__(self, name, lora_int_id, path):
            self.name, self.lora_int_id, self.path = name, lora_int_id, path


def test_vllm_sampler_allows_only_validated_qwen35_dynamic_snapshot(tmp_path):
    raw, compat = _compat_pair(tmp_path, version=7)
    sampler = VLLMSampler(MODEL, engine=object(), api=SimpleNamespace(LoRARequest=_FakeAPI.LoRARequest))

    with pytest.raises(ValueError, match="maps no `model.layers"):
        sampler.advance_policy(str(raw), version=7)

    sampler.advance_policy(str(compat), version=7)
    request = sampler._policy_lora_request()
    assert (request.lora_int_id, request.path) == (7, str(compat))


def test_vllm_sampler_rejects_raw_adapter_for_relocated_qwen35_snapshot(tmp_path):
    snapshot = tmp_path / "models" / "pinned"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}), encoding="utf-8")
    raw = tmp_path / "raw"
    _write_raw_adapter(raw)
    sampler = VLLMSampler(str(snapshot), engine=object(), api=SimpleNamespace(LoRARequest=_FakeAPI.LoRARequest))

    with pytest.raises(ValueError, match="maps no `model.layers"):
        sampler.advance_policy(str(raw), version=7)


class _FakePolicyModel:
    def save_pretrained(self, directory: str) -> None:
        _write_raw_adapter(Path(directory))


class _FakeTrainingBackend:
    renderer_source = "hf"
    sampler = "vllm"
    use_lora = True
    vllm_options = {}

    def __init__(self) -> None:
        self.model = _FakePolicyModel()
        self.shutdown_calls = 0

    def setup(self, **_kwargs) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _passing_effect() -> dict[str, object]:
    return {
        "worker_v2_minus_base": {"max_abs_difference": 0.25},
        "coordinator_updated_minus_base": {"max_abs_difference": 0.24},
        "cosine_similarity": 0.999,
    }


def _write_worker_parity_attestation(
    path: Path,
    *,
    raw: Path,
    compat: Path,
    version: int,
    worker_gpus: list[dict[str, object]],
    worker_engine_kwargs: dict[str, object],
) -> None:
    effect = _passing_effect()
    write_qwen35_rollout_worker_parity_attestation(
        path,
        model=MODEL,
        raw_adapter=raw,
        vllm_adapter=compat,
        adapter_version=version,
        worker_gpus=worker_gpus,
        worker_engine_kwargs=worker_engine_kwargs,
        fixed_token_probe={
            "kind": "post-update-worker-score-completions-v1",
            "score_rows": len(worker_gpus),
            "completion_token_count": len(worker_gpus),
            "score_inputs_sha256": "a" * 64,
        },
        aggregate_effect_parity=effect,
        per_worker_effect_parity=[
            {
                "worker_index": index,
                "worker_gpu": gpu,
                "effect_parity": effect,
            }
            for index, gpu in enumerate(worker_gpus)
        ],
    )


class _FakePool:
    def __init__(self, status_dir: Path) -> None:
        self.status_dir = status_dir
        self.worker_count = 1
        self.gpus = (RolloutGPU(1, "GPU-1"),)
        self.engine_kwargs = {
            "gpu_memory_utilization": 0.75,
            "logprobs_mode": "processed_logprobs",
            "tensor_parallel_size": 1,
        }
        self.published: list[tuple[str, int]] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    def publish_adapter_sync(self, adapter_path: str | Path, *, version: int) -> None:
        self.published.append((str(adapter_path), version))

    async def publish_adapter(self, adapter_path: str | Path, *, version: int) -> None:
        self.published.append((str(adapter_path), version))

    def shutdown(self) -> None:
        return None


def test_parallel_backend_publishes_only_translated_dynamic_qwen35_paths(tmp_path):
    training = _FakeTrainingBackend()
    pool = _FakePool(tmp_path / "workers")
    raw, compat = _compat_pair(tmp_path / "preflight", version=2)
    _write_worker_parity_attestation(
        pool.status_dir / WORKER_PARITY_ATTESTATION_NAME,
        raw=raw,
        compat=compat,
        version=2,
        worker_gpus=[gpu.as_dict() for gpu in pool.gpus],
        worker_engine_kwargs=pool.engine_kwargs,
    )
    backend = RolloutParallelBackend(training, gpus=(), status_dir=pool.status_dir, pool=pool)

    backend.setup(model=MODEL, lora=object())
    asyncio.run(backend.refresh_policy_sampler("after-update"))

    assert [version for _path, version in pool.published] == [1, 2]
    for path_text, version in pool.published:
        published = Path(path_text)
        assert published.name == "vllm_compat"
        validate_qwen35_vllm_rollout_compat_adapter(published, expected_version=version, strict_tensor_content=True)
        root_manifest = json.loads((published.parent / "ctm_rollout_adapter.json").read_text())
        assert root_manifest["vllm_adapter_relative_path"] == "vllm_compat"
        assert root_manifest["qwen35_vllm_compatibility"]["raw_adapter_relative_path"] == "raw"
        preflight = root_manifest["qwen35_vllm_compatibility"]["worker_parity_preflight"]
        assert preflight["snapshot_raw_adapter_model_sha256"] == file_sha256(
            published.parent / "raw" / "adapter_model.safetensors"
        )
        assert preflight["snapshot_vllm_adapter_model_sha256"] == file_sha256(published / "adapter_model.safetensors")
    assert all((Path(path).parent / "raw" / "adapter_model.safetensors").is_file() for path, _ in pool.published)
    backend.shutdown()
    assert training.shutdown_calls == 1


def test_qwen35_rollout_workers_fail_closed_without_parity_attestation(tmp_path):
    training = _FakeTrainingBackend()
    pool = _FakePool(tmp_path / "workers")
    backend = RolloutParallelBackend(training, gpus=(), status_dir=pool.status_dir, pool=pool)

    with pytest.raises(FileNotFoundError, match="parity attestation is missing"):
        backend.setup(model=MODEL, lora=object())

    assert training.shutdown_calls == 1


def test_parallel_backend_revalidates_preflight_before_each_qwen_snapshot(tmp_path):
    training = _FakeTrainingBackend()
    pool = _FakePool(tmp_path / "workers")
    raw, compat = _compat_pair(tmp_path / "preflight", version=2)
    _write_worker_parity_attestation(
        pool.status_dir / WORKER_PARITY_ATTESTATION_NAME,
        raw=raw,
        compat=compat,
        version=2,
        worker_gpus=[gpu.as_dict() for gpu in pool.gpus],
        worker_engine_kwargs=pool.engine_kwargs,
    )
    backend = RolloutParallelBackend(training, gpus=(), status_dir=pool.status_dir, pool=pool)
    backend.setup(model=MODEL, lora=object())

    tensors = load_file(str(raw / "adapter_model.safetensors"))
    first = next(iter(tensors))
    tensors[first] = tensors[first] + 1
    save_file(tensors, str(raw / "adapter_model.safetensors"))

    with pytest.raises(ValueError, match="source adapter weight hash mismatch"):
        asyncio.run(backend.refresh_policy_sampler("after-preflight-mutation"))

    assert [version for _path, version in pool.published] == [1]
    backend.shutdown()


def test_worker_parity_attestation_rejects_mutated_preflight_adapter(tmp_path):
    raw, compat = _compat_pair(tmp_path / "preflight", version=2)
    report = tmp_path / WORKER_PARITY_ATTESTATION_NAME
    worker_gpus = [{"logical_index": 1, "device_token": "GPU-1"}]
    engine_kwargs = {
        "gpu_memory_utilization": 0.75,
        "logprobs_mode": "processed_logprobs",
        "tensor_parallel_size": 1,
    }
    _write_worker_parity_attestation(
        report,
        raw=raw,
        compat=compat,
        version=2,
        worker_gpus=worker_gpus,
        worker_engine_kwargs=engine_kwargs,
    )
    validate_qwen35_rollout_worker_parity_attestation(
        report,
        expected_model=MODEL,
        expected_worker_gpus=worker_gpus,
        expected_worker_engine_kwargs=engine_kwargs,
    )

    tensors = load_file(str(raw / "adapter_model.safetensors"))
    first = next(iter(tensors))
    tensors[first] = tensors[first] + 1
    save_file(tensors, str(raw / "adapter_model.safetensors"))

    with pytest.raises(ValueError, match="source adapter weight hash mismatch"):
        validate_qwen35_rollout_worker_parity_attestation(report)
