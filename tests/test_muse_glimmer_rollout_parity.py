"""CPU-only tests for the Muse Glimmer rollout-worker parity receipt."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ctm.backends.local.muse_glimmer import (
    MODEL_ID,
    MODEL_REVISION,
    PARITY_ATTESTATION_SCHEMA,
    TRANSFORMERS_VERSION,
    VLLM_COMMIT,
    is_muse_glimmer_model_name,
    validate_muse_rollout_worker_parity_attestation,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _effect(gate_kind: str = "policy_minus_base") -> dict:
    return {
        "gate_kind": gate_kind,
        "passed": True,
        "count": 8,
        "hf_l2": 2.0,
        "vllm_l2": 2.0001,
        "difference_l2": 0.001,
        "relative_to_hf_l2_error": 0.0005,
        "vllm_to_hf_l2_ratio": 1.00005,
        "cosine_similarity": 0.99999,
        "pearson_r": 0.99999,
        "max_abs_error": 0.0001,
    }


def _fixture(tmp_path: Path):
    snapshot = tmp_path / MODEL_REVISION
    snapshot.mkdir()
    config = snapshot / "config.json"
    config.write_text(json.dumps({"model_type": "muse_glimmer"}) + "\n")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_model = adapter_dir / "adapter_model.safetensors"
    adapter_config.write_text('{"r": 8}\n')
    adapter_model.write_bytes(b"nonzero-muse-lora")
    data = tmp_path / "train.jsonl"
    manifest = tmp_path / "train.manifest.json"
    source = tmp_path / "source.py"
    data.write_text('{"id": 1}\n')
    manifest.write_text('{"schema": "test"}\n')
    source.write_text("VALUE = 1\n")
    runtime_receipt = tmp_path / "runtime.json"
    pip_freeze = tmp_path / "pip-freeze.txt"
    model_snapshot_receipt = tmp_path / "model-snapshot.json"
    runtime_receipt.write_text('{"schema": "test"}\n')
    pip_freeze.write_text("vllm==test\n")
    model_snapshot_receipt.write_text('{"revision": "test"}\n')
    source_record = {"relative_path": "source.py", **_identity(source)}
    source_compact = [
        {
            "relative_path": source_record["relative_path"],
            "sha256": source_record["sha256"],
            "size_bytes": source_record["size_bytes"],
        }
    ]
    gpus = [
        {"logical_index": 1, "device_token": "GPU-1"},
        {"logical_index": 2, "device_token": "GPU-2"},
        {"logical_index": 3, "device_token": "GPU-3"},
    ]
    options = {
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.9,
        "language_model_only": True,
        "logprobs_mode": "processed_logprobs",
        "max_model_len": 131072,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 128,
        "seed": 42,
        "tensor_parallel_size": 1,
    }
    document = {
        "schema": PARITY_ATTESTATION_SCHEMA,
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_path": str(snapshot.resolve()),
            "config_sha256": _sha(config),
        },
        "runtime": {
            "transformers_version": TRANSFORMERS_VERSION,
            "vllm_commit": VLLM_COMMIT,
            "torch_version": "2.10.0+cu129",
            "platform": "Linux-aarch64-GH200",
            "receipts": {
                "runtime": _identity(runtime_receipt),
                "pip_freeze": _identity(pip_freeze),
                "model_snapshot": _identity(model_snapshot_receipt),
            },
        },
        "worker_gpus": gpus,
        "worker_engine_kwargs": options,
        "adapter": {
            "adapter_config": _identity(adapter_config),
            "adapter_model": _identity(adapter_model),
            "nonzero_tensor_count": 2,
        },
        "probe": {
            "kind": "vllm-prompt-logprobs-uncapped-eos-tail-v1",
            "max_tokens": None,
            "termination": "eos_only",
            "source_prompt_count": 6,
            "score_row_count": 9,
            "anchor_source_index": 1,
            "source_indices": [1, 1, 1, 0, 1, 2, 3, 4, 5],
            "request_count": 18,
            "non_eos_termination_count": 0,
            "score_inputs_sha256": "a" * 64,
        },
        "scientific_contract": {
            "frozen_spec_sha256": "b" * 64,
            "data": _identity(data),
            "manifest": _identity(manifest),
            "generation": {
                "max_tokens": None,
                "termination": "eos_only",
                "non_eos_termination_policy": "fail_run",
            },
        },
        "source_manifest": {
            "schema": "muse-glimmer-source-manifest-v1",
            "sha256": hashlib.sha256(
                (json.dumps(source_compact, indent=2, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "files": [source_record],
        },
        "aggregate_base_parity": _effect("raw_logprob"),
        "aggregate_policy_parity": _effect("raw_logprob"),
        "aggregate_effect_parity": _effect(),
        "per_worker_effect_parity": [
            {"worker_index": index, "worker_gpu": gpu, "effect_parity": _effect()}
            for index, gpu in enumerate(gpus)
        ],
    }
    receipt = tmp_path / "attestation.json"
    receipt.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return snapshot, gpus, options, document, receipt


def test_muse_model_detection_uses_official_id_or_local_config(tmp_path):
    snapshot, *_ = _fixture(tmp_path)

    assert is_muse_glimmer_model_name(MODEL_ID)
    assert is_muse_glimmer_model_name(snapshot)
    assert not is_muse_glimmer_model_name("Qwen/Qwen3.5-9B")


def test_valid_receipt_binds_snapshot_runtime_topology_adapter_and_uncapped_probe(tmp_path):
    snapshot, gpus, options, document, receipt = _fixture(tmp_path)

    validated = validate_muse_rollout_worker_parity_attestation(
        receipt,
        expected_model=snapshot,
        expected_worker_gpus=gpus,
        expected_worker_engine_kwargs=options,
    )

    assert validated == document
    assert validated["probe"]["max_tokens"] is None


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("probe", "max_tokens", 1), "uncapped EOS-only"),
        (("probe", "non_eos_termination_count", 1), "uncapped EOS-only"),
        (("scientific_contract", "generation", {"max_tokens": 1}), "uncapped EOS-only"),
        (("runtime", "vllm_commit", "0" * 40), "pinned Transformers/vLLM"),
        (("aggregate_effect_parity", "hf_l2", 0.0), "non-zero LoRA effect"),
        (("aggregate_effect_parity", "cosine_similarity", 0.89), "direction gate"),
    ],
)
def test_receipt_fails_closed_on_unsafe_or_non_parity_evidence(tmp_path, mutation, message):
    snapshot, gpus, options, document, receipt = _fixture(tmp_path)
    section, key, value = mutation
    changed = copy.deepcopy(document)
    changed[section][key] = value
    receipt.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=message):
        validate_muse_rollout_worker_parity_attestation(
            receipt,
            expected_model=snapshot,
            expected_worker_gpus=gpus,
            expected_worker_engine_kwargs=options,
        )


def test_receipt_rebinds_adapter_bytes(tmp_path):
    snapshot, gpus, options, document, receipt = _fixture(tmp_path)
    adapter_model = Path(document["adapter"]["adapter_model"]["path"])
    adapter_model.write_bytes(b"mutated")

    with pytest.raises(ValueError, match="bytes differ"):
        validate_muse_rollout_worker_parity_attestation(
            receipt,
            expected_model=snapshot,
            expected_worker_gpus=gpus,
            expected_worker_engine_kwargs=options,
        )
