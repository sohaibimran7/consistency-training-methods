"""CPU-only contracts for the RMCT-256 runtime-policy receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infra.isambard import rmct256_convergence_segment_contract as contract
from infra.isambard import rmct256_runtime_policy_receipt as runtime_policy


class _FakeSamplingParams:
    """Small vLLM-shaped value object; no model or vLLM installation needed."""

    def __init__(
        self,
        *,
        n: int = 1,
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        stop_token_ids: list[int] | None = None,
        ignore_eos: bool = False,
        logprobs: int | None = None,
    ):
        self.n = n
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty
        self.stop = None
        self.stop_token_ids = stop_token_ids
        self.ignore_eos = ignore_eos
        self.logprobs = logprobs
        self.best_of = n
        self._all_stop_token_ids = set(stop_token_ids or [])


def test_sampling_record_uses_real_rmct_constructor_and_retains_distributional_defaults():
    record = runtime_policy._sampling_params_record(_FakeSamplingParams, stop_token_ids=[151645])

    assert record["constructor"] == {
        "n": 96,
        "max_tokens": 20480,
        "temperature": 1.0,
        "ignore_eos": False,
        "logprobs": 0,
        "stop_token_ids": [151645],
    }
    assert record["effective"]["n"] == 96
    assert record["effective"]["max_tokens"] == 20480
    assert record["effective"]["temperature"] == 1.0
    assert record["effective"]["stop_token_ids"] == [151645]
    assert record["effective"]["ignore_eos"] is False
    assert record["effective"]["logprobs"] == 0
    assert record["effective"]["top_p"] == record["baseline_defaults"]["top_p"] == 1.0
    assert record["effective"]["top_k"] == record["baseline_defaults"]["top_k"] == 0
    assert record["effective"]["min_p"] == record["baseline_defaults"]["min_p"] == 0.0
    assert record["effective"]["presence_penalty"] == 0.0
    assert record["effective"]["frequency_penalty"] == 0.0
    assert record["effective"]["repetition_penalty"] == 1.0
    # `best_of` and the private stop-ID set are intentionally represented in
    # the full public snapshot, so a vLLM release cannot change an implicit
    # distributional constraint without changing the receipt.
    assert "best_of" in record["effective"]
    assert "_all_stop_token_ids" in record["effective"]


def test_sampling_record_fails_if_rmct_constructor_changes_a_default_penalty():
    class WrongPenalty(_FakeSamplingParams):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if self.n == 96:
                self.presence_penalty = 0.25

    with pytest.raises(runtime_policy.RuntimePolicyError, match="presence_penalty"):
        runtime_policy._sampling_params_record(WrongPenalty, stop_token_ids=[])


def test_worker_dtype_receipt_requires_the_explicit_bfloat16_vllm_option():
    class Config:
        torch_dtype = "torch.bfloat16"
        _attn_implementation = "sdpa"

    class Tokenizer:
        eos_token_id = 151645

    worker_dtype, attention = runtime_policy._worker_dtype_and_attention(transformers=object(), torch_module=object(), config=Config(), tokenizer=Tokenizer())

    assert worker_dtype["model_config_torch_dtype"] == "bfloat16"
    assert worker_dtype["vllm_dtype_argument"] == "bfloat16"
    assert worker_dtype["resolved_worker_dtype"] == "bfloat16"
    assert worker_dtype["assertion"] == "explicit_local_dtype_bfloat16"
    assert attention == {"implementation": "sdpa", "resolution": "AutoConfig"}


def test_runtime_versions_accept_the_pinned_vllm_release_with_a_recorded_cuda_build_suffix(
    monkeypatch: pytest.MonkeyPatch,
):
    """The official Arm wheel may label itself `0.21.0+cu129`."""

    class Vllm:
        __version__ = "0.21.0+cu129"

    class Transformers:
        __version__ = "5.5.4"

    class Torch:
        __version__ = "2.10.0+cu129"
        version = SimpleNamespace(cuda="12.9")

    class Peft:
        __version__ = "0.19.0"

    installed = {
        "vllm": "0.21.0+cu129",
        "transformers": "5.5.4",
        "torch": "2.10.0+cu129",
        "peft": "0.19.0",
    }
    monkeypatch.setattr(runtime_policy.importlib.metadata, "version", installed.__getitem__)

    record = runtime_policy._runtime_versions(
        {"vllm": Vllm, "transformers": Transformers, "torch": Torch, "peft": Peft},
        runtime={"vllm_version": "0.21.0", "transformers_version": "5.5.4", "vllm_cuda": "12.9"},
    )

    assert record["vllm"]["distribution"] == "0.21.0+cu129"
    assert record["vllm"]["module"] == "0.21.0+cu129"
    assert record["vllm_distribution_release"] == "0.21.0"
    assert record["vllm_module_release"] == "0.21.0"


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _fake_document(root: Path, plan: Path, segment: contract.Segment, *, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema": runtime_policy.SCHEMA,
        "scope": {
            "condition": contract.CONDITION,
            "segment": {
                "global_segment_index": segment.global_index,
                "target": segment.target,
                "run_name": segment.run_name,
                "worker_seed_base": segment.worker_seed_base,
            },
            "plan": _identity(plan),
            "compiled_target_args_sha256": "a" * 64,
        },
        "runtime_policy": payload,
        "runtime_policy_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def test_capture_is_immutable_and_validate_recomputes_the_runtime_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    root.mkdir()
    plan = root / "plan.yaml"
    plan.write_text("name: fixture\n", encoding="utf-8")
    output = root / "logs" / "runtime-policy.json"
    segment = contract.segment_for_index(0)
    state = {"value": 1}

    def fake_build(
        repository: Path,
        plan_path: Path,
        requested_segment: contract.Segment,
        *,
        reference_receipt: Path | None = None,
    ) -> dict[str, object]:
        assert repository == root.resolve()
        assert plan_path == plan.resolve()
        assert requested_segment == segment
        assert reference_receipt is None
        return _fake_document(root.resolve(), plan.resolve(), segment, payload={"value": state["value"]})

    monkeypatch.setattr(runtime_policy, "build_runtime_policy_document", fake_build)

    first = runtime_policy.capture_runtime_policy_receipt(root, plan, segment, output=output)
    assert first["status"] == "written"
    assert output.is_file()
    assert runtime_policy.validate_runtime_policy_receipt(root, output, expected_segment=segment)["runtime_policy"] == {"value": 1}
    resumed = runtime_policy.capture_runtime_policy_receipt(root, plan, segment, output=output)
    assert resumed["status"] == "resumed"

    state["value"] = 2
    with pytest.raises(runtime_policy.RuntimePolicyError, match="no longer matches"):
        runtime_policy.validate_runtime_policy_receipt(root, output, expected_segment=segment)


def test_receipt_rejects_tampered_runtime_policy_hash_before_a_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    root.mkdir()
    plan = root / "plan.yaml"
    plan.write_text("name: fixture\n", encoding="utf-8")
    output = root / "receipt.json"
    segment = contract.segment_for_index(1)
    document = _fake_document(root.resolve(), plan.resolve(), segment, payload={"value": 1})
    document["runtime_policy_sha256"] = "0" * 64
    output.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")

    def fake_build(*_args, **_kwargs):
        return document

    monkeypatch.setattr(runtime_policy, "build_runtime_policy_document", fake_build)
    with pytest.raises(runtime_policy.RuntimePolicyError, match="payload hash is invalid"):
        runtime_policy.validate_runtime_policy_receipt(root, output, expected_segment=segment)


def test_source_manifest_rejects_imported_module_shadowing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    root.mkdir()
    expected = root / "fixture.py"
    expected.write_text("VALUE = 1\n", encoding="utf-8")
    shadow = tmp_path / "shadow.py"
    shadow.write_text("VALUE = 2\n", encoding="utf-8")

    class Imported:
        __file__ = str(shadow)

    monkeypatch.setattr(runtime_policy, "METHOD_SOURCE_MODULES", (("fixture_module", "fixture.py"),))
    monkeypatch.setattr(runtime_policy.importlib, "import_module", lambda name: Imported())

    with pytest.raises(runtime_policy.RuntimePolicyError, match="PYTHONPATH shadowing"):
        runtime_policy._method_source_manifest(root)
