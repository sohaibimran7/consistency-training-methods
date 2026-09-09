"""Fail-closed identity and rollout-parity checks for Muse Glimmer.

Muse Glimmer support landed in vLLM after the 0.27.1 release, and the
multimodal-to-language-model LoRA mapping was corrected separately.  A
production RMCT run therefore must not infer compatibility merely because an
adapter loads.  It consumes an immutable preflight receipt that binds the
exact model snapshot, runtime commits, worker topology, uncapped EOS-only
probe, and a non-zero HF/PEFT-versus-vLLM adapter-effect comparison.

This module validates that receipt only.  The GPU preflight which creates it
lives under ``infra/isambard`` so importing the training backend remains cheap
and platform independent.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL_ID = "meta-models/Muse-Glimmer-30B"
MODEL_REVISION = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
VLLM_COMMIT = "8c2bbe00d58a930c6c09a80495728b26b79d9200"
VLLM_VERSION = "0.26.1rc1.dev1136+g8c2bbe00d"
VLLM_WHEEL_URL = (
    "https://wheels.vllm.ai/8c2bbe00d58a930c6c09a80495728b26b79d9200/"
    "vllm-0.26.1rc1.dev1136%2Bg8c2bbe00d-cp38-abi3-manylinux_2_28_aarch64.whl"
)
VLLM_WHEEL_SHA256 = "9b26de5f7bf0f2b7c7b722d6368d7fe18a211744ccad4598c468dfe86fb15e02"
TRANSFORMERS_VERSION = "5.15.1"
PARITY_ATTESTATION_SCHEMA = "muse-glimmer-rollout-worker-parity-v2"
PARITY_ATTESTATION_NAME = "muse-glimmer-rollout-worker-parity-attestation.json"
_SHA256_LENGTH = 64


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def is_muse_glimmer_model_name(model: str | Path) -> bool:
    """Recognise the official model ID or a local snapshot by its config."""

    text = str(model)
    normalized = text.rstrip("/").lower()
    if normalized in {MODEL_ID.lower(), "muse-glimmer-30b"} or normalized.endswith("/muse-glimmer-30b"):
        return True
    config_path = Path(text) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return isinstance(config, Mapping) and config.get("model_type") == "muse_glimmer"


def _canonical(value: Any, *, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Muse parity {label} is not canonical JSON") from exc


def _normalized_worker_gpus(values: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if isinstance(value, Mapping):
            logical = value.get("logical_index")
            token = value.get("device_token")
        else:
            logical = getattr(value, "logical_index", None)
            token = getattr(value, "device_token", None)
        if isinstance(logical, bool) or not isinstance(logical, int) or logical < 0:
            raise ValueError(f"Muse parity worker GPU {index} has an invalid logical_index")
        if not isinstance(token, str) or not token:
            raise ValueError(f"Muse parity worker GPU {index} has an invalid device_token")
        normalized.append({"logical_index": logical, "device_token": token})
    if not normalized or len({item["logical_index"] for item in normalized}) != len(normalized):
        raise ValueError("Muse parity worker GPUs must be non-empty and unique")
    return normalized


def _require_file_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Muse parity {label} must be an object")
    required = {"path", "sha256", "size_bytes"}
    if set(value) != required:
        raise ValueError(f"Muse parity {label} fields differ from {sorted(required)}")
    path, digest, size = value["path"], value["sha256"], value["size_bytes"]
    if not isinstance(path, str) or not path or not _is_sha256(digest):
        raise ValueError(f"Muse parity {label} has an invalid path or SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"Muse parity {label} has an invalid size")
    actual = Path(path).resolve()
    if actual.is_symlink() or not actual.is_file():
        raise ValueError(f"Muse parity {label} is absent or not a regular file: {actual}")
    if actual.stat().st_size != size or file_sha256(actual) != digest:
        raise ValueError(f"Muse parity {label} bytes differ from the attested identity")
    return {"path": str(actual), "sha256": digest, "size_bytes": size}


def _require_parity(value: Any, *, label: str, gate_kind: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Muse parity {label} must be an object")
    required = {
        "gate_kind",
        "passed",
        "count",
        "hf_l2",
        "vllm_l2",
        "difference_l2",
        "relative_to_hf_l2_error",
        "vllm_to_hf_l2_ratio",
        "cosine_similarity",
        "pearson_r",
        "max_abs_error",
    }
    if set(value) != required:
        raise ValueError(f"Muse parity {label} fields differ from {sorted(required)}")
    if value.get("passed") is not True:
        raise ValueError(f"Muse parity {label} did not pass")
    if value.get("gate_kind") != gate_kind:
        raise ValueError(f"Muse parity {label} uses the wrong numerical gate")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError(f"Muse parity {label} has too few compared logits")
    numeric = {key: value[key] for key in required - {"gate_kind", "passed", "count"}}
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
        for item in numeric.values()
    ):
        raise ValueError(f"Muse parity {label} contains a non-finite metric")
    hf_l2 = float(value["hf_l2"])
    vllm_l2 = float(value["vllm_l2"])
    if gate_kind == "raw_logprob":
        if hf_l2 <= 1.0e-8 or vllm_l2 <= 1.0e-8:
            raise ValueError(f"Muse parity {label} has an empty raw-logprob vector")
        if float(value["cosine_similarity"]) < 0.999 or float(value["pearson_r"]) < 0.999:
            raise ValueError(f"Muse parity {label} falls below the raw-score correlation gates")
        if float(value["max_abs_error"]) > 0.50:
            raise ValueError(f"Muse parity {label} exceeds the raw-score absolute-error gate")
    elif gate_kind == "policy_minus_base":
        if hf_l2 < 0.50 or vllm_l2 < 0.50:
            raise ValueError(f"Muse parity {label} does not prove a measurable non-zero LoRA effect")
        if float(value["cosine_similarity"]) < 0.90:
            raise ValueError(f"Muse parity {label} falls below the adapter-effect direction gate")
        ratio = float(value["vllm_to_hf_l2_ratio"])
        if not 0.80 <= ratio <= 1.25:
            raise ValueError(f"Muse parity {label} falls outside the adapter-effect norm-ratio gate")
        if float(value["relative_to_hf_l2_error"]) > 0.25:
            raise ValueError(f"Muse parity {label} exceeds the adapter-effect relative-L2 gate")
    else:
        raise ValueError(f"Muse parity {label} uses an unsupported numerical gate")
    return _canonical(dict(value), label=label)


def validate_muse_rollout_worker_parity_attestation(
    path: str | Path,
    *,
    expected_model: str | Path,
    expected_worker_gpus: Sequence[Any],
    expected_worker_engine_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and byte-rebind one Muse production parity receipt."""

    attestation_path = Path(path).resolve()
    if attestation_path.is_symlink() or not attestation_path.is_file():
        raise ValueError(f"Muse rollout parity attestation must be a regular file: {attestation_path}")
    try:
        document = json.loads(attestation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Muse rollout parity attestation is not valid JSON: {attestation_path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Muse rollout parity attestation must be an object")

    required = {
        "schema",
        "model",
        "runtime",
        "worker_gpus",
        "worker_engine_kwargs",
        "adapter",
        "probe",
        "scientific_contract",
        "source_manifest",
        "aggregate_base_parity",
        "aggregate_policy_parity",
        "aggregate_effect_parity",
        "per_worker_effect_parity",
    }
    if set(document) != required:
        raise ValueError(f"Muse rollout parity attestation fields differ from {sorted(required)}")
    if document.get("schema") != PARITY_ATTESTATION_SCHEMA:
        raise ValueError("Muse rollout parity attestation schema is unsupported")

    model = document.get("model")
    if not isinstance(model, Mapping) or set(model) != {
        "repo_id",
        "revision",
        "snapshot_path",
        "config_sha256",
    }:
        raise ValueError("Muse rollout parity model identity is malformed")
    expected_path = Path(expected_model).resolve()
    snapshot_path = Path(str(model.get("snapshot_path"))).resolve()
    if (
        model.get("repo_id") != MODEL_ID
        or model.get("revision") != MODEL_REVISION
        or snapshot_path != expected_path
        or snapshot_path.is_symlink()
        or not is_muse_glimmer_model_name(snapshot_path)
    ):
        raise ValueError("Muse rollout parity model identity does not match the pinned production snapshot")
    config_path = snapshot_path / "config.json"
    if not _is_sha256(model.get("config_sha256")) or file_sha256(config_path) != model["config_sha256"]:
        raise ValueError("Muse rollout parity config hash differs from the pinned snapshot")

    runtime = document.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "transformers_version",
        "vllm_commit",
        "torch_version",
        "platform",
        "receipts",
    }:
        raise ValueError("Muse rollout parity runtime identity is malformed")
    if runtime.get("transformers_version") != TRANSFORMERS_VERSION or runtime.get("vllm_commit") != VLLM_COMMIT:
        raise ValueError("Muse rollout parity does not bind the pinned Transformers/vLLM runtime")
    if not isinstance(runtime.get("torch_version"), str) or not runtime["torch_version"]:
        raise ValueError("Muse rollout parity lacks a torch version")
    if not isinstance(runtime.get("platform"), str) or not runtime["platform"]:
        raise ValueError("Muse rollout parity lacks a platform identity")
    runtime_receipts = runtime.get("receipts")
    if not isinstance(runtime_receipts, Mapping) or set(runtime_receipts) != {
        "runtime",
        "pip_freeze",
        "model_snapshot",
    }:
        raise ValueError("Muse rollout parity runtime receipts are malformed")
    for label in ("runtime", "pip_freeze", "model_snapshot"):
        _require_file_identity(runtime_receipts.get(label), label=f"runtime receipt {label}")

    observed_gpus = _normalized_worker_gpus(document.get("worker_gpus", []))
    expected_gpus = _normalized_worker_gpus(expected_worker_gpus)
    if observed_gpus != expected_gpus:
        raise ValueError("Muse rollout parity worker GPUs differ from the production worker topology")
    observed_options = _canonical(document.get("worker_engine_kwargs"), label="worker_engine_kwargs")
    expected_options = _canonical(dict(expected_worker_engine_kwargs), label="expected worker_engine_kwargs")
    if observed_options != expected_options:
        raise ValueError("Muse rollout parity vLLM engine options differ from production")
    if observed_options.get("language_model_only") is not True:
        raise ValueError("Muse rollout parity must use vLLM language_model_only=True")

    adapter = document.get("adapter")
    if not isinstance(adapter, Mapping) or set(adapter) != {"adapter_config", "adapter_model", "nonzero_tensor_count"}:
        raise ValueError("Muse rollout parity adapter identity is malformed")
    _require_file_identity(adapter.get("adapter_config"), label="adapter_config")
    _require_file_identity(adapter.get("adapter_model"), label="adapter_model")
    nonzero = adapter.get("nonzero_tensor_count")
    if isinstance(nonzero, bool) or not isinstance(nonzero, int) or nonzero < 1:
        raise ValueError("Muse rollout parity adapter contains no non-zero LoRA tensors")

    probe = document.get("probe")
    if not isinstance(probe, Mapping) or set(probe) != {
        "kind",
        "max_tokens",
        "termination",
        "source_prompt_count",
        "score_row_count",
        "anchor_source_index",
        "source_indices",
        "request_count",
        "non_eos_termination_count",
        "score_inputs_sha256",
    }:
        raise ValueError("Muse rollout parity probe is malformed")
    if (
        probe.get("kind") != "vllm-prompt-logprobs-uncapped-eos-tail-v1"
        or probe.get("max_tokens") is not None
        or probe.get("termination") != "eos_only"
        or probe.get("non_eos_termination_count") != 0
        or isinstance(probe.get("source_prompt_count"), bool)
        or not isinstance(probe.get("source_prompt_count"), int)
        or probe["source_prompt_count"] < len(expected_gpus)
        or isinstance(probe.get("score_row_count"), bool)
        or not isinstance(probe.get("score_row_count"), int)
        or isinstance(probe.get("anchor_source_index"), bool)
        or not isinstance(probe.get("anchor_source_index"), int)
        or not 0 <= probe["anchor_source_index"] < probe["source_prompt_count"]
        or not isinstance(probe.get("source_indices"), list)
        or probe["source_indices"]
        != [probe["anchor_source_index"]] * len(expected_gpus)
        + list(range(probe["source_prompt_count"]))
        or probe["score_row_count"] != len(probe["source_indices"])
        or isinstance(probe.get("request_count"), bool)
        or not isinstance(probe.get("request_count"), int)
        or probe["request_count"] != probe["score_row_count"] * 2
        or not _is_sha256(probe.get("score_inputs_sha256"))
    ):
        raise ValueError("Muse rollout parity probe is not an uncapped EOS-only worker probe")

    scientific = document.get("scientific_contract")
    if not isinstance(scientific, Mapping) or set(scientific) != {
        "frozen_spec_sha256",
        "data",
        "manifest",
        "generation",
    }:
        raise ValueError("Muse rollout parity scientific contract is malformed")
    if not _is_sha256(scientific.get("frozen_spec_sha256")):
        raise ValueError("Muse rollout parity scientific contract lacks a frozen-spec hash")
    _require_file_identity(scientific.get("data"), label="scientific data")
    _require_file_identity(scientific.get("manifest"), label="scientific manifest")
    if scientific.get("generation") != {
        "max_tokens": None,
        "termination": "eos_only",
        "non_eos_termination_policy": "fail_run",
    }:
        raise ValueError("Muse rollout parity scientific generation is not uncapped EOS-only")

    source_manifest = document.get("source_manifest")
    if not isinstance(source_manifest, Mapping) or set(source_manifest) != {"schema", "sha256", "files"}:
        raise ValueError("Muse rollout parity source manifest is malformed")
    files = source_manifest.get("files")
    if (
        source_manifest.get("schema") != "muse-glimmer-source-manifest-v1"
        or not _is_sha256(source_manifest.get("sha256"))
        or not isinstance(files, list)
        or not files
    ):
        raise ValueError("Muse rollout parity source manifest identity is malformed")
    compact = []
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"Muse rollout parity source file {index} is malformed")
        relative = record.get("relative_path")
        if not isinstance(relative, str) or not relative or relative in seen:
            raise ValueError("Muse rollout parity source paths must be non-empty and unique")
        seen.add(relative)
        identity = _require_file_identity(
            {key: record[key] for key in ("path", "sha256", "size_bytes")},
            label=f"source file {relative}",
        )
        compact.append(
            {
                "relative_path": relative,
                "sha256": identity["sha256"],
                "size_bytes": identity["size_bytes"],
            }
        )
    source_digest = hashlib.sha256(
        (json.dumps(compact, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    if source_manifest.get("sha256") != source_digest:
        raise ValueError("Muse rollout parity source-manifest digest changed")

    _require_parity(
        document.get("aggregate_base_parity"),
        label="aggregate_base_parity",
        gate_kind="raw_logprob",
    )
    _require_parity(
        document.get("aggregate_policy_parity"),
        label="aggregate_policy_parity",
        gate_kind="raw_logprob",
    )
    _require_parity(
        document.get("aggregate_effect_parity"),
        label="aggregate_effect_parity",
        gate_kind="policy_minus_base",
    )
    per_worker = document.get("per_worker_effect_parity")
    if not isinstance(per_worker, Sequence) or isinstance(per_worker, (str, bytes)) or len(per_worker) != len(expected_gpus):
        raise ValueError("Muse rollout parity must contain one effect result per worker")
    seen: set[int] = set()
    for entry in per_worker:
        if not isinstance(entry, Mapping) or set(entry) != {"worker_index", "worker_gpu", "effect_parity"}:
            raise ValueError("Muse rollout parity per-worker entry is malformed")
        worker_index = entry.get("worker_index")
        if isinstance(worker_index, bool) or not isinstance(worker_index, int) or worker_index in seen:
            raise ValueError("Muse rollout parity per-worker index is invalid or duplicated")
        if not 0 <= worker_index < len(expected_gpus) or entry.get("worker_gpu") != expected_gpus[worker_index]:
            raise ValueError("Muse rollout parity per-worker GPU identity is inconsistent")
        _require_parity(
            entry.get("effect_parity"),
            label=f"worker[{worker_index}].effect_parity",
            gate_kind="policy_minus_base",
        )
        seen.add(worker_index)
    if seen != set(range(len(expected_gpus))):
        raise ValueError("Muse rollout parity omits a production worker")
    return _canonical(dict(document), label="attestation")


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "PARITY_ATTESTATION_NAME",
    "PARITY_ATTESTATION_SCHEMA",
    "TRANSFORMERS_VERSION",
    "VLLM_COMMIT",
    "VLLM_VERSION",
    "VLLM_WHEEL_SHA256",
    "VLLM_WHEEL_URL",
    "file_sha256",
    "is_muse_glimmer_model_name",
    "validate_muse_rollout_worker_parity_attestation",
]
