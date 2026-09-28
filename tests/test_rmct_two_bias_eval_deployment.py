"""Focused deployment-boundary tests for the two-bias evaluation suite."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from experiments.rmct_two_bias_eval import deployment, raw_preflight
from experiments.rmct_two_bias_eval.contract import (
    BASE_MODEL,
    VLLM_GENERATION_CONFIG,
    VLLM_MODEL_ARGS,
    VLLM_PARITY_LOGPROBS_MODE,
    VLLM_PARITY_MAX_LOGPROBS,
    VLLM_PARITY_SAMPLING,
    VLLM_PARITY_SCORE_TRANSPORT,
    VLLM_PARITY_TOP_TOKEN_COUNT,
    VLLM_SAMPLER_RUNTIME,
    EvaluationContractError,
    validate_r003_parallel_parity_report,
    validate_r004_parallel_parity_report,
    validate_r005_parallel_parity_report,
    vllm_compatibility_identity,
)

_BIASES = (
    "unbiased",
    "wrong_argument",
    "suggested_answer",
    "distractor_fact",
    "post_hoc",
    "spurious_few_shot_squares",
    "wrong_few_shot",
)


def test_materialize_rebases_pinned_manifest_with_nonexistent_workstation_paths(monkeypatch, tmp_path: Path):
    """Only copied paths are opened; canonical /Users paths need not exist."""

    artifact_root = tmp_path / "copied-stage2"
    populations: dict[str, dict[str, object]] = {}
    for population in ("in_domain", "hle"):
        entries: dict[str, object] = {}
        for bias in _BIASES:
            filename = "clean.jsonl" if bias == "unbiased" else f"{bias}.jsonl"
            payload = f"{population}/{bias}\n".encode()
            copied = artifact_root / population / filename
            copied.parent.mkdir(parents=True, exist_ok=True)
            copied.write_bytes(payload)
            entries[bias] = {
                "path": f"/Users/absent/workstation-stage2/{population}/{filename}",
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
        populations[population] = {"artifacts": entries}
    source = tmp_path / "canonical-source-manifest.json"
    source.write_text(json.dumps({"populations": populations}), encoding="utf-8")
    assert not Path("/Users/absent/workstation-stage2/in_domain/clean.jsonl").exists()

    real_sha256 = deployment._sha256_file
    monkeypatch.setattr(
        deployment,
        "_sha256_file",
        lambda path: (
            deployment.CANONICAL_SOURCE_MANIFEST_SHA256
            if Path(path).resolve() == source.resolve()
            else real_sha256(Path(path))
        ),
    )
    # The production source is authenticated by its immutable SHA.  This
    # focused unit document intentionally omits the unrelated full schema.
    materialize = types.ModuleType("experiments.stage2_ood_hle.materialize")
    materialize.validate_manifest = lambda path: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "experiments.stage2_ood_hle.materialize", materialize)
    output = tmp_path / "deployment-manifest.json"
    record = deployment.materialize_deployment_manifest(source, artifact_root, output)

    assert record["source_manifest_sha256"] == deployment.CANONICAL_SOURCE_MANIFEST_SHA256
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["populations"]["in_domain"]["artifacts"]["unbiased"]["path"] == str(
        (artifact_root / "in_domain" / "clean.jsonl").resolve()
    )
    assert document["deployment_provenance"]["copied_artifacts"]["in_domain/unbiased"]["source_path"] == (
        "/Users/absent/workstation-stage2/in_domain/clean.jsonl"
    )


def test_raw_preflight_runtime_keeps_vllm_adapter_and_parity_identities(tmp_path: Path):
    """A preflight report cannot collapse vLLM to an unbound model string."""

    digest = "a" * 64
    compat = tmp_path / "compat"
    raw = tmp_path / "raw-r4"
    report = tmp_path / "parity-report.json"
    runtime = {
        "profile": "vllm",
        "base_model": BASE_MODEL,
        "checkpoint": str(compat),
        "raw_checkpoint": str(raw),
        "provider": "vllm",
        "generation": VLLM_GENERATION_CONFIG,
        "model_args": VLLM_MODEL_ARGS,
        "sampler": VLLM_SAMPLER_RUNTIME,
        "compatibility_adapter": {
            "path": str(compat),
            "adapter_model_sha256": digest,
            "adapter_config_sha256": digest,
            "compatibility_manifest": {"path": str(compat / "compatibility-manifest.json"), "sha256": digest},
            "parity_attestation": {
                "path": str(compat / "vllm-parity-attestation.json"),
                "sha256": digest,
                "schema": "qwen35-vllm-parity-attestation-v1",
            },
            "parity_reports": [{"path": str(report), "sha256": digest}],
            "source_raw_checkpoint": str(raw),
            "source_raw_adapter_model_sha256": digest,
            "base_model": BASE_MODEL,
        },
    }
    model, preserved = raw_preflight._validate_report_runtime(runtime)
    assert model == f"vllm/{BASE_MODEL}:{compat}"
    assert preserved == runtime
    assert VLLM_SAMPLER_RUNTIME["parity_top_token_count"] == VLLM_PARITY_TOP_TOKEN_COUNT == 16
    assert VLLM_SAMPLER_RUNTIME["parity_max_logprobs"] == VLLM_PARITY_MAX_LOGPROBS == 29
    assert VLLM_SAMPLER_RUNTIME["parity_logprobs_mode"] == VLLM_PARITY_LOGPROBS_MODE
    assert VLLM_SAMPLER_RUNTIME["parity_score_transport"] == VLLM_PARITY_SCORE_TRANSPORT
    assert VLLM_SAMPLER_RUNTIME["parity_sampling"] == VLLM_PARITY_SAMPLING

    stale_r003_runtime = json.loads(json.dumps(runtime))
    stale_r003_runtime["sampler"]["parity_max_logprobs"] = 20
    with pytest.raises(ValueError, match="invalid vLLM runtime"):
        raw_preflight._validate_report_runtime(stale_r003_runtime)

    wrong_top_count_runtime = json.loads(json.dumps(runtime))
    wrong_top_count_runtime["sampler"]["parity_top_token_count"] = 17
    with pytest.raises(ValueError, match="invalid vLLM runtime"):
        raw_preflight._validate_report_runtime(wrong_top_count_runtime)

    raw_logprobs_runtime = json.loads(json.dumps(runtime))
    raw_logprobs_runtime["sampler"]["parity_logprobs_mode"] = "raw_logprobs"
    with pytest.raises(ValueError, match="invalid vLLM runtime"):
        raw_preflight._validate_report_runtime(raw_logprobs_runtime)

    nonneutral_runtime = json.loads(json.dumps(runtime))
    nonneutral_runtime["sampler"]["parity_sampling"]["top_p"] = 0.95
    with pytest.raises(ValueError, match="invalid vLLM runtime"):
        raw_preflight._validate_report_runtime(nonneutral_runtime)

    runtime["compatibility_adapter"] = dict(runtime["compatibility_adapter"])
    runtime["compatibility_adapter"]["path"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="incomplete vLLM compatibility"):
        raw_preflight._validate_report_runtime(runtime)


def test_vllm_compatibility_config_must_be_the_raw_checkpoint_config(tmp_path: Path):
    """The converter is allowed to rename tensor keys, not alter PEFT config."""

    compat = tmp_path / "compat"
    compat.mkdir()
    (compat / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": BASE_MODEL}), encoding="utf-8"
    )
    for name in ("adapter_model.safetensors", "compatibility-manifest.json", "vllm-parity-attestation.json"):
        (compat / name).write_bytes(b"present")
    with pytest.raises(EvaluationContractError, match="configuration differs"):
        vllm_compatibility_identity(
            compat,
            raw_checkpoint={
                "path": str(tmp_path / "raw-r4"),
                "adapter_model_sha256": "a" * 64,
                "adapter_config_sha256": "b" * 64,
            },
        )


def _r005_parallel_report(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "parity"
    root.mkdir()
    compatibility = tmp_path / "compat"
    compatibility.mkdir()
    variants = ("full", "linear_only", "self_attn_only", "evaluator_path")
    names = {
        "full": "full",
        "linear_only": "linear_only",
        "self_attn_only": "self_attn_only",
        "evaluator_path": str(compatibility.resolve()),
    }
    plan: dict[str, dict[str, object]] = {}
    model_ids: dict[str, list[str]] = {}
    for index, variant in enumerate(variants):
        log = root / f"vllm-server-{variant}.log"
        log.write_text(f"{variant} server\n", encoding="utf-8")
        cache_directories: dict[str, str] = {}
        for name, leaf in (
            ("XDG_CACHE_HOME", "xdg-cache"),
            ("TRITON_CACHE_DIR", "triton-cache"),
            ("TORCHINDUCTOR_CACHE_DIR", "torchinductor-cache"),
        ):
            cache = root / "vllm-server-caches" / variant / leaf
            cache.mkdir(parents=True)
            cache_directories[name] = str(cache.resolve())
        plan[variant] = {
            "device_token": str(index),
            "port": 8789 + index,
            "server_log": str(log.resolve()),
            "max_loras": 1,
            "max_logprobs": VLLM_PARITY_MAX_LOGPROBS,
            "logprobs_mode": VLLM_PARITY_LOGPROBS_MODE,
            "cache_directories": cache_directories,
        }
        model_ids[variant] = ["base", names[variant]]
    document: dict[str, object] = {
        "schema": "qwen35-lora-runtime-parity-v1",
        "model": BASE_MODEL,
        "adapter": {"path": str(compatibility.resolve())},
        "token_protocol": {
            "requested_result_variants": list(variants),
            "top_token_count": VLLM_PARITY_TOP_TOKEN_COUNT,
            "requested_token_ids": [[11, 12, 13], [21, 22]],
            "reference_token_ids": [11, 22],
            "vllm_score_transport": VLLM_PARITY_SCORE_TRANSPORT,
            "vllm_allowed_token_ids": "requested_token_ids",
            "vllm_response_token_ids": "exactly_requested_token_ids",
        },
        "results": {variant: {"verdict": "hf_vllm_effects_agree"} for variant in variants},
        "backends": {
            "vllm": {
                "isolate_vllm_variants": True,
                "parallel_isolated_vllm_variants": True,
                "enforce_eager": True,
                "gdn_prefill_backend": "triton",
                "max_logprobs": VLLM_PARITY_MAX_LOGPROBS,
                "logprobs_mode": VLLM_PARITY_LOGPROBS_MODE,
                "score_transport": VLLM_PARITY_SCORE_TRANSPORT,
                "parity_sampling": VLLM_PARITY_SAMPLING,
                "parallel_isolated_server_plan": plan,
                "runtime_adapter_names": names,
                "model_ids_after_load": model_ids,
            }
        },
    }
    report = root / "report.json"
    report.write_text(json.dumps(document), encoding="utf-8")
    return report, compatibility, document


def test_r005_parity_report_requires_restricted_processed_logprobs_on_four_servers(tmp_path: Path):
    report, compatibility, document = _r005_parallel_report(tmp_path)
    assert validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility) == document

    duplicate_device = json.loads(json.dumps(document))
    duplicate_device["backends"]["vllm"]["parallel_isolated_server_plan"]["self_attn_only"]["device_token"] = "1"
    report.write_text(json.dumps(duplicate_device), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="reuses a device token"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    missing_identity = json.loads(json.dumps(document))
    missing_identity["backends"]["vllm"]["model_ids_after_load"]["evaluator_path"] = ["base"]
    report.write_text(json.dumps(missing_identity), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="exact adapter identity"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    shared_cache = json.loads(json.dumps(document))
    shared_cache["backends"]["vllm"]["parallel_isolated_server_plan"]["linear_only"]["cache_directories"] = (
        shared_cache["backends"]["vllm"]["parallel_isolated_server_plan"]["full"]["cache_directories"]
    )
    report.write_text(json.dumps(shared_cache), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="reuses a cache directory"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    stale_r003 = json.loads(json.dumps(document))
    stale_r003["backends"]["vllm"]["max_logprobs"] = 20
    report.write_text(json.dumps(stale_r003), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="pinned isolated native-vLLM runtime"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    wrong_top_count = json.loads(json.dumps(document))
    wrong_top_count["token_protocol"]["top_token_count"] = VLLM_PARITY_TOP_TOKEN_COUNT + 1
    report.write_text(json.dumps(wrong_top_count), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="canonical four-variant protocol"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    wrong_server_limit = json.loads(json.dumps(document))
    wrong_server_limit["backends"]["vllm"]["parallel_isolated_server_plan"]["full"]["max_logprobs"] = 20
    report.write_text(json.dumps(wrong_server_limit), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="invalid isolation settings"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    raw_mode = json.loads(json.dumps(document))
    raw_mode["backends"]["vllm"]["logprobs_mode"] = "raw_logprobs"
    report.write_text(json.dumps(raw_mode), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="processed-logprob runtime protocol"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    raw_server_mode = json.loads(json.dumps(document))
    raw_server_mode["backends"]["vllm"]["parallel_isolated_server_plan"]["full"]["logprobs_mode"] = (
        "raw_logprobs"
    )
    report.write_text(json.dumps(raw_server_mode), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="does not bind processed logprobs"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    nonneutral = json.loads(json.dumps(document))
    nonneutral["backends"]["vllm"]["parity_sampling"]["temperature"] = 1.0
    report.write_text(json.dumps(nonneutral), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="processed-logprob runtime protocol"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    unsupported_transport = json.loads(json.dumps(document))
    unsupported_transport["token_protocol"]["logprob_token_ids"] = "requested_token_ids"
    report.write_text(json.dumps(unsupported_transport), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="allowed-token score protocol"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)

    duplicate_requested_token = json.loads(json.dumps(document))
    duplicate_requested_token["token_protocol"]["requested_token_ids"][0] = [11, 11]
    report.write_text(json.dumps(duplicate_requested_token), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="invalid requested token set"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)


def test_r003_r004_aliases_preserve_legacy_report_validation(tmp_path: Path):
    report, compatibility, r005 = _r005_parallel_report(tmp_path)
    legacy = json.loads(json.dumps(r005))
    for field in (
        "vllm_score_transport",
        "vllm_allowed_token_ids",
        "vllm_response_token_ids",
        "requested_token_ids",
        "reference_token_ids",
    ):
        legacy["token_protocol"].pop(field)
    for field in ("logprobs_mode", "score_transport", "parity_sampling"):
        legacy["backends"]["vllm"].pop(field)
    for server in legacy["backends"]["vllm"]["parallel_isolated_server_plan"].values():
        server.pop("logprobs_mode")
    report.write_text(json.dumps(legacy), encoding="utf-8")

    assert validate_r004_parallel_parity_report(report, compatibility_adapter=compatibility) == legacy
    assert validate_r003_parallel_parity_report(report, compatibility_adapter=compatibility) == legacy
    with pytest.raises(EvaluationContractError, match="allowed-token score protocol"):
        validate_r005_parallel_parity_report(report, compatibility_adapter=compatibility)
