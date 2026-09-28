import copy
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from ctm.evals.qwen35_vllm_attestation import is_verified_qwen35_vllm_compat_adapter
from ctm.evals.qwen35_vllm_scope import (
    ACTIVE_VERDICT, ATTESTATION_SCHEMA, MODULES, NOOP_VERDICT, SCOPE,
    make_scope_attestation, scope_identity, scoped_verdict, validate_scope_report,
)
from experiments.act_repair_gate.runtime_parity import compare_effects, prepare_adapter_variants
from experiments.act_repair_gate.vllm_compat_adapter import attest_compat_adapter, make_compat_adapter


@pytest.fixture
def scoped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = {"base_model_name_or_path": "Qwen/Qwen3.5-9B", "r": 2, "lora_alpha": 4,
           "lora_dropout": 0.0, "target_modules": ["model.layers." + m for m in MODULES]}
    (raw / "adapter_config.json").write_text(json.dumps(cfg))
    tensors = {"base_model.model.model.layers." + m + f".lora_{p}.weight": torch.ones((2, 4) if p == "A" else (4, 2))
               for m in MODULES for p in ("A", "B")}
    save_file(tensors, raw / "adapter_model.safetensors")
    compat = tmp_path / "compat"
    make_compat_adapter(raw, compat)
    variants = {"hf": prepare_adapter_variants(raw, tmp_path, directory_name="hf", adapter_scope=SCOPE),
                "vllm": prepare_adapter_variants(compat, tmp_path, directory_name="vl", adapter_scope=SCOPE)}
    ident = scope_identity(compat)
    results = {}
    for variant in ("full", "linear_only", "self_attn_only", "evaluator_path"):
        vector = [0.0, 0.0] if variant == "linear_only" else [0.0, 0.2]
        results[variant] = {"summary": compare_effects(vector, vector),
                            "hf_effect_vector": vector, "vllm_effect_vector": vector,
                            "verdict": NOOP_VERDICT if variant == "linear_only" else ACTIVE_VERDICT}
    report = {"schema": "qwen35-lora-runtime-parity-v1", "model": cfg["base_model_name_or_path"],
              "adapter": {"path": str(compat), "adapter_model_sha256": ident["adapter_model_sha256"],
                          "hf_path": str(raw), "hf_adapter_model_sha256": scope_identity(raw)["adapter_model_sha256"]},
              "adapter_scope": ident, "adapter_variants": variants,
              "token_protocol": {"requested_result_variants": list(results), "requested_token_ids": [[1, 2]]},
              "results": results}
    return raw, compat, report, tmp_path


def test_explicit_scope_is_required_and_absent_family_is_not_faked(scoped):
    raw, compat, report, root = scoped
    with pytest.raises(ValueError, match="both Qwen3.5"):
        prepare_adapter_variants(raw, root, directory_name="old_contract")
    assert report["adapter_variants"]["hf"]["linear_only"]["zeroed_tensors"] == 16
    assert report["adapter_variants"]["hf"]["self_attn_only"]["zeroed_tensors"] == 0
    assert validate_scope_report(compat, report) == scope_identity(compat)


def test_scope_attestation_roundtrip_and_hash_tamper(scoped):
    _, compat, report, root = scoped
    path = root / "report.json"
    path.write_text(json.dumps(report))
    attestation = attest_compat_adapter(compat, path)
    assert attestation["schema"] == ATTESTATION_SCHEMA
    assert attestation["verdicts"]["linear_only"] == NOOP_VERDICT
    assert is_verified_qwen35_vllm_compat_adapter(compat)
    report["results"]["full"]["hf_effect_vector"][1] = 0.7
    path.write_text(json.dumps(report))
    assert not is_verified_qwen35_vllm_compat_adapter(compat)


@pytest.mark.parametrize("variant", ["full", "self_attn_only", "evaluator_path"])
def test_trained_effect_cannot_be_waived(scoped, variant):
    _, compat, report, _ = scoped
    report["results"][variant] = {"summary": compare_effects([0.0, 0.0], [0.0, 0.0]),
                                  "hf_effect_vector": [0.0, 0.0], "vllm_effect_vector": [0.0, 0.0],
                                  "verdict": ACTIVE_VERDICT}
    with pytest.raises(ValueError, match="trained adapter effect"):
        validate_scope_report(compat, report)


def test_nonzero_absent_family_fails_even_with_a_passing_label(scoped):
    _, compat, report, _ = scoped
    item = report["results"]["linear_only"]
    item["hf_effect_vector"] = item["vllm_effect_vector"] = [0.0, 0.1]
    item["summary"] = compare_effects([0.0, 0.1], [0.0, 0.1])
    with pytest.raises(ValueError, match="absent family"):
        validate_scope_report(compat, report)


def test_real_isolation_payload_is_checked(scoped):
    _, compat, report, _ = scoped
    variant = report["adapter_variants"]["vllm"]["linear_only"]
    from pathlib import Path
    from ctm.evals.qwen35_vllm_scope import sha
    path = Path(variant["path"]) / "adapter_model.safetensors"
    tensors = load_file(path)
    key = next(k for k in tensors if ".lora_B." in k)
    tensors[key].fill_(1)
    save_file(tensors, path)
    variant["adapter_sha256"] = sha(path)
    with pytest.raises(ValueError, match="exact family isolation"):
        validate_scope_report(compat, report)


def test_missing_pair_or_extra_family_cannot_claim_narrow_scope(scoped):
    raw, _, _, _ = scoped
    path = raw / "adapter_model.safetensors"
    tensors = load_file(path)
    tensors["base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"] = torch.ones(4, 2)
    save_file(tensors, path)
    with pytest.raises(ValueError, match="exactly 16 complete pairs"):
        scope_identity(raw)


@pytest.mark.parametrize("mutation", ["nan", "missing_result", "wrong_length", "lying_summary"])
def test_malformed_numerical_proof_fails(scoped, mutation):
    _, compat, report, _ = scoped
    if mutation == "nan":
        report["results"]["full"]["hf_effect_vector"] = [0.0, float("nan")]
    elif mutation == "missing_result":
        del report["results"]["evaluator_path"]
    elif mutation == "wrong_length":
        report["results"]["full"]["vllm_effect_vector"] = [0.0]
    else:
        report["results"]["full"]["summary"]["cosine_similarity"] = 0.5
    with pytest.raises(ValueError):
        validate_scope_report(compat, report)


def test_noop_threshold_is_not_relaxed():
    assert scoped_verdict("linear_only", {"hf_max_abs": 0.0, "vllm_max_abs": 0.0}) == NOOP_VERDICT
    assert scoped_verdict("linear_only", {"hf_max_abs": 0.0, "vllm_max_abs": 1e-5}) != NOOP_VERDICT


def test_existing_snapshot_alias_is_not_a_model_substitution(tmp_path):
    from ctm.evals.qwen35_vllm_scope import same_model_snapshot

    model = tmp_path / "revision-a"
    model.mkdir()
    alias = tmp_path / "scratch-alias"
    alias.symlink_to(model, target_is_directory=True)
    other = tmp_path / "revision-b"
    other.mkdir()
    assert same_model_snapshot(str(model), str(alias))
    assert not same_model_snapshot(str(model), str(other))
    assert not same_model_snapshot(str(model), str(tmp_path / "missing"))
    assert not same_model_snapshot(str(model), "Qwen/Qwen3.5-9B")
