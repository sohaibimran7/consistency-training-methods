from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.act_repair_gate.amplify_self_attention import make_amplified_self_attention_adapter
from experiments.act_repair_gate import vllm_compat_adapter


def _adapter(source):
    source.mkdir()
    (source / "adapter_config.json").write_text("{}")
    (source / "manifest.json").write_text('{"frozen": true}\n')
    tensors = {
        "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight": torch.tensor([[1.0]]),
        "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight": torch.tensor([[2.0]]),
        "base_model.model.model.layers.1.linear_attn.in_proj_qkv.lora_B.weight": torch.tensor([[3.0]]),
        "base_model.model.model.layers.1.mlp.up_proj.lora_B.weight": torch.tensor([[4.0]]),
    }
    save_file(tensors, str(source / "adapter_model.safetensors"), metadata={"format": "pt"})


def test_amplified_adapter_keeps_only_fourfold_self_attention_lora_b(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _adapter(source)

    manifest = make_amplified_self_attention_adapter(source, destination)
    tensors = load_file(str(destination / "adapter_model.safetensors"))

    assert tensors["base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight"].item() == 1.0
    assert tensors["base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"].item() == 8.0
    assert tensors["base_model.model.model.layers.1.linear_attn.in_proj_qkv.lora_B.weight"].item() == 0.0
    assert tensors["base_model.model.model.layers.1.mlp.up_proj.lora_B.weight"].item() == 0.0
    assert (destination / "manifest.json").read_bytes() == (source / "manifest.json").read_bytes()
    persisted = json.loads((destination / "parity-amplification-manifest.json").read_text())
    assert persisted == manifest
    assert manifest["scale"] == 4.0
    assert manifest["kept_self_attn_lora_b"] == ["base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"]
    assert manifest["zeroed_non_self_attn_lora_b"] == [
        "base_model.model.model.layers.1.linear_attn.in_proj_qkv.lora_B.weight",
        "base_model.model.model.layers.1.mlp.up_proj.lora_B.weight",
    ]


def test_amplification_refuses_to_overwrite_destination(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _adapter(source)
    destination.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        make_amplified_self_attention_adapter(source, destination)


def test_amplification_fails_closed_for_missing_manifest_or_unknown_lora_b_family(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _adapter(source)
    (source / "manifest.json").unlink()

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        make_amplified_self_attention_adapter(source, destination)

    (source / "manifest.json").write_text("{}")
    tensors = load_file(str(source / "adapter_model.safetensors"))
    tensors["base_model.model.model.layers.1.unknown_proj.lora_B.weight"] = torch.tensor([[4.0]])
    save_file(tensors, str(source / "adapter_model.safetensors"), metadata={"format": "pt"})

    with pytest.raises(ValueError, match="unsupported LoRA-B projection family"):
        make_amplified_self_attention_adapter(source, destination)
    assert not destination.exists()


def test_composite_cli_mode_reaches_composite_attestation(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    primary = tmp_path / "primary.json"
    amplified = tmp_path / "amplified.json"
    called = {}

    def fake_attest(path, *, primary_report_path, amplified_self_attention_report_path):
        called.update(
            {
                "adapter": path,
                "primary": primary_report_path,
                "amplified": amplified_self_attention_report_path,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(vllm_compat_adapter, "attest_compat_adapter_with_amplified_self_attention", fake_attest)
    vllm_compat_adapter.main(
        [
            "--adapter",
            str(adapter),
            "--primary-parity-report",
            str(primary),
            "--amplified-self-attn-report",
            str(amplified),
        ]
    )
    assert called == {"adapter": adapter, "primary": primary, "amplified": amplified}
