"""Focused fail-closed tests for the fresh Qwen3.5 AttCT/MLPCT gate."""

from __future__ import annotations

import asyncio
import json
import random

import pytest
import torch
from tinker import types

from ctm.backends.local import engine as engine_module
from ctm.backends.local.engine import HAS_PEFT, LocalBackend
from ctm.core.config import AdamConfig, LoRAConfig
from ctm.training import sft as sft_module
from ctm.training.sft import SFTConfig, _write_immutable_consistency_preflight, train_sft


ACT_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
]


def _tiny_qwen35_model():
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(4)
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        max_position_embeddings=32,
        attention_dropout=0.2,
    )
    return Qwen3_5ForCausalLM(config)


def test_qwen35_auto_causallm_loader_extracts_text_config_before_weight_loading(tmp_path, monkeypatch):
    """The production AutoModel path must not instantiate the outer multimodal wrapper.

    ``AutoConfig`` returns ``Qwen3_5Config`` for the repository, whose outer
    architecture is conditional generation.  ``LocalBackend.setup`` uses
    ``AutoModelForCausalLM.from_pretrained``; Transformers must extract the
    nested text config before it hands control to the selected model loader.
    Capture that handoff with a local config and a patched classmethod, so this
    is structural only: it downloads no weights and allocates no model.
    """

    from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    outer_config = Qwen3_5Config()
    outer_config.save_pretrained(tmp_path)
    captured: dict[str, object] = {}
    sentinel = object()

    def capture_from_pretrained(cls, path, *args, config, **kwargs):
        captured.update({"class": cls, "path": path, "config": config, "kwargs": kwargs})
        return sentinel

    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", classmethod(capture_from_pretrained))

    result = AutoModelForCausalLM.from_pretrained(str(tmp_path), local_files_only=True)

    assert result is sentinel
    assert captured["class"] is Qwen3_5ForCausalLM
    assert isinstance(captured["config"], Qwen3_5TextConfig)
    assert captured["config"].model_type == "qwen3_5_text"
    assert captured["config"].num_hidden_layers == 32
    assert captured["config"].layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ] * 8


def _paired_datum(prefix_token: int = 1) -> types.Datum:
    scalar = lambda value: types.TensorData.from_torch(torch.tensor([value], dtype=torch.long))  # noqa: E731
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[prefix_token, 2, 3, 4, 5, 6, 7]),
        loss_fn_inputs={
            "clean_tokens": types.TensorData.from_torch(torch.tensor([3, 4, 5, 6, 7], dtype=torch.long)),
            "start_index": scalar(2),
            "clean_start_index": scalar(0),
            "clean_len": scalar(5),
            "match_len": scalar(5),
        },
    )


def _preflight_backend(*, method_config: dict | None = None, targets: list[str] | None = None) -> LocalBackend:
    backend = LocalBackend(
        device="cpu",
        dtype=torch.float32,
        use_lora=True,
        model_instance=_tiny_qwen35_model(),
        consistency_loss_options=method_config or {"layer_selection": "all"},
    )
    backend.setup(
        model="Qwen/Qwen3.5-9B",
        lora=LoRAConfig(
            rank=2,
            dropout=0.1,
            target_modules=targets or ["q_proj", "v_proj"],
            train_mlp=False,
            train_attn=False,
            train_unembed=False,
            seed=17,
        ),
    )
    return backend


@pytest.mark.skipif(not HAS_PEFT, reason="Qwen3.5 consistency preflight needs PEFT")
@pytest.mark.parametrize(("method", "group_size"), [("act", 1), ("attct", 1), ("mlpct", 8)])
def test_qwen35_preflight_runs_real_paired_backward_and_restores_observational_state(method, group_size):
    backend = _preflight_backend(targets=ACT_TARGETS if method == "act" else None)
    model = backend.model
    assert model is not None
    first_trainable = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    sentinel_grad = torch.ones_like(first_trainable)
    first_trainable.grad = sentinel_grad.clone()
    backend._gradient_accumulations = 7
    parameter_bytes_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if ".lora_" in name
    }
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()
    attention_implementation_before = model.config._attn_implementation

    report = backend.run_qwen35_consistency_preflight(
        [_paired_datum(prefix) for prefix in range(1, group_size + 1)],
        method=method,
        expected_group_size=group_size,
    )

    assert report["passed"], report["errors"]
    assert report["probe"]["datums_used"] == group_size
    assert report["probe"]["expected_group_size"] == group_size
    assert report["probe"]["loss"] > 0
    assert backend._gradient_accumulations == 7
    assert backend._optimizer is None
    assert first_trainable.grad is not None
    torch.testing.assert_close(first_trainable.grad, sentinel_grad)
    assert random.getstate() == python_rng_before
    torch.testing.assert_close(torch.get_rng_state(), torch_rng_before)
    for name, parameter in model.named_parameters():
        if name in parameter_bytes_before:
            torch.testing.assert_close(parameter.detach(), parameter_bytes_before[name])
    state = report["state_restoration"]
    assert state["rng_restored"]
    assert state["rng_restoration_verified"]
    assert state["gradients_restored"]
    assert state["gradient_restoration_verified"]
    assert state["gradient_accumulations_restored"]
    assert state["module_training_modes_restored"]
    assert state["optimizer_identity_preserved"]
    assert state["adapter_restoration_verified"]
    assert state["attention_implementation_restored"]
    assert state["backend_caches_restored"]
    assert model.config._attn_implementation == attention_implementation_before
    assert backend._consistency_loss_modules == {}
    assert backend._mlp_hooks is None
    assert backend._base_mlp_hooks is None

    if method == "act":
        assert report["probe"]["variant_hidden_state_counts"] == [33]
        assert report["probe"]["reference_hidden_state_counts"] == [33]
        assert report["probe"]["act_matching_suffix_lengths"] == [5]
        assert report["lora"]["positive_lora_b_gradients_by_family"]["self_attn"] >= 1
        assert report["lora"]["positive_lora_b_gradients_by_family"]["linear_attn"] >= 1
        assert {gradient["family"] for gradient in report["lora"]["lora_b_gradients"]} == {
            "self_attn",
            "linear_attn",
        }
    elif method == "attct":
        assert report["probe"]["variant_attention_counts"] == [8]
        assert report["probe"]["reference_attention_counts"] == [8]
        terminal_v = next(
            gradient
            for gradient in report["lora"]["lora_b_gradients"]
            if gradient["physical_layer"] == 31 and gradient["projection"] == "v_proj"
        )
        assert terminal_v["exempt_terminal_attct_v"]
        assert terminal_v["valid"]
    else:
        assert report["probe"]["variant_mlp_hook_counts"] == [32] * group_size
        assert report["probe"]["reference_mlp_hook_counts"] == [32] * group_size
        assert all(gradient["valid"] for gradient in report["lora"]["lora_b_gradients"])


@pytest.mark.skipif(not HAS_PEFT, reason="Qwen3.5 consistency preflight needs PEFT")
def test_qwen35_preflight_fails_closed_when_a_reported_restoration_invariant_is_false(monkeypatch):
    backend = _preflight_backend()
    digest = engine_module._adapter_parameter_digest
    calls = 0

    def mismatching_restored_digest(model):
        nonlocal calls
        calls += 1
        value, count = digest(model)
        # The third call is the post-restore verification. The adapter itself
        # remains untouched; this proves a false recorded invariant is enough
        # to reject the run rather than merely annotate a passing report.
        return ("different-after-restore", count) if calls == 3 else (value, count)

    monkeypatch.setattr(engine_module, "_adapter_parameter_digest", mismatching_restored_digest)

    report = backend.run_qwen35_consistency_preflight([_paired_datum()], method="attct", expected_group_size=1)

    assert not report["passed"]
    assert report["state_restoration"]["adapter_restoration_verified"] is False
    assert any("state-restoration invariant failed: adapter_restoration_verified" in error for error in report["errors"])


@pytest.mark.skipif(not HAS_PEFT, reason="Qwen3.5 consistency preflight needs PEFT")
def test_qwen35_preflight_fails_closed_for_non_qv_adapter_without_running_probe():
    backend = _preflight_backend(targets=["q_proj"])

    report = backend.run_qwen35_consistency_preflight([_paired_datum()], method="attct", expected_group_size=1)

    assert not report["passed"]
    assert report["probe"]["datums_used"] == 0
    assert any("exact target_modules" in error for error in report["errors"])
    assert report["state_restoration"]["optimizer_identity_preserved"]
    assert report["state_restoration"]["adapter_restoration_verified"]


@pytest.mark.skipif(not HAS_PEFT, reason="Qwen3.5 consistency preflight needs PEFT")
def test_qwen35_act_preflight_fails_closed_without_a_nonzero_gradient_in_each_attention_family(monkeypatch):
    backend = _preflight_backend(targets=ACT_TARGETS)
    original = LocalBackend._consistency_forward_backward

    def zero_lora_b_gradients(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        assert self.model is not None
        for name, parameter in self.model.named_parameters():
            if ".lora_B." in name:
                parameter.grad = torch.zeros_like(parameter)
        return result

    monkeypatch.setattr(LocalBackend, "_consistency_forward_backward", zero_lora_b_gradients)
    report = backend.run_qwen35_consistency_preflight([_paired_datum()], method="act", expected_group_size=1)

    assert not report["passed"]
    assert report["probe"]["loss"] > 0
    assert report["lora"]["positive_lora_b_gradients_by_family"] == {}
    assert any("ACT preflight requires at least one finite positive LoRA-B gradient" in error for error in report["errors"])


def test_consistency_preflight_evidence_is_create_once(tmp_path):
    report = {"schema_version": 1, "passed": True, "errors": []}

    path = _write_immutable_consistency_preflight(tmp_path, report)

    assert json.loads(path.read_text(encoding="utf-8")) == report
    with pytest.raises(RuntimeError, match="immutable consistency preflight"):
        _write_immutable_consistency_preflight(tmp_path, {"schema_version": 1, "passed": False, "errors": ["new"]})
    assert json.loads(path.read_text(encoding="utf-8")) == report


def _word_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"prefix": 0, "question": 1, "[UNK]": 2}
    vocabulary.update({f"word{index}": index + 3 for index in range(16)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


class _DummyLogger:
    def log_hparams(self, _values):
        pass

    def log_metrics(self, _values, step=None):
        pass

    def close(self):
        pass


class _FailingPreflightBackend:
    renderer_source = "hf"
    captured_datums = None
    expected_group_size = None

    def setup(self, **_kwargs):
        pass

    def run_qwen35_consistency_preflight(self, datums, *, method, expected_group_size):
        self.captured_datums = list(datums)
        self.expected_group_size = expected_group_size
        return {
            "schema_version": 1,
            "kind": "qwen35-consistency-preflight",
            "method": method,
            "passed": False,
            "probe": {"datums_used": len(self.captured_datums), "expected_group_size": expected_group_size},
            "errors": ["deliberate test failure"],
        }


def test_sft_passes_the_deterministic_full_mlpct_accumulation_group_to_preflight(tmp_path, monkeypatch):
    data_path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "variant_messages": [{"role": "user", "content": f"prefix word{index} question"}],
            "reference_messages": [{"role": "user", "content": "question"}],
        }
        for index in range(16)
    ]
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backend = _FailingPreflightBackend()
    monkeypatch.setattr(sft_module, "get_renderer_and_tokenizer", lambda *_args, **_kwargs: (None, _word_tokenizer()))
    monkeypatch.setattr(sft_module, "setup_logging", lambda **_kwargs: _DummyLogger())
    monkeypatch.setattr(sft_module, "get_git_state", lambda: {})
    monkeypatch.setattr(sft_module, "warn_if_dirty", lambda _git: None)
    monkeypatch.setattr(sft_module, "write_run_manifest", lambda *_args, **_kwargs: None)
    cfg = SFTConfig(
        experiment_name="preflight-test",
        run_name="mlpct-group",
        method="mlpct",
        model="Qwen/Qwen3.5-9B",
        optimizer=AdamConfig(learning_rate=1e-4),
        lora=LoRAConfig(seed=123),
        batch_size=1,
        gradient_accumulation_steps=8,
        qwen35_consistency_preflight=True,
        log_base_dir=str(tmp_path / "logs"),
    )

    with pytest.raises(RuntimeError, match="failed before any optimizer step"):
        asyncio.run(train_sft(data_path, config=cfg, backend=backend))

    assert backend.expected_group_size == 8
    assert backend.captured_datums is not None and len(backend.captured_datums) == 8
    expected_indices = list(range(16))
    random.Random(123).shuffle(expected_indices)
    captured_word_tokens = [datum.model_input.to_ints()[1] for datum in backend.captured_datums]
    assert captured_word_tokens == [index + 3 for index in expected_indices[:8]]
    evidence = tmp_path / "logs" / "preflight-test" / "mlpct-group" / "consistency-preflight.json"
    assert json.loads(evidence.read_text(encoding="utf-8"))["probe"] == {
        "datums_used": 8,
        "expected_group_size": 8,
    }


def test_sft_uses_one_paired_datum_for_act_preflight_even_with_gradient_accumulation(tmp_path, monkeypatch):
    data_path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "variant_messages": [{"role": "user", "content": f"prefix word{index} question"}],
            "reference_messages": [{"role": "user", "content": "question"}],
        }
        for index in range(16)
    ]
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    backend = _FailingPreflightBackend()
    monkeypatch.setattr(sft_module, "get_renderer_and_tokenizer", lambda *_args, **_kwargs: (None, _word_tokenizer()))
    monkeypatch.setattr(sft_module, "setup_logging", lambda **_kwargs: _DummyLogger())
    monkeypatch.setattr(sft_module, "get_git_state", lambda: {})
    monkeypatch.setattr(sft_module, "warn_if_dirty", lambda _git: None)
    monkeypatch.setattr(sft_module, "write_run_manifest", lambda *_args, **_kwargs: None)
    cfg = SFTConfig(
        experiment_name="preflight-test",
        run_name="act-one-pair",
        method="act",
        model="Qwen/Qwen3.5-9B",
        optimizer=AdamConfig(learning_rate=1e-4),
        lora=LoRAConfig(seed=123),
        batch_size=1,
        gradient_accumulation_steps=8,
        qwen35_consistency_preflight=True,
        log_base_dir=str(tmp_path / "logs"),
    )

    with pytest.raises(RuntimeError, match="failed before any optimizer step"):
        asyncio.run(train_sft(data_path, config=cfg, backend=backend))

    assert backend.expected_group_size == 1
    assert backend.captured_datums is not None and len(backend.captured_datums) == 1


def test_sft_rejects_an_existing_preflight_namespace_before_any_provenance_write(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs" / "preflight-test" / "reused-run"
    log_dir.mkdir(parents=True)
    manifest = log_dir / "manifest.json"
    evidence = log_dir / "consistency-preflight.json"
    manifest_bytes = b'{"prior":"manifest"}\n'
    evidence_bytes = b'{"prior":"preflight"}\n'
    manifest.write_bytes(manifest_bytes)
    evidence.write_bytes(evidence_bytes)
    calls: list[str] = []

    def unexpected_logging(**_kwargs):
        calls.append("setup_logging")
        raise AssertionError("setup_logging must not run for a reused preflight namespace")

    def unexpected_manifest(*_args, **_kwargs):
        calls.append("write_run_manifest")
        raise AssertionError("write_run_manifest must not run for a reused preflight namespace")

    monkeypatch.setattr(sft_module, "setup_logging", unexpected_logging)
    monkeypatch.setattr(sft_module, "write_run_manifest", unexpected_manifest)
    cfg = SFTConfig(
        experiment_name="preflight-test",
        run_name="reused-run",
        method="mlpct",
        model="Qwen/Qwen3.5-9B",
        qwen35_consistency_preflight=True,
        log_base_dir=str(tmp_path / "logs"),
    )

    with pytest.raises(RuntimeError, match="Refusing to reuse Qwen3.5 consistency-preflight run namespace"):
        asyncio.run(train_sft(tmp_path / "unused.jsonl", config=cfg, backend=_FailingPreflightBackend()))

    assert calls == []
    assert manifest.read_bytes() == manifest_bytes
    assert evidence.read_bytes() == evidence_bytes
