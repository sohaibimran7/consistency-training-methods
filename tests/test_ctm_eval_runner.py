"""Offline tests for the task-factory Inspect bridge and Tinker resolution."""

import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from inspect_ai.model import GenerateConfig, ModelAPI

from ctm.evals import local_model as local_model_module
from ctm.evals import runner as runner_module
from ctm.evals import tinker_model as tinker_model_module
from ctm.evals.local_model import read_local_checkpoint
from ctm.evals.runner import (
    build_tasks,
    effective_provider_generation_config,
    load_task_factory,
    normalize_generation_config,
    normalize_task_indices,
    parse_json_object,
    resolve_eval_model,
    run_task_evals,
    select_tasks,
    validate_tinker_generation_config,
)
from ctm.evals.tinker_model import tinker_base_model, tinker_checkpoint_model


class _Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _Service:
    def __init__(self, *, base_model="unit/model", renderer="unit_renderer"):
        self.training_run = SimpleNamespace(
            base_model=base_model,
            user_metadata={"renderer_name": renderer},
        )
        self.sampling_calls = []

    def create_rest_client(self):
        return self

    def get_training_run_by_tinker_path(self, checkpoint):
        self.checkpoint = checkpoint
        return _Future(self.training_run)

    def create_sampling_client(self, **kwargs):
        self.sampling_calls.append(kwargs)
        return object()


class _CookbookAPI(ModelAPI):
    def __init__(self, *, renderer_name, model_name, sampling_client, config, **kwargs):
        super().__init__(model_name, config=config)
        self.renderer_name = renderer_name
        self.sampling_client = sampling_client
        self.kwargs = kwargs

    async def generate(self, input, tools, tool_choice, config):
        raise NotImplementedError


def test_tinker_checkpoint_model_uses_checkpoint_owned_identity(monkeypatch):
    monkeypatch.setattr(tinker_model_module, "InspectAPIFromTinkerSampling", _CookbookAPI)
    service = _Service()
    model = tinker_checkpoint_model(
        "tinker://run/sampler_weights/final",
        config=GenerateConfig(max_tokens=8, temperature=0.25),
        service_client=service,
    )
    assert model.api.model_name == "unit/model"
    assert model.api.renderer_name == "unit_renderer"
    assert model.config.max_tokens == 8
    assert service.sampling_calls == [{"model_path": "tinker://run/sampler_weights/final", "base_model": "unit/model"}]


def test_tinker_base_model_uses_direct_base_sampling_client(monkeypatch):
    monkeypatch.setattr(tinker_model_module, "InspectAPIFromTinkerSampling", _CookbookAPI)
    monkeypatch.setattr(tinker_model_module.model_info, "get_recommended_renderer_name", lambda _: "recommended")
    service = _Service()
    model = tinker_base_model(
        "unit/base",
        config=GenerateConfig(max_tokens=8),
        include_reasoning=True,
        service_client=service,
    )
    assert model.api.model_name == "unit/base"
    assert model.api.renderer_name == "recommended"
    assert model.api.kwargs["include_reasoning"] is True
    assert service.sampling_calls == [{"base_model": "unit/base"}]


def test_tinker_checkpoint_adapter_rejects_invalid_modes(monkeypatch):
    monkeypatch.setattr(tinker_model_module, "InspectAPIFromTinkerSampling", _CookbookAPI)
    with pytest.raises(ValueError, match="tinker://"):
        tinker_checkpoint_model(
            "local/path",
            service_client=_Service(),
        )
    with pytest.raises(ValueError, match="does not match checkpoint"):
        tinker_checkpoint_model("tinker://x", base_model="wrong", service_client=_Service())
    with pytest.raises(ValueError, match="renderer_name"):
        tinker_checkpoint_model("tinker://x", renderer_name="wrong", service_client=_Service())
    with pytest.raises(ValueError, match="no renderer metadata"):
        tinker_checkpoint_model("tinker://x", service_client=_Service(renderer=None))
    with pytest.raises(ValueError, match="exactly one"):
        resolve_eval_model()


def test_parse_json_object_inline_or_file(tmp_path):
    assert parse_json_object('{"x": 1}', label="config") == {"x": 1}
    assert parse_json_object(
        '{"long": "' + ("x" * 300) + '"}',
        label="config",
    ) == {"long": "x" * 300}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"y": 2}))
    assert parse_json_object(str(path), label="config") == {"y": 2}
    with pytest.raises(ValueError, match="decode to an object"):
        parse_json_object("[]", label="config")


def test_task_factory_is_an_explicit_import_path(monkeypatch):
    assert load_task_factory("ctm.evals.runner:normalize_generation_config") is normalize_generation_config
    with pytest.raises(ValueError, match="module:callable"):
        load_task_factory("sycophancy")
    monkeypatch.setattr(runner_module, "load_task_factory", lambda _: lambda **kwargs: [kwargs])
    assert build_tasks("unit:suite", task_args={"dataset": "heldout"}) == [{"dataset": "heldout"}]


def test_task_selection_is_stable_1_based_and_validated():
    tasks = ["unbiased", "bias-a", "bias-b"]
    selected, indices = select_tasks(tasks, [1, 3])
    assert selected == ["unbiased", "bias-b"]
    assert indices == [1, 3]
    assert normalize_task_indices(None, task_count=3) == [1, 2, 3]
    with pytest.raises(ValueError, match="out of range"):
        select_tasks(tasks, [4])
    with pytest.raises(ValueError, match="more than once"):
        select_tasks(tasks, [2, 2])


def test_normal_inspect_model_resolution_uses_provider_registry():
    model = resolve_eval_model(model="mockllm/unit", generation_config={"max_tokens": 37})
    assert model.api.model_name == "unit"
    assert model.config.max_tokens == 37
    assert model.config.temperature == 0.0


def test_direct_hf_muse_text_only_option_is_consumed_and_detaches_vision(monkeypatch):
    import torch

    class FakeMuse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="muse_glimmer")
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Linear(2, 2)
            self.model.vision_tower = torch.nn.Linear(2, 2)
            self.model.vision_adapter = torch.nn.Linear(2, 2)
            self.model.vision_projection = torch.nn.Linear(2, 2)
            self.model.perception_emb_norm = torch.nn.LayerNorm(2)

    captured = {}
    wrapped = SimpleNamespace(api=SimpleNamespace(model=FakeMuse()))

    def fake_get_model(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return wrapped

    monkeypatch.setattr("inspect_ai.model.get_model", fake_get_model)
    result = resolve_eval_model(
        model="hf/unit/muse",
        model_args={"device": "cuda:0", "hf_language_model_only": True},
    )

    assert result is wrapped
    assert captured["name"] == "hf/unit/muse"
    assert captured["kwargs"]["device"] == "cuda:0"
    assert "hf_language_model_only" not in captured["kwargs"]
    assert wrapped.api.model.model.language_model is not None
    assert wrapped.api.model.model.vision_tower is None
    assert wrapped.api.model.model.vision_adapter is None
    assert wrapped.api.model.model.vision_projection is None
    assert wrapped.api.model.model.perception_emb_norm is None


def test_hf_language_model_only_rejects_non_hf_provider():
    with pytest.raises(ValueError, match="explicit hf"):
        resolve_eval_model(
            model="mockllm/unit",
            model_args={"hf_language_model_only": True},
        )


def test_tinker_rejects_provider_model_args():
    with pytest.raises(ValueError, match="model_args"):
        resolve_eval_model(tinker_checkpoint="tinker://x", model_args={"base_url": "http://localhost"})


def test_local_checkpoint_manifest_is_validated(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"backend": "local", "model": "unit/base", "lora": True}))
    (checkpoint / "adapter_config.json").write_text("{}")
    directory, manifest = read_local_checkpoint(f"file://{checkpoint}")
    assert directory == checkpoint.resolve()
    assert manifest["model"] == "unit/base"

    (checkpoint / "manifest.json").write_text(json.dumps({"backend": "local", "model": "unit/base", "lora": False}))
    with pytest.raises(ValueError, match="weights.pt"):
        read_local_checkpoint(checkpoint)
    (checkpoint / "weights.pt").write_bytes(b"placeholder")
    directory, manifest = read_local_checkpoint(checkpoint)
    assert directory == checkpoint.resolve()
    assert manifest["lora"] is False


def test_local_checkpoint_resolution_uses_local_bridge(monkeypatch):
    expected = object()
    monkeypatch.setattr(local_model_module, "local_checkpoint_model", lambda *args, **kwargs: expected)
    resolved = resolve_eval_model(
        local_checkpoint="file:///checkpoint",
        base_model="unit/base",
        model_args={"device": "cpu"},
        generation_config={"max_tokens": 8},
    )
    assert resolved is expected


def test_local_checkpoint_vllm_bridge_uses_manifest_base_and_adapter_path(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"backend": "local", "model": "unit/base", "lora": True}))
    (checkpoint / "adapter_config.json").write_text("{}")
    captured = {}

    def fake_get_model(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr("inspect_ai.model.get_model", fake_get_model)
    from ctm.evals.local_model import local_checkpoint_model

    result = local_checkpoint_model(
        checkpoint,
        model_args={"provider": "vllm", "gpu_memory_utilization": 0.9, "max_model_len": 32768},
        generation_config={"max_tokens": 128},
    )
    assert result is not None
    assert captured["name"] == f"vllm/unit/base:{checkpoint.resolve()}"
    assert captured["kwargs"]["gpu_memory_utilization"] == 0.9
    assert captured["kwargs"]["max_model_len"] == 32768
    assert captured["kwargs"]["config"].max_tokens == 128


def test_local_checkpoint_vllm_refuses_qwen35_lora_that_would_silently_evaluate_base(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"backend": "local", "model": "Qwen/Qwen3.5-9B", "lora": True})
    )
    (checkpoint / "adapter_config.json").write_text("{}")

    from ctm.evals.local_model import local_checkpoint_model

    with pytest.raises(ValueError, match="maps no `model.layers"):
        local_checkpoint_model(checkpoint, model_args={"provider": "vllm"})


def test_native_vllm_identity_allows_only_byte_attested_qwen35_compat_adapter(tmp_path):
    checkpoint = tmp_path / "qwen35-compat"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps({"backend": "local", "model": "Qwen/Qwen3.5-9B", "lora": True})
    )
    (checkpoint / "adapter_config.json").write_text("{}")
    weights = checkpoint / "adapter_model.safetensors"
    weights.write_bytes(b"compatibility-adapter")
    adapter_hash = sha256(weights.read_bytes()).hexdigest()
    variants = ["full", "linear_only", "self_attn_only", "evaluator_path"]
    report = tmp_path / "parity-report.json"
    report.write_text(
        json.dumps(
            {
                "adapter": {"path": str(checkpoint.resolve()), "adapter_model_sha256": adapter_hash},
                "results": {variant: {"verdict": "hf_vllm_effects_agree"} for variant in variants},
            }
        )
    )
    (checkpoint / "vllm-parity-attestation.json").write_text(
        json.dumps(
            {
                "schema": "qwen35-vllm-parity-attestation-v1",
                "adapter_path": str(checkpoint.resolve()),
                "adapter_model_sha256": adapter_hash,
                "report_path": str(report.resolve()),
                "report_sha256": sha256(report.read_bytes()).hexdigest(),
                "required_variants": variants,
                "verdicts": {variant: "hf_vllm_effects_agree" for variant in variants},
            }
        )
    )

    from ctm.evals.local_model import native_vllm_request_identity

    assert native_vllm_request_identity(
        local_checkpoint=checkpoint,
        model_args={"provider": "vllm"},
    ) == ("Qwen/Qwen3.5-9B", str(checkpoint.resolve()))


def test_native_vllm_identity_allows_only_hash_bound_amplified_self_attention_evidence(tmp_path):
    """A weak native self-attention signal needs the separate 4x transport proof."""

    def write_adapter(path, payload):
        path.mkdir()
        (path / "adapter_config.json").write_text("{}")
        (path / "adapter_model.safetensors").write_bytes(payload)
        return sha256(payload).hexdigest()

    checkpoint = tmp_path / "production-compat"
    primary_hf = tmp_path / "primary-hf"
    amplified_hf = tmp_path / "amplified-hf"
    amplified_vllm = tmp_path / "amplified-vllm"
    production_hash = write_adapter(checkpoint, b"production")
    primary_hf_hash = write_adapter(primary_hf, b"primary-hf")
    amplified_hf_hash = write_adapter(amplified_hf, b"amplified-hf")
    amplified_vllm_hash = write_adapter(amplified_vllm, b"amplified-vllm")
    model = "Qwen/Qwen3.5-9B"

    amplification_manifest = amplified_hf / "parity-amplification-manifest.json"
    amplification_manifest.write_text(
        json.dumps(
            {
                "schema": "qwen35-self-attn-amplified-parity-adapter-v1",
                "source": {"path": str(primary_hf.resolve()), "sha256": primary_hf_hash},
                "destination": {"path": str(amplified_hf.resolve()), "sha256": amplified_hf_hash},
                "scale": 4.0,
                "kept_self_attn_lora_b": ["layers.1.self_attn.q_proj.lora_B.default.weight"],
                "zeroed_non_self_attn_lora_b": ["layers.0.linear_attn.in_proj_qkv.lora_B.default.weight"],
            }
        )
    )
    compatibility_manifest = amplified_vllm / "compatibility-manifest.json"
    compatibility_manifest.write_text(
        json.dumps(
            {
                "schema": "qwen35-vllm-compat-adapter-v1",
                "source": {"path": str(amplified_hf.resolve()), "adapter_model_sha256": amplified_hf_hash},
                "destination": {"path": str(amplified_vllm.resolve()), "adapter_model_sha256": amplified_vllm_hash},
                "translation": {
                    "source_prefix": "base_model.model.model.layers.",
                    "destination_prefix": "base_model.model.model.language_model.layers.",
                    "tensor_count": 2,
                    "translated_tensor_count": 2,
                },
            }
        )
    )

    primary_report = tmp_path / "primary-report.json"
    primary_report.write_text(
        json.dumps(
            {
                "schema": "qwen35-lora-runtime-parity-v1",
                "model": model,
                "adapter": {
                    "path": str(checkpoint.resolve()),
                    "adapter_model_sha256": production_hash,
                    "hf_path": str(primary_hf.resolve()),
                    "hf_adapter_model_sha256": primary_hf_hash,
                },
                "results": {
                    "full": {"verdict": "hf_vllm_effects_agree"},
                    "linear_only": {"verdict": "hf_vllm_effects_agree"},
                    "evaluator_path": {"verdict": "hf_vllm_effects_agree"},
                    "self_attn_only": {
                        "verdict": "nonzero_but_delta_mismatch",
                        "summary": {"hf_max_abs": 0.4, "vllm_max_abs": 0.4, "cosine_similarity": 0.85},
                    },
                },
            }
        )
    )
    amplified_report = tmp_path / "amplified-report.json"
    amplified_report.write_text(
        json.dumps(
            {
                "schema": "qwen35-lora-runtime-parity-v1",
                "model": model,
                "adapter": {
                    "path": str(amplified_vllm.resolve()),
                    "adapter_model_sha256": amplified_vllm_hash,
                    "hf_path": str(amplified_hf.resolve()),
                    "hf_adapter_model_sha256": amplified_hf_hash,
                },
                "token_protocol": {"requested_result_variants": ["self_attn_only"]},
                "backends": {"vllm": {"enforce_eager": True, "isolate_vllm_variants": True}},
                "results": {
                    "self_attn_only": {
                        "verdict": "hf_vllm_effects_agree",
                        "summary": {"hf_max_abs": 1.6, "vllm_max_abs": 1.5, "cosine_similarity": 0.98},
                    }
                },
            }
        )
    )

    from experiments.act_repair_gate.vllm_compat_adapter import (
        attest_compat_adapter_with_amplified_self_attention,
    )

    attestation = attest_compat_adapter_with_amplified_self_attention(
        checkpoint,
        primary_report_path=primary_report,
        amplified_self_attention_report_path=amplified_report,
    )
    assert attestation["schema"] == "qwen35-vllm-composite-parity-attestation-v1"

    (checkpoint / "manifest.json").write_text(json.dumps({"backend": "local", "model": model, "lora": True}))
    from ctm.evals.local_model import native_vllm_request_identity

    assert native_vllm_request_identity(
        local_checkpoint=checkpoint,
        model_args={"provider": "vllm"},
    ) == (model, str(checkpoint.resolve()))

    # The side-car hash is bound into the attestation, so a later mutation
    # invalidates the exception rather than silently allowing a different
    # amplified proof.
    amplification_manifest.write_text("{}")
    with pytest.raises(ValueError, match="maps no `model.layers"):
        native_vllm_request_identity(local_checkpoint=checkpoint, model_args={"provider": "vllm"})


def test_generation_config_rejects_unknown_fields_and_has_portable_defaults():
    assert normalize_generation_config({}) == {"temperature": 0.0}
    with pytest.raises(ValueError, match="unknown Inspect"):
        normalize_generation_config({"max_new_tokens": 12})


def test_native_vllm_top_k_is_routed_through_extra_body_without_changing_other_providers():
    effective = effective_provider_generation_config(
        {"max_tokens": 32, "top_p": 0.95, "top_k": 20},
        model="vllm/unit/base",
    )
    assert effective["top_k"] == 20
    assert effective["extra_body"] == {"top_k": 20}

    local_effective = effective_provider_generation_config(
        {"top_k": 20},
        local_checkpoint="file:///checkpoint",
        model_args={"provider": "vllm"},
    )
    assert local_effective["extra_body"] == {"top_k": 20}

    assert "extra_body" not in effective_provider_generation_config(
        {"top_k": 20},
        model="mockllm/unit",
    )
    with pytest.raises(ValueError, match="conflicting top_k"):
        effective_provider_generation_config(
            {"top_k": 20, "extra_body": {"top_k": 7}},
            model="vllm/unit/base",
        )


def test_tinker_generation_config_rejects_fields_the_cookbook_ignores():
    validate_tinker_generation_config({"max_tokens": 12, "temperature": 0.0, "seed": 42})
    with pytest.raises(ValueError, match="frequency_penalty"):
        validate_tinker_generation_config({"frequency_penalty": 0.5})
    with pytest.raises(ValueError, match="stop_seqs"):
        resolve_eval_model(tinker_checkpoint="tinker://x", generation_config={"stop_seqs": ["END"]})


def test_eval_runner_records_canonical_provenance(monkeypatch):
    captured = {}
    monkeypatch.setattr(runner_module, "build_tasks", lambda *args, **kwargs: ["task"])
    monkeypatch.setattr(runner_module, "resolve_eval_model", lambda **kwargs: "resolved-model")

    import inspect_ai

    def fake_eval(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(status="success")]

    monkeypatch.setattr(inspect_ai, "eval", fake_eval)
    logs = run_task_evals(
        "upstream.tasks:suite",
        model="mockllm/unit",
        task_args={"slice": "heldout"},
        model_args={"base_url": "http://localhost", "api_key": "must-redact"},
        generation_config={
            "max_tokens": 12,
            "extra_headers": {"Authorization": "must-redact"},
        },
        metadata={
            "task_factory": "spoofed",
            "selection_candidate": {"domain": "sycophancy", "candidate_id": "unit"},
        },
        max_tasks=3,
    )
    assert logs[0].status == "success"
    metadata = captured["metadata"]
    assert metadata["task_factory"] == "upstream.tasks:suite"
    assert metadata["task_args"] == {"slice": "heldout"}
    assert metadata["model"] == "mockllm/unit"
    assert metadata["model_args"]["api_key"] == "<redacted>"
    assert metadata["generation_config"]["max_tokens"] == 12
    assert metadata["generation_config"]["extra_headers"] == "<redacted>"
    assert metadata["include_reasoning"] is False
    assert metadata["selection_candidate"] == {"domain": "sycophancy", "candidate_id": "unit"}
    assert metadata["max_tasks"] == 3
    assert captured["max_tasks"] == 3


def test_eval_runner_records_tinker_reasoning_mode(monkeypatch):
    captured = {}
    resolved = SimpleNamespace(api=SimpleNamespace(model_name="unit/base", renderer_name="unit_renderer"))
    monkeypatch.setattr(runner_module, "build_tasks", lambda *args, **kwargs: ["task"])
    monkeypatch.setattr(runner_module, "resolve_eval_model", lambda **kwargs: resolved)

    import inspect_ai

    monkeypatch.setattr(
        inspect_ai,
        "eval",
        lambda **kwargs: captured.update(kwargs) or [SimpleNamespace(status="success")],
    )
    run_task_evals(
        "upstream.tasks:suite",
        tinker_checkpoint="tinker://run/sampler_weights/final",
        include_reasoning=True,
    )
    assert captured["metadata"]["include_reasoning"] is True
    assert captured["metadata"]["checkpoint"] == "tinker://run/sampler_weights/final"


def test_eval_runner_passes_only_selected_original_tasks(monkeypatch):
    captured = {}
    tasks = [object(), object(), object()]
    monkeypatch.setattr(runner_module, "build_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(runner_module, "resolve_eval_model", lambda **kwargs: "resolved-model")

    import inspect_ai

    monkeypatch.setattr(
        inspect_ai,
        "eval",
        lambda **kwargs: captured.update(kwargs) or [SimpleNamespace(status="success")],
    )
    run_task_evals("upstream.tasks:suite", model="mockllm/unit", task_indices=[2])
    assert captured["tasks"] == [tasks[1]]
    assert captured["metadata"]["task_indices"] == [2]
    assert captured["metadata"]["task_count"] == 3


def test_eval_cli_rejects_inline_api_keys_before_confirmation():
    from scripts.run_evals import main

    with pytest.raises(SystemExit):
        main(
            [
                "--task-factory",
                "mcq_bias.tasks:suite_tasks",
                "--model",
                "mockllm/unit",
                "--model-args",
                '{"api_key":"do-not-print"}',
            ]
        )


@pytest.mark.parametrize(
    ("flag", "config"),
    [
        ("--task-args", '{"headers":{"X-Custom":"ultra-secret"}}'),
        ("--model-args", '{"proxy-authorization":"ultra-secret"}'),
        ("--metadata", '{"credentials":{"token":"ultra-secret"}}'),
        ("--generation-config", '{"extra_headers":{"Authorization":"ultra-secret"}}'),
    ],
)
def test_eval_cli_rejects_secrets_in_every_printed_config(flag, config, capsys):
    from scripts.run_evals import main

    with pytest.raises(SystemExit):
        main(["--task-factory", "mcq_bias.tasks:suite_tasks", "--model", "mockllm/unit", flag, config])
    captured = capsys.readouterr()
    assert "ultra-secret" not in captured.out
    assert "ultra-secret" not in captured.err


def test_eval_cli_defers_task_construction_until_after_confirmation(monkeypatch, capsys):
    from scripts.run_evals import main

    monkeypatch.setattr("builtins.input", lambda _: "n")
    main(
        [
            "--task-factory",
            "mcq_bias.tasks:suite_tasks",
            "--model",
            "mockllm/unit",
            "--limit",
            "2",
        ]
    )
    output = capsys.readouterr().out
    assert "preflight_samples=deferred" in output
    assert "bound source samples per task" in output


def test_eval_cli_dry_run_constructs_neither_tasks_nor_models(monkeypatch, capsys):
    from scripts import run_evals

    monkeypatch.setattr(
        run_evals,
        "run_task_evals",
        lambda *_args, **_kwargs: pytest.fail("dry run started evaluation"),
    )
    run_evals.main(
        [
            "--task-factory",
            "mcq_bias.tasks:suite_tasks",
            "--model",
            "mockllm/unit",
            "--dry-run",
        ]
    )

    assert "Dry run complete; no task or model was constructed." in capsys.readouterr().out


def test_eval_cli_isolates_selected_tasks_in_fresh_children(monkeypatch):
    from scripts import run_evals as cli

    calls = []
    monkeypatch.setattr(cli, "build_tasks", lambda *args, **kwargs: ["one", "two", "three"])
    monkeypatch.setattr(
        cli,
        "run_task_evals",
        lambda *args, **kwargs: pytest.fail("the isolation parent must not run Inspect"),
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )

    cli.main(
        [
            "--task-factory",
            "unit.tasks:suite",
            "--model",
            "mockllm/unit",
            "--isolate-tasks",
            "--task-index",
            "1",
            "--task-index=3",
            "--yes",
        ]
    )

    assert len(calls) == 2
    for expected_index, (command, kwargs) in zip((1, 3), calls, strict=True):
        assert "--isolate-tasks" not in command
        assert command.count("--task-index") == 1
        assert command[command.index("--task-index") + 1] == str(expected_index)
        assert kwargs == {"cwd": cli.PROJECT_ROOT, "check": False}


def test_eval_cli_reports_exact_failed_isolated_task_resume(monkeypatch):
    from scripts import run_evals as cli

    monkeypatch.setattr(cli, "build_tasks", lambda *args, **kwargs: ["one", "two"])
    return_codes = iter((0, 17))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=next(return_codes)),
    )

    with pytest.raises(SystemExit, match=r"(?s)isolated task 2/2 exited.*--task-index 2"):
        cli.main(
            [
                "--task-factory",
                "unit.tasks:suite",
                "--model",
                "mockllm/unit",
                "--isolate-tasks",
                "--yes",
            ]
        )


def test_isolated_crash_accepts_only_one_new_exact_success_header(monkeypatch):
    from inspect_ai import log as inspect_log
    from scripts import run_evals as cli

    old = SimpleNamespace(name="old.eval")
    new = SimpleNamespace(name="new.eval")
    header = SimpleNamespace(
        status="success",
        eval=SimpleNamespace(metadata={"task_indices": [2], "task_count": 7}),
    )
    monkeypatch.setattr(inspect_log, "list_eval_logs", lambda *args, **kwargs: [old, new])
    monkeypatch.setattr(inspect_log, "read_eval_log", lambda *args, **kwargs: header)

    assert (
        cli._successful_isolated_log(
            "logs/evals",
            previous_logs={"old.eval"},
            task_index=2,
            task_count=7,
        )
        == "new.eval"
    )

    header.eval.metadata["task_indices"] = [3]
    assert (
        cli._successful_isolated_log(
            "logs/evals",
            previous_logs={"old.eval"},
            task_index=2,
            task_count=7,
        )
        is None
    )


def test_eval_cli_accepts_verified_success_log_after_late_child_crash(monkeypatch, capsys):
    from scripts import run_evals as cli

    monkeypatch.setattr(cli, "build_tasks", lambda *args, **kwargs: ["one"])
    monkeypatch.setattr(cli, "_eval_log_names", lambda log_dir: {"old.eval"})
    monkeypatch.setattr(
        cli,
        "_successful_isolated_log",
        lambda *args, **kwargs: "file:///logs/new-success.eval",
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=17),
    )

    cli.main(
        [
            "--task-factory",
            "unit.tasks:suite",
            "--model",
            "mockllm/unit",
            "--isolate-tasks",
            "--yes",
        ]
    )

    output = capsys.readouterr().out
    assert "after writing a verified successful log; continuing" in output
    assert "new-success.eval" in output
