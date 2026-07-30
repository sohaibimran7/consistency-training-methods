"""Offline tests for the task-factory Inspect bridge and Tinker resolution."""

import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import GenerateConfig, ModelAPI

from ctm.evals import runner as runner_module
from ctm.evals.runner import (
    build_tasks,
    load_task_factory,
    normalize_generation_config,
    parse_json_object,
    resolve_eval_model,
    run_task_evals,
    validate_tinker_generation_config,
)
from ctm.evals import tinker_model as tinker_model_module
from ctm.evals.tinker_model import tinker_base_model, tinker_checkpoint_model
from ctm.evals import local_model as local_model_module
from ctm.evals.local_model import read_local_checkpoint


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


def test_normal_inspect_model_resolution_uses_provider_registry():
    model = resolve_eval_model(model="mockllm/unit", generation_config={"max_tokens": 37})
    assert model.api.model_name == "unit"
    assert model.config.max_tokens == 37
    assert model.config.temperature == 0.0


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


def test_generation_config_rejects_unknown_fields_and_has_portable_defaults():
    assert normalize_generation_config({}) == {"temperature": 0.0}
    with pytest.raises(ValueError, match="unknown Inspect"):
        normalize_generation_config({"max_new_tokens": 12})


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
        metadata={"task_factory": "spoofed"},
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
