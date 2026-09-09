"""Offline contract tests for parent-owned vLLM evaluation servers."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from inspect_ai.model import GenerateConfig

from ctm.evals import local_model as local_model_module
from ctm.evals.local_model import (
    PERSISTENT_VLLM_CHILD_METADATA_ENV,
    PersistentNativeVLLMServer,
    native_vllm_request_identity,
)


class _Process:
    def __init__(self) -> None:
        self.pid = 7319
        self.returncode = None

    def poll(self):
        return self.returncode


class _Response:
    def __init__(self, model_ids):
        self._model_ids = model_ids

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {"data": [{"id": model_id} for model_id in self._model_ids]}


class _API:
    def __init__(self, *, adapter_name: str | None) -> None:
        self._init_base_url = None
        self.api_key = "unit-secret-key"
        self.base_model = "unit/base"
        self.adapter = SimpleNamespace(name=adapter_name) if adapter_name is not None else None
        self.server_args = {"gpu_memory_utilization": 0.9, "nested": {"api_key": "redact-me"}}
        self._server = SimpleNamespace(process=None, base_url=None, api_key=None)
        self.process = _Process()
        self.start_calls = 0
        self.close_calls = 0

    def _resolve_server(self) -> None:
        self.start_calls += 1
        logging.getLogger("inspect_ai._util.local_server").info("launch with %s", self.api_key)
        self._server.process = self.process
        self._server.base_url = "http://127.0.0.1:8123/v1"
        self._server.api_key = self.api_key

    def close(self) -> None:
        self.close_calls += 1
        self.process.returncode = 0


@pytest.mark.parametrize("adapter_name", [None, "/exact/checkpoint/adapter"])
def test_persistent_native_server_records_exact_base_or_lora_identity(tmp_path, monkeypatch, adapter_name):
    api = _API(adapter_name=adapter_name)
    monkeypatch.setattr(local_model_module, "_require_native_vllm_api", lambda model: model.api)
    expected_ids = {api.base_model, adapter_name or api.base_model}
    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: _Response(expected_ids))

    with PersistentNativeVLLMServer.start(
        SimpleNamespace(api=api),
        log_dir=tmp_path,
        source_metadata={"model_args": api.server_args},
    ) as server:
        assert api.start_calls == 1
        assert server.base_model == "unit/base"
        assert server.adapter_name == adapter_name
        assert server.served_model == (adapter_name or "unit/base")
        child_env = server.child_environment({"INHERITED": "yes"})
        assert child_env["VLLM_BASE_URL"] == "http://127.0.0.1:8123/v1"
        assert child_env["VLLM_API_KEY"] == "unit-secret-key"
        child_metadata = json.loads(child_env[PERSISTENT_VLLM_CHILD_METADATA_ENV])
        assert child_metadata["adapter"] == adapter_name
        assert child_metadata["served_model"] == (adapter_name or "unit/base")

        metadata = json.loads((tmp_path / "vllm-server.json").read_text())
        assert metadata["status"] == "ready"
        assert metadata["adapter"] == adapter_name
        assert metadata["served_model"] == (adapter_name or "unit/base")
        assert metadata["server_args"]["nested"]["api_key"] == "<redacted>"

    assert api.close_calls == 1
    assert json.loads((tmp_path / "vllm-server.json").read_text())["status"] == "completed"
    server_log = (tmp_path / "vllm-server.log").read_text()
    assert "unit-secret-key" not in server_log
    assert "<redacted>" in server_log


def test_native_vllm_request_identity_canonicalizes_local_lora(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"backend": "local", "model": "unit/base", "lora": True}))
    (checkpoint / "adapter_config.json").write_text("{}")

    assert native_vllm_request_identity(model="vllm/unit/base") == ("unit/base", None)
    assert native_vllm_request_identity(
        local_checkpoint=checkpoint,
        model_args={"provider": "vllm"},
    ) == ("unit/base", str(checkpoint.resolve()))


def test_native_vllm_request_identity_refuses_qwen35_lora_noop_route(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()

    with pytest.raises(ValueError, match="maps no `model.layers"):
        native_vllm_request_identity(model=f"vllm/Qwen/Qwen3.5-9B:{adapter}")


class _OwnedServer:
    instances = []

    def __init__(self) -> None:
        self.health_calls = 0
        self.closed = False
        self.source_metadata = None

    @classmethod
    def start(cls, model, **kwargs):
        instance = cls()
        instance.model = model
        instance.source_metadata = kwargs["source_metadata"]
        cls.instances.append(instance)
        return instance

    def assert_healthy(self) -> None:
        self.health_calls += 1

    def child_environment(self):
        return {
            "VLLM_BASE_URL": "http://127.0.0.1:9000/v1",
            "VLLM_API_KEY": "child-key",
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False


@pytest.mark.parametrize(
    ("model_argv", "expected_resolution"),
    [
        (["--model", "vllm/unit/base"], {"model": "vllm/unit/base"}),
        (
            [
                "--local-checkpoint",
                "/unit/checkpoint",
                "--base-model",
                "unit/base",
                "--model-args",
                '{"provider":"vllm","gpu_memory_utilization":0.9}',
            ],
            {"local_checkpoint": "/unit/checkpoint", "base_model": "unit/base"},
        ),
    ],
)
def test_eval_cli_starts_one_persistent_server_for_all_isolated_tasks(monkeypatch, model_argv, expected_resolution):
    from scripts import run_evals as cli

    _OwnedServer.instances.clear()
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv(PERSISTENT_VLLM_CHILD_METADATA_ENV, raising=False)
    monkeypatch.setattr(cli, "PersistentNativeVLLMServer", _OwnedServer)
    monkeypatch.setattr(cli, "build_tasks", lambda *args, **kwargs: ["one", "two"])
    resolutions = []
    monkeypatch.setattr(
        cli,
        "resolve_eval_model",
        lambda **kwargs: resolutions.append(kwargs) or "parent-model",
    )
    monkeypatch.setattr(cli, "_eval_log_names", lambda log_dir: set())
    child_calls = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda command, **kwargs: child_calls.append((command, kwargs)) or SimpleNamespace(returncode=0),
    )

    cli.main(
        [
            "--task-factory",
            "unit.tasks:suite",
            *model_argv,
            "--isolate-tasks",
            "--persistent-vllm-server",
            "--yes",
        ]
    )

    assert len(resolutions) == 1
    for key, value in expected_resolution.items():
        assert resolutions[0][key] == value
    assert len(_OwnedServer.instances) == 1
    assert _OwnedServer.instances[0].closed is True
    assert len(child_calls) == 2
    for task_index, (command, options) in enumerate(child_calls, start=1):
        assert "--isolate-tasks" not in command
        assert "--persistent-vllm-server" not in command
        assert command[command.index("--task-index") + 1] == str(task_index)
        assert options["env"]["VLLM_BASE_URL"] == "http://127.0.0.1:9000/v1"
        assert options["env"]["VLLM_API_KEY"] == "child-key"


def test_healthy_server_survives_late_child_failure_and_serves_next_task(monkeypatch):
    from scripts import run_evals as cli

    server = _OwnedServer()
    returncodes = iter((19, 0))
    child_calls = []
    monkeypatch.setattr(cli, "_eval_log_names", lambda log_dir: set())
    monkeypatch.setattr(
        cli,
        "_successful_isolated_log",
        lambda *args, **kwargs: "logs/task-1.eval" if kwargs["task_index"] == 1 else None,
    )
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: child_calls.append((args, kwargs)) or SimpleNamespace(returncode=next(returncodes)),
    )

    cli._run_isolated_tasks(
        [
            "--task-factory",
            "unit.tasks:suite",
            "--model",
            "vllm/unit/base",
            "--isolate-tasks",
            "--persistent-vllm-server",
            "--yes",
        ],
        log_dir="logs/unit",
        task_count=2,
        task_indices=None,
        persistent_server=server,
    )

    assert len(child_calls) == 2
    assert server.health_calls == 3  # before each task and after the failed child


def test_unhealthy_server_blocks_late_crash_acceptance_and_persistent_resume(monkeypatch):
    from scripts import run_evals as cli

    class UnhealthyAfterChild(_OwnedServer):
        def assert_healthy(self):
            self.health_calls += 1
            if self.health_calls > 1:
                raise RuntimeError("endpoint lost exact adapter")

    server = UnhealthyAfterChild()
    monkeypatch.setattr(cli, "_eval_log_names", lambda log_dir: set())
    monkeypatch.setattr(cli, "_successful_isolated_log", lambda *args, **kwargs: "logs/task.eval")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=19))

    with pytest.raises(SystemExit) as raised:
        cli._run_isolated_tasks(
            [
                "--task-factory",
                "unit.tasks:suite",
                "--model",
                "vllm/unit/base",
                "--isolate-tasks",
                "--persistent-vllm-server",
                "--yes",
            ],
            log_dir="logs/unit",
            task_count=1,
            task_indices=None,
            persistent_server=server,
        )

    message = str(raised.value)
    assert "server is unhealthy" in message
    assert "--isolate-tasks" in message
    assert "--persistent-vllm-server" in message
    assert "--task-index 1" in message


def test_persistent_parent_cleans_up_when_child_fails(monkeypatch):
    from scripts import run_evals as cli

    _OwnedServer.instances.clear()
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv(PERSISTENT_VLLM_CHILD_METADATA_ENV, raising=False)
    monkeypatch.setattr(cli, "PersistentNativeVLLMServer", _OwnedServer)
    monkeypatch.setattr(cli, "resolve_eval_model", lambda **kwargs: "parent-model")
    monkeypatch.setattr(cli, "build_tasks", lambda *args, **kwargs: ["one"])
    monkeypatch.setattr(cli, "_eval_log_names", lambda log_dir: set())
    monkeypatch.setattr(cli, "_successful_isolated_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=23))

    with pytest.raises(SystemExit, match="isolated task 1/1 exited"):
        cli.main(
            [
                "--task-factory",
                "unit.tasks:suite",
                "--model",
                "vllm/unit/base",
                "--isolate-tasks",
                "--persistent-vllm-server",
                "--yes",
            ]
        )

    assert len(_OwnedServer.instances) == 1
    assert _OwnedServer.instances[0].closed is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--model", "vllm/unit/base", "--persistent-vllm-server"],
        ["--model", "mockllm/unit", "--isolate-tasks", "--persistent-vllm-server"],
        [
            "--local-checkpoint",
            "/unit/checkpoint",
            "--isolate-tasks",
            "--persistent-vllm-server",
        ],
        [
            "--tinker-base-model",
            "unit/base",
            "--isolate-tasks",
            "--persistent-vllm-server",
        ],
        [
            "--model",
            "vllm/unit/base",
            "--model-args",
            '{"base_url":"http://127.0.0.1:9000/v1"}',
            "--isolate-tasks",
            "--persistent-vllm-server",
        ],
    ],
)
def test_persistent_eval_mode_rejects_unsupported_or_unowned_modes(monkeypatch, argv):
    from scripts import run_evals as cli

    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv(PERSISTENT_VLLM_CHILD_METADATA_ENV, raising=False)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--task-factory", "unit.tasks:suite", *argv, "--yes"])
    assert raised.value.code == 2


def test_effective_vllm_request_forwards_top_k_through_extra_body():
    """Pin the effective protocol rather than Inspect's ignored top-level field."""

    from inspect_ai.model._providers.vllm import VLLMAPI
    from ctm.evals.runner import effective_provider_generation_config

    api = VLLMAPI("unit/top-k-contract", base_url="http://127.0.0.1:1/v1")
    try:
        api._resolve_server()
        config = effective_provider_generation_config(
            {"max_tokens": 32, "temperature": 0.0, "top_p": 0.95, "top_k": 20},
            model="vllm/unit/top-k-contract",
        )
        params = api.completion_params(
            GenerateConfig(**config),
            tools=False,
        )
        assert params["top_p"] == 0.95
        assert params["extra_body"]["top_k"] == 20
    finally:
        api.close()
