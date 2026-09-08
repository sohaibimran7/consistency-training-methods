"""Command-line lifecycle tests for the shared Isambard jobs controller."""

from __future__ import annotations

import json
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from infra.isambard import jobs


class FakeBackend:
    instances = []

    def __init__(self, host: str) -> None:
        self.host = host
        self.scope = "fake.isambard:22/alice"
        self.canonicalized = []
        FakeBackend.instances.append(self)

    def canonicalize(self, request: dict) -> dict:
        self.canonicalized.append(json.loads(json.dumps(request)))
        return json.loads(json.dumps(request))

    def snapshot(self, requests: list) -> list:
        raise AssertionError("status/enqueue must not query the scheduler")

    def submit(self, request: dict) -> str:
        raise AssertionError("enqueue must not submit directly")

    def cancel(self, job_id: str) -> None:
        raise AssertionError("not used")


def _request(
    *, request_id: str, owner: str, remote_dir: str, output_root: str, mode: str, minutes: int, env: dict
) -> dict:
    return {
        "id": request_id,
        "owner": owner,
        "mode": mode,
        "remote_dir": remote_dir,
        "output_roots": [output_root],
        "script": "#!/usr/bin/env bash\nprintf '%s\\n' program-body-not-public\n",
        "resources": {"nodes": 1, "gpus": 1, "minutes": minutes, "memory_mb": 1_024, "cpus_per_task": 4},
        "env": env,
    }


def test_prepare_enqueue_status_and_start_preserve_a_private_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    output_root = remote / "campaign"
    manifest = tmp_path / "requests" / "request.json"
    state_dir = tmp_path / "state"
    built = []
    started = []
    FakeBackend.instances = []

    adapter = types.ModuleType("infra.isambard.job_adapters")

    def build_request(profile, **kwargs):
        built.append((profile, kwargs))
        return _request(
            request_id=kwargs["request_id"],
            owner=kwargs["owner"],
            remote_dir=kwargs["remote_dir"],
            output_root=kwargs["output_root"],
            mode=kwargs["mode"],
            minutes=kwargs["minutes"],
            env=kwargs["env"],
        )

    adapter.build_request = build_request
    monkeypatch.setitem(sys.modules, "infra.isambard.job_adapters", adapter)
    monkeypatch.setattr(jobs, "SSHBackend", FakeBackend)
    monkeypatch.setattr(jobs, "start_runner", lambda args: started.append(args) or 4242)

    assert (
        jobs.main(
            [
                "--state-dir",
                str(state_dir),
                "prepare",
                "rmct-r5-interactive",
                "--id",
                "rmct-001",
                "--owner",
                "agent-a",
                "--checkout",
                str(checkout),
                "--remote-dir",
                str(remote),
                "--output-root",
                str(output_root),
                "--mode",
                "interactive",
                "--minutes",
                "30",
                "--env",
                "REPO_DIR=" + str(remote),
                "--env",
                "HARMLESS_VALUE=must-not-appear-in-console",
                "--output",
                str(manifest),
            ]
        )
        == 0
    )
    prepared_output = capsys.readouterr().out
    prepared = json.loads(prepared_output)
    assert prepared["manifest"] == str(manifest.resolve())
    assert "script" not in prepared["request"]
    assert "env" not in prepared["request"]
    assert "program-body-not-public" not in prepared_output
    assert "must-not-appear-in-console" not in prepared_output
    assert built[0][0] == "rmct-r5-interactive"
    assert built[0][1]["checkout"] == checkout

    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored["script"].endswith("program-body-not-public\n")
    assert stored["env"]["HARMLESS_VALUE"] == "must-not-appear-in-console"
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600

    assert jobs.main(["--state-dir", str(state_dir), "enqueue", str(manifest), "--start"]) == 0
    enqueue_output = capsys.readouterr().out
    enqueued = json.loads(enqueue_output)
    assert enqueued["runner_pid"] == 4242
    assert enqueued["request"]["requests"][0]["id"] == "rmct-001"
    assert "script" not in enqueued["request"]["requests"][0]
    assert "env" not in enqueued["request"]["requests"][0]
    assert "program-body-not-public" not in enqueue_output
    assert "must-not-appear-in-console" not in enqueue_output
    assert len(FakeBackend.instances) == 1
    assert FakeBackend.instances[0].canonicalized[0]["id"] == "rmct-001"
    assert len(started) == 1

    assert jobs.main(["--state-dir", str(state_dir), "status"]) == 0
    status_output = capsys.readouterr().out
    status = json.loads(status_output)
    assert status["requests"][0]["id"] == "rmct-001"
    assert status["requests"][0]["status"] == "queued"
    assert "script" not in status["requests"][0]
    assert "env" not in status["requests"][0]
    assert "program-body-not-public" not in status_output
    assert "must-not-appear-in-console" not in status_output

    assert jobs.main(["--state-dir", str(state_dir), "start"]) == 0
    started_output = json.loads(capsys.readouterr().out)
    assert started_output["runner_pid"] == 4242
    assert len(started) == 2


class _LoopController:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.tick_calls = 0

    def tick(self) -> dict:
        self.tick_calls += 1
        return self.state

    def status(self) -> dict:
        return self.state


def test_runner_uses_controller_status_field_and_pauses_on_snapshot_error(tmp_path: Path) -> None:
    backend = SimpleNamespace(scope="fake.isambard:22/alice")
    completed = _LoopController({"requests": [{"id": "done", "status": "terminal"}]})

    idle = jobs.run_loop(
        completed,
        backend,
        SimpleNamespace(state_dir=tmp_path / "idle-state", duration=1),
    )

    assert idle == {"runner": "idle"}
    assert completed.tick_calls == 1

    blocked = _LoopController(
        {
            "requests": [{"id": "waiting", "status": "queued"}],
            "last_snapshot_error": "authentication needs attention",
        }
    )
    paused = jobs.run_loop(
        blocked,
        backend,
        SimpleNamespace(state_dir=tmp_path / "blocked-state", duration=1),
    )

    assert paused == {
        "runner": "paused",
        "reason": "authentication needs attention",
        "resume": "isambard-jobs start",
    }
    assert blocked.tick_calls == 1
