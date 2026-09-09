"""Black-box tests for the small remote scheduler transport program."""

from __future__ import annotations

import base64
import json
import os
import shlex
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import pytest

from infra.isambard.job_transport import REMOTE_PROGRAM, SSHBackend, validate_request

_FAKE_SCHEDULER = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
record = {
    "program": name,
    "argv": sys.argv[1:],
    "environment": {
        key: os.environ.get(key)
        for key in (
            "SBATCH_SHOULD_NOT_LEAK",
            "USER_SETTING",
            "REPO_DIR",
            "CTM_ISAMBARD_REQUEST_ID",
        )
    },
}
if name == "sbatch":
    path = Path(sys.argv[-1])
    if path.exists():
        record["script"] = path.read_text(encoding="utf-8")
with Path(os.environ["FAKE_SCHEDULER_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\\n")

if name == "sbatch":
    print("placement-ok" if "--test-only" in sys.argv else "4182;fake-cluster")
elif name == "squeue":
    sys.stdout.write(os.environ.get("FAKE_SQUEUE_OUTPUT", ""))
    if "--all" in sys.argv:
        sys.stdout.write(os.environ.get("FAKE_HIDDEN_SQUEUE_OUTPUT", ""))
elif name == "sacct":
    sys.stdout.write(os.environ.get("FAKE_SACCT_OUTPUT", ""))
elif name == "scancel":
    print("cancelled")
else:
    raise SystemExit("unexpected scheduler command: " + name)
"""


def _fake_scheduler(tmp_path: Path) -> tuple[Path, Path]:
    binaries = tmp_path / "fake-bin"
    binaries.mkdir()
    log = tmp_path / "scheduler.jsonl"
    for name in ("sbatch", "squeue", "sacct", "scancel"):
        path = binaries / name
        path.write_text(_FAKE_SCHEDULER, encoding="utf-8")
        path.chmod(0o755)
    return binaries, log


def _invoke_remote(
    payload: dict,
    *,
    binaries: Path,
    log: Path,
    extra_environment: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PATH"] = str(binaries) + os.pathsep + environment.get("PATH", "")
    environment["FAKE_SCHEDULER_LOG"] = str(log)
    if extra_environment:
        environment.update(extra_environment)
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return subprocess.run(
        [sys.executable, "-c", REMOTE_PROGRAM, encoded],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _records(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def _request(remote_dir: Path, *, mode: str, resources: dict) -> dict:
    return {
        "id": "request-001",
        "owner": "agent-a",
        "mode": mode,
        "remote_dir": str(remote_dir),
        "output_roots": [str(remote_dir / "outputs" / "campaign")],
        "token": "ctm-submit-token-001",
        "resources": resources,
        "env": {
            "REPO_DIR": str(remote_dir),
            "USER_SETTING": "value with spaces; $metacharacter",
        },
        "script": "#!/usr/bin/env bash\nprintf '%s\\n' worker-body\n",
    }


def test_submit_uses_exact_interactive_sbatch_contract_and_keeps_values_out_of_stdout(tmp_path: Path) -> None:
    binaries, log = _fake_scheduler(tmp_path)
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    request = _request(
        remote_dir,
        mode="interactive",
        resources={"nodes": 1, "gpus": 4, "minutes": 480, "memory_mb": 200_000, "cpus_per_task": 64},
    )

    completed = _invoke_remote(
        {"action": "submit", "user": "alice", "request": request},
        binaries=binaries,
        log=log,
        extra_environment={"SBATCH_SHOULD_NOT_LEAK": "caller-value"},
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"job_id": "4182"}
    assert "worker-body" not in completed.stdout
    assert "value with spaces" not in completed.stdout

    calls = _records(log)
    assert len(calls) == 2
    assert all(call["program"] == "sbatch" for call in calls)
    spool = remote_dir / ".isambard-jobs" / request["token"]
    script_path = str(spool / "job.sh")
    common = [
        "--parsable",
        "--job-name",
        request["token"],
        "--nodes",
        "1",
        "--gpus",
        "4",
        "--time",
        "480",
        "--mem",
        "200000M",
        "--chdir",
        str(remote_dir),
        "--output",
        str(spool / "slurm-%j.out"),
        "--error",
        str(spool / "slurm-%j.err"),
        "--export=NONE",
        "--no-requeue",
        "--open-mode=append",
        "--cpus-per-task",
        "64",
        "--reservation=interactive",
    ]
    assert calls[0]["argv"] == common + ["--test-only", script_path]
    assert calls[1]["argv"] == common + [script_path]
    for call in calls:
        assert "--partition=workq" not in call["argv"]
        assert not any("qos" in item.lower() for item in call["argv"])
        assert call["environment"] == {
            "SBATCH_SHOULD_NOT_LEAK": None,
            "USER_SETTING": None,
            "REPO_DIR": None,
            "CTM_ISAMBARD_REQUEST_ID": None,
        }

    captured = calls[0]["script"]
    assert captured.startswith("#!/usr/bin/env bash\nset -euo pipefail\numask 077\n")
    assert "export REPO_DIR=" + shlex.quote(str(remote_dir)) in captured
    assert "export USER_SETTING='value with spaces; $metacharacter'" in captured
    assert "export SLURM_EXPORT_ENV=ALL" in captured
    assert "export CTM_ISAMBARD_REQUEST_ID=request-001" in captured
    assert "export CTM_ISAMBARD_OWNER=agent-a" in captured
    assert "unset CUDA_VISIBLE_DEVICES\n" not in captured
    assert captured.endswith(request["script"])
    assert stat.S_IMODE((spool / "job.sh").stat().st_mode) == 0o600
    receipt = json.loads((spool / "submission.json").read_text(encoding="utf-8"))
    assert receipt["job_id"] == "4182"
    assert receipt["token"] == request["token"]


def test_submit_preserves_full_batch_topology_without_interactive_options(tmp_path: Path) -> None:
    binaries, log = _fake_scheduler(tmp_path)
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    request = _request(
        remote_dir,
        mode="batch",
        resources={
            "nodes": 4,
            "gpus": 16,
            "minutes": 720,
            "memory_mb": 400_000,
            "cpus_per_gpu": 16,
            "ntasks": 16,
            "ntasks_per_node": 4,
            "gpus_per_node": 4,
        },
    )

    completed = _invoke_remote({"action": "submit", "user": "alice", "request": request}, binaries=binaries, log=log)

    assert completed.returncode == 0, completed.stderr
    actual = _records(log)[1]["argv"]
    assert actual == [
        "--parsable",
        "--job-name",
        request["token"],
        "--nodes",
        "4",
        "--gpus",
        "16",
        "--time",
        "720",
        "--mem",
        "400000M",
        "--chdir",
        str(remote_dir),
        "--output",
        str(remote_dir / ".isambard-jobs" / request["token"] / "slurm-%j.out"),
        "--error",
        str(remote_dir / ".isambard-jobs" / request["token"] / "slurm-%j.err"),
        "--export=NONE",
        "--no-requeue",
        "--open-mode=append",
        "--cpus-per-gpu",
        "16",
        "--ntasks",
        "16",
        "--ntasks-per-node",
        "4",
        "--gpus-per-node",
        "4",
        "--partition=workq",
        str(remote_dir / ".isambard-jobs" / request["token"] / "job.sh"),
    ]
    assert "--reservation=interactive" not in actual
    assert not any("qos" in item.lower() for item in actual)
    assert "unset CUDA_VISIBLE_DEVICES\n" not in _records(log)[0]["script"]


def test_cpu_only_batch_omits_all_gpu_sbatch_flags(tmp_path: Path) -> None:
    binaries, log = _fake_scheduler(tmp_path)
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    request = _request(
        remote_dir,
        mode="batch",
        resources={
            "nodes": 1,
            "gpus": 0,
            "minutes": 360,
            "memory_mb": 65_536,
            "cpus_per_task": 16,
            "ntasks": 1,
            "ntasks_per_node": 1,
        },
    )

    completed = _invoke_remote({"action": "submit", "user": "alice", "request": request}, binaries=binaries, log=log)

    assert completed.returncode == 0, completed.stderr
    calls = _records(log)
    assert len(calls) == 2
    for call in calls:
        assert "--gpus" not in call["argv"]
        assert "--cpus-per-gpu" not in call["argv"]
        assert "--gpus-per-node" not in call["argv"]
        assert "--partition=workq" in call["argv"]
        assert "--reservation=interactive" not in call["argv"]
        assert "--no-requeue" in call["argv"]
    assert calls[0]["argv"][-2:] == ["--test-only", str(remote_dir / ".isambard-jobs" / request["token"] / "job.sh")]
    assert calls[1]["argv"][-1] == str(remote_dir / ".isambard-jobs" / request["token"] / "job.sh")
    assert "--cpus-per-task" in calls[1]["argv"]
    assert "16" in calls[1]["argv"]
    assert "--ntasks-per-node" in calls[1]["argv"]
    captured = calls[0]["script"]
    assert "unset CUDA_VISIBLE_DEVICES\n" in captured
    assert captured.index("unset CUDA_VISIBLE_DEVICES\n") > captured.index("export USER_SETTING=")
    assert captured.index("unset CUDA_VISIBLE_DEVICES\n") < captured.index(request["script"])


@pytest.mark.parametrize(
    ("mode", "gpus", "extra"),
    [
        ("batch", -1, {}),
        ("batch", True, {}),
        ("interactive", 0, {}),
        ("batch", 0, {"cpus_per_gpu": 1}),
        ("batch", 0, {"cpus_per_gpu": 0}),
        ("batch", 0, {"gpus_per_node": 1}),
        ("batch", 0, {"gpus_per_node": 0}),
    ],
)
def test_transport_validation_rejects_invalid_cpu_only_gpu_shapes(tmp_path: Path, mode, gpus, extra) -> None:
    candidate = _request(
        tmp_path / "remote",
        mode=mode,
        resources={"nodes": 1, "gpus": gpus, "minutes": 60, "memory_mb": 1024, "cpus_per_task": 1, **extra},
    )

    with pytest.raises(ValueError):
        validate_request(candidate)


def test_canonicalize_resolves_remote_checkout_and_output_aliases(tmp_path: Path) -> None:
    binaries, log = _fake_scheduler(tmp_path)
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    campaign = remote_dir / "campaign"
    campaign.mkdir()
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(remote_dir, target_is_directory=True)

    completed = _invoke_remote(
        {
            "action": "canonicalize",
            "user": "alice",
            "request": {
                "remote_dir": str(alias),
                "output_roots": [str(alias / "unused" / ".." / "campaign")],
            },
        },
        binaries=binaries,
        log=log,
    )

    assert completed.returncode == 0, completed.stderr
    canonical = json.loads(completed.stdout)
    assert canonical["remote_dir"] == str(remote_dir.resolve())
    assert canonical["output_roots"] == [str(campaign.resolve())]
    assert _records(log) == []


def test_snapshot_includes_interactive_jobs_in_hidden_partitions(tmp_path: Path) -> None:
    binaries, log = _fake_scheduler(tmp_path)
    completed = _invoke_remote(
        {"action": "snapshot", "user": "alice", "requests": []},
        binaries=binaries,
        log=log,
        extra_environment={"FAKE_HIDDEN_SQUEUE_OUTPUT": "991|external-debug|RUNNING|interactive|alice\n"},
    )
    assert completed.returncode == 0, completed.stderr
    records = json.loads(completed.stdout)
    assert records == [
        {
            "job_id": "991",
            "token": "external-debug",
            "state": "RUNNING",
            "reservation": "interactive",
            "source": "squeue",
        }
    ]


def test_snapshot_merges_live_queue_with_accounting_and_never_infers_missing_terminal_state(tmp_path: Path) -> None:
    binaries, log = _fake_scheduler(tmp_path)
    requests = [
        {
            "id": "finished",
            "token": "ctm-finished",
            "job_id": "101",
            "status": "submitted",
            "created_at": 1_700_000_000,
        },
        {
            "id": "unknown-end",
            "token": "ctm-unknown",
            "job_id": "102",
            "status": "submitted",
            "created_at": 1_700_000_000,
        },
        {"id": "absent", "token": "ctm-absent", "job_id": "103", "status": "submitted", "created_at": 1_700_000_000},
    ]
    completed = _invoke_remote(
        {"action": "snapshot", "user": "alice", "requests": requests},
        binaries=binaries,
        log=log,
        extra_environment={
            "FAKE_SQUEUE_OUTPUT": "200|ctm-live|RUNNING|interactive|alice\n",
            "FAKE_SACCT_OUTPUT": textwrap.dedent("""\
                101|ctm-finished|COMPLETED+|2026-09-08T12:00:00|
                102|ctm-unknown|FAILED|Unknown|
                103.batch|ctm-absent|COMPLETED|2026-09-08T12:00:00|
                999|unrelated|COMPLETED|2026-09-08T12:00:00|
                """),
        },
    )

    assert completed.returncode == 0, completed.stderr
    rows = json.loads(completed.stdout)
    assert rows == [
        {"job_id": "200", "token": "ctm-live", "state": "RUNNING", "reservation": "interactive", "source": "squeue"},
        {"job_id": "101", "token": "ctm-finished", "state": "COMPLETED", "reservation": "", "source": "sacct"},
        {"job_id": "102", "token": "ctm-unknown", "state": "UNKNOWN", "reservation": "", "source": "sacct"},
    ]
    calls = _records(log)
    assert [call["program"] for call in calls] == ["squeue", "sacct"]
    assert calls[0]["argv"] == ["--all", "--noheader", "--array", "--user", "alice", "--format", "%i|%128j|%T|%v|%u"]
    assert calls[1]["argv"][:5] == ["--allocations", "--noheader", "--parsable2", "--user", "alice"]
    assert "--starttime" in calls[1]["argv"]
    assert "--format" in calls[1]["argv"]


def test_snapshot_forwards_only_controller_status_metadata_to_remote() -> None:
    backend = object.__new__(SSHBackend)
    captured = {}

    def remote(action, **kwargs):
        captured["action"] = action
        captured.update(kwargs)
        return []

    backend._remote = remote
    SSHBackend.snapshot(
        backend,
        [
            {
                "id": "queued",
                "token": "ctm-queued",
                "status": "queued",
                "created_at": 123,
                "script": "#!/bin/sh\nsecret-body\n",
                "env": {"HARMLESS_VALUE": "not-for-transport"},
            },
            {
                "id": "submitted",
                "token": "ctm-submitted",
                "job_id": "412",
                "status": "submitted",
                "created_at": 124,
                "script": "#!/bin/sh\nother-body\n",
                "env": {"HARMLESS_VALUE": "also-not-for-transport"},
            },
            {"id": "done", "token": "ctm-done", "status": "terminal", "created_at": 125},
        ],
    )

    assert captured == {
        "action": "snapshot",
        "requests": [
            {"id": "queued", "token": "ctm-queued", "status": "queued", "created_at": 123},
            {"id": "submitted", "token": "ctm-submitted", "job_id": "412", "status": "submitted", "created_at": 124},
        ],
    }
