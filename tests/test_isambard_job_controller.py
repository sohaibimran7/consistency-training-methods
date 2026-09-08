"""Focused safety tests for the shared Isambard job controller."""

from __future__ import annotations

import copy
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from infra.isambard.job_controller import (
    ConflictError,
    Controller,
    ControllerError,
    OwnershipError,
    SubmissionRejected,
)


class Clock:
    def __init__(self, value=1_000.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeBackend:
    def __init__(self):
        self.jobs = []
        self.snapshot_error = None
        self.submit_error = None
        self.snapshot_calls = []
        self.submissions = []
        self.cancellations = []
        self.next_job_id = "1001"

    def snapshot(self, requests):
        self.snapshot_calls.append(copy.deepcopy(requests))
        if self.snapshot_error is not None:
            raise self.snapshot_error
        return copy.deepcopy(self.jobs)

    def submit(self, request):
        self.submissions.append(copy.deepcopy(request))
        if self.submit_error is not None:
            raise self.submit_error
        return self.next_job_id

    def cancel(self, job_id):
        self.cancellations.append(job_id)


def request(
    request_id,
    *,
    owner="agent-a",
    mode="batch",
    output_root=None,
    nodes=1,
    gpus=1,
    minutes=30,
):
    output_root = output_root or "/remote/results/{}".format(request_id)
    return {
        "id": request_id,
        "owner": owner,
        "mode": mode,
        "output_roots": [output_root],
        "script": "#!/bin/sh\necho run\n",
        "resources": {
            "nodes": nodes,
            "gpus": gpus,
            "minutes": minutes,
            "cpus_per_task": 4,
            "memory_mb": 1024,
        },
        "remote_dir": "/remote/work/{}".format(request_id),
        "env": {"RUN_ID": request_id},
    }


def single_record(result, request_id):
    return next(record for record in result["requests"] if record["id"] == request_id)


def test_one_tick_admits_one_batch_and_one_interactive_with_stable_token(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    controller = Controller(tmp_path / "state", backend, "sohaib.a5v@ai-p2", clock)
    controller.enqueue(request("batch", output_root="/remote/results/batch"))
    controller.enqueue(
        request("interactive", mode="interactive", output_root="/remote/results/interactive", gpus=4, minutes=60)
    )

    result = controller.tick()

    assert [submission["id"] for submission in backend.submissions] == ["batch", "interactive"]
    assert all(submission["token"].startswith("ctm-") for submission in backend.submissions)
    assert single_record(result, "batch")["status"] == "submitted"
    assert single_record(result, "interactive")["status"] == "submitted"
    assert stat.S_IMODE(controller.state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(controller.state_path.stat().st_mode) == 0o600


def test_interactive_capacity_includes_unmanaged_reservation_jobs(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    backend.jobs = [{"job_id": "external-1", "token": None, "state": "RUNNING", "reservation": "interactive"}]
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("debug", mode="interactive", output_root="/remote/results/debug", gpus=4))

    result = controller.tick()

    assert backend.submissions == []
    assert single_record(result, "debug")["status"] == "queued"
    assert "interactive reservation" in single_record(result, "debug")["diagnostic"]


def test_scope_namespace_and_output_ancestor_conflicts_are_shared(tmp_path):
    clock = Clock()
    first = Controller(tmp_path / "state", FakeBackend(), "User@Resolved.Cluster", clock)
    second = Controller(tmp_path / "state", FakeBackend(), "user@resolved.cluster", clock)
    other_scope = Controller(tmp_path / "state", FakeBackend(), "user@other.cluster", clock)
    first.enqueue(request("first", owner="owner-one", output_root="/remote/results/campaign"))

    assert first.state_path == second.state_path
    with pytest.raises(ConflictError, match="owner-one"):
        second.enqueue(request("second", owner="owner-two", output_root="/remote/results/campaign/segment"))
    other_scope.enqueue(request("third", owner="owner-three", output_root="/remote/results/campaign/segment"))
    assert other_scope.state_path != first.state_path


def test_snapshot_failures_are_throttled_and_keep_request_queued(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    backend.snapshot_error = RuntimeError("authentication unavailable")
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("run"))

    failed = controller.tick()
    clock.advance(10)
    throttled = controller.tick()
    clock.advance(50)
    failed_again = controller.tick()

    assert len(backend.snapshot_calls) == 2
    assert failed["actions"][-1]["action"] == "snapshot_failed"
    assert throttled["actions"][-1]["action"] == "snapshot_throttled"
    assert single_record(failed_again, "run")["status"] == "queued"
    assert "snapshot failed" in single_record(failed_again, "run")["diagnostic"]
    assert backend.submissions == []


def test_missing_stale_job_record_never_releases_output_ownership(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("original", output_root="/remote/results/shared"))
    controller.tick()
    clock.advance(60)
    backend.jobs = []  # A backend must consult accounting for this known id; absence is not terminal evidence.

    result = controller.tick()

    assert single_record(result, "original")["status"] == "submitted"
    with pytest.raises(ConflictError, match="original"):
        controller.enqueue(request("replacement", output_root="/remote/results/shared"))


def test_ambiguous_submission_is_never_retried_and_can_recover_by_token(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    backend.submit_error = TimeoutError("connection closed after submit")
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("uncertain", output_root="/remote/results/uncertain"))

    first = controller.tick()
    token = single_record(first, "uncertain")["token"]
    assert single_record(first, "uncertain")["status"] == "unknown"
    assert len(backend.submissions) == 1

    backend.submit_error = None
    clock.advance(60)
    backend.jobs = [{"job_id": "2400", "token": token, "state": "PENDING", "reservation": None}]
    recovered = controller.tick()

    assert single_record(recovered, "uncertain")["status"] == "submitted"
    assert single_record(recovered, "uncertain")["job_id"] == "2400"
    assert len(backend.submissions) == 1
    with pytest.raises(ConflictError):
        controller.enqueue(request("same-output", output_root="/remote/results/uncertain"))


def test_definitive_preflight_rejection_releases_the_namespace(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    backend.submit_error = SubmissionRejected("requested reservation is unavailable")
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("rejected", output_root="/remote/results/rejected"))

    result = controller.tick()

    rejected = single_record(result, "rejected")
    assert rejected["status"] == "terminal"
    assert rejected["terminal_state"] == "REJECTED"
    controller.enqueue(request("replacement", output_root="/remote/results/rejected"))


def test_exact_terminal_evidence_releases_paths_and_protected_jobs_wait_for_all(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("managed", output_root="/remote/results/managed"))
    controller.tick()
    token = single_record(controller.status(), "managed")["token"]
    clock.advance(60)
    backend.jobs = [{"job_id": "1001", "token": token, "state": "COMPLETED", "reservation": None}]
    terminal = controller.tick()

    assert single_record(terminal, "managed")["status"] == "terminal"
    controller.enqueue(request("after-managed", output_root="/remote/results/managed"))

    controller.protect(
        "existing-campaign",
        "agent-b",
        ["2001", "2002"],
        ["/remote/results/protected"],
    )
    clock.advance(60)
    backend.jobs = [
        {"job_id": "2001", "token": None, "state": "COMPLETED", "reservation": None},
        {"job_id": "2002", "token": None, "state": "RUNNING", "reservation": None},
    ]
    still_protected = controller.tick()
    assert single_record(still_protected, "existing-campaign")["status"] == "submitted"
    clock.advance(60)
    backend.jobs[1]["state"] = "FAILED"
    released = controller.tick()
    assert single_record(released, "existing-campaign")["status"] == "terminal"
    controller.enqueue(request("after-protected", output_root="/remote/results/protected"))


def test_terminal_record_with_wrong_token_cannot_release_or_be_cancelled(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("identity", output_root="/remote/results/identity"))
    controller.tick()
    clock.advance(60)
    backend.jobs = [{"job_id": "1001", "token": "ctm-someone-else", "state": "COMPLETED", "reservation": None}]

    result = controller.tick()

    assert single_record(result, "identity")["status"] == "unknown"
    with pytest.raises(ControllerError, match="no confirmed scheduler job"):
        controller.cancel("identity", "agent-a")
    with pytest.raises(ConflictError):
        controller.enqueue(request("replacement", output_root="/remote/results/identity"))


def test_cancellation_requires_owner_and_waits_for_terminal_scheduler_evidence(tmp_path):
    clock = Clock()
    backend = FakeBackend()
    controller = Controller(tmp_path / "state", backend, "user@cluster", clock)
    controller.enqueue(request("owned", owner="agent-a"))
    controller.tick()

    with pytest.raises(OwnershipError):
        controller.cancel("owned", "agent-b")
    assert backend.cancellations == []
    requested = controller.cancel("owned", "agent-a")
    assert single_record(requested, "owned")["status"] == "cancel_requested"
    assert backend.cancellations == ["1001"]
    repeated = controller.cancel("owned", "agent-a")
    assert repeated["actions"][-1]["action"] == "cancel_already_requested"
    assert backend.cancellations == ["1001"]

    clock.advance(60)
    token = single_record(controller.status(), "owned")["token"]
    backend.jobs = [{"job_id": "1001", "token": token, "state": "CANCELLED by 123", "reservation": None}]
    terminal = controller.tick()
    assert single_record(terminal, "owned")["status"] == "terminal"


class ProcessBackend:
    """A tiny filesystem-backed backend used only to prove lock contention."""

    def __init__(self, log_path):
        self.log_path = log_path

    def snapshot(self, requests):
        return []

    def submit(self, submitted_request):
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(submitted_request["id"] + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return "7001"

    def cancel(self, job_id):
        raise AssertionError("not used")


def _contention_worker(state_dir, log_path, request_id, start_event, ready_queue):
    controller = Controller(Path(state_dir), ProcessBackend(log_path), "user@cluster", lambda: 4_000.0)
    controller.enqueue(
        request(
            request_id,
            mode="interactive",
            output_root="/remote/results/{}".format(request_id),
            gpus=4,
        )
    )
    ready_queue.put(request_id)
    start_event.wait(10)
    controller.tick()


@pytest.mark.skipif(os.name == "nt", reason="flock contention test requires POSIX")
def test_two_processes_cannot_double_submit_interactive_capacity(tmp_path):
    context = multiprocessing.get_context("fork")
    state_dir = tmp_path / "state"
    log_path = tmp_path / "submissions.log"
    start = context.Event()
    ready = context.Queue()
    processes = [
        context.Process(target=_contention_worker, args=(str(state_dir), str(log_path), "first", start, ready)),
        context.Process(target=_contention_worker, args=(str(state_dir), str(log_path), "second", start, ready)),
    ]
    for process in processes:
        process.start()
    assert {ready.get(timeout=10), ready.get(timeout=10)} == {"first", "second"}
    start.set()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0

    submissions = log_path.read_text(encoding="utf-8").splitlines()
    assert len(submissions) == 1
    assert submissions[0] in {"first", "second"}
