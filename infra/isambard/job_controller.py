#!/usr/bin/env python3
"""Durable, fail-closed admission control for shared Isambard jobs.

The controller deliberately knows nothing about SSH, Slurm command syntax, or
authentication.  A small backend supplied by the caller performs those
operations while this module owns the durable local state and the critical
section around every backend call.

One controller scope represents one *resolved* account and cluster.  Callers
must canonicalise aliases before constructing a controller; requests never
carry a host, so the scope cannot accidentally be switched per request.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import posixpath
import re
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

SNAPSHOT_INTERVAL_SECONDS = 60.0
STATE_VERSION = 1
DEFAULT_STATE_DIR = Path.home() / ".local" / "state" / "ctm" / "isambard-jobs"

_TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
_ACTIVE_STATUSES = frozenset({"queued", "submitting", "unknown", "submitted", "cancel_requested"})
_CAPACITY_STATUSES = frozenset({"submitting", "unknown", "submitted", "cancel_requested"})
_PROCESS_LOCKS: Dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class ControllerError(RuntimeError):
    """Base class for errors a command-line adapter can present safely."""


class ValidationError(ControllerError):
    """A request is incomplete or violates controller safety invariants."""


class ConflictError(ControllerError):
    """A request would overlap work already owned by this controller scope."""


class OwnershipError(ControllerError):
    """A caller attempted to change a request owned by somebody else."""


class StateError(ControllerError):
    """The durable state cannot be safely interpreted."""


class SubmissionRejected(ControllerError):
    """A preflight proved that Slurm did not receive the submission."""


def _copy_json(value: Any) -> Any:
    """Return a detached JSON value, rejecting non-durable request data."""

    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ValidationError("Request must contain only JSON-compatible values.") from exc


def _error_text(exc: BaseException) -> str:
    """Keep diagnostics useful without turning local state into a log sink."""

    detail = str(exc).strip().replace("\n", " ")
    if len(detail) > 300:
        detail = detail[:297] + "..."
    return "{}{}".format(type(exc).__name__, ": " + detail if detail else "")


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValidationError("{} must be a non-empty text value.".format(field))
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError("resources.{} must be a positive integer.".format(field))
    return value


def _normal_remote_path(value: Any, field: str) -> str:
    """Require an already-normalised non-root POSIX path.

    Remote output ownership needs exact path comparison.  Silently repairing a
    path here would make two callers believe they own different locations, so
    ``.``, ``..``, duplicate separators and trailing separators are rejected.
    """

    path = _require_text(value, field)
    if not path.startswith("/") or path == "/" or path.startswith("//"):
        raise ValidationError("{} must be an absolute, non-root remote path.".format(field))
    normal = posixpath.normpath(path)
    if normal != path or path.endswith("/"):
        raise ValidationError("{} must already be a normalised remote path.".format(field))
    parts = path.split("/")[1:]
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValidationError("{} must already be a normalised remote path.".format(field))
    return path


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _state_name(scope: str) -> str:
    return "controller-{}.json".format(hashlib.sha256(scope.encode("utf-8")).hexdigest())


def _token_for(scope: str, request_id: str) -> str:
    """Generate a stable, Slurm-safe correlation token for a request."""

    identity = "ctm-isambard-controller/v1\x00{}\x00{}".format(scope, request_id)
    return "ctm-{}".format(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def _state_word(value: str) -> str:
    """Extract Slurm's primary state word from live or accounting output."""

    return value.strip().upper().split(None, 1)[0].rstrip("+")


def _terminal_evidence(job: Mapping[str, Any]) -> bool:
    word = _state_word(str(job["state"]))
    if word in _TERMINAL_STATES:
        return True
    # PREEMPTED may describe a requeue rather than final job accounting.  A
    # backend may make final accounting explicit, but an unadorned PREEMPTED
    # record never releases controller ownership.
    return word == "PREEMPTED" and job.get("terminal") is True and str(job.get("source", "")).lower() == "accounting"


def _is_interactive_job(job: Mapping[str, Any]) -> bool:
    reservation = job.get("reservation")
    return isinstance(reservation, str) and reservation.strip().lower() == "interactive"


class Controller:
    """Coordinate submissions for one resolved Isambard account/cluster.

    ``backend.snapshot(records)`` must return live queue jobs plus accounting
    records for known missing or ambiguous controller records.  It returns
    mappings with at least ``job_id``, ``token``, ``state`` and ``reservation``.
    The engine passes each durable record to make token and job-id recovery
    possible.  ``backend.submit(request)`` receives a detached request mapping
    augmented with the durable ``token``; adapters should use that token as the
    Slurm job name.  ``backend.cancel(job_id)`` is invoked only after an
    explicit owner-authorised cancellation request.
    """

    def __init__(
        self,
        state_dir: Path,
        backend: Any,
        scope: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(scope, str) or not scope.strip() or "\x00" in scope:
            raise ValidationError("scope must be a non-empty resolved account-and-cluster string.")
        self.scope = scope.strip().lower()
        self.backend = backend
        self.clock = clock
        try:
            self.state_dir = Path(state_dir).expanduser().resolve()
        except OSError as exc:
            raise StateError("Could not resolve the controller state directory.") from exc
        self.state_path = self.state_dir / _state_name(self.scope)
        self.lock_path = self.state_dir / (self.state_path.stem + ".lock")

    def enqueue(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a validated request in FIFO order without contacting Slurm."""

        prepared = self._validate_request(request)
        with self._locked_state() as state:
            existing = self._find_record(state, prepared["id"])
            if existing is not None:
                if existing.get("kind") != "managed":
                    raise ConflictError(
                        "Request id {!r} is already reserved as a protected record owned by {!r}.".format(
                            prepared["id"], existing["owner"]
                        )
                    )
                if self._request_fields(existing) != prepared:
                    raise ConflictError(
                        "Request id {!r} is already owned by {!r} with different immutable details.".format(
                            prepared["id"], existing["owner"]
                        )
                    )
                return self._result(state, actions=[{"action": "already_enqueued", "request_id": prepared["id"]}])

            conflict = self._output_conflict(state, prepared["output_roots"])
            if conflict is not None:
                other, left, right = conflict
                raise ConflictError(
                    "Output path {!r} overlaps {!r}, owned by request {!r} ({!r}, status {}).".format(
                        left, right, other["id"], other["owner"], other["status"]
                    )
                )

            now = self._now()
            record: Dict[str, Any] = dict(prepared)
            record.update(
                {
                    "kind": "managed",
                    "token": _token_for(self.scope, prepared["id"]),
                    "status": "queued",
                    "job_id": None,
                    "created_at": now,
                    "updated_at": now,
                    "diagnostic": None,
                    "events": [],
                }
            )
            self._event(record, now, "enqueued")
            state["requests"].append(record)
            self._save_state(state)
            return self._result(state, actions=[{"action": "enqueued", "request_id": prepared["id"]}])

    def protect(
        self,
        request_id: str,
        owner: str,
        job_ids: Any,
        output_roots: Any,
        mode: str = "batch",
    ) -> Dict[str, Any]:
        """Register externally submitted jobs as observation-only ownership.

        This is intended for existing campaigns that cannot be moved into the
        controller retrospectively.  Protected records are never cancelled by
        :meth:`cancel`; every supplied job id needs exact terminal scheduler
        evidence before their output paths are released.
        """

        request_id = _require_text(request_id, "id")
        owner = _require_text(owner, "owner")
        if mode not in ("interactive", "batch"):
            raise ValidationError("mode must be either 'interactive' or 'batch'.")
        if isinstance(job_ids, str):
            job_ids = [job_ids]
        if not isinstance(job_ids, Sequence) or isinstance(job_ids, (str, bytes)) or not job_ids:
            raise ValidationError("job_ids must be a non-empty list of scheduler job ids.")
        normal_job_ids: List[str] = []
        for index, value in enumerate(job_ids):
            job_id = _require_text(value, "job_ids[{}]".format(index)).strip()
            if not re.fullmatch(r"[0-9]+", job_id):
                raise ValidationError("job_ids[{}] must be a numeric Slurm allocation id.".format(index))
            if job_id in normal_job_ids:
                raise ValidationError("job_ids must not contain duplicates.")
            normal_job_ids.append(job_id)
        if isinstance(output_roots, str):
            output_roots = [output_roots]
        normal_roots = self._validate_output_roots(output_roots)

        with self._locked_state() as state:
            existing = self._find_record(state, request_id)
            if existing is not None:
                expected = {
                    "id": request_id,
                    "owner": owner,
                    "job_ids": normal_job_ids,
                    "output_roots": normal_roots,
                    "mode": mode,
                    "kind": "protected",
                }
                actual = {key: existing.get(key) for key in expected}
                if actual != expected:
                    raise ConflictError(
                        "Request id {!r} is already owned by {!r} with different protected-job details.".format(
                            request_id, existing["owner"]
                        )
                    )
                return self._result(state, actions=[{"action": "already_protected", "request_id": request_id}])

            conflict = self._output_conflict(state, normal_roots)
            if conflict is not None:
                other, left, right = conflict
                raise ConflictError(
                    "Output path {!r} overlaps {!r}, owned by request {!r} ({!r}, status {}).".format(
                        left, right, other["id"], other["owner"], other["status"]
                    )
                )
            now = self._now()
            record = {
                "id": request_id,
                "owner": owner,
                "kind": "protected",
                "mode": mode,
                "output_roots": normal_roots,
                "job_ids": normal_job_ids,
                "token": None,
                "status": "submitted",
                "created_at": now,
                "updated_at": now,
                "diagnostic": "Protected external job(s); observation only.",
                "events": [],
            }
            self._event(record, now, "protected")
            state["requests"].append(record)
            self._save_state(state)
            return self._result(state, actions=[{"action": "protected", "request_id": request_id}])

    def tick(self) -> Dict[str, Any]:
        """Reconcile one fresh scheduler snapshot and admit at most two jobs.

        A fresh snapshot is mandatory for every admission.  This sacrifices a
        little responsiveness in exchange for avoiding stale-cache decisions
        when another terminal or user has submitted work outside the
        controller.  Locally submitted jobs are still written into the cache
        and represented by durable active records before the lock is released.
        """

        with self._locked_state() as state:
            now = self._now()
            last_attempt = state.get("last_snapshot_attempt_at")
            if last_attempt is not None and now - float(last_attempt) < SNAPSHOT_INTERVAL_SECONDS:
                remaining = max(0.0, SNAPSHOT_INTERVAL_SECONDS - (now - float(last_attempt)))
                return self._result(
                    state,
                    actions=[{"action": "snapshot_throttled", "retry_after_seconds": remaining}],
                )

            # Persist the attempt before calling the backend.  If this process
            # disappears during a network call, another process still waits the
            # full interval rather than starting a query storm.
            state["last_snapshot_attempt_at"] = now
            state["last_snapshot_error"] = None
            state["last_error"] = None
            state["snapshot_error"] = None
            self._save_state(state)
            try:
                raw_jobs = self.backend.snapshot(self._snapshot_requests(state))
                jobs = self._validate_snapshot(raw_jobs)
            except Exception as exc:  # The backend may represent any transport failure.
                detail = _error_text(exc)
                state["last_snapshot_error"] = detail
                state["last_error"] = detail
                state["snapshot_error"] = detail
                for record in state["requests"]:
                    if record["status"] == "queued":
                        self._set_diagnostic(
                            record,
                            now,
                            "Blocked: scheduler snapshot failed ({}); request remains queued.".format(detail),
                            "snapshot_failed",
                        )
                self._save_state(state)
                return self._result(state, actions=[{"action": "snapshot_failed", "error": detail}])

            state["last_snapshot_success_at"] = now
            state["last_snapshot_error"] = None
            state["last_error"] = None
            state["snapshot_error"] = None
            state["last_snapshot"] = {"at": now, "jobs": jobs}
            self._reconcile(state, jobs, now)
            self._save_state(state)

            actions: List[Dict[str, Any]] = [{"action": "snapshot", "jobs": len(jobs)}]
            batch = self._next_queued(state, "batch")
            if batch is not None:
                actions.append(self._submit(state, batch, now))

            interactive = self._next_queued(state, "interactive")
            if interactive is not None:
                if self._interactive_capacity_occupied(state, jobs):
                    self._set_diagnostic(
                        interactive,
                        now,
                        "Waiting for the account-wide interactive reservation slot.",
                        "interactive_blocked",
                    )
                    self._save_state(state)
                    actions.append({"action": "interactive_blocked", "request_id": interactive["id"]})
                else:
                    actions.append(self._submit(state, interactive, now))
            return self._result(state, actions=actions)

    def status(self) -> Dict[str, Any]:
        """Return the durable state without making a backend or SSH call."""

        with self._locked_state() as state:
            return self._result(state, actions=[])

    def cancel(self, request_id: str, owner: str) -> Dict[str, Any]:
        """Request cancellation of one known submitted job, never automatically."""

        request_id = _require_text(request_id, "id")
        owner = _require_text(owner, "owner")
        with self._locked_state() as state:
            record = self._require_owned_record(state, request_id, owner)
            if record["kind"] == "protected":
                raise ControllerError(
                    "Protected request {!r} is observation-only and cannot be cancelled here.".format(request_id)
                )
            if record["status"] == "terminal":
                return self._result(state, actions=[{"action": "already_terminal", "request_id": request_id}])
            if record["status"] == "cancel_requested":
                return self._result(state, actions=[{"action": "cancel_already_requested", "request_id": request_id}])
            if record["status"] != "submitted" or not record.get("job_id"):
                raise ControllerError(
                    "Request {!r} has no confirmed scheduler job to cancel; ownership remains {}.".format(
                        request_id, record["status"]
                    )
                )
            if not self._last_snapshot_identity_confirmed(state, record):
                raise ControllerError(
                    "Request {!r} has no identity-confirmed scheduler job to cancel; run a fresh tick first.".format(
                        request_id
                    )
                )

            now = self._now()
            record["status"] = "cancel_requested"
            record["updated_at"] = now
            record["diagnostic"] = "Cancellation requested; waiting for terminal scheduler evidence."
            self._event(record, now, "cancel_requested")
            # The intent is durable before the backend call.  A timeout or
            # crash cannot turn into a retry or an unowned release.
            self._save_state(state)
            try:
                self.backend.cancel(str(record["job_id"]))
            except Exception as exc:
                detail = _error_text(exc)
                self._set_diagnostic(
                    record,
                    now,
                    "Cancellation outcome is unknown ({}); waiting for terminal scheduler evidence.".format(detail),
                    "cancel_failed",
                )
                self._save_state(state)
                return self._result(
                    state,
                    actions=[{"action": "cancel_unknown", "request_id": request_id, "error": detail}],
                )
            self._save_state(state)
            return self._result(state, actions=[{"action": "cancel_requested", "request_id": request_id}])

    def withdraw(self, request_id: str, owner: str) -> Dict[str, Any]:
        """Withdraw a request that has not begun submission."""

        request_id = _require_text(request_id, "id")
        owner = _require_text(owner, "owner")
        with self._locked_state() as state:
            record = self._require_owned_record(state, request_id, owner)
            if record["kind"] == "protected":
                raise ControllerError("Protected request {!r} cannot be withdrawn.".format(request_id))
            if record["status"] == "terminal":
                return self._result(state, actions=[{"action": "already_terminal", "request_id": request_id}])
            if record["status"] != "queued":
                raise ControllerError(
                    "Request {!r} is {}; only a queued request can be withdrawn safely.".format(
                        request_id, record["status"]
                    )
                )
            now = self._now()
            record["status"] = "terminal"
            record["terminal_state"] = "WITHDRAWN"
            record["updated_at"] = now
            record["diagnostic"] = "Withdrawn before submission."
            self._event(record, now, "withdrawn")
            self._save_state(state)
            return self._result(state, actions=[{"action": "withdrawn", "request_id": request_id}])

    def resolve_unsubmitted(self, request_id: str, owner: str, note: str) -> Dict[str, Any]:
        """Release one ambiguous request after an owner has investigated it.

        This escape hatch is deliberately narrow: it cannot touch a confirmed
        scheduler job, and callers must provide an explicit human-attestation
        note that the request never reached Slurm.  It is never used by
        :meth:`tick` or any automatic recovery path.
        """

        request_id = _require_text(request_id, "id")
        owner = _require_text(owner, "owner")
        note = _require_text(note, "note")
        with self._locked_state() as state:
            record = self._require_owned_record(state, request_id, owner)
            if record.get("kind") != "managed" or record.get("status") != "unknown" or record.get("job_id"):
                raise ControllerError(
                    "Request {!r} is not an unconfirmed submission without a scheduler job id.".format(request_id)
                )
            now = self._now()
            record["status"] = "terminal"
            record["terminal_state"] = "CONFIRMED_NEVER_SUBMITTED"
            record["updated_at"] = now
            record["diagnostic"] = "Owner confirmed after investigation that no scheduler job was submitted: {}".format(
                note
            )
            self._event(record, now, "resolved_unsubmitted", note)
            self._save_state(state)
            return self._result(state, actions=[{"action": "resolved_unsubmitted", "request_id": request_id}])

    def _validate_request(self, request: Any) -> Dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValidationError("request must be an object.")
        required = ("id", "owner", "mode", "output_roots", "script", "resources", "remote_dir", "env")
        missing = [field for field in required if field not in request]
        if missing:
            raise ValidationError("Request is missing required field(s): {}.".format(", ".join(missing)))
        request_id = _require_text(request["id"], "id")
        owner = _require_text(request["owner"], "owner")
        mode = request["mode"]
        if mode not in ("interactive", "batch"):
            raise ValidationError("mode must be either 'interactive' or 'batch'.")
        script = _require_text(request["script"], "script")
        remote_dir = _normal_remote_path(request["remote_dir"], "remote_dir")
        output_roots = self._validate_output_roots(request["output_roots"])

        resources_raw = request["resources"]
        if not isinstance(resources_raw, Mapping):
            raise ValidationError("resources must be an object.")
        resource_names = ("nodes", "gpus", "minutes", "memory_mb")
        missing_resources = [field for field in resource_names if field not in resources_raw]
        if missing_resources:
            raise ValidationError("resources is missing: {}.".format(", ".join(missing_resources)))
        resources = {field: _require_positive_int(resources_raw[field], field) for field in resource_names}
        optional_resource_names = ("cpus_per_task", "cpus_per_gpu", "ntasks", "ntasks_per_node", "gpus_per_node")
        for field in optional_resource_names:
            if field in resources_raw:
                resources[field] = _require_positive_int(resources_raw[field], field)
        if mode == "interactive":
            if resources["nodes"] > 4 or resources["gpus"] > 16 or resources["minutes"] > 480:
                raise ValidationError("Interactive requests are limited to 4 nodes, 16 GPUs, and 480 minutes.")
        elif resources["minutes"] > 24 * 60:
            raise ValidationError("Batch requests are limited to 24 hours (1440 minutes).")

        env_raw = request["env"]
        if not isinstance(env_raw, Mapping):
            raise ValidationError("env must be an object.")
        env: Dict[str, str] = {}
        for key, value in env_raw.items():
            env_key = _require_text(key, "env key")
            if not isinstance(value, str) or "\x00" in value:
                raise ValidationError("env values must be text values.")
            env[env_key] = value

        # Explicitly retain only immutable scheduling input.  This keeps the
        # idempotency comparison meaningful if a caller adds presentation-only
        # fields to its own CLI request in the future.
        return _copy_json(
            {
                "id": request_id,
                "owner": owner,
                "mode": mode,
                "output_roots": output_roots,
                "script": script,
                "resources": resources,
                "remote_dir": remote_dir,
                "env": env,
            }
        )

    def _validate_output_roots(self, roots: Any) -> List[str]:
        if not isinstance(roots, Sequence) or isinstance(roots, (str, bytes)) or not roots:
            raise ValidationError("output_roots must be a non-empty list of absolute remote paths.")
        normal: List[str] = []
        for index, value in enumerate(roots):
            path = _normal_remote_path(value, "output_roots[{}]".format(index))
            if path in normal:
                raise ValidationError("output_roots must not contain duplicates.")
            normal.append(path)
        for index, left in enumerate(normal):
            for right in normal[index + 1 :]:
                if _paths_overlap(left, right):
                    raise ValidationError("output_roots must not contain ancestor/descendant paths.")
        return normal

    def _next_queued(self, state: Mapping[str, Any], mode: str) -> Optional[Dict[str, Any]]:
        for record in state["requests"]:
            if record["kind"] == "managed" and record["mode"] == mode and record["status"] == "queued":
                return record
        return None

    def _interactive_capacity_occupied(self, state: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> bool:
        if any(_is_interactive_job(job) and not _terminal_evidence(job) for job in jobs):
            return True
        return any(
            record["mode"] == "interactive" and record["status"] in _CAPACITY_STATUSES for record in state["requests"]
        )

    def _submit(self, state: Dict[str, Any], record: Dict[str, Any], now: float) -> Dict[str, Any]:
        record["status"] = "submitting"
        record["updated_at"] = now
        record["diagnostic"] = "Submission intent saved; awaiting scheduler acknowledgement."
        self._event(record, now, "submitting")
        # This is intentionally a separate atomic write before the side effect.
        self._save_state(state)
        payload = self._backend_request(record)
        try:
            job_id_raw = self.backend.submit(payload)
            if not isinstance(job_id_raw, str) or not job_id_raw.strip():
                raise ControllerError("backend.submit returned no scheduler job id")
            job_id = job_id_raw.strip()
            if not re.fullmatch(r"[0-9]+", job_id):
                raise ControllerError("backend.submit returned an invalid scheduler job id")
        except SubmissionRejected as exc:
            detail = _error_text(exc)
            record["status"] = "terminal"
            record["terminal_state"] = "REJECTED"
            record["updated_at"] = now
            record["diagnostic"] = "Submission was rejected before Slurm received it: {}.".format(detail)
            self._event(record, now, "submission_rejected", detail)
            self._save_state(state)
            return {"action": "submission_rejected", "request_id": record["id"], "error": detail}
        except Exception as exc:
            detail = _error_text(exc)
            record["status"] = "unknown"
            record["updated_at"] = now
            record["diagnostic"] = "Submission outcome is unknown ({}); automatic retry is disabled.".format(detail)
            self._event(record, now, "submission_unknown", detail)
            self._save_state(state)
            return {"action": "submission_unknown", "request_id": record["id"], "error": detail}

        record["status"] = "submitted"
        record["job_id"] = job_id
        record["updated_at"] = now
        record["diagnostic"] = "Submitted as scheduler job {}.".format(job_id)
        self._event(record, now, "submitted", job_id)
        # Keep a local representation of our own submission even before the
        # scheduler's next snapshot.  It is never used as permission to skip a
        # fresh snapshot for a later submission.
        last_snapshot = state.get("last_snapshot")
        if isinstance(last_snapshot, dict) and isinstance(last_snapshot.get("jobs"), list):
            last_snapshot["jobs"].append(
                {
                    "job_id": job_id,
                    "token": record["token"],
                    "state": "PENDING",
                    "reservation": "interactive" if record["mode"] == "interactive" else None,
                    "source": "local",
                }
            )
        self._save_state(state)
        return {"action": "submitted", "request_id": record["id"], "job_id": job_id}

    def _reconcile(self, state: Dict[str, Any], jobs: Sequence[Mapping[str, Any]], now: float) -> None:
        by_id: Dict[str, List[Mapping[str, Any]]] = {}
        by_token: Dict[str, List[Mapping[str, Any]]] = {}
        for job in jobs:
            by_id.setdefault(str(job["job_id"]), []).append(job)
            token = job.get("token")
            if isinstance(token, str) and token:
                by_token.setdefault(token, []).append(job)

        for record in state["requests"]:
            if record["status"] == "terminal":
                continue
            if record["kind"] == "protected":
                self._reconcile_protected(record, by_id, now)
            else:
                self._reconcile_managed(record, by_id, by_token, now)

    def _reconcile_protected(
        self, record: Dict[str, Any], by_id: Mapping[str, Sequence[Mapping[str, Any]]], now: float
    ) -> None:
        evidence: List[Mapping[str, Any]] = []
        for job_id in record["job_ids"]:
            records = by_id.get(str(job_id), ())
            if not records or not self._all_terminal(records):
                return
            evidence.append(records[0])
        states = ", ".join("{}={}".format(job["job_id"], job["state"]) for job in evidence)
        self._mark_terminal(record, now, states, "protected_terminal")

    def _reconcile_managed(
        self,
        record: Dict[str, Any],
        by_id: Mapping[str, Sequence[Mapping[str, Any]]],
        by_token: Mapping[str, Sequence[Mapping[str, Any]]],
        now: float,
    ) -> None:
        job_id = record.get("job_id")
        if job_id:
            exact = by_id.get(str(job_id), ())
            if exact:
                if any(job.get("token") != record["token"] for job in exact):
                    record["status"] = "unknown"
                    self._set_diagnostic(
                        record,
                        now,
                        "Scheduler job id evidence does not carry this request token; ownership remains blocked.",
                        "job_identity_mismatch",
                    )
                    return
                if self._all_terminal(exact):
                    states = ", ".join(str(job["state"]) for job in exact)
                    self._mark_terminal(record, now, states, "terminal")
                    return
                if record["status"] != "cancel_requested":
                    record["status"] = "submitted"
                    record["updated_at"] = now
                    record["diagnostic"] = "Scheduler reports {} as {}.".format(job_id, exact[0]["state"])
                return
            conflicting_tokens = [job for job in by_token.get(record["token"], ()) if str(job["job_id"]) != str(job_id)]
            if conflicting_tokens:
                self._set_diagnostic(
                    record,
                    now,
                    "Conflicting scheduler job(s) share this request token; ownership remains blocked.",
                    "token_conflict",
                )
            # Missing known job ids are intentionally not released: the backend
            # is responsible for querying accounting records for them.
            return

        matches = by_token.get(record["token"], ())
        distinct_ids = {str(job["job_id"]) for job in matches}
        if len(distinct_ids) == 1:
            matched_id = next(iter(distinct_ids))
            matching = [job for job in matches if str(job["job_id"]) == matched_id]
            record["job_id"] = matched_id
            record["updated_at"] = now
            if self._all_terminal(matching):
                states = ", ".join(str(job["state"]) for job in matching)
                self._mark_terminal(record, now, states, "recovered_terminal")
            else:
                record["status"] = "submitted"
                record["diagnostic"] = "Recovered scheduler job {} from durable token.".format(matched_id)
                self._event(record, now, "recovered", matched_id)
            return
        if len(distinct_ids) > 1:
            record["status"] = "unknown"
            self._set_diagnostic(
                record,
                now,
                "Multiple scheduler jobs share this request token; automatic retry is disabled.",
                "token_ambiguous",
            )
            return
        if record["status"] == "submitting":
            record["status"] = "unknown"
            self._set_diagnostic(
                record,
                now,
                "Submission was interrupted before acknowledgement; no scheduler evidence was found, so ownership remains blocked.",
                "submission_unconfirmed",
            )

    @staticmethod
    def _all_terminal(records: Sequence[Mapping[str, Any]]) -> bool:
        return bool(records) and all(_terminal_evidence(job) for job in records)

    def _mark_terminal(self, record: Dict[str, Any], now: float, state_text: str, event: str) -> None:
        record["status"] = "terminal"
        record["terminal_state"] = state_text
        record["updated_at"] = now
        record["diagnostic"] = "Terminal scheduler evidence: {}.".format(state_text)
        self._event(record, now, event, state_text)

    def _output_conflict(
        self, state: Mapping[str, Any], requested_roots: Sequence[str]
    ) -> Optional[Tuple[Mapping[str, Any], str, str]]:
        for other in state["requests"]:
            if other.get("status") not in _ACTIVE_STATUSES:
                continue
            for left in requested_roots:
                for right in other.get("output_roots", ()):
                    if _paths_overlap(left, str(right)):
                        return other, left, str(right)
        return None

    @staticmethod
    def _request_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "id": record["id"],
            "owner": record["owner"],
            "mode": record["mode"],
            "output_roots": record["output_roots"],
            "script": record["script"],
            "resources": record["resources"],
            "remote_dir": record["remote_dir"],
            "env": record["env"],
        }

    @staticmethod
    def _find_record(state: Mapping[str, Any], request_id: str) -> Optional[Dict[str, Any]]:
        for record in state["requests"]:
            if record.get("id") == request_id:
                return record
        return None

    def _require_owned_record(self, state: Mapping[str, Any], request_id: str, owner: str) -> Dict[str, Any]:
        record = self._find_record(state, request_id)
        if record is None:
            raise ControllerError("No request with id {!r} exists in this scope.".format(request_id))
        if record.get("owner") != owner:
            raise OwnershipError("Request {!r} is owned by {!r}.".format(request_id, record.get("owner")))
        return record

    @staticmethod
    def _last_snapshot_identity_confirmed(state: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
        """Avoid cancelling an id that a later scheduler record has reused."""

        snapshot = state.get("last_snapshot")
        if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("jobs"), list):
            return False
        exact = [job for job in snapshot["jobs"] if str(job.get("job_id")) == str(record.get("job_id"))]
        return bool(exact) and all(job.get("token") == record.get("token") for job in exact)

    @staticmethod
    def _backend_request(record: Mapping[str, Any]) -> Dict[str, Any]:
        fields = Controller._request_fields(record)
        fields["token"] = record["token"]
        return _copy_json(fields)

    @staticmethod
    def _snapshot_requests(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Send only active correlation fields to a snapshot backend.

        This avoids passing historical scripts and environments through the
        transport and bounds accounting lookups as the local audit grows.
        """

        records: List[Dict[str, Any]] = []
        for record in state["requests"]:
            if record.get("status") == "terminal":
                continue
            item = {
                "id": record["id"],
                "token": record.get("token"),
                "status": record.get("status"),
                "created_at": record.get("created_at"),
            }
            if record.get("job_id"):
                item["job_id"] = record["job_id"]
            if record.get("job_ids"):
                item["job_ids"] = record["job_ids"]
            records.append(item)
        return _copy_json(records)

    def _validate_snapshot(self, raw_jobs: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_jobs, list):
            raise ControllerError("backend.snapshot must return a list of scheduler jobs.")
        jobs: List[Dict[str, Any]] = []
        for index, raw in enumerate(raw_jobs):
            if not isinstance(raw, Mapping):
                raise ControllerError("backend.snapshot job {} is not an object.".format(index))
            try:
                job_id = _require_text(raw.get("job_id"), "snapshot.job_id").strip()
                state = _require_text(raw.get("state"), "snapshot.state").strip()
            except ValidationError as exc:
                raise ControllerError("Invalid backend snapshot job {}: {}".format(index, exc)) from exc
            token = raw.get("token")
            reservation = raw.get("reservation")
            if token is not None and not isinstance(token, str):
                raise ControllerError("Invalid backend snapshot token for job {}.".format(job_id))
            if reservation is not None and not isinstance(reservation, str):
                raise ControllerError("Invalid backend snapshot reservation for job {}.".format(job_id))
            job: Dict[str, Any] = {
                "job_id": job_id,
                "token": token,
                "state": state,
                "reservation": reservation,
            }
            # Optional accounting provenance is deliberately narrow.  It lets a
            # backend distinguish a final PREEMPTED accounting entry from a
            # transient queue record without accepting arbitrary state data.
            if "source" in raw:
                if not isinstance(raw["source"], str):
                    raise ControllerError("Invalid backend snapshot source for job {}.".format(job_id))
                job["source"] = raw["source"]
            if "terminal" in raw:
                if not isinstance(raw["terminal"], bool):
                    raise ControllerError("Invalid backend snapshot terminal flag for job {}.".format(job_id))
                job["terminal"] = raw["terminal"]
            jobs.append(job)
        return jobs

    def _set_diagnostic(self, record: Dict[str, Any], now: float, text: str, event: str) -> None:
        if record.get("diagnostic") != text:
            record["diagnostic"] = text
            record["updated_at"] = now
            self._event(record, now, event, text)

    @staticmethod
    def _event(record: Dict[str, Any], now: float, event: str, detail: Optional[str] = None) -> None:
        item: Dict[str, Any] = {"at": now, "event": event}
        if detail:
            item["detail"] = detail
        events = record.setdefault("events", [])
        events.append(item)
        # Retain a compact audit trail while keeping a long-running controller
        # state file bounded.
        if len(events) > 128:
            del events[:-128]

    def _now(self) -> float:
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StateError("clock must return a numeric Unix timestamp.")
        return float(value)

    def _initial_state(self) -> Dict[str, Any]:
        now = self._now()
        return {
            "version": STATE_VERSION,
            "scope": self.scope,
            "created_at": now,
            "updated_at": now,
            "last_snapshot_attempt_at": None,
            "last_snapshot_success_at": None,
            "last_snapshot_error": None,
            # These aliases are kept for simple adapters that only need to
            # decide whether the latest scheduler poll made progress.
            "last_error": None,
            "snapshot_error": None,
            "last_snapshot": None,
            "requests": [],
        }

    @contextmanager
    def _locked_state(self) -> Iterator[Dict[str, Any]]:
        self._ensure_state_dir()
        lock_key = str(self.lock_path)
        with _PROCESS_LOCKS_GUARD:
            process_lock = _PROCESS_LOCKS.setdefault(lock_key, threading.RLock())
        with process_lock:
            descriptor = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.chmod(self.lock_path, 0o600)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                state = self._load_state()
                yield state
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                finally:
                    os.close(descriptor)

    def _ensure_state_dir(self) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.state_dir, 0o700)
        except OSError as exc:
            raise StateError("Could not create secure controller state directory {!s}.".format(self.state_dir)) from exc

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return self._initial_state()
        try:
            os.chmod(self.state_path, 0o600)
            raw = self.state_path.read_text(encoding="utf-8")
            state = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StateError("Controller state is unreadable; refusing to submit or release work.") from exc
        if not isinstance(state, dict) or state.get("version") != STATE_VERSION or state.get("scope") != self.scope:
            raise StateError("Controller state does not match this scope or version; refusing to continue.")
        if not isinstance(state.get("requests"), list):
            raise StateError("Controller state has no valid request list; refusing to continue.")
        return state

    def _save_state(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = self._now()
        try:
            encoded = (
                json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise StateError("Controller state contains a non-JSON value; refusing to continue.") from exc
        descriptor: Optional[int] = None
        temporary: Optional[str] = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=self.state_path.name + ".", suffix=".tmp", dir=str(self.state_dir)
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            temporary = None
            os.chmod(self.state_path, 0o600)
            self._fsync_directory()
        except OSError as exc:
            raise StateError("Could not atomically save controller state.") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _fsync_directory(self) -> None:
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(str(self.state_dir), os.O_RDONLY)
            os.fsync(descriptor)
        except OSError:
            # The state file itself has already been fsynced.  Some local file
            # systems do not support directory fsync; do not weaken the atomic
            # replacement solely because of that platform limitation.
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _result(self, state: Mapping[str, Any], actions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return _copy_json(
            {
                "scope": state["scope"],
                "last_snapshot_attempt_at": state.get("last_snapshot_attempt_at"),
                "last_snapshot_success_at": state.get("last_snapshot_success_at"),
                "last_snapshot_error": state.get("last_snapshot_error"),
                "last_error": state.get("last_error", state.get("last_snapshot_error")),
                "snapshot_error": state.get("snapshot_error", state.get("last_snapshot_error")),
                "last_snapshot": state.get("last_snapshot"),
                "requests": state["requests"],
                "actions": list(actions),
            }
        )
