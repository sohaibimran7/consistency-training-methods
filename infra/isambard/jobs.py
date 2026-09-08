#!/usr/bin/env python3
"""Shared Isambard queue, submission ownership and cached scheduler status."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    from .job_controller import Controller
    from .job_transport import SSHBackend, validate_request
except ImportError:
    from job_controller import Controller
    from job_transport import SSHBackend, validate_request

DEFAULT_STATE = Path.home() / ".local/share/ctm-isambard/jobs"
DEFAULT_HOST = "a5v.aip2.isambard"


def _code_fingerprint():
    root = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("jobs.py", "job_controller.py", "job_transport.py", "job_adapters.py")
    }


LOADED_CODE_FINGERPRINT = _code_fingerprint()


def public_state(value):
    """Print ownership/status, not captured programs or environment values."""
    if isinstance(value, dict):
        return {key: public_state(item) for key, item in value.items() if key not in {"script", "env"}}
    if isinstance(value, list):
        return [public_state(item) for item in value]
    return value


def emit(value):
    print(json.dumps(public_state(value), indent=2, sort_keys=True), flush=True)


def secure_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def write_manifest(path, request):
    validate_request(request)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # A request is an immutable reviewable artifact. Choose a fresh filename
    # rather than overwriting a previous submission contract.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(request, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def start_runner(args):
    secure_directory(args.state_dir)
    fd = os.open(str(args.state_dir / "runner.log"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, "a") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--host",
                args.host,
                "--state-dir",
                str(args.state_dir),
                "run",
                "--duration",
                "43200",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    return process.pid


def run_loop(controller, backend, args):
    """One bounded local collector. Every SSH operation closes independently."""
    secure_directory(args.state_dir)
    suffix = hashlib.sha256(backend.scope.encode()).hexdigest()[:20]
    fd = os.open(str(args.state_dir / ("runner-" + suffix + ".lock")), os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"runner": "already-running"}
        info = {
            "pid": os.getpid(),
            "scope": backend.scope,
            "started_at": time.time(),
            "code_fingerprint": LOADED_CODE_FINGERPRINT,
        }
        lock.seek(0)
        lock.truncate()
        json.dump(info, lock, sort_keys=True)
        lock.flush()
        os.fsync(lock.fileno())
        deadline = time.monotonic() + args.duration
        previous = None
        while True:
            if _code_fingerprint() != LOADED_CODE_FINGERPRINT:
                return {"runner": "code-updated", "resume": "isambard-jobs start"}
            result = controller.tick()
            safe = public_state(result)
            digest = json.dumps(safe, sort_keys=True)
            if digest != previous:
                emit(safe)
                previous = digest
            state = controller.status()
            requests = state.get("requests", [])
            if isinstance(requests, dict):
                requests = list(requests.values())
            active = [
                r
                for r in requests
                if r.get("status") not in {"completed", "terminal", "withdrawn", "cancelled", "rejected", "finished"}
            ]
            # Authentication/transport failures pause dispatch. Do not create
            # a background certificate-renewal loop or retry unknown submits.
            error = state.get("last_error") or state.get("snapshot_error") or state.get("last_snapshot_error")
            if error:
                return {"runner": "paused", "reason": error, "resume": "isambard-jobs start"}
            if not active:
                return {"runner": "idle"}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"runner": "duration-ended", "resume": "isambard-jobs start"}
            time.sleep(min(60, remaining))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE,
        help="shared state directory; use a different directory only in isolated tests",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="read shared cached state without network access")
    sub.add_parser("tick", help="reconcile and dispatch once, respecting the shared 60-second limit")
    sub.add_parser("start", help="start one bounded local collector; no persistent SSH or renewal loop")
    run = sub.add_parser("run", help="run the local collector in the foreground")
    run.add_argument("--duration", type=int, default=43200)
    enqueue = sub.add_parser("enqueue", help="validate/canonicalize an immutable request and queue it")
    enqueue.add_argument("manifest", type=Path)
    enqueue.add_argument("--start", action="store_true", help="also start the local collector")
    for name in ("cancel", "withdraw"):
        cmd = sub.add_parser(name, help="explicit owner-only " + name)
        cmd.add_argument("request_id")
        cmd.add_argument("--owner", required=True)
    resolve = sub.add_parser(
        "resolve-unsubmitted", help="operator-attested recovery of an unknown request that never submitted"
    )
    resolve.add_argument("request_id")
    resolve.add_argument("--owner", required=True)
    resolve.add_argument("--note", required=True, help="evidence checked; stored in the audit record")
    resolve.add_argument(
        "--confirmed-never-submitted",
        action="store_true",
        required=True,
        help="explicit attestation after investigation; absence from squeue alone is insufficient",
    )
    protect = sub.add_parser("protect", help="observe existing jobs and reserve their outputs without changing them")
    protect.add_argument("--id", required=True)
    protect.add_argument("--owner", required=True)
    protect.add_argument("--job-id", action="append", required=True)
    protect.add_argument("--output-root", action="append", required=True)
    protect.add_argument("--remote-dir", required=True)
    protect.add_argument("--mode", choices=("interactive", "batch"), default="batch")
    prepare = sub.add_parser("prepare", help="create a reviewable experiment request; does not submit")
    prepare.add_argument("profile")
    prepare.add_argument("--id", required=True)
    prepare.add_argument("--owner", required=True)
    prepare.add_argument("--checkout", type=Path, required=True)
    prepare.add_argument("--remote-dir", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--mode", choices=("interactive", "batch"), required=True)
    prepare.add_argument("--minutes", type=int, required=True)
    prepare.add_argument(
        "--env", action="append", default=[], metavar="NAME=VALUE", help="non-secret configuration only"
    )
    prepare.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            try:
                from .job_adapters import build_request
            except ImportError:
                from job_adapters import build_request
            env = {}
            for entry in args.env:
                key, sep, value = entry.partition("=")
                if not sep or key in env:
                    raise ValueError("each --env must be a unique NAME=VALUE")
                env[key] = value
            request = build_request(
                args.profile,
                request_id=args.id,
                owner=args.owner,
                checkout=args.checkout,
                remote_dir=args.remote_dir,
                output_root=args.output_root,
                mode=args.mode,
                minutes=args.minutes,
                env=env,
            )
            write_manifest(args.output, request)
            emit({"manifest": str(args.output.resolve()), "request": request})
            return 0
        if args.command == "start":
            emit({"runner_pid": start_runner(args), "log": str(args.state_dir / "runner.log")})
            return 0
        backend = SSHBackend(args.host)
        controller = Controller(args.state_dir, backend, backend.scope)
        if args.command == "status":
            result = controller.status()
        elif args.command == "tick":
            result = controller.tick()
        elif args.command == "run":
            if not 1 <= args.duration <= 43200:
                raise ValueError("duration must be between 1 and 43200 seconds")
            result = run_loop(controller, backend, args)
        elif args.command == "enqueue":
            request = validate_request(json.loads(args.manifest.read_text()))
            request = backend.canonicalize(request)
            result = controller.enqueue(request)
            if args.start:
                result = {"request": result, "runner_pid": start_runner(args)}
        elif args.command == "cancel":
            result = controller.cancel(args.request_id, args.owner)
        elif args.command == "withdraw":
            result = controller.withdraw(args.request_id, args.owner)
        elif args.command == "resolve-unsubmitted":
            result = controller.resolve_unsubmitted(args.request_id, args.owner, args.note)
        elif args.command == "protect":
            claims = backend.canonicalize(dict(remote_dir=args.remote_dir, output_roots=args.output_root))
            result = controller.protect(args.id, args.owner, args.job_id, claims["output_roots"], mode=args.mode)
        else:
            raise ValueError("unsupported command")
        emit(result)
        if args.command in {"tick", "run"} and (
            result.get("last_snapshot_error") or result.get("runner") in {"paused", "code-updated"}
        ):
            return 2
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("isambard-jobs: " + str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped the local controller; queued requests and submitted jobs remain recorded.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
