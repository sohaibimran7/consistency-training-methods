"""Fresh SSH transport for the shared laptop Isambard job controller.

No credentials or job output are collected. All scheduler reads go through the
controller's rate limiter. Remote helpers run briefly on the login node; the
workload itself is submitted to Slurm.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import shlex
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path


def load_auth():
    candidate = Path(__file__).with_name("auth.py")
    if not candidate.exists():
        candidate = Path.home() / ".local/bin/isambard-auth"
    name = "_ctm_isambard_job_auth"
    if name in sys.modules:
        return sys.modules[name]
    loader = SourceFileLoader(name, str(candidate))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    return module


# Kept as a standalone program so the remote side needs only Python's stdlib.
# Structured arguments are encoded, shell-quoted, then decoded by this helper.
REMOTE_PROGRAM = r"""
import base64, datetime, hashlib, json, os, pathlib, re, subprocess, sys

data = json.loads(base64.b64decode(sys.argv[1]).decode())
action = data["action"]
user = data["user"]
os.environ["TZ"] = "UTC"
for key in list(os.environ):
    if key.startswith("SBATCH_"):
        del os.environ[key]

def command(argv, *, body=None):
    p = subprocess.run(argv, input=body, universal_newlines=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=22)
    if p.returncode:
        # Scheduler diagnostics may contain user paths, but never include script
        # bodies, environment values or job stdout in the control-plane response.
        raise RuntimeError(argv[0] + " failed: " + p.stderr.strip()[:600])
    return p.stdout

if action == "canonicalize":
    request = data["request"]
    root = pathlib.Path(request["remote_dir"]).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("remote_dir is not a directory")
    request["remote_dir"] = str(root)
    request["output_roots"] = [str(pathlib.Path(p).resolve()) for p in request["output_roots"]]
    if "/" in request["output_roots"]:
        raise RuntimeError("cannot reserve filesystem root")
    print(json.dumps(request))
elif action == "snapshot":
    rows = []
    live_ids = set()
    # The interactive partition is hidden from a default squeue listing on
    # AIP2. --all is essential even when restricting the query to one user.
    text = command(["squeue", "--all", "--noheader", "--array", "--user", user,
                    "--format", "%i|%128j|%T|%v|%u"])
    for line in text.splitlines():
        fields = [v.strip() for v in line.split("|")]
        if len(fields) != 5 or fields[4] != user:
            raise RuntimeError("unexpected squeue record")
        job_id, name, state, reservation, _ = fields
        live_ids.add(job_id)
        rows.append(dict(job_id=job_id, token=name, state=state,
                         reservation=reservation, source="squeue"))
    requests = data["requests"]
    # One bounded accounting query covers both known IDs and ambiguous submits.
    # Query by user/time rather than trusting absence from the live queue.
    watched = [r for r in requests if r.get("status") not in
               ("queued", "withdrawn", "completed", "terminal", "rejected")]
    tokens = {r.get("token", "") for r in watched}
    ids = {str(r["job_id"]) for r in watched if r.get("job_id")}
    for r in watched:
        ids.update(str(j) for j in r.get("job_ids", []))
    if watched:
        earliest = min(float(r.get("created_at", r.get("created", 0))) for r in watched)
        # The engine supplies epoch timestamps. Keep a margin around submission.
        if earliest <= 0:
            earliest = datetime.datetime.now(datetime.timezone.utc).timestamp() - 7*86400
        since = datetime.datetime.fromtimestamp(earliest - 3600, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        accounting = command(["sacct", "--allocations", "--noheader", "--parsable2",
                              "--user", user, "--starttime", since,
                              "--format", "JobIDRaw,JobName%128,State%40,End"])
        for line in accounting.splitlines():
            fields = line.split("|")
            if len(fields) < 4:
                raise RuntimeError("unexpected sacct record")
            job_id, name, state, end = [v.strip() for v in fields[:4]]
            if job_id in live_ids or (job_id not in ids and name not in tokens):
                continue
            # Ignore job steps; this controller submits non-array allocations.
            if not re.fullmatch(r"[0-9]+", job_id):
                continue
            state = state.split()[0].rstrip("+")
            # A historical PREEMPTED record can later be requeued. Never release
            # a lease on that ambiguous evidence. Missing end times are likewise
            # not terminal proof.
            if end in ("", "Unknown", "None") or state == "PREEMPTED":
                state = "UNKNOWN"
            rows.append(dict(job_id=job_id, token=name, state=state,
                             reservation="", source="sacct"))
    print(json.dumps(rows))
elif action == "submit":
    r = data["request"]
    res = r["resources"]
    token = r["token"]
    paths = [r["remote_dir"]] + r["output_roots"]
    if any(str(pathlib.Path(p).resolve()) != p for p in paths):
        print(json.dumps(dict(rejected="An output/repository alias changed after admission; prepare a new request.")))
        sys.exit(0)
    if not re.fullmatch(r"ctm-[A-Za-z0-9_-]{8,80}", token):
        raise RuntimeError("invalid controller token")
    spool = pathlib.Path(r["remote_dir"]) / ".isambard-jobs" / token
    # An existing intent directory is deliberately never replayed. If the SSH
    # response was lost, reconciliation must find the original scheduler job.
    spool.mkdir(mode=0o700, parents=True, exist_ok=False)
    script = "#!/usr/bin/env bash\nset -euo pipefail\numask 077\n"
    import shlex
    for name, value in sorted(r.get("env", {}).items()):
        script += "export " + name + "=" + shlex.quote(value) + "\n"
    script += "export SLURM_EXPORT_ENV=ALL\n"
    script += "export CTM_ISAMBARD_REQUEST_ID=" + shlex.quote(r["id"]) + "\n"
    script += "export CTM_ISAMBARD_OWNER=" + shlex.quote(r["owner"]) + "\n"
    script += r["script"]
    script_path = spool / "job.sh"
    with script_path.open("x") as f:
        os.chmod(str(script_path), 0o600)
        f.write(script)
        f.flush()
        os.fsync(f.fileno())
    argv = ["sbatch", "--parsable", "--job-name", token,
            "--nodes", str(res["nodes"]), "--gpus", str(res["gpus"]),
            "--time", str(res["minutes"]), "--mem", str(res["memory_mb"]) + "M",
            "--chdir", r["remote_dir"], "--output", str(spool / "slurm-%j.out"),
            "--error", str(spool / "slurm-%j.err"), "--export=NONE",
            "--no-requeue", "--open-mode=append"]
    for key, option in (("cpus_per_task", "--cpus-per-task"),
                        ("cpus_per_gpu", "--cpus-per-gpu"),
                        ("ntasks", "--ntasks"), ("ntasks_per_node", "--ntasks-per-node"),
                        ("gpus_per_node", "--gpus-per-node")):
        if key in res:
            argv += [option, str(res[key])]
    if r["mode"] == "interactive":
        argv += ["--reservation=interactive"]
    else:
        argv += ["--partition=workq"]
    # Allocation-wide output paths were canonicalized before local admission.
    # The submitted shell exports only the declared non-secret configuration.
    try:
        command(argv + ["--test-only", str(script_path)])
    except Exception as exc:
        print(json.dumps(dict(rejected=str(exc))))
        sys.exit(0)
    result = command(argv + [str(script_path)]).strip()
    job_id = result.split(";")[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError("sbatch did not return a job ID; reconcile before retrying")
    receipt = spool / "submission.json"
    with receipt.open("x") as f:
        os.chmod(str(receipt), 0o600)
        json.dump(dict(job_id=job_id, token=token,
                       script_sha256=hashlib.sha256(script.encode()).hexdigest()), f)
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(job_id=job_id)))
elif action == "cancel":
    job_id = data["job_id"]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError("invalid job ID")
    command(["scancel", "--user", user, job_id])
    print("{}")
else:
    raise RuntimeError("unsupported action")
"""


class SSHBackend:
    def __init__(self, host="a5v.aip2.isambard"):
        self.host = host
        self.auth = load_auth()
        report = self.auth.collect_status(host)
        self.user = report.target.user
        self.scope = "{}:{}/{}".format(report.target.hostname, report.target.port, self.user)
        self._checked = False

    def _remote(self, action, **kwargs):
        report = self.auth.collect_status(self.host)
        if not report.usable:
            raise RuntimeError(
                "Isambard authentication needs attention; run isambard-auth status/check and renew on demand."
            )
        if not self._checked:
            check = self.auth.fresh_check(report)
            if check.exit_code:
                raise RuntimeError(check.message)
            self._checked = True
        payload = dict(action=action, user=self.user, **kwargs)
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        argv = self.auth.build_check_command(report)
        argv[-1] = " ".join(shlex.quote(p) for p in ("python3", "-c", REMOTE_PROGRAM, encoded))
        try:
            result = self.auth.run_fresh_ssh(argv)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Fresh SSH operation timed out; reconcile any submission before retrying.") from exc
        if result.returncode:
            # Report the remote exception's last line rather than a traceback
            # (which can include the generated command/source).
            last_line = result.stderr.strip().splitlines()[-1:] or ["remote operation failed"]
            raise RuntimeError(last_line[0][:800])
        try:
            return json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Remote response was incomplete; retain submission ownership.") from exc

    def canonicalize(self, request):
        return self._remote("canonicalize", request=request)

    def snapshot(self, requests):
        fields = ("id", "token", "job_id", "job_ids", "status", "created_at", "created")
        small = [{key: r[key] for key in fields if key in r} for r in requests if r.get("status") != "terminal"]
        return self._remote("snapshot", requests=small)

    def submit(self, request):
        result = self._remote("submit", request=request)
        if "rejected" in result:
            try:
                from .job_controller import SubmissionRejected
            except ImportError:
                from job_controller import SubmissionRejected
            raise SubmissionRejected(result["rejected"])
        return result["job_id"]

    def cancel(self, job_id):
        self._remote("cancel", job_id=str(job_id))


def validate_request(request):
    """Validate the transport boundary before saving or shell-quoting a request."""
    script = request.get("script", "")
    if not isinstance(script, str) or not script.startswith("#!") or len(script.encode()) > 65536 or "\x00" in script:
        raise ValueError("script must be a captured shell script of at most 64 KiB")
    if re.search(r"^\s*#SBATCH\b", script, re.MULTILINE):
        raise ValueError("remove #SBATCH directives; declare all resources in the request")
    resources = request.get("resources", {})
    allowed = {
        "nodes",
        "gpus",
        "minutes",
        "memory_mb",
        "cpus_per_task",
        "cpus_per_gpu",
        "ntasks",
        "ntasks_per_node",
        "gpus_per_node",
    }
    if set(resources) - allowed:
        raise ValueError("unsupported resource fields")
    for key in ("nodes", "gpus", "minutes", "memory_mb"):
        if key not in resources:
            raise ValueError("missing resource field: " + key)
    for key, value in resources.items():
        if type(value) is not int or value < 1:
            raise ValueError("resources must be positive integers: " + key)
    if "cpus_per_task" in resources and "cpus_per_gpu" in resources:
        raise ValueError("choose cpus_per_task or cpus_per_gpu, not both")
    if "gpus_per_node" in resources and resources["gpus_per_node"] * resources["nodes"] != resources["gpus"]:
        raise ValueError("GPU resource fields disagree")
    if resources["gpus"] > 4 * resources["nodes"]:
        raise ValueError("AIP2 nodes have four GPUs")
    if request.get("mode") == "interactive" and (
        resources["nodes"] > 4 or resources["gpus"] > 16 or resources["minutes"] > 480
    ):
        raise ValueError("interactive limit is 4 nodes, 16 GPUs, 480 minutes")
    for name, value in request.get("env", {}).items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) or not isinstance(value, str) or "\x00" in value:
            raise ValueError("invalid environment entry")
        if re.search(r"PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|PRIVATE_KEY|CREDENTIAL", name):
            raise ValueError("credential-like environment values must not be stored in controller requests: " + name)
        if name.startswith(("SBATCH_", "SLURM_", "CTM_ISAMBARD_")):
            raise ValueError("scheduler/controller environment is reserved: " + name)
    return request
