#!/usr/bin/env python3
"""Check and renew the short-lived Clifton SSH certificate used by Isambard.

``status`` only reads the local SSH configuration and public certificate
metadata.  ``check`` makes one fresh, non-interactive connection after a
usable certificate has been confirmed.  ``renew`` is deliberately the only
command that starts Clifton's interactive device flow.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import shlex
import signal
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

DEFAULT_HOST = "a5v.aip2.isambard"
CONFIG_TIMEOUT_SECONDS = 10
CHECK_TIMEOUT_SECONDS = 35
CHECK_CONNECT_TIMEOUT_SECONDS = 12
RENEW_COMMAND = ["clifton", "auth", "--open-browser", "false", "--show-qr", "false", "--write-config", "true"]
LOCK_PATH = Path.home() / ".cache" / "ctm" / "isambard-auth.lock"

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_CERTIFICATE = 3
EXIT_AUTH = 4
EXIT_TRANSPORT = 5
EXIT_RENEW_FAILED = 6
EXIT_BUSY = 7

_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HOST_WITH_USER_RE = re.compile(r"^(?:[A-Za-z0-9][A-Za-z0-9._-]*@)?[A-Za-z0-9][A-Za-z0-9._-]*$")
_FINGERPRINT_RE = re.compile(r"\b(SHA256:[A-Za-z0-9+/=]+)")
_VALIDITY_RE = re.compile(r"^\s*Valid:\s+from\s+(\S+)\s+to\s+(\S+)\s*$", re.MULTILINE)


class ToolError(Exception):
    """A concise, user-safe problem that maps to a stable CLI exit code."""

    def __init__(self, category: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.category = category
        self.message = message
        self.exit_code = exit_code

    def __str__(self) -> str:
        return self.message


class ConfigError(ToolError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(category, message, EXIT_CONFIG)


class CertificateError(ToolError):
    def __init__(self, category: str, message: str) -> None:
        super().__init__(category, message, EXIT_CERTIFICATE)


class RenewalError(ToolError):
    def __init__(self, category: str, message: str, exit_code: int = EXIT_RENEW_FAILED) -> None:
        super().__init__(category, message, exit_code)


@dataclass(frozen=True)
class Endpoint:
    """The relevant effective settings for one SSH destination."""

    invocation: str
    host: str
    user: str
    hostname: str
    port: str
    identity_file: str
    certificate_file: str
    proxy_jump: Optional[str] = None


@dataclass(frozen=True)
class CertificateMetadata:
    path: str
    fingerprint: str
    valid_from: datetime
    valid_until: datetime
    principals: Tuple[str, ...]


@dataclass(frozen=True)
class Issue:
    category: str
    message: str
    exit_code: int = EXIT_CERTIFICATE


@dataclass
class StatusReport:
    requested_host: str
    target: Endpoint
    jump: Endpoint
    certificates: Dict[str, CertificateMetadata]
    identity_fingerprints: Dict[str, str]
    issues: List[Issue] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return max((issue.exit_code for issue in self.issues), default=EXIT_OK)

    @property
    def usable(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class CheckResult:
    category: str
    message: str
    exit_code: int


def run_command(
    args: Sequence[str],
    *,
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    stdin: object = None,
) -> subprocess.CompletedProcess:
    """Run a fixed argv command without a shell.

    Commands that inspect public SSH metadata are captured so diagnostics can
    be reduced to safe summaries.  Clifton renewal deliberately inherits the
    terminal and therefore passes ``capture_output=False``.
    """

    kwargs = {"check": False}
    if timeout is not None:
        kwargs["timeout"] = timeout
    if env is not None:
        kwargs["env"] = env
    if stdin is not None:
        kwargs["stdin"] = stdin
    if capture_output:
        kwargs.update({"stdout": subprocess.PIPE, "stderr": subprocess.PIPE, "text": True})
    return subprocess.run(list(args), **kwargs)  # type: ignore[arg-type]


def _output(result: subprocess.CompletedProcess, stream: str) -> str:
    value = getattr(result, stream, "")
    return value if isinstance(value, str) else ""


def _validate_host(value: str, *, allow_user: bool = False) -> str:
    pattern = _HOST_WITH_USER_RE if allow_user else _HOST_RE
    if not pattern.fullmatch(value):
        raise ConfigError(
            "invalid-host", "Host must be a plain SSH alias or hostname; ports and shell syntax are unsupported."
        )
    return value


def parse_ssh_g(output: str) -> Dict[str, List[str]]:
    """Parse OpenSSH's machine-readable ``ssh -G`` output."""

    values: Dict[str, List[str]] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(" ")
        if not separator or not key or not value:
            continue
        values.setdefault(key.lower(), []).append(value.strip())
    return values


def _first_setting(settings: Dict[str, List[str]], name: str) -> Optional[str]:
    for value in settings.get(name, []):
        if value and value.lower() != "none":
            return value
    return None


def _display_host(invocation: str) -> str:
    return invocation.rsplit("@", 1)[-1]


def effective_endpoint(invocation: str, *, require_proxy_jump: bool) -> Endpoint:
    """Read an endpoint's effective local SSH configuration without connecting."""

    _validate_host(invocation, allow_user=True)
    try:
        result = run_command(["ssh", "-G", invocation], timeout=CONFIG_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise ConfigError("ssh-unavailable", "ssh is not available on this computer.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ConfigError("ssh-config-timeout", "Timed out while reading local SSH configuration.") from exc
    if result.returncode != 0:
        raise ConfigError("ssh-config-error", "ssh could not read the local configuration for {!r}.".format(invocation))

    settings = parse_ssh_g(_output(result, "stdout"))
    hostname = _first_setting(settings, "hostname")
    user = _first_setting(settings, "user")
    port = _first_setting(settings, "port") or "22"
    identity_file = _first_setting(settings, "identityfile")
    certificate_file = _first_setting(settings, "certificatefile")
    proxy_jump = _first_setting(settings, "proxyjump")
    requested_name = _display_host(invocation)

    if not certificate_file:
        if hostname == requested_name and not proxy_jump:
            raise ConfigError(
                "unknown-alias",
                "SSH alias {!r} has no Clifton configuration (no HostName, CertificateFile, or ProxyJump override).".format(
                    requested_name
                ),
            )
        raise ConfigError(
            "missing-certificate", "SSH configuration for {!r} has no CertificateFile.".format(invocation)
        )
    if not identity_file:
        raise ConfigError("missing-identity", "SSH configuration for {!r} has no IdentityFile.".format(invocation))
    if not hostname or not user:
        raise ConfigError(
            "incomplete-config", "SSH configuration for {!r} is missing HostName or User.".format(invocation)
        )
    if require_proxy_jump and not proxy_jump:
        raise ConfigError("missing-proxyjump", "SSH configuration for {!r} has no ProxyJump.".format(invocation))

    return Endpoint(
        invocation=invocation,
        host=requested_name,
        user=user,
        hostname=hostname,
        port=port,
        identity_file=os.path.expanduser(identity_file),
        certificate_file=os.path.expanduser(certificate_file),
        proxy_jump=proxy_jump,
    )


def expand_proxy_jump(proxy_jump: str, endpoint: Endpoint) -> str:
    """Expand the single ProxyJump hop used by the local Isambard config."""

    raw = proxy_jump.strip()
    if not raw or raw.lower() == "none":
        raise ConfigError("missing-proxyjump", "The target SSH configuration has no usable ProxyJump.")
    if "," in raw:
        raise ConfigError("multi-hop-proxyjump", "This helper only supports a single ProxyJump hop.")

    substitutions = {"r": endpoint.user, "n": endpoint.host, "h": endpoint.hostname, "p": endpoint.port, "%": "%"}
    expanded: List[str] = []
    index = 0
    while index < len(raw):
        character = raw[index]
        if character != "%":
            expanded.append(character)
            index += 1
            continue
        if index + 1 >= len(raw) or raw[index + 1] not in substitutions:
            raise ConfigError("unsupported-proxyjump", "ProxyJump uses an unsupported SSH token.")
        expanded.append(substitutions[raw[index + 1]])
        index += 2

    jump = "".join(expanded)
    _validate_host(jump, allow_user=True)
    return jump


def _parse_timestamp(value: str) -> datetime:
    cleaned = value.rstrip("Z")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise CertificateError(
            "malformed-certificate", "Certificate validity timestamp is not a valid UTC time."
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_certificate_metadata(path: str, output: str) -> CertificateMetadata:
    """Extract only certificate metadata needed to assess authentication."""

    fingerprint_match = re.search(r"^\s*Public key:\s+\S+\s+(SHA256:[A-Za-z0-9+/=]+)\s*$", output, re.MULTILINE)
    validity_match = _VALIDITY_RE.search(output)
    if not fingerprint_match or not validity_match:
        raise CertificateError(
            "malformed-certificate", "Certificate metadata is missing its public-key fingerprint or validity period."
        )

    principals: List[str] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith("Principals:"):
            continue
        inline = line.split(":", 1)[1].strip()
        if inline and inline.lower() != "(none)":
            principals.append(inline)
        for candidate in lines[index + 1 :]:
            if not candidate.startswith((" ", "\t")):
                break
            stripped = candidate.strip()
            if not stripped or stripped == "(none)" or re.match(r"^(Critical Options|Extensions):", stripped):
                break
            principals.append(stripped)
        break

    if not principals:
        raise CertificateError("malformed-certificate", "Certificate metadata has no login principals.")

    valid_from = _parse_timestamp(validity_match.group(1))
    valid_until = _parse_timestamp(validity_match.group(2))
    if valid_until <= valid_from:
        raise CertificateError("malformed-certificate", "Certificate validity period is not ordered correctly.")
    return CertificateMetadata(
        path=path,
        fingerprint=fingerprint_match.group(1),
        valid_from=valid_from,
        valid_until=valid_until,
        principals=tuple(principals),
    )


def inspect_certificate(path: str) -> CertificateMetadata:
    """Run ssh-keygen in UTC so its local-time validity text is unambiguous."""

    environment = os.environ.copy()
    environment["TZ"] = "UTC"
    try:
        result = run_command(["ssh-keygen", "-L", "-f", path], timeout=CONFIG_TIMEOUT_SECONDS, env=environment)
    except FileNotFoundError as exc:
        raise ConfigError("ssh-keygen-unavailable", "ssh-keygen is not available on this computer.") from exc
    except subprocess.TimeoutExpired as exc:
        raise CertificateError(
            "certificate-read-timeout", "Timed out while reading the public certificate metadata."
        ) from exc
    if result.returncode != 0:
        raise ConfigError("missing-certificate", "CertificateFile {!r} cannot be read.".format(path))
    return parse_certificate_metadata(path, _output(result, "stdout"))


def identity_public_path(identity_file: str) -> str:
    return identity_file if identity_file.endswith(".pub") else identity_file + ".pub"


def inspect_public_key_fingerprint(identity_file: str) -> str:
    public_path = identity_public_path(identity_file)
    try:
        result = run_command(["ssh-keygen", "-l", "-f", public_path], timeout=CONFIG_TIMEOUT_SECONDS)
    except FileNotFoundError as exc:
        raise ConfigError("ssh-keygen-unavailable", "ssh-keygen is not available on this computer.") from exc
    except subprocess.TimeoutExpired as exc:
        raise ConfigError("identity-read-timeout", "Timed out while reading the public identity metadata.") from exc
    if result.returncode != 0:
        raise ConfigError("missing-identity-public-key", "Public key {!r} cannot be read.".format(public_path))
    fingerprint_match = _FINGERPRINT_RE.search(_output(result, "stdout"))
    if not fingerprint_match:
        raise ConfigError(
            "malformed-identity-public-key", "Public key fingerprint could not be read from {!r}.".format(public_path)
        )
    return fingerprint_match.group(1)


def _certificate_issues(
    endpoint: Endpoint, certificate: CertificateMetadata, identity_fingerprint: str, now: datetime
) -> List[Issue]:
    issues: List[Issue] = []
    if certificate.fingerprint != identity_fingerprint:
        issues.append(
            Issue(
                "certificate-key-mismatch",
                "CertificateFile {!r} does not match {}.pub.".format(certificate.path, endpoint.identity_file),
            )
        )
    if endpoint.user not in certificate.principals:
        issues.append(
            Issue(
                "certificate-principal-mismatch",
                "CertificateFile {!r} does not permit SSH user {!r}.".format(certificate.path, endpoint.user),
            )
        )
    if now < certificate.valid_from:
        issues.append(
            Issue("certificate-not-yet-valid", "CertificateFile {!r} is not valid yet.".format(certificate.path))
        )
    elif now >= certificate.valid_until:
        issues.append(Issue("certificate-expired", "CertificateFile {!r} has expired.".format(certificate.path)))
    return issues


def collect_status(host: str = DEFAULT_HOST, *, now: Optional[datetime] = None) -> StatusReport:
    """Collect local configuration and public certificate health without SSHing."""

    _validate_host(host)
    target, jump = resolve_route(host)

    certificates: Dict[str, CertificateMetadata] = {}
    identity_fingerprints: Dict[str, str] = {}
    for endpoint in (target, jump):
        if endpoint.certificate_file not in certificates:
            certificates[endpoint.certificate_file] = inspect_certificate(endpoint.certificate_file)
        if endpoint.identity_file not in identity_fingerprints:
            identity_fingerprints[endpoint.identity_file] = inspect_public_key_fingerprint(endpoint.identity_file)

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)

    issues: List[Issue] = []
    seen_issues = set()
    for endpoint in (target, jump):
        for issue in _certificate_issues(
            endpoint,
            certificates[endpoint.certificate_file],
            identity_fingerprints[endpoint.identity_file],
            current_time,
        ):
            key = (issue.category, issue.message, issue.exit_code)
            if key not in seen_issues:
                issues.append(issue)
                seen_issues.add(key)
    return StatusReport(
        requested_host=host,
        target=target,
        jump=jump,
        certificates=certificates,
        identity_fingerprints=identity_fingerprints,
        issues=issues,
    )


def resolve_route(host: str = DEFAULT_HOST) -> Tuple[Endpoint, Endpoint]:
    """Validate the two configured SSH hops without requiring a current cert."""

    _validate_host(host)
    target = effective_endpoint(host, require_proxy_jump=True)
    jump_invocation = expand_proxy_jump(target.proxy_jump or "", target)
    return target, effective_endpoint(jump_invocation, require_proxy_jump=False)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _duration_text(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    if total_seconds < 0:
        return "expired {} ago".format(_duration_text(-delta))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    pieces = []
    if days:
        pieces.append("{}d".format(days))
    if hours or days:
        pieces.append("{}h".format(hours))
    pieces.append("{}m".format(minutes))
    return " ".join(pieces)


def render_status(report: StatusReport, *, out: TextIO = sys.stdout, now: Optional[datetime] = None) -> None:
    """Print a short metadata-only status report."""

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)
    else:
        current_time = current_time.astimezone(timezone.utc)
    print("host: {} -> {}@{}".format(report.requested_host, report.target.user, report.target.hostname), file=out)
    print("jump: {} -> {}@{}".format(report.jump.invocation, report.jump.user, report.jump.hostname), file=out)
    for certificate in report.certificates.values():
        print("certificate: {}".format(certificate.path), file=out)
        print("fingerprint: {}".format(certificate.fingerprint), file=out)
        print("principals: {}".format(", ".join(certificate.principals)), file=out)
        print("valid from: {}".format(_utc_text(certificate.valid_from)), file=out)
        print(
            "valid until: {} ({})".format(
                _utc_text(certificate.valid_until), _duration_text(certificate.valid_until - current_time)
            ),
            file=out,
        )
    if report.issues:
        for issue in report.issues:
            print("status: {} — {}".format(issue.category, issue.message), file=out)
    else:
        print("status: certificate and SSH configuration are usable", file=out)


def _fresh_options() -> List[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout={}".format(CHECK_CONNECT_TIMEOUT_SECONDS),
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
    ]


def _shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def build_check_command(report: StatusReport) -> List[str]:
    """Build an explicitly two-hop, fresh SSH check without a shell locally."""

    jump_command = ["ssh"] + _fresh_options() + ["-o", "ProxyJump=none", "-W", "%h:%p", report.jump.invocation]
    proxy_command = _shell_join(jump_command)
    # ProxyCommand must appear before the configured ProxyJump.  OpenSSH uses
    # the first of these mutually exclusive settings it sees.
    return (
        ["ssh"] + _fresh_options() + ["-o", "ProxyCommand={}".format(proxy_command), report.target.invocation, "true"]
    )


def classify_ssh_failure(output: str) -> CheckResult:
    """Classify a nonzero fresh SSH check without dumping SSH's raw output."""

    text = output.lower()
    if any(
        marker in text
        for marker in ("permission denied", "publickey", "too many authentication failures", "sign_and_send_pubkey")
    ):
        return CheckResult("authentication", "SSH rejected the certificate or public-key authentication.", EXIT_AUTH)
    if any(
        marker in text
        for marker in (
            "could not resolve hostname",
            "name or service not known",
            "nodename nor servname",
            "temporary failure in name resolution",
        )
    ):
        return CheckResult("dns", "SSH could not resolve an Isambard host name.", EXIT_TRANSPORT)
    if any(
        marker in text
        for marker in (
            "host key verification failed",
            "remote host identification has changed",
            "no matching host key type found",
        )
    ):
        return CheckResult(
            "host-key",
            "SSH rejected the configured host key; inspect known_hosts and SSH host-key settings.",
            EXIT_CONFIG,
        )
    if any(
        marker in text
        for marker in (
            "bad configuration option",
            "unknown configuration option",
            "unknown option",
            "invalid proxycommand",
            "illegal option",
        )
    ):
        return CheckResult("configuration", "SSH rejected the local connection configuration.", EXIT_CONFIG)
    if any(
        marker in text
        for marker in (
            "connection timed out",
            "operation timed out",
            "connection refused",
            "no route to host",
            "network is unreachable",
            "connection closed",
            "kex_exchange_identification",
        )
    ):
        return CheckResult("transport", "SSH could not establish the fresh Isambard connection.", EXIT_TRANSPORT)
    return CheckResult("transport", "Fresh SSH check failed before completing the remote command.", EXIT_TRANSPORT)


def run_fresh_ssh(args: Sequence[str]) -> subprocess.CompletedProcess:
    """Run the owned probe in its own process group and clean it up on timeout."""

    process = subprocess.Popen(
        list(args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=CHECK_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        raise
    return subprocess.CompletedProcess(list(args), process.returncode, stdout, stderr)


def fresh_check(report: StatusReport) -> CheckResult:
    """Run a bounded, non-interactive `true` over the explicit two-hop path."""

    if not report.usable:
        return CheckResult("certificate", "Certificate status is not usable; SSH was not attempted.", report.exit_code)
    try:
        result = run_fresh_ssh(build_check_command(report))
    except FileNotFoundError:
        return CheckResult("configuration", "ssh is not available on this computer.", EXIT_CONFIG)
    except subprocess.TimeoutExpired:
        return CheckResult(
            "transport", "Fresh SSH check timed out after {} seconds.".format(CHECK_TIMEOUT_SECONDS), EXIT_TRANSPORT
        )
    if result.returncode == 0:
        return CheckResult("ok", "Fresh two-hop SSH check succeeded.", EXIT_OK)
    return classify_ssh_failure("{}\n{}".format(_output(result, "stdout"), _output(result, "stderr")))


def clifton_process_is_running() -> Optional[bool]:
    """Look for another user-owned Clifton process without reading its arguments."""

    try:
        result = run_command(["ps", "-x", "-o", "comm="], timeout=CONFIG_TIMEOUT_SECONDS)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return any(Path(line.strip()).name == "clifton" for line in _output(result, "stdout").splitlines() if line.strip())


@contextmanager
def renewal_lock(path: Path = LOCK_PATH) -> Iterator[None]:
    """Acquire the per-user advisory renewal lock; leave the lock file intact."""

    descriptor = -1
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.chmod(str(path.parent), 0o700)
        except OSError:
            pass
        descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise RenewalError("lock-error", "Could not prepare the per-user Clifton renewal lock.") from exc
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RenewalError(
                "renewal-in-progress", "Another Isambard certificate renewal is already in progress.", EXIT_BUSY
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def renew(host: str = DEFAULT_HOST, *, out: TextIO = sys.stdout, lock_path: Path = LOCK_PATH) -> int:
    """Run the deliberate Clifton device flow, then require status and SSH success."""

    with renewal_lock(lock_path):
        try:
            resolve_route(host)
        except ToolError as error:
            print("{}: {}".format(error.category, error.message), file=out)
            return error.exit_code
        process_running = clifton_process_is_running()
        if process_running is None:
            print(
                "renewal-check-failed: could not verify whether another Clifton process is already running.", file=out
            )
            return EXIT_BUSY
        if process_running:
            print("renewal-in-progress: another Clifton process is already running.", file=out)
            return EXIT_BUSY
        try:
            result = run_command(RENEW_COMMAND, capture_output=False)
        except FileNotFoundError:
            print("renewal-failed: clifton is not available on this computer.", file=out)
            return EXIT_RENEW_FAILED
        if result.returncode != 0:
            print("renewal-failed: clifton auth exited without a renewed certificate.", file=out)
            return EXIT_RENEW_FAILED

        try:
            report = collect_status(host)
        except ToolError as error:
            print("{}: {}".format(error.category, error.message), file=out)
            return error.exit_code
        render_status(report, out=out)
        if not report.usable:
            return report.exit_code
        check = fresh_check(report)
        print("check: {} — {}".format(check.category, check.message), file=out)
        return check.exit_code


def _run_status(host: str, out: TextIO) -> int:
    try:
        report = collect_status(host)
    except ToolError as error:
        print("{}: {}".format(error.category, error.message), file=out)
        return error.exit_code
    render_status(report, out=out)
    return report.exit_code


def _run_check(host: str, out: TextIO) -> int:
    try:
        report = collect_status(host)
    except ToolError as error:
        print("{}: {}".format(error.category, error.message), file=out)
        return error.exit_code
    render_status(report, out=out)
    if not report.usable:
        print("check: certificate — SSH was not attempted.", file=out)
        return report.exit_code
    check = fresh_check(report)
    print("check: {} — {}".format(check.category, check.message), file=out)
    return check.exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("status", "check", "renew"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("host", nargs="?", default=DEFAULT_HOST, help="SSH alias (default: %(default)s)")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, out: TextIO = sys.stdout) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_host(args.host)
    except ToolError as error:
        print("{}: {}".format(error.category, error.message), file=out)
        return error.exit_code
    if args.command == "status":
        return _run_status(args.host, out)
    if args.command == "check":
        return _run_check(args.host, out)
    try:
        return renew(args.host, out=out)
    except ToolError as error:
        print("{}: {}".format(error.category, error.message), file=out)
        return error.exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
