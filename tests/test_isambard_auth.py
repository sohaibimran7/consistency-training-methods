"""Focused tests for the local Isambard/Clifton authentication helper."""

from __future__ import annotations

import io
import shlex
import shutil
import subprocess
from datetime import datetime, timezone

import pytest

from infra.isambard import auth

FINGERPRINT = "SHA256:certificateFingerprint0123456789abc="
IDENTITY = "/tmp/id_ed25519"
CERTIFICATE = "/tmp/isambard-cert.pub"
TARGET = "a5v.aip2.isambard"
JUMP = "sohaib.a5v@jump.a5v.aip2.isambard"
NOW = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


def ssh_config(*, host, hostname, user="sohaib.a5v", certificate=CERTIFICATE, identity=IDENTITY, proxy_jump=None):
    lines = ["host {}".format(host), "user {}".format(user), "hostname {}".format(hostname), "port 22"]
    if identity is not None:
        lines.append("identityfile {}".format(identity))
    if certificate is not None:
        lines.append("certificatefile {}".format(certificate))
    if proxy_jump is not None:
        lines.append("proxyjump {}".format(proxy_jump))
    return "\n".join(lines) + "\n"


def certificate_output(
    *,
    fingerprint=FINGERPRINT,
    valid_from="2026-08-25T10:00:00",
    valid_until="2026-08-25T22:00:00",
    principal="sohaib.a5v",
):
    return """{path}:
        Type: ssh-ed25519-cert-v01@openssh.com user certificate
        Public key: ED25519-CERT {fingerprint}
        Signing CA: ED25519 SHA256:authority
        Valid: from {valid_from} to {valid_until}
        Principals:
                {principal}
        Critical Options: (none)
        Extensions:
                permit-pty
""".format(
        path=CERTIFICATE,
        fingerprint=fingerprint,
        valid_from=valid_from,
        valid_until=valid_until,
        principal=principal,
    )


def install_status_runner(
    monkeypatch, *, certificate=None, identity_fingerprint=FINGERPRINT, target_config=None, jump_config=None
):
    calls = []
    certificate = certificate or certificate_output()
    target_config = target_config or ssh_config(
        host=TARGET,
        hostname="ai-p2.access.isambard.ac.uk",
        proxy_jump="%r@jump.%n",
    )
    jump_config = jump_config or ssh_config(host="jump.a5v.aip2.isambard", hostname="ai.login.isambard.ac.uk")

    def fake_run(args, **kwargs):
        args = list(args)
        calls.append((args, kwargs))
        if args[:2] == ["ssh", "-G"]:
            output = jump_config if args[2] == JUMP else target_config
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:3] == ["ssh-keygen", "-L", "-f"]:
            return subprocess.CompletedProcess(args, 0, certificate, "")
        if args[:3] == ["ssh-keygen", "-l", "-f"]:
            return subprocess.CompletedProcess(args, 0, "256 {} test-key (ED25519)\n".format(identity_fingerprint), "")
        raise AssertionError("unexpected command: {!r}".format(args))

    monkeypatch.setattr(auth, "run_command", fake_run)
    return calls


def valid_report():
    certificate = auth.CertificateMetadata(
        path=CERTIFICATE,
        fingerprint=FINGERPRINT,
        valid_from=datetime(2026, 8, 25, 10, tzinfo=timezone.utc),
        valid_until=datetime(2026, 8, 25, 22, tzinfo=timezone.utc),
        principals=("sohaib.a5v",),
    )
    target = auth.Endpoint(
        invocation=TARGET,
        host=TARGET,
        user="sohaib.a5v",
        hostname="ai-p2.access.isambard.ac.uk",
        port="22",
        identity_file=IDENTITY,
        certificate_file=CERTIFICATE,
        proxy_jump="%r@jump.%n",
    )
    jump = auth.Endpoint(
        invocation=JUMP,
        host="jump.a5v.aip2.isambard",
        user="sohaib.a5v",
        hostname="ai.login.isambard.ac.uk",
        port="22",
        identity_file=IDENTITY,
        certificate_file=CERTIFICATE,
    )
    return auth.StatusReport(TARGET, target, jump, {CERTIFICATE: certificate}, {IDENTITY: FINGERPRINT})


def _report_with_issue():
    report = valid_report()
    report.issues.append(auth.Issue("certificate-expired", "expired"))
    return report


def test_status_reads_both_effective_hops_and_public_metadata_in_utc(monkeypatch):
    calls = install_status_runner(monkeypatch)

    report = auth.collect_status(now=NOW)

    assert report.usable
    assert report.target.hostname == "ai-p2.access.isambard.ac.uk"
    assert report.jump.invocation == JUMP
    assert report.jump.hostname == "ai.login.isambard.ac.uk"
    assert report.certificates[CERTIFICATE].principals == ("sohaib.a5v",)
    assert [args for args, _ in calls if args[:2] == ["ssh", "-G"]] == [["ssh", "-G", TARGET], ["ssh", "-G", JUMP]]
    certificate_call = next((args, kwargs) for args, kwargs in calls if args[:3] == ["ssh-keygen", "-L", "-f"])
    assert certificate_call[1]["env"]["TZ"] == "UTC"
    assert all(args[0] != "ssh" or args[1] == "-G" for args, _ in calls)


def test_unknown_alias_and_missing_certificate_are_configuration_errors(monkeypatch):
    unknown = ssh_config(host="isambard", hostname="isambard", user="work", certificate=None, proxy_jump=None)
    install_status_runner(monkeypatch, target_config=unknown)
    with pytest.raises(auth.ConfigError) as unknown_error:
        auth.collect_status("isambard", now=NOW)
    assert unknown_error.value.category == "unknown-alias"
    assert unknown_error.value.exit_code == auth.EXIT_CONFIG

    configured_but_missing_cert = ssh_config(
        host=TARGET,
        hostname="ai-p2.access.isambard.ac.uk",
        certificate=None,
        proxy_jump="%r@jump.%n",
    )
    install_status_runner(monkeypatch, target_config=configured_but_missing_cert)
    with pytest.raises(auth.ConfigError) as missing_error:
        auth.collect_status(now=NOW)
    assert missing_error.value.category == "missing-certificate"


@pytest.mark.parametrize(
    "output",
    [
        "not a certificate\n",
        "        Public key: ED25519-CERT {}\n".format(FINGERPRINT),
        "        Valid: from 2026-08-25T10:00:00 to 2026-08-25T22:00:00\n",
        certificate_output(valid_from="not-a-time"),
        certificate_output(valid_from="2026-08-25T22:00:00", valid_until="2026-08-25T10:00:00"),
    ],
)
def test_malformed_certificate_metadata_is_rejected(output):
    with pytest.raises(auth.CertificateError) as error:
        auth.parse_certificate_metadata(CERTIFICATE, output)
    assert error.value.category == "malformed-certificate"


def test_missing_certificate_file_is_reported_separately(monkeypatch):
    monkeypatch.setattr(
        auth,
        "run_command",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1, "", "missing"),
    )

    with pytest.raises(auth.ConfigError) as error:
        auth.inspect_certificate(CERTIFICATE)

    assert error.value.category == "missing-certificate"


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 8, 25, 9, tzinfo=timezone.utc), "certificate-not-yet-valid"),
        (datetime(2026, 8, 25, 23, tzinfo=timezone.utc), "certificate-expired"),
    ],
)
def test_status_distinguishes_not_yet_valid_and_expired_certificates(monkeypatch, now, expected):
    install_status_runner(monkeypatch)

    report = auth.collect_status(now=now)

    assert not report.usable
    assert {issue.category for issue in report.issues} == {expected}
    assert report.exit_code == auth.EXIT_CERTIFICATE


def test_status_accepts_matching_key_and_principal_but_reports_mismatches(monkeypatch):
    calls = install_status_runner(monkeypatch)
    assert auth.collect_status(now=NOW).usable
    assert calls

    install_status_runner(
        monkeypatch,
        certificate=certificate_output(fingerprint="SHA256:other", principal="other-user"),
        identity_fingerprint="SHA256:identity",
    )
    report = auth.collect_status(now=NOW)
    assert {issue.category for issue in report.issues} == {"certificate-key-mismatch", "certificate-principal-mismatch"}


def test_proxyjump_expansion_uses_target_user_and_original_alias():
    endpoint = valid_report().target

    assert auth.expand_proxy_jump("%r@jump.%n", endpoint) == JUMP
    with pytest.raises(auth.ConfigError, match="single ProxyJump"):
        auth.expand_proxy_jump("first,second", endpoint)
    with pytest.raises(auth.ConfigError, match="unsupported"):
        auth.expand_proxy_jump("%x@jump", endpoint)


def test_fresh_check_uses_fresh_options_on_both_hops_and_safe_proxycommand():
    command = auth.build_check_command(valid_report())
    outer_options = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-o"]
    proxy = next(value.split("=", 1)[1] for value in outer_options if value.startswith("ProxyCommand="))
    inner = shlex.split(proxy)
    inner_options = [inner[index + 1] for index, value in enumerate(inner[:-1]) if value == "-o"]

    for option_list in (outer_options, inner_options):
        assert "BatchMode=yes" in option_list
        assert "ConnectTimeout=12" in option_list
        assert "ConnectionAttempts=1" in option_list
        assert "StrictHostKeyChecking=yes" in option_list
        assert "ControlMaster=no" in option_list
        assert "ControlPath=none" in option_list
    assert "ProxyJump=none" in inner_options
    assert inner[-3:] == ["-W", "%h:%p", JUMP]
    assert command[-2:] == [TARGET, "true"]


@pytest.mark.skipif(shutil.which("ssh") is None, reason="OpenSSH is required for this precedence check")
def test_actual_ssh_g_keeps_command_line_proxycommand_ahead_of_configured_proxyjump(tmp_path):
    config = tmp_path / "ssh_config"
    config.write_text(
        "Host {}\n  HostName target.example\n  ProxyJump configured@jump.example\n".format(TARGET), encoding="utf-8"
    )
    command = auth.build_check_command(valid_report())
    probe = ["ssh", "-F", str(config), "-G"] + command[1:-1]

    result = subprocess.run(probe, check=True, capture_output=True, text=True)

    assert any(line.startswith("proxycommand ") for line in result.stdout.splitlines())
    assert "proxyjump configured@jump.example" not in result.stdout


def test_fresh_check_skips_ssh_for_bad_certificate_and_classifies_ssh_errors(monkeypatch):
    invalid = valid_report()
    invalid.issues.append(auth.Issue("certificate-expired", "expired"))
    monkeypatch.setattr(auth, "run_fresh_ssh", lambda args: pytest.fail("SSH must not run with an invalid cert"))
    assert auth.fresh_check(invalid).category == "certificate"

    report = valid_report()
    seen = []

    def rejected(command):
        seen.append(command)
        return subprocess.CompletedProcess(command, 255, "", "Permission denied (publickey).")

    monkeypatch.setattr(auth, "run_fresh_ssh", rejected)
    check = auth.fresh_check(report)
    assert check.category == "authentication"
    assert check.exit_code == auth.EXIT_AUTH
    assert seen


@pytest.mark.parametrize(
    ("message", "category", "exit_code"),
    [
        ("Could not resolve hostname ai.login: Name or service not known", "dns", auth.EXIT_TRANSPORT),
        ("Host key verification failed.", "host-key", auth.EXIT_CONFIG),
        ("Bad configuration option: controlpath", "configuration", auth.EXIT_CONFIG),
        ("ssh: connect to host ai.login port 22: Connection timed out", "transport", auth.EXIT_TRANSPORT),
    ],
)
def test_ssh_failure_classification(message, category, exit_code):
    result = auth.classify_ssh_failure(message)
    assert (result.category, result.exit_code) == (category, exit_code)


def test_clifton_process_detection_reads_only_executable_names(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0, "/usr/local/bin/clifton\n/usr/bin/ssh\n", "")

    monkeypatch.setattr(auth, "run_command", fake_run)
    assert auth.clifton_process_is_running() is True
    assert calls[0][0] == ["ps", "-x", "-o", "comm="]


def test_renew_uses_exact_command_and_stops_on_nonzero_without_status(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(auth, "resolve_route", lambda host: (valid_report().target, valid_report().jump))
    monkeypatch.setattr(auth, "clifton_process_is_running", lambda: False)

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 1)

    monkeypatch.setattr(auth, "run_command", fake_run)
    out = io.StringIO()
    code = auth.renew(lock_path=tmp_path / "auth.lock", out=out)

    assert code == auth.EXIT_RENEW_FAILED
    assert calls == [(auth.RENEW_COMMAND, {"capture_output": False})]
    assert "renewal-failed" in out.getvalue()


@pytest.mark.parametrize(
    ("report", "check_result", "expected"),
    [
        (
            lambda: _report_with_issue(),
            None,
            auth.EXIT_CERTIFICATE,
        ),
        (
            valid_report,
            auth.CheckResult("authentication", "rejected", auth.EXIT_AUTH),
            auth.EXIT_AUTH,
        ),
        (
            valid_report,
            auth.CheckResult("ok", "connected", auth.EXIT_OK),
            auth.EXIT_OK,
        ),
    ],
)
def test_renew_success_is_gated_by_revalidated_status_and_fresh_ssh(
    monkeypatch, tmp_path, report, check_result, expected
):
    calls = []
    monkeypatch.setattr(auth, "resolve_route", lambda host: (valid_report().target, valid_report().jump))
    monkeypatch.setattr(auth, "clifton_process_is_running", lambda: False)
    monkeypatch.setattr(auth, "collect_status", lambda host: report())

    def fake_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(auth, "run_command", fake_run)
    if check_result is None:
        monkeypatch.setattr(
            auth, "fresh_check", lambda status: pytest.fail("SSH must not run after failed revalidation")
        )
    else:
        monkeypatch.setattr(auth, "fresh_check", lambda status: check_result)

    assert auth.renew(lock_path=tmp_path / "auth.lock", out=io.StringIO()) == expected
    assert calls == [(auth.RENEW_COMMAND, {"capture_output": False})]


def test_renew_rejects_bad_route_before_launching_clifton(monkeypatch, tmp_path):
    monkeypatch.setattr(
        auth,
        "resolve_route",
        lambda host: (_ for _ in ()).throw(auth.ConfigError("unknown-alias", "wrong alias")),
    )
    monkeypatch.setattr(auth, "run_command", lambda *args, **kwargs: pytest.fail("clifton must not be launched"))

    out = io.StringIO()
    assert auth.renew(lock_path=tmp_path / "auth.lock", out=out) == auth.EXIT_CONFIG
    assert "unknown-alias" in out.getvalue()


def test_renewal_lock_reports_contention_without_removing_lock(monkeypatch, tmp_path):
    real_flock = auth.fcntl.flock

    def contended(descriptor, operation):
        if operation & auth.fcntl.LOCK_NB:
            raise BlockingIOError
        return real_flock(descriptor, operation)

    monkeypatch.setattr(auth.fcntl, "flock", contended)
    lock = tmp_path / "auth.lock"

    with pytest.raises(auth.RenewalError) as error:
        with auth.renewal_lock(lock):
            pass

    assert error.value.category == "renewal-in-progress"
    assert error.value.exit_code == auth.EXIT_BUSY
    assert lock.exists()
