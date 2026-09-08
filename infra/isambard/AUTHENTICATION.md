# Isambard authentication and renewal

Run these commands on the laptop, before starting remote work. The helper uses
Python 3.9+ and the existing OpenSSH and Clifton installations; it does not need
the research environment or provider API keys.

```bash
python3 infra/isambard/auth.py status a5v.aip2.isambard
python3 infra/isambard/auth.py check a5v.aip2.isambard
```

On the shared laptop, `isambard-auth status` and `isambard-auth check` work from
any directory. `a5v.aip2.isambard` is the default. Phase 1 uses
`a5v.aip1.isambard`. Use the project alias supplied by Clifton rather than the
gateway hostname or a guessed `isambard` alias.

`status` resolves the effective OpenSSH configuration for the destination and
jump host, checks their certificate time windows in UTC, checks the SSH user
against the certificate principals, and compares public-key fingerprints. It
does not read private keys or contact Isambard. `check` then executes only
`true` through a fresh, bounded, non-interactive SSH connection. Both hops use
batch mode, normal host-key verification, and no shared control connection.
Neither command submits, resumes, or cancels remote jobs.

## Expected expiry

Isambard certificates last **12 hours**. The
[official login guide](https://docs.isambard.ac.uk/user-documentation/guides/login/)
requires rerunning `clifton auth` after expiry and prohibits maintaining
persistent SSH connections to bypass expiry, including for AI agents. Existing
connections are not proof that a new connection will authenticate. Do not
create a keepalive tunnel, renewal daemon, or scheduled authentication loop to
work around this policy.

The durable workflow is to check before remote work, renew on demand, and
verify a fresh connection. Certificate renewal cannot guarantee permanent
access. The installed Clifton 0.3.0 device flow requires identity-provider
authorization; its browser/QR/config flags do not remove that requirement.
See the [Clifton usage documentation](https://github.com/isambard-sc/clifton/blob/0.3.0/docs/use.rst).

## Work that must continue beyond certificate expiry

The helper alone does not provide continuous unattended access. A search of the
public Isambard documentation on 7 September 2026 found no documented service
account or non-browser refresh-token renewal option. Browser-driven renewal
using existing SSO was subsequently verified below. An alternative machine
authentication method would need confirmation from Isambard support before
relying on it.
The [AI agent guide](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/)
also prohibits sharing passwords/private keys with agents and bypassing access
controls.

[Slurm batch jobs](https://docs.isambard.ac.uk/user-documentation/guides/slurm/#batch-jobs-sbatch)
run whether or not the user remains logged in. Stages can be submitted with
[job dependencies](https://docs.isambard.ac.uk/user-documentation/guides/slurm/#job-dependencies)
so already-authorized work continues without another laptop connection.
Certificate expiry still prevents new remote commands until renewal. Planning
such a job pipeline is separate from repairing authentication; this task did
not submit or modify any jobs.

## One renewal at a time

When the certificate is expired and browser completion is available:

```bash
python3 infra/isambard/auth.py renew a5v.aip2.isambard
# Or, from any directory on the shared laptop:
isambard-auth renew a5v.aip2.isambard
```

The helper serializes renewals across checkouts with a per-user lock and checks
for an existing Clifton process. Direct invocations outside the helper cannot
participate in that lock, so all tasks should use the helper. It runs:

```bash
clifton auth --open-browser false --show-qr false --write-config true
```

Keep that process running while completing its displayed device URL in Chrome.
Do not start a second attempt or reuse a URL from an older attempt. Do not
record device codes, token-bearing URLs, passwords, or private keys in files,
notes, or logs. The helper leaves the device-flow output attached to the
terminal and does not save it. After Clifton exits successfully, the helper
rechecks local metadata and verifies fresh SSH before reporting success.

In a sandboxed task, renewal needs authorized process inspection, network
access, and writes to the per-user lock, Clifton cache, and SSH configuration.
Use the terminal tool's normal approval mechanism if these are restricted.
Filesystem/network approval does not relax the browser URL restriction below.

The authorized university path on this laptop is:

1. **University Login (MyAccessID)**, then **Lancaster University**.
2. Use university username **`imrans1`**.
3. Click the password field and choose Chrome's saved **Password for imrans1**.
4. Submit the university login and approve the Clifton SSH-signing grant.

Use the saved credential only through Chrome autofill. Never inspect, reveal,
read, copy, export, log, or store its value. The certificate key ID may contain
`s.imran1@lancaster.ac.uk`; that is certificate metadata, not the login username.
The SSH project account is separate and comes from the generated SSH config.

## Current automated-browser limitation

On 7 September 2026, the prior authorized device flow successfully reached
Lancaster WebLogin, but Computer Use explicitly stopped the session because
that browser URL is prohibited, even when the user navigates there themselves.
This is a platform access restriction. It is not evidence of a wrong password,
missing Chrome extension, or a broken Clifton SSH route.

Do not retry an explicitly denied UI through another tool, browser route, or
native automation mechanism. No supported setting that relaxes the recorded
Lancaster restriction was identified. The user's presence does not lift an
explicit tool restriction. However, this denial is not evidence that every
normal device flow is blocked: a fresh university session subsequently allowed
the automated SSO renewal below. The saved-password challenge after SSO expiry
remains unverified. Never request a password or start repeated device flows
after an explicit denial.

## Verified automatic renewal using existing SSO

On 8 September 2026, after the user had signed in manually, they requested one
supervised recheck using Computer Use. The agent navigated Chrome's normal
address bar to the fresh Clifton device URL. Existing SSO went directly to
**Grant Access to clifton (SSH key signing)**. The agent clicked **Yes** and
observed **Device Login Successful**, with no user interaction required.
Clifton exited successfully and the helper's fresh two-hop SSH check passed.

The certificate from that test was valid from `2026-09-08T12:21:12Z` through
`2026-09-09T00:21:12Z`. These dates are a historical verification record; always
inspect the current certificate. No Lancaster password form or Chrome password
picker appeared, so this test establishes automatic renewal with usable SSO,
not saved-password control after the browser session expires.

For subsequent on-demand renewal:

1. Run the shared preflight. If access is healthy, continue the authorized work
   without renewing. A valid-certificate recheck should only be performed when
   explicitly requested, as this supervised test was.
2. When renewal is needed and the normal browser flow can proceed, keep one
   `isambard-auth renew` process live and open its fresh initial device URL in
   Chrome through the permitted interface. Let the existing SSO session work;
   do not force logout, clear sessions, or create a password challenge.
3. If the normal flow reaches the Clifton signing grant, approve it as already
   authorized. If a tool explicitly denies any page, stop there and report the
   exact limitation. Do not switch tools/routes or reset a stopped session to
   evade that denial. Saved-password use remains subject to the credential
   rules above and the tool's actual permission to operate that page.
4. Require Clifton success, current certificate validation, and the helper's
   fresh SSH check before reporting success. Then resume only already-authorized
   remote work. Preserve running jobs if renewal fails.

This successful SSO flow should be used when available without requiring a
manual handoff. It does not guarantee future access after SSO expires or when
the university introduces an additional challenge.

## Verified manual fallback

On 8 September 2026, the user completed a browser handoff successfully. They
subsequently clarified that their desired outcome is unattended authentication,
so this fallback does not satisfy the full requirement. It must not be recorded
as a replacement for unattended access. The credential handling restrictions
remain in force.

1. Run `isambard-auth status` before remote work. If the certificate is usable,
   run `isambard-auth check` and proceed when it succeeds. Renew only when
   needed; do not repeat authentication as a demonstration while access works.
2. If renewal is needed and the user explicitly offers to complete the browser
   steps, establish that they are ready before creating a short-lived device
   flow. Otherwise, use the verified SSO flow when available; if the actual
   authentication step is blocked, report that limitation and preserve existing
   remote jobs.
3. Start one `isambard-auth renew` process and keep its terminal session live.
   Open its fresh initial device verification URL in Chrome and hand over to
   the user. Do not reuse or save an earlier device URL/code.
4. The user completes university sign-in and the Clifton access grant. They
   can follow the university route above. The agent does not inspect or control
   the blocked Lancaster page, read credentials, or switch tools to work around
   the restriction.
5. Watch the running helper for Clifton's successful completion, certificate
   validation, and the fresh two-hop SSH check. Report the result and expiry
   time. A browser success screen alone does not establish working SSH.
6. Resume only the already-authorized remote work. Authentication is not
   authorization to start or restart a training/evaluation run.

This procedure was verified on 8 September: the user completed the browser
authorization, Clifton renewed project `a5v`, and fresh SSH to
`a5v.aip2.isambard` succeeded. That certificate was valid from
`2026-09-08T12:06:26Z` through `2026-09-09T00:06:26Z`; this is a historical
verification record, not a substitute for checking current certificate status.

## Distinguish failures before changing anything

| Finding | Next step |
| --- | --- |
| Expired certificate | Renew once when browser completion is available, then verify fresh SSH. |
| Certificate not yet valid | Check the laptop clock and certificate dates before renewing. |
| Unknown alias, missing certificate, wrong principal, or public-key mismatch | Inspect effective `ssh -G` output and Clifton-generated paths. Avoid repeated authentication attempts and do not replace the private key. |
| Valid local certificate, SSH public-key rejection | Check both hops and project access. Local metadata alone cannot prove server authorization. |
| DNS failure, connection timeout, or routing error | Diagnose network/service availability; renewing a valid certificate does not repair transport. |
| Host-key verification failure | Verify the expected host key through an authoritative source. Do not disable host-key checking or erase known hosts. |
| Renewal already running | Coordinate with its owner and use its live flow. Do not kill unrelated processes or remote sessions. |
| Clifton exits successfully but SSH fails | Treat access as unverified and diagnose the reported failure; do not claim success. |

Exit codes are `0` for success, `2` for configuration/local metadata problems,
`3` for unusable certificates, `4` for SSH authentication rejection, `5` for
connection failures, `6` for renewal failure, and `7` for a renewal already in
progress or a process check that could not verify it was safe to start one.
A successful `status` is only a local preflight; successful `check`
also verifies access. Check the printed category before deciding to renew.

Clifton manages `~/.ssh/config_clifton`, included by `~/.ssh/config`. Inspect
effective values with `ssh -G a5v.aip2.isambard` and
`ssh -G sohaib.a5v@jump.a5v.aip2.isambard`. Avoid hand-editing the generated file;
successful authorized renewal with `--write-config true` writes its configuration.

The 7 September laptop diagnosis found matching certificate/public-key
fingerprints and the expected principal `sohaib.a5v` on both Phase 1 and Phase 2.
Both certificates expired on **25 August 2026 at 22:29:38 UTC**. The configured
Phase 2 route reached `ai.login.isambard.ac.uk` and a fresh SSH `true` failed
with public-key rejection. No live Clifton or SSH processes were present at
the time of inspection. No key replacement or SSH configuration rewrite was
indicated by these findings.
