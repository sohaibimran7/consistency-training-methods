# Shared Isambard job controller

`jobs.py` is the canonical source for the shared-laptop controller.  The
installed `isambard-jobs` command is an independent copy, so it remains usable
when a worktree is removed.  Before installation, substitute
`python3 infra/isambard/jobs.py` for `isambard-jobs` below.

The controller is an admission and recovery layer around Slurm.  It owns local
queue state, output-path claims, and short fresh SSH operations; it does not
hold a remote shell, renew a certificate, or run an agent on a login node.

## Scope and limits

State defaults to `~/.local/share/ctm-isambard/jobs`, outside every worktree.
It is namespaced by the resolved Isambard hostname, port, and Unix user, so
requests for different accounts or clusters do not share a queue.  Use the
default path for normal work; a different `--state-dir` is only for isolated
tests.

This controller coordinates participating agents on this laptop.  It cannot
reserve capacity against another laptop or a manual `sbatch`/`srun` submitted
after its snapshot.  Before it admits an interactive request, it inspects the
same user's live jobs and treats any existing `interactive`-reservation job as
occupying the slot. The collector includes hidden partitions (`squeue --all`):
on this cluster a default queue listing can omit a running interactive job.

For `a5v.aip2.isambard`, apply these controller limits:

- `interactive` means a bounded batch job submitted with
  `--reservation=interactive`, not a detached PTY or persistent SSH session.
  There may be **one submitted job total** for the user: pending **or**
  running.  Requests are FIFO while that slot is held.
- The interactive reservation permits at most 4 nodes, 16 GPUs, and 8 hours.
  Use it only for short diagnostics, debugging, and exploratory work.  Do not
  use it for training campaigns, high-throughput work, or automated chains.
- `batch` requests go to `workq`.  Main training belongs there and is capped by
  the controller at 24 hours per job.

The controller protects scheduler health by allowing only one fresh scheduler
snapshot per scope every 60 seconds.  `status` is cached and never contacts
Isambard.  A runner makes no per-agent polling loop.

## Prepare, review, then enqueue

Use an adapter profile to construct a manifest rather than hand-writing a
Slurm command.  A typical lifecycle is:

```bash
isambard-jobs prepare PROFILE \
  --id REQUEST_ID --owner OWNER \
  --checkout /absolute/local/consistency-training-methods \
  --remote-dir /absolute/isambard/checkout \
  --output-root /absolute/isambard/output-root \
  --mode interactive-or-batch --minutes MINUTES \
  --env NAME=NON_SECRET_VALUE \
  --output /absolute/local/REQUEST_ID.json

# Review the generated local JSON manifest before continuing.  Regenerate it,
# with a new request id, if any code/configuration/resource/output decision changes.

isambard-jobs enqueue /absolute/local/REQUEST_ID.json --start
```

`prepare` creates the manifest with exclusive local creation, so it never
overwrites a previous review artifact.  `enqueue` validates the manifest,
resolves its remote paths, records its output claim, and—when `--start` is
present—starts the bounded local collector.

A manifest captures the launch script and declared non-secret configuration;
it does **not** freeze the scientific repository, data, checkpoint, model, or
runtime.  Before enqueueing, ensure the remote code/configuration and inputs
are frozen at the intended revision and that the pinned runtime is compatible
with the GH200/AArch64 allocation.  Do not enqueue from a mutable checkout
simply because its wrapper was captured locally.

Only non-secret `--env` values belong in a request.  Credential-like names are
rejected.  The local state and runner log store controller metadata, not job
stdout/stderr or credential values.

## Running and observing the queue

```bash
isambard-jobs status
isambard-jobs tick
isambard-jobs run --duration 43200
isambard-jobs start
```

- `status` reads only the cached durable state.
- `tick` reconciles once and can admit at most one batch and one interactive
  candidate after a fresh snapshot.  If another snapshot was attempted less
  than 60 seconds ago, it reports throttling instead of opening SSH.
- `run` is the foreground collector.  Its duration must be from 1 to 43,200
  seconds; 43,200 seconds is 12 hours.
- `start` launches one detached **local** collector with that same 12-hour
  bound, reports its PID, and writes `runner.log` under the state directory.

Every remote action uses a new bounded SSH process with connection reuse
disabled.  If authentication or transport fails, the collector pauses instead
of retrying, opening a browser, or extending an SSH session.  Restore access
with the normal authentication workflow, then start a new collector:

```bash
isambard-auth status
isambard-auth check
isambard-jobs start
```

An uncertain submission stays `unknown`: its output claim and interactive
capacity remain blocked.  Never enqueue a replacement or retry automatically
after a lost submission response.  Investigate it with the owner first; do not
edit controller state files to force progress.

After investigation establishes that an unknown request **never submitted**,
its owner can explicitly attest that finding with
`resolve-unsubmitted REQUEST_ID --owner OWNER --note 'evidence checked' --confirmed-never-submitted`.
This records the attestation and releases the request. It is unavailable when
a job ID is known. An empty live queue alone is not evidence that submission
never happened; accounting may be delayed. Agents must not invoke this as an
automatic recovery step.

## Current adapter profiles

`job_adapters.py` validates profile-specific inputs and captures the maintained
wrapper body at prepare time.  Profiles that depend on a wrapper fail closed if
that exact wrapper is absent from the supplied local checkout.

| Profile | Allowed mode | Fixed allocation | Intended use |
| --- | --- | --- | --- |
| `rmct_r5_segment` | batch | 1 node, 4 GPUs, 720 min, 204800 MiB, 16 CPUs/GPU | RMCT R5 continuation. Requires `SCRATCHDIR` and `CTM_RMCT_SEGMENT_INDEX` (11–31); output root must equal remote checkout. |
| `rmct_r5_interactive_gpu_diagnostic` | interactive | 1 node, 4 GPUs, 1–30 min, 204800 MiB, 64 CPUs/task | Controller-owned hardware-only probe; it does not import a model or train. |
| `gemma_main_16gpu` | batch | 4 nodes, 16 GPUs, 720 min, 409600 MiB, 16 CPUs/GPU, 4 GPUs/node | Main Gemma campaign. |
| `gemma_smoke` | interactive or batch | 1 node, 1 GPU, 120 min, 98304 MiB, 16 CPUs/GPU, 1 GPU/node | Bounded Gemma smoke run. |
| `gemma_eos_debug` | interactive or batch | 1 node, 1 GPU, 30 min, 98304 MiB, 16 CPUs/GPU, 1 GPU/node | Bounded Gemma EOS diagnostic. |

The Gemma profiles require a non-secret snapshot path; the EOS diagnostic also
requires its manifest path.  The adapter rejects unrecognised profile
configuration rather than silently passing it through.  Do not add
`--exclusive` to a one-GPU top-level request: it reserves a whole GH200 node.
Do not pass `--qos=interactive_qos`; reservation selection is handled by the
controller.

## Existing jobs and output ownership

Existing campaigns remain where they are.  Before queueing controller-managed
work that could overlap their outputs, register each legacy chain for
observation only:

```bash
isambard-jobs protect \
  --id LEGACY_ID --owner OWNER \
  --job-id SLURM_JOB_ID [--job-id ANOTHER_JOB_ID] \
  --output-root /absolute/isambard/output-root \
  --remote-dir /absolute/isambard/checkout \
  --mode batch
```

`protect` does not alter, cancel, move, or resubmit the legacy jobs.  It keeps
their output roots unavailable until every registered Slurm allocation has
terminal scheduler evidence.  Protected records cannot be cancelled or
withdrawn through this controller.

For controller-managed work, only the matching owner may change a request:

```bash
isambard-jobs withdraw REQUEST_ID --owner OWNER  # only while still queued
isambard-jobs cancel REQUEST_ID --owner OWNER    # only after a confirmed job id
```

Cancellation is always explicit.  It is recorded before `scancel`, then the
controller waits for terminal scheduler evidence; it never cancels a legacy
protected job or retries an uncertain cancellation.

## Interfaces

`job_controller.py` owns durable FIFO admission, output-root conflict checks,
owner checks, and fail-closed reconciliation.  `job_transport.py` owns the
fresh SSH boundary, remote path canonicalisation, Slurm snapshot/submit/cancel
calls, and rejects scheduler directives embedded in captured scripts.
`job_adapters.py` produces the profile-specific immutable request used by
`jobs.py`; it does not submit or inspect remote state.

The source workflow is intentionally narrow: all resource fields are declared
outside the captured shell script, remote output roots are normalised absolute
paths, and each successful submission receives a controller job-name token.
This makes a lost acknowledgement recoverable by evidence while keeping an
ambiguous outcome blocked.

## Installation verification, 8 September 2026

The installed controller passed fresh SSH and real queue/accounting reads.
The live check identified and fixed compatibility with the login node's older
system Python and the hidden interactive partition. Existing RMCT chain/smoke
and Gemma diagnostic jobs were registered for observation without changing
their scheduler state.

A reviewed two-minute, one-GPU hardware pilot was queued locally. Admission
correctly kept it waiting behind the existing interactive smoke job. The
unsubmitted pilot was then withdrawn locally; no GPU allocation or scheduler
cancellation was made by this validation. Successful real GPU submission through
the controller remains to be exercised when capacity is available. Submission
commands, crash recovery and concurrent requests were tested with a simulated
scheduler locally.

## Primary references

- [Isambard-AI Phase 2 job scheduling and reservation limits](https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/)
- [Isambard Slurm guide, including scheduler query guidance](https://docs.isambard.ac.uk/user-documentation/guides/slurm/)
- [Isambard guidance for AI agents](https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/)
- [Isambard login-node and persistent-session policy](https://docs.isambard.ac.uk/user-documentation/guides/login/)
