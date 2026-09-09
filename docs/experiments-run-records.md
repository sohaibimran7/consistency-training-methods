# Experiment run records

`scripts/run_experiment.py` creates one immutable attempt after an approved
invocation has saved its resolved plan and before it launches a child command.
Dry runs and a declined confirmation return before any attempt, source bundle,
or command record is written.

For an ordinary experiment, records live at
`logs/experiments/<experiment>/attempts/<attempt-uuid>/`. Targeted runs use
`logs/experiments/<experiment>/targets/<target>/attempts/<attempt-uuid>/`.
The attempt descriptor is `attempt.json`; its source snapshot points to the
local shared content-addressed store at `logs/experiments/source-bundles/`.
That bundle is retained locally and is not dependent on an external tracking
service.

An attempt descriptor records the full runner argv, the exact bytes identity
of the authored YAML and resolved selected plan, a source snapshot, and the
runner's Python/platform/distribution inventory. It also retains those exact
YAML and resolved-plan bytes as immutable regular files under the attempt's
`inputs/` directory. The authored YAML is stable-read without following a
symlink before it is retained. The source root comes from
`ctm.provenance.default_source_root()`, so it does not depend on the shell's
current directory. The runtime inventory describes the parent runner process
only. A child process may have a different environment; the record only states
the CUDA-visible-device placement explicitly assigned by the runner, or notes
that it inherited ambient CUDA visibility.

Each launched command has a separate UUID directory under `commands/`:

- `start.json` contains the exact child argv, CUDA placement, and input
  references.
- `end.json` means the process returned zero.
- `error.json` records a nonzero return or another execution error.
- `interrupted.json` records a `KeyboardInterrupt` observed by the runner.

A command or attempt with no terminal event is `incomplete`, never successful.
This is the expected state if the runner itself is killed before it can write a
terminal event. Attempts and command records are never overwritten, so retrying
an experiment creates a distinct UUID record without changing prior evidence.

Command YAML may declare local references without changing child argv:

```yaml
analysis:
  - name: summarize
    command: [python, summarize.py]
    args:
      data_manifest: data/eval.manifest.json
    inputs: [data/eval.manifest.json]
    outputs: [artifacts/summary.json]
```

`inputs` and `outputs` accept one path or a list of paths. They are recorded as
declared coverage. The records layer also recognizes path-like values for
common data, manifest, checkpoint, source, input, attestation, and output
flags; those are marked as heuristic inferred coverage and do not claim a
complete argv inventory. Local regular files and directories are streamed while hashing and
checked for changes; symlinks, missing paths, special files, unstable paths,
and external URIs remain explicit unverified entries. URI records retain only
their scheme and a digest of a credential-free form, never a credential URL.
An announced `CTM_FINAL_CHECKPOINT` is retained as checkpoint lineage along
with any declared outputs.

`ctm.experiments.records` provides `create_attempt_record`,
`start_command_record`, `complete_command_record`, and
`complete_attempt_record` for other local launchers. `verify_attempt_record`
checks the retained source bundle, parent runtime inventory, and the retained
attempt-local YAML and resolved-plan copies by default. With
`allowed_path_roots`, it additionally rechecks the original source YAML and
shared resolved-plan path when one was recorded. `verify_command_record` can additionally rehash local
references, but callers must pass `allowed_path_roots`; verification refuses to read a recorded path
outside those roots. Failed immutable writes preserve their `.partial` bytes
under a local `_archive/failed-experiment-record-writes/` directory instead of
deleting them.
