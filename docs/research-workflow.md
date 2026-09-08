# Research workflow

Start with an experiment ID in [the catalogue](../experiments/README.md), then
follow its protocol and source references. The catalogue separates scientific
status from job state and file availability. A successful job or rendered
figure alone does not make an experiment complete.

## Before a run

1. Define the question, comparison, inputs, split, seeds, generation and grading
   settings, analysis estimand, exclusions and completion criteria in the
   experiment's protocol. Give a changed scientific protocol a new ID and
   record its parent. A recovery attempt of the same protocol gets a new attempt
   record, not an overwritten history.
2. Use a committed branch or preserve the exact dirty source. Generic runner
   attempts and training manifests now retain portable source bundles. A Git
   SHA alone does not describe modified or untracked code.
3. Select the [supported environment](../environments/README.md). The Linux CPU
   lock is tested from a clean install. GPU profiles document distinct training
   and Figure 6 stacks; they are not complete reconstructed historical locks.
4. Keep input JSONL and its manifest together. Record exact model, dataset,
   tokenizer, adapter and judge revisions, not only human-readable names.
   Verify required bytes are reachable. Avoid mutable aliases for accepted
   inputs and checkpoints.
5. Preview the resolved commands with `scripts/run_experiment.py --dry-run`.
   Include explicit `inputs` and `outputs` on ordinary YAML commands where
   possible. Obtain the required approval for paid or remote experiments before
   execution, following [the project instructions](../CLAUDE.md).

## During and after a run

The [generic run recorder](experiments-run-records.md) retains one immutable
attempt per actual invocation and one command record per subprocess, including
parallel commands. It records argv, source, the parent environment, input/output
identities and lifecycle events. Missing terminal events mean incomplete.
Trainer manifests retain their own configuration and source identity. Bespoke
historical launchers still use their existing launch, custody and completion
contracts; this consolidation does not retroactively turn them into generic
runner attempts or infer a child environment from the parent.

Keep raw responses, scores, parser failures, log files and checkpoint sidecars.
Analysis must name its parents, preserve sample IDs and pairing, explain missing
values, and report the intended denominator. Re-render from retained analysis
data and chart recipes. Shared plotting changes layout mechanics; each adapter
still determines the scientific quantity being plotted.

Update the catalogue after checking the protocol's acceptance criteria. Record
the result's restricted scope and any excluded attempts. Do not classify a
diagnostic or partial screen as a completed full comparison. Use
`python scripts/experiment_catalog.py validate experiments/catalog.json --repository .`
to verify local references and hashes as well as schema and lineage. CI runs
that check, the environment contract and the offline suite.

## Retention and worktrees

Every retained result needs retrievable input, source, environment, checkpoint
and analysis identities. Follow the [recovery guide](consolidation/README.md)
and preserve data before retiring its only location. Hashes detect changes;
they cannot recover missing bytes. New run records and outputs are local until
they are deliberately copied to recovery storage.

Keep active worktrees for active work. Integrate useful changes through a
reviewable branch, preserve unique dirty source and ignored artifacts, then
archive the whole inactive directory and its Git registration. Check running
processes, remote deployments, task ownership and editable environment links
first. Keep a restoration record. Never clean up an active experiment's source
directory to make the main checkout look tidy.

## Boundaries that remain deliberate

- `ctm/` is generic library code. It must not import `ctm_data`, benchmark
  packages, `experiments` or `scripts`; architecture tests enforce this.
- Benchmark adapters own their statistical interpretation. MCQ pairing,
  conditional missing values and significance cannot be replaced by a generic
  average merely to remove similar-looking code.
- Versioned recovery plans remain when later protocols import them or their
  byte contracts are needed to explain an accepted result. The r5 continuation
  still depends on earlier convergence plans.
- Raw and translated vLLM adapters remain separate where parity validators
  require both forms. Tensor equivalence does not make their paths disposable.
- Hardware validation is specific to source, model, environment and topology.
  The retained four-GH200 phase-sharing preflight proves its stated synthetic
  case; it does not certify the consolidated branch for production PPO/GRPO,
  OPCT, multi-update resume or eight-GPU behavior. Fresh production validation
  remains an explicit gate before adopting this source in a campaign.

These conventions apply the supplied [Good Research Code Handbook](https://goodresearch.dev/)
to CTM's actual protocols and recovery constraints. The remaining historical
limitations are recorded alongside each experiment rather than hidden by the
new infrastructure.
