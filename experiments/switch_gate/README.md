# Switch-gate input preflight

This directory contains local-only preparation and Inspect task factories for
the switch-gate experiment. The preparation step filters already-frozen
matched prompt pairs. It does not generate arguments, evaluate a model, import
a training backend, or make a Tinker call.

The complete frozen design, exact paid commands, stopping rule, and cost
envelope are in [`preregistration.md`](preregistration.md). Machine-readable
parameters and provenance are in [`protocol.yaml`](protocol.yaml),
[`audit.json`](audit.json), and [`cost.json`](cost.json). No paid command in
the preregistration may run until the user approves it after seeing the
resolved commands and parameters.

## Freeze the screen and confirmation inputs

Run from the repository root:

```bash
uv run python -m experiments.switch_gate.prepare \
  --source-pairs artifacts/switch-gate/source/distractor-argument-pairs.jsonl \
  --output-dir artifacts/switch-gate/prepared \
  --manifest-output artifacts/switch-gate/split-manifest.json
```

Only the first 2,048 source rows are authoritative. Preparation checks that
they have unique question IDs and contain exactly 472 LogiQA and 1,576
HellaSwag matched pairs. With seed `20260729`, SHA-256 ordering within each
dataset selects a 23/77 screen split. The remaining 449/1,499 rows form nested
confirmation files at n = 600, 800, 1,000, 1,200, 1,600, and 1,948.

The manifest records the source hash, ordered IDs, stratum counts, and hash of
every output. Existing outputs are never overwritten; move an obsolete run to
an `_archive/` directory before preparing another one.

## Inspect task factories

`experiments.switch_gate.tasks:training_tasks` reads a manifest plus a split
name (`screen` or `confirmation-n<N>`). It returns four tasks in safe order:
the LogiQA and HellaSwag unbiased baselines first, followed by their matched
wrong-argument tasks. Separate per-source tasks keep each sample's
`source_dataset` equal to the task header's `dataset` argument, so the upstream
`switch_scorer` can match baseline logs by model, dataset, and prompt style.

For example:

```bash
uv run python scripts/run_evals.py \
  --task-factory experiments.switch_gate.tasks:training_tasks \
  --task-args '{"manifest":"artifacts/switch-gate/split-manifest.json","split":"screen","unbiased_log":"logs/switch-gate/screen"}' \
  --model <inspect-model> \
  --log-dir logs/switch-gate/screen \
  -y
```

Use the same log directory for `unbiased_log` and `--log-dir`. The decorated
unbiased tasks run first; each biased task carries the switch scorer, which
waits for the matching completed baseline if necessary. Do not use Inspect's
`--limit`: sample counts are fixed by the prepared files and manifest.

`experiments.switch_gate.tasks:hle_tasks` consumes the frozen HLE files
directly. It returns the unbiased HLE task first and then one task per mapping
entry:

```json
{
  "unbiased_file": "artifacts/switch-gate/source/hle-eval/hle-text-mc_unbiased_none_n100_seed42_ids-1dc073edc4.jsonl",
  "bias_files": {
    "distractor_fact": "artifacts/switch-gate/source/hle-eval/hle-text-mc_distractor_fact_none_n100_seed42_ids-1dc073edc4.jsonl",
    "post_hoc": "artifacts/switch-gate/source/hle-eval/hle-text-mc_post_hoc_none_n100_seed42_ids-1dc073edc4.jsonl",
    "spurious_few_shot_squares": "artifacts/switch-gate/source/hle-eval/hle-text-mc_spurious_few_shot_squares_none_n100_seed42_ids-1dc073edc4.jsonl",
    "suggested_answer": "artifacts/switch-gate/source/hle-eval/hle-text-mc_suggested_answer_none_n100_seed42_ids-1dc073edc4.jsonl",
    "wrong_argument": "artifacts/switch-gate/source/hle-eval/hle-text-mc_wrong_argument_none_n100_seed42_args-vllm-google-gemma-4-31b-it_ids-1dc073edc4.jsonl",
    "wrong_few_shot": "artifacts/switch-gate/source/hle-eval/hle-text-mc_wrong_few_shot_none_n100_seed42_ids-1dc073edc4.jsonl"
  },
  "unbiased_log": "logs/switch-gate/hle"
}
```

Pass that JSON as the task arguments and use the same directory as the eval
log directory. Both factories explicitly omit the `bias_acknowledged` model
grader; they use only the ordinary MCQ scorers and the paired switch scorer.

## Analyze completed logs

The local-only analyzer reads completed Inspect logs and never runs a model or
grader. It validates exact frozen IDs, filters provider-qualified log model
names to the three preregistered candidates, and refuses output overwrite.

```bash
uv run python -m experiments.switch_gate.analyze screen \
  --run Qwen/Qwen3.5-4B=logs/switch-gate/20260729/screen \
  --run Qwen/Qwen3.5-9B=logs/switch-gate/20260729/screen \
  --run Qwen/Qwen3-8B=logs/switch-gate/20260729/screen \
  --expected-split artifacts/switch-gate/prepared/screen.jsonl \
  --output artifacts/switch-gate/screen-analysis.json
```

The exact confirmatory command is frozen in the preregistration. It applies
the fixed six-cell Holm family and the 50,000-replicate stratified/clustered
bootstrap, retaining absent cells as non-rejections.
