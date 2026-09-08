# Stage 1 IID diagnostic

This offline preparation freezes two paired-prompt populations from the exact
3,000-row Qwen3.5 Stage 1 source (`SHA-256 dfc10d55e51de48566488d107cdc90a0965ac5a73a80f3af68e605e40eff24ad`):

- `train_eval`: source rows 1–200, 100 LogiQA and 100 HellaSwag;
- `heldout_in_domain`: source rows 2,049–2,248, likewise 100 per dataset.

The manifest also records source rows 1–64 as the exact RMCT/RMCT-control
prefix. The other bias-aware methods consumed at least all 200 `train_eval`
rows. Preparation preserves the source JSON objects, refuses overwrite, and
makes no model or remote-service call:

```bash
python -m experiments.stage1_iid_diagnostic.prepare \
  --source artifacts/rmct-hle-dense-models-shared/data/distractor-argument-pairs.jsonl \
  --output-dir artifacts/stage1-iid-diagnostic
```

Use `experiments.stage1_iid_diagnostic.tasks:diagnostic_tasks` with
`scripts/run_evals.py`. Task arguments require `manifest`, `split`, and
`unbiased_log`. Each split returns clean LogiQA, clean HellaSwag, biased
LogiQA, and biased HellaSwag in that order. Generation deliberately sets
`include_bias_acknowledged=false`: Stage 1 logs are locally parsed first and
Luna grading is a separate reproducible pass.

## Posthoc Luna grading

### Raw-log provenance preflight

Before copying raw logs away from their generation host or invoking Luna, run
the CPU-only preflight against the frozen manifest and its two verified split
files. It rejects missing/duplicate native biased cells, a changed split file,
wrong question IDs, prompt style, source identity, paired-switch schema, or
(when supplied) checkpoint/decode configuration. Its immutable report records
the exact raw-log hashes that are eligible for grading.

```bash
python -m experiments.stage1_iid_diagnostic.raw_preflight \
  --raw-log-root logs/evals/CONDITION/iid-native \
  --manifest artifacts/stage1-iid-diagnostic/manifest.json \
  --split-file train_eval=artifacts/stage1-iid-diagnostic/train-eval-n200.jsonl \
  --split-file heldout_in_domain=artifacts/stage1-iid-diagnostic/heldout-in-domain-n200.jsonl \
  --condition CONDITION \
  --expected-base-model Qwen/Qwen3.5-9B \
  --expected-checkpoint /absolute/path/to/verified-vllm-compat-adapter \
  --output artifacts/stage1-iid-logs/CONDITION/raw-preflight.json
```

The script makes no model or API call. Copy this report, both frozen split
files, and only the listed raw logs into the local Luna staging root.

The posthoc grader is pinned to
`openrouter/openai/gpt-5.6-luna-20260709`, low reasoning effort, 256 output
tokens, and at most 500 aggregate connections. By default the runner uses five
fixed process shards, each processing its logs serially with 100 scorer
connections. It rejects every configuration where
`workers * connections_per_worker > 500`. Inputs are sorted and assigned by a
stable hash of condition/split/dataset, so newly completed logs cannot change
the shard recorded for already discovered cells. The runner selects only the newest successful
`stage1_iid_biased` retry in each condition/split/dataset cell and supplies
`mockllm/model` as Inspect's inert primary rescore model. The repository scorer
makes the only remote calls.

Raw generation logs must have this split-safe layout:

```text
RAW_ROOT/
  rmct/train_eval/*.eval
  rmct/heldout_in_domain/*.eval
  bct/train_eval/*.eval
  ...
```

Run a small paid capability smoke only when explicitly ready to use the API:

```bash
python -m experiments.stage1_iid_diagnostic.grade_luna \
  --raw-log-root artifacts/stage1-iid-logs \
  --output-root artifacts/stage1-iid-luna \
  --smoke --smoke-samples 2
```

The smoke grades two samples from one biased log and uses `-luna-smoke`
artifacts, so it cannot be mistaken for the full analysis input. The full pass
is:

```bash
python -m experiments.stage1_iid_diagnostic.grade_luna \
  --raw-log-root artifacts/stage1-iid-logs \
  --output-root artifacts/stage1-iid-luna \
  --workers 5 --connections-per-worker 100
```

The output and raw roots must be separate, non-nested directories. For every selected source log,
the runner creates a new `.eval`, a row-level `.jsonl` export of Luna verdicts,
and a `.provenance.json` sidecar under the same
`<condition>/<split>/` hierarchy. It never overwrites a raw log. Rerunning
resumes only when all three outputs exist, the derived EvalLog is successful,
and its source hash, worker/shard assignment, and grading configuration match
exactly. An incomplete or conflicting output set fails closed; archive it
before retrying.

## IID diagnostic report

After the full grading pass:

```bash
python -m experiments.stage1_iid_diagnostic.analyze \
  --graded-root artifacts/stage1-iid-luna \
  --manifest artifacts/stage1-iid-diagnostic/manifest.json \
  --output artifacts/stage1-iid-luna/analysis.json
```

Analysis is local-only and ignores smoke artifacts. It reports pooled and
per-dataset counts/rates for every condition and split:

- joint parsing successes and failures;
- `TBSR = P(biased answer = bias answer | clean answer != bias answer)`;
- away-from-bias rate conditional on the clean answer equalling the bias;
- total switch rate among jointly parsed pairs;
- generation and Luna-grader max-token cap hits;
- parsed/malformed Luna verdict counts, overall Luna YES rate, and Luna YES
  among toward-bias switches.

For `rmct/train_eval` and `rmct-control/train_eval`, the report additionally
emits the exact manifest-defined RMCT first-64 subset, pooled and by dataset.
The JSON report records hashes of its manifest, raw logs, and derived logs and
is idempotent: an identical existing report resumes, while a differing one is
never overwritten.

Render the final nine-condition report as pooled TBSR and Luna
bias-verbalization figures (each in PNG and SVG form) with:

```bash
python -m experiments.stage1_iid_diagnostic.plot \
  --analysis artifacts/stage1-iid-luna/analysis.json \
  --output-dir artifacts/stage1-iid-luna/figures
```

This plotting step is local-only. It requires all nine conditions, including
the untrained base model, and both diagnostic splits. It adapts the exact-count
report to the same chart-row boundary and publication renderer used by the main
Stage 1 figures: training-domain samples are the left group, held-out in-domain
samples are the right group, and method colors, control outlines, backgrounds,
legend order, and two-standard-error bars are shared. An identical rerun
resumes; the command refuses to overwrite any differing figure.
