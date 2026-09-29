# Historical MCQ preprocessing and statistical replay

This bundle contains the **actual code**, not just finished figures. Run with
Python 3.11, NumPy, Matplotlib, and inspect-ai. No external worktree code is
imported; the small plotting/parser dependency closure is vendored. Private
EvalLogs, saved grader receipts, sample tables, and the chart template are
explicit inputs. Nothing invokes a model, provider, SSH, or scheduler.

Validated runtime: Python3.11.12, NumPy2.4.6, Matplotlib3.11.1,
inspect-ai0.3.251. Exact numerical replication across other NumPy versions is
not asserted; use `--reference` to test it.

## Commands

Use a fresh output directory for every command. Existing outputs and overlap
with input directories are rejected. All writes occur inside that new output.

```sh
python replay/run.py reconcile \
  --legacy-artifacts /data/original-artifacts \
  --path-map /data/path-map.json \
  --output /output/reconciled-run \
  --reference /data/parser-fixed-20260923

python replay/run.py merge \
  --fixed /output/reconciled-run/reconciled \
  --recovery /data/recovered-publication-20260922 \
  --output /output/merged-run \
  --reference /data/parser-fixed-64k-20260925

python replay/run.py figures \
  --samples /output/merged-run/merged/samples.json \
  --template /data/chart-spec.json \
  --output /output/figures-run \
  --reference /data/parser-fixed-64k-20260925/figures-final

python replay/test_replay.py
```

`--reference` enables comparison, not input substitution. `--path-map` is an
optional JSON object mapping original absolute path prefixes to relocated
prefixes. Longest-prefix matching is used and original SHA checks remain.
`figures --check-only` checks inputs, isolated imports, parser and helper
definitions **without computing statistics**. `--only switch conditional
verbalisation accuracy` selects endpoints (all four are the default).

### Required private inputs

- Reconciliation: original artifacts subtrees
  `methods-complete-with-opct-20260916/towards_bias_switch-vs-base/manifest.json`,
  `methods-verbalisation-all-seven-20260916/complete.json`, and
  `paper-behavioural-plots-20260917/samples.json`; referenced clean and biased
  EvalLogs must resolve either directly or through the explicit path map.
- Merge: corrected `samples.json`; recovery directory containing
  `source-hashes.json`, `merge-audit.json`, `missing-bct.json`, original
  `build.py` (hashed, **not executed**), `grades/*.json`, and
  `inputs/*/raw/shard-*/*.eval`. Recovery source manifests are remapped by
  the complete relative path below `/inputs/`; basename collisions are not
  accepted. Exactly 552 recovery and 700 top-up outputs are required.
- Figures: seven-method merged samples and original nine-bias/bucket chart
  template. Optional `--sources` copies the upstream provenance ledger; its
  omission does not claim verification of raw logs. Sample/template hashes
  are always recorded. This is a saved-row analysis boundary, not a proof
  that every underlying generation was independently regenerated.

## Preserved statistics and bounded transformations

The endpoint scripts and numerical helpers are unmodified snapshots verified
against `source-lock.json`. The runner changes **only** the helper's hardcoded
repository path to an isolated vendored directory and narrows an unused
`checkpoint_publication` import to its original `contract.SEEN_BIASES`.
The historical raw-log `extract()` helper is not exposed as an entrypoint;
use `reconcile`, then `merge`, then `figures` instead.
`recipes/historical-regenerate.py` records the larger historical 39-figure
campaign for provenance; it is not an exposed command. This portable bundle
executes the four manuscript endpoint families listed by `figures --help`,
not every ancillary figure from that older campaign.

Seeds (including historically named `away-switch` tags), populations,
eligibility, missingness, method-specific denominators, 10,000 bootstrap
draws, 108,000 permutations, plus-one p-values and Holm families are unchanged.
There is no reduced-permutation smoke run disguised as publication output.
The full figure replay compares JSON objects, not rounded plotted values.
The raw merge compares every sample field except `biased_source` and
`clean_source`, which legitimately change with input relocation.

Historical captions are preserved, including the known obsolete RMCT50+50
caption in the historical accuracy spec. Do not treat that caption as the
actual denominator: rows contain the topped-up 100 QIDs/dataset. Corrected
presentation can be performed separately without altering these statistical
reproduction outputs.

## Historical parser recovery and validation

The current ac02 parser is **not** the September 23/25 parser. A first replay
using the current parser correctly failed the raw-merge comparison; that
failure receipt is retained in `validation/merge-current-parser-mismatch.json`.
The historical parser was recovered from the unchanged d6d6 baseline plus the
two exact September 22 patches recorded in session
`01a0c89e-24b1-7fa1-8d68-343b05b7bb9c`. Its SHA-256 is exactly
`5fe2e4e5295a20381f2073ce01b9b5dd25b5b972884e67aaaffb280976500b30`,
matching both historical provenance manifests. This is a historical replay,
not an endorsement of that older parser for new experiments.

- `validation/full-endpoints.json`: **all four endpoints EXACT JSON equality**
  against the saved September 25 results at the full permutation count, rerun
  with the final hash-pinned historical-parser bundle. The earlier successful
  statistical replay is retained as `full-endpoints-initial.json`.
- `validation/merge-historical-parser.json`: **exact sample equality except
  relocated source paths**, using the hash-matched historical parser.
- `validation/import-only.json`: successful cheap import/staging check; not a
  statistical test. The parser in this earlier receipt predates the historical
  parser recovery; later merge and contract checks cover the pinned version.
- `validation/reconciliation.json`: **EXACT JSON equality** for all seven
  methods after reparsing the original archived logs. The 144 verified source
  hashes are in `validation/reconciled-source-hashes.json`. Missing Base clean
  outputs remain missing; this does not assert a complete Base-clean reparse.
- Eight fast contract tests passed (including generator snapshot hashes and
  compilation of every bundled Python source).

Receipts retain original execution paths and source hashes for provenance;
none of those original code paths is a runtime dependency. Wrapper changes
are visible in `run.py`; copied snapshots are immutable under their lock.

## Generator implementations (archival; do not execute for replay)

`generators/` includes original selection, task factories, workers and Slurm
recipes for the targeted 64k retries and RMCT IID top-up, the attested runtime
`evaluate.py`, and the recovery merge/grading implementation. These historical
scripts intentionally retain original machine paths and may launch generation
or grading if executed. **They are provenance code, not portable commands or
part of the offline replay.** Checkpoints, shard-selection files, private logs
and grading receipts remain external inputs.

The canonical PR must also retain the generator implementation modules used
by these snapshots: `scripts/run_evals.py`, `ctm/evals/`, `ctm/backends/`,
`experiments/rmct_two_bias_eval/{checkpoint_publication,contract}.py`, and
`ctm_data/adapters/mcq_bias/luna_scorer_no_cap.py` plus its scorer/config
dependencies. Those are repository code, not external-worktree imports.
The attested runtime also imports
`infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.py` and
`ctm/evals/qwen35_vllm_scope.py`; preserve these in the canonical PR.
Generator snapshots are hashed separately in `generator-source-lock.json`.
No claims are made here that a new inference run reproduces the old outputs.

## Scientific limits unchanged

Selective 64k recovery (not uniform64k); original20k outputs otherwise remain.
Twelve BCT retries are missing (seven biased, five clean); nine parsed responses
lack saved verbalisation grades. Base clean labels are only partly verified.
Training-parser issues are not repaired by regrading these historical answers.
Intervals quantify question-level uncertainty, not training-seed uncertainty.
