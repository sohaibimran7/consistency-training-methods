# Offline paper compute and data-exposure audit

Python 3.10+ standard library only. No SSH, model loading, job submission,
generation, grading, network access or large raw data is part of this package.
The six scripts port the frozen September17/26 analyses without changing their
scientific calculations. This is a historical-results replay, not a general
training controller or a claim of current checkpoint availability.

## Inputs and safe verification

Supply an **artifact root** containing the dated subdirectories, not a repository
root. `input-manifest.json` lists every required external input and SHA256 plus
the archived output baselines. Preserve relative paths when retrieving the
evidence bundle. No automatic download or invented public archive URL is provided.
Raw snapshots remain external; do not commit them with this package.

From the repository root:

```sh
python3 experiments/paper_compute/verify.py --artifact-root /path/to/artifacts
```

Verification hashes all11 external inputs and8 archived baselines, links only
read-only inputs into a temporary tree, runs all six commands there, and compares
outputs. Seven outputs must be byte-identical; the data README differs only in
its two updated reproduction-command strings. The generated data `SHA256.json`
is checked against its actual directory contents, not against unrelated files
that happened to coexist in the historical artifact directory. The optional
`--report /path/to/receipt.json` writes a verification receipt. The bundled
`verification.json` records the successful owner replay, not a guarantee that
another installation has those inputs.

## Explicit production of derived outputs

These commands **overwrite their named derived files under the supplied artifact
root**. Use the isolated verifier above when the original evidence must remain
untouched. Each command requires `--artifact-root`; there is no worktree default.
Do not use Python `-O`, which would disable historical audit assertions.

```sh
python3 experiments/paper_compute/summarize.py --artifact-root /path/to/artifacts
python3 experiments/paper_compute/per_step.py --artifact-root /path/to/artifacts
python3 experiments/paper_compute/analyze.py --artifact-root /path/to/artifacts
python3 experiments/paper_compute/restricted_search.py --artifact-root /path/to/artifacts
python3 experiments/paper_compute/data_analyze.py --artifact-root /path/to/artifacts
python3 experiments/paper_compute/data_report.py --artifact-root /path/to/artifacts
```

| Command | Subdirectory under artifact root | Derived outputs |
|---|---|---|
| summarize | compute-audit-20260917 | summary.json |
| per_step | compute-audit-20260917 | per-step.json, PER_STEP.md |
| analyze | compute-matched-selection-20260926 | selection.json |
| restricted_search | compute-matched-selection-20260926 | restricted-search.json |
| data_analyze | data-matched-checkpoints-20260926 | analysis.json |
| data_report | data-matched-checkpoints-20260926 | CHECKPOINTS.md, README.md, SHA256.json |

Run in the displayed order. Data analysis also reads the two frozen pool-order
manifests listed in `input-manifest.json`. It fails if either manifest glob is
missing or ambiguous rather than choosing arbitrary input. Historical absolute
checkpoint paths within the input/output records are provenance identifiers;
the scripts do not access those remote checkpoint paths.

## Interpretation boundaries

- Allocated GPU-hours and recorded loop durations are **not measured FLOPs**.
  Retry inclusion differs between the audited BCT/OPCT and RMCT lineages.
- Restricted search replays all9504 retained BCT/OPCT/RMCT combinations and
  selects224/112/32 at35.409722/36.425556/35.711111 allocated GPU-hours. It does
  not select by evaluation performance or launch evaluation.
- Data analysis reproduces the maximum retained question-count budget of576
  QIDs, including Qwen288. This is **not the later step224 reuse recommendation**
  in `data-matched-checkpoints-20260926/REUSE_PLAN-20260927.md`. Keep that later
  discussion in the paper's synthesis; do not silently rewrite historical output.
- Equal question counts do not imply equal training QIDs or gradient exposure;
  RMCT uses a different pool/order. Historical Gemma results are thinking-off,
  not a corrected thinking-enabled replication. Global attempted groups and
  optimizer updates differ where groups were skipped.
- These scripts neither encode nor authorize today's evaluation token caps.
  Preserved historical settings are not new generation policy.

## Selective import

Import only this directory's `.py`, `.md`, and small manifest/receipt `.json`
files. Exclude `__pycache__`, external raw snapshots and generated large search
outputs. `source-sha256.json` identifies the exact source contribution files
(excluding itself). No shared runtime code was changed for this port.
