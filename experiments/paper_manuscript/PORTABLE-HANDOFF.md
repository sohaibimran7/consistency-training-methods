# Portable manuscript figure assembly

For selective integration into the ONE canonical reproducibility PR owned by task `01a0e8e0-24a5-7a40-83cc-240005b29ce7`. This task has not edited that checkout or created a second PR.

## Files to integrate

- `build_figures.py`: explicit input-root/output CLI, complete hash preflight, two 2x2 panels, byte-preserving copies and output inventory.
- `figure-sources.json`: 14 pinned source files, including the shared chart specification; 13 result entries (11 figure entries and two manually transcribed table sources).
- `test_build_figures.py`: seven offline contract tests using synthetic fixtures.
- `upstream-command-map.json`: recipe source hashes and source-inspected aggregation commands for all 13 entries, suitable for mapping into `docs/paper-command-map.json`.
- This handoff document. Keep the manifest next to the script or pass `--manifest` explicitly.

Runtime: Python >=3.10 and Matplotlib >=3.6, with its normal dependencies. Tested here using Python 3.11.12 and Matplotlib 3.11.1. Check-only mode does not import Matplotlib. This builder needs no CTM package, Inspect, network, model or grader. Upstream aggregation recipes have additional dependencies recorded separately. Matplotlib/font/platform differences can change rendered bytes; output hashes record the actual artifacts rather than claim cross-version byte identity. PDF creation timestamps are suppressed.

## Run from any checkout

The roots below are examples of staged artifacts, not embedded machine paths. Preserve each root's relative structure from `figure-sources.json`.

```sh
python /path/to/build_figures.py \
  --corrected-figures-root /data/parser-fixed-64k-20260925/figures-final \
  --monitor-root /data/parser-fixed-20260923/luna-overview \
  --ctm-artifacts-root /data/ctm-artifacts \
  --organism-artifacts-root /data/organism-artifacts \
  --output-dir /output/new-manuscript-assembly \
  --check-only
```

Remove `--check-only` to render/copy. The output root receives `figures/provisional/` (13 PDF/PNG files) and `figure-inventory.json`. The two table source entries are explicitly **source-only**, with empty generated-output lists; the builder does not write or validate the numerical transcription in LaTeX. Existing listed outputs in the chosen output root are overwritten; use a new directory to preserve an earlier assembly. Inputs and source manifest cannot be output targets, and relative-path traversal/symlink escapes are rejected.

All source hashes and main-panel rows are validated before the output directory is created. A mismatch, missing file, missing/duplicate cell, undefined rate, invalid denominator, or output collision fails preflight. Validation is fail-closed: do not silently refresh hashes to accept different research results. Review a changed source and its provenance, then explicitly revise the lock manifest. Verified source bytes are held in memory and used directly for rendering/copying.

The generated inventory records root-relative inputs, hashes, output hashes, entry operation type, method warnings, builder/manifest hashes and runtime versions. It records zero model/grader calls and that no upstream aggregation occurred. It intentionally does not disclose machine-specific absolute input roots.

## What is reproduced

The main panels reuse saved point estimates, bootstrap intervals, significance markers and denominators. They preserve rows = seen/held-out datasets, columns = seen/held-out biases, the existing palette and method order with RMCT last. The visible warning now explicitly includes flawed RMCT training. Expanded appendix assets are byte-preserving copies, retaining the publication pipeline layout. Their manuscript provisional captions remain mandatory: the copies alone are not publication-cleared figures. No claim of data-matched training is made.

This is **figure assembly**, not end-to-end reproduction of training, inference, parsing, grading or statistical estimation. `upstream-command-map.json` separately identifies inspected aggregation recipes and their dependencies. None of its upstream commands was executed for this handoff.

## Upstream findings that the command map must preserve

1. MCQ main and expanded figures: the live historical drivers are `tmp/merge_recovery_into_parser_fixed.py` and `tmp/regenerate_64k_figures.py` in the parser checkout. Archived copies under `artifacts/.../reproduction/` compute the wrong repository root if executed there unchanged. The direct staged build recipes require the sibling helper/sample/template tree and CTM imports; preserve `CTM_CONDITIONAL_PERMUTATIONS=108000` for conditional verbalisation.
2. Monitoring: use the original `tmp/regenerate_monitor_figures.py` wrapper. It injects `corrected()` into the recipe execution scope; `luna-overview/recipe.py` alone is not executable. The saved score roots remain machine-specific and need staging/refactoring before a portable upstream run. Reused scores do not cover newly recovered/top-up responses.
3. AITA: `plot_results.py --retry64` validates saved result/plan receipts and aggregates them. `run.py` and `retry64.py` are generation/collection implementation pointers, not a freshly validated launch recipe. Do not execute their submit/worker modes for figure reproduction.
4. Images: `experiments/qwen_image_pilot/paper_split.py` now has explicit `--root`, `--template`, `--output`. Its current `recovery.py` and `REPRODUCIBILITY.md` are separately owned work; integrate those through their owner. The historical image provenance did not pin its template, so current recipe discovery does not prove exact historical template/source-byte identity.
5. Convergence and compute: local `plot.py`/`summarize.py` consume saved histories/accounting. Remote exporters are unnecessary for reproducing these saved analyses. Historical own-loss stopping is not later validation selection; recorded consumption is not matched-compute performance.
6. Organism table: `consolidation-20260915/build_report.py --output-dir NEW` aggregates saved scored logs with native Inspect statistics. Manual table transcription remains separate. It describes one organism's checkpoints, without verified RMCT intervention linkage.

Every command-map recipe hash describes the bytes inspected now, not proof that those exact bytes generated a historical figure. New clean RMCT runs, actual exposure matching, primary monitoring target comparison and replacement coverage remain unassigned/unverified in this handoff. No token caps or run configurations were changed.

## Verification performed

`python test_build_figures.py`: seven tests passed, covering no-write check-only, late-source tampering before any output, duplicate/undefined cells, path escape, output collision, exact figure copying and not claiming manual tables were generated.

The full builder ran against the real pinned artifacts into a separate `portable-smoke/` directory: all 14 source hashes verified, 13 figure files assembled, two table sources recorded without generation. Both 2x2 grids rendered; the switch panel was visually inspected for preserved layout and readable warning. The existing compiled manuscript and its original figure inventory were not overwritten.
