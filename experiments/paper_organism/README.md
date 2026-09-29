# Historical organism table: saved-score replay

`build_report.py` recomputes native Inspect mean/stderr from saved scored logs.
It never invokes a model or grader. Python3.11 and InspectAI0.3.246 were used
for the verified replay; full presentation additionally requires Matplotlib.

```sh
python experiments/paper_organism/build_report.py \
  --old-analysis-dir "$OLD_ANALYSIS" --prior-analysis-dir "$PRIOR_ANALYSIS" \
  --output-dir "$NEW_OUTPUT" --statistics-only
```

The old directory supplies `results.json`; the prior directory supplies
`report.json`. These point to hash-verified raw/scored Inspect logs. For relocated
logs pass `--path-map exact-path-map.json`, an original-path to new-path mapping;
content hashes remain mandatory. No frozen data manifest is rewritten.
Use a new output directory. `analysis.json` contains all rows, condition and
pooled statistics, source hashes and installed metric implementation hashes.
`manifest.json` binds outputs. Manual LaTeX transcription remains separate.

For the original full report and figures omit `--statistics-only` and provide
`--petri-root` and `--lasr-root` containing the historical linked figures,
reviews and source references. The bundled report template may be supplied
explicitly with `--template`. Full link checking remains mandatory; statistics
mode deliberately does not claim to reproduce that presentation layer.

Integration replay on28September:232 cohort memberships,224 unique trajectories,
277 sources checked, zero model/grader calls. All saved rows, pooled summaries,
condition statistics and reuse counts exactly matched the historical
`evalaware-v1-v4-consolidated-20260915/inspect-se-v2/analysis.json`.
The full presentation also ran successfully in a separate output directory:
three figures, report/condition/log pages, 283 checked source files, and all
local links validated. No original output was overwritten.

Interpretation stays historical: one organism's checkpoints, separate valid
denominators, expressed awareness not latent awareness, conditional repeat
standard errors not confidence intervals, no verified RMCT intervention effect.
