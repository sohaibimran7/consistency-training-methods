# Historical monitor overview reproduction

`overview.py` replaces the `luna-overview` branch of `tmp/regenerate_monitor_figures.py` and its injected `corrected()` / saved external recipe dependency. It directly reads corrected samples, saved scores and private labels. No API calls or external Python execution.

```sh
OPENBLAS_NUM_THREADS=1 python scripts/paper_luna/overview.py \
  --samples "$LUNA_ANALYSIS/parser-fixed-20260923/samples.json" \
  --ledger "$LUNA_SCORES/monitorability-luna-rerun-20260920/luna-results.jsonl" \
  --labels "$LUNA_SCORES/monitorability-luna-rerun-20260920/full-private-labels.json" \
  --output "$LUNA_OUT/overview"
```

Requires Python, NumPy, Matplotlib. Inputs remain private/external. `input-provenance.json` records all three SHA-256 input identities and parameters; `label-audit.json` reports changed targets, exclusions and missing sources. Outputs are `monitorability-overview.png/pdf`, `fnr-fpr-curves.png/pdf`, and `results.json` including matched and all-available metrics, dataset/bias breakdowns, paired contrasts and historical usage.

The exact historical cohort is retained: intersection across BCT/RMCT × minimal/xhigh × CoT-only/sequential/prompt+CoT/prompt-only (16 conditions). Corrected eligibility requires valid clean/biased answers and clean != promoted. Target is biased == promoted in that cohort. Saved monitor scores are not modified. Last successful score per request is retained; failed records never become negatives. Historical planned requests default to 12,092 (`--planned-requests` explicit override), and successful/failed counts are ledger-level, not eligible-case denominators.

Default 10,000 paired dataset-stratified question-cluster resamples, seed 20260920, matching the original. `--bootstrap`/`--seed` are explicit sensitivity options. AUROC ties receive half credit. One-class bootstrap draws are omitted and valid counts reported. Intervals are descriptive/unadjusted 95%; no multiplicity claim. FNR–FPR curves reproduce historical point estimates (no newly added bands). CoT uplift is paired AUROC(prompt+CoT) minus AUROC(prompt-only).

Historical RMCT training caveats remain: parser-corrected evaluation does not repair training rewards. Observed switches are not causal-influence ground truth. Historical usage aggregates ledger entries with usage exactly as before; it is not a new billing reconciliation. Plot warnings are explicit; all underlying statistical results are retained unchanged.

Verified 2026-09-28 at defaults: entire `results.json` equals historical `parser-fixed-20260923/luna-overview/results.json`, including all metrics, intervals, contrasts, breakdowns and usage. Cohort: 479 cases / 118 question clusters (HLE 122, LogiQA 171, HellaSwag 186). Twelve combined offline Luna tests pass. Small layout-only changes prevent annotation/title overlap; PDFs are additionally exported. Input SHA-256s: samples `6edd599000b3f14e85f5aa0c3a3375a4637860811bb3b5b58f5f8035f6643566`; ledger `0e3b9efa135cde786da5767bb7f8ed15bae180fcca895408e31e77355e460d4d`; labels `0230b687f2347432b97fd6735c28b11d1c4e09bd14425fb976a893278f97fdcc`.
