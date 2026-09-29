# Offline Luna paper figures

These commands recompute figures and statistics from saved scores; they never call a model or provider. Python 3.11+, NumPy and Matplotlib are required. Set `OPENBLAS_NUM_THREADS=1` for small matrix bootstraps. Each command requires explicit input/output paths and writes `input-provenance.json` with SHA-256 input hashes, seed, draw count and limitations. Do not publish private trajectories/ledgers in the code PR.

## Historical status and interpretation

**Historical RMCT outputs must not be presented as results from an unaffected or corrected training run.** Terminal-answer parser corrections repair evaluation labels, not training rewards. These figures remain historical/sensitivity analyses. Base clean labels are unverified. ICL calibration examples retain historical labels. Observed single-pair towards-bias switches are not causal influence ground truth. Answer disclosures inside CoT remain visible to the monitor. No output, truncated response or missing score is imputed as a negative.

## Commands

Set `LUNA_SCORES` to the directory containing the three original ledger folders, `LUNA_ANALYSIS` to the directory containing the saved analysis artifacts, and `LUNA_OUT` to a new output directory. These roots may differ. Original resolved locations and exact hashes are in `handoff-manifest.json`.

```sh
export OPENBLAS_NUM_THREADS=1
python scripts/paper_luna/filter_comparison.py \
  --data "$LUNA_ANALYSIS/luna-filtered-unfiltered-20260928/with-topup/data.json" \
  --output "$LUNA_OUT/filter"
python scripts/paper_luna/rare_analysis.py \
  --data "$LUNA_ANALYSIS/luna-rare-cache-first-20260928/analysis/data.json" \
  --counts "$LUNA_ANALYSIS/luna-rare-cache-first-20260928/analysis/counts.json" \
  --output "$LUNA_OUT/rare"
for plot in icl_summary icl_configurations icl_quadrants; do
  python "scripts/paper_luna/$plot.py" \
    --samples "$LUNA_ANALYSIS/parser-fixed-20260923/samples.json" \
    --ledger-dir "$LUNA_SCORES/luna-icl-20260922" \
    --ledger-dir "$LUNA_SCORES/monitorability-base-cot-xhigh-20260920" \
    --ledger-dir "$LUNA_SCORES/monitorability-luna-rerun-20260920" \
    --output "$LUNA_OUT/$plot"
done
python -m unittest discover -s tests -p test_paper_luna.py
```

Use defaults (10,000 draws; seed 20260922 for ICL, 20260928 otherwise) for reproduction. `--bootstrap` and `--seed` permit explicitly labelled sensitivity/smoke runs; do not substitute them silently for paper defaults.

## Command to figure map

All names below have PNG and PDF versions. Each command also writes statistics JSON.

| Command | Outputs | Cohort |
|---|---|---|
| `filter_comparison.py` | `curves-available`, `curves-matched`, `auroc`, `already-matched-diagnostic`, `four-quadrant-diagnostic` | Available scored pairs; matched sensitivity separately |
| `icl_summary.py` | `fnr-fpr`, `auroc`; paired AUROC contrasts | Original 273-case all-model/all-config intersection; historical summary includes zoom |
| `icl_configurations.py` | `fnr-fpr-{zero_shot,distribution_only,examples_only,both}`, `fnr-fpr-configurations` | Same 273 matched cases; preferred full-range method overlays |
| `icl_quadrants.py` | `score-quadrants-CONFIG`, `fnr-fpr-acknowledgement-CONFIG` | Same 273 cases, split by model-specific acknowledgement; absent acknowledgement excluded only from strata |
| `rare_analysis.py` | `quadrants-CONFIG`, `fnr-fpr-CONFIG`, `weighted-vs-existing-CONFIG` | Cached common cases plus all available rare cells; not a random sample |

The filter diagnostic uses clean-match × biased-match, NOT acknowledgement × switch. Only clean ≠ cue AND biased = cue is positive. Each of its four panels has one class, so within-panel AUROC is undefined. Score-distribution curves are shown instead. Higher scores are desirable for positive cases, lower for negatives.

## Denominators and uncertainty

Available filter cohorts (Base/BCT/RMCT): filtered n=487/777/853, positives=154/154/206; unfiltered n=645/946/1056 with the same positives. Already-matched cases added=158/169/203. Matched sensitivity uses 439 filtered vs 588 unfiltered cases: separate intersections, not a pure within-cohort filter effect. Primary comparisons use different available-score populations across methods.

Rare selected cohorts n=320/314/319; historical cached subsets n=298/286/284. Weighting calibrates acknowledgement × switch masses to saved eligible-population counts. Unknown historical inclusion probabilities mean these are sampling sensitivities, not unbiased population estimates. Silent-switch cells have only 1 Base and 4 RMCT cases.

All bands are pointwise percentile 95% intervals, not simultaneous bands. Bootstrap units are `(dataset,qid)`, shared across methods/configurations and biases, stratified by dataset. One-class ROC draws are omitted; rare poststratified draws missing a required population stratum are omitted. Valid counts are recorded. ICL banks are fixed: resampling does not capture demonstration-selection variability. Duplicate FPR interpolation uses its rightmost (lowest FNR) point, then linear interpolation.

## Input boundary and dependencies

The new ICL loader replaces `exec(recipe.py)` entirely: it reads `samples.json` and each ledger's `luna-results.jsonl`/`full-private-labels.json`, applies corrected terminal labels and eligibility, then recomputes intersection, metrics and bands. Ledger order is significant and matches the historical recipe.

The filtered and rare commands start from normalized, frozen row-level JSON, not saved plot coordinates/PDFs. `data.json` contains method/question/bias identifiers, observed labels, scores and request IDs. Rare data additionally contains `ack`, `baseline`, all four `scores`, with `counts.json` holding population quadrant counts. Rebuild these using the integrated `build_inputs.py`; [UPSTREAM.md](UPSTREAM.md) gives explicit private-log, template, ledger and path-map inputs. The owner reproduced all 2,647 filtered rows and 953 rare rows exactly against frozen JSON; `upstream-verification.json` records source hashes and verification. Corrected sample labels, eligible-population and rare-case selection remain explicit upstream inputs. No parser adjudication, API inference or training is rerun by these commands.

Replaced sources: `tmp/plot_filter_comparison.py`, `tmp/plot_rare_analysis.py`, `tmp/plot_icl_by_configuration.py`, `tmp/plot_icl_quadrants.py`, and the saved ICL `recipe.py`. All portable plotting dependencies are in this directory plus NumPy/Matplotlib and the explicitly supplied data files. No absolute worktree path or dynamic recipe execution is used by the code.
