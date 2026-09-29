# Offline upstream row reconstruction

`build_inputs.py` replaces `tmp/prepare_filter_comparison.py` and `tmp/build_rare_analysis_data.py`. It reads existing private Inspect logs and provider result ledgers only. It does not submit requests or generate responses. Python 3.11+ and `inspect-ai` are needed; NumPy/Matplotlib are not required for this stage. `builder-manifest.json` records the actual Inspect version, builder SHA-256, all input hashes (including raw logs), prompt/template identities, model/effort and serialization protocol, output hashes, and explicit arguments.

Private logs, ledgers, calibration examples and generated missing-request messages must stay outside the public code repository. Supply them through an authorized data archive. The manifest contains identities/paths/hashes, not trajectory or prompt contents. Paths can be relocated using `--path-map mappings.json`, whose keys are original exact `biased_source` strings and values are accessible new paths. Original source identifiers stay unchanged in output rows for provenance and equality checks.

## Filtered/unfiltered builder

Set `LUNA_ANALYSIS`, `LUNA_SCORES`, `LUNA_BUILD` to appropriate external input/output roots. Ledger order is significant: later successful records override earlier ones, exactly as in the original scripts. Failed records never erase a valid score.

```sh
python scripts/paper_luna/build_inputs.py filter \
  --samples "$LUNA_ANALYSIS/parser-fixed-64k-20260925/samples.json" \
  --system-template "$LUNA_ANALYSIS/luna-icl-prompt-review-20260922/base-both.json" \
  --ledger "$LUNA_SCORES/monitorability-base-cot-xhigh-20260920/luna-results.jsonl" \
  --ledger "$LUNA_SCORES/monitorability-luna-rerun-20260920/luna-results.jsonl" \
  --ledger "$LUNA_SCORES/luna-icl-20260922/luna-results.jsonl" \
  --ledger "$LUNA_SCORES/luna-rare-cache-first-20260928/luna-results.jsonl" \
  --ledger "$LUNA_SCORES/luna-unfiltered-topup-20260928/luna-results.jsonl" \
  --output "$LUNA_BUILD/filter"
python scripts/paper_luna/filter_comparison.py \
  --data "$LUNA_BUILD/filter/data.json" --output "$LUNA_BUILD/filter-figures"
```

Valid parsed pairs with exactly one closing reasoning boundary and nonempty reasoning are eligible. Request identity is reconstructed from the unchanged reasoning-only JSON, system prompt, Luna model ID and xhigh effort. Only exact request-hash hits are reused. Missing scores are counted and excluded, never assigned zero. Already-matched missing-request records are emitted for audit only: this script cannot submit them. The operational target is `clean != promoted AND biased == promoted`.

Outputs: `data.json`, `audit.json` (coverage, prompt, missing count), `missing-already-matched-requests.json` (private), and `builder-manifest.json`. Coverage matches the original audit; historical approval narration is deliberately not fabricated for a fresh build.

## Rare-quadrant builder

```sh
python scripts/paper_luna/build_inputs.py rare \
  --population "$LUNA_ANALYSIS/luna-stratified-20260928/population.json" \
  --rare-labels "$LUNA_ANALYSIS/luna-rare-cache-first-20260928/rare-private-labels.json" \
  --templates "$LUNA_ANALYSIS/luna-icl-prompt-review-20260922" \
  --historical-ledger "$LUNA_SCORES/luna-icl-20260922/luna-results.jsonl" \
  --historical-ledger "$LUNA_SCORES/monitorability-base-cot-xhigh-20260920/luna-results.jsonl" \
  --historical-ledger "$LUNA_SCORES/monitorability-luna-rerun-20260920/luna-results.jsonl" \
  --ledger "$LUNA_SCORES/luna-rare-cache-first-20260928/luna-results.jsonl" \
  --output "$LUNA_BUILD/rare"
python scripts/paper_luna/rare_analysis.py \
  --data "$LUNA_BUILD/rare/data.json" --counts "$LUNA_BUILD/rare/counts.json" \
  --output "$LUNA_BUILD/rare-figures"
```

Requires nine template files `{base,bct,rmct352}-{distribution_only,examples_only,both}.json`. Model-specific system and calibration text are used unchanged; the exact original-message marker separates calibration from the current reasoning. Each template is hashed, so any wording/version change is visible. The zero-shot request has no calibration prefix. Historical membership requires all four hashes in historical ledgers. Selection is historical membership OR explicit rare-label membership; every selected row must have all four scores. Missing required scores or malformed eligible reasoning fail loudly. Output quadrant population counts derive from the supplied eligible population, not selected examples.

## Verification and provenance boundary

On 2026-09-28 both full builders ran using accessible private raw logs. Parsed-JSON equality holds for filter `data.json` (2,647 rows), missing requests (0), and coverage; rare `data.json` (953 rows) and `counts.json`. JSON whitespace differs from original compact files but row content and order are identical. `upstream-verification.json` records verification and complete input/output manifests. All ten tests across `test_paper_luna*.py` pass, including exact request serialization, template boundaries, remapped recovery metadata, duplicate log keys, missingness, ledger precedence and rare inclusion.

The corrected `samples.json`, eligible `population.json` and rare-label manifest are explicit upstream inputs, not recreated by these two builders. Their historical construction/parser adjudication and population-selection code belong to the broader paper pipeline. This handoff closes raw-log/score-to-normalized-row reconstruction, not those earlier stages or training. Exact historical identities remain required; no model call is needed to reproduce these analyses.

Historical RMCT training remains potentially flawed. Correcting labels does not repair training rewards. Base clean labels are unverified; calibration labels remain historical; observed switches are not causal influence. Available cohorts are selected and rare weighting does not establish population unbiasedness. These limitations must accompany downstream figures.
