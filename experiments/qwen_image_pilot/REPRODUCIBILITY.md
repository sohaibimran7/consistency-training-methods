# Image-suite offline reproduction

Historical full-suite results used **65,536 generated tokens on all datasets**,
including LogiQA/HellaSwag, with 131,072 total context and thinking requested.
The earlier feasibility pilot used 20,480. Do not relabel historical outputs.
The new policy communicated by the authorized consolidation task is **20,480
for LogiQA/HellaSwag and 65,536 for other datasets**. This contribution documents
that policy only: it does not change generation code, launch jobs, or rewrite
archived manifests/results. A future run needs an explicit per-dataset budget
implementation and a new manifest; current full_suite.py still reproduces the
historical uniform 65,536 allowance. Reasoning consumes the generation budget.

Run the following from the repository root using its Python environment.
Every output directory must be new. Input roots are read-only. Paths below are
examples; supply the paths to your staged artifacts and chosen chart template.

```sh
python experiments/qwen_image_pilot/collect_full.py --run-root /data/original --run-root /data/act-retry --run-root /data/gemma-retry --output /tmp/new-suite/collected
python experiments/qwen_image_pilot/collect_recovered.py --root /tmp/new-suite --retry-root /data/error-retries --output /tmp/new-recovery
python experiments/qwen_image_pilot/report_full.py --root /data/staged-suite --output /tmp/new-report
python experiments/qwen_image_pilot/paper_full.py --root /data/staged-suite --template /data/chart-spec.json --output /tmp/new-paper
python experiments/qwen_image_pilot/paper_split.py --root /data/staged-suite --template /data/chart-spec.json --output /tmp/new-splits
python experiments/qwen_image_pilot/link_full_logs.py --root /data/staged-suite --output /tmp/new-inspect-links
python experiments/qwen_image_pilot/report.py --root /data/pilot --output /tmp/new-pilot-report
python experiments/qwen_image_pilot/plot.py --root /data/staged-pilot --output /tmp/new-pilot-plots
python -m unittest discover -s experiments/qwen_image_pilot -p 'test_*.py'
```

Recovery can use `--retry-rows /data/retry-rows.json` instead of `--retry-root`
for entirely offline testing. It writes successful-only `recovered-rows.json`,
`merged-rows.json`, and `unresolved-manifest.json`. Stage original `rows.json`
plus the recovered overlay under a new suite's `collected/` directory for plots;
retain the unresolved manifest alongside them. Do not substitute merged rows
as originals while also applying the overlay. Missing/failed retries retain
original failures; duplicate successes and changed identities fail closed.
Unparsed or token-limited completed outputs are retained as such, not discarded
or relabelled successful answers. Only request success determines recovery.

Pilot plotting expects `scored-rows.json` and the original `manifest.json` in
its staged input root. Raw-log collectors require Inspect and local copies of
logs in the original `runs/<model>/rank-*/logs/` structure. Collectors do not
download logs. The plotting scripts require the shared paper renderer and
checkpoint_publication statistics dependencies. Templates are explicit inputs,
not inferred from a sibling artifact directory.

No statistical definitions changed: paper accuracy excludes request errors;
descriptive report accuracy counts scheduled requests. TBSR requires parsed
clean/biased outputs and clean answer different from the bias target. Split
figures use whole-question bootstrap intervals and paired permutations.
Historical final-checkpoint identities remain those in launch-plan.json, not
later retrainings. Fonts/images are not included in this source contribution.
