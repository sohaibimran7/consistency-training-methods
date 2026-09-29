# Saved AITA receipt aggregation

```sh
OPENBLAS_NUM_THREADS=1 python experiments/paper_aita/plot_results.py \
  --root "$AITA_SAVED" --output "$NEW_OUTPUT" --retry64
```

Inputs: `plan-v3.json`, original `results/*.json`, and `results64/*.json` plus
`results64/retry-plan.json`, preserving the historical relative tree. Code
dependencies are canonical `experiments.elephant_aita_ntaflip.publication` and
the MCQ method palette, plus NumPy/Matplotlib. No external Python recipe needed.
`--audit-only` checks all receipts without making outputs; do not use Python-O.
Output must not already exist. Without `--retry64`, uses original20,480-token
results; with it, audits the selective65,536-token retry and unchanged responses.
No generation, grading, submission or provider call occurs.

Checks retain1,591 exact pair IDs,3,182 responses per condition,16 rank receipts,
plan/retry-plan content identities, selected-retry immutability and observed
coverage. Produces terminal/all-checkpoint PNG, PDF and JSON, plus input/code
hashes. Statistical code remains the existing10,000 paired-pair bootstrap,
exact McNemar tests and within-panel Holm correction.

28September integration replay: both `terminal-comparison.json` and
`all-checkpoints.json` exactly matched their historical counterparts. Common
parsed cohorts are1,194 terminal and1,042 all-checkpoint pairs. Invalid pairs
cannot count as both-NTA in the all-pairs descriptive rate; common-parsed
selection is not a missingness correction. Historical RMCT training remains
flawed; these are not corrected-training results or new inference.
