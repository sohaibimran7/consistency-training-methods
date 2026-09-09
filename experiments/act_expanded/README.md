# Expanded ACT question freezer

`selection.py` prepares an expanded ACT condition without generating a model
response.  It deliberately separates question selection from wrong-argument
generation so the existing ACT-Max targets cannot be mixed silently with a
new generator.

The input is three separately attested legacy JSONLs:

- the complete 1,500-per-dataset legacy source;
- the existing 1,400-per-dataset ACT-Max training selection; and
- the already-frozen 100-per-dataset IID selection.

It also consumes local three-field snapshots of the two pinned public training
splits (`question`, `options`, and `ground_truth_idx`).  The snapshots are
rendered into the repository MCQ format, then matched against the complete
legacy population by a conservative normalized-content key rather than only by
question ID.

The required order is:

1. Freeze each staged pinned source with `freeze-source`.
2. Run `audit`. This refuses the run unless the runtime-derived collision
   counts reproduce the pinned conservative audit: 1,390 physical / 1,387
   unique LogiQA collisions and 3 HellaSwag collisions.
3. Review and retain the content-addressed audit JSON.
4. Run `materialize` with that audit. It replays the audit and publishes only:
   a new 100-per-dataset fresh IID reserve and a question-only generation
   candidate JSONL.
5. Generate one new wrong argument for every eventual training candidate using
   the single required generator, `openrouter/google/gemma-4-31b-it`; record a
   target provenance artifact before training. Legacy `biasing_text` is not a
   valid input to this condition.

Assuming no additional collision with a separately supplied frozen evaluation
population, the audited candidate ceilings are:

| Reservation | LogiQA candidates | HellaSwag candidates | Balanced total ceiling |
| --- | ---: | ---: | ---: |
| Existing frozen IID only | 7,376 | 41,302 | 14,752 |
| Plus new fresh IID (100/dataset) | 7,276 | 41,202 | 14,552 |

HellaSwag is intentionally stored as staged oversampling.  The actual balanced
training count remains `N = floor(successful_logiqa / optimizer_granularity) *
optimizer_granularity`; generate HellaSwag in its frozen order until it has at
least `N` successful homogeneous targets.

No Hugging Face, OpenRouter, GPU, credential, or network operation is made by
this package.
