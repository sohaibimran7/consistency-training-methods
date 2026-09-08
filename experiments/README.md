# Experiment catalogue

[`catalog.json`](catalog.json) is the tracked index of observed protocols and
results. Scientific status is separate from file availability and job state.
`unconfirmed` means that the available evidence has not established the
accepted result; it does not mean that the experiment never ran. Entries marked
`diagnostic_only` retain their restricted scientific scope even when their
individual jobs completed. `complete` requires recorded completion evidence for the stated scope.

The initial inventory was taken on 8 September 2026. Active tasks can change
source and job state after that time. Update an entry when its protocol,
accepted result, location or successor changes; preserve earlier attempt
records. Keep large raw results outside Git and record their hashes/locations.

```bash
python scripts/experiment_catalog.py list experiments/catalog.json
python scripts/experiment_catalog.py show experiments/catalog.json rmct-convergence-r5
python scripts/experiment_catalog.py validate experiments/catalog.json
python scripts/experiment_catalog.py render experiments/catalog.json
```

The `show` command prints every protocol, source, environment, artifact and
current-task reference for one ID, plus its completion criteria and caveats.
Validation checks shape, declared completion evidence, known lineage targets
and cycles. It does not contact remote storage or infer success from a path.

[Source restoration and integration status](../docs/consolidation/README.md)
explains the recovery packages, artifact inventory and outstanding work.

| Experiment ID | Status | Scientific question |
|---|---|---|
| `irpan-2510-27062` | planned | Can the registered Irpan reconstruction be reproduced with the pinned sources, checkpoints and evaluations? |
| `mcq-wrong-argument-cross-bias` | unconfirmed | How does training on distractor arguments transfer to held-out MCQ biases? |
| `internal-consistency-method-comparison` | unconfirmed | How do the specified internal-consistency objectives compare under a common experiment plan? |
| `rmct-paper-more-methods` | unconfirmed | How do the registered RMCT and supervised/internal-consistency methods compare on MCQ bias evaluations? |
| `dense-qwen-stage1` | unconfirmed | How do the Stage 1 dense-Qwen methods compare with matched data and declared generation settings? |
| `act-repair-gate` | unconfirmed | Does the repaired ACT objective pass the registered correctness gate? |
| `act-expanded` | unconfirmed | How does expanded ACT training behave under its separate protocol? |
| `act-max` | unconfirmed | How does the ACT maximum-scale continuation behave under its separate protocol? |
| `stage1-iid-cot` | diagnostic_only | What are the paired in-domain bias-switch and acknowledgement rates on the frozen CoT diagnostic populations? |
| `stage1-iid-no-cot` | diagnostic_only | What are the paired in-domain diagnostic rates with the separately frozen no-CoT protocol? |
| `stage2-ood-hle` | unconfirmed | How do Stage 1 checkpoints transfer to the frozen HLE out-of-domain evaluation? |
| `rmct-256` | unconfirmed | What is the outcome of the frozen 256-scale RMCT training protocol? |
| `rmct-512` | unconfirmed | What changes under the explicit 512-scale RMCT continuation? |
| `rmct-256-convergence` | unconfirmed | Does the registered 256-scale continuation reach its stopping criterion? |
| `rmct-convergence` | unconfirmed | Does the original registered RMCT convergence protocol reach its target? |
| `rmct-convergence-accelerated` | unconfirmed | Does the accelerated RMCT continuation preserve its stated update protocol? |
| `rmct-convergence-r4` | superseded | What did the r4 repaired RMCT continuation establish before its failure and r5 amendment? |
| `rmct-convergence-r5` | active | Does uncapped RMCT continuation reach the registered threshold with the amended patience rule? |
| `rmct-step176-two-bias` | diagnostic_only | What are the observed MCQ switch and acknowledgement metrics for the sealed r005 step-176 checkpoint? |
| `elephant-aita-r006` | complete | What are final-answer-only NTA/NTA rates for base and RMCT steps 16, 64 and 176 under the modified AITA protocol? |
| `gemma-base-screen-20260825` | invalidated | Does the Gemma base model meet the registered susceptibility screen across both biases? |
| `gemma-eos-diagnostic-20260908` | diagnostic_only | Does EOS-only decoding finish the two selected long HLE examples without truncation? |
| `gemma-base-screen-repaired-20260908` | planned | Does the repaired Gemma two-bias screen meet the complete clean/biased pairing contract? |
| `muse-glimmer-rmct` | unconfirmed | Can the pinned Muse Glimmer RMCT replication pass its runtime and training gates? |
| `switch-gate` | unconfirmed | Does the preregistered switch gate satisfy its fixed protocol and acceptance rule? |
| `rmct-tbsr` | unconfirmed | How does the separately defined TBSR RMCT protocol behave? |
| `eval-awareness-snr-per-item` | unconfirmed | How does the specified per-item SNR evaluation-awareness intervention behave? |
| `figure6-qwen-subset` | unconfirmed | How do the three selected Qwen models compare across the registered Figure 6 conditions? |
| `figure6-midtrained-crossover` | diagnostic_only | Which parts of the midtrained discrepancy are attributable to targets, judges or generation cohorts? |
| `figure6-runtime-sentinel` | diagnostic_only | Does the sealed paired sentinel isolate a serving-runtime contribution to the midtrained discrepancy? |
| `phase-shared-parity` | diagnostic_only | Does phase-shared replicated training preserve the declared gradient and checkpoint semantics? |
