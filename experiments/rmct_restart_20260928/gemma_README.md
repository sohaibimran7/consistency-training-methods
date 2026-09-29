# Gemma-only integration contribution

No standalone launcher: the integration coordinator owns job submission and the
shared runner/parser. Do not invoke the old Gemma patience driver: it has the
wrong stopping controller for this fresh campaign.

`gemma_config.py` defines the new policy, not a complete training CLI config.
The integrated runner must enforce fresh original weights/optimizer/cache,
explicit thinking mode, per-dataset generated-token caps including reasoning,
and the validation-only selection contract. Bind `approval_reference` to the
actual user authorization, not merely to this document. Historical checkpoints
and validation histories must not be reused.

After incorporating the contribution into a clean, committed deployment:

```sh
python -m experiments.rmct_restart_20260928.gemma_preflight \
  --source-root "$SOURCE_ROOT" --source-commit "$SOURCE_COMMIT" \
  --model "$GEMMA_SNAPSHOT" --validation-manifest "$VALIDATION_MANIFEST" \
  --approval-reference "$APPROVAL_REFERENCE" --output "$NEW_CPU_RECEIPT"
```

This does NOT authorize optimizer work. It validates native thinking-on prompt
tokens against the actual renderer, records source/module/dependency provenance,
and binds the validation file hash. Dataset membership, disjointness, ordering
and stopping-controller semantics still require the coordinator's integrated
validation tests. Full deployment provenance must also cover the shared runner,
parser and data loaders; the CPU receipt alone is not that full manifest.

Fresh GPU base/nonzero-LoRA HF-vLLM parity against the incorporated runtime is
mandatory. Do not reuse old parity receipts or blindly run the old parity script:
its uncapped scoring-tail path is not the approved new dataset-cap policy.
Keep disposable parity adapters separate from production initialization. Record
GPU placement and actual generation/termination settings in the integrated probe.

Local policy tests:
`python3.12 -m unittest experiments.rmct_restart_20260928.gemma_test -v`

Status: contribution only; no committed deployment, GPU preflight, fresh
validation history or optimizer updates claimed. No job was submitted here.
