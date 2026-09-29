# Paper reproducibility and corrected restart integration (draft)

## Scope

Consolidate historical research code in one review branch and integrate fail-closed
completion handling, terminal-answer parsing, explicit thinking templates, fresh
start preparation, immutable CPU provenance and native worker parity preparation.
No historical results are promoted to corrected-training evidence.

## Evidence

- 13 historical paper source hashes verified with `scripts/verify_paper_inputs.py`.
- Focused completion, parser, renderer, IPC and RL tests run locally; native GPU
  tests are separate, not covered by CPU successes.
- Original Qwen weights are checked against independently recorded cache content
  identities; Gemma independent weight pins remain a required launch gate.
- Qwen preparation uses a tracked original-argv fixture, with all resume flags
  removed at initialization. Subsequent clean-lineage segments require strict state.

## Not ready to merge or launch

- Native GPU parity and deployed end-to-end RL regression have not run.
- The Qwen scheduled launcher is deliberately preflight-only.
- Gemma launcher and independent model-identity checks remain incomplete.
- Full per-figure regeneration pipelines are not yet consolidated; the 13-entry
  manifest currently verifies historical inputs only.
- Production parser review and validation-controller integration remain open.

This draft includes substantial historical research code imported from the
existing research snapshot. Review imported code separately from restart-critical
changes. Raw trajectories, model weights and licensed image fonts are excluded.
