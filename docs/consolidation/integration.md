# Integration decisions

Consolidation starts from main `9d574e8`. Preserved source snapshots and the
Git bundle remain the authority for recovering the original implementations;
the consolidated branch is a new source state. No current campaign checkout
has been replaced with this branch.

| Source | Decision and destination | Validation / remaining gate |
| --- | --- | --- |
| PR #9, `5d9aa7b` | Integrated in `b846e6d`; retains explicit phase sharing, replicated updates, worker barriers, timeout separation, checkpoint/RNG custody and preflights. Later experimental additions are reconciled against these safeguards. | Baseline clean Linux CI passed. The merged backend needs its own final test result; no new GPU parity claim. |
| PR #6, `6790866` | Tinker async client setup and bounded SFT pipelining integrated in `7e56b71`. Its local controls are reconciled with the newer datum/token microbatch limits, gradient checkpointing, vLLM sizing and opt-in sleep lifecycle. RL refresh remains tied to actual optimizer updates. | Both pipelining and baseline integration tests pass. Old PR remains open until final feature disposition and replacement review are complete. |
| Figure 6, `3984e08` | Integrated in `04e9a06`. Its serving stack has an explicit `figure6` GPU profile; RMCT training keeps the `training` profile. | 265 focused Figure 6/infrastructure tests and the full baseline suite passed. Historical smoke/sentinel outputs do not establish full-matrix completion. |
| `d6d6` initial source | Preserved as `c78cafa`, then reconciled into the consolidation branch. Adds named experimental protocols, EOS-only paths, publication/custody code and recovery tooling. Older import locations are updated to the main package boundary. | Combined suite initially had 1,987 passes, 15 failures and one skip. Integration regressions and untracked-fixture dependencies are being resolved before the merge is finalized. |
| Stable authentication/controller source | Integrated in `c798d70`; installed laptop tools and shared controller state remain separate. | 59 focused tests passed; fresh two-hop access was verified. |
| Latest Gemma source | A 23-file targeted source delta preserves the completed decoder/binding fixes and new tracing source. Integration is separate from the initial d6d6 merge. | 31 launcher/binding/tracer tests passed here. Generic EOS-kernel extraction and final source-fence coverage remain in progress. |
| `9e79` plotting refactor | Reconcile its shared rendering mechanics against current adapters and Figure 6 additions. Preserve adapter-specific estimates and missing-data rules. | In progress; existing scientific JSON byte identities must remain unchanged. |
| `3f19`, `a3ce`, `c9b7` and smaller worktrees | All selected source variants are preserved. Compare remaining unique behavior before choosing integration or archival; an overlapping file is not automatically redundant. | Feature disposition and safe retirement checks remain outstanding. |

The expedited-screen importer combines a particular HLE source, model and
historical overlap policy. It therefore lives with
`experiments/rmct_paper_vast_dense_models`, rather than in the reusable MCQ
adapter. The shared MCQ experiment compiler and authored HLE/Alpaca sources
retain their established home under `scripts/rmct_paper_vast_more_methods`.

Two small, exact historical contract documents now live under
`tests/fixtures/rmct_history`. Their provenance file records original paths
and hashes. Tests use these fixtures instead of requiring untracked campaign
artifacts in every checkout; production contracts and paths are unchanged.

An offline test that validates a version-pinned remote grader now supplies
the declared package inventory and separately verifies rejection of a
different inventory. It does not require the general CPU test environment
to impersonate that historical grader runtime.
