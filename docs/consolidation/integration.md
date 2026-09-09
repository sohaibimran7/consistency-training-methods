# Integration decisions

Consolidation starts from main `9d574e8`. Preserved source snapshots and the
Git bundle remain the authority for recovering the original implementations;
the consolidated branch is a new source state. No current campaign checkout
has been replaced with this branch.

| Source | Decision and destination | Validation / remaining gate |
| --- | --- | --- |
| PR #9, `5d9aa7b` | Integrated in `b846e6d`; retains explicit phase sharing, replicated updates, worker barriers, timeout separation, checkpoint/RNG custody and preflights. Later experimental additions are reconciled against these safeguards. | Baseline clean Linux CI passed. The reconciled source passed all 2,003 offline tests; no new GPU parity claim. |
| PR #6, `6790866` | Tinker async client setup and bounded SFT pipelining integrated in `7e56b71`. Its local controls are reconciled with the newer datum/token microbatch limits, gradient checkpointing, vLLM sizing and opt-in sleep lifecycle. RL refresh remains tied to actual optimizer updates. | Both pipelining and baseline integration tests pass. PR #6 is closed as superseded by ready-for-review PR #10; original history remains preserved. |
| Figure 6, `3984e08` | Integrated in `04e9a06`. Its serving stack has an explicit `figure6` GPU profile; RMCT training keeps the `training` profile. | 265 focused Figure 6/infrastructure tests and the full baseline suite passed. Historical smoke/sentinel outputs do not establish full-matrix completion. |
| `d6d6` initial source | Preserved as `c78cafa`, then reconciled into the consolidation branch. Adds named experimental protocols, EOS-only paths, publication/custody code and recovery tooling. Older import locations are updated to the main package boundary. | Reconciled merge `090152f` passed all 2,003 offline tests with one skip, after resolving integration regressions and untracked-fixture dependencies. |
| Stable authentication/controller source | Integrated in `c798d70`; installed laptop tools and shared controller state remain separate. | 59 focused tests passed; fresh two-hop access was verified. |
| Latest Gemma source | A 23-file targeted source delta preserves the completed decoder/binding fixes and new tracing source. Integration is separate from the initial d6d6 merge. | 31 launcher/binding/tracer tests passed here. Integrated in `d102a98`: generic kernel, saved logits suppression, final-token prefill, binding/tracing and exact Inspect metadata registration. The missed metadata delta was separately preserved; 55 coupled checks passed after integration. Both Gemma and AITA source fences bind the kernel. No active runtime was changed. |
| `9e79` plotting refactor | Integrated in `c35f8b7`: shared rendering/registry/log-reading mechanics, with adapter-specific estimates and missing-data rules retained. | 85 focused checks and real 9e79 smoke rendering passed; original recipe bytes unchanged. Package registry TOMLs are included and verified in an isolated wheel. |
| `3f19`, `a3ce`, `c9b7` and smaller worktrees | [Variant review](remaining-variants.md) confirms covered backend features and mapped legacy data helpers. All 35 paper assets are archived; four legacy chart recipes remain snapshot-only because their old presentation semantics are unsupported. | Inactive a3ce was archived intact after checks; 3f19/c9b7 remain attached task workspaces. No missing backend patch was identified. |

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

The stable controller extension in `7dd793e` adds the owner-validated Bayesian
batch and CPU-only Gemma Luna profiles. Source/test/documentation hashes match
its v2 receipt;96 focused checks passed here. Installed service state remains
owned by the authentication/controller task.

`8e47948` adds immutable generic-runner attempts and child command records.
Each attempt retains the exact authored YAML and resolved-plan bytes, source
bundle and parent runtime identity. 52 focused recorder/runner checks and 18
backend-integration checks passed. Fresh source fences also bind the new
recording/provenance/topology helpers.

The final code revision `8e47948` passed 2,114 offline tests with one gated-HLE
skip both locally and from the committed lock on clean Linux
([run 34246401841](https://github.com/sohaibimran7/consistency-training-methods/actions/runs/34246401841)).
See [the exact verification record](final-verification.json).

PRs #6 and #9 are now closed as superseded by ready-for-review PR #10. The
clean dedicated PR9 checkout was archived intact after process/environment
checks; its restore record is [here](retired-phase-shared-worktree.json).
GitHub main has not been merged or deployed from the replacement branch.
