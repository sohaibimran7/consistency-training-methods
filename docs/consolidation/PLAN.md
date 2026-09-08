# Experiment consolidation

Approved following the 8 September 2026 codebase audit. The full objective is to preserve the experiment implementations, make their results findable and repeatable, integrate the useful work on current main, consolidate repeated infrastructure, and archive superseded material safely.

## Completion requirements

- [ ] Exact source states preserved for meaningful local worktrees and remote campaign copies, with independently verified restoration and references from the catalogue.
- [ ] Required experiment data, checkpoints and result artifacts indexed with retrievable locations and content identities; retention/backup limitations explicit.
- [ ] Validated experiment catalogue and readable index covering active, completed, diagnostic, invalidated and superseded protocols; no inferred completion from tests or file presence.
- [ ] PR #6, PR #9, Figure 6 branch, experimental development worktrees, plotting refactor and authentication changes reconciled against current main in reviewable changes.
- [ ] Shared run record captures source snapshot, exact parameters/argv, environment, inputs, lifecycle and output lineage; immutable attempt records survive retries.
- [ ] Repeated provenance, execution, validation and plotting mechanics consolidated without changing historical hashes or scientific protocol semantics.
- [ ] Reproducible supported environment definitions and automated offline integration/replay gates; numerical/distributed changes get appropriate GPU evidence.
- [ ] Superseded worktrees and probes archived only after unique content is preserved and ongoing tasks/jobs no longer depend on their paths.
- [ ] Final requirement-by-requirement verification proves the above against current local and external state.

## Work ownership

The consolidation branch is `codex/experiment-consolidation` in the `ad75` worktree. Existing experiment worktrees and deployed runtime directories are protected while their tasks continue. Changes are integrated here first.

Parent task owns preservation, catalogue population, integration, environment/CI work and documentation. Subagents are implementing the catalogue API/CLI and generic source/runtime manifest capture in disjoint files. The existing “Check adaptor duplication” task owns its adapter investigation; coordinate before changing adapters.

## Current progress

- Consolidation branch created from current GitHub main `9d574e8e448d4c8bf32373600bb5050cbfb1368f`.
- Initial Git bundle created and verified, preserving 89 refs and complete referenced history.
- Source snapshots captured from all 20 existing local worktrees. Shared blob storage holds 1,080 distinct contents (19,219,078 bytes). Snapshot identities include new/modified source and executable modes; omitted non-source paths are listed.
- Remote source capture started; restoration validation and catalogue references still pending.
- Catalogue API/CLI and shared manifest provenance implementation delegated; not yet integrated or verified.
- Coordination requests sent to active Gemma, RMCT and Figure 6 tasks. No experiment code/jobs changed by consolidation.

## Preservation location

Initial local recovery material is outside Git at:

`/Users/work/.codex/visualizations/2026/09/08/01a08105-5843-7be1-9f22-0a739c1a40fe/preservation/20260908-initial/`

It contains `repository.bundle`, `index.json`, `snapshots/`, `blobs/` and the remote source archive. This is currently a laptop recovery copy, not proof of an off-device backup. Do not retire source worktrees or result locations on this basis alone.

## Protected active experiments

- Gemma: diagnostic runtime `ctm-gemma4-12b-base-eval-20260908/repo`; prepared full campaign `ctm-gemma4-12b-base-eval-repaired-20260908/repo`; historical failed attempt under `ctm-gemma4-12b-base-eval-20260825/repo`. All under `/projects/a5v/sohaib.a5v`. No completed suitability result yet. Keep the shared `.venv-muse-cu129` runtime untouched.
- RMCT: `ctm-rmct-convergence-gcall-r2-20260814/repo` uses a mixed source state with PR #9 backends and later r5 continuation code. Await current owner's exact source/attempt status before integration or retirement decisions.
- Figure 6: primary local checkout on `codex/evalaware-figure6-isambard` remains owned by its active task. No in-place rebasing or source changes.

## Audit evidence

The detailed audit and point-in-time evidence are stored at:

`/Users/work/.codex/visualizations/2026/09/08/01a08105-5843-7be1-9f22-0a739c1a40fe/codebase-audit/audit.md`

The initial audit passed 100 focused offline tests. That evidence applies to the audited source states, not to future consolidated changes.
