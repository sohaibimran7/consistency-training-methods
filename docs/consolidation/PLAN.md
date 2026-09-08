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

Parent task owns preservation, catalogue population, integration, environment/CI work and documentation. Subagents are resolving disjoint backend, trainer and adapter conflicts and implementing generic source/runtime manifest capture. The existing “Check adaptor duplication” task owns its adapter investigation; coordinate before changing adapters.

## Current progress

- Consolidation branch created from current GitHub main `9d574e8e448d4c8bf32373600bb5050cbfb1368f`.
- Initial Git bundle created and verified, preserving 89 refs and complete referenced history.
- Source snapshots captured from all 20 existing local worktrees. Shared blob storage holds 1,080 distinct contents (19,219,078 bytes). Snapshot identities include new/modified source and executable modes; omitted non-source paths are listed.
- Captured source snapshots from 15 remote campaign copies; independently verified all blob/tree hashes. Source-only capture has 1,038 distinct contents (16,426,291 bytes). An initial capture that also included an archived environment was retained under the preservation `_archive/`, then narrowed explicitly.
- Independently restored all 616 selected `d6d6` files to a new temporary directory and verified their hashes and modes. Local source snapshots from all 20 worktrees passed identity verification.
- Integrated PR #9, PR #6's Tinker/SFT changes, the Figure 6 branch, and stable authentication/job-controller source. GitHub main and the open PRs remain unchanged.
- Baseline commit `af9a6a4` passed 1,056 offline tests. The remaining three CPU distributed tests passed separately with loopback socket access; no GPU experiment was run. Clean Linux CI also passed all 1,059 tests from the committed dependency lock (run `34237627335`).
- Added a single-schema catalogue and CLI with 31 experiment entries. Historical states without sufficient acceptance evidence remain unconfirmed. The catalogue and retention guide are linked from the main README.
- Produced a hashed Linux CPU dependency lock and offline CI workflow, plus separate training and Figure 6 GPU profiles. GPU profile installation and numerical validation are separate gates.
- Verified all 116 referenced AITA r006 files (912,210,600 bytes) and replayed the preserved paired analysis: 1,591 pairs produced exactly the four recorded counts. This establishes completion only for its documented modified protocol.
- Preserved the d6d6 source as commit `c78cafa7ad848498c589e522a2d3742ad17bbae1` with tag `archive/d6d6-source-20260908-initial`. Its merge into the baseline is in progress; backend, trainer and MCQ adapter conflicts have separate owners.
- Local artifact inventory covers 20 roots, with 8,620 distinct contents totalling 17.664 GiB. The deduplicated Isambard recovery archive has verified 8,619 blobs (18,967,069,872 bytes). One older active RMCT scheduling receipt changed after inventory and is explicitly unavailable; a delta retains its newer state. Remote artifact locations and partial hashing limitations are indexed separately.
- Four Git registrations for already-missing temporary Figure 6 worktrees were moved into `.git/_archive/worktree-registrations-20260908/` and verified. No existing worktree was moved.
- Shared manifest provenance is under review; generic command/input/output lifecycle records and retry coverage remain outstanding.
- Coordination requests sent to active Gemma, RMCT and Figure 6 tasks. No experiment code/jobs changed by consolidation.

## Preservation location

Initial local recovery material is outside Git at:

`/Users/work/.codex/visualizations/2026/09/08/01a08105-5843-7be1-9f22-0a739c1a40fe/preservation/20260908-initial/`

It contains `repository.bundle`, `index.json`, `snapshots/`, `blobs/` and the remote source archive. The source package has a verified second copy at
`a5v.aip2.isambard:/projects/a5v/sohaib.a5v/_archive/ctm-consolidation-preservation-20260908-initial/ctm-source-preservation-20260908-initial.tar.gz`.
Both copies have SHA-256 `8da62113a0925182e504f4bc7ae18fc89ffb1892f04939868783b2ee8e9c7c43` (26,948,474 bytes). This source backup excludes model/result payloads; source preservation alone does not justify retiring their locations.

## Protected active experiments

- Gemma: diagnostic runtime `ctm-gemma4-12b-base-eval-20260908/repo`; prepared full campaign `ctm-gemma4-12b-base-eval-repaired-20260908/repo`; historical failed attempt under `ctm-gemma4-12b-base-eval-20260825/repo`. All under `/projects/a5v/sohaib.a5v`. No completed suitability result yet. Keep the shared `.venv-muse-cu129` runtime untouched.
- RMCT: `ctm-rmct-convergence-gcall-r2-20260814/repo` uses a mixed source state with PR #9 backends and later r5 continuation code. Await current owner's exact source/attempt status before integration or retirement decisions.
- Figure 6: primary local checkout on `codex/evalaware-figure6-isambard` remains owned by its active task. No in-place rebasing or source changes.

## Audit evidence

The detailed audit and point-in-time evidence are stored at:

`/Users/work/.codex/visualizations/2026/09/08/01a08105-5843-7be1-9f22-0a739c1a40fe/codebase-audit/audit.md`

The initial audit passed 100 focused offline tests. That evidence applies to the audited source states, not to future consolidated changes.
