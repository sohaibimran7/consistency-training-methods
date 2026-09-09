# Experiment preservation and selective promotion

The 8 September consolidation is retained on `codex/experiment-consolidation`,
based on GitHub main `9d574e8`. On 9 September the user clarified that main should
inherit only DRY code supporting useful functionality. The recommendation to
merge PR #10 wholesale is withdrawn; this branch is an integration reference
for selective review. Existing campaign checkouts remain separate.

## Current scope

Experiments and historical implementations can remain on their own branches.
The catalogue should identify their exact commits or recoverable snapshots,
environments and artifacts; indexing a result does not require merging its
implementation into main. Preserve uncommitted source before retiring its
worktree, and retain stable references for historical results.

Each proposed change to main must identify the useful capability or concrete
defect it addresses, its actual callers, and its required dependencies. Prefer
existing shared code; extract common mechanics only where real use warrants
the abstraction. Keep scientific choices and campaign-specific recovery code
with their experiments. A useful capability can have one current caller;
speculative reuse is not a reason to create a framework.

Review the smallest sufficient change and verify its behavior. This applies
equally to provenance, catalogue tooling, plotting and runtime helpers: none
is automatically approved for main because it is labelled infrastructure.
Passing the integration suite or preserving a historical implementation does
not establish that it belongs in the maintained codebase.

PRs #6 and #9 were closed under the earlier consolidation plan. Their code and
branches remain available for independent assessment; that closure is not
evidence that their changes have reached main.

## Historical consolidation checklist (8 September)

The completed items below record work performed under the earlier scope.
They are not current merge acceptance criteria.

- [x] Preserve meaningful local and remote source states, independently verify
  restoration, and reference recovery packages from the catalogue.
- [x] Index required experiment artifact locations and content identities;
  distinguish verified retrieval from recovery copies and state retention gaps.
- [x] Maintain a validated catalogue with active, completed, diagnostic,
  invalidated and superseded protocols. Never infer scientific completion from
  tests, filenames or a successful job.
- [x] Reconcile PR #6, PR #9, Figure 6, experimental worktree variants, plotting
  and stable authentication/controller changes into reviewable commits.
- [x] Capture source, exact YAML/plan/argv, parent environment, input/output
  lineage and subprocess lifecycle in immutable attempt records that survive
  retries. Keep bespoke historical launch contracts intact.
- [x] Share provenance, execution records, topology metadata, EOS decoding and
  plotting mechanics while retaining scientific protocol and historical byte
  contracts.
- [x] Provide a clean Linux CPU lock and automated environment, catalogue and
  offline gates. Retain the historical GPU evidence with its precise scope;
  keep fresh numerical/runtime validation as a gate before campaign deployment.
- [x] Archive superseded inactive worktrees only after preserving unique content
  and checking process, task and environment dependencies. Retain explicit
  dispositions for the remaining workspaces.
- [x] Verify final frozen-source tests and clean Linux CI: 2,114 passed, one
  gated-HLE skip on both platforms.
- [x] Mark PR #10 ready for review and close superseded PRs #6/#9, retaining
  branches/history. GitHub main remains unchanged.
- [ ] Resolve the additional recovery-package upload decision. The 15.9 MB
  package is locally verified; its upload was blocked by automatic approval
  review and awaits explicit user authorization. Earlier off-laptop recovery
  copies are verified and remain available.

## Delivered evidence

| Area | Evidence | Practical limit |
| --- | --- | --- |
| Original source | [Source preservation](source-preservation.json): 20 local and 15 remote snapshots, 89 initial Git refs, verified source package on the laptop and Isambard. Restored d6d6 and remote RMCT snapshots independently. | Observed source is not retroactive proof of the bytes executed by every historical attempt. |
| Integrated source | d6d6 snapshot commit `c78cafa`, reconciled merge `090152f`, then focused topology/EOS/plot/controller/recorder commits. [Integration decisions](integration.md). | Active Gemma/RMCT runtimes and new unfinished work are separate. |
| Artifacts | [Recovery archive](artifact-backup.json): 8,619 distinct verified contents, 18,967,069,872 bytes; actual selected AITA restoration succeeded. [Remote verification](remote-artifact-verification.json): all 25,292 initial paths, 61,813,967,440 bytes, zero errors. | One old changed scheduling receipt is unavailable; its newer bytes are retained. About 47.9 GB of distinct remote content lacks a verified independent recovery copy. Base caches and new outputs are outside the inventory. |
| Catalogue | 31 entries; 44 available repository references verified by path/hash. CI checks schema, lineage, declared completion and local references. | Historical acceptance remains unconfirmed unless supported by evidence. |
| AITA r006 | [Completion evidence](aita-completion-evidence.json): 116 referenced files verified; paired replay reproduced all four recorded counts from 1,591 pairs. | Completion is only for the documented modified final-answer-only protocol. |
| Validation | [Baseline](baseline-validation.json): 1,059 clean Linux passes. [Reconciled experiment merge](experiment-merge-validation.json): 2,003 passes/one skip locally and on clean Linux. [Final source](final-verification.json): 2,114 passes/one skip locally and in clean Linux CI. | CPU integration is not GPU numerical equivalence or production throughput. |
| Plotting | [Plot validation](plot-validation.json): 85 focused checks; real 9e79 smoke log rendering; unchanged recipe bytes; registry files verified in an isolated wheel. | Smoke rendering is not an accepted scientific result. |
| Controller | [Stable extension receipt](controller-extensions.json): 11 copied files matched owner hashes, 96 focused tests passed here; owner supplied 122 test/install checks. | Consolidation did not install helpers, change shared state or submit jobs. |
| Retirement | Four already-missing registrations, four legacy Claude worktrees, inactive a3ce and the clean PR9 checkout moved into recoverable archives. [Claude recovery](retired-worktrees.json), [a3ce recovery](retired-a3ce-worktree.json), [PR9 recovery](retired-phase-shared-worktree.json). | 14 registered worktrees remain; [their dispositions](retained-worktrees.json) distinguish active and unrelated task paths from canonical source. |

## Protected campaigns and scope cutoff

Gemma job 6404189 was reported submitted by its owning task from a separately
frozen checkout. Its GPU runtime remains untouched. New isolated CPU Luna
runtime preparation is still owner work and is outside this source cut.
The validated controller extension is included; no grading job was submitted
by consolidation.

RMCT's deployed convergence/recovery repository and scheduling receipts remain
protected. Figure 6's primary checkout, the shared primary Python environment,
and the installed authentication/controller service also remain in place.
No in-place rebasing, reset, runtime upgrade or campaign-source replacement was
performed by consolidation.

The [research workflow](../research-workflow.md) is the maintained entrypoint for
new work. Original audit evidence remains outside Git at
`/Users/work/.codex/visualizations/2026/09/08/01a08105-5843-7be1-9f22-0a739c1a40fe/codebase-audit/audit.md`.
