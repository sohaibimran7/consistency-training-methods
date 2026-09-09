# Experiment consolidation and recovery

Start with the [experiment catalogue](../../experiments/README.md). Each ID
links a scientific question to its protocol, source snapshot, environment,
outputs, owner and acceptance evidence. The [scope and historical ledger](PLAN.md)
records the 9 September correction: PR #10 is an integration reference for
selective review, not a proposal to merge every experiment into main. Experiment
branches and recoverable snapshots remain valid homes for scientific history.

## Restore preserved source

[`source-preservation.json`](source-preservation.json) identifies all 20 local
and 15 remote source snapshots, plus four registrations whose directories
were already missing. The archive contains complete selected source bytes,
including dirty/new files, modes and symlink targets. It also contains a Git
bundle of all 89 referenced histories observed before consolidation.

The recovery archive has SHA-256
`8da62113a0925182e504f4bc7ae18fc89ffb1892f04939868783b2ee8e9c7c43`.
Its local and Isambard locations are in the preservation record. Both copies
have matching digests. A laptop-only recovery copy is no longer the sole copy;
the storage service's backup policy and retention remain unverified.

After retrieving the archive to a new recovery directory and checking that
digest, extract it there. Use the included helper with the exact snapshot
member named by the catalogue:

```bash
python 20260908-initial/restore.py \
  20260908-initial/snapshots/d6d6.json \
  --destination /absolute/new/source-directory
```

The destination must not exist. For remote snapshots use the corresponding
`20260908-initial/remote/snapshots/` entry; the helper finds its sibling blob
store. Omitting `--destination` verifies the snapshot without restoring it.
The helper verifies the tree identity, each blob and its size before restore.
The d6d6 source (616 files) and the composite remote RMCT source (326 files)
were independently restored and checked during consolidation.

These snapshots preserve observed source states. They do not retrospectively
prove that a historical attempt executed those bytes. Keep its own launch
contract, protected-source attestation, environment and raw results too.

## Retrieve artifacts

[`artifact-locations.json`](artifact-locations.json) groups the local and
Isambard artifact inventories by location. Each inventory has a content hash
and contains the individual filenames, sizes and available SHA-256 hashes.
The local pass hashed all regular files in the declared artifact/log/data/
checkpoint/result roots. The initial remote pass hashed only smaller files. A follow-up read verified
all 25,292 inventoried files (61,813,967,440 bytes), including large checkpoints,
without errors or changed earlier hashes. See
[`remote-artifact-verification.json`](remote-artifact-verification.json) for
coverage and recovery-copy overlap. About 47.9 GB of distinct remote content
remains outside the laptop-artifact recovery archive, so its original locations
remain protected. Base-model caches and later outputs are outside this pass.
An inventory is not an artifact backup.

The initial local regular artifacts now have a separately verified recovery
archive on Isambard: see [`artifact-backup.json`](artifact-backup.json) for its
URI, hash, scope and one unavailable older scheduling receipt. Verify or
selectively restore it with `python scripts/restore_artifacts.py ARCHIVE`.
Restoration additionally takes `--root ORIGINAL_ROOT --prefix RELATIVE_PATH
--destination NEW_DIRECTORY`. It requires a new destination, verifies every
blob and reports missing selected identities. It never overwrites a live run.

Do not retire a location until the experiment's required inputs, checkpoint,
raw outputs and analysis parents have verified retrievable copies. Missing
or unverified dependencies remain visible in the catalogue. A hash proves
identity once bytes are found; it does not recover missing bytes.

## Integration boundaries

Work happens on `codex/experiment-consolidation`. PR #9 and the committed
Figure 6 branch and PR #6's Tinker/SFT contribution are integrated there. Existing campaign directories and
installed laptop helpers are separate from this checkout.

The RMCT training and Figure 6 serving GPU stacks have separate explicit
setup profiles and constraints. A setup profile is not yet a fully locked,
reconstructed GPU environment. CPU lock and CI evidence are described in
[`environments/README.md`](../../environments/README.md).

Preserve historical scientific choices, JSON byte contracts and output
identities while sharing execution/validation/plotting mechanics. In
particular, r5 still imports earlier convergence plans, incomplete Gemma
screens are not completed susceptibility results, and Figure 6 diagnostics
are not strict full-matrix results.


## Retired worktrees

Four fully merged legacy Claude worktrees were moved intact to the primary
checkout's `_archive/worktrees-20260908/`. Their Git registration directories
were separately retained under `.git/_archive/worktrees-20260908/`, so they
no longer appear as active registered worktrees. See
[`retired-worktrees.json`](retired-worktrees.json) for exact paths, original
heads, metadata hashes and the restoration command. The check verified no
process or primary Python environment reference before each archival pass.
All ignored files and environments moved with their directories.

The earlier four stale registrations for already-missing temporary Figure 6
worktrees were archived separately. Existing experiment/diagnostic worktrees
remain until their task ownership and remaining source variants are resolved.

The integration reference is [PR #10](https://github.com/sohaibimran7/consistency-training-methods/pull/10);
the recommendation to merge it wholesale has been withdrawn.
[`experiment-merge-validation.json`](experiment-merge-validation.json) records
local and clean Linux evidence for the initial reconciled experiment merge.

The inactive `a3ce` worktree was also archived intact after its 260 selected
source files matched preservation and its unique chart recipes received an
explicit historical disposition. Its task was already archived.
[`retired-a3ce-worktree.json`](retired-a3ce-worktree.json) records restoration.
The dedicated PR9 checkout was subsequently archived after the superseded PR
closed; see [its restore record](retired-phase-shared-worktree.json). There are
now 14 registered worktrees. [Remaining workspace dispositions](retained-worktrees.json)
explain the active and unrelated task paths.

The final code passed 2,114 offline tests with one gated-HLE skip locally and in
clean Linux CI. See [the final verification](final-verification.json).
The 15.9 MB additional recovery package is verified locally; automatic approval
review stopped its Isambard upload pending explicit authorization. Its exact
hash, intended destination and current state are in [the backup record](artifact-backup.json).
