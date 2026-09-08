# Experiment consolidation and recovery

Start with the [experiment catalogue](../../experiments/README.md). Each ID
links a scientific question to its protocol, source snapshot, environment,
outputs, owner and acceptance evidence. The [completion ledger](PLAN.md)
records remaining integration and validation work; this directory is not a
claim that consolidation is finished.

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
checkpoint/result roots. The initial remote pass hashed small metadata and
log/data files; remaining large files and symlink targets are explicitly
unverified. An inventory is not an artifact backup.

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
