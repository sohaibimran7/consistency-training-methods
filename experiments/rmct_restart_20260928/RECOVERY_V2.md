# Separate batch and optimizer counters

The active controller uses v2 receipts. Existing v1 helpers remain only for
historical verification/tests; they cannot be used by the v2 training loop.
`step` is the actual optimizer count. `progress.sampled_batches` is the number
of consumed two-QID batches, including zero-signal batches. Validation and
patience use only actual optimizer steps at exact multiples of 64.

Each child loads a verified original 32-QID segment with `batch_offset` and
`batch_count` (both in two-QID batches). Its count is no larger than the
remaining optimizer updates. After successful child completion, actual saved
counters determine the next slice. The loader is checked before launch by
comparing the requested slice against the full verified segment. No consumed
batch is replayed to make up a skipped optimizer update.

Five hundred all-zero batches without a progressing child trigger an
operational failure requiring investigation, never a scientific convergence
decision. This guard does not modify TBSR patience. A child with at least one
verified optimizer update resets this conservative child-level guard.

## Explicit first-child recovery

No automatic legacy migration exists. The new plan must pin a
`recovery_contract` file using `{path, sha256, bytes}`. The contract contains:

```json
{
  "schema": "rmct-first-child-recovery-contract-v2",
  "destination_source_commit": "<exact newly incorporated commit>",
  "campaign_id": "<unchanged original run-name>",
  "origin_plan": {"path": "<original plan>", "sha256": "<hash>", "bytes": 0},
  "origin_started": {"path": "<original segment0 started.json>", "sha256": "<hash>", "bytes": 0},
  "audit": {"path": "<failure-6952064-audit.json>", "sha256": "<hash>", "bytes": 0},
  "checkpoint": "<exact original saved checkpoint directory>"
}
```

The displayed zero sizes and bracketed fields are placeholders, not receipts.
Do not deploy this example. The contract pins the audit and original start
command; file hashes, original source HEAD/cleanliness, complete strict
optimizer/RNG metadata, 16-batch/12-update history and original training data
are rechecked. Consumed QIDs come from the full manifest-verified setting,
not raw JSONL order. The original plan's scientific recipe and Python runtime
must remain identical; only script source and fresh preflight-attestation
paths may move. The new plan needs new source-bound gates. Live scheduler
state must confirm the original job failed before any recovered child starts.

The migration creates a `rmct-clean-recovery-v2` receipt, without modifying
the original checkpoint, source, failed receipts or audit. Subsequent child
seals use `rmct-clean-checkpoint-v2`. Both explicitly label continuation as
`optimizer_data_segment`, **not bitwise-identical vLLM sampling**.

`verify_seal_v2(receipt, plan, binding)` in `qwen_train_window.py` reproduces
the complete lineage against exact plan/source binding and current checkpoint
bytes. Build the binding with `plan_binding(plan, plan_path)`. A consumer
must also check its own current source/runtime binding. For validation, require
`step == progress.optimizer_updates`, positive and divisible by 64. The
recovery root at step 12 is not a validation target.

Receipts are exclusive-create. An existing start claim or run directory
blocks uncertain replay. A later operational interruption requires explicit
recovery; this patch does not add automatic retries. A successful
`training-complete.json` is written only at the exact optimizer boundary.

## Evidence limit

The CPU tests exercise the real metadata reader and synthetic checkpoint
files, plus the saved failure audit when locally available. They do not prove
GPU optimizer deserialization or authorize a job. Canonical integration,
source-bound gates and live recovery validation are still required. No
performance settings or scientific hyperparameters change in this patch.
