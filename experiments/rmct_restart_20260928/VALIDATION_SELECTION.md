# Shared TBSR selection interface

`validation_selection.py` is the model-independent selection component. Model
adapters still own native rendering, completion parsing, exact response coverage,
actual runtime/scheduler evidence and strict optimizer/RNG checkpoint verification.
The shared component invokes both adapter verifiers every time it ingests or
replays a receipt. Its CPU tests do not establish a working Gemma validator.

## Immutable inputs

All input references use `qwen_checkpoint.identity(path)`:
`{path, sha256, bytes}`. Originals remain unchanged.

The contract schema is `ctm-tbsr-selection-contract-v1` with `campaign_id`,
`method`, `model`, `source_commit`, `population` (frozen JSON identity),
`response_count=600`, `pair_count=400`, model-appropriate `settings` containing
`enable_thinking=true`, and `policy=validation_selection.POLICY`. Policy is
TBSR, every 64 actual updates, strict decrease, earliest ties, patience two and
maximum 4096 actual updates. Model adapters must verify the full settings and
200-question/600-prompt population, not infer correctness from these counts.

Normalized progress schema is `ctm-training-progress-v1`, with the same four
identity fields and `actual_optimizer_step`, `next_attempt_index`,
`sampled_batches` (integer or null where sampling is separately logged),
`checkpoint` (directory), and `checkpoint_files` (map of file identities).
Adapters also preserve original RNG/data-order/consumed-QID evidence. The
checkpoint verifier must reproduce that original evidence and return the exact
normalized progress; a schema conversion alone is insufficient.

The validation reference points to an immutable JSON artifact from the
model-specific validation workflow. `verify_validation(record, progress,
contract)` rechecks all saved requests and responses, native template/parameters,
actual checkpoint use and scheduler completion. It returns reproduced metrics:
`step`, `campaign_id`, `response_count`, exact `checkpoint_files`,
`towards_switches`, `eligible_pairs`, `tbsr`. Invalid/truncated responses remain
excluded explicitly in the model-specific score/coverage artifact. Complete
response coverage is still required; an invalid completion cannot become a
successful negative.

## Calls and persistence

Before the first checkpoint exists, `bootstrap_budget(contract_record,
start_record, requested_updates=..., verify_start=...)` requires an immutable
`ctm-training-start-v1` record with the same campaign/method/model/source,
`contract`, zero actual updates/next attempt, `resume_from=null`, and
`optimizer=fresh`. The adapter verifies original base/data/native gate evidence
and the exclusive scheduled-start claim. It grants at most 64 actual updates;
later windows use checkpoint progress and accepted validation receipts.

1. `accept_validation(folder, contract_record, progress_record,
   validation_record, verify_checkpoint=..., verify_validation=...)` verifies
   all evidence before creating an exclusive `validation/step-NNNNNN.json`.
   Repeating the identical call returns the replayed state without counting a
   checkpoint twice. A conflicting receipt fails. A failed native validator
   never creates an acceptance receipt.
2. `replay(entries_from_folder(folder), contract_record, ...)` reproduces the
   decision from the same verified artifacts. Steps must be exactly 64, 128,
   and so on; the consumed-attempt cursor must increase. Exact rational TBSR
   comparisons keep the first checkpoint on ties. Two nonimprovements stop
   the run; receipt ingestion after this first stopping event fails.
3. `continuation_budget(progress, state, requested_updates=...)` grants at most
   the actual updates remaining to the next boundary. Call it with freshly
   checkpoint-verified progress and replayed state. At an unvalidated boundary
   it raises; after a stopping decision it returns zero. Skipped attempts do
   not consume the optimizer budget. Training adapters must independently bound
   each attempted slice and rederive progress after every saved child.
4. `selected_evaluation_manifest(state)` returns the selected checkpoint/file
   identities and original validation reference, independently of the latest
   checkpoint. Selection is `provisional` until the run stops and `terminal`
   afterward. Evaluation consumers must reverify the referenced checkpoint.

## Qwen saved-state load probe

`qwen_resume_load_gate.py` wraps the actual production CLI/backend setup,
constructs AdamW and explicitly loads its staged state without an optimizer
step. Every GPU compares the materialized optimizer state digest with the
saved state. Replicated setup verifies adapter equality and restores each
rank RNG; the probe reads those states back and also verifies coordinator RNG.
Generation, backward, optimizer steps and checkpoint writes are prohibited.
The original checkpoint is verified again after the probe.

For the explicit 16-batch/12-update recovery, the controller requires
`--resume-load-gate <completed receipt.json>`. It checks exact source/plan,
parent and prerequisite receipts, four distinct GPUs, equal optimizer digests,
RNG evidence, zero training/generation operations and scheduler completion.
This receipt is diagnostic and cannot serve as a new production checkpoint.
The probe must run from the incorporated source with fresh same-source gates.
