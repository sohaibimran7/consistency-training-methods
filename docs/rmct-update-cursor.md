# RMCT optimizer updates and data cursor

The first merged-source production segment consumed 16 two-QID batches but
performed 12 optimizer updates: four batches had no training signal. Its final
checkpoint retains `global_step=16`, `step=16`, `optimizer_step=12`, no accumulated
gradients, and final segment state. These counters must not be relabelled.

The core RL loop already records these quantities separately. The restart
controller incorrectly required equality. The repair must keep two independent
coordinates:

- **sampled batches** locate the next two QIDs in the frozen order;
- **optimizer updates** determine the exact 64-update validation boundary.

`SharedQidTwoBiasSetting.load_datapoints` now accepts optional `batch_offset`
and `batch_count`. The entire manifest/data artifact and original segment width
remain validated; `n_datapoints=32` still refers to that verified full segment.
The returned subset contains exactly twice `batch_count` QIDs in the original
LogiQA/HellaSwag order. Run metadata records the selected IDs/hash and sampled
batch start/end. Defaults preserve full-segment loading.

A controller can request at most the smaller of the remaining data segment and
remaining optimizer updates. Every batch yields at most one optimizer update
under this campaign's accumulation=1 contract, so this cannot overshoot the
validation boundary. A zero-signal batch advances the data cursor, not the
optimizer counter. The next child resumes from the real saved counters.

Checkpoint migration/recovery requires an explicit source/command/file-hash
audit, strict optimizer and rank/coordinator RNG state, and a versioned seal.
Presence of a checkpoint alone is not resume clearance. Existing weights and
all evidence are retained; no failed attempt is silently replayed.

Continuation is `optimizer_data_segment`, not bitwise-uninterrupted sampling:
vLLM's private sampling RNG is not serialized by the existing backend contract.
Any operational no-progress pause must be reported separately from scientific
convergence. This document does not itself authorize a launch or declare the
controller repair complete.

## Local integration evidence (30 September 2026)

The active controller now implements v2 lineage verification and bounded slices.
Validation preparation calls the same full verifier before adapter translation
or GPU work, checks the current source/interpreter/model binding, and rejects
anything other than a positive actual optimizer count divisible by 64.
The original 12-update recovery root cannot be validated as step 16 or 64.

The expanded deployed regression inventory passed locally: 234 passed, one
skipped. The skipped case requires the immutable failed-job audit, which is not
present in this canonical checkout. The focused restart/loader/patience suite
passed 74 tests with the same skip. These runs used the local project Python
and the pinned mcq-bias source; they are not deployed-runtime receipts.

Remaining gates: independently review the incorporated source; publish and
incorporate its exact commit; pin the actual failure audit/original plan/start
receipt into an explicit recovery contract; rerun the same-source regression,
native preflight and disposable RL gates in the deployed environment; verify
strict GPU optimizer/RNG deserialization and the saved 16/12 counters before
continuing with the next unconsumed batch. No checkpoint bytes were changed by
this integration. Only the designated Qwen submitter may launch recovery.
