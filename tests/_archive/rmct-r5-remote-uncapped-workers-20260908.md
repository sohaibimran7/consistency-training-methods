# Archived RMCT r5 remote-worker AST fixture

This fixture originally parsed
`tmp/rmct-r5-remote-worker-fix-20260908/rollout_workers.py`, an unversioned
local source copy that is not part of a portable checkout. It is retained as
`test_rmct_r5_remote_uncapped_workers.py.txt` for historical reference only.

Its EOS-only `max_tokens=None` regression coverage now executes the repository
implementation in `tests/test_ctm_rollout_workers.py`: worker dispatch,
`RolloutWorkerPool` validation and transport, policy sampling, and both
worker-pool and in-process frozen-base sampling paths.
