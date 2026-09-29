# Fresh repaired-ACT tiny behavioral gate

Run this after the fresh repaired-ACT checkpoint is published and before any
long generated-answer evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b_repaired_act_tiny_behavioral_gate_20260803.yaml \
  --target repaired-act-gate \
  --checkpoint file:///absolute/path/to/repaired-act-adapter \
  --yes
```

It first verifies/recreates the attested canonical repaired-ACT split files.
It then scores a deterministic balanced four-per-dataset prefix in each split
through native Transformers/PEFT. The resulting immutable attestation binds
the raw adapter hash, frozen input hashes, selected question IDs, and report
hash. It fails closed unless the train split has at least one base toward-bias
switch and the adapter has a strictly lower direct TBSR.

The scorer is in the `evaluation` stage and the attester is in the subsequent
ordered `analysis` stage. This is deliberate: even when the runner is invoked
with `--parallel`, its stage barrier prevents the attester racing a partially
written direct-answer report.

This is a cheap checkpoint sanity gate. It is not a substitute for the full
generated-answer metric, Luna verbalisation scoring, or the separate Qwen3.5
HF/vLLM adapter-parity attestation required for a vLLM evaluator.

For the fresh no-CoT repaired-ACT trajectory, do not launch a long evaluation
directly after this command. First use
[`scripts/ctm_repaired_act_long_eval_guard_20260803.sh`](../../../scripts/ctm_repaired_act_long_eval_guard_20260803.sh),
which writes (or revalidates) the chain attestation binding the target-scoped
training output, this passed tiny gate, and the vLLM compatibility/parity
evidence. Pass that chain attestation again to the no-CoT raw preflight after
generation.
