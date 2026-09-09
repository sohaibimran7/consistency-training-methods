# No-CoT Stage 1 IID diagnostic

This is the final-Qwen3.5 replacement for the historical `encourage_cot`
diagnostic.  It is deliberately separate: it accepts only the approved
content-level legacy-G4 recovery (SHA-256
`7d113ee1858426721d09b23a78f4bbb0e9b16e7576b3ee5ab7da4924c3a0ef3b`) and
requires its recovery manifest on every task-factory call.

It freezes the same populations, without a shuffle:

- `train_eval`: source rows 1–200, 100 LogiQA + 100 HellaSwag;
- `heldout_in_domain`: source rows 2,049–2,248, again 100 + 100;
- `rmct_first64`: rows 1–64, recorded for RMCT/RMCT-control subset analysis.

Preparation is CPU-only and does not call a model or remote service:

```bash
python -m experiments.stage1_iid_diagnostic_none.prepare \
  --source artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.jsonl \
  --source-manifest artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/data/distractor-argument-pairs.manifest.json \
  --output-dir artifacts/stage1-iid-diagnostic-none-20260801
```

It creates `train-eval-n200.jsonl`, `heldout-in-domain-n200.jsonl`, and
`manifest.json` under that output directory.  Existing outputs are never
overwritten.  The manifest records every selected ID/hash and source offset;
task construction revalidates the full recovered source and verifies each
split's bytes against those exact source rows.

Use `experiments.stage1_iid_diagnostic_none.tasks:diagnostic_tasks` for model
evaluation.  Each split yields clean LogiQA, clean HellaSwag, biased LogiQA,
and biased HellaSwag in that order.  Keep a distinct no-CoT raw-log root so
the historic and final populations cannot be mixed.  The pinned evaluator
settings are in `config.py`; they use `max_tokens=20480` and intentionally do
not set Qwen's `enable_thinking=false`—`prompt_style: none` removes the
explicit user CoT instruction while retaining the reasoning model's native
behaviour.

Before moving any raw responses to the Luna staging root, produce the matching
no-CoT preflight report on the generation host. It rejects old-CoT task
headers, old source digests, changed split files, duplicate IDs, or a missing
biased cell:

```bash
python -m experiments.stage1_iid_diagnostic_none.raw_preflight \
  --raw-log-root logs/evals/CONDITION/iid-none \
  --manifest artifacts/stage1-iid-diagnostic-none-20260801/manifest.json \
  --split-file train_eval=artifacts/stage1-iid-diagnostic-none-20260801/train-eval-n200.jsonl \
  --split-file heldout_in_domain=artifacts/stage1-iid-diagnostic-none-20260801/heldout-in-domain-n200.jsonl \
  --condition CONDITION \
  --expected-base-model Qwen/Qwen3.5-9B \
  --expected-checkpoint /absolute/path/to/verified-vllm-compat-adapter \
  --runtime-profile vllm \
  --output artifacts/stage1-iid-none-logs/CONDITION/raw-preflight.json
```

### Fresh repaired-ACT launch chain

For the fresh no-CoT repaired-ACT checkpoint, the supported long-evaluation
entry point is the CPU-only wrapper below, not a direct evaluator command. It
creates or revalidates a write-once chain attestation *before* it `exec`s the
long evaluator: target-scoped 4,000-step training output → raw adapter →
passed tiny native-HF gate → translated vLLM adapter and its live parity
attestation.

```bash
export CTM_REPAIRED_ACT_TRAINING_OUTPUT_STATE=.../logs/experiments/rmct_paper_vast_dense_qwen3_5_9b_stage1_supervised_recovery_none_calibration_20260803/targets/repaired-act/outputs.json
export CTM_REPAIRED_ACT_RAW_CHECKPOINT=file:///absolute/path/to/fresh-repaired-act-adapter
export CTM_REPAIRED_ACT_TINY_GATE_ATTESTATION=.../repaired-act-tiny-hf-behavioral-gate/attestation.json
export CTM_REPAIRED_ACT_VLLM_COMPAT_ADAPTER=/absolute/path/to/repaired-act-vllm-compat-adapter
export CTM_REPAIRED_ACT_CHAIN_ATTESTATION=.../repaired-act-long-eval-chain.json

bash scripts/ctm_repaired_act_long_eval_guard_20260803.sh -- \
  LONG_EVALUATOR_COMMAND [ARGS...]
```

The subsequent no-CoT raw preflight must carry the same attestation:

```bash
  --repaired-act-chain-attestation "$CTM_REPAIRED_ACT_CHAIN_ATTESTATION"
```

That extra handoff revalidates the chain after generation and binds the
preflight's served vLLM adapter identity to it. It is intentionally optional
for historical/non-ACT conditions so their existing provenance contracts stay
unchanged.

Copy only the four hash-bound `.eval` files named by that report into a
separate staging layout `RAW_ROOT/CONDITION/<split>/`. Then use the no-CoT
grader wrapper, not the generic historical entry point:

```bash
python -m experiments.stage1_iid_diagnostic_none.grade_luna \
  --raw-log-root artifacts/stage1-iid-none-logs \
  --output-root artifacts/stage1-iid-none-luna \
  --preflight-report artifacts/stage1-iid-none-logs/CONDITION/raw-preflight.json \
  --workers 5 --connections-per-worker 100
```

The wrapper validates the report's source digest, prompt style, four cells,
and absence of an alternate prompt file before it invokes the existing
500-connection-capped GPT-5.6 Luna grader. It therefore cannot silently grade
a historical-CoT preflight report.

For an adapted `--runtime-profile vllm` checkpoint, preflight also verifies
the live Qwen3.5 HF/vLLM parity attestation and writes immutable hashes for
the served adapter weights/config, compatibility-translation manifest, parity
attestation, and raw-source adapter weights into
`contract.vllm_compatibility_adapter_identity`. The final matrix accepts an
adapted vLLM condition only with that complete identity record. The untrained
vLLM baseline has no local adapter checkpoint and is intentionally exempt.

## Accelerated RMCT-control report canonicalization (offline)

If the accelerated native-HF evaluator is published under the operational
condition `rmct-control-b8-accelerated`, do **not** rename or rewrite its raw
or Luna-graded EvalLogs.  Recompute a separate paper-facing `rmct-control`
analysis from those immutable logs instead:

```bash
python -m experiments.stage1_iid_diagnostic_none.canonicalize_rmct_control \
  --alias-analysis artifacts/stage1-onpolicy-final/analysis/rmct-control-b8-accelerated-final.json \
  --graded-root artifacts/stage1-onpolicy-final/luna \
  --manifest artifacts/stage1-iid-diagnostic-none-20260801/manifest.json \
  --output artifacts/stage1-onpolicy-final/analysis/rmct-control-final.json \
  --provenance-output artifacts/stage1-onpolicy-final/analysis/rmct-control-canonicalization.json
```

The tool only reads the input logs.  It checks that all four graded cells and
their ordered IDs match the frozen manifest, re-extracts the cells, recomputes
the `rmct_first64` training-prefix subset from manifest IDs, and refuses to
publish unless the alias report's pooled/per-dataset values reproduce exactly
apart from the condition label.  Report/provenance outputs must be outside the
graded-log root and are write-once (an identical rerun resumes).

## Final nine-condition figure adapter (offline)

Once all nine final analysis reports exist, make the one plot-ready 18-cell
adapter with the offline merger.  It reads immutable analysis reports *and*
their raw-preflight reports: the latter are needed to prove that every graded
raw-log hash came from the frozen no-CoT source, split bytes, and per-dataset
question IDs.  It does not read or modify EvalLogs, call Inspect, call Luna,
or launch a model.

```bash
python -m experiments.stage1_iid_diagnostic_none.merge_final_matrix \
  --report ANALYSIS_WITH_BASE_ACT_ATTCT_MLPCT.json \
  --report ANALYSIS_WITH_BCT_AND_BCT_CONTROL.json \
  --report OPCT_ANALYSIS.json \
  --report RMCT_ANALYSIS.json \
  --report RMCT_CONTROL_CANONICAL_ANALYSIS.json \
  --preflight BASE_PREFLIGHT.json \
  --preflight ACT_PREFLIGHT.json \
  --preflight ATTCT_PREFLIGHT.json \
  --preflight MLPCT_PREFLIGHT.json \
  --preflight BCT_PREFLIGHT.json \
  --preflight BCT_CONTROL_PREFLIGHT.json \
  --preflight OPCT_PREFLIGHT.json \
  --preflight RMCT_PREFLIGHT.json \
  --preflight RMCT_CONTROL_ACCELERATED_PREFLIGHT.json \
  --population-manifest artifacts/stage1-iid-diagnostic-none-20260801/manifest.json \
  --rmct-control-canonicalization RMCT_CONTROL_CANONICALIZATION.json \
  --output artifacts/stage1-iid-final/analysis/final-matrix.json \
  --provenance-output artifacts/stage1-iid-final/analysis/final-matrix-provenance.json

python -m experiments.stage1_iid_diagnostic.plot \
  --analysis artifacts/stage1-iid-final/analysis/final-matrix.json \
  --output-dir artifacts/stage1-iid-final/figures
```

The merger accepts the attested operational names (such as `base-vllm` and
`repaired-act-vllm-compat`) but produces only the chart's canonical conditions:
`untrained`, `act`, `attct`, `mlpct`, `bct`, `bct-control`, `opct`, `rmct`, and
`rmct-control`.  It requires exactly two 200-sample cells for each condition;
each cell must retain its two 100-sample per-dataset sources.  It also retains
source report/preflight SHA-256s and every source-cell provenance inside the
new adapter and sidecar.  If RMCT-control was not published from the
accelerated operational alias, omit `--rmct-control-canonicalization`.
