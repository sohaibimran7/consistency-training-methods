# Five-method two-bias convergence comparison

This dedicated experiment uses the frozen expanded LogiQA/HellaSwag pool. It
does not invoke the legacy experiment factory, generate more wrong arguments,
add Alpaca, or inherit implicit generation defaults. BCT/OPCT training now uses
the user-approved 20,480-generated-token cap matching RMCT; evaluation settings
and existing internal-method runs are unchanged.

## Approved prefix amendment and startup repair

The initial full-pool native-prompt audit found 5,733 suggested-answer rows
that fail the required complete-clean-suffix contract, including 3,784 rows
that cannot align the complete clean user message at all. All 7,680
wrong-argument rows pass. Suggested-answer cues are placed before, within, or
after the question by the original injector.

The user explicitly approved prefixing for ACT/AttCT/MLPCT. The v2 training
view now places the **verbatim suggested-answer cue**, two newlines, then the
**complete unchanged clean question** in the final user message. Wrong-argument
prompts are untouched. BCT/OPCT use the original shared-pool prompts; evaluation
artifacts and prompts are never modified. The shared source remains byte-identical.

The three internal methods are blocked before model loading unless a passing
full-pool audit for `suggested_answer_prefix_internal_v1` is supplied. Old native
audits and the earlier unapproved candidate diagnostic cannot satisfy this gate.

The first BCT/OPCT attempts (`6447612`, `6447613`) failed during model loading,
before optimizer updates, because the runner selected a Muse-only multimodal
detachment option. The v2 runner removes that option: Qwen loads through the
ordinary causal-LM loader. Failed attempts and their source bundle are retained.
New jobs use an isolated v2 bundle and run root; no running source is overwritten.

The v2 BCT attempt (`6448996`) subsequently failed at its post-generation EOS
check, before an optimizer update. The HF sampler accepts both renderer/chat
EOS and generation-config EOS, while the experiment validator mistakenly
accepted only the latter. OPCT shares that check; the three internal methods
do not sample completions during training. The repair factors the sampler's
unchanged EOS resolution into one helper used by both sampling and validation.
Exact completion tokens, including the emitted terminator, are retained; no
token cap, new stop token, or generation-policy change is introduced. Repaired
BCT/OPCT use a fresh immutable bundle; progressing v2 internal runs are untouched.

## Scientific contract

The user requested matching RMCT's "20k" cap. Its deployed value is exactly
20,480 generated tokens (reasoning plus final answer, excluding the prompt).
The capped amendment accepts either real EOS or a length stop at that boundary.
Length-stopped sequences remain in training without synthetic EOS or resampling;
BCT cache and OPCT rollout records identify the termination reason. The existing
temperature-only cached sampler is retained with an explicit length check, so
this change does not activate default top-k processors or retain a full-vocabulary
score tensor per token. The new option is opt-in; other backend callers retain
their existing behavior. New BCT/OPCT runs use a separate immutable source root.

| Setting | This comparison | Relation to RMCT step 176 |
|---|---|---|
| Base | Qwen3.5-9B, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Same |
| Data | 3,840 LogiQA + 3,840 HellaSwag QIDs, each with both biases | Expanded; protected evaluation QIDs excluded |
| Semantic batch | 2 QIDs, both biases per QID, 4 equally weighted backwards/update | Same QID/bias grouping |
| Ordering | Frozen per-dataset permutation, alternating LogiQA/HellaSwag | Same algorithm; no epoch shuffle |
| Suggested-answer cue placement | Prefix for ACT/AttCT/MLPCT training only; original for BCT/OPCT | Explicitly approved internal-method formatting difference |
| LR | `1e-4`, constant, no warmup or decay | Same |
| Adam | betas `.9/.95`, epsilon `1e-8`, weight decay `0`, clip norm `1` | Same |
| LoRA | rank `8`, alpha `16`, dropout `0`; embeddings/head frozen | Same shared parameters |
| Stop | Max 4,096 updates; 8 complete non-improving 16-update windows | Method-loss plateau; not RMCT's rate-gap threshold |
| Epochs | May stop before one pass; repeats only after update 3,840 | No artificial one-epoch stop |
| Instruction replay | None | Same no-replay decision |
| Training output length | EOS or 20,480 generated tokens, including reasoning | Matches RMCT's approved cap and retained length-stop handling |

There were 7,892 accepted wrong arguments. The balanced, 16-QID-per-dataset
segment realization uses 7,680; the other 212 remain in the generation journal.
Nothing is deleted and no more wrong arguments are requested.

## Method-specific settings

| Method | Trainable LoRA scope | Objective / method-specific differences |
|---|---|---|
| BCT | All attention + MLP | Cross-entropy on one frozen-base clean completion per QID, reused identically for both biases. Exact sampled tokens retain reasoning and EOS when emitted. |
| ACT | Ordinary Q/V + DeltaNet fused `in_proj_qkv` | All-layer residual MSE, unnormalized. Explicit hybrid approximation to paper Q/V: fused K also changes; MLP frozen. |
| AttCT | Ordinary Q/V | All available full-attention JSD, uniform layer weights. DeltaNet does not expose ordinary attention matrices. |
| MLPCT | Ordinary Q/V; MLP frozen | Hidden MLP-state cosine distance, all layers, uniform weights, no normalization. The objective observes MLP states; it does not imply training MLP parameters. |
| OPCT | All attention + MLP | Historical 4 rollouts/bias prompt, temperature `.7`, reverse-KL coefficient `2`, discount `.9`, importance-sampling loss. Teacher is always the frozen base on the clean prompt, including resumed jobs. |

The shared optimizer is checked against the original RMCT compiler in tests.
The scoped ACT gate is explicit: the historical broad-attention gate remains
unchanged unless `act_qv_fused_qkv` is selected. This is not presented as a
slice-exact paper Q/V implementation.

The stopping statistic is the arithmetic mean of the four variant metrics,
then the mean of 16 updates. A strict new global best resets patience. No
common absolute loss threshold is imposed on incomparable objectives. OPCT
uses its sampled per-token reverse-KL estimate, not its signed surrogate
optimization loss. These are training-stream curves, not held-out validation
curves; noisy batches can affect plateau detection.

## Implementation and validation

- `plan.py`: pinned pool, exact configuration, immutable source hashes,
  grouped ordering, and pure patience controller.
- `train.py`: existing CTM losses/backend, four backwards per update, cached
  explicitly capped BCT targets, OPCT frozen-base teacher, sealed optimizer continuation.
- `audit.py`: every pair checked with the pinned tokenizer; no generation.
- `package.py`: allowlisted source/data bundle for a fresh remote directory.
- `infra/isambard/run_methods_two_bias_convergence.sbatch`: one GPU per method,
  128-update job slices, state-driven continuation after successful exit.

The single-GPU HF execution path is an implementation choice, not a claim of
RMCT throughput parity. All five methods can run independently. Checkpoints
are saved every 16 updates with optimizer, loss-window evidence and hashes.
A failed job is not automatically retried. Slurm's advance signal requests a
stop at the next 16-update boundary; the hard wall time may interrupt a long
generation before that boundary, leaving the previous checkpoint as
the safe resume point.

Per-update seeds make resumed sampling deterministic at an optimizer boundary.
Two simultaneous jobs cannot modify one method's run directory. Resume verifies
checkpoint bytes and plan identity; it never resets optimizer steps, patience,
or the OPCT teacher. Unsealed attempt logs are retained separately from sealed
window metrics. Loss curves should read checkpoint `window-metrics.json` files
referenced by receipts, not concatenate failed attempt logs.

## Prepare and launch

From the source worktree, using its dependency-complete Python environment:

```bash
python -m experiments.methods_two_bias_convergence.audit --output artifacts/methods-two-bias-convergence-20260910/prefix-approved-v2/alignment.json
python -m experiments.methods_two_bias_convergence.plan prepare --plan artifacts/methods-two-bias-convergence-20260910/prefix-approved-v2/plan.json
python -m experiments.methods_two_bias_convergence.package --plan artifacts/methods-two-bias-convergence-20260910/prefix-approved-v2/plan.json --alignment-audit artifacts/methods-two-bias-convergence-20260910/prefix-approved-v2/alignment.json --output artifacts/methods-two-bias-convergence-20260910/prefix-approved-v2/source-data.tar.gz
```

Deploy that bundle into a **new** Isambard directory. Do not synchronize over
an existing experiment checkout. Bind `CTM_METHODS_PYTHON` to the established
Qwen training environment after checking its installed package versions, and
verify the offline base snapshot. Set `REPO_DIR` to the isolated source,
`CTM_METHODS_PLAN` to its `campaign-plan.json`, and `CTM_METHODS_RUN_ROOT` to a
new campaign output directory. Record each returned Slurm job ID:

For the three internal methods, also set `CTM_METHODS_ALIGNMENT_AUDIT` to the
bundled `alignment-audit.json` for the exact approved prompt transformation.
Native BCT/OPCT do not consume that transformed training view.

```bash
for CTM_METHOD in bct act attct mlpct opct; do
  export CTM_METHOD
  sbatch --parsable --job-name="ctm-two-bias-$CTM_METHOD" "$REPO_DIR/infra/isambard/run_methods_two_bias_convergence.sbatch"
done
```

Submit scheduled jobs before interactive GPU debugging. Do not cancel unrelated
jobs. The wrapper queues successors only when a successfully completed slice
has a sealed `continue` decision. It stops chaining on plateau or 4,096 updates.

## Evaluation handoff

No evaluation result is implied by preparing or submitting training. Evaluate
the terminal checkpoint using the exact RMCT-176 populations, with **16 GPUs**,
the standard switch-rate/verbalisation pipeline and paired significance, plus
AITA-NTA-FLIP with the final-output-only parser and uncapped sampling. Do not
reuse the historical methods launcher's hard-coded adapter hashes as if these
were the old adapters. Register the newly sealed checkpoints with their new
hashes before using that evaluation infrastructure. The current training
wrapper does not automatically submit evaluation jobs.
