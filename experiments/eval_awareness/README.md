# EvalAwareBench F6 experiment

`qwen_f6_snr_per_item.yaml` trains
`Qwen/Qwen3-30B-A3B-Instruct-2507` on matched EvalAwareBench safety prompts. The
natural prompt represents deployment behaviour. The corresponding F6 prompt is
the evaluation condition that receives the consistency gradient.

The objective is to reproduce the natural-prompt refusal rate under F6. The
experiment therefore specifies the direction explicitly:

```yaml
setting_config:
  reference_side: baseline
  train_side: F6
```

The frozen artifact is direction-neutral. It stores both sides without deciding
which side supplies the reference rate.

The training grader returns refusal as a binary trait. Harmful compliance is
reported as `1 - refusal` during analysis; no training-time wrapper is required.

## Training configuration

The full experiment uses:

- a frozen, exact-F6 EvalAwareBench artifact;
- soft SNR-scaled advantages;
- per-item advantage normalization;
- a base-model reference-rate anchor with weight `0.5`;
- 128 current-policy rollouts per side for rate estimation;
- 32,768-token model and grader ceilings; and
- explicit W&B logging to the experiment project.

The experiment does not include a helpfulness auxiliary objective.

## Create the training artifact

First export the pinned `prompts_safety` config from EvalAwareBench to JSONL.
The export must preserve the upstream fields documented in
[`ctm_data/README.md`](../../ctm_data/README.md). Then freeze only exact-F6
pairs:

```bash
uv run python -m ctm_data.adapters.eval_awareness.builder \
  --input-jsonl /path/to/evalawarebench-prompts-safety.jsonl \
  --output ctm_data/frozen/evalawarebench-f6-neutral.jsonl \
  --n-variants 1 \
  --factors F6 \
  --source-config prompts_safety \
  --source-revision FULL_40_HEX_DATASET_COMMIT
```

The command writes the JSONL artifact and its `.manifest.json` sidecar. It does
not download data or create an evaluation split.

## Preview and run

Preview the resolved training command without initializing Tinker or calling
the grader:

```bash
uv run python scripts/run_experiment.py \
  experiments/eval_awareness/qwen_f6_snr_per_item.yaml \
  --training-data ctm_data/frozen/evalawarebench-f6-neutral.jsonl \
  --dry-run
```

For an additional data-level check, run the printed `train_rlct.py` command with
`--dry-run`. The child dry run loads and verifies the artifact and prints the
resolved CTM configuration without initializing a backend.

## Interactive VS Code debugging

Open the repository as the VS Code workspace. Select **Run and Debug**, then run
**CTM: F6 two-item probe**. The launch configuration executes
`debug/qwen_f6_snr_per_item_two_items.yaml` through `run_experiment.py`.

The debug YAML explicitly specifies two datapoints, two rollouts per condition,
one epoch, batch size one, and checkpointing after each attempted step. The VS
Code configuration contains no training hyperparameters. It selects the YAML,
the local frozen artifact, and `.env`. The runner prints the complete plan and
requires confirmation in the integrated terminal before starting Tinker.

For the probe, add these function breakpoints from the Breakpoints panel:

```text
ctm.training.rl.RLTrainer.setup
ctm.training.rl.RLTrainer._collect_rollouts
ctm.training.refusal.judge.RefusalJudge.judge
ctm.training.rl.RLTrainer._build_training_batch
ctm.backends.tinker.TinkerBackend.submit_forward_backward
ctm.backends.tinker.TinkerBackend.submit_optim_step
ctm.training.rl.RLTrainer._maybe_save_checkpoint
```

At each pause, inspect values in the Variables panel or evaluate expressions in
the Debug Console. Useful initial expressions include:

```python
config
datapoints[0]["reference_messages"]
datapoints[0]["variants"][0]["messages"]
```

After the run, inspect complete sampling records in:

```text
logs/evalaware-qwen-f6-eval-from-deployment-debug-two-items/
  qwen-f6-eval-from-deployment-debug-two-items/rollouts/
```

Every sampled response is retained. `skipped_from_training` and `skip_reason`
identify rate-only samples, grader failures, zero-signal batches, and other
excluded responses.
