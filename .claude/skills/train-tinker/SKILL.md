---
name: train-tinker
description: Preview and launch CTM SFT, representation-consistency, RL, or YAML-defined experiment runs with explicit data, backend, logging, and approval settings.
---

# Train CTM models

Use the repository's generic training entry points. Specify benchmark-specific
data through explicit paths and adapter factories.

## Safety gate

Before a paid run:

1. Collect the model, exact data paths, experiment/run names, backend, and
   non-default hyperparameters.
2. Run the applicable RL dry run or experiment preview. For SFT, print the
   command and rely on the CLI's pre-training confirmation.
3. Show the exact command and configuration to the user.
4. Launch only after the user approves, or by passing `-y` when approval was
   already explicit.

## Preferred: experiment config

```bash
uv run python scripts/run_experiment.py experiments/example_rlct.yaml \
  --training-data /absolute/path/to/train.jsonl
```

The config owns which adapter provides training data and which upstream task
factory provides each evaluation. `${checkpoint}` may appear in evaluations
only when the selected config has a single training stage.

## RLCT

```bash
uv run python scripts/train_rlct.py \
  --setting-factory package.adapter:create_setting \
  --setting-config '{"data_paths":["/absolute/path/to/train.jsonl"]}' \
  --experiment-name EXPERIMENT --run-name RUN \
  --dry-run
```

Important current defaults:

- `--backend tinker`
- `--normalization per_item`
- `--anchor-weight 0.5`
- `--anchor-model base`
- `--n-ref-rollouts 128` and `--n-train-rollouts 128`
- complete rollout persistence under `logs/<experiment>/<run>/rollouts/`
- no W&B logging unless `--wandb-project` is supplied

Use `--load-config` for adapter loading options such as `path_limits`. Training
data selection belongs to the setting configuration rather than generic RLCT
arguments.

## SFT / representation consistency

```bash
uv run python scripts/train_bct.py \
  --method bct \
  --data /absolute/path/to/train.jsonl \
  --experiment-name EXPERIMENT --run-name RUN
```

`act`, `attct`, and `mlpct` require the local backend and generic pair fields by
default. Supply method-specific loss settings through `--method-config` or the
corresponding YAML argument. Native mcq-bias rows must map their fields
explicitly:

```bash
uv run python scripts/train_bct.py \
  --backend local --method act \
  --data /absolute/path/to/native-mcq-bias.jsonl \
  --reference-messages-field unbiased_messages \
  --variant-messages-field biased_messages \
  --experiment-name EXPERIMENT --run-name RUN
```

Use `--resume-from` to restore weights. Add `--resume-with-optimizer` only when
the run must continue with the saved optimizer state. Consult `--help` for the
complete current interface and resolved defaults.

For a multi-method paper comparison, start from
`experiments/internal_consistency/method_comparison.yaml`. Named training
results are referenced as `${training.NAME.checkpoint}` and persisted under
`logs/experiments/<experiment>/outputs.json` for later evaluation stages.
