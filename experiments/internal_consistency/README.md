# Internal-consistency method experiments

`method_comparison.yaml` defines one explicit training run for each supervised
family method:

| Method | Objective | Required backend | Input rows |
|---|---|---|---|
| `bct` | Cross-entropy on completed responses | Tinker or local | `messages` |
| `act` | Residual-stream activation consistency | Local | Paired prompts |
| `attct` | Jensen-Shannon attention consistency | Local | Paired prompts |
| `mlpct` | MLP representation consistency | Local | Paired prompts |

Paired-prompt rows use `reference_messages` and `variant_messages` by default.
The final user content from the reference must occur verbatim in the variant so
the tokenizer can align their shared region. Native `mcq-bias` files instead
use `unbiased_messages` and `biased_messages`; specify those field names in the
YAML when such files are selected.

The method is selected by the `method` argument passed to
`scripts/train_bct.py`. `method_config` contains method-specific loss options
and rejects unknown keys. The current options are:

- ACT: `weight`, `layer_selection`, and `normalize`;
- AttCT: `weight`, `layer_selection`, and `layer_weights`;
- MLPCT: `weight`, `variant`, `layer_selection`, `layer_weights`,
  `distance_metric`, and `normalize`.

All three internal-consistency methods require the local backend. With LoRA,
their clean pass uses the recorded base model with the adapter disabled. With
full-parameter training, the backend retains an immutable initial-model copy
for the clean pass.

`lora_config` and `optimizer_config` expose the complete shared training
configuration. The portable LoRA fields are `rank`, `alpha`, `dropout`,
`target_modules`, `train_mlp`, `train_attn`, `train_unembed`, and `seed`.
Exact `target_modules` selection is local-only. Adam fields are `learning_rate`, `lr_schedule`, `beta1`,
`beta2`, `eps`, `weight_decay`, and `grad_clip_norm`. These objects are recorded
in the run manifest and passed to the selected backend.

## Run the matrix

Copy the YAML into a plot-specific directory, replace its data paths and
scientific settings, and preview every command:

```bash
uv run python scripts/run_experiment.py \
  experiments/internal_consistency/method_comparison.yaml \
  --dry-run
```

The runner executes training entries in their listed order. A training entry
named `act` publishes its final checkpoint as
`${training.act.checkpoint}`. Evaluation entries can therefore consume the
correct checkpoint even when the YAML contains several training runs. The
unnamed `${checkpoint}` placeholder remains invalid for a multi-training plan.
Published checkpoints are stored in
`logs/experiments/internal-consistency-method-comparison/outputs.json`, so a
later run can start from evaluation without repeating training.

Local LoRA checkpoints can be evaluated without a serving process:

```yaml
evaluation:
  - name: act-benchmark
    command: ["${python}", scripts/run_evals.py]
    args:
      task_factory: benchmark.tasks:suite
      local_checkpoint: "${training.act.checkpoint}"
      task_args:
        split: heldout
      log_dir: logs/evals/internal-consistency-method-comparison/act
      "yes": true
```

Add one evaluation entry per model and an `analysis` entry for the exact figure
script. The YAML then records the complete set of commands and inputs for that
figure. Plot-specific statistical choices remain in the analysis script or its
explicit YAML arguments rather than in the generic runner.
