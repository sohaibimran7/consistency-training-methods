# Training data and adapters

`ctm_data/adapters/` contains benchmark-specific training adapters. Each adapter
translates an external or local dataset schema into CTM's generic `Setting`
protocol or fixed prompt-family schema. The generic `ctm/` package does not
import these adapters or their dependencies.

Use these gitignored directories for data payloads:

- `ctm_data/local/` for source exports or private local datasets;
- `ctm_data/frozen/` for generated JSONL training artifacts and manifests.

Store, copy, and archive each generated JSONL file together with its
`.manifest.json` sidecar. The loader verifies both files before training.
Builders refuse to overwrite existing artifacts; use a new output path when the
source revision, source rows, seed, factor selection, or variant count changes.

## Native mcq-bias files

The `mcq_bias` adapter reads native frozen rows without renaming their fields or
generating a CTM-specific copy:

```yaml
setting_factory: ctm_data.adapters.mcq_bias:create_setting
setting_config:
  data_paths:
    - /absolute/path/to/suggested-answer.jsonl
    - /absolute/path/to/wrong-few-shot.jsonl
load_config:
  path_limits:
    /absolute/path/to/suggested-answer.jsonl: 200
    /absolute/path/to/wrong-few-shot.jsonl: 100
```

When `path_limits` is omitted, `n_datapoints` is divided as evenly as possible
across the selected files. The loader raises if any requested count cannot be
satisfied. Rows with `bias_type="are_you_sure"` are rejected because they
require staged multi-turn generation, which this training path does not
implement.

For ACT, AttCT, or MLPCT, select the native pair fields explicitly:

```bash
uv run python scripts/train_bct.py \
  --backend local \
  --method act \
  --data /absolute/path/to/native-mcq-bias.jsonl \
  --reference-messages-field unbiased_messages \
  --variant-messages-field biased_messages \
  --experiment-name mcq-bias-act \
  --run-name main
```

## WildJailbreak families

The WildJailbreak builder expects JSONL rows containing:

- `data_type`, equal to `adversarial_harmful` or `adversarial_benign`;
- `vanilla`, the reference request;
- `adversarial`, the related jailbreak request; and
- `tactics`, represented as a list or a Python-list string.

Freeze exactly the supplied rows:

```bash
uv run python -m ctm_data.adapters.wildjailbreak.builder \
  --input-jsonl /absolute/path/to/wildjailbreak-export.jsonl \
  --output ctm_data/frozen/wildjailbreak-k4.jsonl \
  --n-variants 4 \
  --seed 42
```

Every vanilla request must have at least `n_variants` distinct adversarial rows.
The builder raises instead of silently dropping incomplete families. Source
completion fields are ignored. The resulting setting uses refusal as the
training trait for harmful and benign families:

```yaml
setting_factory: ctm_data.adapters.wildjailbreak:create_setting
setting_config:
  family_path: ctm_data/frozen/wildjailbreak-k4.jsonl
  n_variants: 4
  grader_model: openrouter/openai/gpt-oss-120b:nitro
```

## EvalAwareBench families

The EvalAwareBench builder expects a JSONL export with these upstream fields:

`task_id`, `task_name`, `valence`, `description`, `factors_varied`,
`num_factors_varied`, `config`, and `prompt`.

`source_revision` must be the full 40-character commit of the exported dataset.
For example, freeze matched baseline/F6 safety families with:

```bash
uv run python -m ctm_data.adapters.eval_awareness.builder \
  --input-jsonl /absolute/path/to/evalawarebench-prompts-safety.jsonl \
  --output ctm_data/frozen/evalawarebench-f6-neutral.jsonl \
  --n-variants 1 \
  --factors F6 \
  --source-config prompts_safety \
  --source-revision FULL_40_HEX_DATASET_COMMIT
```

The artifact is direction-neutral. It stores the natural baseline as the
canonical reference and labels the selected factor side. The experiment must
state the scientific direction explicitly:

```yaml
setting_factory: ctm_data.adapters.eval_awareness:create_setting
setting_config:
  data_path: ctm_data/frozen/evalawarebench-f6-neutral.jsonl
  n_variants: 1
  reference_side: baseline
  train_side: F6
  grader_model: openrouter/openai/gpt-oss-120b:nitro
```

Reversing `reference_side` and `train_side` is supported for a single variant.
Builders and settings do not create or reserve an evaluation split.
