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

## Irpan et al. (`2510.27062`) paper suite

`ctm_data.adapters.irpan_2510_27062` is the offline, provenance-first data path
for *Consistency Training Helps Stop Sycophancy and Jailbreaks*. It is distinct
from the fixed-family WildJailbreak RLCT adapter documented below. The paper
suite assigns these roles:

| Source | Paper use | Adapter route |
| --- | --- | --- |
| ARC | sycophancy training | clean MCQ plus deterministic incorrect user suggestion |
| OpenBookQA | sycophancy training | clean MCQ plus deterministic incorrect user suggestion |
| BIG-Bench Hard | sycophancy training | clean MCQ plus deterministic incorrect user suggestion |
| MMLU | sycophancy/capability evaluation | clean accuracy and wrong-suggestion tasks |
| HarmBench | jailbreak training and safety validation | local harmful-request export; clean/wrapped vulnerability filter |
| OR-Bench | helpfulness validation | answered-benign rate |
| ClearHarm | final safety evaluation | attack success rate |
| WildGuardTest | final safety evaluation | human-labelled adversarial-harmful rows; attack success rate |
| XSTest | final helpfulness evaluation | answered-benign rate for this paper |
| WildJailbreak | final helpfulness evaluation | `adversarial_benign`; answered-benign rate |

The paper does not release exact source revisions, splits, sycophancy prompt
template, jailbreak wrapper catalogue, judge prompt/parser, or bootstrap seed
and replicate count. The adapter therefore records each of those as a
reconstruction choice instead of presenting it as paper-authored. All source
imports take explicit local files, and every derived JSONL has an immutable
manifest, stable example IDs, content hashes, parent hashes, configuration
hashes, and producer-code hashes. Imports, task construction, filtering, and
dry-runs never acquire data or call a model.

Training manifests are role-bound. A canonical clean/wrapped pair view feeds
ACT, AttCT, MLPCT, OPCT, and the paper-specific RMCT settings; BCT instead
requires a separately verified fresh-target artifact. Evaluation-role rows are
rejected by every training path, and the reconstructed HarmBench training and
validation partitions must have disjoint stable IDs.

WildGuardMix/WildGuardTest and WildJailbreak are gated by AI2 terms. Accept the
upstream terms yourself, export the selected rows locally, and keep them under
`ctm_data/local/` or another gitignored path. The adapter will fail with an
acquisition URL when a local export is missing; it does not bypass either gate
or redistribute the rows. See the paper runbook at
`experiments/paper_reproductions/irpan_2510_27062/README.md` for the artifact
DAG, selection boundary, and smoke/full experiment plans.

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

Materialize a balanced training file without running an evaluation:

```bash
uv run python -m ctm_data.adapters.mcq_bias.materialize \
  --bias-type wrong_argument \
  --datasets logiqa hellaswag \
  --n-questions 250 \
  --dataset-dir artifacts/mcq_bias/train \
  --output artifacts/data/wrong-argument-pairs.jsonl \
  --manifest-output artifacts/data/wrong-argument-pairs.manifest.json
```

The command asks the public `mcq_bias` task constructor to materialize each
native frozen file, then round-robin interleaves those files into the explicit
training output. Interleaving prevents a smaller global training prefix from
silently containing only the first dataset.

`mcq_bias` does not generate BCT responses. CTM samples them through its frozen
base sampler:

```bash
uv run python scripts/prepare_bct_targets.py \
  --backend local \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data artifacts/data/wrong-argument-pairs.jsonl \
  --source-messages-field unbiased_messages \
  --main-messages-field biased_messages \
  --control-messages-field unbiased_messages \
  --main-output artifacts/data/bct.jsonl \
  --control-output artifacts/data/bct-control.jsonl \
  --manifest-output artifacts/data/bct-targets.manifest.json
```

Every source row is validated before the backend is initialized. CTM samples
one reference completion per row and reuses it in both output files. Existing
outputs are never overwritten.

### Pinned HLE export

The RMCT paper evaluates on the text-only multiple-choice subset of Humanity's
Last Exam. Export that gated source to the local JSONL schema accepted by
`mcq_bias`:

```bash
uv run python -m ctm_data.sources.hle \
  --output ctm_data/local/hle-text-mc.jsonl \
  --manifest-output ctm_data/local/hle-text-mc.manifest.json
```

The exporter pins the upstream revision, excludes image-dependent questions,
requires exactly 513 rows, and records the output hash. The generated benchmark
file is gitignored and must not be redistributed. Pass its JSONL path as a
normal `mcq_bias` dataset value.

### Cleaned Alpaca prompts

The RMCT paper mixes fresh base-model instruction responses into bias-augmented
consistency training. Export a deterministic prompt-only subset with:

```bash
uv run python -m ctm_data.sources.cleaned_alpaca \
  --output artifacts/data/cleaned-alpaca-prompts.jsonl \
  --manifest-output artifacts/data/cleaned-alpaca-prompts.manifest.json \
  --count 2048 \
  --seed 42
```

The exporter downloads one pinned Cleaned Alpaca revision, verifies its
SHA-256 digest, and ignores every source response. Pass the resulting prompts
to `scripts/prepare_bct_targets.py`; CTM then samples responses through the
same frozen base model selected by the experiment.

The analysis command accepts repeated condition names to pool independent
replicates while retaining only the latest task retry inside each directory.
`towards_bias_switch` is conditional on the unbiased answer not matching the
bias; `away_from_bias_switch` is conditional on the unbiased answer matching
the bias:

```bash
uv run python -m ctm_data.adapters.mcq_bias.analysis \
  --run rmct=logs/evals/lr-1e-4 \
        rmct=logs/evals/lr-2.86e-4 \
        rmct=logs/evals/lr-5e-4 \
  --metric towards_bias_switch \
  --stderr binomial \
  --output artifacts/results/rmct-hle.json
```

For unconditional bias verbalisation, use `--metric bias_acknowledged`. For the
paper's towards-bias-switch subset, additionally pass
`--where-metric towards_bias_switch`. To condition on a switch in either
direction, pass `--where-metric abs_switch`; `mcq_bias` uses that internal field
for total bias switch. These reports reuse the same per-sample verbalisation
grade and do not make additional grader calls. When an experiment does not
report verbalisation, set
`include_bias_acknowledged: false` in `mcq_bias` task arguments to avoid grader
calls.

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
