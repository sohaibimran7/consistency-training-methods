# Wrong-argument cross-bias method comparison

This experiment compares an untrained base model with RLCT, ACT, and BCT, plus
one matched control for each method. All trained models use the local backend;
run the experiment inside a provisioned Vast.ai instance.

The training cue called `distractor_argument` in older experiments is named
`wrong_argument` by the current `mcq_bias` package. Training uses LogiQA and
HellaSwag. Evaluation uses the six other packaged biases on MMLU and
TruthfulQA. These choices are explicit in `experiment.yaml` and may be changed
without modifying CTM.

## Conditions

| Condition | Training input |
|---|---|
| Base | No training |
| RLCT | Independent rollouts from unbiased and wrong-argument prompts |
| RLCT control | Two independent rollout groups from the unbiased prompt |
| ACT | Current-model activations on the biased prompt matched to frozen-base activations on the unbiased prompt |
| ACT control | Current-model and frozen-base activations on the unbiased prompt |
| BCT | Biased prompt followed by a frozen-base completion sampled from the corresponding unbiased prompt |
| BCT control | Unbiased prompt followed by the same frozen-base completion used by BCT |

CTM samples every BCT target once before either BCT run. The generated main and
control files therefore contain identical assistant completions and differ only
in the prompt side. Their shared manifest records the source-file hashes, field
mapping, model, backend, and generation configuration.

ACT reference activations are not stored. The local backend computes them under
the frozen base model during training. This avoids large model- and
method-specific activation artifacts.

## Run

Install both Python and chart dependencies on the Vast.ai instance:

```bash
uv venv
uv pip install -r requirements.txt
uv pip install -e . --no-deps
npm ci
```

Review every resolved command without executing it:

```bash
uv run python scripts/run_experiment.py \
  experiments/mcq_bias/wrong_argument_cross_bias/experiment.yaml \
  --dry-run
```

Run the complete pipeline without further prompts:

```bash
uv run python scripts/run_experiment.py \
  experiments/mcq_bias/wrong_argument_cross_bias/experiment.yaml \
  --yes
```

The `wrong_argument` materializer may call the configured OpenRouter argument
model when its argument store is incomplete. Supply credentials through the
environment; do not put them in the YAML.

The final analysis reads the official `mcq_bias` sample scores, checks that all
seven conditions contain the same successful task matrix, pools
`matches_bias` over the configured evaluation datasets, and writes:

- `artifacts/results/wrong-argument-cross-bias/matches-bias.json`;
- `figures/wrong-argument-cross-bias/matches-bias.svg`; and
- the compiled Vega-Lite specification beside the SVG.

The chart is compiled by the pinned Flint dependency. Statistical aggregation
is performed before rendering; Flint does not decide which samples or metrics
belong in the figure.

## Compare BCT execution platforms

`bct_backends.yaml` contains one BCT condition for Tinker, Vast.ai, and
Isambard. All three entries inherit the same data path, model, LoRA object,
Adam object, batch size, epoch count, and evaluation configuration. This is a
convenience provided by YAML anchors, not a runtime equality check.

Generate the BCT data once with the main experiment's first two stages. Copy
the resulting JSONL and manifest without changing either file, then run the
appropriate target in each environment:

```bash
uv run python scripts/run_experiment.py \
  experiments/mcq_bias/wrong_argument_cross_bias/experiment.yaml \
  --stages data_generation,data_preparation --yes

uv run python scripts/run_experiment.py \
  experiments/mcq_bias/wrong_argument_cross_bias/bct_backends.yaml \
  --target tinker --yes

uv run python scripts/run_experiment.py \
  experiments/mcq_bias/wrong_argument_cross_bias/bct_backends.yaml \
  --target vast --yes

sbatch infra/isambard/run_experiment.sbatch \
  experiments/mcq_bias/wrong_argument_cross_bias/bct_backends.yaml \
  --target isambard --yes
```

Copy the three evaluation log directories back to the controller under the
paths declared in the YAML. Install the pinned chart dependencies and run:

```bash
uv run python scripts/run_experiment.py \
  experiments/mcq_bias/wrong_argument_cross_bias/bct_backends.yaml \
  --target controller --yes
```

The target selector only filters commands. It does not provision machines,
transfer artifacts, or require supposedly comparable entries to be equal.
