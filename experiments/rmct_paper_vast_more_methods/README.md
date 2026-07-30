# GPT-OSS-20B six-method comparison on HLE

This experiment compares six consistency-training methods on the HLE
evaluation setting from Figures 4 and 5 of *Rate Matching Consistency
Training* (arXiv:2606.02211):

1. rate matching;
2. bias-augmented consistency training (BCT);
3. on-policy consistency training (OPCT);
4. activation consistency training (ACT);
5. attention consistency training (AttCT); and
6. MLP consistency training (MLPCT).

The untrained `openai/gpt-oss-20b` model is evaluated once. Rate matching and
BCT have matched controls. Every trained condition runs at three learning rates
except OPCT, which uses its paper value of `5e-6`. This produces twenty-two
trained checkpoints and twenty-three evaluated model states. The RMCT paper
directly reports the untrained, rate-matching and BCT families; ACT, AttCT and
MLPCT are extensions on the same data and evaluation setting. OPCT follows
*On-Policy Consistency Training Improves LLM Safety with Minimal Capability
Degradation* (arXiv:2605.21834).

The concise specification is in [`experiment.yaml`](experiment.yaml). Its
`experiment_factory` expands the condition and learning-rate matrix into the
complete command plan. One invocation prepares the data, trains the twenty-two
runs, evaluates every checkpoint and the base model, aggregates the results,
and renders seven publication-style SVG charts from chart-ready JSON.
Rendering runs as a separate final `rendering` stage, so charts can also be
re-rendered later — for example on another machine, from the synced results,
with `--start-from rendering`.

## Experimental design

The consistency data uses the distractor-argument bias on LogiQA and
HellaSwag. Evaluation uses the text-only multiple-choice subset of Humanity's
Last Exam with the following six biases:

- suggested answer;
- distractor fact;
- distractor argument (`wrong_argument` in `mcq_bias`);
- post hoc rationalisation;
- spurious few-shot squares; and
- wrong few-shot answers.

Bias-augmented consistency training uses 2,048 biased examples and 2,048 fresh
base-model responses to Cleaned Alpaca prompts. ACT, AttCT and MLPCT use the
same 2,048 unbiased/biased prompt pairs. OPCT also uses those pairs: it samples
four biased-prompt responses online and scores each under the frozen base model
conditioned on the paired unbiased prompt. Its configuration uses `lambda=2.0`,
`gamma=0.9`, temperature `0.7`, maximum 2,048 generated tokens, and learning
rate `5e-6`. Rate matching uses 64 datapoints, 128 reference rollouts and 128
biased rollouts per datapoint. All six methods use rank-8, alpha-16 LoRA over
attention and MLP modules. The supervised methods use microbatches of one with
128-step gradient accumulation. Conditions other than OPCT run at learning
rates `1e-4`, `2.86e-4`, and `5e-4`; evaluation samples are pooled across each
condition's available checkpoints.

MLPCT compares the GPT-OSS MoE block output because GPT-OSS stores expert
projections as fused parameters rather than individual down-projection modules.

All six methods run through the same local PyTorch/PEFT backend. The complete plan
must run on one platform, either a Vast.ai instance or an Isambard GH200 node.
Mixing local and Tinker checkpoints in this comparison is not supported.

The controls remove the biased prompt while preserving each method's training
path. Rate matching uses the unbiased prompt for both perturbations. BCT trains
on frozen base-model responses to unbiased prompts. ACT, AttCT and MLPCT have
no control conditions: pairing the unbiased prompt with itself makes their
consistency losses identically zero, so such a run would only duplicate the
untrained condition. OPCT has no control in this matrix; clean-to-clean OPCT is
a non-zero self-distillation intervention rather than a null control. The
experiment compiler rejects control conditions for the representation methods.

The reports contain:

- conditional towards-bias switch rate, where the denominator contains only
  questions for which a switch towards the biased answer was possible;
- unconditional bias verbalisation over every successfully graded response;
- bias verbalisation conditioned on a towards-bias switch; and
- bias verbalisation conditioned on a total bias switch in either direction;
- away-from-bias switch rate;
- total switch rate; and
- accuracy on the shared unbiased evaluation.

The `mcq_bias` scorer calls the underlying verbalisation value
`bias_acknowledged`. Unfiltered reports use Inspect's stored means and standard
errors. Conditional reports reuse saved sample scores, are marked as derived in
the JSON, and do not make additional grader calls. Aggregation across datasets,
learning rates, and held-out biases is sample-count weighted. Two-proportion
tests against the untrained condition are computed before rendering and stored
as `p_value` and `significance`; the renderer does not infer statistics.

Additional report items may set `ratio: true` to emit percentage change from
the configured untrained significance baseline. They may also replace the
`given` shorthand with a declarative `where` predicate for arbitrary post-hoc
conditioning. Chart recipes can enable `sample_labels`, override automatic
model/training-bias faceting, and customize the publication theme. Common
labels, ordering, and method colors come from the mcq-bias presentation TOML
registry and remain overridable per chart.

## Reproduction boundary

This is a reproduction through the current CTM and `mcq_bias` interfaces, not
a claim that the original experiment artifacts are byte-identical.

- This repository uses conditional directional switch metrics. Values need not
  equal the paper's unconditional directional switch result even when model
  behaviour is identical.
- Cleaned Alpaca prompts are selected deterministically, but their GPT-OSS
  responses are generated during the run.
- Distractor arguments are generated with
  `openrouter/google/gemma-4-31b-it` when the selected store is empty. Reusing a
  materialised store preserves its arguments; starting from an empty store
  creates a new frozen dataset and records its hashes.
- The supplied reports prefer the exact metric and standard error recorded by
  Inspect. Post-hoc conditional subsets necessarily use sample-derived standard
  errors because Inspect has no aggregate for those subsets.
- OPCT, ACT, AttCT and MLPCT were not conditions in the referenced RMCT figure.
  Their inclusion tests additional CTM methods under the same experimental setting.

## Prerequisites

Install the Python dependencies described in the repository README. This
experiment's publication renderer does not require Node or Flint. The complete
run requires:

- one 96 GB local GPU environment prepared with either the Vast.ai or Isambard
  launcher;
- `OPENROUTER_API_KEY` for distractor-argument generation and verbalisation grading;
- a Hugging Face token with access to the gated HLE dataset; and
- `WANDB_API_KEY`, because experiment tracking is explicitly enabled.

The data and result writers refuse to overwrite existing files. To repeat a
run, retain the old artifacts under an archive directory and use new output
paths or a new experiment name.

The two source-selection executables are paper-owned because they pin this
reproduction's exact revisions, filters, and counts. The experiment runner
invokes them automatically; they can also be inspected independently:

```bash
uv run python -m scripts.rmct_paper_vast_more_methods.hle_source --help
uv run python -m scripts.rmct_paper_vast_more_methods.cleaned_alpaca_source --help
```

The HLE command excludes image-dependent rows and requires the configured
text-only multiple-choice count. Its generated data remains local under the
upstream distribution terms. The Cleaned Alpaca command verifies the pinned
source hash, ignores source responses, and selects only prompts; CTM generates
fresh targets with the experiment's frozen base model.

### Integration smoke test

Before submitting the full matrix on a new machine image, run the explicit
integration configuration. It covers all six methods, both controls, the
untrained evaluation, checkpoint loading, reporting, and both remote-model
uses. Its counts and token limits are not suitable for scientific comparison.

```bash
uv run python scripts/run_experiment.py \
  experiments/rmct_paper_vast_more_methods/debug/smoke.yaml \
  --parallel 4 --gpus 0,1,2,3
```

## Commands

Run these commands inside a Vast.ai host with eight visible GPUs. The same YAML
can run on fewer GPUs by shortening `--gpus` and reducing `--parallel`. Inspect
the complete resolved plan without making model calls:

```bash
uv run python scripts/run_experiment.py \
  experiments/rmct_paper_vast_more_methods/experiment.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 \
  --dry-run
```

After reviewing the exact commands, run the complete plan:

```bash
uv run python scripts/run_experiment.py \
  experiments/rmct_paper_vast_more_methods/experiment.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7
```

The runner asks for one final confirmation. It does not start a remote model or
training stage unless that confirmation is given. Each GPU runs at most one
command at a time. Data generation, data preparation, training, and evaluation
are separate barriers. The evaluation suite is frozen once during preparation,
so the twenty-two parallel evaluations read identical files without competing
to generate them. Analysis remains sequential, and chart rendering runs as the
final stage. A completed plan can resume at a named boundary, for
example:

```bash
uv run python scripts/run_experiment.py \
  experiments/rmct_paper_vast_more_methods/experiment.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 \
  --start-from evaluation
```

Named local checkpoints are persisted in
`logs/experiments/rmct_paper_vast_more_methods/outputs.json`. The expanded
plan is saved beside that file as `resolved-plan.yaml` after approval.

## Workload and cost boundary

Before retries or failed distractor arguments, the plan requests:

- 4,096 local generations for bias-augmented consistency targets;
- 98,304 local rollouts for the six rate-matching runs;
- 15,400 local evaluation completions: 700 for each of twenty-two model states;
- 24,576 supervised sequences across the six BCT runs;
- 18,432 paired examples across the nine ACT, AttCT and MLPCT runs;
- 13,200 OpenRouter verbalisation-grader calls; and
- up to 6,200 OpenRouter calls to construct an empty distractor-argument store.

The dominant resource is local GPU time. Vast.ai cost depends on the selected
offer and the measured duration of a representative run. Isambard consumes an
allocation rather than a per-token Tinker charge. The configured 20,480-token
value is a ceiling, not an expected completion length. OpenRouter cost depends
on argument-generation retries and grader response length. A monetary estimate
must use the selected platform price and a measured smoke-run duration before
the thirty-run matrix is submitted.

No remote stage should be started until this workload and a spending or
allocation limit have been reviewed explicitly.
