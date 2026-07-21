# GPT-OSS-20B rate-matching comparison on HLE

This experiment compares the five conditions reported in Figures 4 and 5 of
*Rate Matching Consistency Training* (arXiv:2606.02211):

1. the untrained `openai/gpt-oss-20b` base model;
2. behaviour consistency training;
3. the behaviour consistency control;
4. rate matching consistency training; and
5. the rate matching control.

The concise experiment specification is in [`experiment.yaml`](experiment.yaml).
Its `experiment_factory` expands the condition and learning-rate matrix into
the complete command plan. One invocation prepares the data, trains the twelve
paper runs, evaluates every resulting checkpoint and the base model, aggregates
the results, and renders four Flint charts.

## Experimental design

The consistency data uses the distractor-argument bias on LogiQA and
HellaSwag. Evaluation uses the multiple-choice subset of Humanity's Last Exam
with the following six biases:

- suggested answer;
- distractor fact;
- distractor argument (`wrong_argument` in `mcq_bias`);
- post hoc rationalisation;
- spurious few-shot squares; and
- wrong few-shot answers.

Behaviour consistency training uses 2,048 consistency examples and 2,048
fresh base-model responses to Cleaned Alpaca prompts. Rate matching uses 64
datapoints, 128 reference rollouts and 128 biased rollouts per datapoint. Both
methods use rank-8, alpha-16 LoRA over attention and MLP modules. Each trained
condition is run at learning rates `1e-4`, `2.86e-4`, and `5e-4`; evaluation
samples are pooled across those three checkpoints.

The outputs are:

- conditional towards-bias switch rate, where the denominator contains only
  questions for which a switch toward the biased answer was possible;
- unconditional bias verbalisation rate over every response with a successful
  verbalisation grade;
- bias verbalisation rate conditioned on a towards-bias switch; and
- bias verbalisation rate conditioned on a total bias switch, meaning a switch
  either towards or away from the biased answer.

The `mcq_bias` scorer calls the underlying verbalisation value
`bias_acknowledged`. It grades both the structured reasoning channel and the
final answer. CTM calculates the two conditional rates from the same saved
sample scores, so they do not make additional grader calls.

## Reproduction boundary

This is a reproduction through the current CTM and `mcq_bias` interfaces, not
a claim that the original experiment artifacts are byte-identical.

- The paper prints an unconditional directional switch equation. This
  repository intentionally uses the conditional switch definition. Values in
  the first chart therefore need not equal Figure 5 even when model behaviour
  is identical.
- Cleaned Alpaca prompts are selected deterministically from the cited source,
  but their fresh GPT-OSS responses are generated during this run.
- Distractor arguments are generated with
  `openrouter/google/gemma-4-31b-it` when the selected store is empty. Reusing a
  previously materialised store preserves its exact arguments; starting from
  an empty store creates a new frozen dataset and records its hashes.
- The experiment reports one binomial standard error. The paper plots two
  binomial standard errors and adds significance markers. The aggregate JSON
  contains the counts required to construct those intervals, but the supplied
  Flint chart does not add significance tests.

## Prerequisites

Install the Python and JavaScript dependencies as described in the repository
README. The complete run requires:

- `TINKER_API_KEY` for sampling and training;
- `OPENROUTER_API_KEY` for distractor-argument generation and verbalisation
  grading;
- a Hugging Face token with access to the HLE dataset; and
- `WANDB_API_KEY`, because experiment tracking is explicitly enabled in the
  YAML.

The data and result writers refuse to overwrite existing files. To repeat a
run, retain the old artifacts under an archive directory and use new output
paths or a new experiment name.

## Commands

Inspect the resolved plan without making model calls:

```bash
uv run python scripts/run_experiment.py \
  experiments/paper_reproductions/rmct_hle_gpt_oss_20b/experiment.yaml \
  --dry-run
```

Run the complete plan after reviewing the printed commands:

```bash
uv run python scripts/run_experiment.py \
  experiments/paper_reproductions/rmct_hle_gpt_oss_20b/experiment.yaml
```

The runner asks for one final confirmation. It does not start a paid stage
unless that confirmation is given. A previously completed plan can resume at
a named boundary, for example:

```bash
uv run python scripts/run_experiment.py \
  experiments/paper_reproductions/rmct_hle_gpt_oss_20b/experiment.yaml \
  --start-from evaluation
```

Named Tinker checkpoint outputs are persisted under
`logs/experiments/rmct-hle-gpt-oss-20b-five-condition/outputs.json`, so the
evaluation stage resolves each checkpoint to the training run that produced
it. After approval, the expanded plan is saved beside that file as
`resolved-plan.yaml`. The runner rejects later changes under the same experiment
name, preventing a resumed run from silently using a different matrix.

## Workload and cost boundary

Before retries or failed distractor arguments, the plan requests:

- 4,096 Tinker completions for behaviour-consistency targets;
- 98,304 Tinker rollouts for the six rate-matching runs;
- 9,100 Tinker evaluation completions: 700 for each of thirteen evaluated
  model states;
- 24,576 supervised training sequences across the six behaviour-consistency
  runs;
- 7,800 OpenRouter verbalisation-grader calls; and
- up to 6,200 OpenRouter calls to construct an empty distractor-argument
  store, because each of 3,100 questions may receive two generation attempts.

The exact charge cannot be known before sampling because GPT-OSS stops at
variable lengths. The configured 20,480-token value is a ceiling, not an
expected completion length. At the Tinker prices current on 21 July 2026,
sampling GPT-OSS-20B costs $0.45 per million generated tokens, prefill costs
$0.18 per million input tokens, and training costs $0.396 per million tokens.
The full plan is therefore plausibly in the low hundreds of US dollars, but a
reasoning-length tail can make it materially higher. The theoretical ceiling
from the configured token limits is not a useful budget and exceeds $1,000 in
sampling alone.

Gemma 4 31B currently starts near $0.12 per million input tokens and $0.35 per
million output tokens on OpenRouter. Those calls should be a small fraction of
the Tinker charge, but their exact cost depends on argument-generation retries
and grader prompt length.

No remote stage should be started until this workload and a spending limit
have been reviewed explicitly.
