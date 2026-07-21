# GPT-OSS-20B five-method comparison on HLE

This experiment compares five consistency-training methods and their controls
on the HLE evaluation setting from Figures 4 and 5 of *Rate Matching
Consistency Training* (arXiv:2606.02211):

1. rate matching;
2. bias-augmented consistency training (BCT);
3. activation consistency training (ACT);
4. attention consistency training (AttCT); and
5. MLP consistency training (MLPCT).

The untrained `openai/gpt-oss-20b` model is evaluated once. Every trained method
has a matched control and is trained at three learning rates, producing thirty
trained checkpoints and thirty-one evaluated model states. The RMCT paper
directly reports the untrained, rate-matching and BCT families; ACT, AttCT and
MLPCT are an extension on the same data and evaluation setting.

The concise specification is in [`experiment.yaml`](experiment.yaml). Its
`experiment_factory` expands the condition and learning-rate matrix into the
complete command plan. One invocation prepares the data, trains the thirty
runs, evaluates every checkpoint and the base model, aggregates the results,
and renders four Flint charts.

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
same 2,048 unbiased/biased prompt pairs. Rate matching uses 64 datapoints, 128
reference rollouts and 128 biased rollouts per datapoint. All five methods use
rank-8, alpha-16 LoRA over attention and MLP modules. The supervised methods use
microbatches of one with 128-step gradient accumulation. Each trained condition
is run at learning rates `1e-4`, `2.86e-4`, and `5e-4`; evaluation samples are
pooled across those three checkpoints.

MLPCT compares the GPT-OSS MoE block output because GPT-OSS stores expert
projections as fused parameters rather than individual down-projection modules.

All methods run through the same local PyTorch/PEFT backend. The complete plan
must run on one platform, either a Vast.ai instance or an Isambard GH200 node.
Mixing local and Tinker checkpoints in this comparison is not supported.

The controls remove the biased prompt while preserving each method's training
path. Rate matching uses the unbiased prompt for both perturbations. BCT trains
on frozen base-model responses to unbiased prompts. ACT, AttCT and MLPCT compare
the adapter-enabled unbiased prompt with the frozen-base unbiased prompt.

The reports contain:

- conditional towards-bias switch rate, where the denominator contains only
  questions for which a switch towards the biased answer was possible;
- unconditional bias verbalisation over every successfully graded response;
- bias verbalisation conditioned on a towards-bias switch; and
- bias verbalisation conditioned on a total bias switch in either direction.

The `mcq_bias` scorer calls the underlying verbalisation value
`bias_acknowledged`. The conditional reports reuse saved sample scores and do
not make additional grader calls.

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
- The supplied reports use one binomial standard error. They do not implement
  the paper's significance-marker tests.
- ACT, AttCT and MLPCT were not conditions in the referenced RMCT figure. Their
  inclusion tests additional CTM methods under the same experimental setting.

## Prerequisites

Install the Python and JavaScript dependencies described in the repository
README. The complete run requires:

- one 96 GB local GPU environment prepared with either the Vast.ai or Isambard
  launcher;
- `OPENROUTER_API_KEY` for distractor-argument generation and verbalisation grading;
- a Hugging Face token with access to the gated HLE dataset; and
- `WANDB_API_KEY`, because experiment tracking is explicitly enabled.

The data and result writers refuse to overwrite existing files. To repeat a
run, retain the old artifacts under an archive directory and use new output
paths or a new experiment name.

## Commands

Run these commands inside the selected Vast.ai or Isambard environment. Inspect
the complete resolved plan without making model calls:

```bash
uv run python scripts/run_experiment.py \
  experiments/paper_reproductions/rmct_hle_gpt_oss_20b/experiment.yaml \
  --dry-run
```

After reviewing the exact commands, run the complete plan:

```bash
uv run python scripts/run_experiment.py \
  experiments/paper_reproductions/rmct_hle_gpt_oss_20b/experiment.yaml
```

The runner asks for one final confirmation. It does not start a remote model or
training stage unless that confirmation is given. A completed plan can resume
at a named boundary, for example:

```bash
uv run python scripts/run_experiment.py \
  experiments/paper_reproductions/rmct_hle_gpt_oss_20b/experiment.yaml \
  --start-from evaluation
```

Named local checkpoints are persisted in
`logs/experiments/rmct-hle-gpt-oss-20b-five-method/outputs.json`. The expanded
plan is saved beside that file as `resolved-plan.yaml` after approval.

## Workload and cost boundary

Before retries or failed distractor arguments, the plan requests:

- 4,096 local generations for bias-augmented consistency targets;
- 98,304 local rollouts for the six rate-matching runs;
- 21,700 local evaluation completions: 700 for each of thirty-one model states;
- 24,576 supervised sequences across the six BCT runs;
- 36,864 paired examples across the eighteen ACT, AttCT and MLPCT runs;
- 18,600 OpenRouter verbalisation-grader calls; and
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
