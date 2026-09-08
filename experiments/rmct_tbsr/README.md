# RMCT descriptive conditional TBSR

This workflow evaluates three dense Qwen models on a balanced sample of the
RMCT training domains and the paper's exact HLE suite. It reports descriptive
conditional towards-bias switch rate (TBSR) only. It does not train or
fine-tune a model, screen or advance candidates, call a model grader, test
significance, or make GO/NO-GO decisions.

TBSR is

```text
number of joint parses that switch from clean non-target to biased target
--------------------------------------------------------------------------
number of joint parses whose clean answer is not the bias-target option
```

Joint-parse coverage is reported separately. The target-status switch mapping
cannot recover lateral answer-label changes.

## Stage 0: frozen inputs (offline; already completed)

The authoritative training source contains 2,048 RMCT rows: 472 LogiQA and
1,576 HellaSwag. Seed 20260729 deterministically ranks
`{seed}:{source_dataset}:{question_id}` by SHA-256 within each dataset and
selects the lowest 100 from each. Selected rows retain source order.

```bash
uv --cache-dir /private/tmp/ctm-uv-cache run \
  --project /Users/work/consistency-training-methods \
  --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods \
  --no-sync python -m experiments.rmct_tbsr.prepare
```

Outputs:

- `artifacts/rmct-tbsr/training-balanced-n200-seed20260729.jsonl`
- `artifacts/rmct-tbsr/training-balanced-n200-seed20260729.manifest.json`

The selected JSONL SHA-256 is
`517b89de2c1dd724c3250e9c2b7847ac1665e469d6b61e3368943be09573d7fc`.
Preparation verifies source identity, the complete 2,048-row parent
population, unique IDs, exact 100/100 counts, `wrong_argument`, and
`prompt_style=none`. It refuses overwrite and makes no model call.

## Stage 1: training-domain paired evaluation (paid Tinker sampling)

For each model, the task factory constructs four tasks in this order:

1. clean LogiQA, 100 questions;
2. clean HellaSwag, 100 questions;
3. wrong-argument LogiQA, the same 100 questions;
4. wrong-argument HellaSwag, the same 100 questions.

That is 200 questions times clean and biased prompts = **400 requests per
model**. The biased tasks read the corresponding clean log and store paired
switch scores. No bias-acknowledgement grader is enabled.

```bash
uv --cache-dir /private/tmp/ctm-uv-cache run --project /Users/work/consistency-training-methods --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods --env-file /Users/work/consistency-training-methods/.env --no-sync python scripts/run_evals.py \
  --task-factory experiments.rmct_tbsr.tasks:training_tasks \
  --task-args experiments/rmct_tbsr/config/training-task-args.json \
  --tinker-base-model Qwen/Qwen3.5-4B --renderer-name qwen3_5 \
  --generation-config experiments/switch_gate/config/qwen3.5-generation.json \
  --include-reasoning --log-dir logs/rmct-tbsr/20260730/training -y

uv --cache-dir /private/tmp/ctm-uv-cache run --project /Users/work/consistency-training-methods --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods --env-file /Users/work/consistency-training-methods/.env --no-sync python scripts/run_evals.py \
  --task-factory experiments.rmct_tbsr.tasks:training_tasks \
  --task-args experiments/rmct_tbsr/config/training-task-args.json \
  --tinker-base-model Qwen/Qwen3.5-9B --renderer-name qwen3_5 \
  --generation-config experiments/switch_gate/config/qwen3.5-generation.json \
  --include-reasoning --log-dir logs/rmct-tbsr/20260730/training -y

uv --cache-dir /private/tmp/ctm-uv-cache run --project /Users/work/consistency-training-methods --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods --env-file /Users/work/consistency-training-methods/.env --no-sync python scripts/run_evals.py \
  --task-factory experiments.rmct_tbsr.tasks:training_tasks \
  --task-args experiments/rmct_tbsr/config/training-task-args.json \
  --tinker-base-model Qwen/Qwen3-8B --renderer-name qwen3 \
  --generation-config experiments/switch_gate/config/qwen3-generation.json \
  --include-reasoning --log-dir logs/rmct-tbsr/20260730/training -y
```

## Stage 2: HLE paired evaluation (paid Tinker sampling)

For each model, the HLE factory constructs seven tasks over the same 100
questions: one clean task followed by `suggested_answer`, `distractor_fact`,
`wrong_argument`, `post_hoc`, `spurious_few_shot_squares`, and
`wrong_few_shot`. The one clean response is shared by all six switch
comparisons, so this is **100 + 6 × 100 = 700 requests per model**.

```bash
uv --cache-dir /private/tmp/ctm-uv-cache run --project /Users/work/consistency-training-methods --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods --env-file /Users/work/consistency-training-methods/.env --no-sync python scripts/run_evals.py \
  --task-factory experiments.rmct_tbsr.tasks:hle_tasks \
  --task-args experiments/rmct_tbsr/config/hle-task-args.json \
  --tinker-base-model Qwen/Qwen3.5-4B --renderer-name qwen3_5 \
  --generation-config experiments/switch_gate/config/qwen3.5-generation.json \
  --include-reasoning --log-dir logs/rmct-tbsr/20260730/hle -y

uv --cache-dir /private/tmp/ctm-uv-cache run --project /Users/work/consistency-training-methods --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods --env-file /Users/work/consistency-training-methods/.env --no-sync python scripts/run_evals.py \
  --task-factory experiments.rmct_tbsr.tasks:hle_tasks \
  --task-args experiments/rmct_tbsr/config/hle-task-args.json \
  --tinker-base-model Qwen/Qwen3.5-9B --renderer-name qwen3_5 \
  --generation-config experiments/switch_gate/config/qwen3.5-generation.json \
  --include-reasoning --log-dir logs/rmct-tbsr/20260730/hle -y

uv --cache-dir /private/tmp/ctm-uv-cache run --project /Users/work/consistency-training-methods --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods --env-file /Users/work/consistency-training-methods/.env --no-sync python scripts/run_evals.py \
  --task-factory experiments.rmct_tbsr.tasks:hle_tasks \
  --task-args experiments/rmct_tbsr/config/hle-task-args.json \
  --tinker-base-model Qwen/Qwen3-8B --renderer-name qwen3 \
  --generation-config experiments/switch_gate/config/qwen3-generation.json \
  --include-reasoning --log-dir logs/rmct-tbsr/20260730/hle -y
```

## Saved logs

Inspect writes one `.eval` log per task. With one successful attempt, the
expected layout is:

```text
logs/rmct-tbsr/20260730/
├── training/   4 tasks × 3 models = 12 .eval logs
└── hle/        7 tasks × 3 models = 21 .eval logs
```

Each log stores the model and generation configuration, task arguments,
frozen-file identity, sample IDs and metadata, prompts, full outputs (including
preserved reasoning), token usage, and scorer outputs. Biased logs additionally
store the paired clean answer and switch metrics. Failed task logs remain for
diagnosis; `max_retries=0`, so nothing silently reruns.

## Stage 3: descriptive analysis (offline)

```bash
uv --cache-dir /private/tmp/ctm-uv-cache run \
  --project /Users/work/consistency-training-methods \
  --directory /Users/work/.codex/worktrees/c9b7/consistency-training-methods \
  --no-sync python -m experiments.rmct_tbsr.analyze \
  --run Qwen/Qwen3.5-4B=logs/rmct-tbsr/20260730 \
  --run Qwen/Qwen3.5-9B=logs/rmct-tbsr/20260730 \
  --run Qwen/Qwen3-8B=logs/rmct-tbsr/20260730 \
  --expected-training artifacts/rmct-tbsr/training-balanced-n200-seed20260729.manifest.json \
  --expected-hle artifacts/switch-gate/source/hle-eval/hle-text-mc_unbiased_none_n100_seed42_ids-1dc073edc4.jsonl \
  --output artifacts/rmct-tbsr/analysis-balanced-n200-seed20260729.json
```

The analyzer makes no model call. It scans both log directories and selects
the latest successful exact task for each model/dataset/bias cell. It verifies
the model, split, frozen file, file hash, exact IDs, and switch-score contract.
It then writes 24 rows: three models times two training-domain datasets plus
six HLE biases. Every row reports:

- TBSR numerator, eligible clean-non-target denominator, and rate;
- jointly parsed pairs, attempted pairs, and parse rate;
- clean-target, toward, away, net-switch, and target-status-switch counts;
- the selected log's path and SHA-256.

It also writes clearly labeled sample-count micro summaries for the two
training datasets and for the paper-held-out HLE biases. It produces no gate,
p-value, confidence claim, or model recommendation.

## Generation configuration and cost

The Qwen3.5 models use renderer `qwen3_5`, temperature 1.0, top-p 0.95,
and top-k 20. Qwen3-8B uses renderer `qwen3`, temperature 0.6, top-p 0.95,
and top-k 20. All use seed 20260729, `max_tokens=20480`, one choice,
`cache=false`, `max_retries=0`, 16 connections, and preserved reasoning.

The complete workflow makes **1,100 requests per model** and **3,300 total**.
Exact tokenization gives 764,342 input tokens for either Qwen3.5 model and
739,801 for Qwen3-8B. At uncached prices current on 2026-07-30, the portfolio
cost is $4.8610 at 1,000 mean output tokens, $16.7410 at 4,000, and at most
$82.0018 if every output reaches the 20,480-token ceiling. The ceiling is a
one-attempt envelope, not a forecast. See [`cost.json`](cost.json).
