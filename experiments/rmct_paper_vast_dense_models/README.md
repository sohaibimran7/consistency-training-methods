# Dense Qwen RMCT/consistency comparison on Vast

This directory preregisters the same local HLE comparison for two dense models:
`Qwen/Qwen3.5-9B` and `Qwen/Qwen3-8B`. Each full model plan evaluates one
untrained base model and trains eight conditions at one provisional common
learning rate, `1e-4`:

1. RMCT and its clean-prompt control;
2. BCT and its clean-prompt control;
3. ACT, AttCT, and MLPCT; and
4. OPCT, without an automatically generated control.

That is eight trained checkpoints plus one base state per model, or 16 trained
checkpoints plus two base states across the portfolio. The model plans are
[`qwen3_5_9b.yaml`](qwen3_5_9b.yaml) and
[`qwen3_8b.yaml`](qwen3_8b.yaml). Both use the compiler at
`scripts.rmct_paper_vast_more_methods.experiment_factory`; it keeps internal methods on
the local backend and rejects Tinker for this comparison.

## Frozen shared data boundary

Run [`shared_data.yaml`](shared_data.yaml) exactly once before either model
plan. It is the sole writer of the shared artifacts under
`artifacts/rmct-hle-dense-models-shared/`:

- the text-only HLE source and manifest;
- a balanced 2,048-example prefix of the imported legacy
  `distractor_argument_g4` prompt-pair stores and its manifest;
- the seeded 2,048 Cleaned Alpaca prompt selection and manifest; and
- one materialized evaluation suite containing 100 HLE questions under each of
  six biases.

Before running either shared-data plan, place the existing 2,400-row LogiQA
and HellaSwag stores at the two paths under
`artifacts/rmct-hle-dense-models-legacy-sources/` declared in the YAML. The
importer filters empty biased prompts, validates the native schema, selects the
first 1,500 valid pairs per source, and round-robins them. Consequently, the
first 2,048 training rows contain exactly 1,024 examples from each dataset.
The stored prompts already include chain-of-thought instructions and are
labelled `prompt_style: encourage_cot`; no training wrong arguments are
regenerated.

The two model plans declare `prepare_shared: false`. They therefore cannot
regenerate those paths: both read the exact same prompt pairs and HLE suite,
while each writes model-dependent BCT and instruction targets beneath its own
artifact root. This prevents a second model from repeating OpenRouter argument
generation or silently changing evaluation prompts. Shared writers refuse to
overwrite existing outputs; preserve completed data and manifests. To repeat
the preparation, move the old directory to an `_archive/` location and author
a new shared root rather than deleting it.

The smoke workflow has the same boundary. Run
[`debug/shared_data_smoke.yaml`](debug/shared_data_smoke.yaml) once, followed by
[`debug/qwen3_5_9b_smoke.yaml`](debug/qwen3_5_9b_smoke.yaml) and
[`debug/qwen3_8b_smoke.yaml`](debug/qwen3_8b_smoke.yaml). It uses 16 training
pairs, 16 instruction prompts, two HLE questions per bias, one shared learning
rate, and one OPCT learning rate. These counts are integration-only.

## Expedited 12-hour screen: separate frozen HLE import

The existing switch-gate HLE suite is **not** the full Stage 1 suite. Both use
the pinned 513-row `cais/hle` text-MCQ source, seed `"42"`, prompt style
`none`, the same six bias names, and the same pinned `mcq-bias` prompt/schema
implementation. However, the switch-gate suite was restricted through a
question-ID allowlist: only 88 of its 100 IDs overlap the unrestricted Stage 1
seeded prefix, leaving 12 IDs unique to each pool. Its wrong arguments also
record `vllm/google/gemma-4-31B-it`, whereas full Stage 1 requests
`openrouter/google/gemma-4-31b-it`. Copying or renaming those files into
`artifacts/rmct-hle-dense-models-shared/` would silently change the full
benchmark and falsely label argument provenance, so it is forbidden.

For an explicitly opt-in, time-bounded screen, the offline plan
[`expedited_screen/shared_data_import.yaml`](expedited_screen/shared_data_import.yaml)
imports the complete audited switch-gate suite byte-for-byte into the disjoint
root `artifacts/rmct-hle-dense-models-shared-expedited-12h/`. The importer:

- verifies the switch-gate audit against its lock;
- verifies the pinned HLE source and manifest, all seven frozen-file hashes,
  exact 100-row schemas and ordered IDs, deterministic bias targets, and the
  question-ID sidecar;
- independently reconstructs the unrestricted Stage 1 prefix and requires the
  known 88-overlap/12-per-side divergence; and
- writes an import manifest labelled
  `EXPEDITED_SCREEN_ONLY_NOT_FULL_STAGE1`, retaining the original filenames
  (including the ID and vLLM argument-model slugs).

It makes no API/model call and refuses overwrite. Run it only after the gated
HLE source, its manifest, the seven switch-gate frozen files, and
`question-ids.jsonl` have been securely copied to their declared
`artifacts/switch-gate/source/` paths on the Vast checkout:

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/expedited_screen/shared_data_import.yaml \
  --stages data_generation --yes
```

Consumers of this root must use the exact file mapping recorded in
`import-manifest.json` through `experiments.switch_gate.tasks:hle_tasks`.
Never point `mcq_bias.tasks:suite_tasks`, `materialize_eval`, a full Stage 1
`shared_root`, or the Stage 1 continuation script at the expedited root.

The full Stage 1 recovery remains separate and fail-closed. Preserve every
accepted OpenRouter argument already present in
`$MCQ_BIAS_DATA_DIR/wrong_arguments/openrouter-google-gemma-4-31b-it.jsonl`.
The pinned generator gives each missing question two attempts per
materialization call. A completion is accepted only if its parsed answer is
the deterministic bias target and it avoids the recipe's banned terms; the
five wrapper rounds therefore gave each persistent miss up to ten attempts.
The upstream code deletes a below-floor frozen `wrong_argument` file but keeps
accepted arguments in the canonical store. Failed completions are not stored,
so a shortfall does not by itself distinguish wrong-target parsing from
banned-term rejection.

The Qwen3.5-9B Stage 1 run has an explicitly approved `min_n_questions=92`
protocol for `wrong_argument`: retry only missing arguments with the pinned
OpenRouter model, prompt, acceptance filter, and `mcq-bias` revision until at
least 92 of the fixed 100-ID pool are accepted, then freeze that subset. Report
its realized sample count and argument-acceptance missingness. Do not restrict
to a covered allowlist or replace questions. Any deliberate switch to the
expedited suite must remain labelled as that different screen.

## RMCT paper-aligned batching remediation

The historical Qwen3.5 RMCT recovery uses a one-datapoint microbatch with
four-way gradient accumulation and per-item advantage normalisation. That is
not equivalent to the released RMCT implementation, which trains a real batch
of four datapoints and standardises its full outcome-reward population before
one optimizer step. In particular, per-item normalisation cancels each
question's rate-gap magnitude.

[`stage1/qwen3_5_9b_rmct_paper_fidelity_20260803.yaml`](stage1/qwen3_5_9b_rmct_paper_fidelity_20260803.yaml)
is a separate, training-only, immutable remediation plan. It restores the
paper's batch/normalisation behavior but is not a literal reproduction: it
retains the
frozen no-reasoning Qwen input store, rank-8/alpha-16 LoRA, KL `0.05`, one
epoch over 64 datapoints, 20,480 generated tokens, and the user-approved
`N=96`, but executes real batches of four with pooled GRPO normalisation and
no gradient accumulation. It deliberately differs from the paper's literal
reproduction in model (Qwen3.5 rather than its reported models), prompt style,
single `1e-4` learning rate, `N=96` rather than `128`, and disables LM-head
LoRA; those deviations are declared in the plan rather than hidden in the
historical namespace.

[`stage1/qwen3_5_9b_opct_recovery_20260803.yaml`](stage1/qwen3_5_9b_opct_recovery_20260803.yaml)
is the separate fresh OPCT-only recovery plan. It keeps the frozen no-CoT pair
source and user-approved `1e-4` learning rate, but contains no RMCT target;
the invalid 2026-08-01 b1/accum4/per-item RMCT branch is therefore not silently
carried into an OPCT rerun.

### Required Qwen3.5 input and rollout-worker gates

Every fresh Qwen3.5 on-policy target must pass three gates on the same
eight-GPU node and with the same `CUDA_VISIBLE_DEVICES` mapping that will run
training:

1. The no-CoT source gate calls the exact legacy-G4 conversion verifier—not a
   `prompt_style` metadata check—and records the source JSONL and recovery
   manifest identities before any model/backend initialization.
2. The fixed-token transport gate performs one non-production LoRA update,
   then verifies on each of the seven actual vLLM workers that the translated
   v2 adapter has a nonzero policy-minus-base effect matching the HF/PEFT
   coordinator. It is a short 3 × 1, 32-token probe, not an HLE or RMCT step.
3. The target hand-off gate writes an immutable sidecar binding the authored
   YAML bytes, entire factory-compiled selected entry, final training-child
   argv, interpreter argv[0] plus executable hash, training-script hash, full
   `CUDA_VISIBLE_DEVICES` allocation (including coordinator 0), source-proof
   sidecar, and worker-parity sidecar. The runner rechecks its second YAML
   read, and `train_rlct.py`/`train_opct.py` independently replay the source
   proof and worker proof before loading data or a backend.

Use the target-specific launcher rather than invoking a plan directly. It
binds the source, run namespace, worker topology, and worker options in one
place, then starts only the selected target. Its dry-run does not write files,
initialize a model, or probe a GPU.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CTM_PYTHON=/path/to/the/configured/python

# No-GPU / no-write contract previews.
bash infra/vastai/run_qwen35_onpolicy_recovery.sh rmct-main --dry-run
bash infra/vastai/run_qwen35_onpolicy_recovery.sh rmct-control --dry-run
bash infra/vastai/run_qwen35_onpolicy_recovery.sh opct --dry-run

# Independent eight-GPU RMCT paper-fidelity targets.
bash infra/vastai/run_qwen35_onpolicy_recovery.sh rmct-main
bash infra/vastai/run_qwen35_onpolicy_recovery.sh rmct-control

# Separate fresh OPCT-only recovery namespace; never reuse the mixed 20260801
# RMCT/OPCT plan, whose RMCT branch had invalid b1/accum4/per-item behavior.
bash infra/vastai/run_qwen35_onpolicy_recovery.sh opct
```

If a machine stopped after the worker preflight completed but before a
production rollout session was created, use the same target with
`--resume-attestation`. It revalidates the existing sidecar against the exact
model, worker topology/options, and hash-bound raw/translated adapter bytes
without repeating the probe. Any direct
`rollout_workers/session-*/` directory—including an empty one created before
IPC or adapters—marks production residue and blocks this mode. The bootstrap's
nested `rollout_workers/workers/session-*/` evidence is intentionally not a
production marker. The launcher also writes
`rollout_workers/qwen35-onpolicy-training-started.json` immediately before the
runner: it blocks resume even if the process dies before it creates a session
directory. Use a fresh experiment/run identity rather than overwrite or restart
a production namespace.

For the two RMCT targets, the immutable worker sidecars are:

- `logs/rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803/rate-matching-lr-1e-4/rollout_workers/qwen35-rollout-worker-parity-attestation.json`
- `logs/rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803/rate-matching-control-lr-1e-4/rollout_workers/qwen35-rollout-worker-parity-attestation.json`

Each run also writes its source identity record under
`logs/<experiment>/<run>/preflight/qwen35-recovered-none-source-attestation.json`
and its target-to-child hand-off record under
`logs/<experiment>/<run>/preflight/qwen35-onpolicy-target-attestation.json`.
The normal Qwen3.5 worker backend refuses to start without a valid worker
sidecar, and records its hash plus the raw and translated adapter hashes for
every policy snapshot. Keep the preflight's nested `workers/session-*` source
snapshot and log beside that sidecar; do not copy a sidecar between targets or
alter worker options/status directories after creation. The older fixed
`preflight_qwen35_rmct_rollout_workers.sh` remains a compatibility/manual
transport diagnostic, but it is not the production launcher because it lacks
the no-CoT source gate.

## Method and decoding choices

For this provisional comparison, RMCT, BCT, ACT, AttCT, MLPCT, and OPCT all use
`1e-4`. This deliberately removes method-specific learning-rate selection from
the initial matrix; any sensitivity analysis can be authored as a later run.
OPCT samples from the biased student field and scores under the frozen run-start
policy using the unbiased teacher field. A clean-to-clean OPCT run is a non-zero
self-distillation intervention, so no OPCT control is included unless one is
explicitly authored.

Every trained condition uses rank-8, alpha-16 LoRA over attention and MLP
modules. Both selected models have dense FFNs, so MLPCT uses `variant: hidden`
to compare the input to each down projection; it does not use the fused-MoE
output workaround from the GPT-OSS experiment.

Evaluation retains the decoding frozen by the repository's TBSR screen:

| Model | Temperature | Top-p | Top-k | Max tokens |
|---|---:|---:|---:|---:|
| Qwen3.5-9B | 1.0 | 0.95 | 20 | 20,480 |
| Qwen3-8B | 0.6 | 0.95 | 20 | 20,480 |

The same generation configuration is passed to the base model and every local
checkpoint. In particular, these thinking models are not evaluated greedily.

Qwen3.5-9B is a hybrid model with 24 Gated DeltaNet layers and eight full
attention layers. The current AttCT hook observes the eight recorded
full-attention tensors only; it does not turn DeltaNet state into attention
tensors. Interpret Qwen3.5 AttCT as consistency over that explicitly narrower
scope. Qwen3.5 remains on the default text-only `AutoModelForCausalLM` route,
which avoids loading an unused vision encoder. Its local configuration also
sets `vllm_language_model_only: true`; this forwards vLLM's explicit
`language_model_only=True` option so rollout sampling skips the vision encoder
and multimodal profiling as well.

## Compatibility gate

Provision a local/Vast host using [`../../infra/vastai/README.md`](../../infra/vastai/README.md)
as the baseline. The Qwen3.5 model card requires a current Transformers build
and a vLLM nightly/main build with Qwen3.5 support; a stale generic Vast image
is not sufficient. Before the full matrix, confirm the installed versions and
complete both model-specific smoke plans. The Qwen3.5 smoke is the required
end-to-end gate for model loading, LoRA target discovery, vLLM sampling,
checkpoint reload, and the eight-layer AttCT scope.

The run also requires a Hugging Face token for HLE, `OPENROUTER_API_KEY` for
missing HLE evaluation arguments and verbalisation grading, and `WANDB_API_KEY`
because the plans opt into W&B. Training arguments come from the imported
legacy stores. Keep credentials in `.env`; never place them in YAML.

## Offline review and dry runs

These commands only compile and print plans. They do not initialize a model,
call a provider, create a Vast instance, or start a paid job:

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/shared_data.yaml --dry-run
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_5_9b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --dry-run
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_8b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --dry-run
```

Use the same pattern for the three `debug/` YAMLs. The focused offline contract
test is:

```bash
python -m pytest -q tests/test_rmct_paper_vast_dense_models.py
```

## Vast execution order

Creating or keeping a Vast instance and running either non-dry plan is the
paid/external side-effect boundary. Review the selected offer, storage, API
costs, and a spending limit before crossing it. This repository change does
not call `vastai create instance`, provision a host, generate remote data, or
launch any paid training/evaluation job.

When sharing an existing eight-GPU host with another experiment on physical
GPUs 1, 2, and 3, reserve this experiment to physical GPUs 0, 4, 5, 6, and 7.
Use a separate checkout, virtual environment, tmux session, temporary
directory, and artifact/log roots; never reboot, stop, or destroy the shared
instance. Both the ambient visibility boundary and the runner allocation must
be explicit:

```bash
export CUDA_VISIBLE_DEVICES=0,4,5,6,7
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_5_9b_smoke.yaml \
  --parallel 5 --gpus 0,4,5,6,7 --yes
```

The runner writes the physical GPU ID passed through `--gpus` into each child
process, so do not renumber that list to `0,1,2,3,4` on this shared host.

Inside an explicitly approved and provisioned Vast host, use a persistent
`tmux` session. First run the tiny shared-data plan once, then the two smoke
plans one at a time:

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/shared_data_smoke.yaml --yes
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_5_9b_smoke.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_8b_smoke.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
```

After both smokes and the workload review pass, run the full shared-data plan
once, then each full model plan. Do not run `shared_data.yaml` a second time.

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/shared_data.yaml --yes
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_5_9b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_8b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
```

Stage barriers ensure the two model-dependent target generators finish before
training, all eight checkpoints finish before evaluation, and all nine evaluations
finish before aggregation and rendering. The chart definitions are reused by
path from `experiments/rmct_paper_vast_more_methods/`; they are not copied.

## Qwen3.5-9B Stage 1 multi-node allocation

The production Qwen3.5 plan is
[`stage1/qwen3_5_9b.yaml`](stage1/qwen3_5_9b.yaml). It retains the full 2,048
training examples, 2,048 instruction examples, a fixed 100-question HLE pool
with the explicitly approved 92-pair floor for `wrong_argument`, all eight
trained conditions, and the common `1e-4` learning rate. Its
`execution.allocations` section changes placement only.

All nodes must use the same immutable checkout and a shared or explicitly
synchronized `artifacts/` and `logs/` tree. Run the target generators once,
then start the eight training targets. The two RMCT targets each require an
eight-GPU node: logical GPU 0 coordinates training and logical GPUs 1-7 own
independent rollout workers. OPCT also uses an eight-GPU bundle with one
coordinator and seven workers. Dedicated RMCT/OPCT workers run vLLM at
`gpu_memory_utilization=0.75`; the training coordinator retains the colocated
`0.34` limit. BCT, BCT control, ACT, AttCT, and MLPCT each have an independent
one-GPU training target; every evaluation also remains a one-GPU process. Each
frozen-base target generator uses a four-GPU bundle
with no GPU coordinator: logical GPUs `0,1,2,3` are all independent
`enable_lora=False` vLLM workers, while the parent process only partitions and
restores prompt order.

```bash
PLAN=experiments/rmct_paper_vast_dense_models/stage1/qwen3_5_9b.yaml

# Preparation owner: two independent four-GPU commands (eight GPUs total).
python scripts/run_experiment.py "$PLAN" --stages data_preparation \
  --target data-preparation --parallel 2 --gpus 0,1,2,3,4,5,6,7 --yes

# Two separate 8-GPU nodes.
python scripts/run_experiment.py "$PLAN" --stages training \
  --target rmct-main --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
python scripts/run_experiment.py "$PLAN" --stages training \
  --target rmct-control --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes

# Five independent one-GPU targets (shown on separate one-GPU allocations).
# Sequential runner invocations inherit CUDA visibility from the environment.
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py "$PLAN" --stages training \
  --target bct-main --parallel 1 --yes
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py "$PLAN" --stages training \
  --target bct-control --parallel 1 --yes
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py "$PLAN" --stages training \
  --target act --parallel 1 --yes
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py "$PLAN" --stages training \
  --target attct --parallel 1 --yes
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py "$PLAN" --stages training \
  --target mlpct --parallel 1 --yes

# One 8-GPU node for OPCT: GPU 0 coordinates training and GPUs 1-7 sample.
python scripts/run_experiment.py "$PLAN" --stages training \
  --target opct --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
```

Each target writes only
`logs/experiments/rmct_paper_vast_dense_qwen3_5_9b_stage1/targets/<target>/resolved-plan.yaml`
and its sibling `outputs.json`; target workers never read-modify-write the
canonical state. After all eight training targets finish, the designated
`stage1-coordinator` is the sole publication owner and runs this local merge:

```bash
python scripts/run_experiment.py "$PLAN" --publish-training-outputs --yes
```

The merge validates that each target published exactly its assigned checkpoint
set, records source-state hashes, and atomically creates the canonical
`outputs.json`. It refuses a missing, stale, conflicting, or previously
different canonical state. If a prior attempt exists, move its experiment log
directory to `_archive/` and use a deliberately new experiment identity; do
not have multiple nodes publish.

Only after publication may the evaluation node resolve all eight checkpoint
placeholders. Each evaluation still requests one GPU; eight run concurrently
and the ninth follows when a GPU is released. Analysis and rendering remain
ordered CPU work owned by the coordinator.

```bash
python scripts/run_experiment.py "$PLAN" --stages evaluation \
  --target evaluation --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
python scripts/run_experiment.py "$PLAN" --stages analysis,rendering --yes
```

Do not launch the unscoped training stage concurrently on multiple nodes: that
legacy mode intentionally uses the single canonical `outputs.json` and is not
the multi-node protocol.

## Artifacts, logs, and resumption

Shared inputs live under `artifacts/rmct-hle-dense-models-shared/`. The two
model-specific artifact roots are `artifacts/rmct-hle-qwen3.5-9b-dense/` and
`artifacts/rmct-hle-qwen3-8b-dense/`; they contain generated targets and report
JSON. Figures use the matching roots beneath `figures/`.

Training metrics, manifests, OPCT rollout records, and adapter checkpoints are
under `logs/<experiment>/<run>/`. Evaluation logs are under
`logs/evals/<experiment>/`. The orchestration state and immutable expanded plan
are written to `logs/experiments/<experiment>/outputs.json` and
`resolved-plan.yaml`. Copy all logs, artifacts, figures, and the shared data
root off an ephemeral Vast host before destroying it.

After all eight checkpoints for one model appear in its `outputs.json`, evaluation
can resume without retraining:

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_5_9b.yaml \
  --start-from evaluation --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
```

Replace the YAML for Qwen3-8B. `--start-from analysis` and
`--start-from rendering` provide later stage boundaries. Local OPCT does not
support mid-run checkpoint resume because its frozen teacher is the run-start
policy; the supported experiment-level resume begins after the complete
training stage. A partial training-stage retry needs deliberate archival and a
new experiment name/path so existing immutable outputs are never overwritten.

The provisional portfolio evaluates 18 model states and each state sees the
shared HLE suite (100 questions under six biases, plus the unbiased evaluation).
OPCT requests 8,192 online student completions per model before retries. Measure
both smokes before committing the full 16-checkpoint matrix.
