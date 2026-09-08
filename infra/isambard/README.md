# Isambard-AI launchers (Phase 2 GH200, AArch64)

Isambard and Vast.ai use the same `LocalBackend` implementation in
`ctm/backends/local/`. Isambard jobs are submitted through SLURM.

## Laptop authentication

Before remote work, use the local helper to check the SSH route and certificate:

```bash
python3 infra/isambard/auth.py status              # local metadata only
python3 infra/isambard/auth.py check               # fresh SSH, runs only true
python3 infra/isambard/auth.py renew               # explicit Clifton device flow
```

The default project alias is `a5v.aip2.isambard`; pass another Clifton alias as
the final argument when needed. Plain `isambard` is not a configured alias on
the shared laptop. On that laptop, the installed `isambard-auth` command exposes
the same subcommands from any checkout.

See [Authentication and renewal](AUTHENTICATION.md) for the browser flow,
failure diagnosis, and the current restriction on automated Lancaster sign-in.

## Shared laptop job coordination

Use the installed `isambard-jobs` controller for new agent-owned submissions
on the shared laptop. It queues interactive requests across worktrees, reserves
experiment output locations, and provides one cached scheduler view. Start with
`isambard-jobs status`; the [job controller guide](JOBS.md) documents preparing
RMCT/Gemma manifests, submitting them, registering existing jobs, and recovery.

Install independent copies with `python3 infra/isambard/install_jobs.py`.
State lives in `~/.local/share/ctm-isambard/jobs`. Keep that shared state path
for normal work. Existing Slurm jobs continue in place during installation.
The raw Slurm examples below describe the underlying cluster commands; agents
on the shared laptop should express new work through the controller.

## One-time setup

```bash
export REPO_DIR="$PROJECTDIR/$USER/consistency-training-methods"
git clone https://github.com/sohaibimran7/consistency-training-methods.git \
    "$REPO_DIR"
cd "$REPO_DIR"
bash infra/isambard/setup_env.sh     # uv venv + base deps + peft (aarch64)
cp /path/to/.env .                   # grader provider keys; add WANDB_API_KEY only when W&B is enabled
```

For a publication run, use a new checkout at an immutable reviewed revision.
Do not update a checkout that contains unrelated or uncommitted work.

The setup script places Hugging Face and uv caches under `$SCRATCHDIR/ctm`.
Keep the checkout, virtual environment, logs, and durable artifacts under
`$PROJECTDIR`; do not use the smaller home filesystem for model weights. The
login-node step deliberately installs CPU PyTorch; the next step replaces it
with the CUDA build selected for the allocated GH200.

Finish the GPU-specific installation from an interactive allocation for
ordinary runs. `--torch-backend=auto` must see a GH200 to select the correct
CUDA-enabled PyTorch wheels. The phase-shared Qwen3.5 preflight uses the
verified vLLM 0.21.0 CUDA-12.9 AArch64 wheel, Transformers 5.5.4, and the
strict Qwen3.5 import check installed by `setup_gpu_env.sh`.

```bash
srun --nodes=1 --gpus-per-node=1 --time=00:30:00 --pty /bin/bash --login
cd "$REPO_DIR"
bash infra/isambard/setup_gpu_env.sh
```

## Interactive reservation

On Isambard-AI Phase 2, `interactive` is a **reservation**, not a partition or
QOS. Request it with `--reservation=interactive`; do not add
`--partition=interactive` or an `interactive_qos` setting. For an interactive
shell:

```bash
srun --reservation=interactive --nodes=1 --gpus-per-node=4 \
  --cpus-per-gpu=16 --mem=200G --time=02:00:00 --pty bash -i
```

The same reservation can run a non-production batch preflight while retaining
durable Slurm logs and artifacts:

```bash
sbatch --reservation=interactive \
  --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/preflight_qwen35_phase_shared.sbatch
```

The documented constraints are at most four nodes / sixteen GPUs and eight
hours per interactive job, with at most one queued or running interactive job
per user. Interactive usage has a 50% allocation premium (one node-hour
consumes 1.5 node-hour resources). The reservation permits one multi-node job,
not multiple simultaneous one-node jobs for the same user.

Use it for environment setup, debugging, short benchmarks, and
non-production validation. Submit sustained training and evaluation to the
normal batch queue. Before replacing a pending normal-queue copy, first submit
the interactive job and confirm that it is `RUNNING`; then cancel only the
duplicate so the workload cannot execute twice.

Useful read-only checks are:

```bash
scontrol show reservation interactive
sbatch --test-only --reservation=interactive --nodes=1 --gpus-per-node=4 \
  --cpus-per-gpu=16 --mem=200G --time=06:00:00 --wrap=true
```

See the [interactive-reservation guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/#interactive-reservation-isambard-ai-phase-2)
and [FAQ](https://docs.isambard.ac.uk/user-documentation/faqs/#how-do-i-run-interactively-on-a-compute-node)
for current site policy.

Data is external to the repository. Copy each JSONL/manifest pair to an explicit
path or run the appropriate `python -m ctm_data.adapters.<name>.builder` command
before submitting a job. Record the resulting path in the experiment or setting
configuration.

## Launch a training run

The Clifton project login supplies the SLURM account association, and Isambard's
default `workq` partition is used when no partition is specified. The launcher
defaults `REPO_DIR` to `$PROJECTDIR/$USER/consistency-training-methods` and is
non-interactive, so first review the identical run locally:

```bash
python scripts/train_rlct.py --dry-run \
    --backend local --local-dtype bfloat16 --local-sampler vllm \
    --model openai/gpt-oss-20b \
    --setting-factory ctm_data.adapters.mcq_bias:create_setting \
    --setting-config '{"data_paths":["/path/to/train.jsonl"]}' \
    --n-datapoints 64 \
    --experiment-name rl_wfs_local --run-name gh200-rlct-wfs
```

Add `--wandb-project PROJECT` to both commands only when remote W&B logging is
required.

Only after approving that resolved configuration, submit the job:

```bash
sbatch infra/isambard/train_rlct.sbatch \
    --model openai/gpt-oss-20b \
    --setting-factory ctm_data.adapters.mcq_bias:create_setting \
    --setting-config '{"data_paths":["/path/to/train.jsonl"]}' \
    --n-datapoints 64 \
    --experiment-name rl_wfs_local --run-name gh200-rlct-wfs
```

Everything after the script name is forwarded to `scripts/train_rlct.py`.
Outputs are written to `logs/<experiment>/<run>/`, including local metrics,
`manifest.json`, complete `rollouts/`, and `checkpoints/<name>/` adapter
directories. W&B receives metrics only when `--wandb-project` is supplied.

For a multi-platform experiment YAML, use the generic launcher instead:

```bash
sbatch infra/isambard/run_experiment.sbatch \
    experiments/mcq_bias/wrong_argument_cross_bias/bct_backends.yaml \
    --target isambard --yes
```

This allocates the node and forwards the remaining arguments to
`scripts/run_experiment.py`. The target selector does not submit remote jobs or
copy artifacts.

## Platform notes

- **Architecture:** PyTorch and vLLM wheels must support AArch64.
  `setup_gpu_env.sh` performs the CUDA-dependent install only where a GH200 is
  visible. Use `--local-sampler hf` for diagnostic runs until it passes.
- **Phase 2 shape:** The documented system has 1,320 nodes and four GH200
  Superchips per node. Each Superchip contributes one GPU with 96 GB HBM,
  72 CPU cores, and 115 GB usable CPU memory.
- **Limits:** The documented `workq` maximum is 24 hours. The project maximum
  is 32 GPUs. Isambard accounts one NHR as four GPU-hours.
- **GRES:** Current known-issues guidance recommends `--gpus-per-node` for
  predictable GPU allocation.
- **Training memory:** gpt-oss-20b LoRA training and its colocated sampler use
  the existing `--local-gpu-mem-util 0.45` setting.
- **Device count:** ordinary `LocalBackend` training remains single-device.
  The experimental RMCT/RLCT phase-shared path uses replicated LoRA trainers;
  neither path implements FSDP or tensor parallelism. Local gpt-oss-120b
  training is therefore unsupported.

## Phase-shared Qwen3.5 preflight

`preflight_qwen35_phase_shared.sbatch` is an explicit, non-production
four-GPU GH200 validation path for the phase-shared local trainer. It leaves
all production datasets, checkpoints, and credentials untouched, and creates a
fresh evidence directory below `artifacts/qwen35-phase-shared-preflight/`.

```bash
sbatch --reservation=interactive --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/preflight_qwen35_phase_shared.sbatch
```

The harness requires a complete Slurm allocation, preserves Slurm's
`CUDA_VISIBLE_DEVICES` mapping (including UUID mappings), and validates the
following sequence: an HF/PEFT trainer and vLLM worker coexist on each GPU;
vLLM level-1 sleep/wake releases and reacquires worker memory; a fixed
single-rank reference update agrees with a four-rank manually replicated
NCCL-SUM update; and rank-zero capacity probes run with all workers asleep.
It is not a production training launcher or a performance benchmark.

The initial successful non-production run was Isambard job `5989987` on four
GH200 GPUs. It passed the pre-optimizer gradient gate (cosine `0.99999915`,
norm ratio `0.9999873`), sleep/wake and worker-effect checks, and the
20,480/40,960/49,152-token capacity-only sweep. The sweep does not establish
throughput or choose between three-lane and four-lane production topologies.

## GPU environment profiles

`setup_gpu_env.sh` defaults to `--profile training` (the phase-shared
vLLM 0.21.0 CUDA 12.9 stack). Figure 6 launchers explicitly select
`--profile figure6` (vLLM 0.26.0 / PyTorch 2.11), using
`vllm-figure6-constraints.txt`. Keep these in separate clean checkouts and
virtual environments. A successful setup records its profile in `.venv/`
and refuses to convert that environment to the other profile.

These are distinct historical runtime requirements. Neither profile upgrades
an existing campaign directory as part of consolidation. The shared install,
dependency check and GPU import gates run only when setup is explicitly invoked.

## EvalAwareBench Figure 6 generation

The seven-model target-generation run has dedicated manifests and launchers:

- `experiments/eval_awareness/figure6/models.yaml` pins every model revision,
  display label, comparison family/stage, prompt protocol, and tensor-parallel
  size.
- `experiments/eval_awareness/figure6/protocol.yaml` pins the 1,800 conditions,
  three samples per condition, generation settings, dataset revision, and
  prompt hashes. The completed Qwen inputs are tracked under the adjacent
  `inputs/` directory; the deferred Llama scratchpad prompt remains external.
- `experiments/eval_awareness/figure6/README.md` is the end-to-end runbook.

The cached model snapshots are approximately 1,088.8 GB. Verify at least
1,300 GB of usable shared project/scratch cache capacity before prefetching.
On 2026-07-29 the proposed scratch path reported about 4.9 TB filesystem-wide
free and the project path about 200 TB, but `lfs quota -u` showed only the
default rather than a personal limit; project-owner confirmation is still
required.
All models serve in bfloat16; the MO checkpoints stored in float32 are
downcast while loading. The standard Llama prefetch excludes `original/*.pth`
so the listed 141.1 GB safetensors estimate remains meaningful.

One 24-hour all-model target array pass has a ceiling of 240 GPU-hours, or
60 NHR. The optional 12-hour one-GPU prefetch adds at most 3 NHR, making the
combined single-pass ceiling 63 NHR. Pilot consumption is separate and should
end as soon as its 300 generations per model complete.

## Evaluating a checkpoint

The generic runner can load a LocalBackend LoRA checkpoint directly through the
Inspect Hugging Face provider. This requires `peft`, which the setup script
installs:

```bash
python scripts/run_evals.py \
    --task-factory mcq_bias.tasks:suite_tasks \
    --local-checkpoint file:///path/to/checkpoint \
    --task-args '{"bias_types":["wrong_few_shot"],"datasets":["truthfulqa"]}' \
    --generation-config '{"max_tokens":32768}'
```
