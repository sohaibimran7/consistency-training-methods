# Isambard-AI launchers (GH200, AArch64)

Isambard and Vast.ai use the same `LocalBackend` implementation in
`ctm/backends/local/`. Isambard jobs are submitted through SLURM.

## One-time setup

```bash
mkdir -p "$PROJECTDIR/$USER"
git clone https://github.com/sohaibimran7/consistency-training-methods.git \
    "$PROJECTDIR/$USER/consistency-training-methods"
cd "$PROJECTDIR/$USER/consistency-training-methods"
bash infra/isambard/setup_env.sh     # uv venv + base deps + peft (aarch64)
cp /path/to/.env .                   # grader provider keys; add WANDB_API_KEY only when W&B is enabled
```

The setup script places Hugging Face and uv caches under `$SCRATCHDIR/ctm`.
Keep the checkout, virtual environment, logs, and durable artifacts under
`$PROJECTDIR`; do not use the smaller home filesystem for model weights. The
login-node step deliberately installs CPU PyTorch; the next step replaces it
with the CUDA build selected for the allocated GH200.

Finish the GPU-specific installation from an interactive allocation. This is
required because `--torch-backend=auto` must see a GH200 to select the correct
CUDA-enabled PyTorch and vLLM wheels:

```bash
srun --reservation=interactive --nodes=1 --gpus=1 \
  --time=00:30:00 --pty bash -i
cd "$PROJECTDIR/$USER/consistency-training-methods"
bash infra/isambard/setup_gpu_env.sh
```

## Interactive reservation

On Isambard-AI Phase 2, `interactive` is a **reservation**, not a partition or
QOS. Request it with `--reservation=interactive`; do not add
`--partition=interactive` or an `interactive_qos` setting. For an interactive
shell:

```bash
srun --reservation=interactive --nodes=1 --gpus=4 \
  --cpus-per-gpu=16 --mem=200G --time=02:00:00 --pty bash -i
```

The same reservation can run a non-production batch preflight while retaining
durable Slurm logs and artifacts:

```bash
sbatch --reservation=interactive \
  --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/preflight_qwen35_phase_shared.sbatch
```

Current service constraints:

- at most 4 nodes / 16 GPUs per interactive job;
- at most 8 hours per interactive job;
- at most one queued or running interactive job per user; and
- interactive usage has a 50% allocation premium (one node-hour consumes 1.5
  node-hour resources).

Thus, the reservation permits **one multi-node job**, rather than one node at a
time: for example, a single job may request four nodes (up to 16 GPUs), subject
to availability. It does not permit several simultaneous one-node interactive
jobs for the same user.

Use the reservation for environment setup, debugging, short benchmarks, and
non-production validation. Submit sustained training and evaluation to the
normal batch queue. Before replacing a pending normal-queue copy, first submit
the interactive job and confirm that it is `RUNNING`; then cancel only the
duplicate so the workload cannot execute twice.

Useful read-only checks are:

```bash
scontrol show reservation interactive
sbatch --test-only --reservation=interactive --nodes=1 --gpus=4 \
  --cpus-per-gpu=16 --mem=200G --time=06:00:00 --wrap=true
```

The second command forecasts placement without creating a queued job. See the
[Isambard interactive-reservation guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/#interactive-reservation-isambard-ai-phase-2)
and [FAQ](https://docs.isambard.ac.uk/user-documentation/faqs/#how-do-i-run-interactively-on-a-compute-node)
for the current site policy.

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
  `setup_gpu_env.sh` installs the released vLLM 0.21.0 CUDA 12.9 Arm64 wheel,
  because the older 0.10.2 recipe predates Qwen3.5 support. It performs the
  CUDA-dependent install only where a GH200 is visible and verifies the
  Qwen3.5 vLLM model class. Its companion constraints also keep the vLLM
  FastAPI/Starlette stack compatible with the repository's optional Streamlit
  UI, so the dependency gate remains strict. The four-GPU `--preflight-only`
  transport probe is still mandatory before on-policy training.
- **Memory:** A Phase 2 GH200 Superchip with 120 GB of HBM can accommodate gpt-oss-20b LoRA
  training and a colocated vLLM engine when
  `--local-gpu-mem-util 0.45` is used.
- **Device count:** `LocalBackend` can use a Transformers/Accelerate device map
  for single-process, layer-wise GPU placement. It does not implement FSDP,
  expert parallelism, or tensor parallelism; gpt-oss-120b training is therefore
  unsupported.

## Isolated GPT-OSS training-memory probe

`probe_training_memory.sbatch` runs a six-case matrix: sequence lengths 4,096,
8,192, and 9,728, each with gradient checkpointing disabled and enabled. Every
case receives a fresh Python process and four GPUs. The probe reports current
and peak allocator memory for every GPU, uses the real LocalBackend BCT loss
path, and includes the first AdamW step.

```bash
sbatch --export=ALL,REPO_DIR="$PWD" infra/isambard/probe_training_memory.sbatch
```

The final `PROFILE_RESULT_JSON=` line in each `slurm-mem-isolated-<job>_<task>.out`
file is machine-readable. CUDA OOM is a measured outcome and therefore exits
successfully; other exceptions fail the array task.

## Qwen3.5 on-policy recovery on one four-GPU Phase 2 node

The `run_qwen35_onpolicy_recovery_phase2.sbatch` wrapper is the explicit
four-GPU variant for the fresh Qwen3.5 OPCT, RMCT, and RMCT-control recovery
runs. It assigns GPU 0 to the HF/PEFT training coordinator and GPUs 1--3 to
independent vLLM rollout workers. It keeps the frozen source, 20,480-token
cap, optimizer settings, and rollout counts unchanged from the eight-GPU
plans; only rollout throughput and worker-local random streams differ.

Before submitting, copy the frozen recovered source into the checkout at its
declared `artifacts/rmct-hle-dense-models-shared-qwen3.5-none-20260801/`
location and validate the GPU environment in a one-GPU allocation. The first
four-GPU `--preflight-only` job performs a short real HF/PEFT-to-vLLM transport
probe on all three workers, records immutable source/worker/target
attestations, and exits before training. A preflight failure writes no
production rollouts.

```bash
# This has no production rollouts or training side effects.
sbatch --time=01:00:00 --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/run_qwen35_onpolicy_recovery_phase2.sbatch opct --preflight-only

# This consumes the attestation above and starts the selected target.
sbatch --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/run_qwen35_onpolicy_recovery_phase2.sbatch opct --resume-attestation

# Later, use distinct jobs/namespaces for these targets:
sbatch --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/run_qwen35_onpolicy_recovery_phase2.sbatch rmct-main --preflight-only
sbatch --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/run_qwen35_onpolicy_recovery_phase2.sbatch rmct-control --preflight-only
```

The wrapper requires exactly four Slurm-visible GPUs and deliberately does not
pass a numeric `--gpus` argument to the child runner, so Slurm UUID device
identifiers remain valid. Do not use `--resume-attestation` after a training
marker or rollout session exists; start a fresh target namespace instead.
The default request is 24 hours, the current `workq` maximum; do not submit a
36-hour request.

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
