# Isambard-AI launchers (Phase 2 GH200, AArch64)

Isambard and Vast.ai use the same `LocalBackend` implementation in
`ctm/backends/local/`. Isambard jobs are submitted through SLURM.

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
Do not update a checkout that contains unrelated or uncommitted work. The
Figure 6 runbook uses a separate clean path for this reason.

The setup script places Hugging Face and uv caches under `$SCRATCHDIR/ctm`.
Keep the checkout, virtual environment, logs, and durable artifacts under
`$PROJECTDIR`; do not use the smaller home filesystem for model weights. The
login-node step deliberately installs CPU PyTorch; the next step replaces it
with the CUDA build selected for the allocated GH200.

Finish the GPU-specific installation from an interactive allocation for
ordinary runs. `--torch-backend=auto` must see a GH200 to select the correct
CUDA-enabled PyTorch wheels. The Figure 6 prefetch job performs this same step
at the start of its existing GPU allocation, so it does not need a separate
interactive request.

```bash
srun --nodes=1 --gpus-per-node=1 --time=00:30:00 --pty /bin/bash --login
cd "$PROJECTDIR/$USER/consistency-training-methods"
bash infra/isambard/setup_gpu_env.sh
```

The GPU step installs the pinned standard-PyPI AArch64 wheel for vLLM 0.26.0,
PyTorch 2.11, and Transformers 5.5.4. It then checks the dependency set, CUDA,
bfloat16 support, the exact vLLM version, and the training backend import.

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
  predictable GPU allocation; the Figure 6 launchers follow it.
- **Training memory:** gpt-oss-20b LoRA training and its colocated sampler use
  the existing `--local-gpu-mem-util 0.45` setting.
- **Device count:** `LocalBackend` currently trains on one device and does not
  implement FSDP or tensor parallelism. Local gpt-oss-120b training is therefore
  unsupported.

The capacity snapshot recorded on 2026-07-29 is only a volatile observation:
there were no fully idle nodes, with mixed capacity across nodes. At
2026-07-29T13:03:58Z, Slurm `--test-only` estimated
2026-08-01T05:32:59Z (about 64 hours 29 minutes later) for the exact one-GPU
12-hour prefetch, one-GPU 24-hour Qwen array-task shape, and two-GPU 24-hour
Llama array-task shape. Two-hour pilot checks at 13:04:29Z estimated 05:33:29Z,
so shortening the request did not improve that snapshot. These estimates are
not reservations or guarantees; check again immediately before submission.

## EvalAwareBench Figure 6 generation

The seven-model target-generation run has dedicated manifests and launchers:

- `experiments/eval_awareness/figure6/models.yaml` pins every model revision,
  display label, comparison family/stage, prompt protocol, and tensor-parallel
  size.
- `experiments/eval_awareness/figure6/protocol.yaml` pins the 1,800 conditions,
  three samples per condition, generation settings, dataset revision, and
  external prompt hashes.
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
