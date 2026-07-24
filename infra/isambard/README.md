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
srun --nodes=1 --gpus=1 --time=00:30:00 --pty /bin/bash --login
cd "$PROJECTDIR/$USER/consistency-training-methods"
bash infra/isambard/setup_gpu_env.sh
```

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
  `setup_gpu_env.sh` follows BriCS's supported vLLM 0.10.2 recipe and performs
  the CUDA-dependent install only where a GH200 is visible. Use
  `--local-sampler hf` for diagnostic runs until that step has passed.
- **Memory:** A Phase 2 GH200 Superchip with 120 GB of HBM can accommodate gpt-oss-20b LoRA
  training and a colocated vLLM engine when
  `--local-gpu-mem-util 0.45` is used.
- **Device count:** `LocalBackend` currently trains on one device and does not
  implement FSDP or tensor parallelism. Local gpt-oss-120b training is therefore
  unsupported.

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
