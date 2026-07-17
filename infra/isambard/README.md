# Isambard-AI launchers (GH200, AArch64)

Isambard and Vast.ai use the same `LocalBackend` implementation in
`ctm/backends/local/`. Isambard jobs are submitted through SLURM.

## One-time setup

```bash
git clone <repo-url> ~/consistency-training-methods
cd ~/consistency-training-methods
bash infra/isambard/setup_env.sh     # uv venv + deps + peft + vllm (aarch64)
cp /path/to/.env .                   # grader provider keys; add WANDB_API_KEY only when W&B is enabled
```

Data is external to the repository. Copy each JSONL/manifest pair to an explicit
path or run the appropriate `python -m ctm_data.adapters.<name>.builder` command
before submitting a job. Record the resulting path in the experiment or setting
configuration.

## Launch a training run

Configure the account, partition, and `REPO_DIR` values in
`train_rlct.sbatch`. The batch launcher
is non-interactive and supplies `-y`, so first review the identical run locally:

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

## Platform notes

- **Architecture:** PyTorch and vLLM wheels must support AArch64.
  `setup_env.sh` first attempts the standard package indexes. If vLLM cannot be
  installed, use `--local-sampler hf` for diagnostic runs until a compatible
  vLLM or NVIDIA NGC environment is available.
- **Memory:** A GH200 node with 96 GB of HBM can accommodate gpt-oss-20b LoRA
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
