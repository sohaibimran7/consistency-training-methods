# Isambard-AI launchers (GH200, aarch64)

Isambard and Vast.ai run the **same** `LocalBackend` (`ctm/backends/local/`);
only the launch mechanics differ — here it's SLURM.

## One-time setup

```bash
git clone <repo-url> ~/consistency-training-methods
cd ~/consistency-training-methods
bash infra/isambard/setup_env.sh     # uv venv + deps + peft + vllm (aarch64)
cp /path/to/.env .                   # OPENAI_API_KEY (judges), WANDB_API_KEY — no Tinker key needed
```

Data: the LFS budget is exceeded, so `dataset_dumps/*.jsonl` may be pointer files
on a fresh clone — copy them from a working machine (`rsync` the `dataset_dumps/`
dir) or regenerate via `scripts/eval_awareness/build_*.py`.

## Launch a training run

Edit `train_rlct.sbatch` once (account/partition/`REPO_DIR`), then:

```bash
sbatch infra/isambard/train_rlct.sbatch \
    --model openai/gpt-oss-20b \
    --bias-types wrong_few_shot --datasets truthfulqa \
    --n-datapoints 64 --prompt-style no_cot \
    --experiment-name rl_wfs_local --run-name gh200-rlct-wfs
```

Everything after the script name is forwarded to `train_rl.py`. Outputs land
exactly as on Tinker: `logs/<experiment>/<run>/` with WandB metrics,
`manifest.json`, `rollouts/`, and `checkpoints/<name>/` (file:// adapter dirs).

## Platform notes

- **aarch64 everywhere**: torch/vllm wheels must be arm64. `setup_env.sh` tries
  the generic wheels; if vLLM fails, `--local-sampler hf` unblocks debugging
  while you sort a platform build (NGC containers are the fallback).
- **Memory**: a GH200 node (96GB HBM + large unified LPDDR) fits gpt-oss-20b
  LoRA training + a colocated vLLM engine (`--local-gpu-mem-util 0.45`).
- **Current limit**: LocalBackend trains on a **single device** (no FSDP/TP yet)
  — gpt-oss-120b local training is not supported until that lands.

## Evaluating a checkpoint

Serve it on a GPU node, then run the suite (from anywhere that reaches the node):

```bash
python -m sycophancy_eval_inspect.run_local_evals \
    --checkpoint file://$PWD/logs/<exp>/<run>/checkpoints/<name> \
    --base-model openai/gpt-oss-20b --print-serve-cmd   # prints the vllm serve line
```
