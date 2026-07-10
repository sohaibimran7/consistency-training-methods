# Vast.ai launchers (rented x86 CUDA GPUs)

Same `LocalBackend` as Isambard (`ctm/backends/local/`); launch mechanics here
are docker + SSH instead of SLURM.

## Instance sizing

| Model | Training (LoRA bf16) | + colocated vLLM sampler | Recommendation |
|---|---|---|---|
| gpt-oss-20b | ~45 GB | `--local-gpu-mem-util 0.45` | 1× H100/A100 **80GB** (tight) or 96GB |
| gpt-oss-120b | — | — | **not yet supported** (LocalBackend is single-device; no FSDP/TP) |

## Bring up an instance

```bash
pip install vastai && vastai set api-key <key>
vastai search offers 'gpu_name=H100_SXM num_gpus=1 disk_space>100'
vastai create instance <offer-id> \
    --image vllm/vllm-openai:latest \
    --disk 120 --ssh --direct
# (or use your prebaked image from infra/vastai/Dockerfile for faster boot)
```

Then inside the instance:

```bash
curl -O https://raw.githubusercontent.com/<you>/consistency-training-methods/<branch>/infra/vastai/provision.sh
REPO_URL=https://<token>@github.com/<you>/consistency-training-methods.git bash provision.sh
cd /workspace/consistency-training-methods
# create .env (OPENAI_API_KEY for judges, WANDB_API_KEY); rsync dataset_dumps/ if LFS pointers
```

## Train

Always inside `tmux` (SSH drops kill bare processes):

```bash
python scripts/tinker_training/train_rl.py \
    --backend local --local-dtype bfloat16 --local-sampler vllm \
    --model openai/gpt-oss-20b \
    --bias-types wrong_few_shot --datasets truthfulqa \
    --n-datapoints 64 --prompt-style no_cot \
    --experiment-name rl_wfs_local --run-name vast-rlct-wfs -y
```

Outputs mirror Tinker runs: `logs/<exp>/<run>/` with WandB metrics, `manifest.json`,
`rollouts/` (inspectable via `ctm.evals.analysis.rollouts`), and
`checkpoints/<name>/` adapter dirs. **Pull artifacts off the box before destroying
the instance** (`rsync -av <instance>:.../logs/ ./logs/`).

## Evaluate a checkpoint

```bash
# on the box: serve the adapter
python -m sycophancy_eval_inspect.run_local_evals \
    --checkpoint file:///workspace/.../checkpoints/<name> \
    --base-model openai/gpt-oss-20b --print-serve-cmd | bash   # or run the printed cmd yourself

# then (same box or anywhere that reaches it):
python -m sycophancy_eval_inspect.run_local_evals \
    --checkpoint file:///workspace/.../checkpoints/<name> \
    --base-model openai/gpt-oss-20b --served-model rlct \
    --base-url http://localhost:8000/v1 \
    --name gpt-oss-20b-rlct-local \
    --bias-types wrong_few_shot --datasets truthfulqa --limit 200
```
