# Vast.ai launchers (x86 CUDA instances)

For the two-host EvalAwareBench Figure 6 generation run, use the pinned,
minimal-runtime instructions in [FIGURE6.md](FIGURE6.md). The training setup
below is intentionally separate.

Vast.ai uses the same `LocalBackend` implementation as Isambard. Instances are
provisioned through Docker and accessed through SSH rather than SLURM.

## Instance sizing

| Model | Training (LoRA bf16) | + colocated vLLM sampler | Recommendation |
|---|---|---|---|
| gpt-oss-20b | approximately 45 GB | `--local-gpu-mem-util 0.45` | One 96 GB GPU is preferred; an 80 GB H100 or A100 has limited headroom. |
| gpt-oss-120b | not applicable | not applicable | Unsupported because `LocalBackend` does not implement multi-device training. |

## Bring up an instance

```bash
pip install vastai && vastai set api-key <key>
vastai search offers 'gpu_name=H100_SXM num_gpus=1 disk_space>100'
vastai create instance <offer-id> \
    --image vllm/vllm-openai:latest \
    --disk 120 --ssh --direct
# Alternatively, use a prebuilt image based on infra/vastai/Dockerfile.
```

Then inside the instance:

```bash
curl -O https://raw.githubusercontent.com/<you>/consistency-training-methods/<branch>/infra/vastai/provision.sh
REPO_URL=git@github.com:<you>/consistency-training-methods.git bash provision.sh
cd /workspace/consistency-training-methods
# create .env with grader-provider credentials
# add WANDB_API_KEY only if the training command also sets --wandb-project
# copy immutable JSONL/manifest pairs to explicit paths, or run an adapter builder module
```

## Train

Run long jobs inside `tmux` so an SSH disconnection does not terminate the
training process. First inspect the resolved run without initializing a backend
or training:

```bash
python scripts/train_rlct.py \
    --backend local --local-dtype bfloat16 --local-sampler vllm \
    --model openai/gpt-oss-20b \
    --setting-factory ctm_data.adapters.mcq_bias:create_setting \
    --setting-config '{"data_paths":["/path/to/train.jsonl"]}' \
    --n-datapoints 64 \
    --experiment-name rl_wfs_local --run-name vast-rlct-wfs --dry-run
```

After approving that output, rerun the same command with `--dry-run` replaced
by `-y`.

For an experiment YAML containing platform-labelled entries, select the Vast.ai
commands inside the provisioned instance:

```bash
python scripts/run_experiment.py \
    experiments/mcq_bias/wrong_argument_cross_bias/bct_backends.yaml \
    --target vast --yes
```

Outputs mirror Tinker runs: `logs/<exp>/<run>/` contains local metrics,
`manifest.json`, complete `rollouts/` records, and `checkpoints/<name>/` adapter
directories. W&B is used only when `--wandb-project` is supplied. Copy artifacts
off the instance before destroying it (`rsync -av <instance>:.../logs/ ./logs/`).

For a matrix of independent runs, prefer one multi-GPU instance with shared
storage. Run one child process per GPU:

```bash
python scripts/run_experiment.py path/to/experiment.yaml \
    --parallel 8 --gpus 0,1,2,3,4,5,6,7 --yes
```

The runner preserves stage barriers and gives each concurrent GPU process an
exclusive `CUDA_VISIBLE_DEVICES` value. This keeps generated data and named
checkpoint metadata on one filesystem. Transfer `.env` to the instance through
SSH; do not commit it or place credentials in experiment YAML.

## Evaluate a checkpoint

The generic runner can load a LocalBackend LoRA checkpoint directly through the
Inspect Hugging Face provider:

```bash
python scripts/run_evals.py \
    --task-factory mcq_bias.tasks:suite_tasks \
    --local-checkpoint file:///path/to/checkpoint \
    --task-args '{"bias_types":["wrong_few_shot"],"datasets":["truthfulqa"]}' \
    --generation-config '{"max_tokens":32768}'
```
