# Vast.ai launchers (x86 CUDA instances)

Vast.ai uses the same `LocalBackend` implementation as Isambard. Instances are
provisioned through Docker and accessed through SSH rather than SLURM. No
repository script searches for, creates, or destroys an instance
automatically; the CLI commands below are manual review gates.

## Instance sizing

Each local training process colocates a bf16 Transformers/PEFT training model
and a vLLM sampler on one GPU. The dense-Qwen matrix therefore uses independent
single-GPU runs, not tensor parallelism. A four-GPU smoke runs up to four jobs
at a time; the full plan can run four waves at a time or use an eight-GPU host.
The runner gives every child an exclusive `CUDA_VISIBLE_DEVICES` value while
all children share the machine's local filesystem.

On a shared eight-GPU instance where another job owns physical GPUs 1, 2, and
3, use a separate workspace and set both
`CUDA_VISIBLE_DEVICES=0,4,5,6,7` and runner argument
`--parallel 5 --gpus 0,4,5,6,7`. The runner assigns the listed physical IDs
directly to child processes; do not pass logical `0,1,2,3,4`. Never reboot,
stop, or destroy a shared instance, and serialize first-time model downloads.

| Model | bf16 weights alone | Architecture relevant to training | Per-run sizing |
|---|---:|---|---|
| `Qwen/Qwen3.5-9B` | about 18 GB | Dense FFNs; 32 hybrid layers: 24 DeltaNet/linear-attention and 8 full-attention layers | At least 80 GB VRAM. H200 (141 GB) is safer for the configured long sequences; start vLLM at `gpu_memory_utilization: 0.45`. |
| `Qwen/Qwen3-8B` | about 16 GB | Dense FFNs; 36 full-attention layers | At least 80 GB VRAM; start vLLM at `gpu_memory_utilization: 0.45`. |
| `openai/gpt-oss-20b` | model-dependent MoE allocation | Existing supported path | One 96 GB GPU is preferred; an 80 GB H100 or A100 has limited headroom. |
| `openai/gpt-oss-120b` | not applicable | Multi-device training required | Unsupported because `LocalBackend` does not implement multi-device training. |

The weight figures are lower bounds, not training-memory estimates. LoRA
activations, optimizer state, loss-specific captured activations/attentions,
the second vLLM copy, and its KV cache also consume VRAM. Treat 80 GB as the
minimum and measure the smoke configurations with `nvidia-smi`; prefer H200 or
reduce sequence length/sampler utilization if an 80 GB smoke has inadequate
headroom.

Allocate at least **500 GB of local disk**. That covers both model snapshots,
the Qwen3.5 nightly Python/CUDA environment and caches, shared prepared data,
parallel rollout logs, and adapter checkpoints with retry headroom. Select 1 TB
instead when retaining multiple complete matrices on the host. Put
`HF_HOME`, `UV_CACHE_DIR`, the checkout, data, checkpoints, and logs below
`/workspace` so every GPU process sees the same local storage.

## Search, review, and only then rent

Install the CLI and register an SSH key on the controller. Keep the API key in
the CLI's credential store; never put it in this repository or an experiment
YAML.

```bash
python -m pip install --upgrade vastai
vastai set api-key <key>
vastai create ssh-key ~/.ssh/id_ed25519.pub
```

This is a read-only, on-demand search for verified machines with at least 99%
reliability, eight H100 SXM or H200 GPUs, at least 80 GB VRAM per GPU, 500 GB
available disk, and a direct port. `--storage=500` makes the displayed total
hourly price account for the intended disk allocation; ascending
`dph_total` orders offers by total hourly price, not per-GPU price.

```bash
vastai search offers \
  'gpu_name in [H100_SXM,H200] num_gpus=8 gpu_ram>=80 disk_space>=500 reliability>=0.99 verified=true rentable=true rented=false direct_port_count>=1' \
  --type=on-demand --storage=500 --order=dph_total --limit=20
```

For the integration smoke, use the same reliability, storage, direct-SSH, and
total-price rules with four 80 GB A100s. This second command is also read-only:

```bash
vastai search offers \
  'gpu_name in [A100_SXM4,A100_PCIE] num_gpus=4 gpu_ram>=80 disk_space>=500 reliability>=0.99 verified=true rentable=true rented=false direct_port_count>=1' \
  --type=on-demand --storage=500 --order=dph_total --limit=20
```

Before any paid action, record and review the exact offer ID and row: GPU model
and count, per-GPU RAM, reliability/verification, driver/CUDA capability,
direct-port availability, disk/bandwidth terms, rental duration, and
`dph_total`. Marketplace availability and prices are live; this document does
not assert a price. Do not substitute a remembered offer ID or rate.

Only after that review, the paid operation has this shape:

```bash
# PAID: do not run until <reviewed-offer-id> and its current dph_total were approved.
vastai create instance <reviewed-offer-id> \
  --image vllm/vllm-openai:latest \
  --disk 500 --ssh --direct
```

The default image/path remains useful for stable, non-Qwen3.5 work. The
Qwen3.5 switch below upgrades that instance explicitly; it is never selected
implicitly.

## Provision the checkout

Inside the instance, place caches on shared local storage and run the
provisioner. Use a deploy key or another non-committed Git credential for
`REPO_URL`.

```bash
export HF_HOME=/workspace/cache/huggingface
export UV_CACHE_DIR=/workspace/cache/uv
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

curl -O https://raw.githubusercontent.com/<you>/consistency-training-methods/<branch>/infra/vastai/provision.sh
REPO_URL=git@github.com:<you>/consistency-training-methods.git bash provision.sh
```

That is the unchanged default behavior: use the image's vLLM when present, or
install a stable vLLM package when absent.

Qwen3.5 currently requires Transformers main/latest and vLLM main/nightly. Its
[model card](https://huggingface.co/Qwen/Qwen3.5-9B) and the
[vLLM GPU installer](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
recommend the nightly index with `uv ... --torch-backend=auto`; the latter
selects Torch for the visible GPU driver/CUDA environment. Opt in explicitly:

```bash
QWEN35_COMPAT=1 \
QWEN35_INSTALL_FAST_KERNELS=1 \
REPO_URL=git@github.com:<you>/consistency-training-methods.git \
bash provision.sh
```

`QWEN35_INSTALL_FAST_KERNELS=1` is deliberately separate. It installs
`flash-linear-attention` (import `fla`) and `causal-conv1d` (import
`causal_conv1d`) only after vLLM has resolved Torch. Without both, Transformers
warns and uses a much slower Torch implementation for the 24 Qwen3.5 DeltaNet
layers. Before installing, the provisioner installs ordinary Debian build
tools when `apt-get` is available, then requires Git, C/C++ compilers, Ninja,
`nvcc`, and CUDA development headers. The supplied Dockerfile also preinstalls
the ordinary build tools; the upstream vLLM image supplies its matching CUDA
toolkit. A generic runtime-only image fails with an actionable error instead of
silently falling back. The preflight below also rejects a missing, too-old, or
ABI-incompatible kernel. Rerunning either provisioning mode is safe; the Qwen
mode updates the explicitly floating main/nightly packages and prints the
versions it resolved.

Create `.env` in the checkout with only the credentials required by the
selected stages. `WANDB_API_KEY` is needed only when the configuration enables
W&B. Never commit `.env`, copy API keys into YAML, or paste them into logs.

For an explicitly synced checkout without Git metadata, retain the image's
resolved Torch/vLLM stack and build an isolated environment with:

```bash
bash infra/vastai/setup_synced_validation_env.sh
```

The script defaults `TORCH_CUDA_ARCH_LIST=9.0` for H100/H200 instead of
compiling unused GPU architectures. Override it when using a different GPU.

## Preflight and zero-execution plan checks

From the repository root, run the preflight before a Qwen smoke:

```bash
EXPECTED_GPUS=4 bash infra/vastai/preflight.sh | tee /workspace/qwen-preflight.txt
```

It prints exact `torch`, `transformers`, `vllm`, and `peft` versions (including
the Transformers Git commit when available), Torch CUDA/cuDNN, every visible
GPU and its memory, and the exact fast-kernel distribution/import versions. It
downloads only the two small Hugging Face config files, then verifies:

- the requested number of visible GPUs with at least 80 GB each;
- `fla` and `causal_conv1d` are accepted by Transformers on CUDA, so Qwen3.5
  will not use its slow fallback;
- `Qwen/Qwen3.5-9B` maps through `AutoModelForCausalLM` to the text-only
  `Qwen3_5ForCausalLM` compatibility class, with 24 linear-attention and 8
  full-attention layers; and
- `Qwen/Qwen3-8B` maps to `Qwen3ForCausalLM`.

The intended smoke host has four GPUs, hence `EXPECTED_GPUS=4`. Use the default
of eight on the eight-GPU full host. Do not lower the 80 GB memory requirement.
For a Git-metadata-free sync, set `CTM_SOURCE_REVISION` to the exact source
commit; preflight refuses an untraceable checkout.

After preflight, the following is a non-production capacity probe, not a
scientific experiment. It performs one real rank-8 LoRA update on GPU 0,
publishes the adapter to seven independent vLLM workers, generates 16 × 4
rollouts, verifies slot/logprob integrity, and writes timing, throughput, peak
memory, and per-worker JSONL logs. It downloads/loads eight model copies and
therefore still requires an explicit reviewed GPU budget.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python infra/vastai/benchmark_qwen35_parallel.py \
  --worker-gpus 1,2,3,4,5,6,7 \
  --sequence-length 512 --prompt-count 16 \
  --samples-per-prompt 4 --max-new-tokens 512
```

The following dry-runs compile and print every smoke and full plan but execute
no stage, load no model weights, contact no grader/model provider, and write no
experiment outputs. They can be run on a configured controller before renting,
then repeated on the instance:

First compile the smoke plans with the intended four-way schedule:

```bash
for config in \
  experiments/rmct_paper_vast_dense_models/debug/shared_data_smoke.yaml \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_5_9b_smoke.yaml \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_8b_smoke.yaml
do
  python scripts/run_experiment.py "$config" \
    --parallel 4 --gpus 0,1,2,3 --dry-run
done
```

Then compile the full plans with the intended eight-way schedule (use
`--parallel 4 --gpus 0,1,2,3` instead on an approved four-GPU full host):

```bash
for config in \
  experiments/rmct_paper_vast_dense_models/shared_data.yaml \
  experiments/rmct_paper_vast_dense_models/qwen3_5_9b.yaml \
  experiments/rmct_paper_vast_dense_models/qwen3_8b.yaml
do
  python scripts/run_experiment.py "$config" \
    --parallel 8 --gpus 0,1,2,3,4,5,6,7 --dry-run
done
```

Review the resolved commands, workload, artifact paths, credentials, and
provider spending limits before removing `--dry-run` from anything.

## Smoke first, then run the full matrices

Start `tmux` after exporting the shared cache variables so an SSH disconnect
does not kill a job. Detach with `Ctrl-b d` and reconnect with
`tmux attach -t qwen-smoke`.

```bash
tmux new-session -s qwen-smoke
```

Inside tmux, prepare the small shared fixture once, then run both model smokes.
These commands intentionally omit `--yes`; inspect and answer the runner's
final confirmation. The smoke stages can make configured provider calls, so
approve their small workload and provider budget too.

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/shared_data_smoke.yaml \
  --parallel 4 --gpus 0,1,2,3

python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_5_9b_smoke.yaml \
  --parallel 4 --gpus 0,1,2,3

python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/debug/qwen3_8b_smoke.yaml \
  --parallel 4 --gpus 0,1,2,3
```

Do not start the full plans until both smokes have completed, peak VRAM/disk
use and wall time have been recorded, outputs have been inspected and copied
off-host, and a full-run spending limit has been approved. Then prepare the
full shared data once and run the model matrices, still retaining the final
interactive confirmation:

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/shared_data.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7

python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_5_9b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7

python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_8b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7
```

The runner preserves stage barriers and stores its resolved plan and named
checkpoint metadata. After a completed stage boundary, resume later stages
without repeating training, for example:

```bash
python scripts/run_experiment.py \
  experiments/rmct_paper_vast_dense_models/qwen3_5_9b.yaml \
  --parallel 8 --gpus 0,1,2,3,4,5,6,7 \
  --start-from evaluation
```

Do not use `--start-from` to pretend a partially completed stage is complete.
Use the checkpoint/resume command printed for the affected training entry, or
start a new run/output path. Preserve abandoned outputs under `_archive/`
rather than deleting or overwriting them.

## Copy artifacts and account for cost

Copy results off the instance after every smoke and at stage boundaries, not
only immediately before shutdown. Use the direct SSH host/port shown by Vast:

```bash
mkdir -p ./vast-artifacts/logs ./vast-artifacts/artifacts ./vast-artifacts/figures
rsync -avP -e 'ssh -p <direct-ssh-port>' \
  root@<direct-ssh-host>:/workspace/consistency-training-methods/logs/ \
  ./vast-artifacts/logs/
rsync -avP -e 'ssh -p <direct-ssh-port>' \
  root@<direct-ssh-host>:/workspace/consistency-training-methods/artifacts/ \
  ./vast-artifacts/artifacts/
rsync -avP -e 'ssh -p <direct-ssh-port>' \
  root@<direct-ssh-host>:/workspace/consistency-training-methods/figures/ \
  ./vast-artifacts/figures/
```

Let `P` be the reviewed `dph_total` from the `--storage=500` search row and
let `T` be billable wall-clock hours from instance creation through final
artifact sync. The base estimate is:

```text
instance cost = P * T
total cost    = instance cost + charged network traffic + external provider calls
```

Because `P` already prices the requested 500 GB, do not add that storage a
second time. Include provisioning, downloads, idle/debug time, and rsync in
`T`. Estimate the full matrix with the measured smoke ratio and a conservative
retry margin; compare the result with the approved cap before proceeding.

## Evaluate a checkpoint directly

The generic runner can load a `LocalBackend` LoRA checkpoint through the
Inspect Hugging Face provider:

```bash
python scripts/run_evals.py \
  --task-factory mcq_bias.tasks:suite_tasks \
  --local-checkpoint file:///path/to/checkpoint \
  --task-args '{"bias_types":["wrong_few_shot"],"datasets":["truthfulqa"]}' \
  --generation-config '{"max_tokens":32768}'
```
