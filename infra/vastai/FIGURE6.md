# EvalAwareBench Figure 6 on two Vast.ai hosts

This is the generation-only Vast.ai path for the pinned seven-model Figure 6
reproduction. It has no SLURM, `PROJECTDIR`, `SCRATCHDIR`, shared-filesystem, or
external grading assumptions. The two hosts may have independent local disks.

## 1. Create the hosts

Use the exact image `vllm/vllm-openai:v0.26.0` on both x86 CUDA hosts. Create
each instance with Vast's SSH runtype and direct SSH connectivity:

```bash
vastai create instance <host-a-offer-id> \
    --image vllm/vllm-openai:v0.26.0 \
    --disk 1000 --ssh --direct

vastai create instance <host-b-offer-id> \
    --image vllm/vllm-openai:v0.26.0 \
    --disk 400 --ssh --direct
```

The disk values are starting points for the checkpoint sizes below; increase
them when the selected offer or cache policy needs more headroom. Do not also
set Docker entrypoint or image-argument overrides: SSH runtype installs and
supervises the SSH entrypoint. If bootstrap commands are needed, supply them
through Vast's `--onstart-cmd` option. No public API port is needed because every
model server binds only to loopback.

The fixed allocation is:

| Host | Required visible GPUs | Models and physical GPU indices |
|---|---:|---|
| A | 8 | `qwen36=0`, `qwen32=1`, `qwen_mo_mid=2`, `qwen_mo_post=3`, `llama33=4,5`, `llama_mo_mid=6,7` |
| B | 2 | `llama_mo_post=0,1` |

Host A needs room for approximately 807 GB of checkpoint snapshots; host B
needs approximately 283 GB, before cache and filesystem headroom. Select GPUs
with bfloat16 support and enough aggregate memory for the assigned models. The
launcher deliberately rejects a host exposing any GPU count other than 8 or 2.

Clone the same repository commit on both hosts. Do not run the generic
`infra/vastai/provision.sh`; it installs the complete training environment.
From the repository root, run:

```bash
bash infra/vastai/provision_figure6.sh
```

That creates `.venv-figure6` with `--system-site-packages`, installs only the
OpenAI-compatible client, PyYAML, and this repository without dependencies, and
asserts that Torch and vLLM still come from the image. Provisioning stops unless
vLLM is exactly 0.26.0, CUDA is available, and the GPU supports bfloat16.

## 2. Put immutable inputs on each host

Each host needs the verified 1,800-row artifact and its adjacent manifest:

```text
/workspace/figure6-inputs/prompts.jsonl
/workspace/figure6-inputs/prompts.jsonl.manifest.json
```

Host A needs both pinned upstream prompt files. Host B needs the explicit
scratchpad prompt. The generator verifies their SHA-256 identities before any
request, so copying a wrong prompt fails closed.

Prefer a token file for Hugging Face access. Create it without putting the token
on a command line, then make its permissions exactly 600:

```bash
mkdir -p /workspace/private
chmod 700 /workspace/private
# Write the single token line using an interactive editor or a secure file copy.
chmod 600 /workspace/private/hf_token
```

The file must be regular (not a symlink), owned by the current user, and contain
one non-empty token line. An inherited `HF_TOKEN` environment variable is also
supported, but never set it at the same time as `HF_TOKEN_FILE`. The scripts do
not print or pass this token as a command-line argument. Do not configure
`OPENAI_API_KEY` on these hosts; the local endpoint client is forced to use a
non-secret dummy token.

## 3. Export the run environment

On host A:

```bash
export HF_HOME=/workspace/huggingface
export FIGURE6_ARTIFACT=/workspace/figure6-inputs/prompts.jsonl
export FIGURE6_OUTPUT_ROOT=/workspace/figure6-output
export PAPER_NATURAL_PROMPT_PATH=/workspace/figure6-inputs/chat_prompt_realistic.txt
export EXPLICIT_SCRATCHPAD_PROMPT_PATH=/workspace/figure6-inputs/chat_prompt_realistic_scratchpad.txt
export HF_TOKEN_FILE=/workspace/private/hf_token
```

On host B, use the same first three variables and token setup, but only the
scratchpad prompt is required:

```bash
export HF_HOME=/workspace/huggingface
export FIGURE6_ARTIFACT=/workspace/figure6-inputs/prompts.jsonl
export FIGURE6_OUTPUT_ROOT=/workspace/figure6-output
export EXPLICIT_SCRATCHPAD_PROMPT_PATH=/workspace/figure6-inputs/chat_prompt_realistic_scratchpad.txt
export HF_TOKEN_FILE=/workspace/private/hf_token
```

Optional operational variables are `FIGURE6_VENV` (absolute venv path),
`FIGURE6_MAX_CONCURRENCY` (default 10), `VLLM_STARTUP_TIMEOUT_SECONDS` (default
1800), and `VLLM_GPU_MEMORY_UTILIZATION` (default 0.92). Model identity,
revision, bfloat16 dtype, tensor parallelism, prompt selection, 8,192-token
model context, 3 replicates, temperature 0.3, and 4,096 output tokens are fixed.

## 4. Canary, then resume the full run

Use a persistent terminal such as `tmux`. Run the canary on each host:

```bash
FIGURE6_LIMIT_CONDITIONS=100 bash infra/vastai/launch_figure6_host.sh A canary
FIGURE6_LIMIT_CONDITIONS=100 bash infra/vastai/launch_figure6_host.sh B canary
```

Inspect every model's generation records and runner/server logs. Once approved,
run full mode on the same host with the same `FIGURE6_OUTPUT_ROOT`:

```bash
bash infra/vastai/launch_figure6_host.sh A full
bash infra/vastai/launch_figure6_host.sh B full
```

Full mode explicitly removes `FIGURE6_LIMIT_CONDITIONS`. The generator resumes
the same per-model `generations.jsonl`, skips successful canary keys, and appends
the remaining records. It never truncates a generation log. Runner logs, vLLM
logs, and host status JSONL are also append-only under `_logs/` and `_status/`.
The launchers hold persistent per-host lock files to prevent accidental duplicate
launches, validate the model registry against the fixed GPU plan and tensor
parallel sizes, reject GPU overlap or gaps, and use unique loopback ports
8100–8106.

The scripts only serve models and call `figure6_generate`; they never start the
paid external judge. After both hosts finish, copy the complete output roots off
the instances before destroying them. Hosts own disjoint model directories, so
combine those directories without rewriting their JSONL or provenance sidecars,
and retain both hosts' `_logs` and `_status` records.
