# EvalAwareBench Figure 6 target-generation runbook

This is a prospective runbook. It does not claim that any model generation,
paid judgment, aggregation, or plot has completed. The immutable scientific
pins live in `models.yaml` and `protocol.yaml`; do not substitute moving model
branches, prompt text, or dataset revisions.

## Fixed workload and model shape

The artifact contains 100 tasks, two valences, and nine conditions (natural
baseline plus F1--F8): exactly 1,800 conditions. Three independent generations
per condition give 5,400 generations per model and 37,800 across seven models.
Generation uses temperature 0.3, at most 4,096 new tokens, and an 8,192-token
serving ceiling.

| Model key | Display label | TP | Approximate cached weights |
|---|---|---:|---:|
| `qwen36` | Qwen3.6-27B | 1 | 55.6 GB |
| `qwen32` | Qwen3-32B | 1 | 65.5 GB |
| `qwen_mo_mid` | Qwen3-32B MO (midtrained) | 1 | 131.1 GB |
| `qwen_mo_post` | Qwen3-32B MO (posttrained) | 1 | 131.1 GB |
| `llama33` | Llama-3.3-70B-Instruct | 2 | 141.1 GB |
| `llama_mo_mid` | Llama-3.3-70B MO (midtrained) | 2 | 282.2 GB |
| `llama_mo_post` | Llama-3.3-70B MO (posttrained) | 2 | 282.2 GB |

The listed total is 1,088.8 GB. Reserve at least 1,300 GB of usable shared
project/scratch cache capacity. The MO repositories store float32 weights;
vLLM downcasts them to bfloat16 while loading. The standard Llama prefetch
excludes `original/*.pth`, leaving the safetensors snapshot represented by the
141.1 GB estimate.

Qwen models use the hash-pinned paper-natural system prompt and vLLM's `qwen3`
reasoning parser. Qwen3.6 is served with `--language-model-only` because its
Transformers class is a multimodal wrapper. Llama models use the separate
hash-pinned explicit-scratchpad system prompt; the adapter extracts the tagged
scratchpad trace before paid judging. This is a protocol adaptation, not a
change to the 1,800 benchmark conditions.

## 1. Create a clean checkout and environment

The existing checkout at
`/projects/a5v/sohaib.a5v/consistency-training-methods` contains substantial
unrelated work. Do not update it in place. After this branch is published,
clone a new checkout and detach it at the reviewed commit:

```bash
export REPO_DIR="$PROJECTDIR/$USER/consistency-training-methods-figure6"
git clone https://github.com/sohaibimran7/consistency-training-methods.git "$REPO_DIR"
git -C "$REPO_DIR" checkout --detach REVIEWED_40_HEX_COMMIT
cd "$REPO_DIR"
bash infra/isambard/setup_env.sh
```

The preferred prefetch job in step 3 installs and smoke-checks the pinned GPU
serving environment at the start of its existing one-GPU allocation, avoiding
another scheduler cycle. If prefetch will be skipped entirely, perform this
manual alternative before the pilot:

```bash
srun --account=brics.a5v --nodes=1 --gpus-per-node=1 --time=01:00:00 \
  --pty /bin/bash --login
cd "$REPO_DIR"
bash infra/isambard/setup_gpu_env.sh
```

Whether run by prefetch or by the manual alternative, `setup_gpu_env.sh`
installs vLLM 0.26.0 from standard PyPI with the automatic GPU-aware PyTorch
backend, then verifies PyTorch 2.11, CUDA, bfloat16, vLLM, the dependency set,
and the repository's training import.

Set durable data/output paths and the shared cache. On the live login used for
this plan, `PROJECTDIR=/projects/a5v` and
`SCRATCHDIR=/scratch/a5v/sohaib.a5v`.

```bash
export ISAMBARD_ACCOUNT=brics.a5v
export HF_HOME="$SCRATCHDIR/ctm/huggingface-figure6"
export FIGURE6_CACHE_CAPACITY_GB=1300
export FIGURE6_SOURCE_ROOT="$PROJECTDIR/$USER/evalawarebench-figure6-source"
export FIGURE6_ARTIFACT="$PROJECTDIR/$USER/evalawarebench-figure6-artifacts/prompts.jsonl"
export FIGURE6_OUTPUT_ROOT="$PROJECTDIR/$USER/evalawarebench-figure6-generations"
mkdir -p "$HF_HOME" "$FIGURE6_SOURCE_ROOT" "$(dirname "$FIGURE6_ARTIFACT")" "$FIGURE6_OUTPUT_ROOT"
```

On 2026-07-29, the proposed scratch cache path reported a 5.0 TB filesystem
with about 4.9 TB available, while the project path reported about 200 TB
available. `lfs quota -u` still showed zero/default rather than an explicit
personal limit. These are filesystem-wide observations, not a personal
capacity guarantee. Setting `FIGURE6_CACHE_CAPACITY_GB=1300` is an explicit
acknowledgement that at least that much usable allocation has been verified
with the project owner.

## 2. Download and materialize the pinned prompt artifact

Download only the two dataset files directly from the immutable Hugging Face
dataset revision:

```bash
source "$REPO_DIR/.venv/bin/activate"
hf download aisa-group/EvalAwareBench \
  data/prompts_safety.jsonl data/prompts_capability.jsonl \
  --repo-type dataset \
  --revision 736bda484525041972fa214eaf72f3beba492666 \
  --local-dir "$FIGURE6_SOURCE_ROOT/evalawarebench"

python -m ctm_data.adapters.eval_awareness.figure6_materialize \
  --prompts-safety "$FIGURE6_SOURCE_ROOT/evalawarebench/data/prompts_safety.jsonl" \
  --prompts-capability "$FIGURE6_SOURCE_ROOT/evalawarebench/data/prompts_capability.jsonl" \
  --output "$FIGURE6_ARTIFACT"
```

The materializer requires the exact 100-task × 2-valence × 9-condition matrix
and writes a hash/provenance manifest beside the JSONL. Preserve both files.

The three paper prompt bodies remain external. Obtain them from the pinned
upstream code revision and record their local paths:

```bash
export FIGURE6_UPSTREAM_CODE="$FIGURE6_SOURCE_ROOT/decomposing-eval-awareness"
git clone https://github.com/aisa-group/decomposing-eval-awareness.git \
  "$FIGURE6_UPSTREAM_CODE"
git -C "$FIGURE6_UPSTREAM_CODE" checkout --detach 446be5c605b56a60d4efe2526f0cbf55522c523a

find "$FIGURE6_UPSTREAM_CODE" -type f \
  \( -name chat_prompt_realistic.txt -o -name chat_prompt_realistic_scratchpad.txt \) -print
```

Export the two paths printed above, then verify their exact bytes:

```bash
export PAPER_NATURAL_PROMPT_PATH=/absolute/path/to/chat_prompt_realistic.txt
export EXPLICIT_SCRATCHPAD_PROMPT_PATH=/absolute/path/to/chat_prompt_realistic_scratchpad.txt

test "$(sha256sum "$PAPER_NATURAL_PROMPT_PATH" | awk '{print $1}')" = \
  8d46fa8eee79ba7372088d0bd138a463cc5255c8043a77f468bed937f609d735
test "$(sha256sum "$EXPLICIT_SCRATCHPAD_PROMPT_PATH" | awk '{print $1}')" = \
  c7c16141e424217e334f4166fb34a5302ea880bddb67598297e7ed38dc103b33
```

The generator independently repeats these hash checks. Do not copy the prompt
bodies into this repository.

## 3. Optionally prefetch the seven pinned snapshots

The standard Llama repository is gated. Request access beforehand, then read a
Hugging Face token without displaying it:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
printf '\n'
export HF_TOKEN
```

Submit the resumable prefetch with the live account association made explicit:

```bash
cd "$REPO_DIR"
PREFETCH_JOB_ID=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --export=ALL \
  infra/isambard/prefetch_figure6.sbatch)
printf 'prefetch job: %s\n' "$PREFETCH_JOB_ID"
```

The job requests one GPU and 72 CPU cores for 12 hours. It first installs and
smoke-checks the GPU environment, then bounds Hugging Face shard downloads to
four workers by default. A 2026-07-29 login-node probe saw
5.95 MB/s for one stream: 1.0888 TB would take about 50.8 hours at that serial
rate. Parallel shards should improve throughput, but 12 hours is not a promise
of completion. Re-submit the same command to resume cached blobs. Target jobs
may also continue cache population through vLLM; use `afterany`, rather than
requiring the optional prefetch to finish successfully.

If prefetch is skipped entirely, omit the `--dependency` option from the pilot
commands below; do not invent a placeholder job ID.

## 4. Run the deterministic 100-condition pilot

The pilot selects the first 100 conditions in the verified artifact order and
generates three samples each: 300 calls per model. It writes directly into the
same seven append-only logs later extended by the full run. Row provenance is
invariant; a separate selection ledger records the deterministic prefix and
its later extension.

```bash
export FIGURE6_LIMIT_CONDITIONS=100

PILOT_QWEN_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterany:$PREFETCH_JOB_ID" \
  --time=12:00:00 \
  --export=ALL \
  infra/isambard/figure6_qwen.sbatch)

PILOT_LLAMA_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterany:$PREFETCH_JOB_ID" \
  --time=12:00:00 \
  --export=ALL \
  infra/isambard/figure6_llama.sbatch)

printf 'pilot jobs: qwen=%s llama=%s\n' "$PILOT_QWEN_JOB" "$PILOT_LLAMA_JOB"
```

The Qwen array is `0-3%4`, one GPU/node, 72 CPU cores/task. The Llama array is
`0-2%3`, two GPUs/node, 144 CPU cores/task. Every server uses the exact model
revision, bfloat16, tensor-parallel size from `models.yaml`, and an 8,192-token
model ceiling. The launcher waits for localhost health, checks endpoint model
identity, and durably appends every terminal request attempt. A timeout leaves
the cache and JSONL histories resumable.

After all pilot tasks stop, estimate remaining time from the explicit request
timestamps/token usage:

```bash
python scripts/estimate_figure6_resources.py \
  --pilot qwen36="$FIGURE6_OUTPUT_ROOT/qwen36/generations.jsonl" \
  --pilot qwen32="$FIGURE6_OUTPUT_ROOT/qwen32/generations.jsonl" \
  --pilot qwen_mo_mid="$FIGURE6_OUTPUT_ROOT/qwen_mo_mid/generations.jsonl" \
  --pilot qwen_mo_post="$FIGURE6_OUTPUT_ROOT/qwen_mo_post/generations.jsonl" \
  --pilot llama33="$FIGURE6_OUTPUT_ROOT/llama33/generations.jsonl" \
  --pilot llama_mo_mid="$FIGURE6_OUTPUT_ROOT/llama_mo_mid/generations.jsonl" \
  --pilot llama_mo_post="$FIGURE6_OUTPUT_ROOT/llama_mo_post/generations.jsonl" \
  --available-storage-gb "$FIGURE6_CACHE_CAPACITY_GB"
```

Timestamp inference measures the span of the requests and excludes model
startup. For a conservative job-level rate, obtain each array task's
`ElapsedRaw` from `sacct` and add, for example,
`--pilot-elapsed-seconds qwen36=SECONDS`. The estimator reports remaining ETA,
GPU-hours, and NHR per model, plus observed completion-token throughput and
mean request latency. Queue wait is always external; any live scheduler start
estimate is volatile.

Inspect errors, trace presence, completion lengths, and the estimator output.
Do not continue merely because the Slurm jobs exited zero; approve the full
run only after this pilot review.

## 5. Extend the same logs to the full target

Remove the pilot limiter. Retain the same artifact, prompt paths, output root,
and model cache:

```bash
unset FIGURE6_LIMIT_CONDITIONS

FULL_QWEN_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterok:$PILOT_QWEN_JOB" \
  --export=ALL \
  infra/isambard/figure6_qwen.sbatch)

FULL_LLAMA_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterok:$PILOT_LLAMA_JOB" \
  --export=ALL \
  infra/isambard/figure6_llama.sbatch)

printf 'full jobs: qwen=%s llama=%s\n' "$FULL_QWEN_JOB" "$FULL_LLAMA_JOB"
```

Each 24-hour array pass has a 60 NHR ceiling: 96 GPU-hours for Qwen plus
144 GPU-hours for Llama. The optional 12-hour prefetch adds at most 3 NHR, for
a 63 NHR single-pass ceiling. Pilot usage is metered separately and should end
after its prefix completes.

If a target job is interrupted or has terminal request errors, submit the same
array again against the same output root. Existing successes are verified and
never called again; failed logical keys receive a new append-only resume
attempt. For example:

```bash
RESUME_QWEN_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterany:$FULL_QWEN_JOB" \
  --export=ALL \
  infra/isambard/figure6_qwen.sbatch)
```

The Llama command is identical apart from the dependency job ID and
`figure6_llama.sbatch`. Override `--array` at submission when only named array
indices need another attempt. No launcher removes a partial output.

## 6. Audit exactly 5,400 successful generations per model

The generator's dry run performs full artifact, prompt, sidecar, append-only
history, and logical-key validation without starting a server or making an API
request. Run it for every output with the full selection (no limiter):

```bash
for MODEL_KEY in qwen36 qwen32 qwen_mo_mid qwen_mo_post; do
  python -m ctm_data.adapters.eval_awareness.figure6_generate \
    --artifact "$FIGURE6_ARTIFACT" \
    --output "$FIGURE6_OUTPUT_ROOT/$MODEL_KEY/generations.jsonl" \
    --model-key "$MODEL_KEY" \
    --prompt-path "$PAPER_NATURAL_PROMPT_PATH" \
    --dry-run
done

for MODEL_KEY in llama33 llama_mo_mid llama_mo_post; do
  python -m ctm_data.adapters.eval_awareness.figure6_generate \
    --artifact "$FIGURE6_ARTIFACT" \
    --output "$FIGURE6_OUTPUT_ROOT/$MODEL_KEY/generations.jsonl" \
    --model-key "$MODEL_KEY" \
    --prompt-path "$EXPLICIT_SCRATCHPAD_PROMPT_PATH" \
    --dry-run
done
```

Every summary must report `required_keys: 5400`, `existing_successes: 5400`,
`pending_keys: 0`, and `complete: true`. Append-only files may contain earlier
error rows; those remain part of the audit history. The judge adapter selects
exactly one terminal success for each logical key and rejects any history
without one.

## 7. Create, approve, submit, and collect paid judge batches

This stage is deliberately outside Slurm. The generation jobs contain no paid
judge submission and do not need `OPENAI_API_KEY`. Obtain the paper judge
template separately from the pinned upstream revision. Locate the file by its
SHA-256 and export its absolute path:

```bash
find "$FIGURE6_UPSTREAM_CODE" -type f -exec sha256sum {} + | \
  awk '$1 == "e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870" {print}'
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/path/printed/by/the/command
test "$(sha256sum "$FIGURE6_JUDGE_TEMPLATE_PATH" | awk '{print $1}')" = \
  e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870
```

Create deterministic request shards and a manifest. This command never
uploads or submits anything:

```bash
export FIGURE6_JUDGE_ROOT="$PROJECTDIR/$USER/evalawarebench-figure6-judge"
mkdir -p "$FIGURE6_JUDGE_ROOT"

python -m ctm_data.adapters.eval_awareness.figure6_judge create \
  --generations \
    "$FIGURE6_OUTPUT_ROOT/qwen36/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen32/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen_mo_mid/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen_mo_post/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama33/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama_mo_mid/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama_mo_post/generations.jsonl" \
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH" \
  --expected-template-sha256 e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870 \
  --judge-model gpt-5 \
  --output "$FIGURE6_JUDGE_ROOT/requests.jsonl"
```

The default safety margins produce at most 5,000 requests and 150 MiB per
shard. Inspect `requests.jsonl.manifest.json`, verify 37,800 rows and every
shard hash, check the provider's current queued-token limits and Batch pricing,
and record a cost estimate. Then require an explicit human decision:

```bash
read -r -p "Type APPROVE_PAID_JUDGE only after reviewing hashes and cost: " JUDGE_APPROVAL
test "$JUDGE_APPROVAL" = APPROVE_PAID_JUDGE
```

Only after that check may an authorized operator upload every manifest-listed
request shard and create the paid external batches using the provider UI or
approved submission tooling. Record each input file ID, batch ID, price basis,
approver, and timestamp next to the manifest. The repository intentionally has
no submission command.

After all batches finish, download their raw output JSONL files, verify their
provider identities against the recorded batch IDs, and collect them:

```bash
python -m ctm_data.adapters.eval_awareness.figure6_judge collect \
  --generations \
    "$FIGURE6_OUTPUT_ROOT/qwen36/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen32/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen_mo_mid/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen_mo_post/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama33/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama_mo_mid/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama_mo_post/generations.jsonl" \
  --batch-output "$FIGURE6_JUDGE_ROOT"/batch-output-*.jsonl \
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH" \
  --expected-template-sha256 e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870 \
  --judge-model gpt-5 \
  --output "$FIGURE6_JUDGE_ROOT/judgments.jsonl"
```

The collector requires exactly one matching output for every generation and
retains raw judge bodies plus model, prompt, generation, and judge provenance.
It refuses to overwrite an existing output.

## 8. Strict aggregation and plotting

Run strict aggregation first. It rejects incomplete/duplicate matrices,
unsuccessful generation or judge records, missing reasoning traces, mixed
provenance, and any denominator other than 300 per model/valence/condition
cell:

```bash
python -m ctm_data.adapters.eval_awareness.figure6_analysis \
  --judgments "$FIGURE6_JUDGE_ROOT/judgments.jsonl" \
  --output-csv "$FIGURE6_JUDGE_ROOT/figure6-aggregation.csv" \
  --summary-json "$FIGURE6_JUDGE_ROOT/figure6-summary.json"

python -m ctm_data.adapters.eval_awareness.figure6_plot \
  --input-csv "$FIGURE6_JUDGE_ROOT/figure6-aggregation.csv" \
  --summary-json "$FIGURE6_JUDGE_ROOT/figure6-summary.json" \
  --output-prefix "$FIGURE6_JUDGE_ROOT/figure6-reproduction"
```

Publication-mode plotting accepts only a complete 37,800-record strict
summary. The output is a computed reproduction from the supplied artifacts,
not paper result data. Preserve the artifact, seven generation logs and their
provenance/selection sidecars, batch manifest and approval record, raw batch
outputs, normalized judgments, summary, CSV, PNG, and PDF together.

## Isambard capacity caveat

Phase 2 has 1,320 nodes, four 96 GB HBM GH200 GPUs per node, 72 CPU cores and
115 GB usable CPU memory per Superchip, a documented 24-hour `workq` maximum,
and a 32-GPU project maximum. One NHR is four GPU-hours.

The recorded 2026-07-29 snapshot had no fully idle nodes and mixed partial
capacity. At 2026-07-29T13:03:58Z, Slurm `--test-only` estimated
2026-08-01T05:32:59Z (about 64 hours 29 minutes later) for the exact one-GPU
12-hour prefetch, one-GPU 24-hour Qwen task, and two-GPU 24-hour Llama task.
Two-hour pilot checks at 13:04:29Z estimated 05:33:29Z, so shorter requests did
not improve that snapshot. These values are volatile observations, not a
reservation or guarantee. Repeat the read-only scheduler checks immediately
before each submission.
