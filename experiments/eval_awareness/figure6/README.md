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

## 3. Validate the GPU environment and prefetch the three selected snapshots

The selected Qwen repositories are not the gated Llama checkpoint, so the
current prefetch does not require an `HF_TOKEN`. Do not broaden the prefetch
arguments to `qwen36` or any Llama key.

First submit the one-GPU, 30-minute environment smoke gate. It installs the
pinned AArch64 serving stack, exposes the CUDA 13 runtime bundled with vLLM,
and imports vLLM on the allocated GH200. It downloads no model weights and
runs no evaluation:

```bash
cd "$REPO_DIR"
GPU_SMOKE_JOB_ID=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --export=ALL \
  infra/isambard/smoke_figure6_gpu.sbatch)
printf 'GPU smoke job: %s\n' "$GPU_SMOKE_JOB_ID"
```

Submit the resumable prefetch only after the smoke gate succeeds:

```bash
cd "$REPO_DIR"
PREFETCH_JOB_ID=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterok:$GPU_SMOKE_JOB_ID" \
  --export=ALL \
  infra/isambard/prefetch_figure6.sbatch \
  qwen32 qwen_mo_mid qwen_mo_post)
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
same three append-only logs later extended by the full run. Row provenance is
invariant; a separate selection ledger records the deterministic prefix and
its later extension.

```bash
export FIGURE6_LIMIT_CONDITIONS=100

PILOT_QWEN_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterok:$GPU_SMOKE_JOB_ID,afterany:$PREFETCH_JOB_ID" \
  --time=12:00:00 \
  --array=1-3%3 \
  --export=ALL \
  infra/isambard/figure6_qwen.sbatch)

printf 'selected-Qwen pilot job: %s\n' "$PILOT_QWEN_JOB"
```

The selected Qwen override is `1-3%3`: `qwen32`, `qwen_mo_mid`, and
`qwen_mo_post`, one GPU/node and 72 CPU cores/task. Array index 0 (`qwen36`)
and all Llama arrays are intentionally skipped. Every server uses the exact model
revision, bfloat16, tensor-parallel size from `models.yaml`, and an 8,192-token
model ceiling. The launcher waits for localhost health, checks endpoint model
identity, and durably appends every terminal request attempt. A timeout leaves
the cache and JSONL histories resumable.

After all pilot tasks stop, estimate remaining time from the explicit request
timestamps/token usage:

```bash
python scripts/estimate_figure6_resources.py \
  --pilot qwen32="$FIGURE6_OUTPUT_ROOT/qwen32/generations.jsonl" \
  --pilot qwen_mo_mid="$FIGURE6_OUTPUT_ROOT/qwen_mo_mid/generations.jsonl" \
  --pilot qwen_mo_post="$FIGURE6_OUTPUT_ROOT/qwen_mo_post/generations.jsonl" \
  --available-storage-gb "$FIGURE6_CACHE_CAPACITY_GB"
```

Timestamp inference measures the span of the requests and excludes model
startup. For a conservative job-level rate, obtain each array task's
`ElapsedRaw` from `sacct` and add, for example,
`--pilot-elapsed-seconds qwen32=SECONDS`. The estimator reports remaining ETA,
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
  --array=1-3%3 \
  --export=ALL \
  infra/isambard/figure6_qwen.sbatch)

printf 'selected-Qwen full job: %s\n' "$FULL_QWEN_JOB"
```

Each selected 24-hour array pass has a 72 GPU-hour ceiling (18 NHR). The
optional 12-hour prefetch adds at most 3 NHR, for a 21 NHR single-pass ceiling.
Pilot usage is metered separately and should end after its prefix completes.

If a target job is interrupted or has terminal request errors, submit the same
array again against the same output root. Existing successes are verified and
never called again; failed logical keys receive a new append-only resume
attempt. For example:

```bash
RESUME_QWEN_JOB=$(sbatch --parsable \
  --account="$ISAMBARD_ACCOUNT" \
  --dependency="afterany:$FULL_QWEN_JOB" \
  --array=1-3%3 \
  --export=ALL \
  infra/isambard/figure6_qwen.sbatch)
```

Keep the `--array=1-3%3` override on every resume so neither `qwen36` nor a
Llama job enters the current scope. No launcher removes a partial output.

## 6. Audit exactly 5,400 successful generations per model

The generator's dry run performs full artifact, prompt, sidecar, append-only
history, and logical-key validation without starting a server or making an API
request. Run it for every output with the full selection (no limiter):

```bash
for MODEL_KEY in qwen32 qwen_mo_mid qwen_mo_post; do
  python -m ctm_data.adapters.eval_awareness.figure6_generate \
    --artifact "$FIGURE6_ARTIFACT" \
    --output "$FIGURE6_OUTPUT_ROOT/$MODEL_KEY/generations.jsonl" \
    --model-key "$MODEL_KEY" \
    --prompt-path "$PAPER_NATURAL_PROMPT_PATH" \
    --dry-run
done
```

Every summary must report `required_keys: 5400`, `existing_successes: 5400`,
`pending_keys: 0`, and `complete: true`. Append-only files may contain earlier
error rows; those remain part of the audit history. The judge adapter selects
exactly one terminal success for each logical key and rejects any history
without one.

## 7. Run the approved GPT-OSS 120B Nitro alternative judge through OpenRouter

The current reproduction uses `openai/gpt-oss-120b:nitro` directly from the
operator laptop through OpenRouter with
the exact hash-pinned paper judge prompt and a 32,768-token output ceiling. This
is a user-pinned alternative judge, not the paper's GPT-5 judge. The registered
`gpt-oss-120b-nitro-direct` profile prohibits proxies and route attestations, uses
the model's maximum supported mandatory reasoning effort (`high`) without returning
the reasoning trace, requests a JSON object, and hash-binds temperature 0, concurrency 4,
the 300-second retry cap, and exact provider routing. The Nitro suffix and the
explicit `provider.sort=throughput` pin throughput-prioritized routing. The profile
accepts the requested Nitro model ID or OpenRouter's base response identity
`openai/gpt-oss-120b`, while rejecting every other response model.
It excludes the Cerebras provider because live paid responses stopped at 8,192
tokens despite the larger advertised endpoint ceiling, and permits OpenRouter to
fall through to the next fastest provider that supports every pinned parameter.
The approved scope is exactly
`qwen32`, `qwen_mo_mid`, and `qwen_mo_post`; do not add `qwen36` or a Llama key. Each model has its own complete
5,400-request paid lifecycle, so it can be judged as soon as that model's
generation log finishes. The three verified outputs are combined only during
strict analysis.

Use clean, detached local checkouts for both this repository and the upstream
prompt source. Record the reviewed 40-hex commit supplied in the handoff; do
not run the paid command from an uncommitted or moving checkout:

```bash
export FIGURE6_CODE_LOCAL=/absolute/local/path/consistency-training-methods-figure6
export FIGURE6_REVIEWED_COMMIT=REVIEWED_40_HEX_COMMIT
git clone https://github.com/sohaibimran7/consistency-training-methods.git \
  "$FIGURE6_CODE_LOCAL"
git -C "$FIGURE6_CODE_LOCAL" checkout --detach "$FIGURE6_REVIEWED_COMMIT"
test "$(git -C "$FIGURE6_CODE_LOCAL" rev-parse HEAD)" = "$FIGURE6_REVIEWED_COMMIT"
test -z "$(git -C "$FIGURE6_CODE_LOCAL" status --porcelain)"

export FIGURE6_UPSTREAM_CODE_LOCAL=/absolute/local/path/decomposing-eval-awareness
git clone https://github.com/aisa-group/decomposing-eval-awareness.git \
  "$FIGURE6_UPSTREAM_CODE_LOCAL"
git -C "$FIGURE6_UPSTREAM_CODE_LOCAL" checkout --detach \
  446be5c605b56a60d4efe2526f0cbf55522c523a
test "$(git -C "$FIGURE6_UPSTREAM_CODE_LOCAL" rev-parse HEAD)" = \
  446be5c605b56a60d4efe2526f0cbf55522c523a

cd "$FIGURE6_CODE_LOCAL"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Retrieve each completed append-only generation log and both provenance
sidecars from Isambard. All variables below are local shell variables; use
absolute roots:

```bash
export ISAMBARD_LOGIN=user@login.isambard.example
export ISAMBARD_OUTPUT_ROOT=/absolute/remote/path/evalawarebench-figure6
export FIGURE6_LOCAL_GENERATION_ROOT=/absolute/local/path/figure6-generations
mkdir -p "$FIGURE6_LOCAL_GENERATION_ROOT"

for MODEL_KEY in qwen32 qwen_mo_mid qwen_mo_post; do
  mkdir -p "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY"
  for FILE_NAME in generations.jsonl \
                   generations.jsonl.provenance.json \
                   generations.jsonl.selections.jsonl; do
    rsync -av --protect-args \
      "$ISAMBARD_LOGIN:$ISAMBARD_OUTPUT_ROOT/$MODEL_KEY/$FILE_NAME" \
      "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/$FILE_NAME"
  done
done

find "$FIGURE6_UPSTREAM_CODE_LOCAL" -type f -exec sha256sum {} + | \
  awk '$1 == "e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870" {print}'
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/path/printed/by/the/command
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
test "$(sha256sum "$FIGURE6_JUDGE_TEMPLATE_PATH" | awk '{print $1}')" = \
  e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870
mkdir -p "$FIGURE6_JUDGE_ROOT"
```

Keep `OPENROUTER_API_KEY` only on that operator machine. Do not configure a
proxy or U.S. route evidence for the current direct GPT-OSS Nitro profile.

### Optional Muse US-proxy profile

The registered `muse-spark-1.1-us-proxy` alternate is not the current judge.
If it is deliberately selected in a separately reviewed plan, its route is
Vast instance `46276309`, `ssh3.vast.ai:36308`, country `US`, with SOCKS on
`127.0.0.1:1080`. Use the dedicated control socket shown here. If a stale
socket exists, move it into the judge archive; never delete it in place.
`ExitOnForwardFailure` makes bind/forward setup fail closed, and `-O check`
proves the exact master is live before any plan or paid request:

```bash
export VAST_INSTANCE_ID=46276309
export VAST_SSH_HOST=ssh3.vast.ai
export VAST_SSH_PORT=36308
export VAST_COUNTRY_CODE=US
export FIGURE6_SOCKS_PORT=1080
export FIGURE6_CONTROL_SOCKET=/private/tmp/figure6-vast-a-socks-control
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
mkdir -p "$FIGURE6_JUDGE_ROOT/_archive"

if test -e "$FIGURE6_CONTROL_SOCKET"; then
  mv "$FIGURE6_CONTROL_SOCKET" \
    "$FIGURE6_JUDGE_ROOT/_archive/figure6-vast-a-socks-control.$(date -u +%Y%m%dT%H%M%SZ)"
fi

ssh -M -S "$FIGURE6_CONTROL_SOCKET" -fN \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -D "127.0.0.1:$FIGURE6_SOCKS_PORT" \
  -p "$VAST_SSH_PORT" "root@$VAST_SSH_HOST"

ssh -S "$FIGURE6_CONTROL_SOCKET" -O check \
  -p "$VAST_SSH_PORT" "root@$VAST_SSH_HOST"
```

Create and preserve a sanitized evidence JSON. Before doing so, verify the
instance ID, SSH endpoint, and U.S. location in the Vast console, save a
credential-free console export/screenshot, then verify the tunneled egress
country is `US`. Set `ROUTE_EXIT_IP`, `ROUTE_EXIT_COUNTRY`, the
`VAST_CONSOLE_EVIDENCE` path, and a real non-secret `ROUTE_ATTESTED_BY` from
those checks. Never put an SSH key, API key, token, cookie, or authorization
header in either evidence file:

```bash
export FIGURE6_PROXY_URL="socks5h://127.0.0.1:$FIGURE6_SOCKS_PORT"
export ROUTE_ATTESTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
: "${ROUTE_ATTESTED_BY:?export a real non-secret reviewer identifier}"
: "${ROUTE_EXIT_IP:?record the tunneled public egress IP}"
: "${ROUTE_EXIT_COUNTRY:?record the independently checked egress country}"
: "${VAST_CONSOLE_EVIDENCE:?export the sanitized Vast console evidence path}"
test "$ROUTE_EXIT_COUNTRY" = US
export VAST_CONSOLE_EVIDENCE_SHA256="$(shasum -a 256 "$VAST_CONSOLE_EVIDENCE" | awk '{print $1}')"

export ROUTE_EVIDENCE="$FIGURE6_JUDGE_ROOT/vast-route-evidence.json"
jq -n \
  --arg instance "$VAST_INSTANCE_ID" \
  --arg host "$VAST_SSH_HOST" \
  --argjson port "$VAST_SSH_PORT" \
  --arg country "$VAST_COUNTRY_CODE" \
  --arg attested_at "$ROUTE_ATTESTED_AT" \
  --arg attested_by "$ROUTE_ATTESTED_BY" \
  --arg socks "$FIGURE6_PROXY_URL" \
  --arg control "$FIGURE6_CONTROL_SOCKET" \
  --arg exit_ip "$ROUTE_EXIT_IP" \
  --arg exit_country "$ROUTE_EXIT_COUNTRY" \
  --arg console_sha "$VAST_CONSOLE_EVIDENCE_SHA256" \
  '{schema:"ctm.eval_awareness.figure6.vast_route_evidence.v1",
    vast_instance_id:$instance, ssh_host:$host, ssh_port:$port,
    country_code:$country, attested_at:$attested_at, attested_by:$attested_by,
    socks_proxy:$socks, ssh_control_socket:$control, exit_ip:$exit_ip,
    exit_country_code:$exit_country, vast_console_evidence_sha256:$console_sha}' \
  > "$ROUTE_EVIDENCE"
export ROUTE_EVIDENCE_SHA256="$(shasum -a 256 "$ROUTE_EVIDENCE" | awk '{print $1}')"
test "${#ROUTE_EVIDENCE_SHA256}" -eq 64
```

For the optional Muse profile, the evidence bytes, concrete route fields, and
exact `socks5h` proxy enter its deterministic plan. The current direct workflow
below supplies none of them. Judge one complete model at a time. The CLI will
reject a partial, extra, mixed, or unregistered matrix before opening the HTTP
client:

```bash
export FIGURE6_LOCAL_GENERATION_ROOT=/absolute/local/path/figure6-generations
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/local/path/to/the/verified/template
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
export FIGURE6_MAX_ATTEMPTS=5
mkdir -p "$FIGURE6_JUDGE_ROOT"

build_judge_args() {
  MODEL_KEY="$1"
  MAX_ATTEMPTS="$2"
  MODEL_JUDGE_ROOT="$FIGURE6_JUDGE_ROOT/$MODEL_KEY"
  mkdir -p "$MODEL_JUDGE_ROOT"

  JUDGE_ARGS=(
    --generations "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl"
    --expected-model-key "$MODEL_KEY"
    --judge-profile gpt-oss-120b-nitro-direct
    --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH"
    --attempt-log "$MODEL_JUDGE_ROOT/openrouter-attempts.jsonl"
    --manifest "$MODEL_JUDGE_ROOT/openrouter-manifest.json"
    --output "$MODEL_JUDGE_ROOT/judgments.jsonl"
    --max-attempts "$MAX_ATTEMPTS"
    --max-retry-after 300
  )
}

for MODEL_KEY in qwen32 qwen_mo_mid qwen_mo_post; do
  build_judge_args "$MODEL_KEY" "$FIGURE6_MAX_ATTEMPTS"
  python scripts/judge_figure6_openrouter.py "${JUDGE_ARGS[@]}" --dry-run \
    | tee "$MODEL_JUDGE_ROOT/plan.dry-run.json"
done
```

Stop here. An independent reviewer must inspect each exact 5,400-request plan,
cost, profile settings, retry policy, identities, and hash. Only in a later
operator session, paste one reviewed hash explicitly; do not derive it in the
same shell block that performs payment:

```bash
export FIGURE6_LOCAL_GENERATION_ROOT=/absolute/local/path/figure6-generations
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/local/path/to/the/verified/template
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
export FIGURE6_MAX_ATTEMPTS=5
export MODEL_KEY=qwen32
export REVIEWED_PLAN_SHA256=PASTE_INDEPENDENTLY_REVIEWED_64_HEX_HASH
export MODEL_JUDGE_ROOT="$FIGURE6_JUDGE_ROOT/$MODEL_KEY"
mkdir -p "$MODEL_JUDGE_ROOT"
JUDGE_ARGS=(
  --generations "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl"
  --expected-model-key "$MODEL_KEY"
  --judge-profile gpt-oss-120b-nitro-direct
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH"
  --attempt-log "$MODEL_JUDGE_ROOT/openrouter-attempts.jsonl"
  --manifest "$MODEL_JUDGE_ROOT/openrouter-manifest.json"
  --output "$MODEL_JUDGE_ROOT/judgments.jsonl"
  --max-attempts "$FIGURE6_MAX_ATTEMPTS"
  --max-retry-after 300
)
test "${#REVIEWED_PLAN_SHA256}" -eq 64
test -n "${OPENROUTER_API_KEY:-}"
python scripts/judge_figure6_openrouter.py "${JUDGE_ARGS[@]}" \
  --expected-plan-sha256 "$REVIEWED_PLAN_SHA256" \
  --yes
test "$(wc -l < "$MODEL_JUDGE_ROOT/judgments.jsonl" | tr -d ' ')" = 5400
```

The paid entry point always requires both `--yes` and the separately reviewed
`--expected-plan-sha256`; there is no one-record/debug paid mode. It holds an
OS lock across history validation, all requests, durable attempt appends, and
the final normalized output. Ambient proxy variables are ignored. Every HTTP
200 raw body, provider response ID/model/usage/content, request identity,
attempt plan, and approval index remains in the audit trail without recording
credentials.

Malformed paid HTTP-200 responses require `--rescore-paid-errors`. Exhausted
transport/408/429/5xx retries—including an exhausted malformed-response
rescore—cannot silently gain another attempt. Raise only `--max-attempts`, run
a new dry plan, review its new hash, and explicitly record the ceiling-only
amendment while preserving all old successes and attempts:

```bash
export FIGURE6_LOCAL_GENERATION_ROOT=/absolute/local/path/figure6-generations
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/local/path/to/the/verified/template
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
export MODEL_KEY=qwen32
export NEW_MAX_ATTEMPTS=7
export MODEL_JUDGE_ROOT="$FIGURE6_JUDGE_ROOT/$MODEL_KEY"
JUDGE_ARGS=(
  --generations "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl"
  --expected-model-key "$MODEL_KEY"
  --judge-profile gpt-oss-120b-nitro-direct
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH"
  --attempt-log "$MODEL_JUDGE_ROOT/openrouter-attempts.jsonl"
  --manifest "$MODEL_JUDGE_ROOT/openrouter-manifest.json"
  --output "$MODEL_JUDGE_ROOT/judgments.jsonl"
  --max-attempts "$NEW_MAX_ATTEMPTS"
  --max-retry-after 300
)
python scripts/judge_figure6_openrouter.py "${JUDGE_ARGS[@]}" --dry-run \
  | tee "$MODEL_JUDGE_ROOT/plan.amendment.dry-run.json"
```

Stop again for independent review. In a later operator session, paste the
reviewed amended hash and authorize only the ceiling increase:

```bash
export FIGURE6_LOCAL_GENERATION_ROOT=/absolute/local/path/figure6-generations
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/local/path/to/the/verified/template
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
export MODEL_KEY=qwen32
export NEW_MAX_ATTEMPTS=7
export AMENDED_PLAN_SHA256=PASTE_INDEPENDENTLY_REVIEWED_AMENDMENT_HASH
export MODEL_JUDGE_ROOT="$FIGURE6_JUDGE_ROOT/$MODEL_KEY"
JUDGE_ARGS=(
  --generations "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl"
  --expected-model-key "$MODEL_KEY"
  --judge-profile gpt-oss-120b-nitro-direct
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH"
  --attempt-log "$MODEL_JUDGE_ROOT/openrouter-attempts.jsonl"
  --manifest "$MODEL_JUDGE_ROOT/openrouter-manifest.json"
  --output "$MODEL_JUDGE_ROOT/judgments.jsonl"
  --max-attempts "$NEW_MAX_ATTEMPTS"
  --max-retry-after 300
)
test "${#AMENDED_PLAN_SHA256}" -eq 64
test -n "${OPENROUTER_API_KEY:-}"
python scripts/judge_figure6_openrouter.py "${JUDGE_ARGS[@]}" \
  --expected-plan-sha256 "$AMENDED_PLAN_SHA256" \
  --amend-attempt-ceiling \
  --yes
```

If the exhausted attempt is a malformed paid response, include both explicit
authorizations on the amended paid command:

```bash
export FIGURE6_LOCAL_GENERATION_ROOT=/absolute/local/path/figure6-generations
export FIGURE6_JUDGE_TEMPLATE_PATH=/absolute/local/path/to/the/verified/template
export FIGURE6_JUDGE_ROOT=/absolute/local/path/evalawarebench-figure6-judge
export MODEL_KEY=qwen32
export NEW_MAX_ATTEMPTS=7
export AMENDED_PLAN_SHA256=PASTE_INDEPENDENTLY_REVIEWED_AMENDMENT_HASH
export MODEL_JUDGE_ROOT="$FIGURE6_JUDGE_ROOT/$MODEL_KEY"
JUDGE_ARGS=(
  --generations "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl"
  --expected-model-key "$MODEL_KEY"
  --judge-profile gpt-oss-120b-nitro-direct
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH"
  --attempt-log "$MODEL_JUDGE_ROOT/openrouter-attempts.jsonl"
  --manifest "$MODEL_JUDGE_ROOT/openrouter-manifest.json"
  --output "$MODEL_JUDGE_ROOT/judgments.jsonl"
  --max-attempts "$NEW_MAX_ATTEMPTS"
  --max-retry-after 300
)
test "${#AMENDED_PLAN_SHA256}" -eq 64
test -n "${OPENROUTER_API_KEY:-}"
python scripts/judge_figure6_openrouter.py "${JUDGE_ARGS[@]}" \
  --expected-plan-sha256 "$AMENDED_PLAN_SHA256" \
  --amend-attempt-ceiling \
  --rescore-paid-errors \
  --yes
```

`Retry-After` is honored exactly; the hashed 300-second cap fails instead of
truncating a longer provider delay. Changing concurrency, route, proxy,
endpoint, response identities, retry cap, template, or any request is a new
core lifecycle, not a ceiling amendment. A transport timeout after provider
acceptance remains a small unavoidable duplicate-charge risk.

### Legacy paper-judge OpenAI Batch path

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

Only after that check may an authorized operator cross the paid API boundary.
Make the API key available through the operator's secure environment setup
(never in an argument or file), then run the separate lifecycle tool with
`--yes`:

```bash
test -n "${OPENAI_API_KEY:-}" # must already be securely exported
export FIGURE6_BATCH_MANIFEST="$FIGURE6_JUDGE_ROOT/openai-batch-lifecycle.json"

python -m ctm_data.adapters.eval_awareness.figure6_batch submit \
  --request-manifest "$FIGURE6_JUDGE_ROOT/requests.jsonl.manifest.json" \
  --manifest "$FIGURE6_BATCH_MANIFEST" \
  --yes
```

`submit` always prints the exact resolved shard paths, hashes and request
counts plus `gpt-5`, `/v1/chat/completions`, 4,096 completion tokens, the pinned
judge-template hash, and the 24-hour completion window before its first paid
call. `--yes` is the explicit confirmation that this reviewed plan may be
submitted. Without it, no lifecycle manifest, upload, or Batch is created.
The atomic lifecycle manifest records the confirmation time, exact request
manifest and shard identities, upload file IDs, Batch IDs, provider statuses,
timestamps, counts, and an append-only event trail. Re-running the identical
command resumes it and does not upload or create Batches for recorded shards.

Check once, or poll read-only until every shard is terminal:

```bash
python -m ctm_data.adapters.eval_awareness.figure6_batch status \
  --manifest "$FIGURE6_BATCH_MANIFEST"

python -m ctm_data.adapters.eval_awareness.figure6_batch status \
  --manifest "$FIGURE6_BATCH_MANIFEST" \
  --wait \
  --poll-interval-seconds 60
```

Download all available output and error files for terminal Batches. This
refreshes status first, verifies downloaded byte counts and SHA-256 hashes,
and refuses to overwrite mismatched local content. It exits nonzero with a
clear error for failed, expired, cancelled, or partially submitted lifecycles:

```bash
python -m ctm_data.adapters.eval_awareness.figure6_batch download \
  --manifest "$FIGURE6_BATCH_MANIFEST" \
  --output-dir "$FIGURE6_JUDGE_ROOT/raw-batch"
```

After every Batch is completed, collect the verified raw output JSONL files:

```bash
python -m ctm_data.adapters.eval_awareness.figure6_judge collect \
  --generations \
    "$FIGURE6_OUTPUT_ROOT/qwen32/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen_mo_mid/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/qwen_mo_post/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama33/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama_mo_mid/generations.jsonl" \
    "$FIGURE6_OUTPUT_ROOT/llama_mo_post/generations.jsonl" \
  --batch-output "$FIGURE6_JUDGE_ROOT"/raw-batch/batch-output.part-*.jsonl \
  --judge-template "$FIGURE6_JUDGE_TEMPLATE_PATH" \
  --expected-template-sha256 e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870 \
  --judge-model gpt-5 \
  --output "$FIGURE6_JUDGE_ROOT/judgments.jsonl"
```

The collector requires exactly one matching output for every generation and
retains raw judge bodies plus model, prompt, generation, and judge provenance.
It refuses to overwrite an existing output.

## 8. Strict aggregation and plotting

The approved current scope is exactly `qwen32`, `qwen_mo_mid`, and
`qwen_mo_post`; `qwen36` and all Llama checkpoints are intentionally deferred.
This is a strict, complete 16,200-row subset (5,400 judgments per selected
model), not the full seven-model Figure 6 reproduction. Run strict aggregation
first. It rejects incomplete/duplicate matrices,
unsuccessful generation or judge records, missing reasoning traces, mixed
provenance, and any denominator other than 300 per model/valence/condition
cell:

```bash
python -m ctm_data.adapters.eval_awareness.figure6_analysis \
  --judgments \
    "$FIGURE6_JUDGE_ROOT/qwen32/judgments.jsonl" \
    "$FIGURE6_JUDGE_ROOT/qwen_mo_mid/judgments.jsonl" \
    "$FIGURE6_JUDGE_ROOT/qwen_mo_post/judgments.jsonl" \
  --judge-manifest "$FIGURE6_JUDGE_ROOT/qwen32/openrouter-manifest.json" \
  --judge-manifest "$FIGURE6_JUDGE_ROOT/qwen_mo_mid/openrouter-manifest.json" \
  --judge-manifest "$FIGURE6_JUDGE_ROOT/qwen_mo_post/openrouter-manifest.json" \
  --expected-model-key qwen32 \
  --expected-model-key qwen_mo_mid \
  --expected-model-key qwen_mo_post \
  --expected-judge-profile gpt-oss-120b-nitro-direct \
  --expected-judge-max-completion-tokens 32768 \
  --output-csv "$FIGURE6_JUDGE_ROOT/figure6-aggregation.csv" \
  --summary-json "$FIGURE6_JUDGE_ROOT/figure6-summary.json"

python -m ctm_data.adapters.eval_awareness.figure6_plot \
  --input-csv "$FIGURE6_JUDGE_ROOT/figure6-aggregation.csv" \
  --summary-json "$FIGURE6_JUDGE_ROOT/figure6-summary.json" \
  --output-prefix "$FIGURE6_JUDGE_ROOT/figure6-qwen-only-reproduction"
```

Strict plotting derives the selected model order from the analysis summary and
accepts this scope only when all three selected registered Qwen models have
exactly 5,400 valid judgments (16,200 total). Strict GPT-OSS Nitro-profile analysis recomputes every judgment
file digest, validates its unique `run_completed` event and approval, walks the
reviewed ceiling-amendment chain, binds every row to its planned custom ID and
generation digest, and checks the exact direct route, provider routing, reasoning/JSON settings, requested
and response model identities, provider request/response IDs, and plan hashes.
It fails closed if any judgment file, manifest, or evidence file is absent,
extra, mismatched, or tampered. Plotting automatically labels the title and source
note with both `Qwen-only subset (3 of 7 models; 16,200 strict judgments)` and
`OpenRouter GPT-OSS 120B Nitro alternative judge`; custom title/source text is rejected if it
omits either exact label. The output is complete only for the requested subset,
not the full seven-model paper reproduction or paper result data. Preserve the
artifact, three selected Qwen generation logs and their provenance/selection sidecars,
OpenRouter manifests and durable attempt logs, normalized judgments, summary,
CSV, PNG, and PDF together. Direct-mode strict analysis rejects route evidence.

Omitting all three `--expected-model-key` options preserves the original strict
seven-model default: 37,800 judgments and 126 aggregate cells.

## 9. Archive and hand off without deleting audit material

Create a new handoff directory; never reuse or clean an older handoff. Copy,
do not move, the original generation histories and sidecars, the exact judge
template, every reviewed dry
plan, append-only attempt log, manifest, per-model judgment output, combined
analysis/plot outputs, and both detached commit identities. Preserve failed
attempts and superseded plan hashes—they are part of the audit trail:

```bash
export FIGURE6_HANDOFF_ROOT=/absolute/local/path/figure6-handoffs
export FIGURE6_HANDOFF="$FIGURE6_HANDOFF_ROOT/$(date -u +%Y%m%dT%H%M%SZ)-qwen-gpt-oss-nitro"
mkdir -p "$FIGURE6_HANDOFF/generations" "$FIGURE6_HANDOFF/judge" \
  "$FIGURE6_HANDOFF/results" "$FIGURE6_HANDOFF/source-identities"

for MODEL_KEY in qwen32 qwen_mo_mid qwen_mo_post; do
  mkdir -p "$FIGURE6_HANDOFF/generations/$MODEL_KEY" \
    "$FIGURE6_HANDOFF/judge/$MODEL_KEY"
  cp -a "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl" \
    "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl.provenance.json" \
    "$FIGURE6_LOCAL_GENERATION_ROOT/$MODEL_KEY/generations.jsonl.selections.jsonl" \
    "$FIGURE6_HANDOFF/generations/$MODEL_KEY/"
  cp -a "$FIGURE6_JUDGE_ROOT/$MODEL_KEY/openrouter-attempts.jsonl" \
    "$FIGURE6_JUDGE_ROOT/$MODEL_KEY/openrouter-manifest.json" \
    "$FIGURE6_JUDGE_ROOT/$MODEL_KEY"/plan.*.json \
    "$FIGURE6_JUDGE_ROOT/$MODEL_KEY/judgments.jsonl" \
    "$FIGURE6_HANDOFF/judge/$MODEL_KEY/"
done

cp -a "$FIGURE6_JUDGE_TEMPLATE_PATH" "$FIGURE6_HANDOFF/source-identities/"
: "${FIGURE6_JUDGE_PROFILE:?export the exact registered profile used by all three manifests}"
if test "$FIGURE6_JUDGE_PROFILE" = muse-spark-1.1-us-proxy; then
  : "${ROUTE_EVIDENCE:?Muse handoff requires the exact route-evidence JSON}"
  : "${VAST_CONSOLE_EVIDENCE:?Muse handoff requires the sanitized console evidence}"
  test -f "$ROUTE_EVIDENCE"
  test -f "$VAST_CONSOLE_EVIDENCE"
  cp -a "$ROUTE_EVIDENCE" "$VAST_CONSOLE_EVIDENCE" \
    "$FIGURE6_HANDOFF/source-identities/"
elif test "$FIGURE6_JUDGE_PROFILE" != gpt-oss-120b-nitro-direct; then
  printf 'Unexpected handoff judge profile: %s\n' "$FIGURE6_JUDGE_PROFILE" >&2
  exit 1
fi
cp -a "$FIGURE6_JUDGE_ROOT/figure6-aggregation.csv" \
  "$FIGURE6_JUDGE_ROOT/figure6-summary.json" \
  "$FIGURE6_JUDGE_ROOT/figure6-qwen-only-reproduction.png" \
  "$FIGURE6_JUDGE_ROOT/figure6-qwen-only-reproduction.pdf" \
  "$FIGURE6_HANDOFF/results/"

git -C "$FIGURE6_CODE_LOCAL" rev-parse HEAD \
  > "$FIGURE6_HANDOFF/source-identities/repository-commit.txt"
git -C "$FIGURE6_UPSTREAM_CODE_LOCAL" rev-parse HEAD \
  > "$FIGURE6_HANDOFF/source-identities/upstream-commit.txt"
find "$FIGURE6_HANDOFF" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 shasum -a 256 > "$FIGURE6_HANDOFF/SHA256SUMS"
```

Verify the copied row counts (5,400 judgments per Qwen model), rerun strict
analysis from the handoff copies, verify `SHA256SUMS`, and hand off the whole
directory read-only. If the optional Muse profile was used, stop its tunnel
only after all judging, verification, and copying finishes:

```bash
ssh -S "$FIGURE6_CONTROL_SOCKET" -O exit \
  -p "$VAST_SSH_PORT" "root@$VAST_SSH_HOST"
```

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
