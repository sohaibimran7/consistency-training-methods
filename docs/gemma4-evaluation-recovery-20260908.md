# Gemma 4 12B evaluation recovery

## Objective and protocol

Complete the base-model susceptibility screen before deciding whether RMCT
training has a meaningful bias problem to address, then (if it does) replicate
RMCT on Gemma and compare with Qwen. Preserve the agreed 3 datasets × 7
conditions × 50 questions = 1,050 generations and all 16 GPUs. No output-token
cap is authorized; smoke tests, model generation and Luna grading must remain
uncapped. Existing incomplete results are diagnostic evidence, not a finished
base-model screen.

## Verified state on 8 September

- Fresh two-hop SSH passed the installed `isambard-auth status` / `check`.
- No job was queued/running for the account at the start of recovery.
- Previous Gemma job 6131938 is terminal: `TIMEOUT`, elapsed 05:31:02.
  Its interactive reservation ended on 25 August at 22:34:12.
- The log contains a stale-file-handle failure and an OOM warning on rank 5.
- Preserved files: `artifacts/gemma4-12b-failed-attempt-audit-20260908/`.
  The original remote attempt remains untouched.
- Clean sample records: LogiQA 50, HellaSwag 50, HLE 48. Two HLE records
  belong to an unfinished log, so these counts do not establish full campaign
  validity.
- There are 47 biased sample records but only 16 with generated answers, all
  from wrong-argument LogiQA. The remaining 31 have no model output.
  The task errors show interruption while waiting for clean-log resolution;
  no completed switch or verbalisation publication exists.

## Pairing defect

The launcher rotates question-ID sharding by task index to balance work.
Consequently a rank's biased question IDs do not generally match that rank's
clean question IDs. The switch scorer also searches logged factory arguments
for a cell-level question-ID restriction that the factory logs do not contain.
Waiting on a rank-local clean directory cannot resolve the required pairing.

Recovery separates generation from switch scoring. All immutable shards must
first be merged into their full 50-ID cells; the standard switch scorer must
then use the corresponding clean cell by exact path. Generation must never
wait for a clean-log discovery operation. Regression tests and a fresh smoke
must pass before a replacement campaign is submitted.

## Live diagnostic handoff

Job **6401027**, `gemma4-eos-debug`, was accepted by normal `workq` on 8
September at 12:45:38 UTC. Initial verified state: `PENDING (Resources)`.
It requests one GPU and a 30-minute diagnostic allocation, not a token limit.
It retries only HLE IDs `7b033238cfa808f91555465b7a1f19042caab411` and
`db498bcf69d44b87e3e6183d6f61017edb07ef0c`, recording memory and final EOS
token evidence. It must never contribute samples to the scored campaign.

Final evidence: the job started on `nid010883`, then ended `TIMEOUT` after
00:30:07. Last telemetry at 13:31:55 UTC shows the first selected question
still decoding after 23,808 model forwards, with 24.69 GB allocated and
24.71 GB peak allocation. No EOS/completion receipt was produced. This is an
incomplete diagnostic, not evidence that the suitability screen passed.

Frozen diagnostic checkout:
`/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-gemma4-12b-base-eval-20260908/repo`.
Telemetry will be in `logs/eos-debug-0001/events.jsonl`; a `completed.json`
exists only after both requested generations reach model EOS. Do not sync
further source edits to this checkout while the diagnostic is queued/running.
Use a separate checkout for the repaired full campaign.

Interactive submission was rejected with `QOSMaxSubmitJobPerUserLimit`.
The account also has 21 pending Qwen RMCT jobs, all using normal QOS. Do not
infer from that coincidence that canceling a Qwen job will necessarily clear
the interactive limit. No other task's jobs were changed. Normal batch
submission was accepted under its own limits, with no QOS/security override.
Scheduler forecasts changed sharply; re-query the existing handle rather
than treating any forecast as a promised start or reason to resubmit.

The uncapped sampler already runs one sequence at a time under
`torch.inference_mode()`; lowering `max_connections` is not a demonstrated
memory fix. The pinned Transformers runtime supports an offloaded
`DynamicCache`, potentially retaining sliding-window layers on GPU and
offloading the growing full-attention layers. This is a candidate, not an
implemented or verified change. Any such change must preserve full-prompt
prefill even if an empty cache object is supplied initially: the existing
loop tests `past_key_values is None` to decide full-prompt versus single-token
input. Test real-model parity and memory behavior before adopting offload.
Do not use a fixed-length static cache or token ceiling as a substitute.

## Decoder and GPU-binding audit

The pinned Gemma text-only, batch-one cached forward path does not require
omitted position IDs or per-layer inputs: the sliding cache reports cumulative
sequence length, so positions do not reset at 1,024 tokens. The EOS union
`{1, 50, 106}` matches the saved model generation configuration. However, the
old custom sampler omitted the saved suppression of image/audio token IDs
258882 and 258883 that standard Transformers generation applies. The fix now
uses the standard suppression processor before temperature/top-k/top-p,
inspects the real saved GenerationConfig diff, and fails closed on unsupported
saved processors. It also passes `logits_to_keep=1` where the forward signature
supports it: returning only last-position logits is a memory optimization, not
a generation-token cap. Do not treat the old diagnostic as evidence for the
repaired decoder.

All 16 old Slurm GPU-binding lines report `usable_gres=0x1`, `bit_alloc=0xF`,
`local_inx=4`, `global_list=0`, `local_list=0`. Source audit of Slurm 24.11.7
shows this is the expected task/cgroup local-index display path for
`per_task:1`: it deliberately selects local bit zero in the task-environment
branch, while physical device enforcement is computed separately. These lines
are not evidence of collocation. Preserve the existing valid binding flags.
See Slurm's [GRES guide](https://slurm.schedmd.com/gres.html) and
[srun reference](https://slurm.schedmd.com/srun.html).
New `gemma_gpu_binding.py` makes
every worker attest its CUDA-visible GPU UUID and requires 16 distinct UUIDs,
four workers per node on four nodes, before any model weights load. Its startup
barrier timeout does not constrain generation. Twelve targeted tests passed;
the one-GPU smoke will verify UUID availability in the real CUDA runtime.
The old log also contains at least two weight-load banners per rank; an earlier
informal statement that each rank loaded weights only once was incorrect.

The launcher now binds generation-source hashes (including sampler and GPU
binding sources) into a fresh contract and checks them before smoke and worker
generation. Existing prepared attempt-0001 is preserved; use attempt-0002 once
the new implementation is final. No source edits may be synced into the live
diagnostic checkout.

The combined pinned-runtime suite after these fixes passed **87 tests** in
29.63 seconds. Files were staged only in the separate, unlaunched repaired
checkout. Local openrsync failed before transport with a process-group error;
the remote paths were verified writable and ordinary SSH file copy succeeded.
The actual Inspect batching audit confirms that one module-global worker
thread runs one generator synchronously per rank. The hook's batch size one
means `max_connections=8` merely queues calls, without concurrent KV caches.
All 21 tasks share one resolved Model in the intended code path. The repeated
historical weight-load banners remain unexplained; do not turn them into an
unsupported claim that eight models/caches were simultaneously resident.

New submissions now use `/Users/work/.local/bin/isambard-jobs`, following
`/Users/work/.local/share/ctm-isambard/JOBS.md`. Read shared cached `status`
instead of independent scheduler polling. The controller observed diagnostic
6401027 as terminal and preserved its output claim history; it did not cancel
the job. The other task's interactive RMCT smoke completed without intervention
here.

## Repaired full-campaign preparation

Separate checkout (smoke enqueued; full campaign not submitted):
`/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-gemma4-12b-base-eval-repaired-20260908/repo`.
Current campaign relative path:
`logs/gemma4-12b-base-evaluations/attempt-0002/gemma4-12b-base-two-bias-50x21-16gpu-v1`.
Its prepared launch-contract SHA-256 is
`c5e4e8cf34e170d83789173362cc329de06a8addc23b57638603de0571db9bdc`
(55,797 bytes). Attempt-0001 and its older preparation remain preserved.

Controller smoke request `gemma4-repaired-smoke-a2-20260908` was enqueued at
13:47 UTC, owned by task `01a080f9-4717-7e51-ae52-60f173ed7ee9`. It requests
one GPU and 120 minutes of interactive diagnostic allocation, not a token cap,
and runs two held-out discarded clean samples. Local review manifest:
`artifacts/gemma4-repaired-smoke-20260908-request.json`.
The controller token is `ctm-dc130d98-2e50-5255-913d-db8236991fb4`.
At enqueue the request was locally queued, without a Slurm ID yet. Read shared
controller status to learn admission/outcome; never duplicate this request.
The subsequent shared snapshot confirmed Slurm job **6402593 RUNNING** with
the interactive reservation. This is the existing request, not a second job.
The repaired remote checkout is now frozen: do not sync further scientific
source changes into it while the request is queued or running.
Preparation/enqueueing is not a GPU smoke receipt or a completed evaluation.

### Corrected smoke results

Shared controller accounting subsequently confirmed job 6402593 **COMPLETED**.
The smoke produced a verified receipt and a successful EvalLog. Both held-out
samples reached EOS with no output-token cap: 431 and 397 output tokens. The
receipt SHA-binds the current contract and the log
`2026-09-08T13-49-44-00-00_stage2-ood-unbiased_JptMv3uLYf4acf6xwxJhKH.eval`
(SHA-256 `d573f9f8aea7d62ccead8fe8e982eca5c121017895bc8133981a1724679b3ebe`).
The launcher's existing smoke-receipt validator and the new generation-source
hash validator both passed on the remote files. These two samples remain
discarded diagnostics, outside the scored 50-question pool.

The real CUDA probe succeeded: one visible NVIDIA GH200 120GB, with a nonempty
UUID from `torch.cuda.get_device_properties(0).uuid`. This runtime returns a
plain UUID string without the `GPU-` prefix; the checker correctly does not
require that cosmetic prefix.

Important open issue: the healthy corrected smoke also emitted exactly **two**
complete 677-tensor weight-load banners, counted in binary-safe stderr, without
the historical filesystem/async errors. The double load is therefore genuinely
reproducible in the ordinary single-task, two-sample Inspect entrypoint. It is
not explained by 21-task transitions or multiple concurrent generation caches.
The direct diagnostic loaded once. Trace this call path before committing the
full 16-GPU allocation; do not label it a proven retained-model leak without
additional evidence. The active decoder/lifecycle explorer is investigating.

Final pinned-runtime validation:

- 25 regression tests passed, including actual Inspect task construction,
  a real 50-question standard switch-scoring integration, serialization into
  a complete receipt-bound 21-cell test campaign, and publication preflight.
- An additional check used the **actual frozen deployment manifest**, built
  all 16 ranks (336 Task objects), and verified all 21 cells have exactly 50
  samples, total 1,050, rank workloads 65/66, and zero live switch scorers.
- The real-object check caught and repaired an extra compatibility issue:
  inner Inspect Tasks lack `task_args` at construction. The helper now
  respects that absence, validates actual metadata, and removes only the
  standard live switch scorer.
- Standard NaN conditional-exclusion scores are preserved and accepted by
  postprocess, not converted to unsuccessful switches or inflated denominators.

Next: observe the controller-owned corrected smoke, inspect its EOS/UUID
evidence, and decide whether additional lifecycle/long-HLE diagnostics are
needed before submitting the full 16-GPU screen.
Only completed screen evidence can support the RMCT training decision.

## Completion criteria

1. The 21 cells have exactly the agreed 50 unique IDs, with matched clean and
   biased pools; every accepted generation has verified EOS-only evidence.
2. Standard switch, answer-parsing/accuracy, and uncapped Luna verbalisation
   results are complete, with exclusions and denominators explicit.
3. Compare to the historical Qwen base without silently equalising sample
   sizes or hiding Qwen's historical cap/protocol differences. Do not attach
   cross-model significance claims without a valid comparison design.
4. Use the susceptibility evidence to decide whether to train; if justified,
   run RMCT to a documented convergence criterion rather than an arbitrary
   short checkpoint, evaluate, and compare with Qwen in the standard style.

The full goal remains active; no suitability or training-success conclusion
has been established by the incomplete attempt.

## Training readiness (read-only audit; no training authorized by results yet)

If the completed screen supports training, a GPU preflight must establish the
full Gemma multimodal-wrapper/text-processor route with a pinned local
snapshot, text-tower-only LoRA targets, real PEFT forward/backward and
base-disabled behavior, and HF/rollout-engine parity with a nonzero adapter.
The native-HF evaluation route does not prove training support. Relevant gaps
are in `ctm/backends/local/engine.py`, `ctm/backends/renderers.py`, and the
currently Qwen/Muse-specific runtime parity gates.

Reuse the existing convergence controller and immutable segment evidence,
not an arbitrary 64-step endpoint. Do not copy training defaults: the generic
training CLI currently defaults to 16,384 generated tokens, Qwen plans contain
20,480, and a standard tail-scoring path uses a one-token completion. None is
approved for Gemma. Model context-window overrides also need explicit
inspection for truncation. These are future preflight requirements, not
reasons to substitute a smaller training objective.

## Comparison audit

The current historical Qwen inputs are the r003 publications under
`artifacts/rmct-step16-step176-standard-switch-rate-by-dataset-significance-key-r003-20260821/`
and
`artifacts/rmct-step16-step176-standard-bias-verbalisation-by-dataset-significance-key-r003-20260821/`.
All 50 Gemma IDs per dataset are contained in the archived Qwen base's 100 IDs
per dataset, with no Gemma-only IDs and matching frozen source identities.
LogiQA and HellaSwag selections match Qwen step 16's 50-ID selections; HLE is
a subset of Qwen's original 100. Re-attest these IDs against final Gemma
receipts; the aborted August attempt is only input-design evidence.

`experiments/gemma4_12b_base_eval/combined_comparison.py` preserves full Qwen
display denominators (100 + 100 held-in, 100 held-out per bias) alongside
Gemma's agreed 50 + 50 / 50. It uses the standard renderer and Wilson
intervals but deliberately emits no cross-model significance stars. Historical
Qwen generation used a 20,480-token ceiling (54/1,800 biased outputs reached
it); its Luna grading used 256 (9 hits, 1,791 valid grades). Gemma generation
and grading are uncapped. These historical facts are not approvals to use any
cap in a new run. A protocol-harmonized significance comparison would require
new comparable Qwen evidence or an explicit different estimand, not silent
retroactive stars. No final Gemma chart rows exist yet.

## Model-load trace in preparation

The healthy smoke's two load sequences warrant a decisive trace. A separate
checkout has been created at
`/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-gemma4-12b-base-eval-traced-20260908/repo`
by copying project source only. The completed repaired checkout stays frozen.
New diagnostic script `infra/isambard/trace_gemma_hf_loads.py` will wrap model
constructors, AutoModel/base loaders, and the actual Transformers tensor-load
loop, recording call stacks and CUDA memory without arguments, credentials,
generation changes, or model-retaining references. The launcher now has an
optional `discarded-smoke --trace-model-loads` mode to instrument its actual
`run_evals.py` child; the maintained smoke wrapper requests this mode.

Reviewed controller manifest:
`artifacts/gemma4-traced-smoke-20260908-request.json`, request
`gemma4-traced-smoke-a1-20260908`. It is another two-sample, one-GPU diagnostic
in a fresh campaign beneath the traced checkout's
`logs/gemma4-12b-base-evaluations/attempt-0001/` directory. The final staged
tracer/integration passed **35 pinned-runtime tests** in 27.37 seconds. A
separate CPU check installed all five hooks in the real pinned libraries and
confirmed that doing so did not initialize CUDA.

The fresh contract is sealed, SHA-256
`36f244084a3032ad8f53d39dc792e794b741f93df200f947080669da188747fc`
(56,064 bytes). The existing reviewed request was then **enqueued** through the
shared controller, initially queued without a Slurm ID. Read shared status for
admission; do not submit a duplicate. Do not mutate either frozen smoke
checkout. Its trace will be `discarded-smoke/model-load-trace.jsonl` beneath the
new campaign, and its receipt will bind that trace if the smoke succeeds.

### Decisive trace: usage metadata constructs a second GPU model

The existing controller request ran as job **6403521** and completed. Its two
discarded clean responses reached EOS. Trace SHA-256:
`867f9bd5a0d57e0155c702768e4258db64dbbfd7ec14893b31a800e7cd0fbe8e`
(15,868 bytes). A local copy and its original receipt are preserved under
`artifacts/gemma4-model-load-audit-20260908/`.

The trace identifies two independent `HuggingFaceAPI.__init__` calls in one
evaluator process. The second is triggered after the first completion by
Inspect's transcript usage reporting:
`event_mapping._build_usage_update -> get_model_info -> _get_model_info ->
_resolve_model_info -> get_model -> HuggingFaceAPI.__init__`.
This is not a second evaluation or a concurrency/KV-cache effect. CUDA
allocation increases from 23.919 GB after the first load to 47.895 GB after
the second; both distinct model object IDs remain alive at final shutdown,
with 47.873 GB allocated (48.103 GB peak). The second copy is therefore a
verified retained duplicate, not only a repeated progress display.

A narrow Gemma metadata-registration fix is being implemented and reviewed.
New launcher code also requires a receipt-bound trace proving exactly one HF
constructor, one concrete weight-load call, and the same single live returned
GPU model before admitting the full screen. These new checks do not change
generation or introduce any token cap. The original traced and repaired
checkouts remain immutable; validation of the fix will use a fresh checkout.

The launcher's focused suite passed **12 tests** locally, including duplicate
constructor/load, extra retained model, missing trace hook/final event, and
normal nested Auto/base loader cases. Applying its new gate to the real
6403521 trace correctly rejected the duplicate HF constructor.

A source-only repair checkout has been created at
`/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-gemma4-12b-base-eval-single-load-20260908/repo`.
Its reviewed request manifest is
`artifacts/gemma4-single-load-smoke-20260908-request.json`, request ID
`gemma4-single-load-smoke-a1-20260908`. At this preparation point it is not
enqueued: stage the stable bridge repair, pass pinned tests, seal its fresh
contract, then enqueue this exact request. Its wrapper/resources are unchanged
from the traced smoke: two held-out samples, one GPU, EOS-only, 120-minute
scheduler allocation. The main 1,050-generation campaign remains unsubmitted.
