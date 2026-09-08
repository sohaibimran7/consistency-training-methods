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

Latest live evidence: the job subsequently started on `nid010883`; telemetry
at 13:09:24 UTC shows the first selected question still decoding after 5,120
model forwards, with about 24.4 GB allocated and no EOS/completion receipt
yet. This is not evidence that the diagnostic or suitability screen passed.

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

## Repaired full-campaign preparation

Separate full-campaign checkout (not submitted):
`/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-gemma4-12b-base-eval-repaired-20260908/repo`.
Campaign relative path:
`logs/gemma4-12b-base-evaluations/attempt-0001/gemma4-12b-base-two-bias-50x21-16gpu-v1`.
Its prepared launch-contract SHA-256 is
`8fac218a3c5c0282aafb84c515aee69367b220b6cf3b7adc907df8e9d47681cd`.
Preparation is not a GPU smoke receipt or a completed evaluation.

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

Next: observe job 6401027 (never duplicate it because observation times out),
inspect memory/EOS evidence, resolve the long-decode risk, run the new
campaign's required discarded smoke, then submit the full 16-GPU screen.
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
