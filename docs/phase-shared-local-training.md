# Phase-shared local training: status and evidence

## Purpose and scope

This document records the current implementation status for the proposed
phase-shared RMCT/RLCT local training path. The intent is to let the same
allocated GPUs serve two sequential phases of an on-policy update:

1. vLLM rollout workers are awake for sampling and scoring.
2. The workers enter vLLM level-1 sleep; replicated HF/PEFT trainers use those
   same GPUs for forward, backward, and optimizer work.
3. Rank zero verifies that every replica agrees, publishes one adapter version,
   and all workers acknowledge that version before rollout resumes.

The implementation is opt-in and records this placement as execution-only
provenance. It does not change the experiment's objective, data selection,
rollout budgets, or optimizer hyperparameters.

Only `scripts/train_rlct.py` exposes the phase-shared command-line mode in this
change. OPCT, BCT, ACT, AttCT, MLPCT, and target-generation entrypoints reject
the option at argument parsing because their training loops have not yet been
integrated with and tested against the same phase-boundary scheduler.

For Qwen3.5, the runtime also fails closed unless the rollout status directory
contains a worker-parity attestation bound to the exact model path, ordered
GPU topology, worker options, and referenced adapter bytes. The generic
preflight deliberately creates a nonportable, local-path-bound attestation;
the normal training CLI does not silently transplant that evidence into a new
production directory. The representative benchmark must therefore include an
explicit fresh attestation handoff in its own immutable launch contract.

**Current limitation:** the Qwen parity-attestation schema binds the model
path, topology, worker options, and adapter hashes, but does not yet
cryptographically bind the installed vLLM or Transformers package/build
versions. Before production, run a fresh live parity probe or extend the
launch contract to bind those runtime versions as well; do not treat an older
attestation as proof of parity after a runtime change.

The terms below are important:

- **Implemented** means the repository contains the code and focused offline
  tests cover its contract.
- **Hardware preflight passed** means the specific non-production Isambard job
  described below exercised that behavior on four GH200 GPUs.
- **Production benchmark pending** means no speed, cost, or topology-selection
  claim should be made yet.

## Current optimisation status

| Intervention | Confidence | Implementation status | Evidence and boundary |
| --- | --- | --- | --- |
| Bounded forward microbatches / selected-token path | High | Implemented | The local engine bounds internal forwards by datum count and padded-token budget while retaining the logical global loss. It is a memory-safety and packing mechanism; its production throughput benefit still needs measurement. |
| Selective gradient checkpointing | High | Implemented | The engine can checkpoint every backbone layer or the first *N* layers. This reduces activation memory at a recomputation cost; the best setting remains workload-dependent. |
| vLLM level-1 sleep/wake | High | Implemented; hardware preflight passed | Four workers slept and woke in Isambard job `5989987`; the preflight recorded stable scores across the first transition. It does not yet establish long-run transition cost under representative RMCT updates. |
| Manual replicated LoRA update | High for correctness; medium for realised speedup | Implemented; fixed-update hardware preflight passed | This is **not `torch.nn.parallel.DistributedDataParallel`**. Independent HF/PEFT replicas deterministically shard work, use one global denominator, and NCCL-**SUM** flattened LoRA gradients before AdamW. A fixed synthetic update passed the gradient and adapter-safety gates; a production-shaped PPO/GRPO equivalence run is still pending. |
| Persistent rollout-worker pool | High | Implemented | Worker processes maintain their own device visibility, seed, adapter version, health receipt, and fail-closed request protocol. Offline lifecycle and compatibility tests pass. |
| Qwen3.5 PEFT-to-vLLM translated adapter | High | Implemented; hardware preflight passed | Every published Qwen3.5 policy version gets a hash-bound vLLM-only adapter sibling. This prevents vLLM from silently accepting the incompatible text-tower key spelling. The preflight checked a non-zero post-update effect on all workers. |
| 2/4/8 GPU topology resolver | High for configuration; limited hardware coverage | Implemented | CPU tests exercise 2-, 4-, and 8-GPU logical plans without assuming device ordinals. The successful hardware run used four GH200 GPUs; it recorded 2- and 4-GPU contracts and correctly marked 8 GPUs unavailable. |
| Checkpoint / strict resume sidecar | Medium–high | Implemented, CPU-tested | Replicated checkpoints include rank-specific RNG and topology state and reject incompatible strict resumes. A real multi-GPU resume cycle is not yet measured. |
| Three-lane partial sharing | Medium | Implemented as a topology option; production benchmark pending | The code supports a partial overlap layout, but no representative 1–2 update benchmark has run. |
| Four-lane full co-residency | Medium | Implemented; coexistence preflight passed | The four-GPU preflight ran a worker and HF trainer on each GPU at different lifecycle points. It was not a representative RMCT PPO/GRPO update benchmark. |
| Choosing three versus four lanes | Not yet known | Not decided | The agreed decision needs comparable 1–2 real-update measurements. We should choose four lanes only if its transition/memory penalty remains below the measured benefit threshold; no such comparison exists yet. |
| FSDP | Not implemented | Deliberately deferred | The current design uses manual exact replicated LoRA training. FSDP should not be added unless the replicated/co-resident route fails its production benchmark or memory gate. |

## What the successful preflight establishes

The non-production Isambard job `5989987` successfully ran on four GH200
GPUs. Its immutable copied evidence is intentionally kept outside Git in the
project's experiment-artifact store, under the run label
`phase-shared-preflight-4h100-20260810/isambard-5989987-20260811`. The full
tree is hundreds of megabytes because it contains gradient tensors, adapters,
worker receipts, and logs; collaborators should obtain that checksum-verified
artifact bundle from the project store rather than treating this code PR as an
evidence archive.

The authoritative receipt labels the work
`non_production_phase_shared_preflight_contract`; it did not touch production
output. It establishes all of the following on that hardware/software stack:

- pinned Qwen3.5 snapshot readiness and all four safetensor shard checks;
- four visible-GPU health checks and all-four-worker / all-four-trainer
  coexistence;
- a synthetic fixed cross-entropy update with eight 512-token datums,
  compared to a single-rank reference;
- pre-optimizer global-gradient cosine `0.99999915` and norm ratio
  `0.9999873` for the manually NCCL-SUM-reduced replica update;
- successful vLLM sleep/wake, adapter publication, and non-zero worker LoRA
  effect checks; and
- rank-zero capacity-only sweeps at 20,480, 40,960, and 49,152 padded tokens,
  with all vLLM workers asleep.

It does **not** establish any of the following:

- a real RMCT PPO, pooled-GRPO, or OPCT update;
- a production multi-update correctness proof;
- a throughput, GPU-hour, or dollar saving;
- a three-lane versus four-lane decision; or
- 8-GPU hardware behavior.

The post-Adam diagnostic differences in the preflight are retained for
calibration but are explicitly monitoring-only; hard pre-optimizer gradient
and adapter-safety gates passed. The implementation must therefore be
described as tolerance-equivalent for the tested synthetic update, not
bitwise-identical.

## Next experiment before production

Run the following two non-production, representative benchmarks serially on
the same four-GPU allocation and frozen input batch:

1. Three-lane partial sharing for one update, then (if healthy) a second warm
   update.
2. Four-lane full co-residency for the same workload and update count.

For each, retain the resolved topology, transition timings, GPU memory
snapshots, rollout and training timing split, completion-token counts,
checkpoint hashes, and a comparison receipt. The workload must be a real
on-policy RMCT/PPO or pooled-GRPO update—not the short fixed CE probe. Only
then compare end-to-end seconds/update and resource cost, decide whether full
co-residency clears the agreed benefit threshold, and permit a production
launch. For Qwen3.5, each benchmark must also generate and bind its own
runtime-path/topology-specific worker-parity attestation before training
starts; copying the prior preflight JSON alone is intentionally unsupported.

## Operating the non-production preflight

On Isambard, first prepare the GPU runtime inside an allocated GH200 session,
then submit the wrapper with an explicit durable checkout:

```bash
sbatch --reservation=interactive --export=ALL,REPO_DIR="$PWD" \
  infra/isambard/preflight_qwen35_phase_shared.sbatch
```

The wrapper preserves Slurm's `CUDA_VISIBLE_DEVICES` allocation, creates a
fresh evidence path, and does not load experiment credentials or production
datasets. It is safe to use for validation only; it must not be presented as a
training result.
