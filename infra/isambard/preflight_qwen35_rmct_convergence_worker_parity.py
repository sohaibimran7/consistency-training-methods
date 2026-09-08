#!/usr/bin/env python3
"""Create the one production-path-bound worker-parity sidecar for RMCT convergence.

This is deliberately narrower than ``infra/vastai/preflight_qwen35_phase_shared.py``.
It does *not* benchmark a topology or perform a packing sweep.  It starts the
same four phase-shared trainer/worker lanes as RMCT convergence, takes one
synthetic replicated LoRA update, and requires the translated vLLM adapter to
have a measurable, HF-consistent effect on every worker.  The resulting
attestation is immutable and path-bound to the exact local Qwen snapshot and
the exact worker options that the production command uses.

The fastpath has two modes:

* a fresh invocation creates ``--output-dir`` and writes all evidence beneath
  it; and
* ``--resume`` is read-only and revalidates an already complete sidecar.

It never writes a production run directory, checkpoint, or training marker.
``SUCCESS`` is created only after worker teardown and sidecar revalidation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "rmct-convergence-worker-parity-fastpath-v1"
MODEL = "Qwen/Qwen3.5-9B"
WORKER_PARITY_ATTESTATION = "qwen35-rollout-worker-parity-attestation.json"
RESULT = "result.json"
SUCCESS = "SUCCESS"
GPU_COUNT = 4
LORA_SEED = 42
WORKER_SEED_BASE = 42
LEARNING_RATE = 1e-4


class PreflightError(RuntimeError):
    """The exact production worker-parity gate could not be proven."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write an immutable receipt once; never replace prior evidence."""

    payload = _canonical_json(dict(value))
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PreflightError(f"refusing to replace immutable evidence: {path}") from exc


def _write_success(path: Path) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("rmct convergence worker parity passed\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PreflightError(f"refusing to replace existing success marker: {path}") from exc


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} must be a regular file: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(parsed, dict):
        raise PreflightError(f"{label} must be a JSON object: {path}")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh global RMCT-convergence worker-parity evidence directory",
    )
    parser.add_argument(
        "--model-snapshot",
        type=Path,
        required=True,
        help="Exact resolved offline Qwen3.5 snapshot path passed to every production segment",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Read-only validation of an existing complete immutable sidecar",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact worker contract without probing CUDA or writing files",
    )
    args = parser.parse_args(argv)
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run are mutually exclusive")
    return args


def _snapshot_path(raw: Path, *, require_exists: bool) -> Path:
    path = raw.expanduser().resolve()
    if require_exists:
        if not path.is_dir() or path.is_symlink() or not (path / "config.json").is_file():
            raise PreflightError(f"model snapshot must be a regular Qwen snapshot directory: {path}")
    return path


def _expected_worker_engine_kwargs() -> dict[str, Any]:
    """Return the normalized worker ABI of the production compiler.

    ``RolloutWorkerPool`` appends ``logprobs_mode`` and
    ``tensor_parallel_size`` when it constructs a pool.  Include them here so
    resume validation compares exactly the same canonical option map as the
    production backend, not an approximate CLI fragment.
    """

    return {
        "gpu_memory_utilization": 0.34,
        "dtype": "bfloat16",
        "enable_sleep_mode": True,
        "language_model_only": True,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "max_model_len": 32768,
        "gdn_prefill_backend": "triton",
        "seed": WORKER_SEED_BASE,
        "logprobs_mode": "processed_logprobs",
        "tensor_parallel_size": 1,
    }


def _worker_contract(*, cuda_visible_devices: str | None) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    """Resolve exactly the four fully-overlapped production lanes without CUDA I/O."""

    from ctm.backends.local.phase_shared import resolve_phase_shared_topology
    from ctm.backends.local.rollout_workers import RolloutGPU

    topology = resolve_phase_shared_topology(
        train_gpus_spec="all",
        rollout_gpus_spec="all",
        cuda_visible_devices=cuda_visible_devices,
        allow_overlap=True,
    )
    if topology.world_size != GPU_COUNT or len(topology.rollout_gpus) != GPU_COUNT:
        raise PreflightError(
            f"RMCT convergence requires exactly {GPU_COUNT} fully phase-shared GPUs; "
            f"got training={len(topology.train_gpus)}, rollout={len(topology.rollout_gpus)}"
        )
    if tuple(gpu.logical_index for gpu in topology.train_gpus) != tuple(range(GPU_COUNT)):
        raise PreflightError("RMCT convergence requires all four logical GPUs in canonical order 0,1,2,3")
    if tuple(gpu.logical_index for gpu in topology.rollout_gpus) != tuple(range(GPU_COUNT)):
        raise PreflightError("RMCT convergence requires all four rollout GPUs in canonical order 0,1,2,3")
    workers = tuple(RolloutGPU(gpu.logical_index, gpu.device_token) for gpu in topology.rollout_gpus)
    return topology, workers, _expected_worker_engine_kwargs()


def _dry_run_document(snapshot: Path, *, cuda_visible_devices: str | None) -> dict[str, Any]:
    topology, workers, worker_kwargs = _worker_contract(cuda_visible_devices=cuda_visible_devices)
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "non_production": True,
        "model": str(snapshot),
        "training_gpus": [gpu.logical_index for gpu in topology.train_gpus],
        "rollout_gpus": [gpu.as_dict() for gpu in workers],
        "worker_engine_kwargs": worker_kwargs,
        "synthetic_update": {
            "replicated_world_size": GPU_COUNT,
            "datums": 2 * GPU_COUNT,
            "sequence_tokens": 512,
            "optimizer_steps": 1,
            "learning_rate": LEARNING_RATE,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "seed": LORA_SEED},
        },
        "gates": [
            "all four phase-shared workers start with exact production engine kwargs",
            "all workers acknowledge level-1 sleep before replicated trainer update",
            "one real synchronized LoRA update is published as v2",
            "every worker has a nonzero HF-consistent translated-LoRA fixed-token effect",
            "attestation validates against the exact production model path/topology/engine kwargs",
        ],
    }


def _validate_existing(*, output_dir: Path, snapshot: Path, cuda_visible_devices: str | None) -> dict[str, Any]:
    """Read-only revalidation used by the launcher before every first segment."""

    from ctm.backends.local.qwen35_vllm_compat import validate_qwen35_rollout_worker_parity_attestation

    if output_dir.is_symlink() or not output_dir.is_dir():
        raise PreflightError(f"worker-parity output directory must be a regular directory: {output_dir}")
    success = output_dir / SUCCESS
    if success.is_symlink() or not success.is_file():
        raise PreflightError(f"completed worker-parity evidence lacks terminal SUCCESS marker: {success}")
    result = _read_json(output_dir / RESULT, label="worker-parity result")
    if result.get("schema") != SCHEMA or result.get("status") != "passed" or result.get("passed") is not True:
        raise PreflightError("worker-parity result is not a passed fastpath receipt")
    if result.get("model_snapshot") != str(snapshot):
        raise PreflightError(
            "worker-parity receipt is bound to a different exact model snapshot: "
            f"recorded={result.get('model_snapshot')!r}, expected={str(snapshot)!r}"
        )
    topology, workers, worker_kwargs = _worker_contract(cuda_visible_devices=cuda_visible_devices)
    if result.get("worker_engine_kwargs") != worker_kwargs:
        raise PreflightError("worker-parity result does not carry the exact production worker options")
    if result.get("worker_gpus") != [gpu.as_dict() for gpu in workers]:
        raise PreflightError("worker-parity result does not carry the exact production worker topology")
    if result.get("training_gpus") != [gpu.logical_index for gpu in topology.train_gpus]:
        raise PreflightError("worker-parity result does not carry the exact production trainer topology")
    attestation = output_dir / WORKER_PARITY_ATTESTATION
    validated = validate_qwen35_rollout_worker_parity_attestation(
        attestation,
        expected_model=str(snapshot),
        expected_worker_gpus=workers,
        expected_worker_engine_kwargs=worker_kwargs,
    )
    if result.get("attestation_sha256") != _sha256(attestation):
        raise PreflightError("worker-parity result checksum does not match its immutable sidecar")
    if result.get("attestation_schema") != validated.get("schema"):
        raise PreflightError("worker-parity result schema does not match its immutable sidecar")
    return {"result": result, "attestation": validated}


async def _run_fresh(*, output_dir: Path, snapshot: Path, cuda_visible_devices: str | None) -> dict[str, Any]:
    """Perform the minimum real production-shaped parity proof.

    The generic phase-shared preflight owns the carefully tested fixed-token
    datum recipe, worker scorer, and mathematical parity gates.  Reusing those
    helpers keeps this fastpath scientifically identical where it matters,
    while intentionally omitting its topology and capacity benchmarks.
    """

    import torch
    from tinker import types
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from transformers import AutoConfig, AutoTokenizer

    from ctm.backends.local.engine import LocalBackend
    from ctm.backends.local.qwen35_vllm_compat import (
        validate_qwen35_rollout_worker_parity_attestation,
        write_qwen35_rollout_worker_parity_attestation,
    )
    from ctm.backends.local.replicated import LocalBackendConstructorSpec, ReplicatedTrainingBackend
    from ctm.backends.local.rollout_workers import RolloutParallelBackend
    from ctm.core.config import AdamConfig, LoRAConfig
    from infra.vastai import preflight_qwen35_phase_shared as generic

    if not torch.cuda.is_available():
        raise PreflightError("RMCT convergence worker-parity fastpath requires CUDA")
    topology, workers, expected_worker_kwargs = _worker_contract(cuda_visible_devices=cuda_visible_devices)
    if torch.cuda.device_count() < GPU_COUNT:
        raise PreflightError(
            f"torch reports fewer visible devices than the four-lane RMCT contract: torch={torch.cuda.device_count()}"
        )
    coordinator = f"cuda:{topology.coordinator.logical_index}"
    torch.cuda.set_device(topology.coordinator.logical_index)

    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    if getattr(config, "model_type", None) != "qwen3_5":
        raise PreflightError(
            f"model snapshot is not Qwen3.5: model_type={getattr(config, 'model_type', None)!r}, snapshot={snapshot}"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    events = output_dir / "events.jsonl"

    def event(name: str, **fields: Any) -> None:
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": _utc_now(), "event": name, **fields}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    contract = _dry_run_document(snapshot, cuda_visible_devices=cuda_visible_devices)
    contract.update({"dry_run": False, "created_at": _utc_now()})
    _write_new_json(output_dir / "contract.json", contract)
    event("contract_written")

    # This map is deliberately the same as the training CLI's *coordinator*
    # map.  It contains no worker seed because its vLLM sampler stays cold;
    # the RolloutParallelBackend owns the four actual seeded worker engines.
    coordinator_vllm_options = {
        key: value
        for key, value in expected_worker_kwargs.items()
        # ``_vllm_options(args, worker=False)`` in the production CLI leaves
        # worker-only scheduling/RNG fields out.  The coordinator sampler
        # remains cold, but retaining this exact option shape eliminates an
        # unnecessary difference between the fastpath and production setup.
        if key not in {"seed", "tensor_parallel_size", "logprobs_mode"}
    }
    training_kwargs = {
        "dtype": torch.bfloat16,
        "use_lora": True,
        "sampler": "vllm",
        "vllm_options": coordinator_vllm_options,
        "gradient_checkpointing": True,
        "gradient_checkpointing_layers": 16,
        "forward_microbatch_max_datums": 8,
        "forward_microbatch_max_tokens": 20_480,
        "target_logprob_chunk_size": 2_048,
    }
    datums = generic._make_cross_entropy_datums(
        tokenizer=tokenizer,
        count=2 * GPU_COUNT,
        sequence_length=512,
        torch=torch,
        types=types,
        datum_from_model_input_weights=datum_from_model_input_weights,
    )
    candidates, completions, fixed_probe_contract = generic._make_update_aligned_probe_candidates(
        tokenizer,
        types,
        count=4,
        sequence_length=512,
        completion_tokens=256,
    )
    torch.manual_seed(LORA_SEED)
    rank_zero = LocalBackend(device=coordinator, **training_kwargs)
    replicated = ReplicatedTrainingBackend(
        rank_zero,
        topology=topology,
        child_backend_spec=LocalBackendConstructorSpec(LocalBackend, kwargs=training_kwargs),
        start_timeout_seconds=1_800.0,
        command_timeout_seconds=7_200.0,
        shutdown_timeout_seconds=30.0,
    )
    backend = RolloutParallelBackend(
        replicated,
        gpus=workers,
        status_dir=output_dir / "rollout_workers",
        worker_vllm_options=expected_worker_kwargs,
        start_timeout_seconds=1_800.0,
        request_timeout_seconds=7_200.0,
        shutdown_timeout_seconds=30.0,
        # This isolated, non-production path is the only route permitted to
        # bootstrap the sidecar consumed later by the production backend.
        qwen35_rollout_parity_bootstrap=True,
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "passed": False,
        "created_at": _utc_now(),
        "model_snapshot": str(snapshot),
        "training_gpus": [gpu.logical_index for gpu in topology.train_gpus],
        "worker_gpus": [gpu.as_dict() for gpu in workers],
        "worker_engine_kwargs": expected_worker_kwargs,
        "fixed_update": {
            "datums": len(datums),
            "sequence_tokens": 512,
            "optimizer_steps": 1,
            "learning_rate": LEARNING_RATE,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "seed": LORA_SEED},
        },
        "fixed_token_probe_contract": fixed_probe_contract,
        "non_production": True,
    }
    started = time.monotonic()
    try:
        event("backend_setup_started")
        backend.setup(
            model=str(snapshot),
            lora=LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.0,
                seed=LORA_SEED,
                train_attn=True,
                train_mlp=True,
                train_unembed=False,
            ),
        )
        if backend.pool is None:
            raise PreflightError("rollout backend setup did not create a worker pool")
        if backend.pool.engine_kwargs != expected_worker_kwargs:
            raise PreflightError(
                "worker pool normalized options differ from the exact production parity contract: "
                f"actual={backend.pool.engine_kwargs!r}, expected={expected_worker_kwargs!r}"
            )
        awake = await backend.pool.health()
        if any(response.get("sleeping") is not False for response in awake):
            raise PreflightError("all workers must start awake before the phase-shared parity update")
        result["initial_worker_health"] = awake
        event("backend_setup_completed", worker_count=len(workers))

        await backend.enter_training_phase()
        asleep = await backend.pool.health()
        if any(response.get("sleeping") is not True for response in asleep):
            raise PreflightError("all workers must acknowledge level-1 sleep before trainer update")
        result["asleep_worker_health"] = asleep
        event("workers_asleep")

        forward_started = time.monotonic()
        forward = await generic._resolve_pending(await backend.submit_forward_backward(datums, "cross_entropy"))
        optimizer_started = time.monotonic()
        await generic._resolve_pending(
            await backend.submit_optim_step(
                learning_rate=LEARNING_RATE,
                adam=AdamConfig(
                    learning_rate=LEARNING_RATE,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                    weight_decay=0.0,
                    grad_clip_norm=1.0,
                ),
            )
        )
        result["replicated_update"] = {
            "loss": float(forward.metrics["loss"]),
            "forward_backward_seconds": time.monotonic() - forward_started,
            "optimizer_seconds": time.monotonic() - optimizer_started,
        }
        event("replicated_update_completed", loss=result["replicated_update"]["loss"])

        await backend.refresh_policy_sampler("rmct_convergence_worker_parity_v2")
        awake_after = await backend.pool.health()
        if any(response.get("sleeping") is not False for response in awake_after):
            raise PreflightError("all workers must be awake after publishing the v2 adapter")
        if backend.pool.adapter_version != 2:
            raise PreflightError(
                f"expected one base and one updated adapter snapshot (v2); got v{backend.pool.adapter_version}"
            )
        result["post_publish_worker_health"] = awake_after
        event("updated_adapter_published", adapter_version=backend.pool.adapter_version)

        effect = await generic._post_update_worker_effect(
            backend=backend,
            rank_zero=rank_zero,
            candidates=candidates,
            candidate_completions=completions,
            fixed_token_probe_contract=fixed_probe_contract,
            worker_count=len(workers),
            worker_gpus=workers,
            min_effect=1e-5,
            cosine_min=0.90,
            norm_ratio_min=0.80,
            norm_ratio_max=1.25,
            relative_l2_max=0.25,
            diagnostic_sink=lambda report: result.__setitem__("worker_effect", report),
        )
        result["worker_effect"] = effect
        adapter_root = backend.pool.status_dir / "adapters" / f"v{backend.pool.adapter_version:08d}"
        attestation_path = output_dir / WORKER_PARITY_ATTESTATION
        attestation = write_qwen35_rollout_worker_parity_attestation(
            attestation_path,
            model=str(snapshot),
            raw_adapter=adapter_root / "raw",
            vllm_adapter=adapter_root / "vllm_compat",
            adapter_version=backend.pool.adapter_version,
            worker_gpus=workers,
            worker_engine_kwargs=backend.pool.engine_kwargs,
            fixed_token_probe=effect["fixed_token_probe"],
            aggregate_effect_parity=effect["formal_aggregate_effect_parity"],
            per_worker_effect_parity=effect["formal_per_worker_effect_parity"],
        )
        revalidated = validate_qwen35_rollout_worker_parity_attestation(
            attestation_path,
            expected_model=str(snapshot),
            expected_worker_gpus=workers,
            expected_worker_engine_kwargs=expected_worker_kwargs,
        )
        if revalidated != attestation:
            raise PreflightError("freshly written worker-parity sidecar did not revalidate identically")
        result.update(
            {
                "attestation_path": str(attestation_path.resolve()),
                "attestation_sha256": _sha256(attestation_path),
                "attestation_schema": attestation["schema"],
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        # Teardown precedes the passed receipt so SUCCESS can never describe
        # an instance whose worker processes failed to shut down.
        backend.shutdown()
        backend = None
        result["status"] = "passed"
        result["passed"] = True
        _write_new_json(output_dir / RESULT, result)
        event("worker_parity_passed", attestation_sha256=result["attestation_sha256"])
        _write_success(output_dir / SUCCESS)
        return result
    except BaseException as exc:
        result.update(
            {
                "status": "failed",
                "passed": False,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        # The output directory is fresh, so a failure receipt cannot replace
        # earlier production or preflight evidence.
        _write_new_json(output_dir / "failure.json", result)
        event("worker_parity_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if backend is not None:
            backend.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot = _snapshot_path(args.model_snapshot, require_exists=not args.dry_run)
    output_dir = args.output_dir.expanduser().resolve()
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.dry_run:
        print(json.dumps(_dry_run_document(snapshot, cuda_visible_devices=cuda_visible_devices), indent=2, sort_keys=True))
        return 0
    if args.resume:
        validated = _validate_existing(
            output_dir=output_dir,
            snapshot=snapshot,
            cuda_visible_devices=cuda_visible_devices,
        )
        print("RMCT_CONVERGENCE_WORKER_PARITY_RESUMED=1")
        print("RMCT_CONVERGENCE_WORKER_PARITY_ATTESTATION=" + str((output_dir / WORKER_PARITY_ATTESTATION).resolve()))
        print("RMCT_CONVERGENCE_WORKER_PARITY_SHA256=" + _sha256(output_dir / WORKER_PARITY_ATTESTATION))
        print("RMCT_CONVERGENCE_WORKER_PARITY_SCHEMA=" + str(validated["attestation"]["schema"]))
        return 0
    if output_dir.exists() or output_dir.is_symlink():
        raise PreflightError(
            f"fresh worker-parity invocation refuses existing output directory; use --resume only for a completed receipt: {output_dir}"
        )
    result = asyncio.run(_run_fresh(output_dir=output_dir, snapshot=snapshot, cuda_visible_devices=cuda_visible_devices))
    print("RMCT_CONVERGENCE_WORKER_PARITY_ATTESTATION=" + result["attestation_path"])
    print("RMCT_CONVERGENCE_WORKER_PARITY_SHA256=" + result["attestation_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
