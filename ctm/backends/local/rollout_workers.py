"""Data-parallel vLLM rollout workers for the local RL backend.

This module deliberately does *not* provide tensor parallelism.  The
coordinator keeps the trainable transformers/PEFT model on its explicitly
selected GPU, while each rollout GPU owns an independent process and one vLLM
engine.  A request's ``(prompt_index, sample_index)`` slots are partitioned
across those workers and reconstructed in the original order.

Large token/logprob matrices never pass through ``multiprocessing.Connection``.
Workers put the exact int64 token ids and binary64 Python logprob values in one
bounded, file-backed payload and send only its descriptor over the pipe.  There
is at most one in-flight command (and therefore one payload) per worker.

Each vLLM engine receives a distinct, deterministic stream seed derived from
one pool seed.  This is an engine-level seed rather than a fixed per-request
``SamplingParams`` seed: the stream advances normally between requests while
workers cannot silently start from vLLM's shared default seed.  Multiple
engines are therefore not expected to be byte-identical to a single engine,
but every slot is sampled with the same generation parameters from the
requested base or explicitly acknowledged policy distribution.
"""

from __future__ import annotations

import array
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import mmap
import multiprocessing
from multiprocessing.connection import Connection
import os
from pathlib import Path
import re
import secrets
import struct
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Optional
import uuid

from ctm.backends.base import SampledSequence
from ctm.backends.local.muse_glimmer import (
    PARITY_ATTESTATION_NAME as MUSE_PARITY_ATTESTATION_NAME,
    file_sha256 as muse_file_sha256,
    is_muse_glimmer_model_name,
    validate_muse_rollout_worker_parity_attestation,
)
from ctm.backends.local.qwen35_vllm_compat import (
    MANIFEST_NAME as QWEN35_COMPAT_MANIFEST_NAME,
    WORKER_PARITY_ATTESTATION_NAME as QWEN35_WORKER_PARITY_ATTESTATION_NAME,
    file_sha256,
    is_qwen35_model_name,
    materialize_qwen35_vllm_rollout_compat_adapter,
    validate_qwen35_rollout_worker_parity_attestation,
)

_PAYLOAD_MAGIC = b"CTMRP002"
_PAYLOAD_HEADER = struct.Struct("<8sQ")
_PAYLOAD_RECORD = struct.Struct("<qqqqq")
_FINISH_REASONS = ("unknown", "stop", "length", "error")
_INT64_BYTES = 8
_FLOAT64_BYTES = 8
_MAX_ENGINE_SEED = 2**31 - 1
_ENGINE_SEED_POLICY = "base_plus_worker_id_v1"


def _validated_engine_seed_base(value: Any, *, worker_count: int) -> int:
    """Return a seed base whose derived worker seeds remain valid integers."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("rollout worker vLLM seed must be an integer")
    if value < 0 or value + worker_count - 1 > _MAX_ENGINE_SEED:
        raise ValueError(
            "rollout worker vLLM seed must be in "
            f"[0, {_MAX_ENGINE_SEED - worker_count + 1}] for {worker_count} worker(s)"
        )
    return value


class RolloutWorkerError(RuntimeError):
    """A rollout worker failed, timed out, or violated the pool protocol."""


def _sampler_sleep_enabled(sampler: Any) -> bool:
    """Read optional lifecycle state without breaking legacy test samplers."""

    return bool(getattr(sampler, "sleep_enabled", False))


def _sampler_sleeping(sampler: Any) -> bool:
    """Read optional lifecycle state without breaking legacy test samplers."""

    return bool(getattr(sampler, "sleeping", False))


@dataclass(frozen=True)
class RolloutGPU:
    """One worker's GPU mapping.

    ``logical_index`` is relative to the coordinator's inherited
    ``CUDA_VISIBLE_DEVICES``.  ``device_token`` is the corresponding physical
    ordinal, GPU UUID, or MIG UUID.  The worker receives only that one token and
    therefore sees its engine device as local ``cuda:0``.
    """

    logical_index: int
    device_token: str

    def as_dict(self) -> dict[str, Any]:
        return {"logical_index": self.logical_index, "device_token": self.device_token}


def resolve_rollout_gpus(
    spec: str,
    *,
    cuda_visible_devices: Optional[str],
    coordinator_device: Optional[str],
) -> tuple[RolloutGPU, ...]:
    """Resolve a comma-separated list of logical rollout GPU indices safely.

    Parallel rollout mode requires an explicit inherited visibility list and an
    explicit coordinator device (for example ``cuda:0``).  This prevents a
    worker from naming or probing any GPU outside the allocation supplied to the
    command.  The returned physical/UUID token is copied from that inherited
    list verbatim; no CUDA discovery call is made.
    """

    raw = (spec or "").strip()
    if not raw:
        raise ValueError("rollout GPU list must not be empty")
    visible_raw = (cuda_visible_devices or "").strip()
    if not visible_raw or visible_raw in {"-1", "NoDevFiles"}:
        raise ValueError("parallel rollout requires an explicit non-empty CUDA_VISIBLE_DEVICES allocation")
    visible = [token.strip() for token in visible_raw.split(",")]
    if any(not token for token in visible):
        raise ValueError(f"invalid CUDA_VISIBLE_DEVICES={visible_raw!r}")
    if len(set(visible)) != len(visible):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")

    match = re.fullmatch(r"cuda:(\d+)", (coordinator_device or "").strip())
    if match is None:
        raise ValueError("parallel rollout requires an explicit --local-device cuda:N coordinator assignment")
    coordinator_index = int(match.group(1))
    if coordinator_index >= len(visible):
        raise ValueError(
            f"coordinator logical GPU {coordinator_index} is outside CUDA_VISIBLE_DEVICES ({len(visible)} device(s))"
        )

    pieces = [piece.strip() for piece in raw.split(",")]
    if any(not piece for piece in pieces):
        raise ValueError(f"invalid rollout GPU list: {spec!r}")
    try:
        logical = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("rollout GPUs must be comma-separated logical integer indices") from exc
    if any(index < 0 for index in logical):
        raise ValueError("rollout GPU logical indices must be non-negative")
    if len(set(logical)) != len(logical):
        raise ValueError("rollout GPU logical indices must be unique")
    outside = [index for index in logical if index >= len(visible)]
    if outside:
        raise ValueError(
            f"rollout logical GPU index/indices {outside} are outside CUDA_VISIBLE_DEVICES ({len(visible)} device(s))"
        )
    if coordinator_index in logical:
        raise ValueError(f"coordinator logical GPU {coordinator_index} cannot also host a rollout worker")
    return tuple(RolloutGPU(index, visible[index]) for index in logical)


def resolve_base_only_gpus(
    spec: str,
    *,
    cuda_visible_devices: Optional[str],
) -> tuple[RolloutGPU, ...]:
    """Resolve logical GPUs for a coordinator-free frozen-base worker pool.

    Unlike training rollout mode, target generation has no Transformers/PEFT
    coordinator model. Every listed logical GPU therefore owns one independent
    vLLM process, including logical GPU 0. The indices are still constrained to
    the command's inherited ``CUDA_VISIBLE_DEVICES`` allocation.
    """

    raw = (spec or "").strip()
    if not raw:
        raise ValueError("base-only GPU list must not be empty")
    visible_raw = (cuda_visible_devices or "").strip()
    if not visible_raw or visible_raw in {"-1", "NoDevFiles"}:
        raise ValueError("base-only workers require an explicit non-empty CUDA_VISIBLE_DEVICES allocation")
    visible = [token.strip() for token in visible_raw.split(",")]
    if any(not token for token in visible):
        raise ValueError(f"invalid CUDA_VISIBLE_DEVICES={visible_raw!r}")
    if len(set(visible)) != len(visible):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate device tokens")

    pieces = [piece.strip() for piece in raw.split(",")]
    if any(not piece for piece in pieces):
        raise ValueError(f"invalid base-only GPU list: {spec!r}")
    try:
        logical = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("base-only GPUs must be comma-separated logical integer indices") from exc
    if any(index < 0 for index in logical):
        raise ValueError("base-only GPU logical indices must be non-negative")
    if len(set(logical)) != len(logical):
        raise ValueError("base-only GPU logical indices must be unique")
    outside = [index for index in logical if index >= len(visible)]
    if outside:
        raise ValueError(
            f"base-only logical GPU index/indices {outside} are outside CUDA_VISIBLE_DEVICES "
            f"({len(visible)} device(s))"
        )
    return tuple(RolloutGPU(index, visible[index]) for index in logical)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"at": _utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _payload_size(records: Sequence[tuple[int, int, SampledSequence]]) -> int:
    total = _PAYLOAD_HEADER.size
    for _prompt_index, _sample_index, sequence in records:
        sequence.validate()
        n_tokens = len(sequence.tokens)
        n_logprobs = -1 if sequence.logprobs is None else len(sequence.logprobs)
        if n_logprobs not in {-1, n_tokens}:
            raise RolloutWorkerError(f"sequence has {n_tokens} token(s) but {n_logprobs} logprob(s)")
        total += _PAYLOAD_RECORD.size + n_tokens * _INT64_BYTES
        if n_logprobs >= 0:
            total += n_logprobs * _FLOAT64_BYTES
    return total


def _write_payload(
    records: Sequence[tuple[int, int, SampledSequence]],
    *,
    ipc_dir: Path,
    worker_id: int,
    request_id: str,
) -> dict[str, Any]:
    """Write a worker result atomically and return its small pipe descriptor."""

    ipc_dir.mkdir(parents=True, exist_ok=True)
    size = _payload_size(records)
    partial = ipc_dir / f"worker-{worker_id}-{request_id}.partial"
    ready = ipc_dir / f"worker-{worker_id}-{request_id}.ready"
    if partial.exists() or ready.exists():
        raise RolloutWorkerError(f"IPC payload path collision for request {request_id}")

    with partial.open("w+b") as handle:
        handle.truncate(size)
        mapped = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_WRITE)
        try:
            offset = 0
            _PAYLOAD_HEADER.pack_into(mapped, offset, _PAYLOAD_MAGIC, len(records))
            offset += _PAYLOAD_HEADER.size
            for prompt_index, sample_index, sequence in records:
                n_tokens = len(sequence.tokens)
                n_logprobs = -1 if sequence.logprobs is None else len(sequence.logprobs)
                _PAYLOAD_RECORD.pack_into(
                    mapped,
                    offset,
                    prompt_index,
                    sample_index,
                    n_tokens,
                    n_logprobs,
                    _FINISH_REASONS.index(sequence.finish_reason) if sequence.finish_reason in _FINISH_REASONS else 0,
                )
                offset += _PAYLOAD_RECORD.size

                token_array = array.array("q", sequence.tokens)
                if token_array.itemsize != _INT64_BYTES:
                    raise RolloutWorkerError("platform signed-long-long is not 64 bits")
                if sys.byteorder != "little":
                    token_array.byteswap()
                mapped[offset : offset + n_tokens * _INT64_BYTES] = token_array
                offset += n_tokens * _INT64_BYTES

                if sequence.logprobs is not None:
                    logprob_array = array.array("d", sequence.logprobs)
                    if logprob_array.itemsize != _FLOAT64_BYTES:
                        raise RolloutWorkerError("platform double is not 64 bits")
                    if sys.byteorder != "little":
                        logprob_array.byteswap()
                    mapped[offset : offset + n_logprobs * _FLOAT64_BYTES] = logprob_array
                    offset += n_logprobs * _FLOAT64_BYTES
            if offset != size:
                raise RolloutWorkerError(f"IPC payload size mismatch: wrote {offset}, expected {size}")
            mapped.flush()
        finally:
            mapped.close()
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, ready)
    return {"path": str(ready), "size": size, "record_count": len(records)}


def _validated_payload_path(descriptor: dict[str, Any], ipc_dir: Path) -> Path:
    path = Path(descriptor["path"])
    resolved_parent = path.resolve().parent
    expected_parent = ipc_dir.resolve()
    if resolved_parent != expected_parent or path.suffix != ".ready":
        raise RolloutWorkerError(f"worker returned an invalid IPC payload path: {path}")
    return path


def _read_payload(
    descriptor: dict[str, Any],
    *,
    ipc_dir: Path,
) -> list[tuple[int, int, SampledSequence]]:
    """Decode a payload without changing token IDs or Python float values."""

    path = _validated_payload_path(descriptor, ipc_dir)
    expected_size = int(descriptor["size"])
    if path.stat().st_size != expected_size:
        raise RolloutWorkerError(f"IPC payload {path} is {path.stat().st_size} bytes, expected {expected_size}")
    records: list[tuple[int, int, SampledSequence]] = []
    with path.open("rb") as handle:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(mapped) < _PAYLOAD_HEADER.size:
                raise RolloutWorkerError(f"truncated IPC payload: {path}")
            magic, record_count = _PAYLOAD_HEADER.unpack_from(mapped, 0)
            if magic != _PAYLOAD_MAGIC:
                raise RolloutWorkerError(f"invalid IPC payload magic in {path}")
            if record_count != int(descriptor["record_count"]):
                raise RolloutWorkerError("IPC record-count descriptor mismatch")
            offset = _PAYLOAD_HEADER.size
            for _ in range(record_count):
                if offset + _PAYLOAD_RECORD.size > len(mapped):
                    raise RolloutWorkerError(f"truncated IPC record header in {path}")
                prompt_index, sample_index, n_tokens, n_logprobs, finish_code = _PAYLOAD_RECORD.unpack_from(mapped, offset)
                if finish_code not in range(len(_FINISH_REASONS)):
                    raise RolloutWorkerError("invalid IPC completion status")
                offset += _PAYLOAD_RECORD.size
                if n_tokens < 0 or n_logprobs not in {-1, n_tokens}:
                    raise RolloutWorkerError(f"invalid IPC lengths tokens={n_tokens}, logprobs={n_logprobs}")
                token_end = offset + n_tokens * _INT64_BYTES
                logprob_end = token_end + (0 if n_logprobs < 0 else n_logprobs * _FLOAT64_BYTES)
                if logprob_end > len(mapped):
                    raise RolloutWorkerError(f"truncated IPC sequence body in {path}")

                token_view = memoryview(mapped)[offset:token_end]
                try:
                    tokens = token_view.cast("q").tolist()
                finally:
                    token_view.release()
                if sys.byteorder != "little":
                    token_values = array.array("q", tokens)
                    token_values.byteswap()
                    tokens = token_values.tolist()
                offset = token_end

                logprobs: Optional[list[float]]
                if n_logprobs < 0:
                    logprobs = None
                else:
                    logprob_view = memoryview(mapped)[offset:logprob_end]
                    try:
                        logprobs = logprob_view.cast("d").tolist()
                    finally:
                        logprob_view.release()
                    if sys.byteorder != "little":
                        logprob_values = array.array("d", logprobs)
                        logprob_values.byteswap()
                        logprobs = logprob_values.tolist()
                    offset = logprob_end
                records.append(
                    (int(prompt_index), int(sample_index), SampledSequence(tokens=tokens, logprobs=logprobs,
                                                                          finish_reason=_FINISH_REASONS[finish_code]))
                )
            if offset != len(mapped):
                raise RolloutWorkerError(f"IPC payload has {len(mapped) - offset} unexpected trailing byte(s)")
        finally:
            mapped.close()
    return records


def _remove_payload(descriptor: Optional[dict[str, Any]], ipc_dir: Path) -> None:
    if not descriptor:
        return
    try:
        path = _validated_payload_path(descriptor, ipc_dir)
    except Exception:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


@dataclass(frozen=True)
class _WorkerProcessConfig:
    worker_id: int
    gpu: RolloutGPU
    model: str
    enable_lora: bool
    engine_kwargs: dict[str, Any]
    ipc_dir: str
    log_path: str


def _worker_main(connection: Connection, config: _WorkerProcessConfig) -> None:
    """Spawn target.  CUDA visibility is narrowed before vLLM initializes CUDA."""

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu.device_token
    log_path = Path(config.log_path)
    ipc_dir = Path(config.ipc_dir)
    sampler = None
    loaded_version = 0
    _append_jsonl(
        log_path,
        "boot",
        worker_id=config.worker_id,
        pid=os.getpid(),
        logical_gpu=config.gpu.logical_index,
        device_token=config.gpu.device_token,
        worker_cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"],
        model=config.model,
        enable_lora=config.enable_lora,
        enable_sleep_mode=bool(config.engine_kwargs.get("enable_sleep_mode", False)),
        engine_seed=config.engine_kwargs.get("seed"),
        engine_seed_policy=_ENGINE_SEED_POLICY,
    )
    try:
        # Lazy import matters: this is the first code in the worker that can
        # initialize CUDA, after visibility has been reduced to one GPU.
        from ctm.backends.local.vllm_sampler import VLLMSampler

        sampler = VLLMSampler(model=config.model, enable_lora=config.enable_lora, **config.engine_kwargs)
        connection.send(
            {
                "ok": True,
                "op": "ready",
                "worker_id": config.worker_id,
                "pid": os.getpid(),
                "logical_gpu": config.gpu.logical_index,
                "device_token": config.gpu.device_token,
                "adapter_version": loaded_version,
                "sleep_enabled": _sampler_sleep_enabled(sampler),
                "sleeping": _sampler_sleeping(sampler),
                "engine_seed": config.engine_kwargs.get("seed"),
            }
        )
        _append_jsonl(
            log_path,
            "ready",
            worker_id=config.worker_id,
            adapter_version=loaded_version,
            sleep_enabled=_sampler_sleep_enabled(sampler),
            sleeping=_sampler_sleeping(sampler),
            engine_seed=config.engine_kwargs.get("seed"),
        )

        while True:
            try:
                command = connection.recv()
            except EOFError:
                break
            op = command.get("op")
            request_id = command.get("request_id")
            try:
                if op == "health":
                    connection.send(
                        {
                            "ok": True,
                            "op": op,
                            "request_id": request_id,
                            "worker_id": config.worker_id,
                            "pid": os.getpid(),
                            "adapter_version": loaded_version,
                            "sleep_enabled": _sampler_sleep_enabled(sampler),
                            "sleeping": _sampler_sleeping(sampler),
                        }
                    )
                    continue
                if op == "sleep":
                    started_at = time.monotonic()
                    sampler.sleep(level=int(command.get("level", 1)))
                    _append_jsonl(
                        log_path,
                        "sleep_completed",
                        worker_id=config.worker_id,
                        request_id=request_id,
                        adapter_version=loaded_version,
                        sleep_enabled=_sampler_sleep_enabled(sampler),
                        sleeping=_sampler_sleeping(sampler),
                        elapsed_seconds=max(0.0, time.monotonic() - started_at),
                    )
                    connection.send(
                        {
                            "ok": True,
                            "op": op,
                            "request_id": request_id,
                            "worker_id": config.worker_id,
                            "adapter_version": loaded_version,
                            "sleep_enabled": _sampler_sleep_enabled(sampler),
                            "sleeping": _sampler_sleeping(sampler),
                        }
                    )
                    continue
                if op == "wake":
                    started_at = time.monotonic()
                    sampler.wake_up()
                    _append_jsonl(
                        log_path,
                        "wake_completed",
                        worker_id=config.worker_id,
                        request_id=request_id,
                        adapter_version=loaded_version,
                        sleep_enabled=_sampler_sleep_enabled(sampler),
                        sleeping=_sampler_sleeping(sampler),
                        elapsed_seconds=max(0.0, time.monotonic() - started_at),
                    )
                    connection.send(
                        {
                            "ok": True,
                            "op": op,
                            "request_id": request_id,
                            "worker_id": config.worker_id,
                            "adapter_version": loaded_version,
                            "sleep_enabled": _sampler_sleep_enabled(sampler),
                            "sleeping": _sampler_sleeping(sampler),
                        }
                    )
                    continue
                if op == "load_adapter":
                    if not config.enable_lora:
                        raise RolloutWorkerError("base-only worker does not accept adapter snapshots")
                    version = int(command["adapter_version"])
                    adapter_path = Path(command["adapter_path"])
                    if not adapter_path.is_dir():
                        raise RolloutWorkerError(f"adapter snapshot does not exist: {adapter_path}")
                    sampler.advance_policy(str(adapter_path), version=version)
                    loaded_version = version
                    _append_jsonl(
                        log_path,
                        "adapter_loaded",
                        worker_id=config.worker_id,
                        adapter_version=loaded_version,
                        adapter_path=str(adapter_path),
                        sleep_enabled=_sampler_sleep_enabled(sampler),
                        sleeping=_sampler_sleeping(sampler),
                    )
                    connection.send(
                        {
                            "ok": True,
                            "op": op,
                            "request_id": request_id,
                            "worker_id": config.worker_id,
                            "adapter_version": loaded_version,
                            "sleep_enabled": _sampler_sleep_enabled(sampler),
                            "sleeping": _sampler_sleeping(sampler),
                        }
                    )
                    continue
                if op == "shutdown":
                    _append_jsonl(log_path, "shutdown_requested", worker_id=config.worker_id)
                    connection.send(
                        {
                            "ok": True,
                            "op": op,
                            "request_id": request_id,
                            "worker_id": config.worker_id,
                            "adapter_version": loaded_version,
                            "sleep_enabled": _sampler_sleep_enabled(sampler),
                            "sleeping": _sampler_sleeping(sampler),
                        }
                    )
                    break
                if op not in {"sample_batch", "score_batch", "score_batch_uncapped_eos_tail"}:
                    raise RolloutWorkerError(f"unknown worker operation: {op!r}")

                use_base = bool(command["use_base"])
                required_version = int(command["adapter_version"])
                if not config.enable_lora and not use_base:
                    raise RolloutWorkerError("base-only worker cannot sample a policy adapter")
                if not use_base and loaded_version != required_version:
                    raise RolloutWorkerError(
                        f"policy request requires adapter v{required_version}, worker has v{loaded_version}"
                    )
                assignments = command["assignments"]
                if op in {"score_batch", "score_batch_uncapped_eos_tail"}:
                    score_started_at = time.monotonic()
                    _append_jsonl(
                        log_path,
                        "score_started",
                        worker_id=config.worker_id,
                        request_id=request_id,
                        use_base=use_base,
                        requested_adapter_version=required_version,
                        loaded_adapter_version=loaded_version,
                        score_indices=[assignment["score_index"] for assignment in assignments],
                    )
                    score_method = (
                        sampler.score_completions_uncapped_eos_tail
                        if op == "score_batch_uncapped_eos_tail"
                        else sampler.score_completions
                    )
                    scored = score_method(
                        [assignment["prompt_tokens"] for assignment in assignments],
                        [assignment["completion_tokens"] for assignment in assignments],
                        use_base=use_base,
                    )
                    if len(scored) != len(assignments):
                        raise RolloutWorkerError(
                            f"worker got {len(scored)} score result(s) for {len(assignments)} assignment(s)"
                        )
                    records = []
                    for assignment, logprobs in zip(assignments, scored):
                        completion = list(assignment["completion_tokens"])
                        values = list(logprobs)
                        if len(values) != len(completion):
                            raise RolloutWorkerError(
                                f"worker score {assignment['score_index']} has {len(completion)} token(s) "
                                f"but {len(values)} logprob(s)"
                            )
                        if any(not math.isfinite(float(value)) for value in values):
                            raise RolloutWorkerError(
                                f"worker score {assignment['score_index']} contains a non-finite logprob"
                            )
                        records.append(
                            (
                                int(assignment["score_index"]),
                                0,
                                SampledSequence(tokens=completion, logprobs=[float(value) for value in values]),
                            )
                        )
                    descriptor = _write_payload(
                        records,
                        ipc_dir=ipc_dir,
                        worker_id=config.worker_id,
                        request_id=request_id,
                    )
                    connection.send(
                        {
                            "ok": True,
                            "op": op,
                            "request_id": request_id,
                            "worker_id": config.worker_id,
                            "adapter_version": loaded_version,
                            "sleep_enabled": _sampler_sleep_enabled(sampler),
                            "sleeping": _sampler_sleeping(sampler),
                            "result": descriptor,
                        }
                    )
                    completion_token_count = sum(len(sequence.tokens) for _, _, sequence in records)
                    elapsed_seconds = max(0.0, time.monotonic() - score_started_at)
                    _append_jsonl(
                        log_path,
                        "score_completed",
                        worker_id=config.worker_id,
                        request_id=request_id,
                        use_base=use_base,
                        requested_adapter_version=required_version,
                        record_count=len(records),
                        completion_tokens=completion_token_count,
                        tail_max_tokens=None if op == "score_batch_uncapped_eos_tail" else 1,
                        tail_termination="eos_only" if op == "score_batch_uncapped_eos_tail" else "not_attested",
                        elapsed_seconds=elapsed_seconds,
                        scored_tokens_per_second=(
                            completion_token_count / elapsed_seconds if elapsed_seconds > 0 else 0.0
                        ),
                        payload_bytes=descriptor["size"],
                    )
                    continue

                sample_started_at = time.monotonic()
                _append_jsonl(
                    log_path,
                    "sample_started",
                    worker_id=config.worker_id,
                    request_id=request_id,
                    use_base=use_base,
                    requested_adapter_version=required_version,
                    loaded_adapter_version=loaded_version,
                    assignments=[
                        {
                            "prompt_index": assignment["prompt_index"],
                            "sample_indices": assignment["sample_indices"],
                        }
                        for assignment in assignments
                    ],
                )

                records: list[tuple[int, int, SampledSequence]] = []
                by_count: dict[int, list[dict[str, Any]]] = {}
                for assignment in assignments:
                    count = len(assignment["sample_indices"])
                    if count:
                        by_count.setdefault(count, []).append(assignment)
                for count in sorted(by_count):
                    group = by_count[count]
                    sampled = sampler.sample_batch(
                        [assignment["prompt_tokens"] for assignment in group],
                        max_tokens=(
                            None
                            if command.get("max_tokens") is None
                            else int(command["max_tokens"])
                        ),
                        temperature=float(command["temperature"]),
                        stop=command["stop"],
                        num_samples=count,
                        use_base=use_base,
                        ignore_eos=bool(command.get("ignore_eos", False)),
                    )
                    if len(sampled) != len(group):
                        raise RolloutWorkerError(
                            f"worker got {len(sampled)} prompt result(s) for {len(group)} assignment(s)"
                        )
                    for assignment, sequences in zip(group, sampled):
                        indices = assignment["sample_indices"]
                        if len(sequences) != len(indices):
                            raise RolloutWorkerError(
                                f"worker got {len(sequences)} sequence(s) for {len(indices)} sample slot(s)"
                            )
                        records.extend(
                            (int(assignment["prompt_index"]), int(sample_index), sequence)
                            for sample_index, sequence in zip(indices, sequences)
                        )
                descriptor = _write_payload(
                    records,
                    ipc_dir=ipc_dir,
                    worker_id=config.worker_id,
                    request_id=request_id,
                )
                connection.send(
                    {
                        "ok": True,
                        "op": op,
                        "request_id": request_id,
                        "worker_id": config.worker_id,
                        "adapter_version": loaded_version,
                        "sleep_enabled": _sampler_sleep_enabled(sampler),
                        "sleeping": _sampler_sleeping(sampler),
                        "result": descriptor,
                    }
                )
                completion_tokens = sum(len(sequence.tokens) for _, _, sequence in records)
                elapsed_seconds = max(0.0, time.monotonic() - sample_started_at)
                _append_jsonl(
                    log_path,
                    "sample_completed",
                    worker_id=config.worker_id,
                    request_id=request_id,
                    use_base=use_base,
                    requested_adapter_version=required_version,
                    record_count=len(records),
                    completion_tokens=completion_tokens,
                    elapsed_seconds=elapsed_seconds,
                    output_tokens_per_second=completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                    payload_bytes=descriptor["size"],
                )
            except BaseException as exc:  # noqa: BLE001 -- errors must cross the worker boundary
                error = {
                    "ok": False,
                    "op": op,
                    "request_id": request_id,
                    "worker_id": config.worker_id,
                    "adapter_version": loaded_version,
                    "sleep_enabled": None if sampler is None else _sampler_sleep_enabled(sampler),
                    "sleeping": None if sampler is None else _sampler_sleeping(sampler),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                _append_jsonl(log_path, "command_failed", **error)
                connection.send(error)
    except BaseException as exc:  # noqa: BLE001 -- startup failure must reach coordinator
        error = {
            "ok": False,
            "op": "startup",
            "worker_id": config.worker_id,
            "adapter_version": loaded_version,
            "sleep_enabled": None if sampler is None else _sampler_sleep_enabled(sampler),
            "sleeping": None if sampler is None else _sampler_sleeping(sampler),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _append_jsonl(log_path, "startup_failed", **error)
        try:
            connection.send(error)
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if sampler is not None:
            try:
                sampler.shutdown()
            except Exception:
                _append_jsonl(
                    log_path,
                    "sampler_shutdown_failed",
                    worker_id=config.worker_id,
                    traceback=traceback.format_exc(),
                )
        _append_jsonl(
            log_path,
            "stopped",
            worker_id=config.worker_id,
            adapter_version=loaded_version,
            sleep_enabled=None if sampler is None else _sampler_sleep_enabled(sampler),
            sleeping=None if sampler is None else _sampler_sleeping(sampler),
        )
        connection.close()


class _ProcessEndpoint:
    """One synchronous pipe guarded by a lock and exposed asynchronously."""

    def __init__(
        self,
        config: _WorkerProcessConfig,
        *,
        start_timeout_seconds: float,
        request_timeout_seconds: float,
        shutdown_timeout_seconds: float,
        context: multiprocessing.context.BaseContext,
        ipc_dir: Path,
        worker_target: Any = _worker_main,
    ):
        self.worker_id = config.worker_id
        self.gpu = config.gpu
        self._config = config
        self._start_timeout = start_timeout_seconds
        self._request_timeout = request_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._context = context
        self._ipc_dir = ipc_dir
        self._worker_target = worker_target
        self._connection: Optional[Connection] = None
        self._process: Optional[multiprocessing.Process] = None
        self._lock = threading.Lock()
        self._closed = False

    def launch_sync(self) -> None:
        if self._process is not None:
            raise RolloutWorkerError(f"worker {self.worker_id} was already launched")
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._worker_target,
            args=(child, self._config),
            name=f"ctm-rollout-{self.worker_id}-gpu{self.gpu.logical_index}",
            daemon=False,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process

    def wait_ready_sync(self) -> dict[str, Any]:
        response = self._recv_sync(self._start_timeout, phase="startup")
        self._check_response(response, expected_op="ready", request_id=None)
        return response

    def _recv_sync(self, timeout: float, *, phase: str) -> dict[str, Any]:
        connection = self._connection
        process = self._process
        if connection is None or process is None:
            raise RolloutWorkerError(f"worker {self.worker_id} has not been launched")
        if not connection.poll(timeout):
            state = f"exitcode={process.exitcode}" if not process.is_alive() else "still alive"
            raise RolloutWorkerError(f"worker {self.worker_id} timed out during {phase} after {timeout:g}s ({state})")
        try:
            response = connection.recv()
        except EOFError as exc:
            raise RolloutWorkerError(
                f"worker {self.worker_id} closed its pipe during {phase} (exitcode={process.exitcode})"
            ) from exc
        if not isinstance(response, dict):
            raise RolloutWorkerError(f"worker {self.worker_id} returned a non-dict response")
        return response

    def _check_response(
        self,
        response: dict[str, Any],
        *,
        expected_op: str,
        request_id: Optional[str],
    ) -> None:
        if not response.get("ok"):
            detail = response.get("traceback") or response.get("error") or repr(response)
            raise RolloutWorkerError(f"worker {self.worker_id} {expected_op} failed:\n{detail}")
        if response.get("op") != expected_op:
            raise RolloutWorkerError(
                f"worker {self.worker_id} response op={response.get('op')!r}, expected {expected_op!r}"
            )
        if request_id is not None and response.get("request_id") != request_id:
            raise RolloutWorkerError(
                f"worker {self.worker_id} response request_id={response.get('request_id')!r}, "
                f"expected {request_id!r}"
            )

    def call_sync(self, command: dict[str, Any], *, timeout: Optional[float] = None) -> dict[str, Any]:
        request_id = command.get("request_id")
        op = command["op"]
        with self._lock:
            if self._closed:
                raise RolloutWorkerError(f"worker {self.worker_id} endpoint is closed")
            process = self._process
            connection = self._connection
            if process is None or connection is None or not process.is_alive():
                exitcode = None if process is None else process.exitcode
                raise RolloutWorkerError(f"worker {self.worker_id} is not alive (exitcode={exitcode})")
            connection.send(command)
            response = self._recv_sync(timeout or self._request_timeout, phase=op)
            self._check_response(response, expected_op=op, request_id=request_id)
            return response

    async def call(self, command: dict[str, Any]) -> dict[str, Any]:
        # Shield the executor future so cancellation (used to flush stale RL
        # prefetch) still drains the one pipe response before another command can
        # acquire this endpoint.  A canceled sample payload is explicitly removed.
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, self.call_sync, command)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                response = await asyncio.shield(future)
                _remove_payload(response.get("result"), self._ipc_dir)
            except Exception:
                pass
            raise

    def close_sync(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        connection = self._connection
        if process is not None and process.is_alive() and connection is not None:
            try:
                # Bypass call_sync because the endpoint has just been marked
                # closed, while retaining the same single-command lock.
                with self._lock:
                    request_id = uuid.uuid4().hex
                    connection.send({"op": "shutdown", "request_id": request_id})
                    response = self._recv_sync(self._shutdown_timeout, phase="shutdown")
                    self._check_response(response, expected_op="shutdown", request_id=request_id)
            except Exception:
                pass
        if connection is not None:
            connection.close()
        if process is not None:
            process.join(self._shutdown_timeout)
            if process.is_alive():
                process.terminate()
                process.join(self._shutdown_timeout)


class RolloutWorkerPool:
    """Partition and merge rollout slots across independent worker engines.

    With ``enable_sleep_mode=True`` in ``engine_kwargs``, all workers can be
    put to sleep as one serialized phase barrier.  The pool then wakes them as
    one barrier before sampling, scoring, or publishing an adapter.  Nothing
    in this lifecycle assumes a particular worker count: one, two, four, or
    eight independent rollout workers follow the same protocol.
    """

    def __init__(
        self,
        *,
        model: str,
        gpus: Sequence[RolloutGPU],
        engine_kwargs: Optional[dict[str, Any]],
        status_dir: str | Path,
        enable_lora: bool = True,
        start_timeout_seconds: float = 1800.0,
        request_timeout_seconds: float = 7200.0,
        shutdown_timeout_seconds: float = 30.0,
        endpoints: Optional[Sequence[Any]] = None,
        worker_entrypoint_for_tests: Optional[Any] = None,
    ):
        if not gpus:
            raise ValueError("RolloutWorkerPool requires at least one rollout GPU")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (start_timeout_seconds, request_timeout_seconds, shutdown_timeout_seconds)
        ):
            raise ValueError("worker timeouts must be finite and positive")
        if len({gpu.logical_index for gpu in gpus}) != len(gpus):
            raise ValueError("rollout GPU logical indices must be unique")
        if len({gpu.device_token for gpu in gpus}) != len(gpus):
            raise ValueError("rollout GPU device tokens must be unique")

        kwargs = dict(engine_kwargs or {})
        requested_sleep_mode = kwargs.get("enable_sleep_mode", False)
        if not isinstance(requested_sleep_mode, bool):
            raise ValueError("rollout worker enable_sleep_mode must be a boolean")
        requested_logprobs_mode = kwargs.get("logprobs_mode", "processed_logprobs")
        if requested_logprobs_mode != "processed_logprobs":
            raise ValueError(
                "rollout workers require logprobs_mode='processed_logprobs'; "
                f"got {requested_logprobs_mode!r}"
            )
        kwargs["logprobs_mode"] = "processed_logprobs"
        tensor_parallel_size = kwargs.pop("tensor_parallel_size", 1)
        if tensor_parallel_size != 1:
            raise ValueError("rollout data parallelism requires tensor_parallel_size=1 per worker")
        if "engine" in kwargs or "api" in kwargs:
            raise ValueError("prebuilt vLLM engine/api hooks cannot cross rollout worker process boundaries")
        kwargs["tensor_parallel_size"] = 1

        # vLLM defaults every independent LLM engine to seed=0.  A data-parallel
        # pool must not let identical engines consume identical RNG streams.
        # Keep the base in the attested common option map, then specialize the
        # actual child configuration by worker id below.  If no seed is supplied
        # (the backwards-compatible general API), draw it once per pool rather
        # than independently in each process.
        if "seed" in kwargs:
            engine_seed_base = _validated_engine_seed_base(kwargs["seed"], worker_count=len(gpus))
        else:
            engine_seed_base = secrets.randbelow(_MAX_ENGINE_SEED - len(gpus) + 2)
        kwargs["seed"] = engine_seed_base

        self.model = model
        self.gpus = tuple(gpus)
        self.enable_lora = enable_lora
        self.engine_kwargs = kwargs
        self.sleep_enabled = requested_sleep_mode
        self.engine_seed_base = engine_seed_base
        self.engine_seed_policy = _ENGINE_SEED_POLICY
        self.worker_engine_seeds = tuple(engine_seed_base + worker_id for worker_id in range(len(self.gpus)))
        base_status_dir = Path(status_dir)
        session_id = (
            f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.status_dir = base_status_dir / session_id
        self.ipc_dir = self.status_dir / "ipc"
        self.log_path = self.status_dir / "coordinator.jsonl"
        self.status_dir.mkdir(parents=True, exist_ok=False)
        self.ipc_dir.mkdir(parents=True, exist_ok=True)
        self._operation_lock = asyncio.Lock()
        self._started = False
        self._closed = False
        self._fatal_error: Optional[str] = None
        self._adapter_version = 0
        self._sleeping = False
        # This is a logical phase barrier, not merely an observation of vLLM
        # memory.  Once training begins, a concurrent sampling task must never
        # wake a worker back onto a GPU that the coordinator is using for
        # forward/backward.  The generic RL loop can use
        # ``sampling_training_overlap_supported`` to avoid scheduling that
        # overlap; the pool itself fails closed if one slips through.
        self._training_phase_active = False
        self._request_counter = 0

        if endpoints is not None:
            if len(endpoints) != len(self.gpus):
                raise ValueError("fake endpoint count must match rollout GPU count")
            self._endpoints = list(endpoints)
        else:
            context = multiprocessing.get_context("spawn")
            self._endpoints = []
            for worker_id, gpu in enumerate(self.gpus):
                worker_engine_kwargs = dict(self.engine_kwargs)
                worker_engine_kwargs["seed"] = self.worker_engine_seeds[worker_id]
                config = _WorkerProcessConfig(
                    worker_id=worker_id,
                    gpu=gpu,
                    model=model,
                    enable_lora=enable_lora,
                    engine_kwargs=worker_engine_kwargs,
                    ipc_dir=str(self.ipc_dir),
                    log_path=str(self.status_dir / f"worker-{worker_id}.jsonl"),
                )
                self._endpoints.append(
                    _ProcessEndpoint(
                        config,
                        start_timeout_seconds=start_timeout_seconds,
                        request_timeout_seconds=request_timeout_seconds,
                        shutdown_timeout_seconds=shutdown_timeout_seconds,
                        context=context,
                        ipc_dir=self.ipc_dir,
                        worker_target=worker_entrypoint_for_tests or _worker_main,
                    )
                )

        _append_jsonl(
            self.log_path,
            "pool_configured",
            model=model,
            workers=[
                {**gpu.as_dict(), "engine_seed": self.worker_engine_seeds[worker_id]}
                for worker_id, gpu in enumerate(self.gpus)
            ],
            enable_lora=enable_lora,
            sleep_enabled=self.sleep_enabled,
            engine_kwargs=self.engine_kwargs,
            engine_seed_base=self.engine_seed_base,
            engine_seed_policy=self.engine_seed_policy,
            worker_engine_seeds=list(self.worker_engine_seeds),
            result_transport="bounded_file_backed_binary_v1",
        )

    @property
    def adapter_version(self) -> int:
        return self._adapter_version

    @property
    def worker_count(self) -> int:
        return len(self._endpoints)

    @property
    def sleeping(self) -> bool:
        """Whether the pool has completed an enabled sleep barrier."""

        return self._sleeping

    @property
    def phase(self) -> str:
        """Current resource phase: ``rollout`` or ``training``."""

        return "training" if self._training_phase_active else "rollout"

    @property
    def sampling_training_overlap_supported(self) -> bool:
        """Whether the caller may overlap worker sampling and coordinator work.

        A resident, separate-GPU rollout pool retains the historical overlap.
        Once sleep mode is enabled, the same GPUs may be reclaimed by a
        trainer, so overlap is intentionally prohibited.
        """

        return not self.sleep_enabled

    def _ensure_usable(self) -> None:
        if not self._started:
            raise RolloutWorkerError("rollout worker pool has not been started")
        if self._closed:
            raise RolloutWorkerError("rollout worker pool has been shut down")
        if self._fatal_error is not None:
            raise RolloutWorkerError(f"rollout worker pool is unhealthy: {self._fatal_error}")

    def _mark_fatal(self, exc: BaseException, *, event: str) -> None:
        self._fatal_error = f"{type(exc).__name__}: {exc}"
        _append_jsonl(self.log_path, event, error=self._fatal_error, traceback=traceback.format_exc())

    def start(self) -> None:
        if self._started:
            return
        if self._closed:
            raise RolloutWorkerError("cannot restart a shut-down rollout worker pool")
        launched: list[Any] = []
        try:
            for endpoint in self._endpoints:
                endpoint.launch_sync()
                launched.append(endpoint)
            ready = [endpoint.wait_ready_sync() for endpoint in self._endpoints]
            health = []
            for endpoint in self._endpoints:
                request_id = self._next_request_id("health")
                health.append(endpoint.call_sync({"op": "health", "request_id": request_id}))
            self._started = True
            _append_jsonl(self.log_path, "pool_started", ready=ready, health=health)
        except BaseException as exc:
            self._mark_fatal(exc, event="pool_start_failed")
            for endpoint in reversed(launched):
                endpoint.close_sync()
            self._closed = True
            raise

    def _next_request_id(self, prefix: str) -> str:
        self._request_counter += 1
        return f"{prefix}-{self._request_counter:08d}-{uuid.uuid4().hex[:8]}"

    def _validate_lifecycle_acks(
        self,
        responses: Sequence[dict[str, Any]],
        *,
        expected_sleeping: bool,
        operation: str,
    ) -> None:
        actual = [response.get("sleeping") for response in responses]
        expected = [expected_sleeping] * self.worker_count
        if actual != expected:
            raise RolloutWorkerError(
                f"worker {operation} acknowledgement mismatch: sleeping={actual}, expected={expected}"
            )
        if self.sleep_enabled:
            enabled = [response.get("sleep_enabled") for response in responses]
            if enabled != [True] * self.worker_count:
                raise RolloutWorkerError(
                    f"worker {operation} acknowledgement mismatch: sleep_enabled={enabled}, expected all True"
                )

    async def _set_sleeping_locked(self, sleeping: bool, *, reason: str) -> None:
        """Transition all worker engines while ``_operation_lock`` is held."""

        self._ensure_usable()
        if not self.sleep_enabled:
            # Keep the default separate-worker topology exactly as it was:
            # there is no memory transition and sampling/training may overlap.
            self._sleeping = False
            return
        if self._sleeping == sleeping:
            return

        op = "sleep" if sleeping else "wake"
        request_id = self._next_request_id(op)
        command: dict[str, Any] = {"op": op, "request_id": request_id}
        if sleeping:
            # Level 1 releases GPU allocations and KV cache but retains weights
            # in host memory.  It is the only level used by the production
            # phase-sharing lifecycle; level 2 is deliberately not exposed as
            # an accidental per-update model reload.
            command["level"] = 1
        _append_jsonl(
            self.log_path,
            f"worker_{op}_started",
            request_id=request_id,
            reason=reason,
            adapter_version=self._adapter_version,
            phase=self.phase,
            worker_count=self.worker_count,
        )
        try:
            responses = await self._call_all([command] * self.worker_count)
            self._validate_lifecycle_acks(responses, expected_sleeping=sleeping, operation=op)
        except BaseException as exc:
            # Partial transitions are operationally ambiguous: keeping the
            # pool alive could allow sampling on one GPU while another holds
            # training memory. Fail closed rather than guessing which phase is
            # safe to resume.
            self._mark_fatal(exc, event=f"worker_{op}_failed")
            raise
        self._sleeping = sleeping
        _append_jsonl(
            self.log_path,
            f"worker_{op}_completed",
            request_id=request_id,
            reason=reason,
            adapter_version=self._adapter_version,
            phase=self.phase,
            acknowledgements=responses,
        )

    async def enter_training_phase(self) -> None:
        """Drain worker commands and release their GPU allocations for training.

        This is idempotent.  It is only active for an opt-in sleep-enabled
        pool; otherwise it preserves the previous overlap-friendly behavior.
        """

        async with self._operation_lock:
            self._ensure_usable()
            if not self.sleep_enabled or self._training_phase_active:
                return
            await self._set_sleeping_locked(True, reason="enter_training_phase")
            self._training_phase_active = True
            _append_jsonl(
                self.log_path,
                "training_phase_entered",
                adapter_version=self._adapter_version,
                worker_count=self.worker_count,
            )

    async def enter_rollout_phase(self) -> None:
        """Restore worker engines and permit rollout/scoring commands again."""

        async with self._operation_lock:
            self._ensure_usable()
            if not self.sleep_enabled:
                return
            await self._set_sleeping_locked(False, reason="enter_rollout_phase")
            if self._training_phase_active:
                self._training_phase_active = False
                _append_jsonl(
                    self.log_path,
                    "rollout_phase_entered",
                    adapter_version=self._adapter_version,
                    worker_count=self.worker_count,
                )

    async def sleep(self) -> None:
        """Compatibility alias for :meth:`enter_training_phase`."""

        await self.enter_training_phase()

    async def wake(self) -> None:
        """Compatibility alias for :meth:`enter_rollout_phase`."""

        await self.enter_rollout_phase()

    async def wake_up(self) -> None:
        """vLLM-named compatibility alias for :meth:`enter_rollout_phase`."""

        await self.enter_rollout_phase()

    async def health(self) -> list[dict[str, Any]]:
        """Return a serialized all-worker health snapshot including phase state."""

        async with self._operation_lock:
            self._ensure_usable()
            request_id = self._next_request_id("health")
            try:
                responses = await self._call_all(
                    [{"op": "health", "request_id": request_id}] * self.worker_count
                )
                if self.sleep_enabled:
                    self._validate_lifecycle_acks(
                        responses,
                        expected_sleeping=self._sleeping,
                        operation="health",
                    )
            except BaseException as exc:
                self._mark_fatal(exc, event="pool_health_failed")
                raise
            _append_jsonl(
                self.log_path,
                "pool_health",
                request_id=request_id,
                phase=self.phase,
                sleeping=self._sleeping,
                responses=responses,
            )
            return responses

    def publish_adapter_sync(self, adapter_path: str | Path, *, version: int) -> None:
        """Initial policy barrier, used by the trainer's synchronous setup."""

        self._ensure_usable()
        if self._training_phase_active:
            raise RolloutWorkerError(
                "cannot synchronously publish an adapter during the training phase; use publish_adapter()"
            )
        if not self.enable_lora:
            raise RolloutWorkerError("base-only rollout pool does not accept adapter snapshots")
        if version <= self._adapter_version:
            raise ValueError(f"adapter version must increase (current={self._adapter_version}, requested={version})")
        request_id = self._next_request_id(f"adapter-v{version}")
        command = {
            "op": "load_adapter",
            "request_id": request_id,
            "adapter_version": version,
            "adapter_path": str(adapter_path),
        }
        _append_jsonl(
            self.log_path,
            "adapter_publish_started",
            request_id=request_id,
            adapter_version=version,
            adapter_path=str(adapter_path),
        )
        try:
            responses = [endpoint.call_sync(command) for endpoint in self._endpoints]
            self._validate_adapter_acks(responses, version)
        except BaseException as exc:
            self._mark_fatal(exc, event="adapter_publish_failed")
            raise
        self._adapter_version = version
        _append_jsonl(
            self.log_path,
            "adapter_publish_completed",
            request_id=request_id,
            adapter_version=version,
            acknowledgements=responses,
        )

    async def publish_adapter(self, adapter_path: str | Path, *, version: int) -> None:
        """Atomically wake, advance, and release every worker for rollout.

        When the pool is in its training phase this is the only transition that
        may both wake workers and make a policy adapter available.  Keeping all
        three actions under one operation lock prevents an old-policy sample
        from entering between wake-up and the adapter-version barrier.
        """

        async with self._operation_lock:
            self._ensure_usable()
            if not self.enable_lora:
                raise RolloutWorkerError("base-only rollout pool does not accept adapter snapshots")
            if version <= self._adapter_version:
                raise ValueError(
                    f"adapter version must increase (current={self._adapter_version}, requested={version})"
                )
            await self._set_sleeping_locked(False, reason="adapter_publish")
            request_id = self._next_request_id(f"adapter-v{version}")
            command = {
                "op": "load_adapter",
                "request_id": request_id,
                "adapter_version": version,
                "adapter_path": str(adapter_path),
            }
            _append_jsonl(
                self.log_path,
                "adapter_publish_started",
                request_id=request_id,
                adapter_version=version,
                adapter_path=str(adapter_path),
            )
            try:
                responses = await self._call_all([command] * self.worker_count)
                self._validate_adapter_acks(responses, version)
            except BaseException as exc:
                # A canceled barrier may have advanced only a subset of workers,
                # or all workers without updating the coordinator version.  Do
                # not permit any later sampling from that ambiguous state.
                self._mark_fatal(exc, event="adapter_publish_failed")
                raise
            self._adapter_version = version
            if self._training_phase_active:
                self._training_phase_active = False
                _append_jsonl(
                    self.log_path,
                    "rollout_phase_entered",
                    adapter_version=self._adapter_version,
                    worker_count=self.worker_count,
                    reason="adapter_publish",
                )
            _append_jsonl(
                self.log_path,
                "adapter_publish_completed",
                request_id=request_id,
                adapter_version=version,
                acknowledgements=responses,
            )

    def _validate_adapter_acks(self, responses: Sequence[dict[str, Any]], version: int) -> None:
        acknowledged = [int(response.get("adapter_version", -1)) for response in responses]
        if acknowledged != [version] * self.worker_count:
            raise RolloutWorkerError(f"adapter v{version} acknowledgement mismatch across workers: {acknowledged}")
        if self.sleep_enabled:
            self._validate_lifecycle_acks(
                responses,
                expected_sleeping=False,
                operation="adapter_publish",
            )

    def _assignments(
        self,
        prompt_tokens_batch: Sequence[Sequence[int]],
        num_samples: int,
    ) -> list[list[dict[str, Any]]]:
        assignments: list[list[dict[str, Any]]] = [[] for _ in self._endpoints]
        per_worker_prompt: list[dict[int, list[int]]] = [dict() for _ in self._endpoints]
        for prompt_index in range(len(prompt_tokens_batch)):
            for sample_index in range(num_samples):
                worker_id = (prompt_index * num_samples + sample_index) % self.worker_count
                per_worker_prompt[worker_id].setdefault(prompt_index, []).append(sample_index)
        for worker_id, by_prompt in enumerate(per_worker_prompt):
            assignments[worker_id] = [
                {
                    "prompt_index": prompt_index,
                    "prompt_tokens": list(prompt_tokens_batch[prompt_index]),
                    "sample_indices": sample_indices,
                }
                for prompt_index, sample_indices in sorted(by_prompt.items())
            ]
        return assignments

    def _score_assignments(
        self,
        prompt_tokens_batch: Sequence[Sequence[int]],
        completion_tokens_batch: Sequence[Sequence[int]],
    ) -> list[list[dict[str, Any]]]:
        """Assign each aligned scoring pair by stable flat index."""

        assignments: list[list[dict[str, Any]]] = [[] for _ in self._endpoints]
        for score_index, (prompt_tokens, completion_tokens) in enumerate(
            zip(prompt_tokens_batch, completion_tokens_batch)
        ):
            worker_id = score_index % self.worker_count
            assignments[worker_id].append(
                {
                    "score_index": score_index,
                    "prompt_tokens": list(prompt_tokens),
                    "completion_tokens": list(completion_tokens),
                }
            )
        return assignments

    async def _call_all(self, commands: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        tasks = [asyncio.create_task(endpoint.call(command)) for endpoint, command in zip(self._endpoints, commands)]
        aggregate = asyncio.gather(*tasks, return_exceptions=True)
        try:
            results = await asyncio.shield(aggregate)
        except asyncio.CancelledError:
            # Shield leaves worker calls alive.  Drain them before releasing the
            # operation lock so a policy refresh can never overtake old-policy
            # generation. Endpoint.call removes any canceled result payload.
            results = await aggregate
            for result in results:
                if isinstance(result, dict):
                    _remove_payload(result.get("result"), self.ipc_dir)
            raise
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            for result in results:
                if isinstance(result, dict):
                    _remove_payload(result.get("result"), self.ipc_dir)
            raise RolloutWorkerError("one or more rollout workers failed: " + "; ".join(str(error) for error in errors))
        return list(results)

    async def sample_batch(
        self,
        prompt_tokens_batch: Sequence[Sequence[int]],
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
        use_base: bool,
        ignore_eos: bool = False,
    ) -> list[list[SampledSequence]]:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not prompt_tokens_batch:
            return []
        if num_samples == 0:
            return [[] for _ in prompt_tokens_batch]

        async with self._operation_lock:
            self._ensure_usable()
            if self._training_phase_active:
                raise RolloutWorkerError(
                    "sampling is prohibited while the rollout worker pool is in the training phase; "
                    "call enter_rollout_phase() after coordinator work completes"
                )
            await self._set_sleeping_locked(False, reason="sample_batch")
            requested_version = self._adapter_version
            if not self.enable_lora and not use_base:
                raise RolloutWorkerError("base-only rollout pool cannot sample a policy adapter")
            if not use_base and requested_version <= 0:
                raise RolloutWorkerError("policy sampling requires an acknowledged adapter snapshot")
            request_id = self._next_request_id("base" if use_base else f"policy-v{requested_version}")
            assignments = self._assignments(prompt_tokens_batch, num_samples)
            commands = [
                {
                    "op": "sample_batch",
                    "request_id": request_id,
                    "assignments": worker_assignments,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "use_base": use_base,
                    "ignore_eos": ignore_eos,
                    "adapter_version": requested_version,
                }
                for worker_assignments in assignments
            ]
            _append_jsonl(
                self.log_path,
                "sample_assigned",
                request_id=request_id,
                use_base=use_base,
                adapter_version=requested_version,
                prompt_count=len(prompt_tokens_batch),
                samples_per_prompt=num_samples,
                ignore_eos=ignore_eos,
                assignments=[
                    {
                        "worker_id": worker_id,
                        "logical_gpu": self.gpus[worker_id].logical_index,
                        "device_token": self.gpus[worker_id].device_token,
                        "prompts": [
                            {
                                "prompt_index": assignment["prompt_index"],
                                "sample_indices": assignment["sample_indices"],
                            }
                            for assignment in worker_assignments
                        ],
                    }
                    for worker_id, worker_assignments in enumerate(assignments)
                ],
            )
            responses: list[dict[str, Any]] = []
            try:
                responses = await self._call_all(commands)
                expected = {
                    (assignment["prompt_index"], sample_index)
                    for worker_assignments in assignments
                    for assignment in worker_assignments
                    for sample_index in assignment["sample_indices"]
                }
                records: dict[tuple[int, int], SampledSequence] = {}
                for response in responses:
                    if int(response.get("adapter_version", -1)) != requested_version:
                        raise RolloutWorkerError(
                            f"worker {response.get('worker_id')} sampled with adapter "
                            f"v{response.get('adapter_version')}, expected v{requested_version}"
                        )
                    descriptor = response.get("result")
                    try:
                        worker_records = _read_payload(descriptor, ipc_dir=self.ipc_dir)
                    finally:
                        _remove_payload(descriptor, self.ipc_dir)
                    for prompt_index, sample_index, sequence in worker_records:
                        key = (prompt_index, sample_index)
                        if key not in expected:
                            raise RolloutWorkerError(f"worker returned unexpected rollout slot {key}")
                        if key in records:
                            raise RolloutWorkerError(f"worker returned duplicate rollout slot {key}")
                        records[key] = sequence
                missing = sorted(expected - set(records))
                if missing:
                    raise RolloutWorkerError(f"workers omitted {len(missing)} rollout slot(s): {missing[:8]}")
                merged = [
                    [records[(prompt_index, sample_index)] for sample_index in range(num_samples)]
                    for prompt_index in range(len(prompt_tokens_batch))
                ]
            except asyncio.CancelledError:
                for response in responses:
                    _remove_payload(response.get("result"), self.ipc_dir)
                _append_jsonl(
                    self.log_path,
                    "sample_cancelled_and_drained",
                    request_id=request_id,
                    use_base=use_base,
                    adapter_version=requested_version,
                )
                raise
            except BaseException as exc:
                for response in responses:
                    _remove_payload(response.get("result"), self.ipc_dir)
                self._mark_fatal(exc, event="sample_failed")
                raise
            _append_jsonl(
                self.log_path,
                "sample_completed",
                request_id=request_id,
                use_base=use_base,
                adapter_version=requested_version,
                prompt_count=len(prompt_tokens_batch),
                sample_count=len(prompt_tokens_batch) * num_samples,
            )
            return merged

    async def score_completions(
        self,
        prompt_tokens_batch: Sequence[Sequence[int]],
        completion_tokens_batch: Sequence[Sequence[int]],
        *,
        use_base: bool,
        _uncapped_eos_tail: bool = False,
    ) -> list[list[float]]:
        """Score aligned prompt/completion pairs across rollout workers."""

        if len(prompt_tokens_batch) != len(completion_tokens_batch):
            raise ValueError(
                "prompt_tokens_batch and completion_tokens_batch must have the same length, "
                f"got {len(prompt_tokens_batch)} and {len(completion_tokens_batch)}"
            )
        if not prompt_tokens_batch:
            return []
        prompts = [list(tokens) for tokens in prompt_tokens_batch]
        completions = [list(tokens) for tokens in completion_tokens_batch]
        for score_index, (prompt, completion) in enumerate(zip(prompts, completions)):
            if not prompt:
                raise ValueError(f"prompt {score_index} is empty")
            if not completion:
                raise ValueError(f"completion {score_index} is empty")

        async with self._operation_lock:
            self._ensure_usable()
            if self._training_phase_active:
                raise RolloutWorkerError(
                    "scoring is prohibited while the rollout worker pool is in the training phase; "
                    "call enter_rollout_phase() after coordinator work completes"
                )
            await self._set_sleeping_locked(False, reason="score_completions")
            requested_version = self._adapter_version
            if not self.enable_lora and not use_base:
                raise RolloutWorkerError("base-only rollout pool cannot score a policy adapter")
            if not use_base and requested_version <= 0:
                raise RolloutWorkerError("policy scoring requires an acknowledged adapter snapshot")
            request_id = self._next_request_id("score-base" if use_base else f"score-policy-v{requested_version}")
            assignments = self._score_assignments(prompts, completions)
            operation = "score_batch_uncapped_eos_tail" if _uncapped_eos_tail else "score_batch"
            commands = [
                {
                    "op": operation,
                    "request_id": request_id,
                    "assignments": worker_assignments,
                    "use_base": use_base,
                    "adapter_version": requested_version,
                }
                for worker_assignments in assignments
            ]
            _append_jsonl(
                self.log_path,
                "score_assigned",
                request_id=request_id,
                use_base=use_base,
                adapter_version=requested_version,
                score_count=len(prompts),
                completion_token_count=sum(len(completion) for completion in completions),
                tail_max_tokens=None if _uncapped_eos_tail else 1,
                tail_termination="eos_only" if _uncapped_eos_tail else "not_attested",
                assignments=[
                    {
                        "worker_id": worker_id,
                        "logical_gpu": self.gpus[worker_id].logical_index,
                        "device_token": self.gpus[worker_id].device_token,
                        "score_indices": [assignment["score_index"] for assignment in worker_assignments],
                    }
                    for worker_id, worker_assignments in enumerate(assignments)
                ],
            )
            responses: list[dict[str, Any]] = []
            try:
                responses = await self._call_all(commands)
                expected = set(range(len(prompts)))
                records: dict[int, list[float]] = {}
                for response in responses:
                    if int(response.get("adapter_version", -1)) != requested_version:
                        raise RolloutWorkerError(
                            f"worker {response.get('worker_id')} scored with adapter "
                            f"v{response.get('adapter_version')}, expected v{requested_version}"
                        )
                    descriptor = response.get("result")
                    try:
                        worker_records = _read_payload(descriptor, ipc_dir=self.ipc_dir)
                    finally:
                        _remove_payload(descriptor, self.ipc_dir)
                    for score_index, sample_index, sequence in worker_records:
                        if score_index not in expected or sample_index != 0:
                            raise RolloutWorkerError(
                                f"worker returned unexpected scoring slot {(score_index, sample_index)}"
                            )
                        if score_index in records:
                            raise RolloutWorkerError(f"worker returned duplicate scoring slot {score_index}")
                        expected_tokens = completions[score_index]
                        if sequence.tokens != expected_tokens:
                            raise RolloutWorkerError(
                                f"worker scoring slot {score_index} returned misaligned completion tokens"
                            )
                        if sequence.logprobs is None or len(sequence.logprobs) != len(expected_tokens):
                            count = None if sequence.logprobs is None else len(sequence.logprobs)
                            raise RolloutWorkerError(
                                f"worker scoring slot {score_index} returned {count} logprob(s) "
                                f"for {len(expected_tokens)} completion token(s)"
                            )
                        if any(not math.isfinite(value) for value in sequence.logprobs):
                            raise RolloutWorkerError(f"worker scoring slot {score_index} returned a non-finite logprob")
                        records[score_index] = sequence.logprobs
                missing = sorted(expected - set(records))
                if missing:
                    raise RolloutWorkerError(f"workers omitted {len(missing)} scoring slot(s): {missing[:8]}")
                merged = [records[score_index] for score_index in range(len(prompts))]
            except asyncio.CancelledError:
                for response in responses:
                    _remove_payload(response.get("result"), self.ipc_dir)
                _append_jsonl(
                    self.log_path,
                    "score_cancelled_and_drained",
                    request_id=request_id,
                    use_base=use_base,
                    adapter_version=requested_version,
                )
                raise
            except BaseException as exc:
                for response in responses:
                    _remove_payload(response.get("result"), self.ipc_dir)
                self._mark_fatal(exc, event="score_failed")
                raise
            _append_jsonl(
                self.log_path,
                "score_completed",
                request_id=request_id,
                use_base=use_base,
                adapter_version=requested_version,
                score_count=len(prompts),
                completion_token_count=sum(len(completion) for completion in completions),
            )
            return merged

    async def score_completions_uncapped_eos_tail(
        self,
        prompt_tokens_batch: Sequence[Sequence[int]],
        completion_tokens_batch: Sequence[Sequence[int]],
        *,
        use_base: bool,
    ) -> list[list[float]]:
        """Preflight-only prompt scoring with an uncapped EOS-only tail."""

        return await self.score_completions(
            prompt_tokens_batch,
            completion_tokens_batch,
            use_base=use_base,
            _uncapped_eos_tail=True,
        )

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for endpoint in reversed(self._endpoints):
            try:
                endpoint.close_sync()
            except Exception:
                _append_jsonl(
                    self.log_path,
                    "worker_shutdown_failed",
                    worker_id=getattr(endpoint, "worker_id", None),
                    traceback=traceback.format_exc(),
                )
        _append_jsonl(
            self.log_path,
            "pool_stopped",
            adapter_version=self._adapter_version,
            phase=self.phase,
            sleeping=self._sleeping,
            fatal_error=self._fatal_error,
        )


class DistributedSamplerHandle:
    """Sampler and raw-policy scorer backed by the rollout worker pool.

    Generated-token logprobs describe the processed behavior distribution;
    ``score_completions`` returns raw policy scores through prompt logprobs.
    """

    def __init__(self, backend: "RolloutParallelBackend", *, use_base: bool):
        self._backend = backend
        self._use_base = use_base
        self._pending: list[tuple[dict[str, Any], asyncio.Future]] = []
        self._flush_task: Optional[asyncio.Task] = None

    async def sample(
        self,
        prompt: Any,
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
    ) -> list[SampledSequence]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending.append(
            (
                {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "num_samples": num_samples,
                },
                future,
            )
        )
        if self._flush_task is None:
            self._flush_task = loop.create_task(self._flush_pending())
        return await future

    async def sample_batch(
        self,
        prompts: Sequence[Any],
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
    ) -> list[list[SampledSequence]]:
        """Sample several prompts in one partitioned worker-pool request."""

        return await self._backend.pool.sample_batch(
            [list(prompt.to_ints()) for prompt in prompts],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=self._use_base,
        )

    @staticmethod
    def _same_generation_params(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return all(left[key] == right[key] for key in ("max_tokens", "temperature", "stop", "num_samples"))

    async def _flush_pending(self) -> None:
        # Coalesce sibling gather()/prefetch tasks into scheduler-visible prompt
        # batches.  Grouping by exact generation parameters preserves semantics.
        await asyncio.sleep(0)
        pending, self._pending = self._pending, []
        try:
            groups: list[list[tuple[dict[str, Any], asyncio.Future]]] = []
            for request, future in pending:
                matching = next(
                    (group for group in groups if self._same_generation_params(group[0][0], request)),
                    None,
                )
                if matching is None:
                    groups.append([(request, future)])
                else:
                    matching.append((request, future))

            for group in groups:
                first = group[0][0]
                try:
                    results = await self.sample_batch(
                        [request["prompt"] for request, _ in group],
                        max_tokens=first["max_tokens"],
                        temperature=first["temperature"],
                        stop=first["stop"],
                        num_samples=first["num_samples"],
                    )
                    if len(results) != len(group):
                        raise RolloutWorkerError(
                            f"rollout pool returned {len(results)} prompt group(s) for {len(group)} requests"
                        )
                except BaseException as exc:  # noqa: BLE001 -- fan out failure/cancellation to every caller
                    for _, future in group:
                        if not future.done():
                            future.set_exception(exc)
                else:
                    for result, (_, future) in zip(results, group):
                        if not future.done():
                            future.set_result(result)
        finally:
            self._flush_task = None
            if self._pending:
                self._flush_task = asyncio.get_running_loop().create_task(self._flush_pending())

    async def score_completions(
        self,
        prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
    ) -> list[list[float]]:
        """Score supplied completions on the rollout-worker vLLM engines."""

        return await self._backend.pool.score_completions(
            [list(prompt.to_ints()) for prompt in prompts],
            [list(tokens) for tokens in completion_tokens],
            use_base=self._use_base,
        )


class FrozenBaseSamplerHandle(DistributedSamplerHandle):
    """Sampler-only handle for target generation from immutable base weights."""

    def __init__(self, backend: "FrozenBaseVLLMBackend"):
        super().__init__(backend, use_base=True)

    async def sample_batch(
        self,
        prompts: Sequence[Any],
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
    ) -> list[list[SampledSequence]]:
        return await self._backend.sample_base_batch(
            [list(prompt.to_ints()) for prompt in prompts],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
        )

    async def score_completions(
        self,
        prompts: Sequence[Any],
        completion_tokens: Sequence[Sequence[int]],
    ) -> list[list[float]]:
        raise NotImplementedError("frozen-base target generation does not provide coordinator scoring")


class FrozenBaseVLLMBackend:
    """vLLM-only backend for immutable target generation.

    No Transformers causal-LM or PEFT policy is constructed. With an explicit
    GPU list, the CPU coordinator starts one ``enable_lora=False`` worker per
    assigned logical GPU and uses every GPU in the bundle. Without a worker
    list, one in-process base-only vLLM engine provides a backwards-compatible
    single-GPU path.
    """

    renderer_source = "hf"
    policy_samplers_are_snapshots = True

    def __init__(
        self,
        *,
        model: str,
        engine_kwargs: Optional[dict[str, Any]] = None,
        gpus: Sequence[RolloutGPU] = (),
        status_dir: str | Path | None = None,
        start_timeout_seconds: float = 1800.0,
        request_timeout_seconds: float = 7200.0,
        shutdown_timeout_seconds: float = 30.0,
        pool: Optional[RolloutWorkerPool] = None,
        sampler_factory: Optional[Any] = None,
    ):
        if pool is None and gpus and status_dir is None:
            raise ValueError("base-only rollout workers require a status directory")
        self.model_name = model
        self.engine_kwargs = dict(engine_kwargs or {})
        self.gpus = tuple(gpus)
        self.status_dir = None if status_dir is None else Path(status_dir)
        self.start_timeout_seconds = start_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.pool = pool
        self._sampler_factory = sampler_factory
        self._sampler: Optional[Any] = None
        self._setup = False
        self._shutdown = False

    def setup(
        self,
        *,
        model: str,
        lora: Any = None,
        resume_from: Optional[str] = None,
        resume_with_optimizer: bool = False,
    ) -> None:
        del lora
        if model != self.model_name:
            raise ValueError(f"frozen-base backend was configured for {self.model_name!r}, got {model!r}")
        if resume_from is not None or resume_with_optimizer:
            raise ValueError("frozen-base target generation cannot resume training state")
        if self._setup:
            return
        try:
            if self.pool is None and self.gpus:
                assert self.status_dir is not None
                self.pool = RolloutWorkerPool(
                    model=model,
                    gpus=self.gpus,
                    engine_kwargs=self.engine_kwargs,
                    status_dir=self.status_dir,
                    enable_lora=False,
                    start_timeout_seconds=self.start_timeout_seconds,
                    request_timeout_seconds=self.request_timeout_seconds,
                    shutdown_timeout_seconds=self.shutdown_timeout_seconds,
                )
            if self.pool is not None:
                if getattr(self.pool, "enable_lora", None) is not False:
                    raise ValueError("frozen-base backend requires an enable_lora=False worker pool")
                self.pool.start()
                print(
                    f"FrozenBaseVLLMBackend: {self.pool.worker_count} base-only worker(s), "
                    f"status={self.pool.status_dir}",
                    flush=True,
                )
            else:
                if self._sampler_factory is None:
                    from ctm.backends.local.vllm_sampler import VLLMSampler

                    self._sampler_factory = VLLMSampler
                self._sampler = self._sampler_factory(model=model, enable_lora=False, **self.engine_kwargs)
                print("FrozenBaseVLLMBackend: one in-process base-only vLLM engine", flush=True)
            self._setup = True
        except BaseException:
            self.shutdown()
            raise

    def base_sampler(self) -> FrozenBaseSamplerHandle:
        if not self._setup:
            raise RolloutWorkerError("FrozenBaseVLLMBackend.setup() must be called before sampling")
        return FrozenBaseSamplerHandle(self)

    def policy_sampler(self, name: str) -> None:
        del name
        raise NotImplementedError("frozen-base target generation has no policy adapter")

    async def sample_base_batch(
        self,
        prompt_tokens_batch: Sequence[Sequence[int]],
        *,
        max_tokens: int | None,
        temperature: float,
        stop: Any,
        num_samples: int,
    ) -> list[list[SampledSequence]]:
        if not self._setup:
            raise RolloutWorkerError("FrozenBaseVLLMBackend.setup() must be called before sampling")
        if self.pool is not None:
            return await self.pool.sample_batch(
                prompt_tokens_batch,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                num_samples=num_samples,
                use_base=True,
            )
        if self._sampler is None:
            raise RolloutWorkerError("frozen-base vLLM sampler is unavailable")
        return await asyncio.to_thread(
            self._sampler.sample_batch,
            [list(tokens) for tokens in prompt_tokens_batch],
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            num_samples=num_samples,
            use_base=True,
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self.pool is not None:
            self.pool.shutdown()
        if self._sampler is not None:
            self._sampler.shutdown()
            self._sampler = None


class RolloutParallelBackend:
    """TrainingBackend proxy replacing local sampling with rollout workers.

    The wrapped backend still owns setup, forward/backward, optimizer, KL,
    checkpoints, and raw-policy scoring.  Crucially, this proxy never calls its
    ``policy_sampler`` or ``base_sampler`` methods, so its in-process vLLM engine
    remains cold on the coordinator GPU.
    """

    renderer_source = "hf"
    policy_samplers_are_snapshots = False

    def __init__(
        self,
        training_backend: Any,
        *,
        gpus: Sequence[RolloutGPU],
        status_dir: str | Path,
        worker_vllm_options: Optional[dict[str, Any]] = None,
        start_timeout_seconds: float = 1800.0,
        request_timeout_seconds: float = 7200.0,
        shutdown_timeout_seconds: float = 30.0,
        qwen35_rollout_parity_attestation: str | Path | None = None,
        muse_rollout_parity_attestation: str | Path | None = None,
        qwen35_rollout_parity_bootstrap: bool = False,
        pool: Optional[RolloutWorkerPool] = None,
    ):
        self.training_backend = training_backend
        self.gpus = tuple(gpus)
        self.status_dir = Path(status_dir)
        self.worker_vllm_options = (
            dict(worker_vllm_options)
            if worker_vllm_options is not None
            else dict(getattr(training_backend, "vllm_options", {}))
        )
        self.start_timeout_seconds = start_timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.qwen35_rollout_parity_attestation = (
            None if qwen35_rollout_parity_attestation is None else Path(qwen35_rollout_parity_attestation)
        )
        self.muse_rollout_parity_attestation = (
            None if muse_rollout_parity_attestation is None else Path(muse_rollout_parity_attestation)
        )
        # This is intentionally not exposed through the normal training CLI.
        # Only the non-production fixed-token preflight harness may bootstrap
        # the first evidence file, before a production backend is allowed to
        # start. See infra/vastai/benchmark_qwen35_opct_group.py.
        self.qwen35_rollout_parity_bootstrap = qwen35_rollout_parity_bootstrap
        self.pool: Optional[RolloutWorkerPool] = pool
        self._adapter_version = 0
        self._adapter_root: Optional[Path] = None
        self._model_name: Optional[str] = None
        self._qwen35_rollout_parity: Optional[dict[str, Any]] = None
        self._qwen35_rollout_parity_path: Optional[Path] = None
        self._qwen35_rollout_parity_sha256: Optional[str] = None
        self._muse_rollout_parity: Optional[dict[str, Any]] = None
        self._muse_rollout_parity_path: Optional[Path] = None
        self._muse_rollout_parity_sha256: Optional[str] = None
        self._shutdown = False

    def __getattr__(self, name: str) -> Any:
        # Forward every non-sampling TrainingBackend operation.
        return getattr(self.training_backend, name)

    @property
    def sampling_training_overlap_supported(self) -> bool:
        """Whether callers may prefetch rollouts during coordinator training.

        The normal distributed-worker topology keeps rollout GPUs distinct from
        the coordinator and continues to support overlap.  An opt-in sleeping
        pool reclaims those GPUs for training, so callers must complete a
        training phase before scheduling the next rollout phase.
        """

        if self.pool is not None:
            return self.pool.sampling_training_overlap_supported
        return not bool(self.worker_vllm_options.get("enable_sleep_mode", False))

    @property
    def rollout_phase(self) -> str:
        """Current worker resource phase, available for orchestration/logging."""

        return "rollout" if self.pool is None else self.pool.phase

    async def enter_training_phase(self) -> None:
        """Sleep rollout workers before coordinator KL/forward/backward work."""

        if self.pool is None:
            raise RolloutWorkerError(
                "RolloutParallelBackend.setup() must be called before entering a training phase"
            )
        await self.pool.enter_training_phase()

    async def enter_rollout_phase(self) -> None:
        """Release trainer caches, then wake rollout workers.

        In phase-shared mode the wrapped backend may own one persistent HF
        replica on every rollout GPU.  Its all-rank barrier must complete
        before any worker asks vLLM to reclaim GPU memory.  Keeping this
        ordering at the outer boundary also makes the lifecycle independent
        of whether the wrapped trainer is single-rank or replicated.
        """

        if self.pool is None:
            raise RolloutWorkerError(
                "RolloutParallelBackend.setup() must be called before entering a rollout phase"
            )
        await self._release_training_memory_before_worker_wake()
        await self.pool.enter_rollout_phase()

    async def _release_training_memory_before_worker_wake(self) -> None:
        """Run the trainer-side release barrier for a real shared-GPU wake."""

        assert self.pool is not None
        # Legacy/injected pools used by compatibility tooling predate the
        # phase-sharing lifecycle.  Treat those as sleep-disabled rather than
        # requiring unrelated callers to emulate resource-phase state.
        if not getattr(self.pool, "sleep_enabled", False) or (
            getattr(self.pool, "phase", "rollout") != "training"
        ):
            return
        release = getattr(self.training_backend, "enter_rollout_phase", None)
        if not callable(release):
            raise RolloutWorkerError(
                "sleep-enabled rollout workers require the wrapped training backend to expose "
                "an async enter_rollout_phase() trainer-memory release barrier"
            )
        await release()

    async def sleep_rollout_workers(self) -> None:
        """Explicit lifecycle alias used by phase-sharing orchestrators."""

        await self.enter_training_phase()

    async def wake_rollout_workers(self) -> None:
        """Explicit lifecycle alias used by phase-sharing orchestrators."""

        await self.enter_rollout_phase()

    async def wake_up_rollout_workers(self) -> None:
        """vLLM-named compatibility alias for :meth:`wake_rollout_workers`."""

        await self.enter_rollout_phase()

    async def incorporate_kl_penalty(
        self,
        datums: Sequence[Any],
        *,
        kl_coef: float,
        kl_discount_factor: float,
    ) -> dict[str, float]:
        # KL-to-base performs coordinator forwards and therefore belongs to
        # the same protected training phase as backward.
        await self.enter_training_phase()
        return await self.training_backend.incorporate_kl_penalty(
            datums,
            kl_coef=kl_coef,
            kl_discount_factor=kl_discount_factor,
        )

    async def submit_forward_backward(self, datums: Sequence[Any], loss_fn: str) -> Any:
        """Run a coordinator update only after workers are safely asleep."""

        await self.enter_training_phase()
        return await self.training_backend.submit_forward_backward(datums, loss_fn)

    async def submit_opct_forward_backward(self, *args: Any, **kwargs: Any) -> Any:
        """OPCT variant of :meth:`submit_forward_backward` with the same barrier."""

        await self.enter_training_phase()
        return await self.training_backend.submit_opct_forward_backward(*args, **kwargs)

    async def submit_optim_step(self, *, learning_rate: float, adam: Any) -> Any:
        """Keep workers asleep through optimizer mutation and adapter snapshotting."""

        await self.enter_training_phase()
        return await self.training_backend.submit_optim_step(learning_rate=learning_rate, adam=adam)

    async def save_checkpoint(self, *, name: str, log_dir: str | Path, loop_state: dict, kind: str) -> Any:
        """Keep workers asleep while rank zero verifies and writes a checkpoint.

        Phase-shared checkpoints hash every trainer replica and materialize the
        canonical adapter on rank zero.  Treat that as trainer work: allowing a
        prefetched rollout to wake vLLM on the same GPUs would reintroduce the
        memory race that the phase barrier exists to prevent.
        """

        await self.enter_training_phase()
        return await self.training_backend.save_checkpoint(
            name=name,
            log_dir=log_dir,
            loop_state=loop_state,
            kind=kind,
        )

    def setup(
        self,
        *,
        model: str,
        lora: Any,
        resume_from: Optional[str] = None,
        resume_with_optimizer: bool = False,
    ) -> None:
        self.training_backend.setup(
            model=model,
            lora=lora,
            resume_from=resume_from,
            resume_with_optimizer=resume_with_optimizer,
        )
        try:
            self._model_name = model
            if getattr(self.training_backend, "renderer_source", None) != "hf":
                raise ValueError("rollout workers can wrap only the local HF-rendered backend")
            if getattr(self.training_backend, "sampler", None) != "vllm":
                raise ValueError("rollout workers require --local-sampler vllm")
            if not getattr(self.training_backend, "use_lora", False):
                raise ValueError("rollout workers require LoRA so policy snapshots can be hot-reloaded")
            model_object = getattr(self.training_backend, "model", None)
            if model_object is None or not callable(getattr(model_object, "save_pretrained", None)):
                raise ValueError("wrapped local backend has no saveable PEFT policy model")

            if self.pool is None:
                self.pool = RolloutWorkerPool(
                    model=model,
                    gpus=self.gpus,
                    engine_kwargs=self.worker_vllm_options,
                    status_dir=self.status_dir,
                    start_timeout_seconds=self.start_timeout_seconds,
                    request_timeout_seconds=self.request_timeout_seconds,
                    shutdown_timeout_seconds=self.shutdown_timeout_seconds,
                )
            if is_qwen35_model_name(model):
                parity_path = (
                    self.qwen35_rollout_parity_attestation
                    if self.qwen35_rollout_parity_attestation is not None
                    else self.status_dir / QWEN35_WORKER_PARITY_ATTESTATION_NAME
                )
                # The translation manifest proves every future raw/vLLM
                # snapshot is byte-preserving. This independent fixed-token
                # preflight proves that the actual worker scorer applies a
                # meaningful nonzero translated LoRA at all, rather than only
                # acknowledging its adapter version.
                if not self.qwen35_rollout_parity_bootstrap:
                    self._qwen35_rollout_parity = validate_qwen35_rollout_worker_parity_attestation(
                        parity_path,
                        expected_model=model,
                        expected_worker_gpus=self.pool.gpus,
                        expected_worker_engine_kwargs=self.pool.engine_kwargs,
                    )
                    self._qwen35_rollout_parity_path = Path(parity_path).resolve()
                    self._qwen35_rollout_parity_sha256 = file_sha256(self._qwen35_rollout_parity_path)
            if is_muse_glimmer_model_name(model):
                parity_path = (
                    self.muse_rollout_parity_attestation
                    if self.muse_rollout_parity_attestation is not None
                    else self.status_dir / MUSE_PARITY_ATTESTATION_NAME
                )
                # Muse requires the post-fix vLLM stack and a real non-zero
                # LoRA transport proof.  There is intentionally no production
                # bootstrap bypass: the separate GPU preflight must finish
                # before a training backend is allowed to publish even v1.
                self._muse_rollout_parity = validate_muse_rollout_worker_parity_attestation(
                    parity_path,
                    expected_model=model,
                    expected_worker_gpus=self.pool.gpus,
                    expected_worker_engine_kwargs=self.pool.engine_kwargs,
                )
                self._muse_rollout_parity_path = Path(parity_path).resolve()
                self._muse_rollout_parity_sha256 = muse_file_sha256(self._muse_rollout_parity_path)
            print(
                f"RolloutParallelBackend: {self.pool.worker_count} worker(s), status={self.pool.status_dir}",
                flush=True,
            )
            self._adapter_root = self.pool.status_dir / "adapters"
            self._adapter_root.mkdir(parents=True, exist_ok=True)
            initial_version = 1
            adapter_path = self._snapshot_adapter(initial_version)
            self.pool.start()
            self.pool.publish_adapter_sync(adapter_path, version=initial_version)
            self._adapter_version = initial_version
        except BaseException:
            self.shutdown()
            raise

    def _snapshot_adapter(self, version: int) -> Path:
        if self._adapter_root is None:
            raise RolloutWorkerError("adapter snapshot root is not initialized")
        if self._model_name is None:
            raise RolloutWorkerError("model name is not initialized for adapter snapshot")
        final = self._adapter_root / f"v{version:08d}"
        if final.exists():
            raise RolloutWorkerError(f"adapter snapshot already exists: {final}")
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".v{version:08d}-",
                dir=str(self._adapter_root),
            )
        )
        try:
            qwen35_compatibility = is_qwen35_model_name(self._model_name)
            muse_compatibility = is_muse_glimmer_model_name(self._model_name)
            if qwen35_compatibility:
                if self._qwen35_rollout_parity is None:
                    if not self.qwen35_rollout_parity_bootstrap:
                        raise RolloutWorkerError(
                            "Qwen3.5 rollout workers require a validated nonzero fixed-token HF/worker parity attestation"
                        )
                elif (
                    self._qwen35_rollout_parity_path is None
                    or self._qwen35_rollout_parity_sha256 is None
                    or file_sha256(self._qwen35_rollout_parity_path) != self._qwen35_rollout_parity_sha256
                ):
                    raise RolloutWorkerError(
                        "Qwen3.5 rollout worker parity attestation changed after setup; refusing to publish a policy snapshot"
                    )
                else:
                    # The sidecar's own digest is not enough: rebind its
                    # referenced nonzero v2 raw/translated bytes before every
                    # dynamic policy version is published. This is cheap
                    # relative to a rollout update and makes a later source
                    # mutation fail closed rather than merely leaving stale
                    # provenance in the version manifest.
                    assert self.pool is not None
                    current_parity = validate_qwen35_rollout_worker_parity_attestation(
                        self._qwen35_rollout_parity_path,
                        expected_model=self._model_name,
                        expected_worker_gpus=self.pool.gpus,
                        expected_worker_engine_kwargs=self.pool.engine_kwargs,
                    )
                    if current_parity != self._qwen35_rollout_parity:
                        raise RolloutWorkerError(
                            "Qwen3.5 rollout worker parity attestation no longer matches the validated preflight"
                        )
            if muse_compatibility:
                if (
                    self._muse_rollout_parity is None
                    or self._muse_rollout_parity_path is None
                    or self._muse_rollout_parity_sha256 is None
                    or muse_file_sha256(self._muse_rollout_parity_path) != self._muse_rollout_parity_sha256
                ):
                    raise RolloutWorkerError(
                        "Muse Glimmer rollout-worker parity attestation is absent or changed after setup"
                    )
                assert self.pool is not None
                current_muse_parity = validate_muse_rollout_worker_parity_attestation(
                    self._muse_rollout_parity_path,
                    expected_model=self._model_name,
                    expected_worker_gpus=self.pool.gpus,
                    expected_worker_engine_kwargs=self.pool.engine_kwargs,
                )
                if current_muse_parity != self._muse_rollout_parity:
                    raise RolloutWorkerError(
                        "Muse Glimmer rollout-worker parity no longer matches the validated preflight"
                    )
            raw_dir = staging / "raw" if qwen35_compatibility else staging
            self.training_backend.model.save_pretrained(str(raw_dir))
            published_relative_path = "."
            compatibility_manifest: dict[str, Any] | None = None
            if qwen35_compatibility:
                compatibility_dir = staging / "vllm_compat"
                compatibility_manifest = materialize_qwen35_vllm_rollout_compat_adapter(
                    raw_dir,
                    compatibility_dir,
                    model=self._model_name,
                    adapter_version=version,
                )
                published_relative_path = "vllm_compat"
            manifest = {
                "adapter_version": version,
                "created_at": _utc_now(),
                "source": "coordinator_training_model",
                "vllm_adapter_relative_path": published_relative_path,
                **(
                    {
                        "qwen35_vllm_compatibility": {
                            "raw_adapter_relative_path": "raw",
                            "vllm_adapter_relative_path": published_relative_path,
                            "manifest_name": QWEN35_COMPAT_MANIFEST_NAME,
                            "source_adapter_model_sha256": compatibility_manifest["source_adapter"][
                                "adapter_model_sha256"
                            ],
                            "vllm_adapter_model_sha256": compatibility_manifest["destination_adapter"][
                                "adapter_model_sha256"
                            ],
                            **(
                                {
                                    "worker_parity_preflight": {
                                        "path": str(self._qwen35_rollout_parity_path),
                                        "sha256": self._qwen35_rollout_parity_sha256,
                                        "schema": self._qwen35_rollout_parity["schema"],
                                        "preflight_adapter_version": self._qwen35_rollout_parity["adapter_version"],
                                        "preflight_raw_adapter_model_sha256": self._qwen35_rollout_parity["raw_adapter"][
                                            "adapter_model_sha256"
                                        ],
                                        "preflight_vllm_adapter_model_sha256": self._qwen35_rollout_parity[
                                            "vllm_adapter"
                                        ]["adapter_model_sha256"],
                                        "snapshot_raw_adapter_model_sha256": compatibility_manifest["source_adapter"][
                                            "adapter_model_sha256"
                                        ],
                                        "snapshot_vllm_adapter_model_sha256": compatibility_manifest[
                                            "destination_adapter"
                                        ]["adapter_model_sha256"],
                                    }
                                }
                                if self._qwen35_rollout_parity is not None
                                else {"worker_parity_bootstrap": True}
                            ),
                        }
                    }
                    if compatibility_manifest is not None
                    else {}
                ),
                **(
                    {
                        "muse_glimmer_vllm_parity": {
                            "path": str(self._muse_rollout_parity_path),
                            "sha256": self._muse_rollout_parity_sha256,
                            "schema": self._muse_rollout_parity["schema"],
                            "vllm_commit": self._muse_rollout_parity["runtime"]["vllm_commit"],
                            "max_tokens": None,
                            "termination": "eos_only",
                        }
                    }
                    if muse_compatibility
                    else {}
                ),
            }
            (staging / "ctm_rollout_adapter.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, final)
        except BaseException:
            # Preserve a failed staging directory for diagnosis rather than
            # recursively deleting evidence from a training run.
            failed = self._adapter_root / f"_failed_v{version:08d}_{uuid.uuid4().hex[:8]}"
            if staging.exists():
                os.replace(staging, failed)
            raise
        return final / published_relative_path if published_relative_path != "." else final

    def policy_sampler(self, name: str) -> DistributedSamplerHandle:
        if self.pool is None:
            raise RolloutWorkerError("RolloutParallelBackend.setup() must be called before sampling")
        return DistributedSamplerHandle(self, use_base=False)

    def base_sampler(self) -> DistributedSamplerHandle:
        if self.pool is None:
            raise RolloutWorkerError("RolloutParallelBackend.setup() must be called before sampling")
        return DistributedSamplerHandle(self, use_base=True)

    async def refresh_policy_sampler(self, name: str) -> DistributedSamplerHandle:
        if self.pool is None:
            raise RolloutWorkerError("RolloutParallelBackend.setup() must be called before refresh")
        verify = getattr(self.training_backend, "verify_policy_publication", None)
        if callable(verify):
            # A replicated backend cannot publish merely because the previous
            # optimizer call usually checked hashes.  Bind this exact snapshot
            # to an unconditional all-rank verification, while leaving the
            # unused rank-zero in-process vLLM cold.
            await verify()
        # publish_adapter() wakes a sleeping pool while holding its operation
        # lock.  Release every persistent trainer replica first; otherwise a
        # peer process can retain caching-allocator blocks and make that wake
        # fail even though the logical training phase has completed.
        await self._release_training_memory_before_worker_wake()
        version = self._adapter_version + 1
        adapter_path = self._snapshot_adapter(version)
        await self.pool.publish_adapter(adapter_path, version=version)
        self._adapter_version = version
        return DistributedSamplerHandle(self, use_base=False)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        try:
            if self.pool is not None:
                self.pool.shutdown()
        finally:
            shutdown = getattr(self.training_backend, "shutdown", None)
            if callable(shutdown):
                shutdown()


__all__ = [
    "DistributedSamplerHandle",
    "FrozenBaseSamplerHandle",
    "FrozenBaseVLLMBackend",
    "RolloutGPU",
    "RolloutParallelBackend",
    "RolloutWorkerError",
    "RolloutWorkerPool",
    "resolve_base_only_gpus",
    "resolve_rollout_gpus",
]
