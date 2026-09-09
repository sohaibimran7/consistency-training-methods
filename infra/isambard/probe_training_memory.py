"""Profile one isolated LocalBackend training step on every visible GPU.

Run exactly one batch/sequence/checkpointing combination per process. Process
isolation is intentional: a failed or retained autograd object from one shape
must not affect the next shape. The forward/backward call uses LocalBackend's
real BCT cross-entropy path, including its float32 log-softmax, and the optional
optimizer phase uses the backend's real AdamW setup.
"""

# ruff: noqa: E402 -- force this checkout ahead of the shared editable install.

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import peft
import torch
import transformers
from tinker import types
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from ctm.backends.local.engine import LocalBackend
from ctm.core.config import AdamConfig, LoRAConfig

GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device-map", default="auto", help="Use 'none' to disable sharding")
    parser.add_argument("--max-memory-per-gpu", default="45GiB")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--optimizer-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the first real AdamW step (and its state allocation)",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.sequence_length < 2:
        parser.error("batch size must be positive and sequence length must be at least 2")
    return args


def gib(value: int) -> float:
    return value / GIB


def synchronize_all() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def memory_snapshot(label: str) -> list[dict[str, Any]]:
    """Read current and peak allocator state for every visible GPU."""

    synchronize_all()
    rows: list[dict[str, Any]] = []
    print(f"\n{label}")
    print("gpu  allocated  reserved  peak_alloc  peak_resv  used_total  free       total (GiB)")
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        row = {
            "gpu": index,
            "allocated_gib": gib(torch.cuda.memory_allocated(index)),
            "reserved_gib": gib(torch.cuda.memory_reserved(index)),
            "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(index)),
            "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(index)),
            "used_total_gib": gib(total - free),
            "free_gib": gib(free),
            "total_gib": gib(total),
        }
        rows.append(row)
        print(
            f"{index:>3}  {row['allocated_gib']:>9.2f}  {row['reserved_gib']:>8.2f}  "
            f"{row['peak_allocated_gib']:>10.2f}  {row['peak_reserved_gib']:>9.2f}  "
            f"{row['used_total_gib']:>10.2f}  {row['free_gib']:>8.2f}  {row['total_gib']:>8.2f}",
            flush=True,
        )
    return rows


def reset_peaks_all() -> None:
    synchronize_all()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)


def make_bct_datums(batch_size: int, sequence_length: int) -> list[Any]:
    """Build the same datum type consumed by LocalBackend BCT training."""

    datums = []
    for batch_index in range(batch_size):
        tokens = [(position + batch_index) % 1000 for position in range(sequence_length)]
        weights = torch.ones(sequence_length, dtype=torch.float32)
        datums.append(datum_from_model_input_weights(types.ModelInput.from_ints(tokens=tokens), weights))
    return datums


async def forward_backward(backend: LocalBackend, datums: list[Any]):
    pending = await backend.submit_forward_backward(datums, "cross_entropy")
    return await pending.result()


async def optimizer_step(backend: LocalBackend, learning_rate: float) -> None:
    adam = AdamConfig(learning_rate=learning_rate)
    pending = await backend.submit_optim_step(learning_rate=learning_rate, adam=adam)
    await pending.result()


def compact_error(exc: BaseException) -> str:
    return " ".join(str(exc).split())[:2000]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("this profiler requires at least one CUDA GPU")

    device_map = None if args.device_map.lower() == "none" else args.device_map
    max_memory = (
        {index: args.max_memory_per_gpu for index in range(torch.cuda.device_count())}
        if device_map and args.max_memory_per_gpu
        else None
    )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    result: dict[str, Any] = {
        "schema_version": 1,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "model": args.model,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "device_map": device_map,
        "max_memory": max_memory,
        "visible_gpu_count": torch.cuda.device_count(),
        "software_versions": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "seed": args.seed,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": False,
        },
        "optimizer_step": args.optimizer_step,
        "phases": {},
    }
    print("PROFILE_CONFIG_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"gpu{index}: {props.name}, {gib(props.total_memory):.2f} GiB", flush=True)

    result["phases"]["process_start"] = memory_snapshot("process start")
    backend: LocalBackend | None = None
    datums: list[Any] | None = None
    step_output = None
    started = time.monotonic()
    exit_code = 0

    try:
        backend = LocalBackend(
            dtype=torch.bfloat16,
            use_lora=True,
            sampler="vllm",
            device_map=device_map,
            max_memory=max_memory,
            vllm_options={"gpu_memory_utilization": 0.25},
        )
        lora = LoRAConfig(
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            seed=args.seed,
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
        )
        backend.setup(model=args.model, lora=lora)
        model = backend._require_model()
        model.train()
        result["model_class"] = type(model).__name__

        peft_config = model.peft_config["default"]
        result["resolved_targets"] = {
            "modules": sorted(peft_config.target_modules),
            "parameters": sorted(peft_config.target_parameters or []),
        }
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        result["trainable_parameters"] = sum(parameter.numel() for _, parameter in trainable)
        result["trainable_bytes"] = sum(parameter.numel() * parameter.element_size() for _, parameter in trainable)
        result["input_device"] = str(backend.device)
        result["hf_device_map"] = {
            str(name): str(device) for name, device in (getattr(model, "hf_device_map", {}) or {}).items()
        }

        if args.gradient_checkpointing:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            if not getattr(model, "is_gradient_checkpointing", False):
                raise RuntimeError("gradient checkpointing was requested but did not activate")
        result["gradient_checkpointing_active"] = bool(getattr(model, "is_gradient_checkpointing", False))
        print(
            "gradient checkpointing: " + ("enabled" if result["gradient_checkpointing_active"] else "disabled"),
            flush=True,
        )

        # Model loading can leave temporary cached blocks. The production
        # sleep/wake lifecycle also empties the cache before the training phase.
        gc.collect()
        torch.cuda.empty_cache()
        result["phases"]["model_resident"] = memory_snapshot("PEFT model resident")

        datums = make_bct_datums(args.batch_size, args.sequence_length)
        model.zero_grad(set_to_none=True)
        reset_peaks_all()
        forward_started = time.monotonic()
        step_output = asyncio.run(forward_backward(backend, datums))
        result["forward_backward_seconds"] = time.monotonic() - forward_started
        result["loss"] = step_output.metrics["loss"]
        result["phases"]["after_forward_backward"] = memory_snapshot("after LocalBackend forward/backward")

        # LocalBackend returns detached CPU logprobs, so no forward graph should
        # remain when the optimizer runs. Drop the returned value explicitly.
        del step_output
        step_output = None
        gc.collect()
        if args.optimizer_step:
            optimizer_started = time.monotonic()
            asyncio.run(optimizer_step(backend, args.learning_rate))
            result["optimizer_seconds"] = time.monotonic() - optimizer_started
            result["phases"]["after_optimizer"] = memory_snapshot("after first AdamW step")
        result["outcome"] = "ok"
    except torch.OutOfMemoryError as exc:
        result["outcome"] = "oom"
        result["error"] = compact_error(exc)
        result["phases"]["at_oom"] = memory_snapshot("at CUDA OOM")
        print(f"CUDA OOM: {result['error']}", flush=True)
    except Exception as exc:
        result["outcome"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = compact_error(exc)
        exit_code = 1
        traceback.print_exc()
    finally:
        if backend is not None and backend.model is not None:
            backend.model.zero_grad(set_to_none=True)
        if step_output is not None:
            del step_output
        if datums is not None:
            del datums
        gc.collect()
        torch.cuda.empty_cache()
        result["phases"]["after_cleanup"] = memory_snapshot("after explicit cleanup")
        result["elapsed_seconds"] = time.monotonic() - started
        print("PROFILE_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
