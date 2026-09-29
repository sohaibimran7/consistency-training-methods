"""Non-production Qwen3.5 coordinator + multi-GPU rollout capacity probe."""

# ruff: noqa: E402 -- force this checkout ahead of any shared editable install.

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tinker import types
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from transformers import AutoTokenizer

from ctm.backends.local.engine import LocalBackend, _selected_token_components
from ctm.backends.local.rollout_workers import RolloutWorkerPool, resolve_rollout_gpus
from ctm.core.config import AdamConfig, LoRAConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--device-map", default="none", help="Transformers device map; 'none' keeps one GPU")
    parser.add_argument("--max-memory-per-gpu", default=None, help="Optional Accelerate placement cap")
    parser.add_argument("--worker-gpus", default="1,2,3,4,5,6,7")
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/ctm-qwen-validation/benchmark"))
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--training-datums", type=int, default=1)
    parser.add_argument("--training-steps", type=int, default=1)
    parser.add_argument("--forward-microbatch-max-datums", type=int, default=8)
    parser.add_argument("--forward-microbatch-max-tokens", type=int, default=2048)
    parser.add_argument("--target-logprob-chunk-size", type=int, default=32)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable non-reentrant Transformers gradient checkpointing on the coordinator model",
    )
    parser.add_argument(
        "--gradient-checkpoint-layers",
        type=int,
        default=None,
        help="Benchmark only: checkpoint the first N backbone layers instead of every layer",
    )
    parser.add_argument(
        "--gradient-checkpoint-full-attention-layers",
        type=int,
        default=None,
        help="Benchmark only: checkpoint the first N full-attention backbone layers",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Run only the coordinator training probe; do not start rollout workers",
    )
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-warmup-prompts", type=int, default=0)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force every benchmark completion to max-new-tokens (stress testing only)",
    )
    parser.add_argument("--worker-gpu-mem-util", type=float, default=0.9)
    args = parser.parse_args()
    if (
        min(
            args.sequence_length,
            args.training_datums,
            args.training_steps,
            args.forward_microbatch_max_datums,
            args.forward_microbatch_max_tokens,
            args.target_logprob_chunk_size,
            args.prompt_count,
            args.samples_per_prompt,
            args.max_new_tokens,
        )
        < 1
    ):
        parser.error("sequence length, training sizes, prompt/sample counts, and max new tokens must be positive")
    if args.generation_warmup_prompts < 0:
        parser.error("--generation-warmup-prompts must be non-negative")
    if args.gradient_checkpoint_layers is not None and args.gradient_checkpoint_layers < 1:
        parser.error("--gradient-checkpoint-layers must be positive")
    if (
        args.gradient_checkpoint_full_attention_layers is not None
        and args.gradient_checkpoint_full_attention_layers < 1
    ):
        parser.error("--gradient-checkpoint-full-attention-layers must be positive")
    if sum(
        option is not None
        for option in (
            True if args.gradient_checkpointing else None,
            args.gradient_checkpoint_layers,
            args.gradient_checkpoint_full_attention_layers,
        )
    ) > 1:
        parser.error("choose full or one selective gradient-checkpointing mode")
    if not 0 < args.worker_gpu_mem_util <= 1:
        parser.error("--worker-gpu-mem-util must be in (0, 1]")
    return args


async def train_probe(
    backend: LocalBackend,
    tokens: list[int],
    *,
    datum_count: int,
    step_count: int,
) -> dict[str, object]:
    datums = []
    for datum_index in range(datum_count):
        # Rotate the synthetic sequence so bucketing sees equally sized but
        # independently materialized datums, like a real logical batch.
        offset = datum_index % len(tokens)
        datum_tokens = tokens[offset:] + tokens[:offset]
        weights = torch.ones(len(datum_tokens), dtype=torch.float32)
        datums.append(datum_from_model_input_weights(types.ModelInput.from_ints(tokens=datum_tokens), weights))

    adam = AdamConfig(learning_rate=1e-4)
    steps: list[dict[str, object]] = []
    for step_index in range(step_count):
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
            torch.cuda.reset_peak_memory_stats(device_index)
        started = time.monotonic()
        pending = await backend.submit_forward_backward(datums, "cross_entropy")
        output = await pending.result()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
        forward_backward_seconds = time.monotonic() - started

        started = time.monotonic()
        pending_optim = await backend.submit_optim_step(learning_rate=1e-4, adam=adam)
        await pending_optim.result()
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
        optimizer_seconds = time.monotonic() - started
        peak_allocated = [torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())]
        peak_reserved = [torch.cuda.max_memory_reserved(index) for index in range(torch.cuda.device_count())]
        steps.append(
            {
                "step_index": step_index,
                "loss": float(output.metrics["loss"]),
                "forward_backward_seconds": forward_backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "coordinator_peak_allocated_bytes": max(peak_allocated),
                "coordinator_peak_reserved_bytes": max(peak_reserved),
                "coordinator_peak_allocated_bytes_by_gpu": peak_allocated,
                "coordinator_peak_reserved_bytes_by_gpu": peak_reserved,
            }
        )

    warm_steps = steps[1:] or steps
    warm_forward_backward = [float(step["forward_backward_seconds"]) for step in warm_steps]
    warm_optimizer = [float(step["optimizer_seconds"]) for step in warm_steps]
    return {
        "datum_count": datum_count,
        "sequence_length": len(tokens),
        "step_count": step_count,
        "steps": steps,
        "steady_state_forward_backward_seconds_median": statistics.median(warm_forward_backward),
        "steady_state_optimizer_seconds_median": statistics.median(warm_optimizer),
        "max_coordinator_peak_allocated_bytes": max(int(step["coordinator_peak_allocated_bytes"]) for step in steps),
        "max_coordinator_peak_reserved_bytes": max(int(step["coordinator_peak_reserved_bytes"]) for step in steps),
    }


def repeated_tokens(seed_tokens: list[int], length: int) -> list[int]:
    if not seed_tokens:
        raise RuntimeError("tokenizer produced an empty probe prompt")
    return (seed_tokens * ((length + len(seed_tokens) - 1) // len(seed_tokens)))[:length]


def length_summary(lengths: list[int], *, cap: int) -> dict[str, object]:
    if not lengths:
        raise ValueError("cannot summarize an empty completion set")
    ordered = sorted(lengths)

    def nearest_rank(fraction: float) -> int:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": nearest_rank(0.50),
        "p90": nearest_rank(0.90),
        "p99": nearest_rank(0.99),
        "max": ordered[-1],
        "cap": cap,
        "cap_hit_count": sum(length == cap for length in ordered),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    workers = (
        []
        if args.skip_generation
        else resolve_rollout_gpus(
            args.worker_gpus,
            cuda_visible_devices=visible,
            coordinator_device="cuda:0",
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed_tokens = tokenizer.encode("Solve the problem carefully and give the final answer.", add_special_tokens=True)
    training_tokens = repeated_tokens(seed_tokens, args.sequence_length)
    prompts = [tokenizer.encode(f"Question {index}: What is {index} plus {index + 1}?", add_special_tokens=True) for index in range(args.prompt_count)]

    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    load_started = time.monotonic()
    device_map = None if args.device_map.lower() == "none" else args.device_map
    max_memory = {index: args.max_memory_per_gpu for index in range(torch.cuda.device_count())} if device_map is not None and args.max_memory_per_gpu else None
    backend = LocalBackend(
        device="cuda:0",
        dtype=torch.bfloat16,
        use_lora=True,
        sampler="hf",
        forward_microbatch_max_datums=args.forward_microbatch_max_datums,
        forward_microbatch_max_tokens=args.forward_microbatch_max_tokens,
        target_logprob_chunk_size=args.target_logprob_chunk_size,
        device_map=device_map,
        max_memory=max_memory,
    )
    backend.setup(
        model=args.model,
        lora=LoRAConfig(
            rank=8,
            alpha=16,
            dropout=0.0,
            seed=42,
            train_attn=True,
            train_mlp=True,
            train_unembed=False,
        ),
    )
    model = backend._require_model()
    if (
        args.gradient_checkpointing
        or args.gradient_checkpoint_layers is not None
        or args.gradient_checkpoint_full_attention_layers is not None
    ):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        if not getattr(model, "is_gradient_checkpointing", False):
            raise RuntimeError("gradient checkpointing was requested but did not activate")
        if (
            args.gradient_checkpoint_layers is not None
            or args.gradient_checkpoint_full_attention_layers is not None
        ):
            components = _selected_token_components(model)
            if components is None or not hasattr(components.backbone, "layers"):
                raise RuntimeError("selective checkpointing could not resolve backbone layers")
            layers = list(components.backbone.layers)
            if args.gradient_checkpoint_layers is not None and args.gradient_checkpoint_layers > len(layers):
                raise ValueError(
                    f"requested {args.gradient_checkpoint_layers} checkpoint layers but model has {len(layers)}"
                )
            if args.gradient_checkpoint_layers is not None:
                selected_indices = set(range(args.gradient_checkpoint_layers))
            else:
                full_attention_indices = [
                    index
                    for index, layer_type in enumerate(components.backbone.config.layer_types)
                    if layer_type == "full_attention"
                ]
                assert args.gradient_checkpoint_full_attention_layers is not None
                if args.gradient_checkpoint_full_attention_layers > len(full_attention_indices):
                    raise ValueError(
                        "requested "
                        f"{args.gradient_checkpoint_full_attention_layers} full-attention checkpoint layers "
                        f"but model has {len(full_attention_indices)}"
                    )
                selected_indices = set(
                    full_attention_indices[: args.gradient_checkpoint_full_attention_layers]
                )
            for index, layer in enumerate(layers):
                layer.gradient_checkpointing = index in selected_indices
    load_seconds = time.monotonic() - load_started
    training_metrics = asyncio.run(
        train_probe(
            backend,
            training_tokens,
            datum_count=args.training_datums,
            step_count=args.training_steps,
        )
    )

    if args.skip_generation:
        result = {
            "schema_version": 4,
            "model": args.model,
            "cuda_visible_devices": visible,
            "load_seconds": load_seconds,
            "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
            "gradient_checkpoint_layers": args.gradient_checkpoint_layers,
            "gradient_checkpoint_full_attention_layers": args.gradient_checkpoint_full_attention_layers,
            "device_map": device_map,
            "max_memory": max_memory,
            "target_logprob_chunk_size": args.target_logprob_chunk_size,
            "training": training_metrics,
            "generation": None,
            "outcome": "ok",
        }
        backend.shutdown()
        result_path = args.output_dir / "result.json"
        serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
        result_path.write_text(serialized, encoding="utf-8")
        print("CTM_QWEN35_PARALLEL_BENCHMARK=" + json.dumps(result, sort_keys=True), flush=True)
        return 0

    adapter_dir = args.output_dir / "adapter-v1"
    backend._require_model().save_pretrained(adapter_dir)
    pool = RolloutWorkerPool(
        model=args.model,
        gpus=workers,
        enable_lora=True,
        status_dir=args.output_dir / "workers",
        engine_kwargs={
            "gpu_memory_utilization": args.worker_gpu_mem_util,
            "max_model_len": 32768,
            "max_num_seqs": 256,
            "language_model_only": True,
        },
    )

    result: dict[str, object] = {
        "schema_version": 3,
        "model": args.model,
        "cuda_visible_devices": visible,
        "workers": [worker.as_dict() for worker in workers],
        "load_seconds": load_seconds,
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "gradient_checkpoint_layers": args.gradient_checkpoint_layers,
        "gradient_checkpoint_full_attention_layers": args.gradient_checkpoint_full_attention_layers,
        "device_map": device_map,
        "max_memory": max_memory,
        "target_logprob_chunk_size": args.target_logprob_chunk_size,
        "training": training_metrics,
        "generation": {
            "prompt_count": args.prompt_count,
            "samples_per_prompt": args.samples_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "warmup_prompt_count": min(args.generation_warmup_prompts, len(prompts)),
            "ignore_eos": args.ignore_eos,
            "worker_gpu_mem_util": args.worker_gpu_mem_util,
        },
    }
    try:
        started = time.monotonic()
        pool.start()
        result["worker_start_seconds"] = time.monotonic() - started
        started = time.monotonic()
        pool.publish_adapter_sync(adapter_dir, version=1)
        result["adapter_publish_seconds"] = time.monotonic() - started
        warmup_prompts = prompts[: args.generation_warmup_prompts]
        if warmup_prompts:
            warmup_samples_per_prompt = math.ceil(pool.worker_count / len(warmup_prompts))
            started = time.monotonic()
            warmup_sampled = asyncio.run(
                pool.sample_batch(
                    warmup_prompts,
                    max_tokens=min(args.max_new_tokens, 32),
                    temperature=0.7,
                    stop=[],
                    num_samples=warmup_samples_per_prompt,
                    use_base=False,
                )
            )
            result["generation"]["warmup_seconds"] = time.monotonic() - started
            result["generation"]["warmup_samples_per_prompt"] = warmup_samples_per_prompt
            result["generation"]["warmup_output_tokens"] = sum(len(sequence.tokens) for sequences in warmup_sampled for sequence in sequences)
        started = time.monotonic()
        sampled = asyncio.run(
            pool.sample_batch(
                prompts,
                max_tokens=args.max_new_tokens,
                temperature=0.7,
                stop=[],
                num_samples=args.samples_per_prompt,
                use_base=False,
                ignore_eos=args.ignore_eos,
            )
        )
        generation_seconds = time.monotonic() - started
        flattened = [sequence for sequences in sampled for sequence in sequences]
        expected = args.prompt_count * args.samples_per_prompt
        if len(flattened) != expected:
            raise RuntimeError(f"rollout pool returned {len(flattened)}/{expected} sequences")
        if any(sequence.logprobs is None or len(sequence.logprobs) != len(sequence.tokens) for sequence in flattened):
            raise RuntimeError("one or more rollouts has missing/misaligned sampled-token logprobs")
        output_tokens = sum(len(sequence.tokens) for sequence in flattened)
        lengths = [len(sequence.tokens) for sequence in flattened]
        result["generation"].update(
            {
                "seconds": generation_seconds,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / generation_seconds,
                "sequence_count": len(flattened),
                "lengths": length_summary(lengths, cap=args.max_new_tokens),
            }
        )
        result["worker_status_dir"] = str(pool.status_dir)
        result["outcome"] = "ok"
    finally:
        pool.shutdown()
        backend.shutdown()

    result_path = args.output_dir / "result.json"
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    result_path.write_text(serialized, encoding="utf-8")
    print("CTM_QWEN35_PARALLEL_BENCHMARK=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
