"""Measure exact OPCT forward/backward packing on one training GPU.

This is a non-production capacity benchmark.  It deliberately excludes
generation and frozen-teacher scoring, replays only completion-length shapes,
and keeps OPCT's public one-prompt reductions intact.  Its purpose is to find
the largest safe physical padded-token budget under full-layer gradient
checkpointing without changing the training objective.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import torch
import tinker
from tinker import types
from transformers import AutoTokenizer

from ctm.backends.local.engine import LocalBackend
from ctm.core.config import LoRAConfig
from ctm.training.opct import OPCTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lengths-json", type=Path, required=True)
    parser.add_argument("--packing-budgets", type=int, nargs="+", required=True)
    parser.add_argument("--target-logprob-chunk-size", type=int, default=2048)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value < 1 for value in args.packing_budgets):
        parser.error("packing budgets must be positive")
    return args


def load_lengths(path: Path) -> list[list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("lengths JSON must be a non-empty list of prompt groups")
    groups: list[list[int]] = []
    for group_index, group in enumerate(payload):
        if not isinstance(group, list) or not group:
            raise ValueError(f"length group {group_index} must be non-empty")
        values = [int(value) for value in group]
        if any(value < 2 for value in values):
            raise ValueError(f"length group {group_index} contains a value below two")
        groups.append(values)
    return groups


def make_datums(tokenizer, lengths: list[int]) -> list[object]:
    prompt_tokens = tokenizer.encode("Solve this problem carefully.", add_special_tokens=False)
    completion_cycle = tokenizer.encode(
        " We need to reason carefully and provide the final answer.",
        add_special_tokens=False,
    )
    if not prompt_tokens or not completion_cycle:
        raise RuntimeError("tokenizer produced an empty synthetic probe")

    datums = []
    for total_length in lengths:
        completion_length = max(1, total_length - len(prompt_tokens))
        completion = [completion_cycle[index % len(completion_cycle)] for index in range(completion_length)]
        behavior = [-2.0] * completion_length
        datum = OPCTTrainer._create_datum(
            types.ModelInput.from_ints(tokens=prompt_tokens),
            completion,
            behavior,
        )
        datum.loss_fn_inputs["opct_teacher_logprobs"] = tinker.TensorData.from_torch(
            torch.full((completion_length,), -2.5, dtype=torch.float32)
        )
        datums.append(datum)
    return datums


async def run_group(backend: LocalBackend, datums: list[object]) -> float:
    started = time.monotonic()
    pending = await backend.submit_opct_forward_backward(
        datums,
        behavior_temperature=0.7,
        kl_coef=2.0,
        kl_discount_factor=0.9,
        loss_fn="importance_sampling",
    )
    result = await pending.result()
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    if not math.isfinite(float(result.metrics["loss"])):
        raise RuntimeError("benchmark produced a non-finite loss")
    backend._require_model().zero_grad(set_to_none=True)
    backend._gradient_accumulations = 0
    return elapsed


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    groups = load_lengths(args.lengths_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    backend = LocalBackend(
        device="cuda:0",
        dtype=torch.bfloat16,
        use_lora=True,
        sampler="hf",
        gradient_checkpointing=True,
        gradient_checkpointing_layers=None,
        forward_microbatch_max_datums=8,
        forward_microbatch_max_tokens=args.packing_budgets[0],
        target_logprob_chunk_size=args.target_logprob_chunk_size,
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
    datums_by_group = [make_datums(tokenizer, lengths) for lengths in groups]
    results = []
    for budget in args.packing_budgets:
        backend.forward_microbatch_max_tokens = budget
        backend._require_model().zero_grad(set_to_none=True)
        backend._gradient_accumulations = 0
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        times = []
        chunk_counts = []
        failure = None
        try:
            for datums in datums_by_group:
                token_counts = [len(datum.model_input.to_ints()) for datum in datums]
                chunk_counts.append(len(backend._forward_microbatches(token_counts)))
                times.append(await run_group(backend, datums))
        except torch.OutOfMemoryError as exc:
            failure = "torch.OutOfMemoryError: " + " ".join(str(exc).split())[:500]
            backend._require_model().zero_grad(set_to_none=True)
            backend._gradient_accumulations = 0
            torch.cuda.empty_cache()
        results.append(
            {
                "packing_budget": budget,
                "passed": failure is None,
                "failure": failure,
                "groups_completed": len(times),
                "forward_chunks_completed": sum(chunk_counts[: len(times)]),
                "seconds_total": sum(times),
                "seconds_mean": statistics.fmean(times) if times else None,
                "seconds_median": statistics.median(times) if times else None,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
    return {
        "schema_version": 1,
        "model": args.model,
        "gradient_checkpointing_layers": "all",
        "target_logprob_chunk_size": args.target_logprob_chunk_size,
        "length_groups": groups,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    result = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
