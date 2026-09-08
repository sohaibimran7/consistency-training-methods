"""Compare a Qwen3.5 LoRA effect through HF and the actual rollout workers.

``runtime_parity`` exercises vLLM's OpenAI server.  On-policy training uses
the in-process ``VLLMSampler.score_completions`` path instead.  This small,
single-use probe holds the model, prompts, candidate tokens, and raw/translated
LoRA snapshot fixed, and changes only that transport boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ctm.backends.local.rollout_workers import RolloutWorkerPool, resolve_rollout_gpus
from experiments.act_repair_gate.runtime_parity import (
    _chat_prompt_ids,
    _effect_vector,
    _letter_candidate_token_ids,
    _select_scores,
    _top_token_ids,
    compare_effects,
    load_prompts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--raw-adapter", type=Path, required=True)
    parser.add_argument("--vllm-adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-gpus", default="1,2,3")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--top-token-count", type=int, default=16)
    parser.add_argument("--worker-gpu-mem-util", type=float, default=0.75)
    parser.add_argument("--worker-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--adapter-version", type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.top_token_count < 1 or args.adapter_version < 1:
        parser.error("samples, top-token-count, and adapter-version must be positive")
    if not 0 < args.worker_gpu_mem_util <= 1:
        parser.error("worker-gpu-mem-util must be in (0, 1]")
    for field in ("raw_adapter", "vllm_adapter", "data"):
        if not getattr(args, field).exists():
            parser.error(f"--{field.replace('_', '-')} does not exist: {getattr(args, field)}")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    return args


def _hf_dense_scores(
    *,
    model: Any,
    tokenizer: Any,
    prompt_token_ids: Sequence[Sequence[int]],
    use_base: bool,
) -> list[dict[int, float]]:
    """Score one unpadded prompt at a time, matching worker input geometry."""

    results: list[dict[int, float]] = []
    context = model.disable_adapter() if use_base else contextlib.nullcontext()
    with context, torch.inference_mode():
        for prompt in prompt_token_ids:
            input_ids = torch.tensor([list(prompt)], dtype=torch.long, device="cuda:0")
            attention_mask = torch.ones_like(input_ids)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, logits_to_keep=1).logits
            results.append({index: float(value) for index, value in enumerate(logits[0, -1].float().cpu().tolist())})
    return results


def _relative_vector(
    scores: Sequence[dict[int, float]], references: Sequence[int], requested: Sequence[Sequence[int]]
) -> list[float]:
    values: list[float] = []
    for row, reference, token_ids in zip(scores, references, requested, strict=True):
        values.extend(float(row[token]) - float(row[reference]) for token in token_ids)
    return values


def _worker_score_rows(
    *,
    pool: RolloutWorkerPool,
    prompt_ids: Sequence[Sequence[int]],
    requested: Sequence[Sequence[int]],
    use_base: bool,
) -> list[dict[int, float]]:
    flat_prompts: list[list[int]] = []
    flat_completions: list[list[int]] = []
    shape: list[list[int]] = []
    for prompt, token_ids in zip(prompt_ids, requested, strict=True):
        ids = [int(token) for token in token_ids]
        shape.append(ids)
        flat_prompts.extend([list(prompt)] * len(ids))
        flat_completions.extend([[token] for token in ids])
    scored = asyncio.run(pool.score_completions(flat_prompts, flat_completions, use_base=use_base))
    if len(scored) != len(flat_completions) or any(len(row) != 1 for row in scored):
        raise RuntimeError("rollout worker returned misaligned one-token scores")
    rows: list[dict[int, float]] = []
    offset = 0
    for ids in shape:
        row = {token: float(scored[offset + index][0]) for index, token in enumerate(ids)}
        rows.append(row)
        offset += len(ids)
    return rows


def _assert_finite(name: str, values: Sequence[float]) -> None:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{name} must be non-empty and finite")


def main() -> int:
    args = _parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    workers = resolve_rollout_gpus(
        args.worker_gpus,
        cuda_visible_devices=visible,
        coordinator_device="cuda:0",
    )
    prompts = load_prompts(args.data, limit=args.samples)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_ids = [_chat_prompt_ids(tokenizer, prompt.messages) for prompt in prompts]

    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda:0")
    model = PeftModel.from_pretrained(base_model, str(args.raw_adapter), is_trainable=False)
    model.eval()
    hf_base_dense = _hf_dense_scores(model=model, tokenizer=tokenizer, prompt_token_ids=prompt_ids, use_base=True)
    top_ids = _top_token_ids(hf_base_dense, count=args.top_token_count)
    candidate_ids = {token for values in _letter_candidate_token_ids(tokenizer).values() for token in values}
    requested = [list(dict.fromkeys([*top, *sorted(candidate_ids)])) for top in top_ids]
    references = [top[0] for top in top_ids]
    hf_base = _select_scores(hf_base_dense, requested)
    hf_policy = _select_scores(
        _hf_dense_scores(model=model, tokenizer=tokenizer, prompt_token_ids=prompt_ids, use_base=False),
        requested,
    )
    del model, base_model, hf_base_dense
    gc.collect()
    torch.cuda.empty_cache()

    args.output.mkdir(parents=True, exist_ok=False)
    pool = RolloutWorkerPool(
        model=args.model,
        gpus=workers,
        enable_lora=True,
        status_dir=args.output / "workers",
        engine_kwargs={
            "gpu_memory_utilization": args.worker_gpu_mem_util,
            "max_model_len": 32768,
            "max_num_seqs": 256,
            "max_num_batched_tokens": args.worker_max_num_batched_tokens,
            "language_model_only": True,
            "logprobs_mode": "processed_logprobs",
        },
    )
    try:
        pool.start()
        pool.publish_adapter_sync(args.vllm_adapter, version=args.adapter_version)
        worker_base = _worker_score_rows(pool=pool, prompt_ids=prompt_ids, requested=requested, use_base=True)
        worker_policy = _worker_score_rows(pool=pool, prompt_ids=prompt_ids, requested=requested, use_base=False)
    finally:
        pool.shutdown()

    hf_effect: list[float] = []
    worker_effect: list[float] = []
    for index, token_ids in enumerate(requested):
        hf_effect.extend(
            _effect_vector(hf_base[index], hf_policy[index], reference_token_id=references[index], token_ids=token_ids)
        )
        worker_effect.extend(
            _effect_vector(
                worker_base[index], worker_policy[index], reference_token_id=references[index], token_ids=token_ids
            )
        )
    hf_base_relative = _relative_vector(hf_base, references, requested)
    worker_base_relative = _relative_vector(worker_base, references, requested)
    hf_policy_relative = _relative_vector(hf_policy, references, requested)
    worker_policy_relative = _relative_vector(worker_policy, references, requested)
    for name, values in {
        "hf_effect": hf_effect,
        "worker_effect": worker_effect,
        "hf_base_relative": hf_base_relative,
        "worker_base_relative": worker_base_relative,
        "hf_policy_relative": hf_policy_relative,
        "worker_policy_relative": worker_policy_relative,
    }.items():
        _assert_finite(name, values)

    result = {
        "schema": "qwen35-direct-rollout-worker-parity-v1",
        "model": args.model,
        "cuda_visible_devices": visible,
        "worker_gpus": [worker.as_dict() for worker in workers],
        "adapter_version": args.adapter_version,
        "raw_adapter": str(args.raw_adapter.resolve()),
        "vllm_adapter": str(args.vllm_adapter.resolve()),
        "question_ids": [prompt.question_id for prompt in prompts],
        "prompt_lengths": [len(prompt) for prompt in prompt_ids],
        "candidate_count_per_prompt": [len(row) for row in requested],
        "comparison": "adapter-minus-base relative next-token score; HF unpadded versus direct worker prompt_logprobs",
        "effect": compare_effects(hf_effect, worker_effect),
        "base_relative": compare_effects(hf_base_relative, worker_base_relative),
        "policy_relative": compare_effects(hf_policy_relative, worker_policy_relative),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("CTM_QWEN35_DIRECT_ROLLOUT_WORKER_PARITY=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
