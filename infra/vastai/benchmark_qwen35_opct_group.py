"""Non-production full-group OPCT benchmark on one coordinator plus vLLM workers."""

# The standalone Vast harness adds the checkout root before importing project
# modules so it can run without installing the in-flight repository snapshot.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from transformers import AutoTokenizer

from ctm.backends.base import SampledSequence
from ctm.backends.local.engine import LocalBackend, _selected_token_components
from ctm.backends.local.qwen35_vllm_compat import (
    WORKER_PARITY_ATTESTATION_NAME,
    file_sha256,
    write_qwen35_rollout_worker_parity_attestation,
)
from ctm.backends.local.rollout_workers import RolloutParallelBackend, resolve_rollout_gpus
from ctm.core.config import AdamConfig, LoRAConfig
from ctm.training.opct import OPCTConfig, OPCTGenerationConfig, OPCTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--worker-gpus", default="1,2,3")
    parser.add_argument(
        "--worker-seed-base",
        type=int,
        required=True,
        help="vLLM engine seed base; worker i receives base+i",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--rollouts-per-prompt", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=20480)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--target-logprob-chunk-size", type=int, default=2048)
    parser.add_argument("--worker-gpu-mem-util", type=float, default=0.75)
    parser.add_argument(
        "--worker-max-model-len",
        type=int,
        default=32768,
        help="Exact max_model_len passed to every vLLM rollout worker",
    )
    parser.add_argument(
        "--worker-max-num-seqs",
        type=int,
        default=256,
        help="Exact max_num_seqs passed to every vLLM rollout worker",
    )
    parser.add_argument("--worker-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--worker-gdn-prefill-backend",
        choices=["flashinfer", "triton"],
        default=None,
        help="Exact optional vLLM GDN prefill backend passed to every rollout worker",
    )
    parser.add_argument(
        "--post-update-min-effect",
        type=float,
        default=1e-5,
        help=(
            "Minimum absolute worker-v2 minus worker-base logprob effect required by the "
            "post-update Qwen3.5 LoRA transport gate"
        ),
    )
    parser.add_argument(
        "--sampled-input",
        type=Path,
        help="Reuse a sampled.json written by an earlier run and skip generation",
    )
    parser.add_argument("--ignore-eos", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.prompt_count,
            args.rollouts_per_prompt,
            args.max_new_tokens,
            args.target_logprob_chunk_size,
            args.worker_max_model_len,
            args.worker_max_num_seqs,
            args.worker_max_num_batched_tokens,
        )
        < 1
    ):
        parser.error("counts, token limits, and chunk size must be positive")
    if not 0 < args.temperature:
        parser.error("--temperature must be positive")
    if not 0 < args.worker_gpu_mem_util <= 1:
        parser.error("--worker-gpu-mem-util must be in (0, 1]")
    if not math.isfinite(args.post_update_min_effect) or args.post_update_min_effect <= 0:
        parser.error("--post-update-min-effect must be finite and positive")
    if not 0 <= args.worker_seed_base <= 2**31 - 1:
        parser.error("--worker-seed-base must be in [0, 2147483647]")
    return args


def _prompt(tokenizer, text: str) -> types.ModelInput:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"expected one encoded prompt, got batch size {len(encoded)}")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not all(isinstance(token, int) for token in encoded):
        raise TypeError(f"chat template returned unsupported token container {type(encoded).__name__}")
    return types.ModelInput.from_ints(tokens=encoded)


def _difference_summary(differences: list[float]) -> dict[str, object]:
    ordered = sorted(differences)
    p99_index = max(0, (99 * len(ordered) + 99) // 100 - 1) if ordered else 0
    return {
        "token_count": len(differences),
        "max_abs_difference": max(differences, default=0.0),
        "mean_abs_difference": statistics.fmean(differences) if differences else 0.0,
        "p99_abs_difference": ordered[p99_index] if ordered else 0.0,
        "first_32_abs_differences": differences[:32],
        "last_32_abs_differences": differences[-32:],
    }


def _flat_score_differences(
    left: list[list[float]],
    right: list[list[float]],
    *,
    label: str,
) -> list[float]:
    """Return aligned score differences, rejecting a silently truncated audit."""

    if len(left) != len(right):
        raise RuntimeError(f"{label} row count differs: {len(left)} and {len(right)}")
    differences = []
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        if len(left_row) != len(right_row):
            raise RuntimeError(f"{label} token count differs at row {index}: {len(left_row)} and {len(right_row)}")
        differences.extend(float(a) - float(b) for a, b in zip(left_row, right_row))
    if not differences:
        raise RuntimeError(f"{label} has no scored token positions")
    if not all(math.isfinite(value) for value in differences):
        raise RuntimeError(f"{label} contains a non-finite score")
    return differences


def _effect_parity_summary(
    *,
    worker_policy: list[list[float]],
    worker_base: list[list[float]],
    coordinator_policy: list[list[float]],
    coordinator_base: list[list[float]],
) -> dict[str, object]:
    """Compare the policy-minus-base LoRA effect, not just raw logprobs."""

    worker_effect = _flat_score_differences(worker_policy, worker_base, label="worker policy/base audit")
    coordinator_effect = _flat_score_differences(
        coordinator_policy,
        coordinator_base,
        label="coordinator policy/base audit",
    )
    effect_difference = _flat_score_differences(
        [worker_effect],
        [coordinator_effect],
        label="worker/coordinator LoRA-effect audit",
    )
    worker_norm = math.sqrt(sum(value * value for value in worker_effect))
    coordinator_norm = math.sqrt(sum(value * value for value in coordinator_effect))
    cosine = (
        sum(left * right for left, right in zip(worker_effect, coordinator_effect)) / (worker_norm * coordinator_norm)
        if worker_norm and coordinator_norm
        else None
    )
    return {
        "worker_v2_minus_base": _difference_summary([abs(value) for value in worker_effect]),
        "coordinator_updated_minus_base": _difference_summary([abs(value) for value in coordinator_effect]),
        "worker_minus_coordinator_effect": _difference_summary([abs(value) for value in effect_difference]),
        "cosine_similarity": cosine,
    }


def _require_post_update_effect_parity(
    summary: dict[str, object],
    *,
    min_effect: float,
    label: str,
) -> None:
    """Fail closed when one actual rollout worker disagrees with HF/PEFT.

    The test deliberately validates the *effect* (policy minus frozen base),
    rather than merely observing an acknowledged adapter version.  A raw
    Qwen3.5 PEFT adapter used to be accepted by vLLM while changing no worker
    logits at all.
    """

    worker_effect = summary["worker_v2_minus_base"]
    coordinator_effect = summary["coordinator_updated_minus_base"]
    if not isinstance(worker_effect, dict) or not isinstance(coordinator_effect, dict):
        raise RuntimeError(f"{label}: malformed post-update effect audit")
    worker_effect_max = float(worker_effect["max_abs_difference"])
    coordinator_effect_max = float(coordinator_effect["max_abs_difference"])
    cosine = summary["cosine_similarity"]
    if worker_effect_max < min_effect:
        raise RuntimeError(
            f"{label}: worker v2 has no measurable LoRA effect relative to worker base: "
            f"max_abs={worker_effect_max:.3e}, required>={min_effect:.3e}. "
            "This is the historical Qwen3.5 vLLM no-op failure."
        )
    if coordinator_effect_max < min_effect:
        raise RuntimeError(
            f"{label}: coordinator has no measurable LoRA effect relative to its frozen base: "
            f"max_abs={coordinator_effect_max:.3e}, required>={min_effect:.3e}. "
            "The synthetic update did not make a usable transport probe."
        )
    if cosine is None or float(cosine) < 0.90:
        raise RuntimeError(
            f"{label}: worker/coordinator LoRA effects disagree: " f"cosine_similarity={cosine!r}, required>=0.90"
        )


def _post_update_probe_source_indices(
    candidate_indices: list[int],
    *,
    worker_count: int,
    anchor_index: int,
) -> list[int]:
    """Schedule a nonzero post-update anchor on every rollout worker.

    ``RolloutWorkerPool.score_completions`` assigns flat score row ``i`` to
    worker ``i % worker_count``.  Repeating the strongest coordinator-side
    candidate for the first ``worker_count`` rows therefore gives every engine
    the same independently checkable post-update prompt/completion.  The
    remaining small candidate set retains prompt diversity for the aggregate
    report without changing any vLLM production option.
    """

    if worker_count < 1:
        raise ValueError("post-update worker coverage requires at least one rollout worker")
    if not candidate_indices:
        raise ValueError("post-update worker coverage requires at least one candidate probe")
    if len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("post-update candidate probe indices must be unique")
    if anchor_index not in candidate_indices:
        raise ValueError("post-update anchor must be one of the candidate probe indices")
    return [anchor_index] * worker_count + [index for index in candidate_indices if index != anchor_index]


def _rollout_rng_diversity_summary(
    sampled: list[list[SampledSequence]],
    *,
    worker_count: int,
) -> dict[str, object]:
    """Detect the systematic equal-lane signature of cloned worker RNGs.

    Global sample slots are assigned round-robin.  For each prompt, local lane
    ``j`` therefore compares slots ``j*W .. j*W+W-1`` generated by the W
    engines from the same request shape. Natural collisions are allowed; a
    run fails only if *every* complete lane group is byte-identical.
    """

    if worker_count < 2:
        raise ValueError("RNG diversity probe requires at least two rollout workers")
    lane_groups: list[dict[str, object]] = []
    for prompt_index, prompt_samples in enumerate(sampled):
        complete_groups = len(prompt_samples) // worker_count
        for local_lane in range(complete_groups):
            start = local_lane * worker_count
            group = prompt_samples[start : start + worker_count]
            hashes = [
                hashlib.sha256(
                    json.dumps(sequence.tokens, separators=(",", ":")).encode("ascii")
                ).hexdigest()
                for sequence in group
            ]
            lane_groups.append(
                {
                    "prompt_index": prompt_index,
                    "local_lane": local_lane,
                    "distinct_completion_hashes": len(set(hashes)),
                    "all_workers_identical": len(set(hashes)) == 1,
                }
            )
    if not lane_groups:
        raise RuntimeError(
            "RNG diversity probe needs at least one prompt with one sample per worker; "
            f"workers={worker_count}"
        )
    identical = sum(bool(group["all_workers_identical"]) for group in lane_groups)
    return {
        "schema": "vllm-worker-rng-diversity-v1",
        "worker_count": worker_count,
        "matched_lane_groups": len(lane_groups),
        "all_worker_identical_lane_groups": identical,
        "passed": identical < len(lane_groups),
        "lane_groups": lane_groups,
    }


def _require_rollout_rng_diversity(summary: dict[str, object]) -> None:
    if not summary.get("passed"):
        raise RuntimeError(
            "all matched same-prompt rollout lanes were identical across vLLM workers; "
            "the engines appear to share one RNG seed/stream"
        )


def _post_update_worker_probe_alignment(
    *,
    worker_policy: list[list[float]],
    worker_base: list[list[float]],
    coordinator_policy: list[list[float]],
    coordinator_base: list[list[float]],
    worker_gpus: list[dict[str, object]],
    sources: list[dict[str, int]],
    min_effect: float,
) -> dict[str, list[dict[str, object]]]:
    """Report and validate direct worker/HF effects by worker and probe row.

    Every row is scored by exactly one rollout engine.  Checking each row
    makes a successful aggregate unable to hide one worker that accepted a
    snapshot but did not apply it.  ``sources`` preserves the original sampled
    prompt and rollout identity when an anchor is intentionally duplicated.
    """

    row_count = len(worker_policy)
    if len(worker_gpus) < 1:
        raise ValueError("post-update worker audit has no rollout workers")
    lengths = {
        "worker_base": len(worker_base),
        "coordinator_policy": len(coordinator_policy),
        "coordinator_base": len(coordinator_base),
        "sources": len(sources),
    }
    mismatches = {name: length for name, length in lengths.items() if length != row_count}
    if mismatches:
        raise RuntimeError(
            "post-update worker audit row count differs from worker-policy scores: "
            f"worker_policy={row_count}, {mismatches}"
        )
    if row_count < len(worker_gpus):
        raise RuntimeError(
            "post-update worker audit has fewer probe rows than rollout workers: "
            f"rows={row_count}, workers={len(worker_gpus)}"
        )

    by_worker_probe: list[dict[str, object]] = []
    worker_rows: list[list[int]] = [[] for _ in worker_gpus]
    for probe_index in range(row_count):
        worker_index = probe_index % len(worker_gpus)
        worker_rows[worker_index].append(probe_index)
        effect_parity = _effect_parity_summary(
            worker_policy=[worker_policy[probe_index]],
            worker_base=[worker_base[probe_index]],
            coordinator_policy=[coordinator_policy[probe_index]],
            coordinator_base=[coordinator_base[probe_index]],
        )
        _require_post_update_effect_parity(
            effect_parity,
            min_effect=min_effect,
            label=f"post-update worker {worker_index} probe {probe_index}",
        )
        policy_parity = _difference_summary(
            [
                abs(value)
                for value in _flat_score_differences(
                    [worker_policy[probe_index]],
                    [coordinator_policy[probe_index]],
                    label=f"post-update worker {worker_index} probe {probe_index} policy audit",
                )
            ]
        )
        by_worker_probe.append(
            {
                "probe_index": probe_index,
                "worker_index": worker_index,
                "worker_gpu": dict(worker_gpus[worker_index]),
                **sources[probe_index],
                "completion_token_count": len(worker_policy[probe_index]),
                "policy_parity": policy_parity,
                "effect_parity": effect_parity,
            }
        )

    by_worker: list[dict[str, object]] = []
    for worker_index, row_indices in enumerate(worker_rows):
        if not row_indices:
            raise RuntimeError(f"post-update worker audit did not assign a probe to worker {worker_index}")
        effect_parity = _effect_parity_summary(
            worker_policy=[worker_policy[index] for index in row_indices],
            worker_base=[worker_base[index] for index in row_indices],
            coordinator_policy=[coordinator_policy[index] for index in row_indices],
            coordinator_base=[coordinator_base[index] for index in row_indices],
        )
        _require_post_update_effect_parity(
            effect_parity,
            min_effect=min_effect,
            label=f"post-update worker {worker_index} aggregate",
        )
        policy_parity = _difference_summary(
            [
                abs(value)
                for value in _flat_score_differences(
                    [worker_policy[index] for index in row_indices],
                    [coordinator_policy[index] for index in row_indices],
                    label=f"post-update worker {worker_index} policy audit",
                )
            ]
        )
        by_worker.append(
            {
                "worker_index": worker_index,
                "worker_gpu": dict(worker_gpus[worker_index]),
                "probe_indices": row_indices,
                "source_audit_indices": [sources[index]["source_audit_index"] for index in row_indices],
                "source_prompt_indices": [sources[index]["source_prompt_index"] for index in row_indices],
                "source_rollout_indices": [sources[index]["source_rollout_index"] for index in row_indices],
                "policy_parity": policy_parity,
                "effect_parity": effect_parity,
            }
        )
    return {"by_worker": by_worker, "by_worker_probe": by_worker_probe}


async def run_group(args: argparse.Namespace) -> dict[str, object]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    workers = resolve_rollout_gpus(
        args.worker_gpus,
        cuda_visible_devices=visible,
        coordinator_device="cuda:0",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    training_backend = LocalBackend(
        device="cuda:0",
        dtype=torch.bfloat16,
        use_lora=True,
        sampler="vllm",
        vllm_options={
            "gpu_memory_utilization": 0.34,
            "max_model_len": args.worker_max_model_len,
            "max_num_seqs": args.worker_max_num_seqs,
            "language_model_only": True,
            "logprobs_mode": "processed_logprobs",
        },
        gradient_checkpointing=True,
        gradient_checkpointing_layers=16,
        forward_microbatch_max_datums=8,
        # This is a physical packing budget, not a sequence-length cap. A
        # 20,480-token completion remains one indivisible full transformer
        # forward/backward; shorter rows may share a physical forward.
        forward_microbatch_max_tokens=20480,
        target_logprob_chunk_size=args.target_logprob_chunk_size,
    )
    backend = RolloutParallelBackend(
        training_backend,
        gpus=workers,
        status_dir=args.output_dir / "workers",
        # Bootstrap is deliberately a Python-only escape hatch, never a
        # training CLI flag. This non-production harness is what creates the
        # fixed-token evidence that a normal RMCT backend subsequently requires.
        qwen35_rollout_parity_bootstrap=True,
        worker_vllm_options={
            "gpu_memory_utilization": args.worker_gpu_mem_util,
            "max_model_len": args.worker_max_model_len,
            "max_num_seqs": args.worker_max_num_seqs,
            "max_num_batched_tokens": args.worker_max_num_batched_tokens,
            "language_model_only": True,
            "logprobs_mode": "processed_logprobs",
            "seed": args.worker_seed_base,
            **(
                {"gdn_prefill_backend": args.worker_gdn_prefill_backend}
                if args.worker_gdn_prefill_backend is not None
                else {}
            ),
        },
    )
    load_started = time.monotonic()
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
    setup_seconds = time.monotonic() - load_started
    try:
        trainable_model = training_backend._require_model()
        selected_components = _selected_token_components(trainable_model)
        if selected_components is None:
            causal_lm = trainable_model.get_base_model()
            raise RuntimeError(
                "Qwen3.5 did not resolve the selected-token scoring path: "
                f"class={type(causal_lm).__name__}, "
                f"model_type={getattr(causal_lm.config, 'model_type', None)!r}"
            )
        reference_prompts = [
            _prompt(tokenizer, f"Solve reference arithmetic problem {index}: what is {index} + {index + 1}?")
            for index in range(args.prompt_count)
        ]
        variant_prompts = [
            _prompt(
                tokenizer,
                f"A source insists the answer is {2 * index + 2}. Solve arithmetic problem {index}: "
                f"what is {index} + {index + 1}?",
            )
            for index in range(args.prompt_count)
        ]

        # Real-engine parity gate for the new worker prompt-logprob scorer.
        parity_completions = [
            tokenizer.encode(f"The answer is {2 * index + 1}.", add_special_tokens=False)
            for index in range(min(4, args.prompt_count))
        ]
        parity_prompts = reference_prompts[: len(parity_completions)]
        worker_scores = await backend.base_sampler().score_completions(
            parity_prompts,
            parity_completions,
        )
        coordinator_scores = await asyncio.to_thread(
            training_backend._score_completions,
            parity_prompts,
            parity_completions,
            use_base=True,
        )
        parity_differences = [
            abs(worker - coordinator)
            for worker_row, coordinator_row in zip(worker_scores, coordinator_scores)
            for worker, coordinator in zip(worker_row, coordinator_row)
        ]
        parity = {
            **_difference_summary(parity_differences),
            "worker_scores": worker_scores,
            "coordinator_scores": coordinator_scores,
        }

        torch.cuda.reset_peak_memory_stats(0)
        if args.sampled_input is None:
            generation_started = time.monotonic()
            sampled = await backend.pool.sample_batch(
                [prompt.to_ints() for prompt in variant_prompts],
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                stop=[],
                num_samples=args.rollouts_per_prompt,
                use_base=False,
                ignore_eos=args.ignore_eos,
            )
            generation_seconds = time.monotonic() - generation_started
            sampled_path = args.output_dir / "sampled.json"
            sampled_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "prompt_count": args.prompt_count,
                        "rollouts_per_prompt": args.rollouts_per_prompt,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "ignore_eos": args.ignore_eos,
                        "worker_seed_base": args.worker_seed_base,
                        "sampled": [
                            [{"tokens": sequence.tokens, "logprobs": sequence.logprobs} for sequence in prompt_group]
                            for prompt_group in sampled
                        ],
                    }
                ),
                encoding="utf-8",
            )
            generation_source = "generated"
        else:
            sampled_payload = json.loads(args.sampled_input.read_text(encoding="utf-8"))
            expected = {
                "prompt_count": args.prompt_count,
                "rollouts_per_prompt": args.rollouts_per_prompt,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "ignore_eos": args.ignore_eos,
                "worker_seed_base": args.worker_seed_base,
            }
            mismatches = {
                key: (sampled_payload.get(key), value)
                for key, value in expected.items()
                if sampled_payload.get(key) != value
            }
            if mismatches:
                raise ValueError(f"sampled input metadata does not match benchmark arguments: {mismatches}")
            sampled = [
                [
                    SampledSequence(
                        tokens=[int(token) for token in sequence["tokens"]],
                        logprobs=[float(value) for value in sequence["logprobs"]],
                    )
                    for sequence in prompt_group
                ]
                for prompt_group in sampled_payload["sampled"]
            ]
            generation_seconds = 0.0
            generation_source = str(args.sampled_input)

        rng_diversity = _rollout_rng_diversity_summary(sampled, worker_count=len(workers))
        _require_rollout_rng_diversity(rng_diversity)

        trainer = OPCTTrainer(
            config=OPCTConfig(
                model=args.model,
                generation=OPCTGenerationConfig(
                    rollouts_per_prompt=args.rollouts_per_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                ),
                batch_size=1,
                gradient_accumulation_steps=args.prompt_count,
                kl_coef=2.0,
                kl_discount_factor=0.9,
                loss_fn="importance_sampling",
                rollout_log="none",
            ),
            backend=backend,
        )
        trainer.sampling_client = backend.policy_sampler("benchmark")
        trainer.reference_policy = backend.base_sampler()
        group_batches = [[(index, {})] for index in range(args.prompt_count)]
        pair_groups = [
            [(index, reference_prompts[index], variant_prompts[index])] for index in range(args.prompt_count)
        ]
        sampled_groups = [[sampled[index]] for index in range(args.prompt_count)]
        teacher_started = time.monotonic()
        group_results = await trainer._build_batch_group(
            group_batches,
            prepared_pair_groups=pair_groups,
            sampled_groups=sampled_groups,
        )
        teacher_and_prepare_seconds = time.monotonic() - teacher_started

        audit_variants = []
        audit_completions = []
        audit_sources = []
        for prompt_index, prompt_samples in enumerate(sampled):
            for rollout_index, sequence in enumerate(prompt_samples):
                audit_index = len(audit_variants)
                audit_variants.append(variant_prompts[prompt_index])
                audit_completions.append(list(sequence.tokens))
                audit_sources.append(
                    {
                        "source_audit_index": audit_index,
                        "source_prompt_index": prompt_index,
                        "source_rollout_index": rollout_index,
                    }
                )
        policy_audit_started = time.monotonic()
        worker_policy_scores = await trainer.sampling_client.score_completions(
            audit_variants,
            audit_completions,
        )
        policy_audit_seconds = time.monotonic() - policy_audit_started

        forward_backward_seconds = []
        forward_backward_metrics = []
        coordinator_policy_scores = []
        fused_forward_backward = False
        for datums, kl_metrics, _, rollout_meta in group_results:
            started = time.monotonic()
            uses_fused_scoring = bool(datums) and all(
                "opct_teacher_logprobs" in datum.loss_fn_inputs for datum in datums
            )
            if uses_fused_scoring:
                fused_forward_backward = True
                pending = await backend.submit_opct_forward_backward(
                    datums,
                    behavior_temperature=args.temperature,
                    kl_coef=trainer.config.kl_coef,
                    kl_discount_factor=trainer.config.kl_discount_factor,
                    loss_fn=trainer.config.loss_fn,
                )
            else:
                pending = await backend.submit_forward_backward(
                    datums,
                    loss_fn="importance_sampling",
                )
            output = await pending.result()
            if uses_fused_scoring:
                trainer._complete_fused_batch(
                    datums,
                    output,
                    kl_metrics,
                    rollout_meta,
                )
            forward_backward_seconds.append(time.monotonic() - started)
            forward_backward_metrics.append(output.metrics)
            for datum, logprobs in zip(datums, output.logprobs):
                mask = datum.loss_fn_inputs["mask"].to_torch().bool()
                coordinator_policy_scores.append(logprobs[mask].tolist())

        if len(worker_policy_scores) != len(coordinator_policy_scores):
            raise RuntimeError(
                "worker/coordinator policy audit result count mismatch: "
                f"{len(worker_policy_scores)} and {len(coordinator_policy_scores)}"
            )
        policy_differences = [
            abs(worker - coordinator)
            for worker_row, coordinator_row in zip(worker_policy_scores, coordinator_policy_scores)
            for worker, coordinator in zip(worker_row, coordinator_row)
        ]
        expected_policy_tokens = sum(len(row) for row in worker_policy_scores)
        if len(policy_differences) != expected_policy_tokens:
            raise RuntimeError(
                "worker/coordinator policy audit token count mismatch: "
                f"{len(policy_differences)} and {expected_policy_tokens}"
            )

        optimizer_started = time.monotonic()
        pending_optimizer = await backend.submit_optim_step(
            learning_rate=1e-4,
            adam=AdamConfig(learning_rate=1e-4, lr_schedule="constant"),
        )
        await pending_optimizer.result()
        optimizer_seconds = time.monotonic() - optimizer_started

        # This is the on-policy boundary the capacity probe exists to test:
        # after a real coordinator update, every worker must receive a fresh
        # translated Qwen3.5 compatibility snapshot before it can produce the
        # next policy rollouts.  The initial LoRA B factor is zero, so merely
        # exercising setup/publish-v1 cannot detect a broken dynamic refresh.
        post_update_publish_started = time.monotonic()
        post_update_policy = await backend.refresh_policy_sampler("benchmark_post_update")
        post_update_publish_seconds = time.monotonic() - post_update_publish_started
        post_update_candidate_count = min(4, len(audit_variants))
        if post_update_candidate_count < 1:
            raise RuntimeError("cannot audit post-update Qwen3.5 policy: no non-empty generated completion")
        post_update_candidate_indices = list(range(post_update_candidate_count))
        post_update_candidate_prompts = [audit_variants[index] for index in post_update_candidate_indices]
        post_update_candidate_completions = [audit_completions[index] for index in post_update_candidate_indices]
        # Find a coordinator-side nonzero anchor after the real optimizer step
        # before duplicating it across the worker engines.  The strongest of a
        # small fixed candidate prefix avoids asking every worker to prove
        # parity on a completion that happened to receive no useful gradient.
        post_update_candidate_coordinator_policy_scores = await asyncio.to_thread(
            training_backend._score_completions,
            post_update_candidate_prompts,
            post_update_candidate_completions,
            use_base=False,
        )
        post_update_candidate_coordinator_base_scores = await asyncio.to_thread(
            training_backend._score_completions,
            post_update_candidate_prompts,
            post_update_candidate_completions,
            use_base=True,
        )
        post_update_candidate_effects = []
        for candidate_position, audit_index in enumerate(post_update_candidate_indices):
            effect = _difference_summary(
                [
                    abs(value)
                    for value in _flat_score_differences(
                        [post_update_candidate_coordinator_policy_scores[candidate_position]],
                        [post_update_candidate_coordinator_base_scores[candidate_position]],
                        label=f"post-update coordinator candidate {audit_index} policy/base audit",
                    )
                ]
            )
            post_update_candidate_effects.append(
                {
                    **audit_sources[audit_index],
                    "coordinator_updated_minus_base": effect,
                }
            )
        post_update_anchor_position = max(
            range(post_update_candidate_count),
            key=lambda position: float(
                post_update_candidate_effects[position]["coordinator_updated_minus_base"]["max_abs_difference"]
            ),
        )
        post_update_anchor_audit_index = post_update_candidate_indices[post_update_anchor_position]
        post_update_anchor_effect = post_update_candidate_effects[post_update_anchor_position][
            "coordinator_updated_minus_base"
        ]
        if float(post_update_anchor_effect["max_abs_difference"]) < args.post_update_min_effect:
            raise RuntimeError(
                "post-update coordinator has no measurable LoRA effect relative to its frozen base among "
                f"{post_update_candidate_count} candidate completion(s): "
                f"max_abs={float(post_update_anchor_effect['max_abs_difference']):.3e}, "
                f"required>={args.post_update_min_effect:.3e}. "
                "The synthetic update did not make a usable transport probe."
            )
        post_update_source_indices = _post_update_probe_source_indices(
            post_update_candidate_indices,
            worker_count=len(workers),
            anchor_index=post_update_anchor_audit_index,
        )
        post_update_probe_count = len(post_update_source_indices)
        post_update_prompts = [audit_variants[index] for index in post_update_source_indices]
        post_update_completions = [audit_completions[index] for index in post_update_source_indices]
        post_update_sources = [dict(audit_sources[index]) for index in post_update_source_indices]
        candidate_position_by_audit_index = {
            audit_index: candidate_position
            for candidate_position, audit_index in enumerate(post_update_candidate_indices)
        }
        post_update_worker_policy_scores = await post_update_policy.score_completions(
            post_update_prompts,
            post_update_completions,
        )
        post_update_worker_base_scores = await backend.base_sampler().score_completions(
            post_update_prompts,
            post_update_completions,
        )
        post_update_coordinator_policy_scores = [
            post_update_candidate_coordinator_policy_scores[candidate_position_by_audit_index[index]]
            for index in post_update_source_indices
        ]
        post_update_coordinator_base_scores = [
            post_update_candidate_coordinator_base_scores[candidate_position_by_audit_index[index]]
            for index in post_update_source_indices
        ]
        if len(post_update_worker_policy_scores) != post_update_probe_count:
            raise RuntimeError(
                "post-update worker policy scorer returned "
                f"{len(post_update_worker_policy_scores)}/{post_update_probe_count} result row(s)"
            )
        post_update_policy_parity = _difference_summary(
            [
                abs(value)
                for value in _flat_score_differences(
                    post_update_worker_policy_scores,
                    post_update_coordinator_policy_scores,
                    label="post-update worker/coordinator policy audit",
                )
            ]
        )
        post_update_effect_parity = _effect_parity_summary(
            worker_policy=post_update_worker_policy_scores,
            worker_base=post_update_worker_base_scores,
            coordinator_policy=post_update_coordinator_policy_scores,
            coordinator_base=post_update_coordinator_base_scores,
        )
        _require_post_update_effect_parity(
            post_update_effect_parity,
            min_effect=args.post_update_min_effect,
            label="post-update worker aggregate",
        )
        post_update_per_worker_alignment = _post_update_worker_probe_alignment(
            worker_policy=post_update_worker_policy_scores,
            worker_base=post_update_worker_base_scores,
            coordinator_policy=post_update_coordinator_policy_scores,
            coordinator_base=post_update_coordinator_base_scores,
            worker_gpus=[worker.as_dict() for worker in workers],
            sources=post_update_sources,
            min_effect=args.post_update_min_effect,
        )

        # The benchmark's post-update check is the only Qwen3.5 parity probe
        # that exercises the production teacher-forced worker scorer. Bind the
        # resulting evidence to this exact nonzero v2 raw/translated snapshot,
        # the worker topology, and the vLLM worker options. A later RMCT run
        # refuses to start unless this immutable sidecar validates.
        assert backend.pool is not None
        post_update_adapter_version = backend.pool.adapter_version
        post_update_adapter_root = backend.pool.status_dir / "adapters" / f"v{post_update_adapter_version:08d}"
        score_inputs = {
            "adapter_version": post_update_adapter_version,
            "sources": post_update_sources,
            "prompts": [prompt.to_ints() for prompt in post_update_prompts],
            "completions": post_update_completions,
        }
        attestation_path = args.output_dir / WORKER_PARITY_ATTESTATION_NAME
        write_qwen35_rollout_worker_parity_attestation(
            attestation_path,
            model=args.model,
            raw_adapter=post_update_adapter_root / "raw",
            vllm_adapter=post_update_adapter_root / "vllm_compat",
            adapter_version=post_update_adapter_version,
            worker_gpus=[worker.as_dict() for worker in workers],
            worker_engine_kwargs=backend.pool.engine_kwargs,
            fixed_token_probe={
                "kind": "post-update-worker-score-completions-v1",
                "score_rows": post_update_probe_count,
                "completion_token_count": sum(len(row) for row in post_update_completions),
                "score_inputs_sha256": hashlib.sha256(
                    json.dumps(score_inputs, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            aggregate_effect_parity=post_update_effect_parity,
            per_worker_effect_parity=post_update_per_worker_alignment["by_worker"],
        )

        lengths = [len(sequence.tokens) for prompt_group in sampled for sequence in prompt_group]
        return {
            "schema_version": 4,
            "model": args.model,
            "cuda_visible_devices": visible,
            "worker_gpus": [worker.as_dict() for worker in workers],
            "setup_seconds": setup_seconds,
            "prompt_count": args.prompt_count,
            "rollouts_per_prompt": args.rollouts_per_prompt,
            "rollout_count": len(lengths),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "ignore_eos": args.ignore_eos,
            "target_logprob_chunk_size": args.target_logprob_chunk_size,
            "forward_microbatch_max_tokens": 20480,
            "fused_forward_backward": fused_forward_backward,
            "worker_max_num_batched_tokens": args.worker_max_num_batched_tokens,
            "worker_max_model_len": args.worker_max_model_len,
            "worker_max_num_seqs": args.worker_max_num_seqs,
            "worker_gdn_prefill_backend": args.worker_gdn_prefill_backend,
            "worker_seed_base": args.worker_seed_base,
            "worker_engine_seeds": list(backend.pool.worker_engine_seeds),
            "rollout_rng_diversity": rng_diversity,
            "post_update_min_effect": args.post_update_min_effect,
            "selected_token_path": {
                "backbone_class": type(selected_components.backbone).__name__,
                "lm_head_class": type(selected_components.lm_head).__name__,
            },
            "response_tokens": sum(lengths),
            "response_length_min": min(lengths),
            "response_length_mean": statistics.fmean(lengths),
            "response_length_max": max(lengths),
            "generation_seconds": generation_seconds,
            "generation_source": generation_source,
            "generation_tokens_per_second": (sum(lengths) / generation_seconds if generation_seconds else None),
            "teacher_and_prepare_seconds": teacher_and_prepare_seconds,
            "policy_audit_seconds": policy_audit_seconds,
            "forward_backward_seconds_total": sum(forward_backward_seconds),
            "forward_backward_seconds_mean_per_prompt": statistics.fmean(forward_backward_seconds),
            "optimizer_seconds": optimizer_seconds,
            "post_update_adapter_version": post_update_adapter_version,
            "post_update_adapter_publish_seconds": post_update_publish_seconds,
            "post_update_candidate_score_rows": post_update_candidate_count,
            "post_update_candidate_coordinator_effects": post_update_candidate_effects,
            "post_update_anchor_audit_index": post_update_anchor_audit_index,
            "post_update_anchor_source": audit_sources[post_update_anchor_audit_index],
            "post_update_probe_source_audit_indices": post_update_source_indices,
            "post_update_probe_score_rows": post_update_probe_count,
            "post_update_worker_coverage_required": len(workers),
            "post_update_policy_score_rows": len(post_update_worker_policy_scores),
            "post_update_policy_scored_tokens": sum(len(row) for row in post_update_worker_policy_scores),
            "post_update_worker_hf_policy_parity": post_update_policy_parity,
            "post_update_worker_hf_effect_parity": post_update_effect_parity,
            "post_update_worker_hf_effect_parity_by_worker": post_update_per_worker_alignment["by_worker"],
            "post_update_worker_hf_effect_parity_by_worker_probe": post_update_per_worker_alignment["by_worker_probe"],
            "worker_parity_attestation": {
                "path": str(attestation_path.resolve()),
                "sha256": file_sha256(attestation_path),
            },
            "measured_group_seconds": (
                generation_seconds
                + teacher_and_prepare_seconds
                + sum(forward_backward_seconds)
                + optimizer_seconds
                + post_update_publish_seconds
            ),
            "coordinator_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "coordinator_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "worker_hf_score_parity": parity,
            "actual_action_worker_hf_policy_parity": _difference_summary(policy_differences),
            "forward_backward_metrics_first": forward_backward_metrics[0],
            "forward_backward_metrics_last": forward_backward_metrics[-1],
        }
    finally:
        backend.shutdown()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run_group(args))
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "result.json").write_text(serialized, encoding="utf-8")
    print("CTM_QWEN35_OPCT_GROUP_BENCHMARK=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
