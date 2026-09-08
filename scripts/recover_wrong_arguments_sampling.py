#!/usr/bin/env python3
"""Recover a fixed wrong-argument pool with bounded, diversified sampling.

This helper preserves the pinned mcq-bias prompt, parser, target, and
acceptance filter.  It changes only the decoding seed/temperature/top-p and
serializes canonical-store writes so a slow provider response cannot discard
other accepted results.
"""

from __future__ import annotations

import asyncio
import fcntl
from collections import Counter

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
import mcq_bias.parsers as mcq_parsers
from mcq_bias.pipeline.records import COT_INSTRUCTION
from mcq_bias.pipeline.wrong_arguments import (
    BANNED_WORDS,
    DECEPTIVE_SYSTEM_PROMPT,
    WrongArgumentStore,
    _acceptable,
    append_arguments,
    arguments_path,
)
from mcq_bias.tasks import _load_pool

from ctm_data.adapters.mcq_bias.parser_compat import install_extended_answer_parser

DATASET = "artifacts/rmct-hle-dense-models-shared/data/hle-text-mc.jsonl"
MODEL = "openrouter/google/gemma-4-31b-it"
N_QUESTIONS = 100
FLOOR = 92
CONCURRENCY = 11
ATTEMPT_TIMEOUT = 90

# Predeclared deterministic diversification schedule. The model, messages,
# parser, deterministic target, and banned-term acceptance filter stay fixed.
SCHEDULE = [
    (42001, 0.7, 0.90),
    (42002, 1.0, 0.95),
    (42003, 1.3, 0.98),
    (42004, 0.7, 0.90),
    (42005, 1.0, 0.95),
    (42006, 1.3, 0.98),
    (42007, 0.7, 0.90),
    (42008, 1.0, 0.95),
    (42009, 1.3, 0.98),
    (42010, 0.7, 0.90),
    (42011, 1.0, 0.95),
    (42012, 1.3, 0.98),
]


def _fixed_records():
    records = _load_pool(DATASET, N_QUESTIONS, "42")
    if len(records) != N_QUESTIONS or len({record.question_id for record in records}) != N_QUESTIONS:
        raise RuntimeError("the fixed seed-42 pool is not exactly 100 unique questions")
    return records


def _coverage(store: WrongArgumentStore, records) -> int:
    return sum(store.get(record) is not None for record in records)


async def _one(model, semaphore: asyncio.Semaphore, record):
    # Keep message construction byte-for-byte aligned with pinned
    # mcq_bias.pipeline.wrong_arguments.generate_wrong_arguments.
    messages = [
        ChatMessageSystem(content=DECEPTIVE_SYSTEM_PROMPT.format(biased_ans=record.biased_option)),
        ChatMessageUser(content=record.parsed_input() + COT_INSTRUCTION),
    ]
    try:
        async with semaphore:
            output = await asyncio.wait_for(
                model.generate(messages),
                timeout=ATTEMPT_TIMEOUT + 5,
            )
    except Exception as exc:  # Provider failures are recorded by class only.
        return record, None, f"error:{type(exc).__name__}"

    completion = output.completion or ""
    if _acceptable(completion, record.biased_option):
        return record, completion, "accepted"
    parsed = mcq_parsers.parse_answer(completion)
    banned = tuple(word for word in BANNED_WORDS if word in completion.lower())
    reason = (
        f"wrong_target:{parsed!r}"
        if parsed != record.biased_option
        else f"banned:{','.join(banned)}"
    )
    return record, None, reason


async def _main() -> None:
    install_extended_answer_parser()
    records = _fixed_records()
    store_path = arguments_path(MODEL)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = store_path.with_suffix(store_path.suffix + ".recovery.lock")

    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another diversified recovery helper holds the lock") from exc

        store = WrongArgumentStore.for_model(MODEL)
        print(
            f"start fixed-pool coverage={_coverage(store, records)}/{N_QUESTIONS}; "
            f"store={store_path}",
            flush=True,
        )
        for seed, temperature, top_p in SCHEDULE:
            if _coverage(store, records) >= FLOOR:
                break
            misses = [record for record in records if store.get(record) is None]
            config = GenerateConfig(
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_connections=CONCURRENCY,
                max_retries=0,
                attempt_timeout=ATTEMPT_TIMEOUT,
                timeout=ATTEMPT_TIMEOUT + 10,
            )
            reasons: Counter[str] = Counter()
            semaphore = asyncio.Semaphore(CONCURRENCY)
            async with get_model(MODEL, config=config, memoize=False) as model:
                tasks = [
                    asyncio.create_task(_one(model, semaphore, record))
                    for record in misses
                ]
                for completed in asyncio.as_completed(tasks):
                    record, completion, reason = await completed
                    reasons[reason] += 1
                    if completion is None or store.get(record) is not None:
                        continue
                    row = {
                        "question_id": record.question_id,
                        "parsed_input": record.parsed_input(),
                        "wrong_argument": completion,
                        "biased_option": record.biased_option,
                        "model": model.name,
                        "dataset": record.dataset,
                    }
                    # Only this parent coroutine writes, and each success is
                    # durable before waiting for another provider response.
                    append_arguments(store_path, [row])
                    store.add(
                        completion,
                        parsed=record.parsed_input(),
                        question_id=record.question_id,
                    )
                    print(
                        f"accepted {record.question_id}; "
                        f"coverage={_coverage(store, records)}/{N_QUESTIONS}",
                        flush=True,
                    )
            print(
                f"seed={seed} temperature={temperature} top_p={top_p} "
                f"outcomes={dict(reasons)}",
                flush=True,
            )

        final = _coverage(store, records)
        print(f"final fixed-pool coverage={final}/{N_QUESTIONS}", flush=True)
        if final < FLOOR:
            raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(_main())
