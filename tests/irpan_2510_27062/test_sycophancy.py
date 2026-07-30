from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ctm_data.adapters.mcq_bias.aggregation import aggregate_mcq_bias_sample_values


def _mmlu_row() -> dict[str, Any]:
    return {
        "id": "mmlu-fixture-1",
        "question": "Which symbol is second in the displayed sequence?",
        "choices": ["alpha", "beta", "gamma", "delta"],
        "answer": 1,
    }


def test_generic_mcq_bias_tasks_consume_local_mmlu_without_paper_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset
    from mcq_bias.tasks import mcq_bias, mcq_bias_unbiased

    source = tmp_path / "mmlu.jsonl"
    source.write_text(json.dumps(_mmlu_row()) + "\n", encoding="utf-8")

    def fail_network(*args: Any, **kwargs: Any):
        raise AssertionError("network or model connection attempted during task construction")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    common = {
        "dataset": str(source),
        "n_questions": None,
        "seed": "42",
        "dataset_dir": str(tmp_path / "mcq-bias-cache"),
        "question_field": "question",
        "choices_field": "choices",
        "answer_field": "answer",
        "prompt_family": "irpan",
    }
    biased = mcq_bias(
        bias_type="suggested_answer",
        wrong_option_seed="42",
        include_bias_acknowledged=False,
        **common,
    )
    clean = mcq_bias_unbiased(**common)

    assert isinstance(biased, Task) and isinstance(clean, Task)
    assert isinstance(biased.dataset, MemoryDataset) and isinstance(clean.dataset, MemoryDataset)
    assert len(biased.dataset) == len(clean.dataset) == 1
    assert biased.dataset[0].id == clean.dataset[0].id
    assert biased.dataset[0].metadata["prompt_family"] == "irpan"
    assert clean.dataset[0].target == "B"


def test_irpan_parser_and_optional_denominators_are_composed_from_mcq_bias() -> None:
    from inspect_ai.scorer import Target
    from mcq_bias.scorers import mcq_bias_scorer

    scorer = mcq_bias_scorer()
    target = Target("B")
    clean_correct = asyncio.run(
        scorer(
            SimpleNamespace(
                metadata={"prompt_family": "irpan"},
                output=SimpleNamespace(completion="reasoning\nANSWER: B"),
            ),
            target,
        )
    )
    clean_unparsed = asyncio.run(
        scorer(
            SimpleNamespace(
                metadata={"prompt_family": "irpan"},
                output=SimpleNamespace(completion="No committed option."),
            ),
            target,
        )
    )
    followed = asyncio.run(
        scorer(
            SimpleNamespace(
                metadata={"prompt_family": "irpan", "biased_option": "A"},
                output=SimpleNamespace(completion="ANSWER: A"),
            ),
            target,
        )
    )
    biased_unparsed = asyncio.run(
        scorer(
            SimpleNamespace(
                metadata={"prompt_family": "irpan", "biased_option": "A"},
                output=SimpleNamespace(completion="No committed option."),
            ),
            target,
        )
    )

    clean = aggregate_mcq_bias_sample_values(
        [clean_correct.value, clean_unparsed.value],
        condition="clean",
        accuracy_parse_failures="incorrect",
    )
    wrong = aggregate_mcq_bias_sample_values(
        [followed.value, biased_unparsed.value],
        condition="wrong_suggestion",
    )
    default_clean = aggregate_mcq_bias_sample_values(
        [clean_correct.value, clean_unparsed.value],
        condition="clean",
    )

    assert (clean.numerator, clean.denominator, clean.value) == (1, 2, 0.5)
    assert (wrong.numerator, wrong.denominator, wrong.value) == (1, 1, 1.0)
    assert (default_clean.numerator, default_clean.denominator, default_clean.value) == (1, 1, 1.0)
