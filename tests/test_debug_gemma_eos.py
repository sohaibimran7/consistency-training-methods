"""Unit contracts for the uncapped Gemma EOS diagnostic selector."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[1] / "infra/isambard/debug_gemma_eos.py"
    spec = importlib.util.spec_from_file_location("debug_gemma_eos_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_question_id_parser_requires_nonempty_unique_json_strings():
    module = _load_module()

    assert module._parse_question_ids('["a", "b"]') == ("a", "b")
    for value in ("[]", '["a", "a"]', '["a", 2]', "not-json"):
        with pytest.raises(Exception, match="question-ids"):
            module._parse_question_ids(value)


def test_selector_preserves_frozen_source_order_for_explicit_failed_ids():
    module = _load_module()
    source_ids = tuple(f"id-{index:03d}" for index in range(100))

    selected = module._selected_ids(
        source_ids,
        task_index=3,
        rank=None,
        question_ids=("id-019", "id-003"),
        shard_ids=lambda *_args, **_kwargs: (),
    )

    assert selected == ("id-003", "id-019")
    with pytest.raises(module.GemmaEOSDebugError, match="outside the scored"):
        module._selected_ids(
            source_ids,
            task_index=3,
            rank=None,
            question_ids=("id-050",),
            shard_ids=lambda *_args, **_kwargs: (),
        )


def test_rank_selector_uses_the_existing_rotated_shard_function_without_length_cap():
    module = _load_module()
    calls = []

    def shard_ids(ids, *, task_index, shard_index):
        calls.append((ids, task_index, shard_index))
        return ("id-001", "id-017", "id-049")

    selected = module._selected_ids(
        tuple(f"id-{index:03d}" for index in range(100)),
        task_index=3,
        rank=5,
        question_ids=None,
        shard_ids=shard_ids,
    )

    assert selected == ("id-001", "id-017", "id-049")
    assert calls[0][1:] == (3, 5)


def test_source_declares_telemetry_cadence_not_a_generation_stopping_rule():
    path = Path(__file__).parents[1] / "infra/isambard/debug_gemma_eos.py"
    source = path.read_text(encoding="utf-8")

    assert "_eos_only_model_generate" in source
    assert "completed_after_model_eos" in source
    assert "telemetry cadence" in source
    assert "max_new_tokens" not in source
    assert "max_tokens" not in source
