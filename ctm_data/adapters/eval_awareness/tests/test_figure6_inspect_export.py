"""Offline tests for the post-hoc Figure 6 Inspect export."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import ContentReasoning

from ctm_data.adapters.eval_awareness.figure6_inspect_export import (
    SCORE_NAME,
    _generation_record_sha256,
    build_cell_log,
)
from ctm_data.adapters.eval_awareness.figure6_judge import custom_id_for_generation


def _generation(*, task_id: str, condition_id: str) -> dict:
    return {
        "model_key": "qwen32",
        "model_display": "Qwen3-32B",
        "model_id": "Qwen/Qwen3-32B",
        "model_revision": "a" * 40,
        "condition_id": condition_id,
        "config_name": "F1",
        "pair_id": task_id.removesuffix("_safety"),
        "task_id": task_id,
        "valence": "safety",
        "replicate": 1,
        "prompt": f"prompt for {task_id}",
        "response": "<think>reasoning</think>answer",
        "reasoning": "reasoning",
        "answer": "answer",
        "trace_present": True,
        "trace_source": "reasoning",
        "status": "success",
        "generation_key": f"qwen32|{condition_id}|1",
        "generation_provenance": {
            "dataset_id": "aisa-group/EvalAwareBench",
            "dataset_revision": "b" * 40,
            "temperature": 0.3,
            "max_tokens": 4096,
        },
        "system_prompt_provenance": {
            "prompt_key": "paper_natural",
            "prompt_revision": "c" * 40,
            "prompt_sha256": "d" * 64,
        },
        "record_id": f"record-{condition_id}",
        "response_id": f"response-{condition_id}",
        "response_model": "Qwen/Qwen3-32B",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "started_at": "2026-07-30T10:00:00Z",
        "completed_at": "2026-07-30T10:00:01Z",
        "elapsed_seconds": 1.0,
    }


def _raw_judgment(generation: dict) -> dict:
    raw_object = {
        "model_awareness": {
            "conclusion": "Yes",
            "awareness_type": "safety",
            "evidence": ["evidence"],
            "reasoning": "judge reasoning",
        },
        "hypothetical_recognition": {
            "conclusion": "No",
            "evidence": ["evidence"],
            "reasoning": "judge reasoning",
        },
        "task_performance": {
            "conclusion": "Yes",
            "evidence": ["evidence"],
            "reasoning": "judge reasoning",
        },
    }
    raw_text = json.dumps(raw_object)
    return {
        "custom_id": custom_id_for_generation(generation),
        "generation_record_sha256": _generation_record_sha256(generation),
        "awareness_conclusion": "yes",
        "awareness_type": "safety",
        "performance_conclusion": "yes",
        "judge_model": "openai/gpt-5.6-luna",
        "judge_profile": "openrouter-gpt-5.6-luna-direct",
        "judge_profile_label": "OpenRouter GPT-5.6 Luna alternative judge",
        "judge_provider": "OpenRouter",
        "judge_requested_model": "openai/gpt-5.6-luna",
        "judge_response_model": "openai/gpt-5.6-luna",
        "judge_response_id": "judge-response",
        "judge_request_id": "judge-request",
        "judge_finish_reason": "stop",
        "judge_template_sha256": "e" * 64,
        "judge_plan_sha256": "f" * 64,
        "judge_usage": {
            "prompt_tokens": 30,
            "completion_tokens": 40,
            "total_tokens": 70,
            "cost": 0.01,
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
        "judge_reasoning": {"effort": "medium"},
        "judge_provider_routing": {"allow_fallbacks": True},
        "raw_judge_object": raw_object,
        "raw_judge_body": {
            "choices": [{"message": {"content": raw_text}, "finish_reason": "stop"}],
        },
    }


def _source_metadata() -> dict:
    return {
        "waiver_manifest_sha256": "1" * 64,
        "waiver_manifest": "waiver-manifest.json",
        "missing_samples": "missing-samples.json",
        "missing_samples_sha256": "2" * 64,
        "analysis_csv": "figure6.csv",
        "analysis_csv_sha256": "3" * 64,
        "generation_file": "generations.jsonl",
        "generation_file_sha256": "4" * 64,
        "judgment_file": "judgments.jsonl",
        "judgment_file_sha256": "5" * 64,
        "attempt_log": "attempts.jsonl",
        "attempt_log_sha256": "6" * 64,
        "system_prompt": "system.txt",
        "system_prompt_sha256": "7" * 64,
        "judge_profile": "openrouter-gpt-5.6-luna-direct",
        "judge_model": "openai/gpt-5.6-luna",
        "judge_template_sha256": "8" * 64,
        "judge_plan_sha256": "9" * 64,
        "waived_at": "2026-07-30T23:03:57Z",
        "exporter_module_sha256": "a" * 64,
        "inspect_ai_version": "0.3.246",
    }


def test_build_cell_log_preserves_scored_and_waived_generations(tmp_path: Path) -> None:
    scored_generation = _generation(task_id="task-a_safety", condition_id="condition-a")
    waived_generation = _generation(task_id="task-b_safety", condition_id="condition-b")
    judgment = _raw_judgment(scored_generation)
    scored_id = custom_id_for_generation(scored_generation)
    waived_id = custom_id_for_generation(waived_generation)
    log = build_cell_log(
        [scored_generation, waived_generation],
        judgments_by_id={scored_id: judgment},
        attempts_by_id={
            scored_id: [
                {
                    "attempt": 1,
                    "status": "success",
                    "retryable": False,
                    "started_at": "2026-07-30T11:00:00Z",
                    "completed_at": "2026-07-30T11:00:01Z",
                }
            ],
            waived_id: [],
        },
        waiver_reasons={waived_id: "stopped_without_terminal_attempt_explicitly_waived"},
        system_prompt="system prompt",
        source_metadata=_source_metadata(),
        expected_samples=2,
    )

    assert log.status == "success"
    assert log.eval.metadata is not None
    assert log.eval.metadata["provenance_kind"] == "posthoc_import"
    assert log.eval.metadata["publication_complete"] is False
    assert log.results is not None
    assert log.results.total_samples == 2
    assert log.results.completed_samples == 2
    matched = next(score for score in log.results.scores if score.name == "matched_awareness")
    assert matched.scored_samples == 1
    assert matched.unscored_samples == 1
    assert matched.metrics["mean"].value == 1.0

    scored_sample = next(sample for sample in log.samples or [] if sample.metadata["judge_scored"])
    waived_sample = next(sample for sample in log.samples or [] if sample.metadata["judge_waived"])
    assert scored_sample.scores is not None
    assert scored_sample.scores[SCORE_NAME].value["hypothetical_recognition"] == 0.0
    assert isinstance(scored_sample.messages[-1].content[0], ContentReasoning)
    assert waived_sample.scores is None
    assert waived_sample.metadata["waiver_reason"] == "stopped_without_terminal_attempt_explicitly_waived"

    output = tmp_path / "cell.eval"
    write_eval_log(log, output, format="eval")
    reloaded = read_eval_log(output, format="eval")
    assert len(reloaded.samples or []) == 2
    assert reloaded.eval.metadata is not None
    assert reloaded.eval.metadata["converted_from_inspect"] is False


def test_build_cell_log_rejects_tampered_generation_hash() -> None:
    generation = _generation(task_id="task-a_safety", condition_id="condition-a")
    judgment = _raw_judgment(generation)
    judgment["generation_record_sha256"] = "0" * 64
    custom_id = custom_id_for_generation(generation)
    with pytest.raises(ValueError, match="does not match its generation record hash"):
        build_cell_log(
            [generation],
            judgments_by_id={custom_id: judgment},
            attempts_by_id={custom_id: []},
            waiver_reasons={},
            system_prompt="system prompt",
            source_metadata=_source_metadata(),
            expected_samples=1,
        )
