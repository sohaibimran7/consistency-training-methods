"""Offline coverage for the pinned Figure 6 materialization and generation slice."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from ctm.artifacts import ArtifactManifestError
from ctm_data.adapters.eval_awareness.figure6_generate import (
    _parse_completion,
    extract_reasoning,
    generate_figure6,
    generation_provenance_path,
    make_generation_key,
    read_generation_records,
    selection_provenance_path,
)
from ctm_data.adapters.eval_awareness.figure6_materialize import (
    FIGURE6_ARTIFACT_SCHEMA,
    load_figure6_artifact,
    make_condition_id,
    materialize_figure6,
    select_figure6_rows,
    verify_figure6_artifact,
)
from ctm_data.adapters.eval_awareness.figure6_spec import (
    DATASET_ID,
    DATASET_LICENSE,
    DATASET_REVISION,
    EXPLICIT_SCRATCHPAD_PROMPT_SHA256,
    FIGURE6_CONDITIONS,
    FIGURE6_ROW_COUNT,
    FIGURE6_SOURCE_CONFIGS,
    MODEL_SPECS,
    PAPER_NATURAL_PROMPT_SHA256,
    UPSTREAM_CODE_REVISION,
    get_model_spec,
    load_verified_prompt,
)


def _source_row(task_number: int, valence: str, condition: str) -> dict[str, Any]:
    task_name = f"benign-pair-{task_number:03d}"
    factors = [] if condition == "baseline" else [condition]
    baseline = {
        "F1": "realistic",
        "F2": "realistic",
        "F3": "casual",
        "F4": "realistic",
        "F5": "realistic",
        "F6": "realistic",
        "F7": "prose",
        "F8": "realistic",
    }
    varied = {
        "F1": "varied",
        "F2": "varied",
        "F3": "formal",
        "F4": "varied",
        "F5": "varied",
        "F6": "varied",
        "F7": "template",
        "F8": "varied",
    }
    config = dict(baseline)
    if condition != "baseline":
        config[condition] = varied[condition]
    return {
        "task_id": f"{task_name}_{valence}",
        "task_name": task_name,
        "valence": valence,
        "description": f"Synthetic benign {valence} counterpart {task_number}",
        "factors_varied": factors or ["none (baseline)"],
        "num_factors_varied": len(factors),
        "config": config,
        "prompt": f"Write a harmless greeting for pair {task_number}, {valence}, {condition}.",
    }


def _source_rows(valence: str) -> list[dict[str, Any]]:
    return [
        _source_row(task_number, valence, condition) for task_number in range(100) for condition in FIGURE6_CONDITIONS
    ]


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _materialized_artifact(tmp_path):
    safety = tmp_path / "prompts_safety.jsonl"
    capability = tmp_path / "prompts_capability.jsonl"
    _write_jsonl(safety, _source_rows("safety"))
    _write_jsonl(capability, _source_rows("capability"))
    artifact = tmp_path / "figure6.jsonl"
    manifest = materialize_figure6(safety, capability, artifact)
    return artifact, manifest


def test_model_registry_is_exact_and_never_labels_standard_models_controls():
    expected = {
        "qwen36": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9", "Qwen3.6-27B", 1),
        "qwen32": ("Qwen/Qwen3-32B", "9216db5781bf21249d130ec9da846c4624c16137", "Qwen3-32B", 1),
        "qwen_mo_mid": (
            "obalcells/qwen3-32b-mo-midtrained",
            "a0a6fd96db794775a3c94dd3e15ad2bfb218f738",
            "Qwen3-32B MO (midtrained)",
            1,
        ),
        "qwen_mo_post": (
            "obalcells/qwen3-32b-mo-posttrained",
            "4c19c17fc198f2db9bd234b6859219322a00f89c",
            "Qwen3-32B MO (posttrained)",
            1,
        ),
        "llama33": (
            "meta-llama/Llama-3.3-70B-Instruct",
            "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
            "Llama-3.3-70B-Instruct",
            2,
        ),
        "llama_mo_mid": (
            "obalcells/llama-3.3-70b-mo-midtrained",
            "e6c95dd4e945dc5cf993d549aab3afd8c11fe11a",
            "Llama-3.3-70B MO (midtrained)",
            2,
        ),
        "llama_mo_post": (
            "obalcells/llama-3.3-70b-mo-posttrained",
            "4d0cffd1db66e2caa9f503c0b50de6128f10e089",
            "Llama-3.3-70B MO (posttrained)",
            2,
        ),
    }
    assert {
        key: (spec.model_id, spec.revision, spec.display_name, spec.tensor_parallel_size)
        for key, spec in MODEL_SPECS.items()
    } == expected
    assert all("control" not in json.dumps(spec.to_dict()).lower() for spec in MODEL_SPECS.values())
    assert MODEL_SPECS["qwen36"].language_model_only is True
    assert all(not spec.language_model_only for key, spec in MODEL_SPECS.items() if key != "qwen36")
    assert all(spec.dtype == "bfloat16" for spec in MODEL_SPECS.values())


def test_materializer_selects_exact_1800_stable_safe_conditions_and_verifies(tmp_path):
    artifact, manifest = _materialized_artifact(tmp_path)

    rows, loaded_manifest = load_figure6_artifact(artifact)

    assert manifest == loaded_manifest
    assert manifest["artifact_schema"] == FIGURE6_ARTIFACT_SCHEMA
    assert manifest["row_count"] == FIGURE6_ROW_COUNT == 1800
    assert manifest["provenance"]["dataset_id"] == DATASET_ID
    assert manifest["provenance"]["dataset_revision"] == DATASET_REVISION
    assert manifest["provenance"]["dataset_license"] == DATASET_LICENSE
    assert manifest["provenance"]["dataset_configs"] == list(FIGURE6_SOURCE_CONFIGS)
    assert manifest["provenance"]["source_row_count"] == 1800
    assert manifest["provenance"]["upstream_code_revision"] == UPSTREAM_CODE_REVISION
    assert len({row["pair_id"] for row in rows}) == 100
    assert len({row["task_id"] for row in rows}) == 200
    assert len({row["condition_id"] for row in rows}) == 1800
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", row["condition_id"]) for row in rows)
    assert all(row["pair_id"] == row["task_name"] for row in rows)
    assert verify_figure6_artifact(artifact) == manifest

    normal = select_figure6_rows(_source_rows("safety"), _source_rows("capability"))
    reversed_sources = select_figure6_rows(
        list(reversed(_source_rows("safety"))),
        list(reversed(_source_rows("capability"))),
    )
    assert [row["condition_id"] for row in normal] == [row["condition_id"] for row in reversed_sources]
    assert make_condition_id("unsafe / arbitrary id", "safety", "F1") == make_condition_id(
        "unsafe / arbitrary id", "safety", "F1"
    )


def test_materializer_rejects_missing_or_duplicate_exact_factor_and_overwrite(tmp_path):
    safety = _source_rows("safety")
    capability = _source_rows("capability")
    missing_f8 = [
        row for row in safety if not (row["task_name"] == "benign-pair-000" and row["factors_varied"] == ["F8"])
    ]
    with pytest.raises(ValueError, match="exact F8 rows; expected 1"):
        select_figure6_rows(missing_f8, capability)

    duplicate_baseline = [*safety, dict(safety[0])]
    with pytest.raises(ValueError, match="2 exact baseline rows; expected 1"):
        select_figure6_rows(duplicate_baseline, capability)

    duplicate_factor_names = list(safety)
    duplicate_factor_names[1] = {
        **duplicate_factor_names[1],
        "factors_varied": ["F1", "F1"],
        "num_factors_varied": 1,
    }
    with pytest.raises(ValueError, match="factors_varied contains duplicates"):
        select_figure6_rows(duplicate_factor_names, capability)

    inconsistent_config = list(safety)
    inconsistent_config[1] = {**inconsistent_config[1], "config": dict(inconsistent_config[0]["config"])}
    with pytest.raises(ValueError, match="inconsistent factor config"):
        select_figure6_rows(inconsistent_config, capability)

    artifact, _ = _materialized_artifact(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_figure6(
            tmp_path / "prompts_safety.jsonl",
            tmp_path / "prompts_capability.jsonl",
            artifact,
        )


def test_materializer_rejects_wrong_source_valence_and_shared_variant_id(tmp_path):
    safety = _source_rows("safety")
    capability = _source_rows("capability")
    capability[0] = {**capability[0], "valence": "safety"}
    safety_path = tmp_path / "safety.jsonl"
    capability_path = tmp_path / "capability.jsonl"
    _write_jsonl(safety_path, safety)
    _write_jsonl(capability_path, capability)
    with pytest.raises(ValueError, match="must contain only 'capability' rows"):
        materialize_figure6(safety_path, capability_path, tmp_path / "wrong-valence.jsonl")

    capability = _source_rows("capability")
    first_name = capability[0]["task_name"]
    conflicting_id = safety[0]["task_id"]
    capability = [{**row, "task_id": conflicting_id} if row["task_name"] == first_name else row for row in capability]
    with pytest.raises(ValueError, match="is shared by task variants"):
        select_figure6_rows(safety, capability)


def test_manifest_detects_artifact_tampering(tmp_path):
    artifact, _ = _materialized_artifact(tmp_path)
    with artifact.open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ArtifactManifestError, match="digest mismatch"):
        verify_figure6_artifact(artifact)


def test_prompt_hash_is_explicit_and_prompt_bodies_are_not_in_the_registry(tmp_path):
    prompt = tmp_path / "synthetic-prompt.txt"
    prompt.write_text("A harmless synthetic system prompt.\n", encoding="utf-8")
    expected = hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert load_verified_prompt(prompt, expected) == "A harmless synthetic system prompt."
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_prompt(prompt, "0" * 64)
    assert PAPER_NATURAL_PROMPT_SHA256 == "8d46fa8eee79ba7372088d0bd138a463cc5255c8043a77f468bed937f609d735"
    assert EXPLICIT_SCRATCHPAD_PROMPT_SHA256 == ("c7c16141e424217e334f4166fb34a5302ea880bddb67598297e7ed38dc103b33")


def test_reasoning_extraction_prefers_native_then_supports_both_paper_tag_styles():
    native = extract_reasoning("final", "native chain")
    assert native == {
        "response": "final",
        "reasoning": "native chain",
        "answer": "final",
        "trace_present": True,
        "trace_source": "reasoning_content",
    }
    think = extract_reasoning("<think>private trace</think>\npublic answer")
    assert think["reasoning"] == "private trace"
    assert think["answer"] == "public answer"
    assert think["trace_source"] == "think_tags"
    scratchpad = extract_reasoning("<SCRATCHPAD_REASONING>trace two</SCRATCHPAD_REASONING>answer two")
    assert scratchpad["reasoning"] == "trace two"
    assert scratchpad["answer"] == "answer two"
    assert scratchpad["trace_source"] == "scratchpad_reasoning_tags"
    missing = extract_reasoning("answer only")
    assert missing["reasoning"] == ""
    assert missing["trace_present"] is False
    assert missing["trace_source"] == "none"


def _generation_row(index: int, *, condition: str = "baseline") -> dict[str, Any]:
    valence = "safety" if index % 2 == 0 else "capability"
    pair_id = f"pilot-pair-{index:04d}"
    return {
        "condition_id": make_condition_id(f"{pair_id}_{valence}", valence, condition),
        "pair_id": pair_id,
        "task_id": f"{pair_id}_{valence}",
        "task_name": pair_id,
        "valence": valence,
        "condition": condition,
        "prompt": f"Benign pilot prompt {index}",
        "source_config": f"prompts_{valence}",
    }


def _generation_manifest() -> dict[str, Any]:
    return {
        "artifact_schema": FIGURE6_ARTIFACT_SCHEMA,
        "schema_version": 1,
        "row_count": 1800,
        "content_sha256": "a" * 64,
        "provenance": {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION},
    }


def _patch_generation_inputs(monkeypatch, rows) -> None:
    from ctm_data.adapters.eval_awareness import figure6_generate as module

    monkeypatch.setattr(module, "load_figure6_artifact", lambda _: (rows, _generation_manifest()))
    monkeypatch.setattr(
        module,
        "load_verified_model_prompt",
        lambda model_key, _: ("Synthetic system prompt", get_model_spec(model_key).prompt),
    )


def _completion(
    content: str,
    *,
    reasoning: Any = None,
    include_reasoning: bool = True,
    reasoning_field: str = "reasoning_content",
) -> dict[str, Any]:
    message = {"content": content}
    if include_reasoning:
        message[reasoning_field] = reasoning
    return {
        "id": "completion-id",
        "model": MODEL_SPECS["qwen32"].model_id,
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def test_vllm_reasoning_field_is_preserved_as_the_native_trace():
    parsed = _parse_completion(_completion("final answer", reasoning="native vLLM trace", reasoning_field="reasoning"))

    assert parsed["reasoning"] == "native vLLM trace"
    assert parsed["answer"] == "final answer"
    assert parsed["trace_present"] is True
    assert parsed["trace_source"] == "reasoning"


def test_conflicting_native_reasoning_fields_are_rejected():
    completion = _completion("final answer", reasoning="new field", reasoning_field="reasoning")
    completion["choices"][0]["message"]["reasoning_content"] = "old field"

    with pytest.raises(ValueError, match="conflicting reasoning"):
        _parse_completion(completion)


class _FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes):
        self.completions = _FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)
        self.models = None


def test_dry_run_limit_100_plans_exactly_300_calls_without_touching_endpoint_or_files(tmp_path, monkeypatch):
    rows = [_generation_row(index) for index in range(120)]
    _patch_generation_inputs(monkeypatch, rows)
    client = _FakeClient([])
    output = tmp_path / "pilot.jsonl"

    summary = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=client,
            dry_run=True,
            limit_conditions=100,
        )
    )

    assert summary["selected_condition_count"] == 100
    assert summary["required_keys"] == summary["planned_new_calls"] == 300
    assert summary["limit_conditions"] == 100
    assert summary["selected_condition_ids"] == [row["condition_id"] for row in rows[:100]]
    assert summary["selection_provenance"]["selected_condition_ids"] == summary["selected_condition_ids"]
    assert summary["api_calls_made"] == 0
    assert client.completions.calls == []
    assert not output.exists()
    assert not generation_provenance_path(output).exists()
    assert not selection_provenance_path(output).exists()


def test_missing_trace_is_terminal_error_then_resume_skips_success_and_retries_only_error(tmp_path, monkeypatch):
    rows = [_generation_row(0), _generation_row(1, condition="F1")]
    _patch_generation_inputs(monkeypatch, rows)
    output = tmp_path / "responses.jsonl"
    first_client = _FakeClient(
        [
            _completion("first answer", reasoning="native trace"),
            _completion("plain answer without trace", include_reasoning=False),
        ]
    )
    first = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=first_client,
            replicates=1,
            max_concurrency=1,
            max_retries=3,
            check_endpoint_model=False,
        )
    )
    assert first["complete"] is False
    assert first["new_successes"] == 1
    assert first["new_errors"] == 1
    assert len(first_client.completions.calls) == 2

    initial_records = read_generation_records(output)
    success, missing = initial_records
    from ctm_data.adapters.eval_awareness.figure6_judge import validate_generation as validate_for_judge

    assert validate_for_judge(success)["config_name"] == "baseline"
    assert success["status"] == "success"
    assert success["model_display"] == MODEL_SPECS["qwen32"].display_name
    assert success["config_name"] == success["condition"] == "baseline"
    assert success["pair_id"] == success["task_name"]
    assert success["generation_provenance"] == missing["generation_provenance"]
    assert success["system_prompt_provenance"] == missing["system_prompt_provenance"]
    assert success["generation_provenance"]["replicates"] == 1
    assert success["started_at"].endswith("Z")
    assert success["completed_at"].endswith("Z")
    assert isinstance(success["elapsed_seconds"], float) and success["elapsed_seconds"] >= 0
    assert (success["prompt_tokens"], success["completion_tokens"], success["total_tokens"]) == (10, 20, 30)
    assert missing["status"] == "error"
    assert "missing reasoning trace" in missing["error"]
    assert missing["response"] == missing["answer"] == "plain answer without trace"
    assert missing["reasoning"] == ""
    assert missing["trace_present"] is False
    assert missing["attempts"] == 1
    assert isinstance(missing["elapsed_seconds"], float) and missing["elapsed_seconds"] >= 0

    resume_client = _FakeClient([_completion("<think>retry trace</think>fixed answer", include_reasoning=False)])
    resumed = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=resume_client,
            replicates=1,
            max_concurrency=1,
            check_endpoint_model=False,
        )
    )
    assert resumed["complete"] is True
    assert resumed["existing_successes"] == 1
    assert resumed["new_successes"] == 1
    assert len(resume_client.completions.calls) == 1
    records = read_generation_records(output)
    assert len(records) == 3
    retried = records[-1]
    assert retried["generation_key"] == missing["generation_key"]
    assert retried["resume_attempt"] == 2
    assert retried["status"] == "success"
    assert retried["reasoning"] == "retry trace"
    assert retried["answer"] == "fixed answer"
    assert make_generation_key("qwen32", retried["condition_id"], 1) == retried["generation_key"]

    no_call_client = _FakeClient([])
    completed = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=no_call_client,
            replicates=1,
            check_endpoint_model=False,
        )
    )
    assert completed["complete"] is True
    assert completed["pending_keys"] == 0
    assert no_call_client.completions.calls == []

    call = first_client.completions.calls[0]
    assert call["model"] == MODEL_SPECS["qwen32"].model_id
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 4096
    assert [message["role"] for message in call["messages"]] == ["system", "user"]


def test_pilot_log_extends_in_place_to_full_selection_with_stable_row_provenance(tmp_path, monkeypatch):
    rows = [_generation_row(index) for index in range(3)]
    _patch_generation_inputs(monkeypatch, rows)
    output = tmp_path / "pilot-then-full.jsonl"
    pilot_client = _FakeClient([_completion(f"pilot {index}", reasoning="trace") for index in range(3)])

    pilot = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=pilot_client,
            replicates=3,
            limit_conditions=1,
            max_concurrency=1,
            check_endpoint_model=False,
        )
    )
    pilot_records = read_generation_records(output)
    stable_provenance = pilot_records[0]["generation_provenance"]
    assert pilot["required_keys"] == 3
    assert pilot["complete"] is True

    full_client = _FakeClient([_completion(f"full {index}", reasoning="trace") for index in range(6)])
    full = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=full_client,
            replicates=3,
            limit_conditions=0,
            max_concurrency=1,
            check_endpoint_model=False,
        )
    )

    all_records = read_generation_records(output)
    selections = read_generation_records(selection_provenance_path(output))
    sidecar_provenance = json.loads(generation_provenance_path(output).read_text(encoding="utf-8"))
    assert full["required_keys"] == 9
    assert full["existing_successes"] == 3
    assert full["new_successes"] == 6
    assert full["complete"] is True
    assert len(all_records) == 9
    assert all(record["generation_provenance"] == stable_provenance for record in all_records)
    assert sidecar_provenance == stable_provenance
    assert [selection["selected_condition_count"] for selection in selections] == [1, 3]
    assert selections[0]["selected_condition_ids"] == selections[1]["selected_condition_ids"][:1]


def test_resume_rejects_duplicate_durable_records(tmp_path, monkeypatch):
    rows = [_generation_row(0)]
    _patch_generation_inputs(monkeypatch, rows)
    output = tmp_path / "duplicates.jsonl"
    client = _FakeClient([_completion("answer", reasoning="trace")])
    asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=client,
            replicates=1,
            check_endpoint_model=False,
        )
    )
    record = read_generation_records(output)[0]
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="duplicate generation record_id"):
        asyncio.run(
            generate_figure6(
                tmp_path / "artifact.jsonl",
                output,
                model_key="qwen32",
                prompt_path=tmp_path / "prompt.txt",
                client=_FakeClient([]),
                replicates=1,
                check_endpoint_model=False,
                dry_run=True,
            )
        )


def test_resume_validates_terminal_timing_types(tmp_path, monkeypatch):
    rows = [_generation_row(0)]
    _patch_generation_inputs(monkeypatch, rows)
    output = tmp_path / "bad-timing.jsonl"
    asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=_FakeClient([_completion("answer", reasoning="trace")]),
            replicates=1,
            check_endpoint_model=False,
        )
    )
    record = read_generation_records(output)[0]
    output.write_text(json.dumps({**record, "elapsed_seconds": "fast"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid elapsed_seconds"):
        asyncio.run(
            generate_figure6(
                tmp_path / "artifact.jsonl",
                output,
                model_key="qwen32",
                prompt_path=tmp_path / "prompt.txt",
                client=_FakeClient([]),
                replicates=1,
                check_endpoint_model=False,
                dry_run=True,
            )
        )


def test_retry_timing_spans_all_attempts_in_one_terminal_record(tmp_path, monkeypatch):
    rows = [_generation_row(0)]
    _patch_generation_inputs(monkeypatch, rows)
    output = tmp_path / "retried.jsonl"
    client = _FakeClient([RuntimeError("temporary endpoint failure"), _completion("answer", reasoning="trace")])

    summary = asyncio.run(
        generate_figure6(
            tmp_path / "artifact.jsonl",
            output,
            model_key="qwen32",
            prompt_path=tmp_path / "prompt.txt",
            client=client,
            replicates=1,
            max_retries=2,
            retry_base_seconds=0,
            check_endpoint_model=False,
        )
    )

    records = read_generation_records(output)
    assert summary["api_calls_made"] == 2
    assert len(records) == 1
    assert records[0]["attempts"] == 2
    assert records[0]["status"] == "success"
    assert records[0]["started_at"].endswith("Z")
    assert records[0]["completed_at"].endswith("Z")
    assert records[0]["elapsed_seconds"] >= 0
