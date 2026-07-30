"""Offline tests for Figure 6 judge batching, aggregation, and rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ctm_data.adapters.eval_awareness import figure6_analysis as analysis
from ctm_data.adapters.eval_awareness import figure6_judge as judge
from ctm_data.adapters.eval_awareness import figure6_openrouter as openrouter
from ctm_data.adapters.eval_awareness.figure6_analysis import (
    CURRENT_QWEN_MODEL_KEY_ORDER,
    EXPECTED_CELL_COUNT,
    EXPECTED_JUDGMENT_COUNT,
    EXPECTED_MODEL_JUDGMENT_COUNT,
    OPENROUTER_MUSE_ALTERNATIVE_LABEL,
    PublicationValidationError,
    STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL,
    analyze_judgments,
    should_annotate_delta,
)
from ctm_data.adapters.eval_awareness.figure6_plot import (
    CONFIG_LABELS,
    MODEL_DISPLAY_ORDER,
    NEGATIVE_DELTA_COLOR,
    POSITIVE_DELTA_COLOR,
    fixture_rows,
    performance_delta_annotation,
    render_figure6,
    validate_plot_rows,
)
from ctm_data.adapters.eval_awareness.figure6_spec import (
    FIGURE6_CONDITIONS,
    FIGURE6_VALENCES,
    MODEL_SPECS,
)


def _generation(
    *,
    model_key: str = "unit",
    task_id: str = "task-000",
    valence: str = "safety",
    config_name: str = "baseline",
    replicate: int = 1,
    pair_id: str = "pair-000",
) -> dict:
    return {
        "model_key": model_key,
        "model_display": "Unit model",
        "model_id": "org/unit-model",
        "model_revision": "a" * 40,
        "condition_id": f"condition-{task_id}-{valence}-{config_name}",
        "pair_id": pair_id,
        "task_id": task_id,
        "valence": valence,
        "config_name": config_name,
        "replicate": replicate,
        "prompt": f"Task {task_id}",
        "response": "reason then answer",
        "reasoning": "private reasoning",
        "answer": "answer",
        "trace_present": True,
        "trace_source": "reasoning_content",
        "status": "success",
        "generation_provenance": {"run": "offline", "temperature": 0.3},
        "system_prompt_provenance": {"sha256": "b" * 64},
    }


def _raw_judge(
    *,
    awareness: str = "Yes",
    awareness_type: str = "Safety",
    performance: str = "Yes",
) -> dict:
    return {
        "model_awareness": {
            "analysis": "benign fixture",
            "conclusion": awareness,
            "awareness_type": awareness_type,
        },
        "hypothetical_recognition": {"analysis": "not used", "conclusion": "No"},
        "task_performance": {"analysis": "benign fixture", "conclusion": performance},
    }


def _batch_output(generation: dict, raw: dict, *, fenced: bool = False) -> dict:
    content = json.dumps(raw)
    if fenced:
        content = f"```json\n{content}\n```"
    return {
        "id": "batch-request-unit",
        "custom_id": judge.custom_id_for_generation(generation),
        "response": {
            "status_code": 200,
            "request_id": "request-unit",
            "body": {
                "id": "completion-unit",
                "model": "gpt-5",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
            },
        },
        "error": None,
    }


def _judgment(
    *,
    model_key: str = "unit",
    task_id: str = "task-000",
    valence: str = "safety",
    config_name: str = "baseline",
    replicate: int = 1,
    awareness: str = "no",
    awareness_type: str = "N/A",
    performance: str = "no",
    pair_id: str = "pair-000",
) -> dict:
    generation = _generation(
        model_key=model_key,
        task_id=task_id,
        valence=valence,
        config_name=config_name,
        replicate=replicate,
        pair_id=pair_id,
    )
    return {
        **generation,
        "generation_status": generation["status"],
        "judge_model": "gpt-5",
        "judge_template_sha256": judge.PAPER_JUDGE_TEMPLATE_SHA256,
        "judge_max_completion_tokens": judge.MAX_JUDGE_TOKENS,
        "judge_status": "ok",
        "awareness_conclusion": awareness,
        "awareness_type": awareness_type,
        "performance_conclusion": performance,
    }


def _small_matrix() -> list[dict]:
    rows = []
    for valence in FIGURE6_VALENCES:
        for config_name in ("baseline", "F1"):
            for replicate in (1, 2, 3):
                rows.append(
                    _judgment(
                        valence=valence,
                        task_id=f"task-000_{valence}",
                        pair_id="pair-000",
                        config_name=config_name,
                        replicate=replicate,
                        awareness="yes" if replicate != 3 else "no",
                        awareness_type=(
                            valence
                            if replicate == 1
                            else ("capability" if valence == "safety" else "safety") if replicate == 2 else "N/A"
                        ),
                        performance="yes" if (config_name == "F1" or replicate == 1) else "no",
                    )
                )
    return rows


def _small_analysis(rows: list[dict], **kwargs):
    return analyze_judgments(
        rows,
        expected_model_keys=("unit",),
        expected_task_ids={
            "safety": ("task-000_safety",),
            "capability": ("task-000_capability",),
        },
        expected_pair_ids=("pair-000",),
        expected_task_count=1,
        expected_valences=FIGURE6_VALENCES,
        expected_configs=("baseline", "F1"),
        **kwargs,
    )


def test_template_render_hash_and_batch_custom_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template_text = 'TASK={task}\nREASONING={reasoning}\nANSWER={answer}\nSCHEMA={{"x": "value"}}'
    template = tmp_path / "upstream-judge.txt"
    template.write_text(template_text, encoding="utf-8")
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    monkeypatch.setattr(judge, "PAPER_JUDGE_TEMPLATE_SHA256", digest)

    loaded = judge.load_judge_template(template, expected_sha256=digest)
    rendered = judge.render_judge_prompt(loaded, task="T {safe}", reasoning="R", answer="A")
    first = _generation(replicate=1)
    second = _generation(replicate=2)
    requests = judge.create_batch_requests([second, first], loaded)

    assert rendered == 'TASK=T {safe}\nREASONING=R\nANSWER=A\nSCHEMA={"x": "value"}'
    assert requests == judge.create_batch_requests([first, second], loaded)
    assert all(row["url"] == "/v1/chat/completions" for row in requests)
    assert all(row["body"]["messages"][0]["role"] == "system" for row in requests)
    assert all(row["body"]["max_completion_tokens"] == 4_096 for row in requests)
    assert all(row["custom_id"].startswith("figure6-") and len(row["custom_id"]) <= 64 for row in requests)
    template.write_text(template_text + "\nchanged", encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        judge.load_judge_template(template, expected_sha256=digest)


def test_batch_request_accepts_generation_aliases_and_rejects_missing_trace():
    generation = _generation()
    generation["condition"] = generation.pop("config_name")
    generation["model_display_name"] = generation.pop("model_display")
    validated = judge.validate_generation(generation)

    assert validated["config_name"] == "baseline"
    assert validated["model_display"] == "Unit model"
    assert validated["pair_id"] == "pair-000"
    with pytest.raises(ValueError, match="must not be sent for paid judging"):
        judge.validate_generation({**generation, "trace_present": False})
    with pytest.raises(ValueError, match="non-blank"):
        judge.validate_generation({**generation, "reasoning": "   "})


def test_append_only_generation_history_selects_one_terminal_success():
    success = {**_generation(), "generation_key": "unit|condition|1", "resume_attempt": 2}
    failure = {
        **success,
        "resume_attempt": 1,
        "status": "error",
        "reasoning": "",
        "trace_present": False,
    }

    assert judge.select_successful_generations([failure, success]) == [success]
    with pytest.raises(ValueError, match="no successful terminal"):
        judge.select_successful_generations([failure])


def test_batch_creation_shards_by_count_and_exact_utf8_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template = tmp_path / "judge.txt"
    template.write_text("{task}\n{reasoning}\n{answer}", encoding="utf-8")
    digest = hashlib.sha256(template.read_bytes()).hexdigest()
    monkeypatch.setattr(judge, "PAPER_JUDGE_TEMPLATE_SHA256", digest)
    generations = [_generation(replicate=replicate) for replicate in (1, 2, 3)]
    requests = judge.create_batch_requests(generations, template.read_text())
    first_line_bytes = len(
        (json.dumps(requests[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    )

    shards = judge.shard_batch_requests(
        requests,
        max_requests_per_shard=10,
        max_bytes_per_shard=first_line_bytes + 2,
    )
    manifest = judge.create_batch_jsonl(
        generations,
        template_path=template,
        expected_template_sha256=digest,
        output_path=tmp_path / "requests.jsonl",
        max_requests_per_shard=2,
        max_bytes_per_shard=10_000,
    )

    assert [len(shard) for shard in shards] == [1, 1, 1]
    assert manifest["shard_count"] == 2
    assert manifest["max_completion_tokens"] == 4_096
    assert [shard["row_count"] for shard in manifest["shards"]] == [2, 1]
    assert all(Path(path).is_file() for path in manifest["shard_paths"])
    disk_manifest = json.loads(Path(manifest["manifest_path"]).read_text())
    assert [item["custom_ids"] for item in disk_manifest["shards"]] == [
        item["custom_ids"] for item in manifest["shards"]
    ]
    assert all(item["utf8_bytes"] <= 10_000 for item in manifest["shards"])


def test_collect_accepts_plain_and_fenced_exact_upstream_schema():
    first = _generation(replicate=1)
    second = _generation(replicate=2)
    outputs = [
        _batch_output(second, _raw_judge(awareness_type="Capability", performance="No"), fenced=True),
        _batch_output(first, _raw_judge()),
    ]

    normalized = judge.collect_batch_outputs([first, second], outputs)

    assert len(normalized) == 2
    assert normalized[0]["judge_status"] == "ok"
    assert normalized[0]["judge_max_completion_tokens"] == 4_096
    assert normalized[0]["raw_judge_object"]["model_awareness"]["conclusion"] == "Yes"
    assert {row["awareness_type"] for row in normalized} == {"safety", "capability"}
    assert {row["performance_conclusion"] for row in normalized} == {"yes", "no"}


def test_collect_rejects_duplicate_missing_error_and_invalid_judge_records():
    first = _generation(replicate=1)
    second = _generation(replicate=2)
    good = _batch_output(first, _raw_judge())
    with pytest.raises(ValueError, match="duplicate batch output"):
        judge.collect_batch_outputs([first], [good, good])
    with pytest.raises(ValueError, match="missing 1 batch output"):
        judge.collect_batch_outputs([first, second], [good])
    error = {**good, "response": None, "error": {"code": "fixture_error"}}
    with pytest.raises(ValueError, match="error record"):
        judge.collect_batch_outputs([first], [error])
    bad_http = {**good, "response": {**good["response"], "status_code": 500}}
    with pytest.raises(ValueError, match="HTTP status 500"):
        judge.collect_batch_outputs([first], [bad_http])
    bad_choices = {
        **good,
        "response": {**good["response"], "body": {**good["response"]["body"], "choices": []}},
    }
    with pytest.raises(ValueError, match="exactly one choice"):
        judge.collect_batch_outputs([first], [bad_choices])
    invalid = _batch_output(first, _raw_judge(awareness="Maybe"))
    with pytest.raises(ValueError, match="awareness_conclusion"):
        judge.collect_batch_outputs([first], [invalid])
    inconsistent = _batch_output(first, _raw_judge(awareness="No", awareness_type="Safety"))
    with pytest.raises(ValueError, match="requires awareness_type=N/A"):
        judge.collect_batch_outputs([first], [inconsistent])


def test_strict_aggregation_counts_matched_and_mismatched_with_exact_denominator():
    matrix = _small_matrix()
    matrix[0]["condition"] = matrix[0].pop("config_name")
    result = _small_analysis(matrix)
    safety_f1 = next(row for row in result.rows if row["valence"] == "safety" and row["config_name"] == "F1")

    assert safety_f1["n"] == 3
    assert safety_f1["awareness_yes_count"] == 2
    assert safety_f1["matched_awareness_count"] == 1
    assert safety_f1["mismatched_awareness_count"] == 1
    assert safety_f1["matched_awareness_fraction"] == pytest.approx(1 / 3)
    assert safety_f1["performance_yes_fraction"] == 1.0
    assert safety_f1["performance_delta_pp"] == pytest.approx(100 - 100 / 3)
    assert safety_f1["annotate_performance_delta"] is True
    assert result.summary["publication_complete"] is True


def test_explicit_registered_scope_still_requires_the_pinned_model_identity():
    model = MODEL_SPECS["qwen36"]
    rows = []
    for replicate in (1, 2, 3):
        row = _judgment(model_key="qwen36", replicate=replicate)
        row.update(
            {
                "model_display": "Imposter Qwen",
                "model_id": model.model_id,
                "model_revision": model.revision,
                "system_prompt_provenance": {
                    "prompt_key": model.prompt.key,
                    "prompt_revision": "446be5c605b56a60d4efe2526f0cbf55522c523a",
                    "prompt_sha256": model.prompt.sha256,
                },
            }
        )
        rows.append(row)

    with pytest.raises(PublicationValidationError, match="pinned Figure 6 registry"):
        analyze_judgments(
            rows,
            expected_model_keys=("qwen36",),
            expected_task_ids=("task-000",),
            expected_pair_ids=("pair-000",),
            expected_task_count=1,
            expected_valences=("safety",),
            expected_configs=("baseline",),
        )


def test_strict_aggregation_rejects_trace_duplicate_and_mixed_provenance():
    missing_trace = _small_matrix()
    missing_trace[0] = {**missing_trace[0], "trace_present": False}
    with pytest.raises(PublicationValidationError, match="trace_present=false"):
        _small_analysis(missing_trace)

    duplicate = _small_matrix()
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(PublicationValidationError, match="duplicate generation key"):
        _small_analysis(duplicate)

    mixed = _small_matrix()
    mixed[0] = {**mixed[0], "generation_provenance": {"run": "different"}}
    with pytest.raises(PublicationValidationError, match="mixed generation provenance"):
        _small_analysis(mixed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_status", "error", "generation_status='error'"),
        ("judge_status", "error", "judge_status='error'"),
    ],
)
def test_strict_aggregation_rejects_generation_and_judge_failures(field: str, value: str, message: str):
    rows = _small_matrix()
    rows[0] = {**rows[0], field: value}
    with pytest.raises(PublicationValidationError, match=message):
        _small_analysis(rows)


def test_strict_aggregation_rejects_uniform_nonpaper_judge_model():
    rows = [{**row, "judge_model": "alternate-judge"} for row in _small_matrix()]
    with pytest.raises(PublicationValidationError, match="pinned paper model"):
        _small_analysis(rows)


def test_strict_muse_rejects_custom_non_qwen_scope_before_publication():
    allowed_models = ["meta/muse-spark-1.1", "meta/muse-spark-1.1-20260709"]
    rows = [
        {
            **row,
            "judge_profile": openrouter.MUSE_US_PROXY_PROFILE,
            "judge_model": "meta/muse-spark-1.1",
            "judge_provider": "OpenRouter",
            "judge_response_model": "meta/muse-spark-1.1-20260709",
            "judge_allowed_response_models": allowed_models,
        }
        for row in _small_matrix()
    ]
    with pytest.raises(PublicationValidationError, match="exactly qwen32"):
        _small_analysis(
            rows,
            expected_judge_profile=openrouter.MUSE_US_PROXY_PROFILE,
            expected_judge_model="meta/muse-spark-1.1",
            expected_judge_provider="OpenRouter",
            expected_judge_response_models=allowed_models,
        )


@pytest.mark.parametrize("response_model", [None, "meta/unexpected-model"])
def test_strict_alternative_analysis_rejects_missing_or_unapproved_response_identity(response_model: str | None):
    allowed_models = ["meta/muse-spark-1.1", "meta/muse-spark-1.1-20260709"]
    rows = [
        {
            **row,
            "judge_profile": openrouter.MUSE_US_PROXY_PROFILE,
            "judge_model": "meta/muse-spark-1.1",
            "judge_provider": "OpenRouter",
            "judge_response_model": response_model,
            "judge_allowed_response_models": allowed_models,
        }
        for row in _small_matrix()
    ]
    result = _small_analysis(
        rows,
        allow_partial=True,
        expected_judge_profile=openrouter.MUSE_US_PROXY_PROFILE,
        expected_judge_model="meta/muse-spark-1.1",
        expected_judge_provider="OpenRouter",
        expected_judge_response_models=(allowed_models if response_model is not None else None),
    )
    assert result.diagnostics["issue_counts"]["judge_response_model_mismatch"] == 1


def test_strict_muse_analysis_does_not_allow_the_identity_set_to_be_widened():
    with pytest.raises(ValueError, match="response identities differ"):
        _small_analysis(
            _small_matrix(),
            allow_partial=True,
            expected_judge_profile=openrouter.MUSE_US_PROXY_PROFILE,
            expected_judge_model="meta/muse-spark-1.1",
            expected_judge_response_models=["meta/muse-spark-1.1", "meta/unexpected-model"],
        )


def test_strict_aggregation_rejects_uniform_nonpaper_judge_token_limit():
    rows = [{**row, "judge_max_completion_tokens": 2_048} for row in _small_matrix()]
    with pytest.raises(PublicationValidationError, match="completion-token provenance"):
        _small_analysis(rows)


def test_allow_partial_is_explicitly_diagnostic_and_never_complete():
    rows = _small_matrix()[:-1]
    result = _small_analysis(rows, allow_partial=True)

    assert result.summary["publication_complete"] is False
    assert result.summary["result_label"] == "diagnostic_partial"
    assert result.diagnostics["issue_counts"]["cell_denominator"] == 1
    assert all(row["publication_complete"] is False for row in result.rows)


def test_annotation_threshold_is_strictly_greater_than_five():
    assert should_annotate_delta(5.0) is False
    assert should_annotate_delta(-5.0) is False
    assert should_annotate_delta(5.01) is True
    assert should_annotate_delta(-5.01) is True
    assert performance_delta_annotation(5.0, config_name="F1") is None
    assert performance_delta_annotation(5.01, config_name="F1")[1] == POSITIVE_DELTA_COLOR
    assert performance_delta_annotation(-5.01, config_name="F1")[1] == NEGATIVE_DELTA_COLOR


def _registered_publication_matrix(model_keys: tuple[str, ...]) -> list[dict]:
    judgments = []
    for model_key in model_keys:
        model = MODEL_SPECS[model_key]
        for valence in FIGURE6_VALENCES:
            for config_name in FIGURE6_CONDITIONS:
                for task_index in range(100):
                    pair_id = f"pair-{task_index:03d}"
                    task_id = f"{pair_id}_{valence}"
                    for replicate in (1, 2, 3):
                        row = _judgment(
                            model_key=model_key,
                            task_id=task_id,
                            valence=valence,
                            config_name=config_name,
                            replicate=replicate,
                            pair_id=pair_id,
                            awareness="yes" if replicate == 1 else "no",
                            awareness_type=valence if replicate == 1 else "N/A",
                            performance="yes" if replicate != 3 else "no",
                        )
                        row.update(
                            {
                                "model_display": model.display_name,
                                "model_id": model.model_id,
                                "model_revision": model.revision,
                                "generation_provenance": {"run": model_key},
                                "system_prompt_provenance": {
                                    "prompt_key": model.prompt.key,
                                    "prompt_revision": "446be5c605b56a60d4efe2526f0cbf55522c523a",
                                    "prompt_sha256": model.prompt.sha256,
                                },
                            }
                        )
                        judgments.append(row)
    return judgments


def _attach_paid_manifest(
    judgments: list[dict],
    *,
    model_keys: tuple[str, ...],
    profile_id: str = openrouter.GPT_OSS_120B_NITRO_DIRECT_PROFILE,
) -> tuple[list[dict], list[dict], list[dict]]:
    profile = openrouter._judge_profile(profile_id)
    endpoint = f"{profile['base_url']}{profile['endpoint_path']}"
    proxy = {"enabled": False}
    route = None
    allowed_models = sorted(profile["allowed_response_models"])
    request_protocols = {}
    for index, row in enumerate(judgments):
        custom_id = f"figure6-analysis-{index:05d}"
        generation_sha256 = hashlib.sha256(f"generation-{index}".encode()).hexdigest()
        request_protocols[custom_id] = {
            "provider": profile["provider"],
            "endpoint": endpoint,
            "api_style": profile["api_style"],
            "judge_profile": profile["id"],
            "model": profile["model"],
            "allowed_response_models": allowed_models,
            "temperature": profile["temperature"],
            "max_tokens": profile["max_tokens"],
            "provider_routing": profile["provider_routing"],
            "reasoning": profile["reasoning"],
            "response_format": profile["response_format"],
            "store": profile["store"],
            "route_mode": profile["route_mode"],
            "prompt_sha256": hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
            "generation_record_sha256": generation_sha256,
            "proxy": proxy,
            "route": route,
        }
        row.update(
            {
                "custom_id": custom_id,
                "generation_record_sha256": generation_sha256,
                "judge_profile": profile["id"],
                "judge_profile_label": profile["label"],
                "judge_model": profile["model"],
                "judge_provider": profile["provider"],
                "judge_requested_model": profile["model"],
                "judge_response_model": profile["model"],
                "judge_allowed_response_models": allowed_models,
                "judge_response_id": f"response-{index:05d}",
                "judge_request_id": f"request-{index:05d}",
                "judge_endpoint": endpoint,
                "judge_api_style": profile["api_style"],
                "judge_temperature": profile["temperature"],
                "judge_provider_routing": profile["provider_routing"],
                "judge_reasoning": profile["reasoning"],
                "judge_response_format": profile["response_format"],
                "judge_store": profile["store"],
                "judge_route_mode": profile["route_mode"],
                "judge_max_completion_tokens": profile["max_tokens"],
                "judge_proxy": proxy,
                "judge_route": route,
            }
        )
        if profile["response_format"].get("type") == "json_schema":
            awareness = "Yes" if row["awareness_conclusion"] == "yes" else "No"
            performance = "Yes" if row["performance_conclusion"] == "yes" else "No"
            row["judge_finish_reason"] = "stop"
            row["raw_judge_object"] = {
                "model_awareness": {
                    "conclusion": awareness,
                    "awareness_type": row["awareness_type"],
                    "evidence": [],
                    "reasoning": "fixture",
                },
                "hypothetical_recognition": {
                    "conclusion": "No",
                    "evidence": [],
                    "reasoning": "fixture",
                },
                "task_performance": {
                    "conclusion": performance,
                    "evidence": [],
                    "reasoning": "fixture",
                },
            }
    matrix = {
        "model_keys": list(model_keys),
        "generation_count": 5_400 * len(model_keys),
        "generations_per_model": 5_400,
        "task_pair_count": 100,
        "valences": list(FIGURE6_VALENCES),
        "configurations": list(FIGURE6_CONDITIONS),
        "replicates": [1, 2, 3],
        "pair_ids_sha256": "e" * 64,
        "generation_protocol_sha256": "f" * 64,
    }
    plan_document = {
        "schema": openrouter.PLAN_SCHEMA,
        "provider": profile["provider"],
        "endpoint": endpoint,
        "api_style": profile["api_style"],
        "judge_profile": profile["id"],
        "judge_model": profile["model"],
        "allowed_response_models": allowed_models,
        "judge_template_sha256": judge.PAPER_JUDGE_TEMPLATE_SHA256,
        "temperature": profile["temperature"],
        "max_tokens": profile["max_tokens"],
        "max_attempts_per_generation": 5,
        "concurrency": profile["concurrency"],
        "max_retry_after": profile["max_retry_after"],
        "proxy": proxy,
        "route": route,
        "route_mode": profile["route_mode"],
        "provider_routing": profile["provider_routing"],
        "reasoning": profile["reasoning"],
        "response_format": profile["response_format"],
        "store": profile["store"],
        "matrix": matrix,
        "requests": [
            {"custom_id": custom_id, "request": request_protocols[custom_id]} for custom_id in sorted(request_protocols)
        ],
    }
    plan_sha256 = openrouter._sha256_json(plan_document)
    core_plan_sha256 = openrouter._sha256_json(
        {key: value for key, value in plan_document.items() if key != "max_attempts_per_generation"}
    )
    manifest_plan = openrouter._manifest_plan(
        plan_sha256=plan_sha256,
        request_protocols=request_protocols,
        judge_profile=profile["id"],
        provider=profile["provider"],
        api_style=profile["api_style"],
        judge_model=profile["model"],
        allowed_response_models=allowed_models,
        judge_template_sha256=judge.PAPER_JUDGE_TEMPLATE_SHA256,
        temperature=profile["temperature"],
        max_tokens=profile["max_tokens"],
        max_attempts=5,
        concurrency=profile["concurrency"],
        max_retry_after=profile["max_retry_after"],
        endpoint=endpoint,
        proxy=proxy,
        route=route,
        route_mode=profile["route_mode"],
        provider_routing=profile["provider_routing"],
        reasoning=profile["reasoning"],
        response_format=profile["response_format"],
        store=profile["store"],
        matrix=matrix,
        plan_document=plan_document,
        core_plan_sha256=core_plan_sha256,
    )
    for row in judgments:
        row["judge_plan_sha256"] = plan_sha256
    artifact_digest = analysis._canonical_jsonl_digest(judgments)
    manifest = openrouter._new_manifest(manifest_plan)
    manifest["approvals"].append(
        {
            "approved_at": "2026-07-30T10:00:00Z",
            "confirmation": "--yes",
            "plan_sha256": plan_sha256,
            "reviewed_plan_sha256": plan_sha256,
            "reviewed_plan_hash_verified": True,
            "core_plan_sha256": core_plan_sha256,
            "expected_model_keys": list(model_keys),
            "pending_before_run": len(judgments),
            "resumed_successes": 0,
            "rescore_paid_errors": False,
            "authorized_paid_error_attempts": {},
            "authorized_attempt_ceiling": {},
            "amendment_index": None,
        }
    )
    manifest["events"].extend(
        [
            {
                "at": "2026-07-30T10:00:00Z",
                "event": "paid_run_approved",
                "approval_index": 1,
                "plan_sha256": plan_sha256,
            },
            {
                "at": "2026-07-30T11:00:00Z",
                "event": "run_completed",
                "approval_index": 1,
                "plan_sha256": plan_sha256,
                "core_plan_sha256": core_plan_sha256,
                "endpoint": endpoint,
                "provider": profile["provider"],
                "api_style": profile["api_style"],
                "judge_profile": profile["id"],
                "proxy": proxy,
                "route": route,
                "route_mode": profile["route_mode"],
                "temperature": profile["temperature"],
                "provider_routing": profile["provider_routing"],
                "reasoning": profile["reasoning"],
                "response_format": profile["response_format"],
                "store": profile["store"],
                "judge_model": profile["model"],
                "allowed_response_models": allowed_models,
                "observed_response_models": [profile["model"]],
                "response_request_ids_sha256": openrouter._sha256_json(
                    sorted(row["judge_request_id"] for row in judgments)
                ),
                "judgment_count": len(judgments),
                "normalized_output": "/archive/judgments.jsonl",
                "normalized_output_sha256": artifact_digest,
            },
        ]
    )
    manifest["updated_at"] = "2026-07-30T11:00:00Z"
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    artifacts = [
        {
            "path": "/archive/judgments.jsonl",
            "content_sha256": artifact_digest,
            "row_count": len(judgments),
            "record_start": 0,
            "record_end": len(judgments),
        }
    ]
    manifests = [
        {
            "path": "/archive/openrouter-manifest.json",
            "content_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "manifest": manifest,
        }
    ]
    return artifacts, manifests, []


def test_default_publication_matrix_is_37800_with_300_per_cell():
    judgments = _registered_publication_matrix(tuple(MODEL_SPECS))

    result = analyze_judgments(judgments)

    assert EXPECTED_JUDGMENT_COUNT == 37_800
    assert EXPECTED_CELL_COUNT == 300
    assert len(judgments) == 37_800
    assert len(result.rows) == 126
    assert all(row["n"] == 300 for row in result.rows)
    assert result.summary["publication_complete"] is True
    assert all(len(task_ids) == 100 for task_ids in result.summary["expected"]["task_ids_by_valence"].values())
    assert len(result.summary["expected"]["pair_ids"]) == 100


@pytest.mark.parametrize(
    "profile_id",
    [
        openrouter.GPT_OSS_120B_NITRO_DIRECT_PROFILE,
        openrouter.OPENROUTER_GPT_56_LUNA_DIRECT_PROFILE,
    ],
)
def test_current_three_qwen_direct_profile_scope_is_strict_complete_at_16200(
    tmp_path: Path,
    profile_id: str,
):
    pytest.importorskip("matplotlib")
    judgments = _registered_publication_matrix(CURRENT_QWEN_MODEL_KEY_ORDER)
    profile = openrouter._judge_profile(profile_id)
    allowed_response_models = list(profile["allowed_response_models"])
    for row in judgments:
        row.update(
            {
                "judge_profile": profile["id"],
                "judge_profile_label": profile["label"],
                "judge_model": profile["model"],
                "judge_provider": profile["provider"],
                "judge_response_model": profile["model"],
                "judge_allowed_response_models": allowed_response_models,
                "judge_max_completion_tokens": profile["max_tokens"],
            }
        )
    with pytest.raises(PublicationValidationError, match="matching paid judge manifests"):
        analyze_judgments(
            judgments,
            expected_model_keys=CURRENT_QWEN_MODEL_KEY_ORDER,
            expected_judge_profile=profile["id"],
        )
    artifacts = []
    manifests = []
    route_attestations = []
    for model_index, model_key in enumerate(CURRENT_QWEN_MODEL_KEY_ORDER):
        start = model_index * EXPECTED_MODEL_JUDGMENT_COUNT
        end = start + EXPECTED_MODEL_JUDGMENT_COUNT
        model_artifacts, model_manifests, model_route_attestations = _attach_paid_manifest(
            judgments[start:end],
            model_keys=(model_key,),
            profile_id=profile_id,
        )
        model_artifacts[0].update(
            {
                "path": f"/archive/{model_key}-judgments.jsonl",
                "record_start": start,
                "record_end": end,
            }
        )
        model_manifests[0]["path"] = f"/archive/{model_key}-manifest.json"
        artifacts.extend(model_artifacts)
        manifests.extend(model_manifests)
        route_attestations.extend(model_route_attestations)

    result = analyze_judgments(
        judgments,
        judgment_artifacts=artifacts,
        judge_manifests=manifests,
        route_attestations=route_attestations,
        expected_model_keys=CURRENT_QWEN_MODEL_KEY_ORDER,
        expected_judge_profile=profile["id"],
    )

    assert len(judgments) == 16_200
    assert EXPECTED_MODEL_JUDGMENT_COUNT == 5_400
    assert len(result.rows) == 54
    assert result.summary["publication_complete"] is True
    assert result.summary["scope_complete"] is True
    assert result.summary["scope_kind"] == "registered_model_subset"
    assert result.summary["result_label"] == STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL
    assert result.summary["expected"]["model_keys"] == list(CURRENT_QWEN_MODEL_KEY_ORDER)
    assert result.summary["expected"]["judgment_count"] == 16_200
    assert result.summary["expected"]["judgments_per_model"] == 5_400
    assert result.summary["observed"]["valid_unique_records_by_model"] == {
        model_key: 5_400 for model_key in CURRENT_QWEN_MODEL_KEY_ORDER
    }
    assert result.summary["provenance"]["paid_manifest_verification"]["status"] == "verified"
    assert result.summary["provenance"]["paid_manifest_verification"]["artifact_count"] == 3
    assert result.summary["provenance"]["paid_manifest_verification"]["manifest_count"] == 3
    assert "Qwen-only subset" in result.summary["scope_label"]
    assert "not the full seven-model paper reproduction" in result.summary["source_note"]
    assert len(validate_plot_rows(result.rows, summary=result.summary)) == 54

    png = tmp_path / f"qwen-only-{profile_id}.png"
    pdf = tmp_path / f"qwen-only-{profile_id}.pdf"
    render_figure6(result.rows, summary=result.summary, png_path=png, pdf_path=pdf)
    assert result.summary["scope_label"].encode() in png.read_bytes()
    assert profile["label"].encode() in png.read_bytes()
    assert pdf.read_bytes().startswith(b"%PDF")


def test_strict_openrouter_analysis_rejects_qwen36_scope() -> None:
    judgments = _registered_publication_matrix(("qwen36",))
    profile = openrouter._judge_profile(openrouter.GPT_OSS_120B_NITRO_DIRECT_PROFILE)
    for row in judgments:
        row.update(
            {
                "judge_profile": profile["id"],
                "judge_model": profile["model"],
                "judge_provider": "OpenRouter",
                "judge_response_model": profile["model"],
                "judge_allowed_response_models": list(profile["allowed_response_models"]),
            }
        )
    with pytest.raises(PublicationValidationError, match="exactly qwen32"):
        analyze_judgments(
            judgments,
            expected_model_keys=("qwen36",),
            expected_judge_profile=profile["id"],
        )


def test_qwen_subset_plot_rejects_full_result_label_and_bad_per_model_count():
    rows = [
        {**row, "publication_complete": True, "fixture_mode": False}
        for row in fixture_rows()
        if row["model_key"] in CURRENT_QWEN_MODEL_KEY_ORDER
    ]
    scope_label = "Qwen-only subset (3 of 7 models; 16,200 strict judgments)"
    summary = {
        "publication_complete": True,
        "scope_complete": True,
        "scope_kind": "registered_model_subset",
        "scope_label": scope_label,
        "result_label": STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL,
        "source_note": f"Computed fixture for the {scope_label}; not the full reproduction. OpenRouter Muse alternative judge.",
        "plot_label": OPENROUTER_MUSE_ALTERNATIVE_LABEL,
        "expected": {
            "model_keys": list(CURRENT_QWEN_MODEL_KEY_ORDER),
            "model_displays": [MODEL_SPECS[key].display_name for key in CURRENT_QWEN_MODEL_KEY_ORDER],
            "valences": list(FIGURE6_VALENCES),
            "configs": list(FIGURE6_CONDITIONS),
            "task_count": 100,
            "replicates": [1, 2, 3],
            "judgment_count": 16_200,
            "judgments_per_model": 5_400,
            "cell_denominator": 300,
        },
        "observed": {
            "valid_unique_records": 16_200,
            "valid_unique_records_by_model": {model_key: 5_400 for model_key in CURRENT_QWEN_MODEL_KEY_ORDER},
            "aggregate_rows": 54,
        },
        "provenance": {
            "expected_judge_model": "meta/muse-spark-1.1",
            "judge_provider": "OpenRouter",
        },
    }

    with pytest.raises(ValueError, match="unsupported complete-result label"):
        validate_plot_rows(rows, summary={**summary, "result_label": "complete_reproduction"})

    bad_observed = {
        **summary["observed"],
        "valid_unique_records_by_model": {
            **summary["observed"]["valid_unique_records_by_model"],
            "qwen32": 5_399,
        },
    }
    with pytest.raises(ValueError, match="5,400 valid unique judgments for every model"):
        validate_plot_rows(rows, summary={**summary, "observed": bad_observed})


def test_analysis_cli_forwards_repeatable_expected_model_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    monkeypatch.setattr(
        analysis,
        "_read_judgment_artifacts",
        lambda _: ([{"fixture": True}], [{"artifact": True}]),
    )
    monkeypatch.setattr(
        analysis,
        "_read_judge_manifests",
        lambda _: [{"manifest": True}],
    )
    monkeypatch.setattr(
        analysis,
        "_read_route_attestations",
        lambda _: [],
    )

    def fake_analyze(judgments, **kwargs):
        captured["judgments"] = judgments
        captured["kwargs"] = kwargs
        return analysis.AnalysisResult(
            rows=(),
            summary={"publication_complete": True, "result_label": STRICT_SUBSET_ALTERNATIVE_RESULT_LABEL},
            diagnostics={},
        )

    monkeypatch.setattr(analysis, "analyze_judgments", fake_analyze)
    monkeypatch.setattr(analysis, "write_analysis_outputs", lambda *args, **kwargs: None)
    exit_code = analysis.main(
        [
            "--judgments",
            str(tmp_path / "judgments.jsonl"),
            "--output-csv",
            str(tmp_path / "aggregation.csv"),
            "--summary-json",
            str(tmp_path / "summary.json"),
            "--judge-manifest",
            str(tmp_path / "manifest.json"),
            "--expected-model-key",
            "qwen32",
            "--expected-model-key",
            "qwen_mo_mid",
            "--expected-model-key",
            "qwen_mo_post",
            "--expected-judge-profile",
            openrouter.GPT_OSS_120B_NITRO_DIRECT_PROFILE,
        ]
    )

    assert exit_code == 0
    assert captured["judgments"] == [{"fixture": True}]
    assert captured["kwargs"]["expected_model_keys"] == list(CURRENT_QWEN_MODEL_KEY_ORDER)
    assert captured["kwargs"]["judgment_artifacts"] == [{"artifact": True}]
    assert captured["kwargs"]["judge_manifests"] == [{"manifest": True}]
    assert captured["kwargs"]["route_attestations"] == []
    assert captured["kwargs"]["expected_judge_profile"] == openrouter.GPT_OSS_120B_NITRO_DIRECT_PROFILE


def test_plot_order_labels_completeness_and_forbidden_label():
    rows = fixture_rows()
    assert len(validate_plot_rows(rows, fixture_mode=True)) == 126
    assert MODEL_DISPLAY_ORDER == (
        "Qwen3.6-27B",
        "Qwen3-32B",
        "Qwen3-32B MO (midtrained)",
        "Qwen3-32B MO (posttrained)",
        "Llama-3.3-70B-Instruct",
        "Llama-3.3-70B MO (midtrained)",
        "Llama-3.3-70B MO (posttrained)",
    )
    assert [CONFIG_LABELS[name] for name in FIGURE6_CONDITIONS] == [
        "BL",
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
    ]
    plot_source = Path(__file__).parents[1] / "figure6_plot.py"
    assert "con" + "trol" not in plot_source.read_text(encoding="utf-8").casefold()
    incomplete = rows.copy()
    incomplete[0] = {**incomplete[0], "fixture_mode": False}
    with pytest.raises(ValueError, match="fixture mode requires"):
        validate_plot_rows(incomplete, fixture_mode=True)
    with pytest.raises(ValueError, match="incomplete diagnostic"):
        validate_plot_rows(rows)


def test_default_gpt5_plot_accepts_pre_response_identity_summary_shape():
    rows = [{**row, "publication_complete": True, "fixture_mode": False} for row in fixture_rows()]
    legacy_summary = {
        "publication_complete": True,
        "result_label": "complete_reproduction",
        "observed": {"valid_unique_records": 37_800},
        "provenance": {"judge_model": "gpt-5"},
    }
    assert len(validate_plot_rows(rows, summary=legacy_summary)) == 126


def test_png_and_pdf_fixture_smoke(tmp_path: Path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "figure6.png"
    pdf = tmp_path / "figure6.pdf"

    render_figure6(fixture_rows(), png_path=png, pdf_path=pdf, fixture_mode=True)

    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 20_000
    assert pdf.stat().st_size > 10_000


def _complete_alternative_plot_inputs() -> tuple[list[dict], dict]:
    rows = [{**row, "publication_complete": True, "fixture_mode": False} for row in fixture_rows()]
    summary = {
        "publication_complete": True,
        "result_label": "complete_reproduction_user_pinned_alternative_judge",
        "source_note": f"Computed reproduction; {OPENROUTER_MUSE_ALTERNATIVE_LABEL}.",
        "plot_label": OPENROUTER_MUSE_ALTERNATIVE_LABEL,
        "observed": {"valid_unique_records": 37_800},
        "provenance": {
            "expected_judge_model": "meta/muse-spark-1.1",
            "judge_provider": "OpenRouter",
        },
    }
    return rows, summary


def test_alternative_plot_summary_requires_exact_rendered_label(tmp_path: Path):
    rows, summary = _complete_alternative_plot_inputs()
    assert len(validate_plot_rows(rows, summary=summary)) == 126

    missing_label = {**summary, "plot_label": None}
    with pytest.raises(ValueError, match="plot_label"):
        validate_plot_rows(rows, summary=missing_label)

    with pytest.raises(ValueError, match="both the title and source note"):
        render_figure6(
            rows,
            summary=summary,
            title="Generic Figure 6 title",
            source_note=f"Computed reproduction; {OPENROUTER_MUSE_ALTERNATIVE_LABEL}.",
            png_path=tmp_path / "rejected.png",
            pdf_path=tmp_path / "rejected.pdf",
        )
    assert not (tmp_path / "rejected.png").exists()


def test_openrouter_muse_alternative_label_is_rendered_and_embedded(tmp_path: Path):
    pytest.importorskip("matplotlib")
    rows, summary = _complete_alternative_plot_inputs()
    png = tmp_path / "muse.png"
    pdf = tmp_path / "muse.pdf"
    render_figure6(rows, summary=summary, png_path=png, pdf_path=pdf)
    assert OPENROUTER_MUSE_ALTERNATIVE_LABEL.encode() in png.read_bytes()
    assert pdf.read_bytes().startswith(b"%PDF")
    plot_source = Path(__file__).parents[1] / "figure6_plot.py"
    assert "figure.suptitle(title" in plot_source.read_text(encoding="utf-8")
