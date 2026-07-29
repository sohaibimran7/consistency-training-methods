"""Offline tests for Figure 6 judge batching, aggregation, and rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ctm_data.adapters.eval_awareness import figure6_judge as judge
from ctm_data.adapters.eval_awareness.figure6_analysis import (
    EXPECTED_CELL_COUNT,
    EXPECTED_JUDGMENT_COUNT,
    PublicationValidationError,
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


def test_default_publication_matrix_is_37800_with_300_per_cell():
    judgments = []
    for model_key, model in MODEL_SPECS.items():
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

    result = analyze_judgments(judgments)

    assert EXPECTED_JUDGMENT_COUNT == 37_800
    assert EXPECTED_CELL_COUNT == 300
    assert len(judgments) == 37_800
    assert len(result.rows) == 126
    assert all(row["n"] == 300 for row in result.rows)
    assert result.summary["publication_complete"] is True
    assert all(len(task_ids) == 100 for task_ids in result.summary["expected"]["task_ids_by_valence"].values())
    assert len(result.summary["expected"]["pair_ids"]) == 100


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


def test_png_and_pdf_fixture_smoke(tmp_path: Path):
    pytest.importorskip("matplotlib")
    png = tmp_path / "figure6.png"
    pdf = tmp_path / "figure6.pdf"

    render_figure6(fixture_rows(), png_path=png, pdf_path=pdf, fixture_mode=True)

    assert png.read_bytes().startswith(b"\x89PNG")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert png.stat().st_size > 20_000
    assert pdf.stat().st_size > 10_000
