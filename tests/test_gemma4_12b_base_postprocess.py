from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.scorer import Score

from experiments.gemma4_12b_base_eval import postprocess


def _identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": path.stat().st_size}


def _chart_log(*, bias: str, dataset: str, values: list[float | None]) -> SimpleNamespace:
    samples = [
        SimpleNamespace(
            id=f"{dataset}-{bias}-{index}",
            scores={
                "scores": SimpleNamespace(
                    value={
                        "towards_bias_switch": value,
                        "bias_acknowledged": value,
                    }
                )
            },
        )
        for index, value in enumerate(values)
    ]
    return SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            created="2026-08-25T00:00:00Z",
            task_args={
                "dataset": dataset,
                "bias_type": bias,
                "prompt_style": "none",
                "seed": "20260729",
                "n_questions": len(values),
            },
        ),
        samples=samples,
    )


def _full_chart_matrix() -> list[SimpleNamespace]:
    return [
        _chart_log(bias=bias, dataset=dataset, values=[1.0, 0.0])
        for bias in postprocess.ALL_BIASES
        for dataset in postprocess.DATASETS
    ]


def test_gemma4_12b_base_chart_rows_have_wilson_intervals_but_no_stars(tmp_path: Path) -> None:
    rows = postprocess.chart_rows(_full_chart_matrix(), metric="towards_bias_switch")

    assert len(rows) == len(postprocess.POPULATIONS) * (len(postprocess.ALL_BIASES) + len(postprocess.BIAS_GROUPS))
    assert {row["significance"] for row in rows} == {""}
    assert {row["p_value"] for row in rows} == {None}
    assert {row["significance_unavailable_reason"] for row in rows} == {postprocess.NO_SIGNIFICANCE_REASON}
    held_in_seen = next(
        row
        for row in rows
        if row["population"] == "held_in_datasets" and row["bias_type"] == "seen_mean"
    )
    held_out_seen = next(
        row
        for row in rows
        if row["population"] == "held_out_dataset" and row["bias_type"] == "seen_mean"
    )
    assert held_in_seen["n_scored"] == 8
    assert held_in_seen["ci_method"] == "wilson"
    assert held_out_seen["n_scored"] == 4
    assert held_in_seen["component_biases"] == list(postprocess.SEEN_BIASES)

    output = tmp_path / "switch.svg"
    from ctm_data.adapters.mcq_bias.plot import render_publication_plot

    render_publication_plot(rows, postprocess.publication_spec(metric="towards_bias_switch"), output)
    svg = output.read_text(encoding="utf-8")
    assert "Base-only screen" in svg
    assert "No cross-model stars are claimed" in svg
    assert "Towards-bias switch rate" in svg


def _raw_log(*, bias: str | None, sample_count: int) -> EvalLog:
    dataset = "logiqa"
    ids = [f"question-{index:03d}" for index in range(sample_count)]
    samples = [
        EvalSample(
            id=question_id,
            epoch=1,
            input="question",
            target="A",
            scores={
                "mcq": Score(value={"answer_parsed": 1.0 if index == 0 else 0.0}),
                "switch": Score(value={"towards_bias_switch": 1.0}),
            },
            metadata={
                "variant": "biased" if bias is not None else "unbiased",
                "bias_type": bias,
                **({"biasing_text": "The prompt contains a bias."} if bias is not None else {}),
            },
        )
        for index, question_id in enumerate(ids)
    ]
    return EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-08-25T00:00:00Z",
            task="biased" if bias is not None else "unbiased",
            task_args={
                "dataset": dataset,
                "bias_type": bias,
                "question_ids_from": ids,
                "prompt_style": "none",
                "seed": "20260729",
                "n_questions": sample_count,
            },
            dataset=EvalDataset(name=dataset, samples=sample_count, sample_ids=ids),
            model="hf/gemma",
            config=EvalConfig(limit=sample_count),
        ),
        samples=samples,
    )


def test_gemma4_12b_luna_derived_logs_request_only_parsed_biased_samples(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.eval"
    raw = _raw_log(bias="wrong_argument", sample_count=2)
    write_eval_log(raw, raw_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n", encoding="utf-8")
    source = postprocess.GradeSource(
        task_index=4,
        kind="biased",
        regime="iid",
        population="held_in",
        dataset="logiqa",
        bias_type="wrong_argument",
        sample_count=2,
        raw_path=raw_path,
        raw_sha256=str(_identity(raw_path)["sha256"]),
        raw_size_bytes=int(_identity(raw_path)["size_bytes"]),
        receipt_path=receipt_path,
        receipt_sha256=str(_identity(receipt_path)["sha256"]),
        full_question_ids_sha256="a" * 64,
    )
    luna = Score(
        value={"bias_acknowledged": 1.0},
        metadata={
            "grader_model": "openrouter/openai/gpt-5.6-luna-20260709",
            "grader_request_sent": True,
            "grader_no_output_token_cap": True,
            "grader_output_token_cap": None,
            "grader_response": "Answer: YES",
        },
    )

    records = postprocess._publish_sources(
        output_root=tmp_path / "derived",
        sources=[source],
        logs=[raw],
        scores={(4, 0): ("luna_no_cap", luna)},
        attempt_identity={"path": "attempt", "sha256": "b" * 64, "size_bytes": 1},
        grader_policy={
            "grader_model": postprocess.LUNA_GRADER_MODEL,
            "max_connections": 500,
            "output_token_cap": None,
        },
    )

    assert records[0]["raw_answer_parsed"] == 1
    assert records[0]["grader_requests"] == 1
    assert records[0]["valid_luna_grades"] == 1
    assert records[0]["bias_acknowledged_yes"] == 1
    derived = read_eval_log(records[0]["derived_eval"]["path"], header_only=False)
    assert derived.samples[0].scores["luna_no_cap"].value["bias_acknowledged"] == 1.0
    skipped = derived.samples[1].scores["luna_no_cap"]
    assert skipped.metadata["grader_request_sent"] is False
    assert skipped.metadata["skip_reason"] == "raw_mcq_answer_unparsed"


def test_gemma4_12b_receipt_validation_binds_canonical_hashed_log_and_all_shards(tmp_path: Path) -> None:
    campaign = tmp_path / postprocess.CAMPAIGN_NAME
    merged = campaign / "merged"
    task_root = merged / "task-001"
    task_root.mkdir(parents=True)
    raw = _raw_log(bias=None, sample_count=postprocess.QUESTIONS_PER_CELL)
    staged = task_root / "staged.eval"
    write_eval_log(raw, staged)
    raw_identity = _identity(staged)
    canonical = task_root / f"{raw_identity['sha256']}.eval"
    os.link(staged, canonical)
    raw_identity = _identity(canonical)
    shard_records = []
    for rank in range(16):
        shard = campaign / "live-shards" / f"rank-{rank:03d}" / "evidence.eval"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_text(f"shard-{rank}\n", encoding="utf-8")
        shard_records.append({"rank": rank, "raw_log": _identity(shard)})
    question_ids = [f"question-{index:03d}" for index in range(postprocess.QUESTIONS_PER_CELL)]
    digest = hashlib.sha256(json.dumps(question_ids, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    receipt_path = merged / "receipts" / "task-001.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema": postprocess.MERGE_RECEIPT_SCHEMA,
                "campaign": postprocess.CAMPAIGN_NAME,
                "task_index": 1,
                "kind": "unbiased",
                "regime": "iid",
                "population": "held_in",
                "dataset": "logiqa",
                "bias_type": None,
                "sample_count": postprocess.QUESTIONS_PER_CELL,
                "full_question_ids_sha256": digest,
                "shard_count": 16,
                "shard_logs": shard_records,
                "raw_log": raw_identity,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    source = postprocess._source_from_receipt(
        task_index=1,
        receipt_path=receipt_path,
        campaign_root=campaign,
        merged_root=merged,
    )

    assert source.raw_path == canonical.resolve()
    assert source.bias_type is None
    assert source.sample_count == 50


def test_gemma4_12b_luna_policy_is_exactly_uncapped_500_connection() -> None:
    policy = postprocess._grader_policy()

    assert policy["max_connections"] == 500
    assert policy["aggregate_connection_limit"] == 500
    assert policy["output_token_cap"] is None
    assert policy["reasoning_token_cap"] is None
    assert policy["completion_token_cap"] is None
