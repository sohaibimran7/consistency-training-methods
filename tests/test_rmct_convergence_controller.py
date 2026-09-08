"""Focused fail-closed contracts for the simple RMCT checkpoint controller."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.rmct_convergence import controller


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _metric_record(step: int, *, gap: float = 0.10, wrong_count: int = 2, suggested_count: int = 2, skipped: int = 0) -> dict:
    record: dict[str, object] = {
        "step": step,
        "train/optimizer_step": step,
        "train/skipped_empty_batch": skipped,
    }
    for index, count in (("1", wrong_count), ("2", suggested_count)):
        record[f"train/consistency_gap_abs_sum_{index}"] = gap * count
        record[f"train/consistency_gap_abs_count_{index}"] = count
        if count:
            record[f"train/consistency_gap_abs_mean_{index}"] = gap
    return record


def _metrics_file(
    root: Path,
    segment_index: int,
    *,
    gap: float = 0.10,
    wrong_count: int = 2,
    suggested_count: int = 2,
    records: list[dict] | None = None,
) -> Path:
    start = segment_index * controller.UPDATES_PER_SEGMENT + 1
    rows = records or [
        _metric_record(step, gap=gap, wrong_count=wrong_count, suggested_count=suggested_count)
        for step in range(start, start + controller.UPDATES_PER_SEGMENT)
    ]
    path = root / f"segment-{segment_index}" / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _source_from_metrics(root: Path, segment_index: int, **kwargs) -> Path:
    metrics = _metrics_file(root, segment_index, **kwargs)
    output = metrics.parent / "segment" / "rmct-convergence-source-metrics.json"
    return controller.extract_source_metrics(metrics_jsonl=metrics, segment_index=segment_index, output=output)


def _boundary_receipts(root: Path, segment_index: int) -> tuple[Path, Path]:
    start = segment_index * controller.UPDATES_PER_SEGMENT + 1
    end = start + controller.UPDATES_PER_SEGMENT - 1
    checkpoint = _write_json(
        root / f"segment-{segment_index}" / "segment" / "checkpoint-receipt.json",
        {
            "schema": controller.CHECKPOINT_RECEIPT_SCHEMA,
            "segment_index": segment_index,
            "optimizer_step": end,
            "sealed": True,
        },
    )
    completion = _write_json(
        root / f"segment-{segment_index}" / "segment" / "completion-receipt.json",
        {
            "schema": controller.COMPLETION_RECEIPT_SCHEMA,
            "segment_index": segment_index,
            "optimizer_step_start": start,
            "optimizer_step_end": end,
            "optimizer_steps": controller.UPDATES_PER_SEGMENT,
            "sealed": True,
        },
    )
    return checkpoint, completion


def _guard(root: Path, segment_index: int, *, predecessor: Path | None = None, **source_kwargs):
    source = _source_from_metrics(root, segment_index, **source_kwargs)
    checkpoint, completion = _boundary_receipts(root, segment_index)
    return controller.guard_from_paths(
        source_metrics=source,
        checkpoint_receipt=checkpoint,
        completion_receipt=completion,
        predecessor_receipt=predecessor,
        output_directory=root / "decisions",
    )


@pytest.mark.parametrize(
    ("rows", "error"),
    [
        (lambda rows: rows[:-1], "missing sealed optimizer steps"),
        (lambda rows: [*rows, dict(rows[0])], "duplicate optimizer step"),
        (lambda rows: [rows[1], rows[0], *rows[2:]], "out-of-order optimizer step"),
        (
            lambda rows: [
                *rows[:4],
                {**rows[4], "train/skipped_empty_batch": 1},
                *rows[5:],
            ],
            "skipped or empty batch",
        ),
    ],
)
def test_extract_source_rejects_missing_duplicate_out_of_order_and_skipped_optimizer_mutations(
    tmp_path: Path, rows, error: str
) -> None:
    original = [_metric_record(step) for step in range(1, 17)]
    metrics = _metrics_file(tmp_path, 0, records=rows(original))

    with pytest.raises(controller.SourceMetricsError, match=error):
        controller.extract_source_metrics(
            metrics_jsonl=metrics,
            segment_index=0,
            output=tmp_path / "source.json",
        )


def test_first_sealed_segment_continues_and_replay_is_idempotent_and_hash_bound(tmp_path: Path) -> None:
    first = _guard(tmp_path, 0, gap=0.30)

    assert first.decision == "continue"
    assert first.receipt["window"]["conditions"]["first_segment"] is True
    replay = _guard(tmp_path, 0, gap=0.30)
    assert replay.path == first.path
    assert replay.status == "resumed"
    assert controller.verify_decision_receipt(first.path)["decision"] == "continue"

    source_path = Path(first.receipt["source_metrics"]["path"])
    source_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(controller.DecisionReceiptError, match="changed after"):
        controller.verify_decision_receipt(first.path)


def test_step32_boundary_converges_at_inclusive_gap_and_change_thresholds(tmp_path: Path) -> None:
    first = _guard(tmp_path, 0, gap=0.11)
    second = _guard(tmp_path, 1, predecessor=first.path, gap=0.10)

    assert second.decision == "converged"
    receipt = controller.verify_decision_receipt(second.path)
    assert receipt["window"]["current"]["weighted_abs_gap"] == pytest.approx(0.10)
    assert receipt["window"]["conditions"]["absolute_gap_change"] == pytest.approx(0.01)
    assert controller.successor_action(second.path) == {"permit_training": False, "successor_action": "no_op"}
    with pytest.raises(controller.NonContinuationDecision, match="no-op"):
        controller.require_continue(second.path)


def test_partial_per_bias_coverage_blocks_an_otherwise_qualifying_step32_window(tmp_path: Path) -> None:
    first = _guard(tmp_path, 0, gap=0.10)
    second = _guard(tmp_path, 1, predecessor=first.path, gap=0.10, wrong_count=1, suggested_count=2)

    assert second.decision == "continue"
    receipt = controller.verify_decision_receipt(second.path)
    current = receipt["window"]["current"]
    # Overall coverage is still 75%, but the named wrong_argument stratum is
    # independently retained and blocks convergence rather than disappearing
    # into a pooled average.
    assert current["coverage"]["by_bias"]["wrong_argument"]["passes"] is False
    assert receipt["window"]["conditions"]["current_coverage_passes"] is False


def test_hard_cap_is_a_terminal_safe_noop_after_512_optimizer_steps(tmp_path: Path) -> None:
    predecessor: Path | None = None
    result = None
    for segment_index in range(32):
        result = _guard(tmp_path, segment_index, predecessor=predecessor, gap=0.20)
        predecessor = result.path

    assert result is not None
    assert result.decision == "capped"
    receipt = controller.verify_decision_receipt(result.path)
    assert receipt["target"]["optimizer_step_end"] == 512
    assert receipt["afterok"] == {"permit_training": False, "successor_action": "no_op"}

