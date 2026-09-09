"""Focused contracts for the RMCT-256 training-gap plateau controller."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from experiments.rmct_256_convergence import plateau

SELECTION_SHA256 = "a" * 64


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _identity(path: Path) -> dict[str, str]:
    import hashlib

    return {"path": str(path.resolve()), "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _question_ids(block: int, *, changed: bool = False) -> list[str]:
    return [
        f"fixed-block-{block}-question-{index}{'-changed' if changed and index == 0 else ''}" for index in range(64)
    ]


def _source_receipt(
    root: Path,
    pass_index: int,
    segment_index: int,
    *,
    abs_sums: int | list[int],
    abs_counts: int | list[int] = 4,
    abs_means: list[float] | None = None,
    heldout: dict | None = None,
    changed_question_ids: bool = False,
) -> Path:
    """Write the extractor-facing evidence bundle that publish-metrics consumes."""

    position = plateau.SegmentPosition(pass_index, segment_index)
    if isinstance(abs_sums, int):
        abs_sums = [abs_sums] * 16
    if isinstance(abs_counts, int):
        abs_counts = [abs_counts] * 16
    assert len(abs_sums) == len(abs_counts) == 16
    if abs_means is not None:
        assert len(abs_means) == 16
    root.mkdir(parents=True, exist_ok=True)
    raw = root / "metrics.jsonl"
    completion = root / "segment-completion.json"
    normalized = root / "rmct256-convergence-gap-metrics.jsonl"
    raw.write_bytes(_json_bytes({"source": "authoritative-tinker-jsonlogger", "segment": position.checkpoint_step}))
    completion.write_bytes(_json_bytes({"checkpoint_step": position.checkpoint_step, "sealed": True}))
    sum_key, count_key, mean_key = plateau._normalized_metric_keys()
    lines = []
    for update_index, (absolute_sum, absolute_count) in enumerate(zip(abs_sums, abs_counts), start=1):
        absolute_mean = (
            abs_means[update_index - 1]
            if abs_means is not None
            else (absolute_sum / absolute_count if absolute_count else 0.0)
        )
        lines.append(
            _json_bytes(
                {
                    "global_step": position.checkpoint_step - 16 + update_index,
                    "metrics": {
                        sum_key: absolute_sum,
                        count_key: absolute_count,
                        mean_key: absolute_mean,
                    },
                }
            )
        )
    normalized.write_bytes(b"".join(lines))
    receipt = {
        "schema": plateau.EXTRACTED_METRICS_SOURCE_SCHEMA,
        "selection": {"content_sha256": SELECTION_SHA256},
        "segment": {
            "pass_index": pass_index,
            "segment_index": segment_index,
            "checkpoint_step": position.checkpoint_step,
            "updates": 16,
            "questions": 64,
            "row_offset": position.row_offset,
            "question_ids": _question_ids(segment_index, changed=changed_question_ids),
        },
        "checkpoint": {
            "path": f"checkpoints/pass-{pass_index}-block-{segment_index}",
            "sha256": f"{position.checkpoint_step:064x}",
            "step": position.checkpoint_step,
        },
        "raw_metrics_jsonl": _identity(raw),
        "normalized_metrics_jsonl": _identity(normalized),
        "segment_completion_receipt": _identity(completion),
    }
    if heldout is not None:
        receipt["heldout_diagnostics"] = heldout
    path = root / "rmct256-convergence-source-receipt.json"
    path.write_bytes(_json_bytes(receipt))
    return path


def _publish_pass(
    metrics_dir: Path,
    pass_index: int,
    sums_by_block: list[int],
    *,
    heldout: dict | None = None,
    changed_block: int | None = None,
) -> list[Path]:
    paths = []
    for block in range(4):
        source = _source_receipt(
            metrics_dir / "_sources" / f"pass-{pass_index}-block-{block}",
            pass_index,
            block,
            abs_sums=sums_by_block[block],
            heldout=heldout,
            changed_question_ids=block == changed_block,
        )
        paths.append(plateau.publish_metrics_from_source_receipt(metrics_dir, source).path)
    return paths


def _guard(metrics_dir: Path, decisions_dir: Path, pass_index: int, segment_index: int = 3):
    return plateau.guard_from_directory(
        metrics_dir,
        output_directory=decisions_dir,
        target=plateau.SegmentPosition(pass_index, segment_index),
        expected_selection_sha256=SELECTION_SHA256,
    )


def test_segment_score_uses_authoritative_abs_sums_and_sample_weighting(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    source = _source_receipt(
        tmp_path / "source",
        1,
        0,
        abs_sums=[4, 0, 2, 0] * 4,
    )

    published = plateau.publish_metrics_from_source_receipt(metrics_dir, source)
    metric = plateau.load_segment_metrics(published.path)

    # 16 updates x four questions; the controller sums absolute per-question
    # gaps rather than trusting a signed or update-level averaged gap.
    assert metric.mean_absolute_gap == Decimal("0.375")
    assert metric.updates[0].absolute_sum == Decimal("4")
    assert metric.updates[0].absolute_count == 4


def test_publish_rejects_any_update_with_missing_per_question_gap(tmp_path: Path):
    source = _source_receipt(tmp_path / "source", 1, 0, abs_sums=2, abs_counts=[4] * 15 + [3])

    with pytest.raises(plateau.MetricValidationError, match="abs_count must be exactly 4"):
        plateau.publish_metrics_from_source_receipt(tmp_path / "metrics", source)


def test_publish_rejects_a_mean_that_does_not_match_the_authoritative_sum_and_count(tmp_path: Path):
    source = _source_receipt(tmp_path / "source", 1, 0, abs_sums=2, abs_means=[0.5] * 15 + [0.75])

    with pytest.raises(plateau.MetricValidationError, match="abs_mean must equal abs_sum / abs_count"):
        plateau.publish_metrics_from_source_receipt(tmp_path / "metrics", source)


def test_complete_passes_use_matched_blocks_and_threshold_qualified_plateau_reference(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 2, [2, 2, 2, 2])  # block reduction 1/4 = 0.25 > default min_delta

    second = _guard(metrics_dir, decisions_dir, 2)
    assert second.decision == "continue"
    second_receipt = plateau.verify_decision_receipt(second.path)
    assert second_receipt["complete_passes"][1]["qualifies_as_improvement"] is True
    assert (
        second_receipt["complete_passes"][1]["matched_block_reductions_from_previous_pass"] == [{"decimal": "0.25"}] * 4
    )
    assert second_receipt["plateau_reference_pass_endpoint"]["pass_index"] == 2
    assert second_receipt["raw_best_pass_endpoint"]["pass_index"] == 2
    assert len(second_receipt["checkpoint_receipts"]) == 8

    _publish_pass(metrics_dir, 3, [2, 2, 2, 2])
    _publish_pass(metrics_dir, 4, [2, 2, 2, 2])
    fourth = _guard(metrics_dir, decisions_dir, 4)
    assert fourth.decision == "converged"
    assert fourth.guard_exit_code == 0
    receipt = plateau.verify_decision_receipt(fourth.path)
    assert receipt["reason"] == "plateau_patience_exhausted_at_hard_cap"
    assert receipt["afterok"] == {"permit_training": False, "successor_action": "no_op"}
    assert receipt["raw_best_pass_endpoint"]["pass_index"] == 2
    assert receipt["plateau_reference_pass_endpoint"]["pass_index"] == 2
    assert len(receipt["checkpoint_receipts"]) == 16


def test_rechecking_a_prior_pass_ignores_later_partial_pass_metrics(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 2, [2, 2, 2, 2])

    # Before each later segment starts, the afterok launcher rechecks pass 1.
    # It must not treat already-written pass-2 artifacts as contamination of
    # the immutable pass-1 decision receipt.
    result = _guard(metrics_dir, decisions_dir, 1)

    assert result.decision == "continue"
    receipt = plateau.verify_decision_receipt(result.path)
    assert len(receipt["source_metrics"]) == 4


def test_hard_cap_while_still_improving_is_capped_not_converged(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [4, 4, 4, 4])
    _publish_pass(metrics_dir, 2, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 3, [2, 2, 2, 2])
    _publish_pass(metrics_dir, 4, [1, 1, 1, 1])

    result = _guard(metrics_dir, decisions_dir, 4)

    assert result.decision == "capped"
    assert result.guard_exit_code == 0
    receipt = plateau.verify_decision_receipt(result.path)
    assert receipt["reason"] == "hard_cap_reached_while_improving"
    assert receipt["afterok"] == {"permit_training": False, "successor_action": "no_op"}
    assert receipt["raw_best_pass_endpoint"]["pass_index"] == 4
    assert receipt["plateau_reference_pass_endpoint"]["pass_index"] == 4


def test_subthreshold_reductions_accumulate_against_the_plateau_reference(tmp_path: Path):
    decisions_dir = tmp_path / "decisions"
    config = plateau.PlateauConfig(min_delta=0.5, patience=2, hard_cap_passes=4)
    metrics_dir = tmp_path / "metrics"
    # Pass 2's 0.25 decrease does not qualify; pass 3's 0.50 decrease against
    # the still-retained pass-1 reference does qualify and resets patience.
    _publish_pass(metrics_dir, 1, [4, 4, 4, 4])
    _publish_pass(metrics_dir, 2, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 3, [2, 2, 2, 2])
    result = plateau.guard_from_directory(
        metrics_dir,
        output_directory=decisions_dir,
        target=plateau.SegmentPosition(3, 3),
        expected_selection_sha256=SELECTION_SHA256,
        config=config,
    )
    receipt = plateau.verify_decision_receipt(result.path)
    assert receipt["decision"] == "continue"
    assert receipt["complete_passes"][1]["qualifies_as_improvement"] is False
    assert receipt["complete_passes"][2]["qualifies_as_improvement"] is True
    assert receipt["plateau_reference_pass_endpoint"]["pass_index"] == 3


def test_heldout_diagnostics_are_bound_but_cannot_change_training_plateau_decision(tmp_path: Path):
    plain_metrics = tmp_path / "plain-metrics"
    diagnostic_metrics = tmp_path / "diagnostic-metrics"
    plain_decisions = tmp_path / "plain-decisions"
    diagnostic_decisions = tmp_path / "diagnostic-decisions"
    for pass_index, score in ((1, 4), (2, 3), (3, 3)):
        _publish_pass(plain_metrics, pass_index, [score] * 4)
        _publish_pass(
            diagnostic_metrics,
            pass_index,
            [score] * 4,
            heldout={"heldout_mean_gap": 0.999 if pass_index == 3 else 0.001, "notes": ["diagnostic only"]},
        )

    plain = _guard(plain_metrics, plain_decisions, 3).receipt
    diagnostic = _guard(diagnostic_metrics, diagnostic_decisions, 3).receipt

    assert plain["decision"] == diagnostic["decision"] == "continue"
    assert [item["pooled_training_mean_absolute_gap"] for item in plain["complete_passes"]] == [
        item["pooled_training_mean_absolute_gap"] for item in diagnostic["complete_passes"]
    ]
    assert [item["qualifies_as_improvement"] for item in plain["complete_passes"]] == [
        item["qualifies_as_improvement"] for item in diagnostic["complete_passes"]
    ]
    assert diagnostic["decision_metric"]["heldout_diagnostics_used_for_decision"] is False
    assert len(diagnostic["heldout_diagnostics"]["source_segments_present"]) == 12


def test_guard_fails_closed_for_missing_metrics_and_writes_an_immutable_fail_receipt(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [3, 3, 3, 3])
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    for path in sorted(metrics_dir.glob("segment-metrics-*.json"))[:2]:
        (partial_dir / path.name).write_bytes(path.read_bytes())

    result = _guard(partial_dir, decisions_dir, 1)

    assert result.decision == "fail"
    assert result.guard_exit_code == 1
    receipt = plateau.load_decision_receipt(result.path)
    assert receipt["afterok"] == {"permit_training": False, "successor_action": "block"}
    assert receipt["reason"] == "metric_validation_failed"
    assert receipt["failure"]["error_type"] == "MetricValidationError"


def test_guard_rejects_question_block_drift_between_passes(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 2, [2, 2, 2, 2], changed_block=2)

    result = _guard(metrics_dir, decisions_dir, 2)

    assert result.decision == "fail"
    assert "matched block question IDs differ" in result.receipt["failure"]["message"]


def test_receipt_replay_detects_tampered_authoritative_metrics_source(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [3, 3, 3, 3])
    result = _guard(metrics_dir, decisions_dir, 1)
    assert result.decision == "continue"

    raw = metrics_dir / "_sources" / "pass-1-block-0" / "metrics.jsonl"
    raw.write_bytes(b"tampered\n")

    with pytest.raises(plateau.DecisionReceiptError, match="source metric is no longer valid"):
        plateau.verify_decision_receipt(result.path)
    with pytest.raises(plateau.DecisionReceiptError, match="source metric is no longer valid"):
        plateau.require_continue(result.path)


def test_metric_prefix_cannot_continue_after_an_earlier_terminal_pass(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 2, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 3, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 4, [2, 2, 2, 2])

    result = _guard(metrics_dir, decisions_dir, 4)

    assert result.decision == "fail"
    assert "continues after terminal pass 3" in result.receipt["failure"]["message"]


def test_cli_terminal_guard_exits_successfully_but_require_continue_returns_terminal_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    metrics_dir = tmp_path / "metrics"
    decisions_dir = tmp_path / "decisions"
    _publish_pass(metrics_dir, 1, [4, 4, 4, 4])
    _publish_pass(metrics_dir, 2, [3, 3, 3, 3])
    _publish_pass(metrics_dir, 3, [2, 2, 2, 2])
    _publish_pass(metrics_dir, 4, [1, 1, 1, 1])

    exit_code = plateau.main(
        [
            "guard",
            "--metrics-directory",
            str(metrics_dir),
            "--output-directory",
            str(decisions_dir),
            "--target-pass",
            "4",
            "--target-segment",
            "3",
            "--selection-sha256",
            SELECTION_SHA256,
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["decision"] == "capped"
    assert output["afterok"]["successor_action"] == "no_op"
    assert plateau.main(["require-continue", "--receipt", output["path"]]) == 2
