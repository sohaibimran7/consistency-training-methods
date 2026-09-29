import pytest

from ctm.core.types import BatchItem
from ctm.training.rl import RLConfig, RLTrainer


class _UnusedBackend:
    """Backend placeholder for logging-only RL trainer tests."""


class _MetricLogger:
    def __init__(self):
        self.metrics: dict[str, float | int] | None = None
        self.step: int | None = None

    def log_metrics(self, metrics, *, step):
        self.metrics = dict(metrics)
        self.step = step


class _Progress:
    def __init__(self):
        self.postfix = None

    def set_postfix(self, postfix):
        self.postfix = postfix


def _item(*, idx: int, p_ref: float | None, p_hat: float | None) -> BatchItem:
    return BatchItem(
        datapoint_idx=idx,
        datapoint={"question": f"question-{idx}"},
        train_rollouts=[],
        anchor_rollouts=[],
        sampled_rollouts=[],
        initial_rollouts=[],
        p_hat={} if p_hat is None else {1: p_hat},
        p_hat_counts={} if p_hat is None else {1: 8},
        p_ref=p_ref,
        p_ref_init=None,
        reference_rates={} if p_ref is None else {0: p_ref},
        reference_rate_counts={} if p_ref is None else {0: 8},
        initial_reference_rates={},
        initial_reference_rate_counts={},
        n_total=0,
        n_parsed=0,
        n_ref_parsed=0,
        n_training_parsed=0,
    )


def _logged_metrics(items: list[BatchItem]) -> dict[str, float | int]:
    trainer = RLTrainer(config=RLConfig(), backend=_UnusedBackend())
    logger = _MetricLogger()
    trainer._log_step_metrics(
        logger,
        global_step=1,
        epoch=0,
        batch_items=items,
        grad_datums=[],
        consistency_rewards=[],
        anchor_rewards=[],
        advantages=[],
        policy_grad_data=[],
        all_rewards=[],
        training_logprobs=[],
        kl_penalty_metrics={},
        parse_rate=1.0,
        total_grader_failures=0,
        grader_failure_rate=0.0,
        grader_sample_count=0,
        total_n_ref_parsed=0,
        total_n_training_parsed=0,
        training_idx=[1],
        need_p_ref_init=False,
        pbar=_Progress(),
        optimizer_step=1,
    )
    assert logger.step == 1
    assert logger.metrics is not None
    return logger.metrics


def test_absolute_per_question_gap_survives_signed_cross_question_cancellation():
    metrics = _logged_metrics(
        [
            _item(idx=0, p_ref=0.25, p_hat=0.75),  # +0.50
            _item(idx=1, p_ref=0.75, p_hat=0.25),  # -0.50
        ]
    )

    # Preserve the historical signed aggregate: the two question-level gaps
    # cancel at the batch level.
    assert metrics["train/consistency_gap_1"] == pytest.approx(0.0)

    # The new monitor metric is computed before cancellation, and the sum/count
    # fields let a 16-update controller calculate a sample-weighted block mean.
    assert metrics["train/consistency_gap_item_signed_sum_1"] == pytest.approx(0.0)
    assert metrics["train/consistency_gap_abs_sum_1"] == pytest.approx(1.0)
    assert metrics["train/consistency_gap_abs_count_1"] == 2
    assert metrics["train/consistency_gap_abs_mean_1"] == pytest.approx(0.5)


def test_unavailable_per_question_rates_have_zero_count_not_a_zero_gap():
    metrics = _logged_metrics(
        [
            _item(idx=0, p_ref=0.5, p_hat=None),  # Training rate unparsed.
            _item(idx=1, p_ref=None, p_hat=0.5),  # Reference rate unparsed.
        ]
    )

    assert metrics["train/consistency_gap_item_signed_sum_1"] == pytest.approx(0.0)
    assert metrics["train/consistency_gap_abs_sum_1"] == pytest.approx(0.0)
    assert metrics["train/consistency_gap_abs_count_1"] == 0
    assert "train/consistency_gap_abs_mean_1" not in metrics
    assert "train/consistency_gap_1" not in metrics


def test_empty_batch_emits_zero_aggregate_count_without_a_perfect_gap_mean():
    metrics = _logged_metrics([])

    assert metrics["train/consistency_gap_item_signed_sum_1"] == pytest.approx(0.0)
    assert metrics["train/consistency_gap_abs_sum_1"] == pytest.approx(0.0)
    assert metrics["train/consistency_gap_abs_count_1"] == 0
    assert "train/consistency_gap_abs_mean_1" not in metrics
