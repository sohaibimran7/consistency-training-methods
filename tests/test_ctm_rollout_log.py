"""Round-trip tests for RL rollout persistence (ctm.training.rollout_log + reader)."""

import json

import pytest

from ctm.core.types import RolloutRecord
from ctm.evals.analysis.rollouts import iter_rollouts, load_index
from ctm.training.rollout_log import INDEX_NAME, RolloutLogger


def make_record(
    step=1, epoch=0, dp=0, pert=1, role="train", trait=1.0, reward=-0.12, advantage=-0.9, p_hat=0.7, p_ref=0.3
) -> RolloutRecord:
    return RolloutRecord(
        step=step,
        epoch=epoch,
        datapoint_idx=dp,
        perturbation_idx=pert,
        role=role,
        sample_source="policy",
        prompt_text="Q: pick one\n(A) x (B) y",
        completion_text="The best answer is: (A)",
        trait_value=trait,
        parsed_successfully=True,
        grader_failed=False,
        reward=reward,
        advantage=advantage,
        skipped_from_training=False,
        p_hat=p_hat if role == "train" else None,
        p_ref=p_ref,
        p_ref_init=0.25,
    )


class TestRolloutLogRoundTrip:
    def test_write_then_read_all_fields(self, tmp_path):
        logger = RolloutLogger(tmp_path)
        original = [
            make_record(step=1, pert=1, trait=1.0, advantage=-0.9),
            make_record(step=1, pert=2, trait=0.0, advantage=0.4),
            make_record(step=1, pert=0, role="anchor", p_hat=None),
        ]
        logger.log_step(original)

        back = list(iter_rollouts(tmp_path))
        assert back == original  # pydantic equality: every field round-trips

    def test_multi_step_index_and_filters(self, tmp_path):
        logger = RolloutLogger(tmp_path)
        logger.log_step([make_record(step=1, pert=1), make_record(step=1, pert=2)])
        logger.log_step([make_record(step=2, pert=1, trait=0.0)])

        index = load_index(tmp_path)
        assert [e["step"] for e in index] == [1, 2]
        assert index[0]["n_records"] == 2 and index[1]["n_records"] == 1
        assert index[0]["n_train"] == 2 and index[0]["n_anchor"] == 0

        # step filter only decompresses matching files; field filters compose
        assert len(list(iter_rollouts(tmp_path, steps=[2]))) == 1
        assert len(list(iter_rollouts(tmp_path, perturbation_idx=1))) == 2
        assert len(list(iter_rollouts(tmp_path, trait=0.0))) == 1
        assert len(list(iter_rollouts(tmp_path, role="anchor"))) == 0
        assert len(list(iter_rollouts(tmp_path, skipped_from_training=False))) == 3

    def test_empty_step_writes_nothing(self, tmp_path):
        logger = RolloutLogger(tmp_path)
        assert logger.log_step([]) is None
        assert load_index(tmp_path) == []

    def test_skipped_record_requires_explicit_reason(self, tmp_path):
        logger = RolloutLogger(tmp_path)
        skipped = make_record().model_copy(update={"skipped_from_training": True})

        with pytest.raises(ValueError, match="non-empty skip_reason"):
            logger.log_step([skipped])

        assert load_index(tmp_path) == []

    def test_resume_overwrites_step_entry(self, tmp_path):
        RolloutLogger(tmp_path).log_step([make_record(step=3)])
        # a resumed run re-logs step 3: entry replaced, not duplicated
        RolloutLogger(tmp_path).log_step([make_record(step=3), make_record(step=3, pert=2)])
        index = load_index(tmp_path)
        assert len(index) == 1 and index[0]["n_records"] == 2

    def test_index_is_valid_json(self, tmp_path):
        RolloutLogger(tmp_path).log_step([make_record()])
        data = json.loads((tmp_path / INDEX_NAME).read_text())
        assert set(data["steps"][0]) >= {"step", "file", "n_records", "p_ref_mean", "trait_mean"}
