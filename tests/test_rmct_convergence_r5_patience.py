from __future__ import annotations

from decimal import Decimal
from pathlib import Path
import pytest

from experiments.rmct_convergence import controller, patience
from experiments.rmct_convergence_r5_patience import plan
from scripts.run_experiment import _argument_tokens


ROOT = Path(__file__).resolve().parents[1]


def _summary(segment_index: int, gap: str, *, coverage: int = 64) -> controller.SegmentSummary:
    identity = controller.SegmentIdentity(
        segment_index=segment_index,
        optimizer_step_start=segment_index * 16 + 1,
        optimizer_step_end=(segment_index + 1) * 16,
    )
    pooled = controller.Coverage(parse_count=coverage, valid_count=coverage, total_count=64)
    by_bias = {
        "suggested_answer": controller.Coverage(parse_count=coverage // 2, valid_count=coverage // 2, total_count=32),
        "wrong_argument": controller.Coverage(parse_count=coverage // 2, valid_count=coverage // 2, total_count=32),
    }
    return controller.SegmentSummary(
        identity=identity,
        absolute_sum=Decimal(gap) * Decimal(64),
        absolute_count=64,
        pooled_coverage=pooled,
        coverage_by_bias=by_bias,
    )


def test_r5_plan_resumes_step176_with_approved_legacy_cap(tmp_path: Path) -> None:
    snapshot = tmp_path / plan.BASE_SNAPSHOT
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")

    record = plan.segment_record(ROOT, plan.START_SEGMENT_INDEX)
    args = plan.segment_args(ROOT, plan.START_SEGMENT_INDEX, model_snapshot=snapshot)
    tokens = _argument_tokens(args)

    assert record["optimizer_step_start"] == 177
    assert record["optimizer_step_end"] == 192
    assert record["parent"]["optimizer_step"] == 176
    assert record["parent"]["run_name"].endswith("r4-s011")
    assert record["generation"] == {"output_token_cap": 20480, "termination": "legacy_stop_or_length"}
    assert args["max_new_tokens"] == 20480
    assert "no_max_new_tokens" not in args
    assert tokens[tokens.index("--max-new-tokens") + 1] == "20480"
    assert "--no-max-new-tokens" not in tokens


def test_patience_requires_eight_consecutive_eligible_windows_and_new_best_resets() -> None:
    state = patience.PatienceState(Decimal("0.09895833333333333"), 10, 176, 0)
    for index in range(11, 18):
        state, new_best, eligible = patience._next_state(state, _summary(index, "0.11"))
        assert eligible is True
        assert new_best is False
        assert state.consecutive_nonimproving_windows == index - 10
    assert state.consecutive_nonimproving_windows == 7

    state, new_best, eligible = patience._next_state(state, _summary(18, "0.11"))
    assert eligible is True
    assert new_best is False
    assert state.consecutive_nonimproving_windows == 8
    assert (18 + 1) * 16 == 304

    state, new_best, eligible = patience._next_state(state, _summary(19, "0.08"))
    assert eligible is True
    assert new_best is True
    assert state.best_gap == Decimal("0.08")
    assert state.best_segment_index == 19
    assert state.consecutive_nonimproving_windows == 0


def test_ineligible_coverage_cannot_consume_patience() -> None:
    before = patience.PatienceState(Decimal("0.09"), 10, 176, 3)
    after, new_best, eligible = patience._next_state(before, _summary(11, "0.2", coverage=48))
    assert after.best_gap == before.best_gap
    assert after.consecutive_nonimproving_windows == 0
    assert new_best is False
    assert eligible is False


def test_patience_policy_and_slurm_sources_restore_cap_without_step_limit() -> None:
    assert patience._policy()["patience_windows"] == 8
    assert patience._policy()["patience_optimizer_steps"] == 128
    assert patience._policy()["output_token_cap"] == 20480
    assert patience._policy()["max_optimizer_steps"] is None
    assert patience._policy()["minimum_improvement"] == 0.01

    runner = (ROOT / "infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.py").read_text(encoding="utf-8")
    sbatch = (ROOT / "infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.sbatch").read_text(encoding="utf-8")
    submitter = (ROOT / "infra/isambard/submit_qwen35_rmct_convergence_r5_patience_chain.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=4" in sbatch
    assert "--dependency=afterok:" in submitter
    assert "plan.execute" in runner
    assert "--max-new-tokens" not in sbatch
    assert "_prior_terminal" in runner
    assert "rmct-capped-patience-20260910/amendment.json" in sbatch
    assert 'submit_qwen35_rmct_convergence_r5_patience_chain.sh" --yes' in sbatch


def test_small_cumulative_improvements_only_reset_at_meaningful_threshold() -> None:
    state = patience.PatienceState(Decimal("0.10"), 10, 176, 0)
    state, improved, _ = patience._next_state(state, _summary(11, "0.095"))
    assert not improved and state.consecutive_nonimproving_windows == 1
    state, improved, _ = patience._next_state(state, _summary(12, "0.09"))
    assert improved and state.consecutive_nonimproving_windows == 0


def test_no_convergence_before_eight_windows_even_beyond_step512() -> None:
    for segment_index in (18, 31, 32, 80):
        for streak in range(8):
            source = controller.ParsedSource(
                identity=_summary(segment_index, "0.11").identity,
                summary=_summary(segment_index, "0.11"),
            )
            decision = patience._build_decision(
                amendment_identity={}, source=source, source_identity={},
                checkpoint_identity={}, completion_identity={}, predecessor_identity=None,
                state_before=patience.PatienceState(Decimal("0.09"), 10, 176, streak),
            )
            assert decision["decision"] == ("converged" if streak == 7 else "continue")


def test_plan_continues_data_order_beyond_original_horizon(tmp_path: Path) -> None:
    snapshot = tmp_path / plan.BASE_SNAPSHOT
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n")
    args = plan.segment_args(ROOT, 32, model_snapshot=snapshot)
    assert args["load_config"]["segment_index"] == 32
    assert args["load_config"]["cycle_segments"] is True
    assert args["max_new_tokens"] == 20480
    assert args["resume_from"].endswith("capped-patience-r5-s032")


def test_data_window_extension_requires_explicit_opt_in() -> None:
    from ctm_data.adapters.mcq_bias.shared_qid_two_bias import SharedQidTwoBiasSetting

    shape = {"segment_datapoints": 32, "segment_count": 32}
    validate = SharedQidTwoBiasSetting._validate_segment_request
    with pytest.raises(ValueError):
        validate(32, 32, shape=shape)
    assert validate(32, 32, shape=shape, cycle_segments=True) == (32, 32)
    for invalid in (-1, True, 3.5):
        with pytest.raises(ValueError):
            validate(32, invalid, shape=shape, cycle_segments=True)
