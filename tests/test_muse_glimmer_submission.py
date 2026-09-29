"""Static safety checks for the sequential Muse RMCT submitter."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_submitter_is_sequential_idempotent_and_has_no_generation_cap():
    source = (ROOT / "infra" / "isambard" / "submit_muse_glimmer_rmct_next.sh").read_text(
        encoding="utf-8"
    )

    assert "contract.validate_preflight(root)" in source
    assert "contract.validate_receipt(root, index)" in source
    assert "contract.validate_convergence(root, index)" in source
    assert "contract.guard(root, next_index)" in source
    assert "plan.MINIMUM_COMPARISON_OPTIMIZER_STEP" in source
    assert '"final_segment": first_passing' in source
    assert "squeue" in source
    assert "sbatch --test-only" in source
    assert '--reservation="$ctm_muse_reservation"' in source
    assert "--time=08:00:00" in source
    assert "max-new-tokens" not in source
    assert "max-tokens" not in source
