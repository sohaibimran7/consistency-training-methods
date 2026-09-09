"""Safety checks for recoverable Muse startup-failure archival."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_startup_archive_is_pretraining_only_recoverable_and_no_cap() -> None:
    source = (ROOT / "infra/isambard/archive_muse_glimmer_failed_startup.py").read_text(
        encoding="utf-8"
    )

    assert 'run / "metrics.jsonl"' in source
    assert 'run / "rollouts"' in source
    assert 'run / "checkpoints"' in source
    assert '"optimizer_updates": 0' in source
    assert '"generation_requests": 0' in source
    assert "control.rename(archived_control)" in source
    assert "run.rename(archived_run)" in source
    assert '"output_token_cap": None' in source
    assert "unlink(" not in source
    assert "rmtree(" not in source
    assert "rm -" not in source
    assert "max_tokens" not in source
    assert "max_new_tokens" not in source

