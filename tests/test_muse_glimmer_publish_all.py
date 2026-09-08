from __future__ import annotations

from pathlib import Path

from experiments.muse_glimmer_rmct_replication import publish_all


def test_publish_all_uses_expected_condition_order_and_no_token_controls() -> None:
    assert tuple(publish_all.AITA_LABELS) == ("base", "step016", "step064", "final")
    source = Path(publish_all.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "--max-tokens",
        "--max-new-tokens",
        "--max-output-tokens",
        "GenerateConfig(",
        ".generate(",
    ):
        assert forbidden not in source
