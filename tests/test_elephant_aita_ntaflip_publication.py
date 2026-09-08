from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.elephant_aita_ntaflip.publication import (
    PARSER_SCHEMA,
    PREFLIGHT_SCHEMA,
    PublicationError,
    _holm_adjust,
    load_condition,
    publish,
)


def _preflight(path: Path, outcomes: list[int], *, pair_hash: str = "pair-hash") -> Path:
    records = [
        {"pair_id": f"p{index:03d}", "final_indicators": {"nta_nta": outcome}}
        for index, outcome in enumerate(outcomes)
    ]
    value = {
        "schema": PREFLIGHT_SCHEMA,
        "benchmark": "elephant-aita-nta-flip",
        "manifest": {
            "schema": "elephant-aita-nta-flip-manifest-v2-r005",
            "pair_count": len(outcomes),
            "pair_ids_sha256": "ids-hash",
            "pair_artifact_sha256": pair_hash,
            "sha256": "manifest-hash",
        },
        "metrics": {
            "final_answer_only": {
                "parser_schema": PARSER_SCHEMA,
                "parsed_pair_coverage": {"count": len(outcomes)},
                "parsed_response_coverage": {"count": 2 * len(outcomes)},
                "outcomes": {"nta_nta": {"count": sum(outcomes)}},
            }
        },
        "pair_records": records,
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    assert _holm_adjust([0.01, 0.03, 0.02]) == pytest.approx([0.03, 0.04, 0.04])


def test_publish_uses_pair_level_outcomes_and_exact_mcnemar(tmp_path: Path) -> None:
    base = _preflight(tmp_path / "base.json", [1, 1, 1, 1, 0, 0, 0, 0])
    better = _preflight(tmp_path / "better.json", [0, 0, 0, 0, 0, 0, 0, 0])
    same = _preflight(tmp_path / "same.json", [1, 1, 1, 1, 0, 0, 0, 0])
    output = tmp_path / "output"
    report = publish(
        [("Base", base), ("Better", better), ("Same", same)],
        output_dir=output,
        expected_pairs=8,
        resamples=200,
        seed=7,
    )
    assert [row["nta_nta_count"] for row in report["conditions"]] == [4, 0, 4]
    assert report["significance"]["family_size"] == 2
    comparison = report["comparisons"][0]
    assert comparison["discordant"] == {
        "baseline_nta_nta_treatment_not": 4,
        "baseline_not_treatment_nta_nta": 0,
        "total": 4,
    }
    assert comparison["p_value_raw"] == pytest.approx(0.125)
    for name in (
        "aita-nta-flip-paired-report.json",
        "aita-nta-flip-paired-results.csv",
        "aita-nta-flip-both-nta-rate.png",
        "aita-nta-flip-both-nta-rate.pdf",
    ):
        assert (output / name).is_file()


def test_rejects_mismatched_pair_custody(tmp_path: Path) -> None:
    first = _preflight(tmp_path / "first.json", [1, 0])
    second = _preflight(tmp_path / "second.json", [1, 0], pair_hash="different")
    base = load_condition("Base", first, expected_pairs=2)
    treatment = load_condition("Treatment", second, expected_pairs=2)
    from experiments.elephant_aita_ntaflip.publication import build_report

    with pytest.raises(PublicationError, match="pair_artifact_sha256"):
        build_report([base, treatment], resamples=100, seed=1)
