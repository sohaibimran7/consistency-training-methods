"""Pure contracts for the post-update Qwen3.5 worker transport gate."""

from __future__ import annotations

import pytest

from infra.vastai.benchmark_qwen35_opct_group import (
    _effect_parity_summary,
    _flat_score_differences,
    _post_update_probe_source_indices,
    _post_update_worker_probe_alignment,
    _require_rollout_rng_diversity,
    _rollout_rng_diversity_summary,
)
from ctm.backends.base import SampledSequence


def test_effect_parity_is_computed_from_policy_minus_base_not_raw_scores():
    result = _effect_parity_summary(
        worker_policy=[[-1.0, -2.0], [-3.0]],
        worker_base=[[-1.2, -2.4], [-3.3]],
        coordinator_policy=[[-1.01, -2.02], [-3.03]],
        coordinator_base=[[-1.21, -2.42], [-3.33]],
    )

    assert result["worker_v2_minus_base"]["max_abs_difference"] == pytest.approx(0.4)
    assert result["coordinator_updated_minus_base"]["max_abs_difference"] == pytest.approx(0.4)
    assert result["worker_minus_coordinator_effect"]["max_abs_difference"] == pytest.approx(0.0)
    assert result["cosine_similarity"] == pytest.approx(1.0)


def test_effect_audit_rejects_misaligned_worker_or_coordinator_scores():
    with pytest.raises(RuntimeError, match="token count differs"):
        _flat_score_differences([[-1.0]], [[-1.0, -2.0]], label="unit")


def test_post_update_probe_coverage_assigns_the_same_anchor_to_every_worker():
    sources = _post_update_probe_source_indices(
        [5, 6, 7, 8],
        worker_count=3,
        anchor_index=7,
    )

    assert sources == [7, 7, 7, 5, 6, 8]
    assert sources[:3] == [7, 7, 7]
    assert {probe_index % 3 for probe_index in range(len(sources))} == {0, 1, 2}


def test_post_update_alignment_reports_and_validates_each_worker_probe():
    worker_policy = [[-0.8, -1.9], [-1.8, -2.9], [-2.8, -3.9]]
    worker_base = [[-1.0, -2.0], [-2.0, -3.0], [-3.0, -4.0]]
    coordinator_policy = [[-0.79, -1.89], [-1.79, -2.89], [-2.79, -3.89]]
    coordinator_base = [[-0.99, -1.99], [-1.99, -2.99], [-2.99, -3.99]]
    worker_gpus = [
        {"logical_index": 1, "device_token": "1"},
        {"logical_index": 2, "device_token": "2"},
        {"logical_index": 3, "device_token": "3"},
    ]
    sources = [
        {"source_audit_index": 4, "source_prompt_index": 1, "source_rollout_index": 0},
        {"source_audit_index": 4, "source_prompt_index": 1, "source_rollout_index": 0},
        {"source_audit_index": 4, "source_prompt_index": 1, "source_rollout_index": 0},
    ]

    report = _post_update_worker_probe_alignment(
        worker_policy=worker_policy,
        worker_base=worker_base,
        coordinator_policy=coordinator_policy,
        coordinator_base=coordinator_base,
        worker_gpus=worker_gpus,
        sources=sources,
        min_effect=1e-5,
    )

    assert [row["worker_index"] for row in report["by_worker_probe"]] == [0, 1, 2]
    assert [row["source_audit_index"] for row in report["by_worker_probe"]] == [4, 4, 4]
    assert [row["worker_index"] for row in report["by_worker"]] == [0, 1, 2]
    assert all(row["effect_parity"]["cosine_similarity"] == pytest.approx(1.0) for row in report["by_worker"])


def test_post_update_alignment_fails_if_any_covered_worker_has_no_effect():
    worker_policy = [[-0.8], [-2.0], [-2.8]]
    worker_base = [[-1.0], [-2.0], [-3.0]]
    coordinator_policy = [[-0.79], [-1.79], [-2.79]]
    coordinator_base = [[-0.99], [-1.99], [-2.99]]
    worker_gpus = [
        {"logical_index": 1, "device_token": "1"},
        {"logical_index": 2, "device_token": "2"},
        {"logical_index": 3, "device_token": "3"},
    ]
    sources = [{"source_audit_index": 0, "source_prompt_index": 0, "source_rollout_index": 0} for _ in worker_gpus]

    with pytest.raises(RuntimeError, match=r"post-update worker 1 probe 1: worker v2 has no measurable"):
        _post_update_worker_probe_alignment(
            worker_policy=worker_policy,
            worker_base=worker_base,
            coordinator_policy=coordinator_policy,
            coordinator_base=coordinator_base,
            worker_gpus=worker_gpus,
            sources=sources,
            min_effect=1e-5,
        )


def test_rng_diversity_probe_detects_cloned_round_robin_worker_lanes():
    cloned = [
        SampledSequence(tokens=[10, 11], logprobs=None),
        SampledSequence(tokens=[10, 11], logprobs=None),
        SampledSequence(tokens=[10, 11], logprobs=None),
        SampledSequence(tokens=[20, 21], logprobs=None),
        SampledSequence(tokens=[20, 21], logprobs=None),
        SampledSequence(tokens=[20, 21], logprobs=None),
    ]
    summary = _rollout_rng_diversity_summary([cloned], worker_count=3)

    assert summary["matched_lane_groups"] == 2
    assert summary["all_worker_identical_lane_groups"] == 2
    assert summary["passed"] is False
    with pytest.raises(RuntimeError, match="share one RNG seed"):
        _require_rollout_rng_diversity(summary)


def test_rng_diversity_probe_allows_natural_collisions_when_any_lane_differs():
    samples = [
        SampledSequence(tokens=[10], logprobs=None),
        SampledSequence(tokens=[10], logprobs=None),
        SampledSequence(tokens=[10], logprobs=None),
        SampledSequence(tokens=[20], logprobs=None),
        SampledSequence(tokens=[21], logprobs=None),
        SampledSequence(tokens=[22], logprobs=None),
    ]
    summary = _rollout_rng_diversity_summary([samples], worker_count=3)

    assert summary["all_worker_identical_lane_groups"] == 1
    assert summary["passed"] is True
    _require_rollout_rng_diversity(summary)
