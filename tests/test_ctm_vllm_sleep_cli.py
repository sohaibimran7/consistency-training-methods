"""Focused CLI contracts for the opt-in vLLM phase-sharing lifecycle."""

from __future__ import annotations

import argparse

import pytest

from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.backends.local.engine import LocalBackend
from ctm.backends.local.rollout_workers import RolloutParallelBackend


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    return parser.parse_args(argv)


def test_sleep_flag_is_explicit_opt_in_and_reaches_the_local_vllm_engine():
    plain = build_backend(_parse(["--backend", "local", "--local-device", "cpu"]))
    enabled_args = _parse(
        ["--backend", "local", "--local-device", "cpu", "--local-vllm-sleep-during-training"]
    )
    enabled = build_backend(enabled_args)

    assert isinstance(plain, LocalBackend)
    assert "enable_sleep_mode" not in plain.vllm_options
    assert isinstance(enabled, LocalBackend)
    assert enabled.vllm_options["enable_sleep_mode"] is True
    assert "vllm_sleep=training_phase" in describe_backend(enabled_args)


@pytest.mark.parametrize(
    "argv",
    [
        ["--local-vllm-sleep-during-training"],
        ["--backend", "local", "--local-sampler", "hf", "--local-vllm-sleep-during-training"],
    ],
    ids=["requires-local", "requires-vllm"],
)
def test_sleep_flag_rejects_unsupported_backend_or_sampler(argv):
    with pytest.raises(ValueError, match="local-vllm-sleep-during-training requires"):
        build_backend(_parse(argv))


def test_sleep_flag_reaches_both_coordinator_and_dedicated_rollout_workers(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    backend = build_backend(
        _parse(
            [
                "--backend",
                "local",
                "--local-device",
                "cuda:0",
                "--local-rollout-gpus",
                "1",
                "--local-rollout-status-dir",
                str(tmp_path / "rollout-workers"),
                "--local-vllm-sleep-during-training",
            ]
        )
    )

    assert isinstance(backend, RolloutParallelBackend)
    assert backend.training_backend.vllm_options["enable_sleep_mode"] is True
    assert backend.worker_vllm_options["enable_sleep_mode"] is True
    assert backend.sampling_training_overlap_supported is False
