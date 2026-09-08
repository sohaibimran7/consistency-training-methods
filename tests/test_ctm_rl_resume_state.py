"""Fail-closed checkpoint boundary tests for local RL data scaling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctm.training.resume_state import (
    RL_LOOP_STATE_SCHEMA,
    capture_runtime_rng_state,
    load_strict_local_rl_resume_state,
)


def _write_final_checkpoint(path: Path, *, accumulated_grads: int = 0) -> None:
    """Write the smallest valid local-RL final checkpoint boundary."""

    path.mkdir()
    # The strict loader validates presence and provenance here; local backend
    # loading separately deserializes this file before a real continuation.
    (path / "optimizer.pt").write_bytes(b"test-optimizer-state")
    runtime_rng = capture_runtime_rng_state()
    # CI commonly has no CUDA device.  The loader deliberately requires a
    # coordinator CUDA state for a protected local continuation, so use the
    # valid CPU RNG bytes as a syntactically valid stand-in for this boundary
    # validation test.  No CUDA state is restored in this test.
    runtime_rng.setdefault("torch_cuda_coordinator_device", 0)
    runtime_rng.setdefault(
        "torch_cuda_rng_state_base64",
        runtime_rng["torch_cpu_rng_state_base64"],
    )
    manifest = {
        "backend": "local",
        "kind": "both",
        "loop_state": {
            "schema": RL_LOOP_STATE_SCHEMA,
            "step": 64,
            "global_step": 64,
            "optimizer_step": 64,
            "completed_epochs": 1,
            "accumulated_grads": accumulated_grads,
            "final": True,
            "runtime_rng": runtime_rng,
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_strict_local_rl_resume_loads_only_a_completed_final_boundary(tmp_path: Path):
    checkpoint = tmp_path / "rmct256-final"
    _write_final_checkpoint(checkpoint)

    state = load_strict_local_rl_resume_state(f"file://{checkpoint}")

    assert state.checkpoint_dir == checkpoint.resolve()
    assert (state.global_step, state.optimizer_step, state.completed_epochs) == (64, 64, 1)


def test_strict_local_rl_resume_rejects_mid_accumulation_boundary(tmp_path: Path):
    checkpoint = tmp_path / "rmct256-not-resumable"
    _write_final_checkpoint(checkpoint, accumulated_grads=1)

    with pytest.raises(ValueError, match="accumulated gradients"):
        load_strict_local_rl_resume_state(checkpoint)
