"""No-GPU contract for the Qwen3.5 RMCT worker-preflight launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "infra" / "vastai" / "preflight_qwen35_rmct_rollout_workers.sh"
EXPERIMENT = "rmct_paper_vast_dense_qwen3_5_9b_batching_repair_20260803"


@pytest.mark.parametrize(
    ("target", "run_name"),
    [
        ("rmct-main", "rate-matching-lr-1e-4"),
        ("rmct-control", "rate-matching-control-lr-1e-4"),
    ],
)
def test_rmct_worker_preflight_dry_run_uses_the_normal_training_status_path(target: str, run_name: str):
    result = subprocess.run(
        ["bash", str(SCRIPT), target, "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    status_dir = ROOT / "logs" / EXPERIMENT / run_name / "rollout_workers"
    assert result.returncode == 0, result.stderr
    assert "QWEN35_RMCT_WORKER_PREFLIGHT_DRY_RUN=1" in result.stdout
    assert f"status_dir={status_dir}" in result.stdout
    assert f"attestation={status_dir}/qwen35-rollout-worker-parity-attestation.json" in result.stdout
    assert "coordinator_logical_gpu=0" in result.stdout
    assert "rollout_worker_logical_gpus=1,2,3,4,5,6,7" in result.stdout
    assert "worker_gpu_memory_utilization=0.75" in result.stdout
    assert "worker_max_model_len=32768" in result.stdout
    assert "worker_max_num_seqs=256" in result.stdout
    assert "worker_max_num_batched_tokens=8192" in result.stdout
    assert "target_logprob_chunk_size=2048" in result.stdout
