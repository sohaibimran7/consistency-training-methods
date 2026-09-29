from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BROKER = ROOT / "artifacts/rmct-r5-relaunch-20260908/run-workq-interactive-broker.sh"


def test_workq_relaunch_broker_is_syntax_valid_and_uses_one_ordinary_allocation_per_window() -> None:
    result = subprocess.run(["bash", "-n", str(BROKER)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    text = BROKER.read_text(encoding="utf-8")

    assert "for ctm_workq_index in {11..31}" in text
    assert "srun --partition=workq --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 --mem=200G" in text
    assert "--time=12:00:00" in text
    assert "--reservation=" not in text
    assert "--qos=" not in text
    assert "sbatch" not in text
    assert "scancel" not in text
    assert "squeue" not in text
    assert "SLURM_JOB_END_TIME is required" in text
    assert "exactly four distinct Slurm-visible GPUs" in text
    assert "workq r5 broker forbids a Slurm reservation" in text
    assert "never submits a successor" not in text
    assert "never cancels, holds, or otherwise alters the scheduled r5" in text

    # The GPU body is intentionally passed as one bash -c argument. Validate
    # that body separately too; outer `bash -n` only sees a quoted string.
    marker = "        bash -c '"
    start = text.index(marker) + len(marker)
    end = text.index("\n        ' bash \"$ctm_workq_python\"", start)
    worker_body = text[start:end]
    worker_result = subprocess.run(["bash", "-n"], input=worker_body, capture_output=True, text=True, check=False)
    assert worker_result.returncode == 0, worker_result.stderr
    assert "'" not in worker_body  # no accidental break-out from outer quote


def test_workq_relaunch_broker_restart_skips_only_helper_validated_sealed_prefix() -> None:
    text = BROKER.read_text(encoding="utf-8")

    # `status` replays the content-addressed patience chain and rejects an
    # incomplete marker before this branch can run. A restart can therefore
    # pass sealed `continue` windows but cannot bypass a partial predecessor.
    assert "ctm_workq_status" in text
    assert "CTM_WORKQ_R5_SKIP_SEALED_SEGMENT" in text
    assert "10#$ctm_workq_next > 10#$ctm_workq_index" in text
    assert "missing/partial predecessor still fails closed" in text
    assert "clone status expects segment" in text


def test_workq_relaunch_broker_replays_custody_smoke_and_uncapped_canonical_command_before_runner() -> None:
    text = BROKER.read_text(encoding="utf-8")

    assert '"$ctm_workq_python" "$ctm_workq_helper" validate' in text
    assert "_require_smoke_success" in text
    assert '"$ctm_job_python" "$ctm_job_parity_helper" --output-dir "$ctm_job_parity"' in text
    assert "--model-snapshot \"$ctm_job_snapshot\" --resume" in text
    assert 'exec "$ctm_job_python" "$ctm_job_runner"' in text
    assert "--repository \"$ctm_job_repo\" --segment-index \"$ctm_job_index\"" in text
    assert "--amendment \"$ctm_job_clone_amendment\" --yes" in text
    assert "first r5 window does not resume canonical r4 step-176 parent" in text
    assert "parent[\"optimizer_step\"] != 176" in text
    assert "--no-max-new-tokens" in text
    for forbidden in ("--max-new-tokens", "--max-tokens", "--max-output-tokens", "--max-completion-tokens"):
        assert forbidden in text
    assert "interactive-smoke-s012" in text
    assert "production r5 command improperly references smoke output" in text
    assert "hard step-512 ceiling" in text
    assert ' prepare_qwen35_rmct_r5_interactive_duplicate.py" run ' not in text
