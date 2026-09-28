"""Exercise scheduling decisions without contacting Slurm."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "infra/isambard/submit_qwen35_rmct_convergence_r5_patience_chain.sh"


@pytest.mark.parametrize("action,queued,expected", [
    ("submit", "123", 0),
    ("complete", "123", 0),
    ("submit", "123\n456", 2),
])
def test_successor_submission_is_conditional_and_preserves_existing_jobs(tmp_path, action, queued, expected):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    artifact = tmp_path / "artifacts/rmct-capped-patience-20260910"
    artifact.mkdir(parents=True)
    fake_python = bin_dir / "python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-m': sys.exit(0)\n"
        "if sys.argv[1] == '-':\n"
        " print(json.dumps({'action': os.environ['TEST_ACTION'], 'segment_index': 32})); sys.exit(0)\n"
        f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n"
    )
    (bin_dir / "squeue").write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_QUEUE"\n')
    (bin_dir / "sbatch").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_CALLS"\n'
        'case "$*" in *--test-only*) exit 0;; esac\nprintf "999\\n"\n'
    )
    for executable in bin_dir.iterdir():
        executable.chmod(0o755)
    calls = tmp_path / "calls.txt"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "REPO_DIR": str(tmp_path), "SCRATCHDIR": str(tmp_path),
           "CTM_RMCT_PYTHON": str(fake_python), "SLURM_JOB_ID": "123",
           "TEST_ACTION": action, "TEST_QUEUE": queued, "TEST_CALLS": str(calls)}
    result = subprocess.run(["bash", str(SCRIPT), "--yes"], env=env, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    if action == "submit" and expected == 0:
        submitted = calls.read_text().splitlines()
        assert len(submitted) == 2  # precheck plus the single real submission
        assert all("--dependency=afterok:123" in line for line in submitted)
        assert all("--kill-on-invalid-dep=no" in line for line in submitted)
        assert all("CTM_RMCT_SEGMENT_INDEX=32" in line for line in submitted)
        assert (artifact / "submissions.log").read_text() == "32 999\n"
    else:
        assert not calls.exists()
