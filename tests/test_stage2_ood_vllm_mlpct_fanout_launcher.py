from __future__ import annotations

import subprocess
from pathlib import Path


def test_mlpct_fanout_launcher_is_syntactically_valid_and_preserves_the_contract():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "ctm_stage2_ood_vllm_mlpct_biased_fanout_20260802.sh"

    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")

    assert "mlpct-vllm-compat" in text
    assert "CTM_OOD_ORIGINAL_PARENT_PID" in text
    assert "CTM_OOD_ORIGINAL_PGID" in text
    assert "verified MLPCT clean barrier" in text
    assert "--isolate-tasks" in text
    assert "--persistent-vllm-server" in text
    assert '"max_tokens":20480' in text
    assert '"prompt_style": "none"' in text
    assert "GROUP_3=(6 17 4 7 8 9)" in text
    assert "GROUP_4=(18 19 5 10 11 12)" in text
    assert "GROUP_6=(20 21 13 14 15 16)" in text
    assert "raw_preflight" in text
    # A stopped parent defers SIGTERM until SIGCONT, which would permit the
    # original serial launcher to produce task 4.  The finalizer must instead
    # re-check the stopped owner and use SIGKILL after raw preflight.
    kill_index = text.index('kill -KILL -- "-$ORIGINAL_PGID"')
    preflight_index = text.index("verified complete MLPCT fanout raw-preflight")
    marker_index = text.index("Stage 2 OOD vLLM raw condition complete")
    assert preflight_index < kill_index < marker_index
    assert "original MLPCT parent is no longer stopped before SIGKILL" in text
    assert 'kill -TERM -- "-$ORIGINAL_PGID"' not in text
    assert 'kill -CONT -- "-$ORIGINAL_PGID"' not in text
    assert "Stage 2 OOD vLLM raw condition complete" in text
    assert "grade_luna" not in text
