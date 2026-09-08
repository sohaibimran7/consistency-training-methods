from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def _write_fake_python(path: Path) -> None:
    """Emulate only the launcher boundaries; never import a model or Inspect."""

    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            capture=${CTM_FAKE_CAPTURE:?}
            first=${1:-}
            if [ "$first" = "-m" ]; then
              output=""
              while [ "$#" -gt 0 ]; do
                if [ "$1" = "--output" ]; then output=$2; break; fi
                shift
              done
              mkdir -p "$(dirname "$output")"
              : > "$output"
              exit 0
            fi
            if [ "$first" = "-c" ]; then
              printf '%s\\n' '{"include_bias_acknowledged": false, "prompt_style": "none"}'
              exit 0
            fi
            if [ "$first" = "-" ]; then
              cat >/dev/null
              clean=0
              biased=0
              for record in "$capture"/run-*; do
                [ -f "$record" ] || continue
                tasks=$(cut -d= -f2 "$record")
                for task in $tasks; do
                  if [ "$task" -le 3 ]; then clean=$((clean + 1)); else biased=$((biased + 1)); fi
                done
              done
              [ "$clean" -eq 3 ] && [ "$biased" -eq 0 ] || exit 97
              : > "$capture/clean-verified"
              printf '%s\\n' 'verified Stage 2 clean barrier: 3 exact clean cells'
              exit 0
            fi
            [ "$first" = "scripts/run_evals.py" ] || exit 98
            tasks=""
            shift
            while [ "$#" -gt 0 ]; do
              if [ "$1" = "--task-index" ]; then tasks="$tasks $2"; shift 2; else shift; fi
            done
            for task in $tasks; do
              if [ "$task" -ge 4 ] && [ ! -f "$capture/clean-verified" ]; then exit 99; fi
            done
            printf 'tasks=%s' "$tasks" > "$capture/run-$$"
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    ("condition", "gpu_list", "expected_clean", "expected_biased"),
    [
        ("bct-hf-peft", "0", {"0": (1, 2, 3)}, {"0": tuple(range(4, 22))}),
        (
            "bct-hf-peft",
            "2,7",
            {"2": (1, 3), "7": (2,)},
            {"2": tuple(range(4, 13)), "7": tuple(range(13, 22))},
        ),
        (
            "bct-hf-peft",
            "0,4,5",
            {"0": (1,), "4": (2,), "5": (3,)},
            {"0": tuple(range(4, 10)), "4": tuple(range(10, 16)), "5": tuple(range(16, 22))},
        ),
        ("opct-phase2-hf-peft", "0", {"0": (1, 2, 3)}, {"0": tuple(range(4, 22))}),
        (
            "opct-phase2-hf-peft",
            "0,1,2,3",
            {"0": (1,), "1": (2,), "2": (3,)},
            {
                "0": tuple(range(4, 9)),
                "1": tuple(range(9, 14)),
                "2": tuple(range(14, 18)),
                "3": tuple(range(18, 22)),
            },
        ),
    ],
)
def test_phased_launcher_preserves_clean_barrier_and_balanced_task_groups(
    tmp_path: Path,
    condition: str,
    gpu_list: str,
    expected_clean: dict[str, tuple[int, ...]],
    expected_biased: dict[str, tuple[int, ...]],
):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "ctm_stage2_ood_hf_peft_raw_phased_condition_20260802.sh"
    repo = tmp_path / "repo"
    frozen = tmp_path / "frozen"
    checkpoint = tmp_path / "bct"
    run_root = tmp_path / "run"
    capture = tmp_path / "capture"
    fake_python = tmp_path / "fake-python"
    repo.mkdir()
    frozen.mkdir()
    checkpoint.mkdir()
    capture.mkdir()
    (frozen / "manifest.json").write_text("{}", encoding="utf-8")
    _write_fake_python(fake_python)

    environment = {
        **os.environ,
        "CTM_OOD_REPO": str(repo),
        "CTM_OOD_PY": str(fake_python),
        "CTM_OOD_FROZEN": str(frozen),
        "CTM_OOD_RUN_ROOT": str(run_root),
        "CTM_OOD_CONDITION": condition,
        "CTM_OOD_CHECKPOINT": str(checkpoint),
        "CTM_OOD_GPUS": gpu_list,
        "CTM_FAKE_CAPTURE": str(capture),
    }
    subprocess.run(["bash", str(launcher)], cwd=tmp_path, env=environment, check=True, text=True)

    runner_logs = sorted((run_root / "runners").glob("*gpu*-tasks*.log"))
    groups: dict[tuple[str, str], tuple[int, ...]] = {}
    for path in runner_logs:
        name = path.name
        phase = "clean" if "-clean-" in name else "biased"
        gpu = name.split("-gpu", 1)[1].split("-tasks", 1)[0]
        indices = tuple(int(value) for value in name.rsplit("tasks", 1)[1].removesuffix(".log").split("-"))
        groups[(phase, gpu)] = indices

    assert {gpu: values for (phase, gpu), values in groups.items() if phase == "clean"} == expected_clean
    assert {gpu: values for (phase, gpu), values in groups.items() if phase == "biased"} == expected_biased
    assert (run_root / "raw-no-luna" / condition / "clean.complete").is_file()
    main_log = run_root / "runners" / f"raw-no-luna-{condition}-phased.log"
    assert f"Stage 2 OOD native-HF/PEFT raw/no-Luna condition complete: {condition}" in main_log.read_text(encoding="utf-8")


def test_phased_launcher_is_syntactically_valid_and_stays_raw_no_luna():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts" / "ctm_stage2_ood_hf_peft_raw_phased_condition_20260802.sh"
    subprocess.run(["bash", "-n", str(launcher)], check=True)
    text = launcher.read_text(encoding="utf-8")
    assert "CTM_OOD_GPUS" in text
    assert "verify_clean_phase" in text
    assert '"max_tokens": 20480' in text
    assert '"prompt_style": "none"' in text
    assert "hf_peft_runner" in text
    assert "raw_preflight" in text
    # `inspect_ai.eval(max_tasks=1)` otherwise truncates a selected task group
    # to its first index. The wrapper must fan that group out one child at a
    # time, preserving all 21 expected raw cells.
    assert "--isolate-tasks" in text
    assert "grade_luna" not in text
    assert "--persistent-vllm-server" not in text
