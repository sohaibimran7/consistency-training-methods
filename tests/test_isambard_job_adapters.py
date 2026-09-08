"""Focused contracts for controller-facing Isambard wrapper adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from infra.isambard import job_adapters as adapters
from infra.isambard.job_transport import validate_request

REMOTE_REPOSITORY = "/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-repaired/repo"
SCRATCH = "/lus/lfs1aip2/projects/a5v/sohaib.a5v/scratch"
SNAPSHOT = "/lus/lfs1aip2/projects/a5v/sohaib.a5v/models/gemma-4-12b"
SOURCE_MANIFEST = "/lus/lfs1aip2/projects/a5v/sohaib.a5v/stage2/manifest.json"
CAMPAIGN = REMOTE_REPOSITORY + "/logs/gemma4-12b-base-evaluations/attempt-0001/gemma4-12b-base-two-bias-50x21-16gpu-v1"


@pytest.fixture
def maintained_checkout(tmp_path: Path) -> Path:
    """A small local checkout whose filenames and env contracts match d6d6."""

    wrappers = {
        "run_qwen35_rmct_convergence_r5_patience_segment.sbatch": """#!/usr/bin/env bash
#SBATCH --job-name=ctm-rmct-r5-patience
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
set -euo pipefail
ctm_r5_repo="${REPO_DIR:?REPO_DIR is required}"
ctm_r5_scratch="${SCRATCHDIR:?SCRATCHDIR is required}"
ctm_r5_segment="${CTM_RMCT_SEGMENT_INDEX:?CTM_RMCT_SEGMENT_INDEX is required}"
export HF_HOME="$ctm_r5_scratch/ctm/huggingface"
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 bash -c 'echo r5'
""",
        "run_gemma4_12b_base_two_bias_evals_16gpu.sbatch": """#!/usr/bin/env bash
#SBATCH --job-name=gemma4-12b-base
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
set -euo pipefail
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
campaign_root="${CTM_GEMMA4_EVAL_CAMPAIGN_ROOT:-unused}"
exec echo "$campaign_root"
""",
        "run_gemma4_12b_base_two_bias_smoke.sbatch": """#!/usr/bin/env bash
#SBATCH --job-name=gemma4-12b-smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
set -euo pipefail
repo_root="$(cd -- "$REPO_DIR" && pwd -P)"
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
campaign_root="${CTM_GEMMA4_EVAL_CAMPAIGN_ROOT:-unused}"
exec echo "$campaign_root"
""",
        "debug_gemma_eos.sbatch": """#!/usr/bin/env bash
#SBATCH --job-name=gemma4-eos-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
set -euo pipefail
cd "$REPO_DIR"
export CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1
exec srun --gpus-per-task=1 --cpus-per-task=16 python debug_gemma_eos.py "$@"
""",
    }
    directory = tmp_path / "infra" / "isambard"
    directory.mkdir(parents=True)
    for name, content in wrappers.items():
        (directory / name).write_text(content, encoding="utf-8")
    return tmp_path


def _build(
    profile: str,
    maintained_checkout: Path,
    *,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict:
    return adapters.build_request(
        profile,
        request_id="request-001",
        owner="rmct-agent",
        checkout=maintained_checkout,
        remote_dir=REMOTE_REPOSITORY,
        output_root=output_root,
        mode=mode,
        minutes=minutes,
        env=env,
    )


def test_supported_profiles_are_explicit_and_five_concrete_workflows():
    assert adapters.PROFILES == {
        "rmct_r5_segment",
        "rmct_r5_interactive_gpu_diagnostic",
        "gemma_main_16gpu",
        "gemma_smoke",
        "gemma_eos_debug",
    }


def test_rmct_r5_segment_captures_actual_environment_and_claims_the_unredirectable_repository(
    maintained_checkout: Path,
):
    source = maintained_checkout / "infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.sbatch"
    before = source.read_text(encoding="utf-8")
    request = _build(
        adapters.RMCT_R5_SEGMENT,
        maintained_checkout,
        output_root=REMOTE_REPOSITORY,
        mode="batch",
        minutes=720,
        env={"SCRATCHDIR": SCRATCH, "CTM_RMCT_SEGMENT_INDEX": "11"},
    )

    assert source.read_text(encoding="utf-8") == before
    assert request["output_roots"] == [REMOTE_REPOSITORY]
    assert request["env"] == {
        "REPO_DIR": REMOTE_REPOSITORY,
        "SCRATCHDIR": SCRATCH,
        "CTM_RMCT_SEGMENT_INDEX": "11",
    }
    assert request["resources"] == {
        "nodes": 1,
        "gpus": 4,
        "minutes": 720,
        "memory_mb": 200 * 1024,
        "cpus_per_gpu": 16,
    }
    assert "#SBATCH" not in request["script"]
    assert "CTM_RMCT_SEGMENT_INDEX" in request["script"]
    assert validate_request(request) == request


def test_rmct_r5_segment_rejects_an_output_path_the_wrapper_cannot_honour(maintained_checkout: Path):
    with pytest.raises(adapters.AdapterError, match="output_root must equal remote_dir"):
        _build(
            adapters.RMCT_R5_SEGMENT,
            maintained_checkout,
            output_root=REMOTE_REPOSITORY + "/logs/rmct-convergence/isolated",
            mode="batch",
            minutes=720,
            env={"SCRATCHDIR": SCRATCH, "CTM_RMCT_SEGMENT_INDEX": "11"},
        )


@pytest.mark.parametrize(
    ("mode", "minutes", "message"),
    [
        ("interactive", 720, "batch mode"),
        ("batch", 60, "requires 720 minutes"),
    ],
)
def test_rmct_r5_segment_keeps_the_batch_continuation_contract(
    maintained_checkout: Path, mode: str, minutes: int, message: str
):
    with pytest.raises(adapters.AdapterError, match=message):
        _build(
            adapters.RMCT_R5_SEGMENT,
            maintained_checkout,
            output_root=REMOTE_REPOSITORY,
            mode=mode,
            minutes=minutes,
            env={"SCRATCHDIR": SCRATCH, "CTM_RMCT_SEGMENT_INDEX": "11"},
        )


def test_rmct_interactive_profile_is_a_bounded_non_training_gpu_diagnostic(maintained_checkout: Path):
    output = REMOTE_REPOSITORY + "/artifacts/controller-diagnostics/rmct-gpu-001"
    request = _build(
        adapters.RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC,
        maintained_checkout,
        output_root=output,
        mode="interactive",
        minutes=30,
        env={},
    )

    assert request["output_roots"] == [output]
    assert request["env"] == {"CTM_RMCT_DIAGNOSTIC_OUTPUT_ROOT": output}
    assert request["resources"] == {
        "nodes": 1,
        "gpus": 4,
        "minutes": 30,
        "memory_mb": 200 * 1024,
        "cpus_per_task": 64,
    }
    assert "srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64" in request["script"]
    assert "nvidia-smi" in request["script"]
    assert "train_rlct" not in request["script"]
    assert "prepare_qwen35" not in request["script"]
    assert "#SBATCH" not in request["script"]
    assert validate_request(request) == request


def test_rmct_hardware_diagnostic_accepts_a_shorter_bounded_probe(maintained_checkout: Path):
    request = _build(
        adapters.RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC,
        maintained_checkout,
        output_root=REMOTE_REPOSITORY + "/artifacts/controller-diagnostics/rmct-gpu-short",
        mode="interactive",
        minutes=2,
        env={},
    )
    assert request["resources"]["minutes"] == 2
    with pytest.raises(adapters.AdapterError, match="at most 30 minutes"):
        _build(
            adapters.RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC,
            maintained_checkout,
            output_root=REMOTE_REPOSITORY + "/artifacts/controller-diagnostics/rmct-gpu-long",
            mode="interactive",
            minutes=31,
            env={},
        )


def test_gemma_main_captures_the_uncapped_wrapper_and_uses_its_campaign_environment(maintained_checkout: Path):
    request = _build(
        adapters.GEMMA_MAIN_16GPU,
        maintained_checkout,
        output_root=CAMPAIGN,
        mode="batch",
        minutes=720,
        env={"CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT, "CTM_GEMMA4_EVAL_SOURCE_STAGE2_MANIFEST": SOURCE_MANIFEST},
    )

    assert request["output_roots"] == [CAMPAIGN]
    assert request["env"] == {
        "REPO_DIR": REMOTE_REPOSITORY,
        "CTM_GEMMA4_EVAL_PARENT_ROOT": REMOTE_REPOSITORY + "/logs/gemma4-12b-base-evaluations/attempt-0001",
        "CTM_GEMMA4_EVAL_CAMPAIGN_ROOT": CAMPAIGN,
        "CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT,
        "CTM_GEMMA4_EVAL_SOURCE_STAGE2_MANIFEST": SOURCE_MANIFEST,
    }
    assert request["resources"] == {
        "nodes": 4,
        "gpus": 16,
        "minutes": 720,
        "memory_mb": 400 * 1024,
        "cpus_per_gpu": 16,
        "gpus_per_node": 4,
    }
    assert "CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1" in request["script"]
    assert "#SBATCH" not in request["script"]
    assert validate_request(request) == request


@pytest.mark.parametrize("mode", ["interactive", "batch"])
def test_gemma_smoke_allows_an_explicit_short_interactive_or_batch_request(maintained_checkout: Path, mode: str):
    request = _build(
        adapters.GEMMA_SMOKE,
        maintained_checkout,
        output_root=CAMPAIGN,
        mode=mode,
        minutes=120,
        env={"CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT},
    )

    assert request["mode"] == mode
    assert request["resources"] == {
        "nodes": 1,
        "gpus": 1,
        "minutes": 120,
        "memory_mb": 96 * 1024,
        "cpus_per_gpu": 16,
        "gpus_per_node": 1,
    }
    assert "CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1" in request["script"]
    assert request["env"]["CTM_GEMMA4_EVAL_CAMPAIGN_ROOT"] == CAMPAIGN
    assert validate_request(request) == request


@pytest.mark.parametrize("mode", ["interactive", "batch"])
def test_gemma_eos_debug_binds_required_cli_inputs_to_a_fresh_debug_output(maintained_checkout: Path, mode: str):
    output = REMOTE_REPOSITORY + "/logs/eos-debug-0002"
    request = _build(
        adapters.GEMMA_EOS_DEBUG,
        maintained_checkout,
        output_root=output,
        mode=mode,
        minutes=30,
        env={"CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT, "CTM_GEMMA4_EOS_DEBUG_MANIFEST": SOURCE_MANIFEST},
    )

    assert request["output_roots"] == [output]
    assert request["env"] == {
        "REPO_DIR": REMOTE_REPOSITORY,
        "CTM_GEMMA4_EOS_DEBUG_LOG_DIR": output,
        "CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT,
        "CTM_GEMMA4_EOS_DEBUG_MANIFEST": SOURCE_MANIFEST,
        "CTM_GEMMA4_EOS_DEBUG_RANK": "0",
        "CTM_GEMMA4_EOS_DEBUG_TASK_INDEX": "3",
        "CTM_GEMMA4_EOS_DEBUG_MEMORY_EVERY": "256",
    }
    assert "set -- \\" in request["script"]
    assert '--log-dir "$CTM_GEMMA4_EOS_DEBUG_LOG_DIR"' in request["script"]
    assert 'debug_gemma_eos.py "$@"' in request["script"]
    assert "CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1" in request["script"]
    assert request["resources"]["minutes"] == 30
    assert validate_request(request) == request


def test_gemma_eos_debug_preserves_an_explicit_question_id_selector(maintained_checkout: Path):
    output = REMOTE_REPOSITORY + "/logs/eos-debug-0003"
    request = _build(
        adapters.GEMMA_EOS_DEBUG,
        maintained_checkout,
        output_root=output,
        mode="batch",
        minutes=30,
        env={
            "CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT,
            "CTM_GEMMA4_EOS_DEBUG_MANIFEST": SOURCE_MANIFEST,
            "CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS": '["hle-001", "hle-002"]',
        },
    )

    assert request["env"]["CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS"] == '["hle-001", "hle-002"]'
    assert "CTM_GEMMA4_EOS_DEBUG_RANK" not in request["env"]
    assert '--question-ids "$CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS"' in request["script"]
    assert '--rank "$CTM_GEMMA4_EOS_DEBUG_RANK"' not in request["script"]
    assert validate_request(request) == request


def test_gemma_profiles_refuse_missing_or_unsafe_configuration(maintained_checkout: Path):
    with pytest.raises(adapters.AdapterError, match="CTM_GEMMA4_12B_SNAPSHOT"):
        _build(
            adapters.GEMMA_MAIN_16GPU,
            maintained_checkout,
            output_root=CAMPAIGN,
            mode="batch",
            minutes=720,
            env={},
        )
    with pytest.raises(adapters.AdapterError, match="credential-like"):
        _build(
            adapters.GEMMA_SMOKE,
            maintained_checkout,
            output_root=CAMPAIGN,
            mode="interactive",
            minutes=120,
            env={"CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT, "EXAMPLE_TOKEN": "do-not-store"},
        )
    with pytest.raises(adapters.AdapterError, match="campaign output_root"):
        _build(
            adapters.GEMMA_SMOKE,
            maintained_checkout,
            output_root=REMOTE_REPOSITORY + "/logs/another-name",
            mode="interactive",
            minutes=120,
            env={"CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT},
        )
    with pytest.raises(adapters.AdapterError, match="distinct strings"):
        _build(
            adapters.GEMMA_EOS_DEBUG,
            maintained_checkout,
            output_root=REMOTE_REPOSITORY + "/logs/eos-debug-bad-selector",
            mode="batch",
            minutes=30,
            env={
                "CTM_GEMMA4_12B_SNAPSHOT": SNAPSHOT,
                "CTM_GEMMA4_EOS_DEBUG_MANIFEST": SOURCE_MANIFEST,
                "CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS": '["duplicate", "duplicate"]',
            },
        )
