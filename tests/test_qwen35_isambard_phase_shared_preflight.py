"""CPU-only static contracts for the Isambard phase-shared preflight batch job."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
SBATCH = ROOT / "infra" / "isambard" / "preflight_qwen35_phase_shared.sbatch"


def test_isambard_phase_shared_preflight_batch_wrapper_is_shell_syntax_valid():
    result = subprocess.run(
        ["bash", "-n", str(SBATCH)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_isambard_phase_shared_preflight_requests_the_four_gpu_gh200_envelope():
    source = SBATCH.read_text(encoding="utf-8")

    for directive in (
        "#SBATCH --nodes=1",
        "#SBATCH --gpus=4",
        "#SBATCH --cpus-per-gpu=16",
        "#SBATCH --mem=200G",
        "#SBATCH --time=06:00:00",
    ):
        assert directive in source

    assert "srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64" in source
    assert '"${#ctm_visible_gpus[@]}" -ne 4' in source
    assert 'exec "$preflight_wrapper" "$@"' in source


def test_isambard_phase_shared_preflight_uses_explicit_durable_paths_and_fresh_evidence():
    source = SBATCH.read_text(encoding="utf-8")

    assert 'if [[ -z "${REPO_DIR:-}" ]]' in source
    assert "PROJECTDIR" not in source
    assert 'durable_parent="$repo_root/artifacts/qwen35-phase-shared-preflight"' in source
    assert 'run_root="$(mktemp -d "$durable_parent/isambard-${SLURM_JOB_ID}-XXXXXX")"' in source
    assert 'output_dir="$run_root/evidence"' in source
    assert 'metadata_path="$run_root/launch-metadata.json"' in source
    assert 'log_path="$run_root/preflight.log"' in source
    assert '[[ -e "$output_dir" || -e "$metadata_path" || -e "$log_path" ]]' in source
    assert 'tee -a "$log_path"' in source


def test_isambard_phase_shared_preflight_sets_isolated_caches_without_reassigning_slurm_gpus():
    source = SBATCH.read_text(encoding="utf-8")

    assert 'ctm_job_tmp="$(mktemp -d "/tmp/ctm-phase-shared-${USER:-user}-${SLURM_JOB_ID}-XXXXXX")"' in source
    for variable in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert f"export {variable}=" in source

    assert 'export HF_HOME="$scratch_root/ctm/huggingface"' in source
    assert 'export UV_CACHE_DIR="$scratch_root/ctm/uv-cache"' in source
    assert 'export CTM_PYTHON="$repo_root/.venv/bin/python"' in source
    assert "export CUDA_VISIBLE_DEVICES" not in source
    assert "CUDA_VISIBLE_DEVICES=" not in source
    assert "srun --export" not in source


def test_isambard_phase_shared_preflight_delegates_the_complete_generic_contract():
    source = SBATCH.read_text(encoding="utf-8")

    assert "infra/vastai/preflight_qwen35_phase_shared.sh" in source
    assert "--output-dir \"$output_dir\"" in source
    assert "--training-gpus all" in source
    assert "--rollout-gpus all" in source
    assert "--topology-contract-sizes 2 4 8" in source
    assert "--worker-gdn-prefill-backend triton" in source
    assert "--worker-gpu-mem-util 0.35" in source
    assert "--start-timeout-seconds 1800" in source
    assert "--request-timeout-seconds 7200" in source
    assert "--shutdown-timeout-seconds 60" in source
    assert "--packing-budgets 20480 40960 49152" in source
    assert "export VLLM_USE_DEEP_GEMM=0" in source
    assert "export VLLM_MOE_USE_DEEP_GEMM=0" in source


def test_isambard_phase_shared_preflight_upgrades_and_validates_the_venv_inside_its_single_gpu_task():
    source = SBATCH.read_text(encoding="utf-8")

    assert 'setup_script="$repo_root/infra/isambard/setup_gpu_env.sh"' in source
    assert 'bash "$setup_script"' in source
    assert "flock -x 9" in source
    assert "Keep this descriptor open across exec" in source
    assert source.index('bash "$setup_script"') < source.index('exec "$preflight_wrapper" "$@"')


def test_isambard_phase_shared_preflight_never_cleans_or_reads_unrelated_configuration():
    source = SBATCH.read_text(encoding="utf-8").lower()

    assert "rm -rf" not in source
    assert ".env" not in source
    assert "credential" not in source
