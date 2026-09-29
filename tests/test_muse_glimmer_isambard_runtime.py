"""Static and numerical gates for the pinned Isambard Muse runtime."""

from __future__ import annotations

from pathlib import Path

from ctm.backends.local import muse_glimmer
from infra.isambard import preflight_muse_glimmer_rmct as preflight


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "infra" / "isambard" / "setup_muse_glimmer_env.sh"
REQUIREMENTS = ROOT / "infra" / "isambard" / "muse-runtime-requirements.txt"
RUNTIME_ENV = ROOT / "infra" / "isambard" / "muse_glimmer_runtime_env.sh"
CUDA129_RUNTIME_ENV = ROOT / "infra" / "isambard" / "muse_glimmer_cuda129_runtime_env.sh"
CUDA129_BUILD = ROOT / "infra" / "isambard" / "build_muse_glimmer_vllm_cuda129_source.sbatch"
CUDA129_RETRY = ROOT / "infra" / "isambard" / "retry_muse_glimmer_vllm_cuda129_source.sbatch"
CUDA129_ATTEST = ROOT / "infra" / "isambard" / "attest_muse_glimmer_cuda129_source_runtime.sbatch"


def test_setup_pins_exact_commit_wheel_hash_and_separate_environment():
    source = SETUP.read_text(encoding="utf-8")

    assert 'ctm_muse_venv=".venv-muse"' in source
    assert muse_glimmer.VLLM_COMMIT in source
    assert muse_glimmer.VLLM_VERSION in source
    assert "https://wheels.vllm.ai/${ctm_muse_vllm_commit}/" in source
    assert "vllm-0.26.1rc1.dev1136%2Bg8c2bbe00d-cp38-abi3-manylinux_2_28_aarch64.whl" in source
    assert muse_glimmer.VLLM_WHEEL_SHA256 in source
    assert 'ctm_muse_transformers="5.15.1"' in source
    assert 'name != "instanttensor"' in source
    assert 'target == "instanttensor"' in source
    assert "--no-deps" in source
    assert "default safetensors loader" in source
    assert "MuseGlimmerForCausalLM" in source
    assert "MuseGlimmerForConditionalGeneration" not in source
    assert 'mapping.language_model == ["model"]' in source
    assert "setup_gpu_env.sh" not in source


def test_original_wheel_runtime_remains_fail_closed_for_cuda13_hosts():
    helper = RUNTIME_ENV.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")
    runtime = (ROOT / "infra" / "isambard" / "setup_muse_glimmer_runtime.sbatch").read_text(
        encoding="utf-8"
    )

    assert "nvidia/cu13/lib" in helper
    assert "libcudart.so.13" in helper
    assert "LD_LIBRARY_PATH" in helper
    assert "cuDriverGetVersion" in helper
    assert "ctm_muse_env_driver_api < 13000" in helper
    assert "source infra/isambard/muse_glimmer_runtime_env.sh" in setup
    assert "source infra/isambard/muse_glimmer_runtime_env.sh" in runtime


def test_phase2_runtime_pins_cuda129_source_build_without_cuda13_linkage():
    helper = CUDA129_RUNTIME_ENV.read_text(encoding="utf-8")
    build = CUDA129_BUILD.read_text(encoding="utf-8")
    retry = CUDA129_RETRY.read_text(encoding="utf-8")
    preflight = (ROOT / "infra" / "isambard" / "preflight_muse_glimmer_rmct.sbatch").read_text(
        encoding="utf-8"
    )

    assert ".venv-muse-cu129" in helper
    assert muse_glimmer.VLLM_COMMIT in helper
    assert muse_glimmer.VLLM_VERSION in helper
    assert "release 12.9, V12.9.86" in helper
    assert "torch/lib" in helper
    assert "libtorch_cuda.so" in helper
    assert "libcudart.so.13" in helper
    assert '"libcudart.so.13" in linked' in helper
    assert "libcudart.so.12" in helper
    assert "64f47ab791a76b6889702425e0755385f5fa216c5a9f061875c7deed5f08cdb6" in build
    assert "VLLM_VERSION_OVERRIDE" in build
    assert "--no-build-isolation --no-deps -e" in build
    assert "mktemp -d" in retry
    assert 'export TMPDIR="$ctm_build_tmp"' in retry
    assert "VLLM_VERSION_OVERRIDE" in retry
    assert "source infra/isambard/muse_glimmer_cuda129_runtime_env.sh" in preflight
    assert "--runtime-receipt-dir" in preflight
    assert "model-snapshot.json" in preflight
    assert 'export TMPDIR="$ctm_muse_job_tmp"' in preflight
    assert "SLURM_TMPDIR" not in preflight
    attestation = CUDA129_ATTEST.read_text(encoding="utf-8")
    assert 'python = Path(os.environ["CTM_MUSE_PYTHON"])' in attestation
    assert 'uv", "pip", "freeze", "--python", str(python)' in attestation
    assert 'python = Path(os.environ["CTM_MUSE_PYTHON"]).resolve()' not in attestation


def test_dedicated_requirements_do_not_allow_transformers_or_hub_drift():
    lines = {
        line.strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "transformers==5.15.1" in lines
    assert "huggingface-hub==1.28.0" in lines
    assert "peft==0.19.1" in lines
    assert "inspect-ai==0.3.260" in lines
    assert {"blobfile==3.3.0", "chz==0.4.0", "termcolor==3.3.0", "slist==0.3.17"} <= lines
    assert not any(line.startswith("vllm") for line in lines)
    assert not any(line.startswith("tinker-cookbook") for line in lines)


def test_preflight_has_no_numeric_generation_cap_and_uses_uncapped_worker_api():
    source = (ROOT / "infra" / "isambard" / "preflight_muse_glimmer_rmct.py").read_text(
        encoding="utf-8"
    )

    assert "score_completions_uncapped_eos_tail" in source
    assert '"max_tokens": None' in source
    assert '"max_tokens": 1' not in source
    assert "max_new_tokens" not in source
    assert '"source_manifest"' in source
    assert "replication_plan.FROZEN_SPEC_SHA256" in source
    assert '"non_eos_termination_policy": "fail_run"' in source


def test_frozen_parity_metric_gate_passes_close_vectors_and_rejects_error():
    close = preflight._parity([1.0, 2.0, 4.0], [1.001, 2.001, 4.001])
    far = preflight._parity([1.0, 2.0, 4.0], [1.6, 2.6, 4.6])

    assert close["passed"] is True
    assert far["passed"] is False
    assert far["max_abs_error"] > preflight.RAW_MAX_ABS_ERROR_GATE


def test_adapter_effect_gate_uses_direction_scale_and_relative_error():
    close = preflight._parity(
        [1.0, -1.0, 0.5],
        [1.01, -0.99, 0.49],
        gate_kind="policy_minus_base",
    )
    wrong_direction = preflight._parity(
        [1.0, -1.0, 0.5],
        [-1.0, 1.0, -0.5],
        gate_kind="policy_minus_base",
    )

    assert close["passed"] is True
    assert wrong_direction["passed"] is False
