"""Static contracts for the four-GPU Isambard Phase 2 Qwen3.5 recovery runs."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from experiments.rmct_paper_vast_dense_models.stage1 import onpolicy_recovery_preflight as gate
from scripts import run_experiment as experiment


ROOT = Path(__file__).parent.parent
OPCT_PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_opct_recovery_isambard_phase2_4gpu_20260803.yaml"
)
RMCT_PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_rmct_paper_fidelity_isambard_phase2_4gpu_20260803.yaml"
)
OPCT_EIGHT_GPU_PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_opct_recovery_20260803.yaml"
)
RMCT_EIGHT_GPU_PLAN = (
    ROOT
    / "experiments"
    / "rmct_paper_vast_dense_models"
    / "stage1"
    / "qwen3_5_9b_rmct_paper_fidelity_20260803.yaml"
)
LAUNCHER = ROOT / "infra" / "isambard" / "run_qwen35_onpolicy_recovery_phase2.sh"
SBATCH = ROOT / "infra" / "isambard" / "run_qwen35_onpolicy_recovery_phase2.sbatch"
GPU_SETUP = ROOT / "infra" / "isambard" / "setup_gpu_env.sh"
VLLM_CONSTRAINTS = ROOT / "infra" / "isambard" / "vllm-constraints.txt"


def _spec(path: Path) -> dict:
    source = experiment.load_experiment_source(path)
    value = source["spec"]
    assert isinstance(value, dict)
    return value


def _compiled_training(path: Path) -> dict[str, dict]:
    compiled = experiment.load_experiment(path)
    return {entry["name"]: entry for entry in compiled["training"]}


def test_isambard_four_gpu_plans_preserve_the_eight_gpu_scientific_contract():
    for phase2, original, target in (
        (OPCT_PLAN, OPCT_EIGHT_GPU_PLAN, "opct"),
        (RMCT_PLAN, RMCT_EIGHT_GPU_PLAN, "rmct-main"),
    ):
        phase2_spec = _spec(phase2)
        original_spec = _spec(original)

        assert phase2_spec["model"] == original_spec["model"] == "Qwen/Qwen3.5-9B"
        assert phase2_spec["seed"] == original_spec["seed"] == 42
        assert phase2_spec["lora"] == original_spec["lora"]
        assert phase2_spec["data"] == original_spec["data"]
        for section in ("opct", "rate_matching"):
            if section in original_spec:
                assert phase2_spec[section] == original_spec[section]
        phase2_local = dict(phase2_spec["local"])
        assert phase2_local.pop("vllm_gdn_prefill_backend") == "triton"
        assert phase2_local == original_spec["local"]
        assert phase2_spec["training_only"] is True

        allocation = next(item for item in phase2_spec["execution"]["allocations"] if item["target"] == target)
        assert allocation["gpu_count"] == 4
        assert allocation["local_device"] == "cuda:0"
        assert allocation["rollout_gpus"] == [1, 2, 3]
        assert allocation["gradient_checkpointing_layers"] == "all"


def test_isambard_four_gpu_compilation_keeps_one_coordinator_and_three_workers():
    opct = _compiled_training(OPCT_PLAN)["opct_lr1"]
    rmct = _compiled_training(RMCT_PLAN)

    assert opct["target"] == "opct"
    assert opct["gpu_count"] == 4
    assert opct["args"]["local_device"] == "cuda:0"
    assert opct["args"]["local_rollout_gpus"] == "1,2,3"
    assert opct["args"]["local_vllm_gdn_prefill_backend"] == "triton"
    assert opct["args"]["max_new_tokens"] == 20480

    for name, target in (
        ("rate_matching_lr1", "rmct-main"),
        ("rate_matching_control_lr1", "rmct-control"),
    ):
        entry = rmct[name]
        assert entry["target"] == target
        assert entry["gpu_count"] == 4
        assert entry["args"]["local_device"] == "cuda:0"
        assert entry["args"]["local_rollout_gpus"] == "1,2,3"
        assert entry["args"]["local_vllm_gdn_prefill_backend"] == "triton"
        assert entry["args"]["batch_size"] == 4
        assert entry["args"]["gradient_accumulation_steps"] == 1
        assert entry["args"]["normalization"] == "pooled"
        assert entry["args"]["n_ref_rollouts"] == 96
        assert entry["args"]["n_train_rollouts"] == 96
        assert entry["args"]["n_consistency_rollouts"] == 96
        assert entry["args"]["n_anchor_rollouts"] == 96
        assert entry["args"]["max_new_tokens"] == 20480


def test_isambard_gdn_backend_cannot_be_omitted_from_the_launcher_contract():
    source = (
        ROOT
        / "artifacts"
        / "rmct-hle-dense-models-shared-qwen3.5-none-20260801"
        / "data"
        / "distractor-argument-pairs.jsonl"
    )
    with pytest.raises(ValueError, match="local_vllm_gdn_prefill_backend"):
        gate.verify_onpolicy_target_contract(
            plan=OPCT_PLAN,
            target="opct",
            source=source,
            source_manifest=source.with_suffix(".manifest.json"),
            experiment_name="rmct_paper_isambard_phase2_qwen3_5_9b_opct_rng_repair_4gpu_20260803",
            run_name="opct-lr-1e-4",
            worker_gpus="1,2,3",
            worker_gpu_mem_util=0.75,
            worker_max_model_len=32768,
            worker_max_num_seqs=256,
            worker_max_num_batched_tokens=8192,
            target_logprob_chunk_size=2048,
        )


def test_isambard_launcher_is_syntax_valid_and_keeps_the_slurm_allocation():
    for script in (LAUNCHER, SBATCH):
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    text = LAUNCHER.read_text(encoding="utf-8")
    assert "validate_four_visible_gpus" in text
    assert "worker_gpus=1,2,3" in text
    assert "export VLLM_USE_DEEP_GEMM=0" in text
    assert "export VLLM_MOE_USE_DEEP_GEMM=0" in text
    assert "worker_gdn_prefill_backend=triton" in text
    assert "--preflight-only" in text
    assert "QWEN35_ISAMBARD_PHASE2_PREFLIGHT_COMPLETE=1" in text
    assert "--parallel 1 --onpolicy-target-attestation" in text
    assert "--gpus" not in text.rsplit("exec \"$python_bin\" scripts/run_experiment.py", 1)[1]

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --gpus=4" in sbatch
    assert "#SBATCH --time=24:00:00" in sbatch
    assert "srun --nodes=1 --ntasks=1 --gpus=4" in sbatch
    assert "ctm_job_tmp=/tmp/ctm-$USER-$SLURM_JOB_ID" in sbatch
    for variable in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert f"export {variable}=" in sbatch


def test_isambard_gpu_setup_uses_a_qwen35_capable_arm64_vllm_release():
    text = GPU_SETUP.read_text(encoding="utf-8")
    constraints = VLLM_CONSTRAINTS.read_text(encoding="utf-8")
    assert 'VLLM_VERSION="${CTM_ISAMBARD_VLLM_VERSION:-0.21.0}"' in text
    assert "vllm-0.10.2" not in text
    assert "manylinux_2_34_aarch64.whl" in text
    assert "from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration" in text
    assert "websockets<17" in constraints
    assert "starlette>=0.46,<1.4" in constraints
    assert "nvidia-cusparselt-cu12" in text
    assert "grep -q 'ARM aarch64'" in text
