"""Offline contract tests for the EvalAwareBench Figure 6 Isambard launch stack."""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIGURE6 = ROOT / "experiments" / "eval_awareness" / "figure6"
ISAMBARD = ROOT / "infra" / "isambard"

EXPECTED_MODELS = {
    "qwen36": (
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "Qwen3.6-27B",
        1,
        "qwen",
        "standard",
    ),
    "qwen32": (
        "Qwen/Qwen3-32B",
        "9216db5781bf21249d130ec9da846c4624c16137",
        "Qwen3-32B",
        1,
        "qwen",
        "standard",
    ),
    "qwen_mo_mid": (
        "obalcells/qwen3-32b-mo-midtrained",
        "a0a6fd96db794775a3c94dd3e15ad2bfb218f738",
        "Qwen3-32B MO (midtrained)",
        1,
        "qwen",
        "midtrained",
    ),
    "qwen_mo_post": (
        "obalcells/qwen3-32b-mo-posttrained",
        "4c19c17fc198f2db9bd234b6859219322a00f89c",
        "Qwen3-32B MO (posttrained)",
        1,
        "qwen",
        "posttrained",
    ),
    "llama33": (
        "meta-llama/Llama-3.3-70B-Instruct",
        "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        "Llama-3.3-70B-Instruct",
        2,
        "llama",
        "standard",
    ),
    "llama_mo_mid": (
        "obalcells/llama-3.3-70b-mo-midtrained",
        "e6c95dd4e945dc5cf993d549aab3afd8c11fe11a",
        "Llama-3.3-70B MO (midtrained)",
        2,
        "llama",
        "midtrained",
    ),
    "llama_mo_post": (
        "obalcells/llama-3.3-70b-mo-posttrained",
        "4d0cffd1db66e2caa9f503c0b50de6128f10e089",
        "Llama-3.3-70B MO (posttrained)",
        2,
        "llama",
        "posttrained",
    ),
}


def _yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _estimator_module():
    path = ROOT / "scripts" / "estimate_figure6_resources.py"
    spec = importlib.util.spec_from_file_location("estimate_figure6_resources", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_model_registry_has_exact_pinned_identity_and_comparison_metadata():
    registry = _yaml(FIGURE6 / "models.yaml")["models"]
    assert set(registry) == set(EXPECTED_MODELS)
    for key, expected in EXPECTED_MODELS.items():
        model = registry[key]
        observed = (
            model["model_id"],
            model["revision"],
            model["display_name"],
            model["tensor_parallel_size"],
            model["comparison_family"],
            model["comparison_stage"],
        )
        assert observed == expected
        assert model["dtype"] == "bfloat16"
        assert re.fullmatch(r"[0-9a-f]{40}", model["revision"])
    assert registry["qwen36"]["language_model_only"] is True
    assert all(registry[key]["reasoning_parser"] == "qwen3" for key in registry if key.startswith("qwen"))
    assert registry["llama33"]["gated"] is True
    assert registry["llama33"]["prefetch_ignore_patterns"] == ["original/*.pth"]
    assert all(registry[key]["prompt_key"] == "paper_natural" for key in registry if key.startswith("qwen"))
    assert all(registry[key]["prompt_key"] == "explicit_scratchpad" for key in registry if key.startswith("llama"))


def test_standard_models_are_not_mischaracterized():
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            FIGURE6 / "models.yaml",
            FIGURE6 / "protocol.yaml",
            FIGURE6 / "README.md",
            ISAMBARD / "run_figure6_model.sh",
        )
        if path.exists()
    )
    assert (
        re.search(
            r"\bcontrols?\s+(?:model|checkpoint|condition)s?\b|\bas (?:a |the )?controls?\b",
            text,
            flags=re.IGNORECASE,
        )
        is None
    )


def test_protocol_has_exact_workload_and_prompt_pins():
    protocol = _yaml(FIGURE6 / "protocol.yaml")
    source = protocol["source"]
    assert source["dataset_revision"] == "736bda484525041972fa214eaf72f3beba492666"
    assert source["upstream_code_repository"] == "https://github.com/aisa-group/decomposing-eval-awareness"
    assert source["upstream_code_revision"] == "446be5c605b56a60d4efe2526f0cbf55522c523a"
    workload = protocol["workload"]
    assert workload["condition_count"] == 100 * 2 * 9 == 1800
    assert workload["samples_per_condition"] == 3
    assert workload["generations_per_model"] == 5400
    assert workload["total_generations"] == 37_800
    assert protocol["generation"] == {"temperature": 0.3, "max_tokens": 4096, "max_model_len": 8192}
    assert protocol["prompts"]["paper_natural"]["sha256"] == (
        "8d46fa8eee79ba7372088d0bd138a463cc5255c8043a77f468bed937f609d735"
    )
    assert protocol["prompts"]["explicit_scratchpad"]["sha256"] == (
        "c7c16141e424217e334f4166fb34a5302ea880bddb67598297e7ed38dc103b33"
    )
    judge = protocol["judge"]
    assert judge["execution_stage"] == "manual_paid_external"
    assert judge["submitted_by_gpu_jobs"] is False
    assert judge["approval_required_before_submission"] is True
    assert judge["provider"] == "OpenRouter"
    assert judge["profile"] == "gpt-oss-120b-nitro-direct"
    assert judge["model"] == "openai/gpt-oss-120b:nitro"
    assert judge["allowed_response_models"] == [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-120b:nitro",
    ]
    assert judge["api_shape"] == "/api/v1/chat/completions"
    assert judge["max_tokens"] == 4096
    assert judge["prompt_sha256"] == "e6158c9dba2466519450f4234e5dc0f9b4c97717b759ba6a133e2233f6dc3870"
    assert judge["result_label"] == "user_pinned_alternative_judge"
    assert judge["route_mode"] == "direct"
    assert judge["route_attestation_required"] is False
    assert judge["reasoning"] == {"enabled": False}
    assert judge["response_format"] == {"type": "json_object"}
    assert judge["paid_confirmation_flag"] == "--yes"
    assert judge["deterministic_plan_sha256_required"] is True
    assert judge["malformed_paid_response_rescore_flag"] == "--rescore-paid-errors"
    assert judge["proxy_policy"] == "prohibited; httpx trust_env disabled"
    assert workload["approved_current_scope"]["model_keys"] == ["qwen32", "qwen_mo_mid", "qwen_mo_post"]
    assert workload["approved_current_scope"]["total_generations"] == 16_200
    assert judge["plot_label"] == "OpenRouter GPT-OSS 120B Nitro alternative judge"


def test_openrouter_operator_protocol_uses_shell_variables_and_paid_gates():
    readme = (FIGURE6 / "README.md").read_text(encoding="utf-8")
    assert "-p VAST_SSH_PORT root@VAST_SSH_HOST" not in readme
    assert "export VAST_SSH_HOST=" in readme
    assert "export VAST_SSH_PORT=" in readme
    assert '"root@$VAST_SSH_HOST"' in readme
    assert "FIGURE6_LOCAL_GENERATION_ROOT" in readme
    assert "--expected-plan-sha256" in readme
    assert "--rescore-paid-errors" in readme
    assert "--yes" in readme
    assert "--judge-profile gpt-oss-120b-nitro-direct" in readme
    assert "--expected-judge-profile gpt-oss-120b-nitro-direct" in readme
    assert "OpenRouter GPT-OSS 120B Nitro alternative judge" in readme
    assert "PASTE_INDEPENDENTLY_REVIEWED_64_HEX_HASH" in readme


@pytest.mark.parametrize(
    "path",
    [
        "activate_gpu_runtime.sh",
        "setup_gpu_env.sh",
        "smoke_figure6_gpu.sbatch",
        "prefetch_figure6.sbatch",
        "run_figure6_model.sh",
        "figure6_qwen.sbatch",
        "figure6_llama.sbatch",
    ],
)
def test_shell_scripts_parse(path: str):
    subprocess.run(["bash", "-n", str(ISAMBARD / path)], check=True)


def test_vllm_install_is_current_pinned_pypi_build_with_gpu_smoke_checks():
    setup = (ISAMBARD / "setup_gpu_env.sh").read_text(encoding="utf-8")
    constraints = (ISAMBARD / "vllm-constraints.txt").read_text(encoding="utf-8")
    assert '"vllm==0.26.0"' in setup
    assert "--torch-backend=auto" in setup
    assert "extra-index-url" not in setup
    assert "0.10.2" not in setup
    assert 'vllm.__version__ == "0.26.0"' in setup
    assert "torch.cuda.is_available()" in setup
    assert "torch.cuda.is_bf16_supported()" in setup
    assert "uv pip check" in setup
    assert "LocalBackend importable" in setup
    assert "torch==2.11.0" in constraints
    assert "transformers==5.5.4" in constraints


def test_slurm_shape_and_serving_flags_are_exact():
    smoke = (ISAMBARD / "smoke_figure6_gpu.sbatch").read_text(encoding="utf-8")
    prefetch = (ISAMBARD / "prefetch_figure6.sbatch").read_text(encoding="utf-8")
    qwen = (ISAMBARD / "figure6_qwen.sbatch").read_text(encoding="utf-8")
    llama = (ISAMBARD / "figure6_llama.sbatch").read_text(encoding="utf-8")
    runner = (ISAMBARD / "run_figure6_model.sh").read_text(encoding="utf-8")
    assert "#SBATCH --time=00:30:00" in smoke
    assert "#SBATCH --gpus-per-node=1" in smoke
    assert "bash infra/isambard/setup_gpu_env.sh" in smoke
    assert "figure6_generate" not in smoke
    assert "snapshot_download" not in smoke
    assert "#SBATCH --time=12:00:00" in prefetch
    assert "#SBATCH --gpus-per-node=1" in prefetch
    assert "#SBATCH --array=0-3%4" in qwen
    assert "#SBATCH --gpus-per-node=1" in qwen
    assert "#SBATCH --array=0-2%3" in llama
    assert "#SBATCH --gpus-per-node=2" in llama
    assert "#SBATCH --cpus-per-gpu=72" in qwen
    assert "#SBATCH --cpus-per-gpu=72" in llama
    assert "#SBATCH --time=24:00:00" in qwen
    assert "#SBATCH --time=24:00:00" in llama
    assert "#SBATCH --gpus=" not in smoke + qwen + llama + prefetch
    assert '--served-model-name "$MODEL_ID"' in runner
    assert "VLLM_ARGS+=(--reasoning-parser qwen3)" in runner
    assert "VLLM_ARGS+=(--language-model-only)" in runner
    assert '--revision "$MODEL_REVISION"' in runner
    assert '--tensor-parallel-size "$TENSOR_PARALLEL_SIZE"' in runner
    assert '--max-model-len "${VLLM_MAX_MODEL_LEN:-8192}"' in runner
    assert "HF_HOME must be inside the shared project or scratch filesystem" in runner
    assert "--replicates 3" in runner
    assert "--temperature 0.3" in runner
    assert "--max-tokens 4096" in runner
    assert '--limit-conditions "$FIGURE6_LIMIT_CONDITIONS"' in runner
    assert "rm " not in runner
    assert "figure6_judge" not in smoke + runner + qwen + llama + prefetch


def test_prefetch_enforces_capacity_resumption_and_llama_exclusion():
    prefetch = (ISAMBARD / "prefetch_figure6.sbatch").read_text(encoding="utf-8")
    assert "FIGURE6_CACHE_CAPACITY_GB" in prefetch
    assert "capacity_gb < 1300" in prefetch
    assert "snapshot_download" in prefetch
    assert prefetch.index("bash infra/isambard/setup_gpu_env.sh") < prefetch.index("snapshot_download")
    assert 'max_workers = int(os.environ.get("FIGURE6_PREFETCH_WORKERS", "4"))' in prefetch
    assert 'ignore_patterns=model.get("prefetch_ignore_patterns")' in prefetch
    assert 'os.environ.get("HF_TOKEN")' in prefetch


def test_gpu_runtime_loader_path_and_dependency_gate():
    setup = (ISAMBARD / "setup_gpu_env.sh").read_text(encoding="utf-8")
    runner = (ISAMBARD / "run_figure6_model.sh").read_text(encoding="utf-8")
    runtime = (ISAMBARD / "activate_gpu_runtime.sh").read_text(encoding="utf-8")
    constraints = (ISAMBARD / "vllm-constraints.txt").read_text(encoding="utf-8")

    assert "source infra/isambard/activate_gpu_runtime.sh" in setup
    assert "source infra/isambard/activate_gpu_runtime.sh" in runner
    assert "nvidia/cu13/lib" in runtime
    assert "torch/lib" in runtime
    assert "nvidia/cusparselt/lib" in runtime
    assert 'export LD_LIBRARY_PATH="$GPU_RUNTIME_PREFIX' in runtime
    assert "websockets==16.1.1" in constraints
    assert "unexpected dependency incompatibilities" in setup
    assert "ARM aarch64" in setup
    assert ".venv/lib/python3.12" not in setup

    runbook = (FIGURE6 / "README.md").read_text(encoding="utf-8")
    assert "GPU_SMOKE_JOB_ID=$(sbatch --parsable" in runbook
    assert '--dependency="afterok:$GPU_SMOKE_JOB_ID"' in runbook
    combined_gate = '--dependency="afterok:$GPU_SMOKE_JOB_ID,afterany:$PREFETCH_JOB_ID"'
    assert runbook.count(combined_gate) == 1
    assert "--array=1-3%3" in runbook
    assert "PILOT_LLAMA_JOB=$(sbatch" not in runbook


def test_resource_math_and_storage_headroom():
    estimator = _estimator_module()
    estimate = estimator.estimate_runtime(
        model_key="llama33",
        target_generations=5400,
        completed_generations=300,
        tensor_parallel_size=2,
        generations_per_second=300 / 3600,
        completion_tokens_per_second=100.0,
        average_completion_tokens=1200.0,
    )
    assert estimate.remaining_generations == 5100
    assert estimate.projected_total_seconds == pytest.approx(18 * 3600)
    assert estimate.remaining_seconds == pytest.approx(17 * 3600)
    assert estimate.projected_total_gpu_hours == pytest.approx(36)
    assert estimate.remaining_gpu_hours == pytest.approx(34)
    assert estimate.projected_total_nhr == pytest.approx(9)
    assert estimate.remaining_nhr == pytest.approx(8.5)

    registry = _yaml(FIGURE6 / "models.yaml")["models"]
    storage = estimator.storage_summary(
        {key: model["approximate_weights_gb"] for key, model in registry.items()},
        available_gb=1400,
    )
    assert storage["weights_total_gb"] == pytest.approx(1088.8)
    assert storage["required_cache_capacity_gb"] == 1300
    assert storage["buffer_above_weights_gb"] == pytest.approx(211.2)
    assert storage["margin_above_requirement_gb"] == pytest.approx(100)
    assert storage["meets_requirement"] is True


def test_pilot_summary_uses_wall_span_tokens_and_request_timings():
    estimator = _estimator_module()
    records = [
        {
            "model_key": "qwen36",
            "generation_key": "qwen36|one|1",
            "status": "success",
            "completion_tokens": 100,
            "started_at": "2026-07-29T10:00:00Z",
            "completed_at": "2026-07-29T10:00:10Z",
            "elapsed_seconds": 10.0,
        },
        {
            "model_key": "qwen36",
            "generation_key": "qwen36|two|1",
            "status": "success",
            "completion_tokens": 300,
            "started_at": "2026-07-29T10:00:02+00:00",
            "completed_at": "2026-07-29T10:00:12+00:00",
            "elapsed_seconds": 10.0,
        },
    ]
    observation = estimator.summarize_pilot_records("qwen36", records)
    assert observation.elapsed_source == "record_timestamps"
    assert observation.elapsed_seconds == 12
    assert observation.successful_generations == 2
    assert observation.completion_tokens == 400
    assert observation.generations_per_second == pytest.approx(1 / 6)
    assert observation.completion_tokens_per_second == pytest.approx(400 / 12)
    assert observation.average_completion_tokens == 200
    assert observation.request_elapsed_seconds == 20
    assert observation.average_request_seconds == 10


def test_one_full_array_pass_plus_prefetch_has_stated_nhr_ceiling():
    qwen_gpu_hours = 4 * 1 * 24
    llama_gpu_hours = 3 * 2 * 24
    prefetch_gpu_hours = 1 * 1 * 12
    assert (qwen_gpu_hours + llama_gpu_hours) / 4 == 60
    assert (qwen_gpu_hours + llama_gpu_hours + prefetch_gpu_hours) / 4 == 63
