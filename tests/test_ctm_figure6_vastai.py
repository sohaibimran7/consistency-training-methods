"""Offline contract tests for the two-host Vast.ai Figure 6 launch stack."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
VAST = ROOT / "infra" / "vastai"
MODELS = ROOT / "experiments" / "eval_awareness" / "figure6" / "models.yaml"


def _plan_module():
    path = VAST / "figure6_plan.py"
    spec = importlib.util.spec_from_file_location("figure6_vast_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_vast_plan_is_the_exact_disjoint_two_host_mapping():
    plan = _plan_module()
    observed = {
        host: [(assignment.model_key, assignment.gpu_ids, assignment.port) for assignment in assignments]
        for host, assignments in plan.HOST_PLANS.items()
    }
    assert observed == {
        "A": [
            ("qwen36", (0,), 8100),
            ("qwen32", (1,), 8101),
            ("qwen_mo_mid", (2,), 8102),
            ("qwen_mo_post", (3,), 8103),
            ("llama33", (4, 5), 8104),
            ("llama_mo_mid", (6, 7), 8105),
        ],
        "B": [("llama_mo_post", (0, 1), 8106)],
    }
    assert plan.HOST_GPU_COUNTS == {"A": 8, "B": 2}
    models = plan.load_models(MODELS)
    plan.validate_global_plan(models)
    for host, assignments in plan.HOST_PLANS.items():
        plan.validate_plan(host, assignments, models, detected_gpu_count=plan.HOST_GPU_COUNTS[host])

    all_assignments = [assignment for assignments in plan.HOST_PLANS.values() for assignment in assignments]
    assert len({assignment.model_key for assignment in all_assignments}) == 7
    assert len({assignment.port for assignment in all_assignments}) == 7


def test_vast_plan_fails_closed_on_gpu_count_assignment_and_tp_drift():
    plan = _plan_module()
    models = plan.load_models(MODELS)
    with pytest.raises(ValueError, match="requires exactly 8 visible GPUs"):
        plan.validate_plan("A", plan.HOST_PLANS["A"], models, detected_gpu_count=7)

    changed_assignment = plan.Assignment("qwen32", (0,), 8101)
    overlapping = (plan.HOST_PLANS["A"][0], changed_assignment, *plan.HOST_PLANS["A"][2:])
    with pytest.raises(ValueError, match="fixed Figure 6 plan"):
        plan.validate_plan("A", overlapping, models, detected_gpu_count=8)

    drifted_models = {key: dict(value) for key, value in models.items()}
    drifted_models["llama33"]["tensor_parallel_size"] = 1
    with pytest.raises(ValueError, match="llama33 requires TP=1; plan assigns 2 GPUs"):
        plan.validate_plan("A", plan.HOST_PLANS["A"], drifted_models, detected_gpu_count=8)


def test_vast_plan_cli_prints_only_validated_host_rows():
    completed = subprocess.run(
        [
            sys.executable,
            str(VAST / "figure6_plan.py"),
            "--models",
            str(MODELS),
            "--host",
            "B",
            "--detected-gpu-count",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout == "llama_mo_post\t0,1\t8106\n"
    assert completed.stderr == ""


@pytest.mark.parametrize(
    "name",
    ["provision_figure6.sh", "run_figure6_model.sh", "launch_figure6_host.sh"],
)
def test_vast_shell_scripts_parse(name: str):
    subprocess.run(["bash", "-n", str(VAST / name)], check=True)


def test_vast_provision_preserves_the_exact_image_runtime_and_is_minimal():
    provision = (VAST / "provision_figure6.sh").read_text(encoding="utf-8")
    requirements = (VAST / "figure6-requirements.txt").read_text(encoding="utf-8")
    assert "--system-site-packages" in provision
    assert '--no-deps -e "$REPO_DIR"' in provision
    assert 'vllm.__version__ == "0.26.0"' in provision
    assert "torch.cuda.is_available()" in provision
    assert "torch.cuda.is_bf16_supported()" in provision
    assert "python -m pip check" in provision
    assert '"$REPO_DIR/requirements.txt"' not in provision
    assert "pip install vllm" not in provision
    assert "pip install torch" not in provision
    installed = {
        line.split("==", 1)[0].lower() for line in requirements.splitlines() if line and not line.startswith("#")
    }
    assert installed == {"openai", "pyyaml"}


def test_vast_runner_uses_pins_local_endpoint_and_append_only_resume():
    runner = (VAST / "run_figure6_model.sh").read_text(encoding="utf-8")
    assert 'MODELS_CONFIG="$REPO_DIR/experiments/eval_awareness/figure6/models.yaml"' in runner
    assert '--revision "$MODEL_REVISION"' in runner
    assert '--served-model-name "$MODEL_ID"' in runner
    assert '--dtype "$MODEL_DTYPE"' in runner
    assert '--tensor-parallel-size "$TENSOR_PARALLEL_SIZE"' in runner
    assert "--max-model-len 8192" in runner
    assert "VLLM_ARGS+=(--reasoning-parser qwen3)" in runner
    assert "VLLM_ARGS+=(--language-model-only)" in runner
    assert 'vllm "${VLLM_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &' in runner
    assert '--output "$GENERATION_LOG"' in runner
    assert "--api-key-env FIGURE6_LOCAL_ENDPOINT_TOKEN" in runner
    assert "FIGURE6_LOCAL_ENDPOINT_TOKEN=vast-local-vllm-dummy" in runner
    assert '--limit-conditions "$FIGURE6_LIMIT_CONDITIONS"' in runner
    assert "--replicates 3" in runner
    assert "--temperature 0.3" in runner
    assert "--max-tokens 4096" in runner
    assert "PROJECTDIR" not in runner
    assert "SCRATCHDIR" not in runner
    assert "SLURM" not in runner
    assert "figure6_judge" not in runner


def test_vast_launcher_sets_canary_unsets_full_and_records_status():
    launcher = (VAST / "launch_figure6_host.sh").read_text(encoding="utf-8")
    assert "export FIGURE6_LIMIT_CONDITIONS=100" in launcher
    assert "unset FIGURE6_LIMIT_CONDITIONS" in launcher
    assert 'python "$SCRIPT_DIR/figure6_plan.py"' in launcher
    assert '"$RUNNER" "$model_key" "$gpu_list" "$port" >> "$runner_log" 2>&1 &' in launcher
    assert 'STATUS_LOG="$FIGURE6_OUTPUT_ROOT/_status/host-${HOST}.jsonl"' in launcher
    assert 'with open(path, "a", encoding="utf-8")' in launcher
    assert "flock -n 9" in launcher
    assert "figure6_judge" not in launcher


def test_vast_runner_rejects_openai_secret_without_disclosing_it():
    secret = "must-not-appear-in-output"
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = secret
    completed = subprocess.run(
        ["bash", str(VAST / "run_figure6_model.sh"), "qwen36", "0", "8100"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    output = completed.stdout + completed.stderr
    assert "OPENAI_API_KEY must not be present" in output
    assert secret not in output


def test_vast_docs_pin_image_and_document_both_resume_phases():
    docs = (VAST / "FIGURE6.md").read_text(encoding="utf-8")
    assert "vllm/vllm-openai:v0.26.0" in docs
    assert docs.count("--ssh --direct") == 2
    assert "--entrypoint" not in docs
    assert "sleep infinity" not in docs
    assert "FIGURE6_LIMIT_CONDITIONS=100 bash infra/vastai/launch_figure6_host.sh A canary" in docs
    assert "bash infra/vastai/launch_figure6_host.sh A full" in docs
    assert "bash infra/vastai/launch_figure6_host.sh B full" in docs
    assert "HF_TOKEN_FILE" in docs
    assert "permissions exactly 600" in docs
    assert "Do not configure\n`OPENAI_API_KEY`" in docs
    assert "never start the\npaid external judge" in docs


def test_plan_tensor_parallelism_matches_pinned_registry():
    plan = _plan_module()
    registry = yaml.safe_load(MODELS.read_text(encoding="utf-8"))["models"]
    for assignments in plan.HOST_PLANS.values():
        for assignment in assignments:
            assert len(assignment.gpu_ids) == registry[assignment.model_key]["tensor_parallel_size"]
