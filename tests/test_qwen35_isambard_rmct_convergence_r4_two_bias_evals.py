"""Focused safety tests for the terminal r4 two-bias evaluation entrypoint."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.py"
SBATCH = ROOT / "infra/isambard/run_qwen35_rmct_convergence_r4_two_bias_evals.sbatch"


def _load_launcher_module():
    spec = importlib.util.spec_from_file_location("r4_two_bias_eval_launcher_for_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_entrypoints_are_executable_syntax_valid_and_never_self_submit_or_chain():
    assert LAUNCHER.stat().st_mode & 0o111
    assert SBATCH.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(SBATCH)], check=False).returncode == 0
    result = subprocess.run([sys.executable, str(LAUNCHER), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    launcher = LAUNCHER.read_text(encoding="utf-8")
    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "sbatch " not in launcher
    assert "sbatch " not in sbatch
    assert "afterok" not in sbatch
    assert "--dependency" not in sbatch
    assert "#SBATCH --partition=workq" in sbatch
    assert "CTM_RMCT_EVAL_SCHEDULER_MODE" in sbatch
    assert "CTM_RMCT_EVAL_PYTHON" in sbatch
    assert "sys.prefix == sys.base_prefix" in sbatch
    assert "readlink -f" not in sbatch
    assert "CTM_RMCT_TRAINING_REPOSITORY" in sbatch
    assert "CTM_RMCT_EVAL_PARITY_MANIFEST" in sbatch
    assert "CTM_RMCT_EVAL_PARITY_DATA" in sbatch
    assert "two-bias-v1-r005" in sbatch
    assert 'export PATH="$python_directory:$PATH"' in sbatch
    assert "command -v vllm" in sbatch
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in sbatch
    assert 'vllm.__version__ != "0.21.0"' in sbatch
    assert 'gpu_tokens="$("$python_bin" - <<PY' in sbatch
    assert '--gpus "$gpu_tokens"' in sbatch
    assert "--gpus 0,1,2,3" not in sbatch
    assert "workq|interactive" in sbatch
    assert "--gpus=4" in sbatch
    assert "--yes" in sbatch


def test_pinned_contract_is_vllm_only_and_preserves_the_full_two_bias_matrix():
    module = _load_launcher_module()
    assert str(module.MODEL_SNAPSHOT) == (
        "/lus/lfs1aip2/scratch/a5v/sohaib.a5v/ctm/huggingface/hub/"
        "models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    )
    assert module.FINAL_OPTIMIZER_STEP == 176
    assert module.FINAL_SEGMENT_INDEX == 10
    assert module.STAGE1_PARITY_MANIFEST_SHA256 == "8cdd4da0575a125b01b2b9b62d9c0e07176142eec6194e72e8abd04401ca2fec"
    assert module.STAGE1_PARITY_TRAIN_EVAL_FILENAME == "train-eval-n200.jsonl"
    assert module.STAGE2_TASK_FACTORY == "experiments.stage2_ood_hle.tasks:ood_tasks"
    assert module.TASK_COUNT == 21
    assert module.CLEAN_TASK_INDICES == (1, 2, 3)
    assert module.BIASED_TASK_INDICES == tuple(range(4, 22))
    assert module.SEEN_BIASES == ("wrong_argument", "suggested_answer")
    assert module.HELD_OUT_BIASES == (
        "distractor_fact",
        "post_hoc",
        "spurious_few_shot_squares",
        "wrong_few_shot",
    )
    assert module.VLLM_MODEL_ARGS["provider"] == "vllm"
    assert module.GENERATION_CONFIG == {
        "max_tokens": 20480,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "extra_body": {"top_k": 20},
    }
    assert module.VLLM_VERSION == "0.21.0"
    assert module.VLLM_MODEL_ARGS["gdn_prefill_backend"] == "triton"
    assert module._vllm_sampler_runtime() == {
        "vllm_version": "0.21.0",
        "environment": {"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "implementation": "pytorch_native",
        "parity_gdn_prefill_backend": "triton",
        "parity_top_token_count": 16,
        "parity_max_logprobs": 29,
        "parity_logprobs_mode": "processed_logprobs",
        "parity_score_transport": "allowed_token_ids_restricted_softmax",
        "parity_sampling": {
            "max_tokens": 1,
            "min_tokens": 0,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "ignore_eos": True,
        },
        "persistent_gdn_prefill_backend": "triton",
        "isolate_vllm_variants": True,
        "parallel_isolated_vllm_variants": True,
        "enforce_eager": True,
        "vllm_device_tokens": ["0", "1", "2", "3"],
        "vllm_device_count": 4,
        "one_adapter_per_server": True,
        "max_loras_per_server": 1,
        "parity_result_variants": ["full", "linear_only", "self_attn_only", "evaluator_path"],
    }

    text = LAUNCHER.read_text(encoding="utf-8")
    assert "attest_compat_adapter" in text
    assert "runtime_parity" in text
    assert "--gdn-prefill-backend" in text
    assert "--parallel-isolated-vllm-variants" in text
    assert "--vllm-device-tokens" in text
    assert "validate_r005_parallel_parity_report" in text
    assert "--max-logprobs" in text
    assert "--parity-data" in text
    assert "--parity-manifest" in text
    assert "r4.DATA_PATH" not in text
    assert "--persistent-vllm-server" in text
    assert "--isolate-tasks" in text
    assert "hf_fallback_permitted\": False" in text
    clean_index = text.index('phase="clean"')
    barrier_index = text.index("_require_clean_barrier(\n        paths", clean_index)
    biased_index = text.index('phase="biased"')
    assert clean_index < barrier_index < biased_index


def test_model_snapshot_allows_only_huggingface_style_config_symlink_within_its_blob_tree(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    cache = tmp_path / "hub" / "models--Qwen--Qwen3.5-9B"
    snapshot = cache / "snapshots" / "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    blob = cache / "blobs" / "config-blob"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir(parents=True)
    blob.write_text('{"architectures": ["Qwen3_5ForConditionalGeneration"]}\n', encoding="utf-8")
    logical = snapshot / "config.json"
    logical.symlink_to("../../blobs/config-blob")
    monkeypatch.setattr(module, "MODEL_SNAPSHOT", snapshot)

    record = module._validate_model_snapshot()
    assert record["path"] == str(snapshot)
    assert record["config"] == {
        "logical_path": str(logical),
        "resolved_path": str(blob),
        "sha256": module._sha256_file(blob),
        "size_bytes": blob.stat().st_size,
    }
    with pytest.raises(module.EvaluationError, match="must be a regular file"):
        module._identity(logical, label="ordinary artifact")

    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    logical.unlink()
    logical.symlink_to(outside)
    with pytest.raises(module.EvaluationError, match="same Hugging Face model blob tree"):
        module._validate_model_snapshot()


def test_vllm_021_native_sampler_runtime_is_immutable_and_fails_closed(monkeypatch):
    module = _load_launcher_module()
    launch = {
        "runtime": {
            "sampler": module._vllm_sampler_runtime(),
            "model_args": dict(module.VLLM_MODEL_ARGS),
        }
    }
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.21.0"))
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")

    assert module._validate_vllm_sampler_runtime(launch) == module._vllm_sampler_runtime()

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    with pytest.raises(module.EvaluationError, match="VLLM_USE_FLASHINFER_SAMPLER must be exactly '0'"):
        module._validate_vllm_sampler_runtime(launch)

    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.20.0"))
    with pytest.raises(module.EvaluationError, match="requires vLLM 0.21.0"):
        module._validate_vllm_sampler_runtime(launch)

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(__version__="0.21.0"))
    launch["runtime"]["model_args"] = {**module.VLLM_MODEL_ARGS, "gdn_prefill_backend": "flashinfer"}
    with pytest.raises(module.EvaluationError, match="launch contract does not bind"):
        module._validate_vllm_sampler_runtime(launch)


def test_parity_report_must_record_the_pinned_r005_processed_logprob_protocol():
    module = _load_launcher_module()
    report = {
        "token_protocol": {
            "requested_result_variants": list(module.PARITY_RESULT_VARIANTS),
            "top_token_count": module.PARITY_TOP_TOKEN_COUNT,
            "vllm_score_transport": "allowed_token_ids_restricted_softmax",
            "vllm_allowed_token_ids": "requested_token_ids",
            "vllm_response_token_ids": "exactly_requested_token_ids",
        },
        "backends": {
            "vllm": {
                "gdn_prefill_backend": "triton",
                "max_logprobs": 29,
                "logprobs_mode": "processed_logprobs",
                "score_transport": "allowed_token_ids_restricted_softmax",
                "parity_sampling": {
                    "max_tokens": 1,
                    "min_tokens": 0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                    "min_p": 0.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "ignore_eos": True,
                },
                "isolate_vllm_variants": True,
                "parallel_isolated_vllm_variants": True,
                "enforce_eager": True,
                "parallel_isolated_server_plan": {
                    variant: {
                        "device_token": str(index),
                        "max_logprobs": 29,
                        "logprobs_mode": "processed_logprobs",
                    }
                    for index, variant in enumerate(module.PARITY_RESULT_VARIANTS)
                },
            }
        },
    }
    module._validate_parity_report_runtime(report, module._vllm_sampler_runtime())

    report["token_protocol"]["top_token_count"] = module.PARITY_TOP_TOKEN_COUNT + 1
    with pytest.raises(module.EvaluationError, match="canonical r005 allowed-token four-server topology"):
        module._validate_parity_report_runtime(report, module._vllm_sampler_runtime())

    report["token_protocol"]["top_token_count"] = module.PARITY_TOP_TOKEN_COUNT
    report["backends"]["vllm"]["gdn_prefill_backend"] = None
    with pytest.raises(module.EvaluationError, match="eager, isolated triton"):
        module._validate_parity_report_runtime(report, module._vllm_sampler_runtime())

    report["backends"]["vllm"]["gdn_prefill_backend"] = "triton"
    report["backends"]["vllm"]["score_transport"] = "top_logprobs"
    with pytest.raises(module.EvaluationError, match="processed-logprob runtime"):
        module._validate_parity_report_runtime(report, module._vllm_sampler_runtime())

    report["backends"]["vllm"]["score_transport"] = "allowed_token_ids_restricted_softmax"
    report["token_protocol"]["vllm_allowed_token_ids"] = "top_logprobs"
    with pytest.raises(module.EvaluationError, match="allowed-token four-server topology"):
        module._validate_parity_report_runtime(report, module._vllm_sampler_runtime())


def test_gpu_tokens_preserve_slurm_uuids_and_are_bound_into_the_parallel_sampler():
    module = _load_launcher_module()
    tokens = module._parse_gpus("GPU-a,GPU-b,GPU-c,GPU-d")
    assert tokens == ("GPU-a", "GPU-b", "GPU-c", "GPU-d")
    sampler = module._vllm_sampler_runtime(vllm_device_tokens=tokens)
    assert sampler["vllm_device_tokens"] == list(tokens)
    assert sampler["parallel_isolated_vllm_variants"] is True
    assert sampler["one_adapter_per_server"] is True
    with pytest.raises(module.EvaluationError, match="four distinct"):
        module._parse_gpus("GPU-a,GPU-a,GPU-c,GPU-d")


def test_parallel_launch_refuses_to_remap_the_inherited_slurm_cuda_tokens(monkeypatch):
    module = _load_launcher_module()
    tokens = ("GPU-a", "GPU-b", "GPU-c", "GPU-d")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(tokens))
    assert module._validate_inherited_cuda_tokens(tokens) == tokens
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b,GPU-a,GPU-c,GPU-d")
    with pytest.raises(module.EvaluationError, match="differ from inherited"):
        module._validate_inherited_cuda_tokens(tokens)


def test_parallel_parity_command_uses_all_immutable_tokens_without_parent_cvd_remapping(tmp_path: Path):
    module = _load_launcher_module()
    sampler = module._vllm_sampler_runtime(vllm_device_tokens=("GPU-a", "GPU-b", "GPU-c", "GPU-d"))
    command = module._parity_command(
        python="/unit/python",
        selected=tmp_path / "compat",
        raw={"path": str(tmp_path / "raw")},
        launch={"parity_data": {"path": str(tmp_path / "parity.jsonl")}},
        attempt=tmp_path / "attempt",
        sampler=sampler,
    )
    assert "--enforce-eager" in command
    assert "--isolate-vllm-variants" in command
    assert "--parallel-isolated-vllm-variants" in command
    token_index = command.index("--vllm-device-tokens")
    assert command[token_index + 1 : token_index + 5] == ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
    assert _option_value(command, "--top-token-count") == "16"
    assert _option_value(command, "--max-logprobs") == "29"


def test_parity_data_binds_a_staged_stage1_train_eval_file_without_opening_the_stale_manifest_path(
    monkeypatch, tmp_path: Path
):
    module = _load_launcher_module()
    data = tmp_path / module.STAGE1_PARITY_TRAIN_EVAL_FILENAME
    payload = b'{"biased_messages": [{"role": "user", "content": "x"}]}\n'
    data.write_bytes(payload)
    manifest = tmp_path / "manifest.json"
    stale_path = "/Users/work/obsolete/stage1/train-eval-n200.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": module.STAGE1_PARITY_MANIFEST_SCHEMA_VERSION,
                "kind": module.STAGE1_PARITY_MANIFEST_KIND,
                "splits": {
                    "train_eval": {
                        "path": stale_path,
                        "content_sha256": module._sha256_file(data),
                        "byte_count": len(payload),
                        "row_count": module.STAGE1_PARITY_TRAIN_EVAL_ROWS,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "STAGE1_PARITY_MANIFEST_SHA256", module._sha256_file(manifest))

    record = module._parity_data(data, manifest)

    assert record["path"] == str(data)
    assert record["canonical_manifest"]["sha256"] == module._sha256_file(manifest)
    assert record["manifest_train_eval"] == {
        "declared_path": stale_path,
        "filename": module.STAGE1_PARITY_TRAIN_EVAL_FILENAME,
        "content_sha256": module._sha256_file(data),
        "byte_count": len(payload),
        "row_count": module.STAGE1_PARITY_TRAIN_EVAL_ROWS,
    }


def test_parity_data_rejects_a_deployed_file_that_does_not_match_the_manifest_entry(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    data = tmp_path / module.STAGE1_PARITY_TRAIN_EVAL_FILENAME
    data.write_bytes(b"one\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": module.STAGE1_PARITY_MANIFEST_SCHEMA_VERSION,
                "kind": module.STAGE1_PARITY_MANIFEST_KIND,
                "splits": {
                    "train_eval": {
                        "path": "/stale/train-eval-n200.jsonl",
                        "content_sha256": "a" * 64,
                        "byte_count": 4,
                        "row_count": module.STAGE1_PARITY_TRAIN_EVAL_ROWS,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "STAGE1_PARITY_MANIFEST_SHA256", module._sha256_file(manifest))

    with pytest.raises(module.EvaluationError, match="SHA-256 differs"):
        module._parity_data(data, manifest)


def test_task_command_is_one_persistent_vllm_server_with_the_exact_raw_task_factory(tmp_path: Path):
    module = _load_launcher_module()
    paths = module._launch_paths(tmp_path / "out")
    manifest = tmp_path / "deployed-stage2.json"
    adapter = tmp_path / "compat-adapter"
    command = module._task_command(
        launch={"deployment_manifest": {"path": str(manifest)}},
        runtime={"compatibility_adapter": {"path": str(adapter)}},
        paths=paths,
        attempt=tmp_path / "attempt",
        task_indices=(4, 8, 12),
        python="/unit/python",
    )

    assert command[:2] == ["/unit/python", str(ROOT / "scripts/run_evals.py")]
    assert _option_value(command, "--task-factory") == module.STAGE2_TASK_FACTORY
    assert _option_value(command, "--local-checkpoint") == str(adapter)
    assert _option_value(command, "--base-model") == str(module.MODEL_SNAPSHOT)
    assert json.loads(_option_value(command, "--model-args")) == module.VLLM_MODEL_ARGS
    assert json.loads(_option_value(command, "--generation-config")) == module.GENERATION_CONFIG
    assert json.loads(_option_value(command, "--task-args")) == {
        "manifest": str(manifest),
        "unbiased_log": str(paths.raw),
        "prompt_style": "none",
        "include_bias_acknowledged": False,
    }
    assert "--isolate-tasks" in command
    assert "--persistent-vllm-server" in command
    assert command.count("--task-index") == 3
    assert [command[index + 1] for index, value in enumerate(command) if value == "--task-index"] == ["4", "8", "12"]


def test_write_once_receipts_refuse_differing_replays(tmp_path: Path):
    module = _load_launcher_module()
    path = tmp_path / "receipt.json"
    assert module._write_immutable_json(path, {"value": 1}, label="test") == "written"
    assert module._write_immutable_json(path, {"value": 1}, label="test") == "resumed"
    with pytest.raises(FileExistsError, match="refusing to overwrite differing"):
        module._write_immutable_json(path, {"value": 2}, label="test")


def test_clean_barrier_refuses_to_release_biased_tasks_when_any_clean_receipt_is_missing(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    paths = module._launch_paths(tmp_path / "out")

    def fake_load(_paths, *, task_index, launch_contract_sha256, evaluation_receipt_sha256):
        assert launch_contract_sha256 == "a" * 64
        assert evaluation_receipt_sha256 == "b" * 64
        return None if task_index == 2 else {"task_index": task_index}

    monkeypatch.setattr(module, "_load_task_receipt", fake_load)
    with pytest.raises(module.EvaluationError, match=r"missing=\[2\]"):
        module._require_clean_barrier(
            paths,
            launch_contract_sha256="a" * 64,
            evaluation_receipt_sha256="b" * 64,
        )


def test_bad_task_index_cannot_be_sent_to_the_frozen_task_factory(tmp_path: Path):
    module = _load_launcher_module()
    with pytest.raises(module.EvaluationError, match="invalid Stage-2 task index"):
        module._task_command(
            launch={"deployment_manifest": {"path": str(tmp_path / "manifest")}},
            runtime={"compatibility_adapter": {"path": str(tmp_path / "adapter")}},
            paths=module._launch_paths(tmp_path / "out"),
            attempt=tmp_path / "attempt",
            task_indices=(0,),
            python=sys.executable,
        )


def test_deployment_write_status_is_not_part_of_the_immutable_launch_contract(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    checkpoint = {
        "path": str(tmp_path / "checkpoint"),
        "decision_receipt": {"path": str(tmp_path / "decision.json")},
    }
    monkeypatch.setattr(module, "validate_final_checkpoint", lambda _repository: checkpoint)
    monkeypatch.setattr(module, "_validate_model_snapshot", lambda: {"path": str(module.MODEL_SNAPSHOT)})
    monkeypatch.setattr(module, "validate_two_bias_substrate", lambda path: {"path": str(path)})
    monkeypatch.setattr(module, "_parity_data", lambda *_args, **_kwargs: {"path": str(tmp_path / "parity.jsonl")})
    calls = iter(("written", "resumed"))

    def fake_materialize(_source, *, artifact_root, output):
        return {
            "status": next(calls),
            "manifest_path": str(Path(output).resolve()),
            "manifest_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "artifact_root": str(Path(artifact_root).resolve()),
            "artifacts": {},
        }

    monkeypatch.setattr(module, "materialize_deployment_manifest", fake_materialize)
    (tmp_path / "source.json").write_text("{}\n", encoding="utf-8")
    values = {
        "repository": tmp_path,
        "source_stage2_manifest": tmp_path / "source.json",
        "stage2_artifact_root": tmp_path / "artifacts",
        "parity_data": tmp_path / "train-eval-n200.jsonl",
        "parity_manifest": tmp_path / "parity-manifest.json",
        "output_root": tmp_path / "output",
        "gpus": ("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
    }
    first = module.build_launch_contract(**values)
    second = module.build_launch_contract(**values)
    assert first == second
    assert "status" not in first["deployment_manifest"]["provenance"]
    assert first["runtime"]["sampler"]["vllm_device_tokens"] == ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]


def test_critical_source_hash_drift_refuses_to_rewrite_an_existing_launch_contract(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    source_root = tmp_path / "source"
    source_root.mkdir()
    critical = source_root / "critical.py"
    critical.write_text("first\n", encoding="utf-8")
    monkeypatch.setattr(module, "PROJECT_ROOT", source_root)
    monkeypatch.setattr(module, "CRITICAL_SOURCES", ("critical.py",))
    monkeypatch.setattr(module, "validate_final_checkpoint", lambda _repository: {"path": str(tmp_path / "checkpoint")})
    monkeypatch.setattr(module, "_validate_model_snapshot", lambda: {"path": str(module.MODEL_SNAPSHOT)})
    monkeypatch.setattr(module, "validate_two_bias_substrate", lambda path: {"path": str(path)})
    monkeypatch.setattr(module, "_parity_data", lambda *_args, **_kwargs: {"path": str(tmp_path / "parity.jsonl")})
    monkeypatch.setattr(
        module,
        "materialize_deployment_manifest",
        lambda _source, *, artifact_root, output: {
            "manifest_path": str(Path(output).resolve()),
            "manifest_sha256": "a" * 64,
            "source_manifest_sha256": "b" * 64,
            "artifact_root": str(Path(artifact_root).resolve()),
            "artifacts": {},
        },
    )
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    values = {
        "repository": tmp_path,
        "source_stage2_manifest": source_manifest,
        "stage2_artifact_root": tmp_path / "artifacts",
        "parity_data": tmp_path / "train-eval-n200.jsonl",
        "parity_manifest": tmp_path / "parity-manifest.json",
        "output_root": tmp_path / "output",
        "gpus": ("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
    }
    module.prepare_launch_contract(**values)
    critical.write_text("second\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite differing"):
        module.prepare_launch_contract(**values)


def test_prior_attempt_salvage_is_scoped_to_its_own_gpu_group(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    paths = module._launch_paths(tmp_path / "out")
    phase_root = paths.attempts / "biased"
    own = phase_root / "gpu0-tasks-4-8-0001"
    other = phase_root / "gpu1-tasks-5-9-0001"
    own.mkdir(parents=True)
    other.mkdir()
    observed: list[Path] = []

    def fake_promote(*, attempt, **_kwargs):
        observed.append(attempt)
        return [4]

    monkeypatch.setattr(module, "_promote_attempt", fake_promote)
    assert module._promote_prior_attempts(
        phase_root=phase_root,
        attempt_label="gpu0-tasks-4-8",
        task_indices=(4, 8),
        paths=paths,
        launch_contract_sha256="a" * 64,
        evaluation_receipt_sha256="b" * 64,
    ) == [4]
    assert observed == [own]


def test_same_group_retry_scans_the_full_group_when_one_cell_is_already_receipted(monkeypatch, tmp_path: Path):
    module = _load_launcher_module()
    paths = module._launch_paths(tmp_path / "out")
    present = {4}
    observed: list[tuple[int, ...]] = []

    def fake_load(_paths, *, task_index, **_kwargs):
        return {"task_index": task_index} if task_index in present else None

    def fake_prior(*, task_indices, **_kwargs):
        observed.append(tuple(task_indices))
        present.add(8)
        return [8]

    monkeypatch.setattr(module, "_load_task_receipt", fake_load)
    monkeypatch.setattr(module, "_promote_prior_attempts", fake_prior)
    result = module._run_group(
        gpu="0",
        task_indices=(4, 8),
        launch={"deployment_manifest": {"path": str(tmp_path / "manifest")}},
        runtime={
            "compatibility_adapter": {"path": str(tmp_path / "adapter")},
            "sampler": module._vllm_sampler_runtime(),
        },
        paths=paths,
        python=sys.executable,
        launch_contract_sha256="a" * 64,
        evaluation_receipt_sha256="b" * 64,
        phase="biased",
    )
    assert observed == [(4, 8)]
    assert result["status"] == "resumed"


def test_completed_replay_reuses_existing_completion_without_rewriting_a_timestamp(tmp_path: Path):
    module = _load_launcher_module()
    paths = module._launch_paths(tmp_path / "out")
    paths.completion.parent.mkdir(parents=True)
    receipt = {"path": str(paths.evaluation_receipt), "sha256": "b" * 64, "status": "resumed"}
    completion = {
        "schema": module.SCHEMA,
        "contract": {"path": str(paths.contract), "sha256": "a" * 64},
        "evaluation_receipt": receipt,
        "completed_at": "2026-08-20T00:00:00+00:00",
        "runtime": {},
        "clean": [],
        "biased": [],
        "preflight": {"status": "resumed", "path": str(paths.root / "stage2" / "preflight" / f"{module.CONDITION}.json")},
    }
    paths.completion.write_text(json.dumps(completion), encoding="utf-8")
    assert module._completed_replay(
        paths=paths,
        launch_contract_sha256="a" * 64,
        evaluation_receipt=receipt,
    ) == completion
