from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.elephant_aita_ntaflip.preflight import parse_final_answer_only
from infra.isambard import run_muse_glimmer_rmct_aita_ntaflip_16gpu as muse
from infra.isambard import run_qwen35_rmct_aita_ntaflip_16gpu as audited


def _policy() -> dict[str, object]:
    return {
        "schema": "test-generic-eos-only",
        "termination": "model_eos_only",
        "output_token_cap": None,
    }


def _cap_keys(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "max_tokens",
                "max_new_tokens",
                "max_output_tokens",
                "max_completion_tokens",
                "max_length",
            } and nested is not None:
                found.append(normalized)
            found.extend(_cap_keys(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.extend(_cap_keys(nested))
    return found


def test_decode_contract_has_no_output_token_cap() -> None:
    assert muse.GENERATION_CONFIG == {
        "temperature": 0.6,
        "top_p": 0.9,
        "seed": 0,
        "top_k": 50,
    }
    assert muse.CONCURRENCY_CONFIG == {"max_connections": 4}
    assert not _cap_keys(muse.RUNTIME_GENERATION_CONFIG)


def test_task_command_selects_generic_eos_only_and_no_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "model"
    checkpoint = tmp_path / "checkpoint"
    snapshot.mkdir()
    checkpoint.mkdir()
    monkeypatch.setattr(audited, "MODEL_SNAPSHOT", snapshot)
    monkeypatch.setattr(muse, "_runtime_policy", _policy)
    runtime = {
        "mode": "native-hf-peft",
        "checkpoint": str(checkpoint),
        "model_args": dict(muse.HF_LOCAL_MODEL_ARGS),
        "generation_config": dict(muse.RUNTIME_GENERATION_CONFIG),
        "reasoning_output_policy": dict(muse.REASONING_OUTPUT_POLICY),
        "no_token_cap_policy": dict(muse.NO_TOKEN_CAP_POLICY),
        "no_token_cap_runtime_policy": _policy(),
    }
    command = muse._task_command(
        python="/venv/bin/python",
        manifest=tmp_path / "manifest.json",
        runtime=runtime,
        attempt=tmp_path / "attempt",
        shard_index=2,
    )
    task_args = json.loads(command[command.index("--task-args") + 1])
    generation = json.loads(command[command.index("--generation-config") + 1])
    assert task_args["eos_only_runtime"] == "generic"
    assert task_args["shard_index"] == 2
    assert generation == muse.RUNTIME_GENERATION_CONFIG
    assert not _cap_keys(command)


def test_reasoning_parser_uses_only_post_cot_output() -> None:
    parsed = parse_final_answer_only("<think>NTA appears in reasoning; reconsider YTA.</think>\nNTA")
    assert (parsed.status, parsed.label, parsed.source) == ("parsed", "NTA", "post_think_tail")
    ignored = parse_final_answer_only("NTA appears in reasoning but there is no final-answer boundary")
    assert ignored.status == "missing_boundary"
    malformed = parse_final_answer_only("<think>NTA</think> NTA plus explanation")
    assert malformed.status == "malformed_tail"


def test_output_attestation_requires_eos_and_no_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = SimpleNamespace(
        id="pair::original",
        output=SimpleNamespace(
            completion="<think>reasoning</think>\nYTA",
            metadata={"ctm_no_output_token_cap": True, "ctm_termination": "model_eos_only"},
        ),
    )
    parsed = muse._validate_output_attestation(SimpleNamespace(samples=[sample]), label="test", require_count=1)
    assert parsed[0]["label"] == "YTA"
    sample.output.metadata["ctm_no_output_token_cap"] = False
    with pytest.raises(audited.EvaluationError, match="EOS-only/no-cap"):
        muse._validate_output_attestation(SimpleNamespace(samples=[sample]), label="test", require_count=1)


def test_sbatch_and_worker_use_all_16_gpus_without_token_cap_flags() -> None:
    root = Path(__file__).resolve().parents[1]
    sbatch = (root / "infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu.sbatch").read_text()
    worker = (root / "infra/isambard/run_muse_glimmer_rmct_aita_ntaflip_16gpu_worker.sh").read_text()
    assert "--nodes=4 --ntasks=16 --ntasks-per-node=4" in sbatch
    assert "--gpus=16 --gpus-per-task=1" in sbatch
    assert "--gpu-bind=verbose,per_task:1" in sbatch
    assert "CTM_HF_EOS_ONLY_NO_TOKEN_CAP=1" in sbatch
    assert "CTM_HF_EOS_ONLY_EXPECTED_INSPECT=0.3.260" in sbatch
    assert "CTM_HF_EOS_ONLY_EXPECTED_TRANSFORMERS=5.15.1" in sbatch
    assert "12) condition=final; shard_index=0" in worker
    for forbidden in ("--max-tokens", "--max_tokens", "--max-new-tokens", "--max_new_tokens"):
        assert forbidden not in sbatch
        assert forbidden not in worker


def test_aita_condition_labels_preserve_global_and_realized_optimizer_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(muse, "_snapshot_path", lambda: Path("/tmp/muse-snapshot"))
    monkeypatch.setattr(muse, "_runtime_policy", _policy)
    muse._configure()
    conditions = {condition.name: condition for condition in audited.CONDITIONS}
    assert conditions["step016"].label == "global/data-step-016"
    assert conditions["step016"].optimizer_step is None
    assert conditions["step064"].label == "global/data-step-064"
    assert conditions["step064"].optimizer_step is None
