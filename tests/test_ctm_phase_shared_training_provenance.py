"""CPU-only provenance contracts for phase-shared RL and OPCT entry points."""

from __future__ import annotations

import json

import pytest


def _phase_shared_args(status_dir, *, train_gpus: str | None = None, rollout_gpus: str | None = None) -> list[str]:
    args = [
        "--backend",
        "local",
        "--local-phase-shared",
        "--local-rollout-status-dir",
        str(status_dir),
        "--local-rollout-start-timeout-seconds",
        "11",
        "--local-rollout-request-timeout-seconds",
        "22",
        "--local-replica-start-timeout-seconds",
        "33",
        "--local-replica-command-timeout-seconds",
        "44",
        "--local-replica-shutdown-timeout-seconds",
        "55",
    ]
    if train_gpus is not None:
        args.extend(["--local-training-gpus", train_gpus])
    if rollout_gpus is not None:
        args.extend(["--local-rollout-gpus", rollout_gpus])
    return args


def _assert_common_phase_shared_provenance(metadata: dict, *, world_size: int) -> None:
    phase = metadata["phase_shared"]
    assert phase["schema_version"] == "local_phase_shared_v1"
    assert phase["execution_only"] is True
    assert phase["training_world_size"] == world_size
    assert phase["vllm_sleep_lifecycle"] == {
        "enabled": True,
        "sleep_level": 1,
        "rollout_phase": "workers awake; sampling and scoring permitted",
        "training_phase": "workers sleep before replicated trainer work; sampling and scoring prohibited",
        "publication": "rank 0 verifies replica state, publishes the adapter, then all workers acknowledge before rollout resumes",
        "transition_failure_policy": "fail_closed",
    }
    assert phase["rollout_worker_timeouts_seconds"] == {"startup": 11.0, "request": 22.0}
    assert phase["replica_timeouts_seconds"] == {"startup": 33.0, "command": 44.0, "shutdown": 55.0}


@pytest.mark.parametrize("gpu_count", [2, 4, 8])
def test_rlct_dry_run_records_default_phase_shared_topology_for_any_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
    gpu_count: int,
):
    import scripts.train_rlct as script

    real_config = script.RLConfig
    captured = {}

    def capture_config(**kwargs):
        config = real_config(**kwargs)
        captured["config"] = config
        return config

    monkeypatch.setattr(script, "RLConfig", capture_config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(f"GPU-{index}" for index in range(gpu_count)))

    script.main(
        [
            "--experiment-name",
            "phase",
            "--run-name",
            f"rlct-{gpu_count}",
            "--setting-factory",
            "tests.test_ctm_setting_runtime:create_unit_setting",
            "--n-datapoints",
            "1",
            "--seed",
            "7",
            "--lora-config",
            '{"dropout": 0}',
            *_phase_shared_args(tmp_path / f"rlct-{gpu_count}"),
            "--dry-run",
        ]
    )

    metadata = captured["config"].run_metadata
    _assert_common_phase_shared_provenance(metadata, world_size=gpu_count)
    phase = metadata["phase_shared"]
    assert phase["visible_devices"] == [f"GPU-{index}" for index in range(gpu_count)]
    assert [(rank["rank"], rank["logical_index"], rank["device_token"]) for rank in phase["training_ranks"]] == [
        (index, index, f"GPU-{index}") for index in range(gpu_count)
    ]
    assert [worker["logical_index"] for worker in phase["rollout_workers"]] == list(range(gpu_count))
    assert [gpu["logical_index"] for gpu in phase["overlap"]] == list(range(gpu_count))
    assert phase["coordinator"] == {
        "rank": 0,
        "logical_index": 0,
        "device_token": "GPU-0",
        "device": "cuda:0",
        "canonical_adapter_publisher": True,
    }

    output = capsys.readouterr().out
    assert f"Phase topology:    world={gpu_count}; rank 0/coordinator=cuda:0" in output
    assert "execution-only; configured objective/data/budgets/hyperparameters unchanged" in output
    assert "level-1 sleep for training; rank-0 publish + worker ACK" in output
    assert "Replica timeouts:  start=33s, command=44s, shutdown=55s" in output


def test_opct_dry_run_records_ordered_nondefault_phase_shared_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
):
    import scripts.train_opct as script

    data = tmp_path / "pairs.jsonl"
    data.write_text(
        json.dumps(
            {
                "reference_messages": [{"role": "user", "content": "clean"}],
                "variant_messages": [{"role": "user", "content": "variant"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    real_config = script.OPCTConfig
    captured = {}

    def capture_config(**kwargs):
        config = real_config(**kwargs)
        captured["config"] = config
        return config

    monkeypatch.setattr(script, "OPCTConfig", capture_config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")

    script.main(
        [
            "--model",
            "unit/model",
            "--data",
            str(data),
            "--experiment-name",
            "phase",
            "--run-name",
            "opct",
            "--seed",
            "7",
            "--lora-config",
            '{"dropout": 0}',
            *_phase_shared_args(
                tmp_path / "opct",
                train_gpus="3,1",
                rollout_gpus="1,3",
            ),
            "--dry-run",
        ]
    )

    metadata = captured["config"].run_metadata
    _assert_common_phase_shared_provenance(metadata, world_size=2)
    phase = metadata["phase_shared"]
    assert phase["visible_devices"] == ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
    assert phase["training_ranks"] == [
        {"rank": 0, "logical_index": 3, "device_token": "GPU-d", "publisher": True},
        {"rank": 1, "logical_index": 1, "device_token": "GPU-b", "publisher": False},
    ]
    assert phase["rollout_workers"] == [
        {"worker_id": 0, "logical_index": 1, "device_token": "GPU-b"},
        {"worker_id": 1, "logical_index": 3, "device_token": "GPU-d"},
    ]
    assert phase["overlap"] == [
        {"logical_index": 3, "device_token": "GPU-d"},
        {"logical_index": 1, "device_token": "GPU-b"},
    ]
    assert phase["coordinator"]["device"] == "cuda:3"
    assert phase["coordinator"]["logical_index"] == 3

    output = capsys.readouterr().out
    assert "Phase topology: world=2; rank 0/coordinator=cuda:3; overlap logical GPUs=[3, 1]" in output
    assert "Training ranks: rank 0 -> logical 3 (GPU-d), rank 1 -> logical 1 (GPU-b)" in output
    assert "Rollout workers: worker 0 -> logical 1 (GPU-b), worker 1 -> logical 3 (GPU-d)" in output
    assert "level-1 sleep for training; rank-0 publish + worker ACK" in output
