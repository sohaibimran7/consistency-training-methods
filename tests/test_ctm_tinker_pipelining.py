"""Offline contracts for Tinker's native setup and SFT submit-ahead queue."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from tinker import types
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

from ctm.backends.base import ForwardBackwardOutput
from ctm.backends.tinker import TinkerBackend
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.training.sft import SFTConfig, train_sft


class _Renderer:
    def build_supervised_example(self, messages):
        sample_id = int(messages[0]["content"][1:])
        return types.ModelInput.from_ints(tokens=[1, sample_id + 2, 3]), torch.tensor([0.0, 1.0, 1.0])


class _RecordingBackend:
    """Backend seam mock that records submission and completion order."""

    renderer_source = "tinker"
    training_submit_ahead = 1

    def __init__(self, *, fail_forward_index: int | None = None, fail_optimizer_index: int | None = None):
        self.events: list[tuple] = []
        self.forward_index = 0
        self.last_forward_index: int | None = None
        self.optim_submissions: list[tuple[int, float]] = []
        self.fail_forward_index = fail_forward_index
        self.fail_optimizer_index = fail_optimizer_index
        self.batches: list[list[list[int]]] = []

    def setup(self, **kwargs):
        self.events.append(("setup", kwargs["resume_from"], kwargs["resume_with_optimizer"]))

    async def submit_forward_backward(self, datums, loss_fn):
        index = self.forward_index
        self.forward_index += 1
        self.last_forward_index = index
        self.batches.append([datum.model_input.to_ints() for datum in datums])
        lengths = [datum.loss_fn_inputs["weights"].to_torch().shape[0] for datum in datums]
        self.events.append(("submit_forward", index, loss_fn))
        backend = self

        class Pending:
            async def result(self):
                backend.events.append(("finish_forward", index))
                if backend.fail_forward_index == index:
                    raise RuntimeError(f"forward {index} failed")
                return ForwardBackwardOutput(
                    logprobs=[-0.5 * torch.ones(length) for length in lengths],
                    metrics={"loss": 0.5},
                )

        return Pending()

    async def submit_optim_step(self, *, learning_rate, adam):
        assert self.last_forward_index is not None
        index = self.last_forward_index
        self.optim_submissions.append((index, learning_rate))
        self.events.append(("submit_optim", index, learning_rate))
        backend = self

        class Pending:
            async def result(self):
                backend.events.append(("finish_optim", index))
                if backend.fail_optimizer_index == index:
                    raise RuntimeError(f"optimizer {index} failed")

        return Pending()

    async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
        self.events.append(("checkpoint", name, dict(loop_state), kind))
        return {"sampler_path": f"fake://{name}", "state_path": f"fake-state://{name}"}


def _write_samples(tmp_path: Path, count: int) -> Path:
    path = tmp_path / "train.jsonl"
    samples = [
        {"messages": [{"role": "user", "content": f"q{index}"}, {"role": "assistant", "content": f"a{index}"}]}
        for index in range(count)
    ]
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    return path


def _config(tmp_path: Path, **overrides) -> SFTConfig:
    values = {
        "experiment_name": "pipelining",
        "run_name": "test",
        "lora": LoRAConfig(seed=0),
        "optimizer": AdamConfig(learning_rate=1e-3, lr_schedule="linear"),
        "batch_size": 1,
        "n_epochs": 1,
        "log_base_dir": str(tmp_path / "logs"),
    }
    values.update(overrides)
    return SFTConfig(**values)


def _run_sft(tmp_path: Path, backend: _RecordingBackend, **config_overrides) -> MagicMock:
    data = _write_samples(tmp_path, config_overrides.pop("sample_count"))
    resume = {
        key: config_overrides.pop(key) for key in ("resume_from", "resume_with_optimizer") if key in config_overrides
    }
    with (
        patch("ctm.training.sft.setup_logging", return_value=MagicMock()) as logging,
        patch("ctm.training.sft.get_renderer_and_tokenizer", return_value=(_Renderer(), object())),
        patch("ctm.training.sft.write_run_manifest"),
    ):
        asyncio.run(train_sft(data, config=_config(tmp_path, **config_overrides), backend=backend, **resume))
    return logging.return_value


def test_submit_ahead_preserves_queue_order_partial_accumulation_and_lr(tmp_path):
    backend = _RecordingBackend()
    _run_sft(tmp_path, backend, sample_count=5, gradient_accumulation_steps=2)

    assert backend.events.index(("submit_forward", 1, "cross_entropy")) < backend.events.index(("finish_forward", 0))
    assert [index for index, _ in backend.optim_submissions] == [1, 3, 4]
    expected_lrs = [
        1e-3
        * max(
            0.0,
            compute_schedule_lr_multiplier(lr_schedule="linear", step=step, total_steps=3),
        )
        for step in range(3)
    ]
    assert [learning_rate for _, learning_rate in backend.optim_submissions] == pytest.approx(expected_lrs)


def test_backends_without_submit_ahead_keep_the_existing_submit_then_wait_order(tmp_path):
    class SequentialBackend(_RecordingBackend):
        training_submit_ahead = 0

    backend = SequentialBackend()
    _run_sft(tmp_path, backend, sample_count=2)

    assert backend.events.index(("finish_forward", 0)) < backend.events.index(("submit_forward", 1, "cross_entropy"))


def test_submit_ahead_drains_before_each_intermediate_checkpoint_boundary(tmp_path):
    backend = _RecordingBackend()
    _run_sft(
        tmp_path,
        backend,
        sample_count=3,
        checkpoint=CheckpointConfig(save_every_n_steps=1, save_state=True),
    )

    checkpoints = [event for event in backend.events if event[0] == "checkpoint"]
    assert [(state, kind) for _, _, state, kind in checkpoints] == [
        ({"epoch": 0, "step": 1}, "both"),
        ({"epoch": 0, "step": 2}, "both"),
        ({"epoch": 1, "step": 3, "final": True}, "both"),
    ]
    assert backend.events.index(checkpoints[0]) < backend.events.index(("submit_forward", 1, "cross_entropy"))
    assert backend.events.index(checkpoints[1]) < backend.events.index(("submit_forward", 2, "cross_entropy"))


@pytest.mark.parametrize("failure", ["forward", "optimizer"])
def test_submit_ahead_propagates_a_pending_failure(tmp_path, failure):
    backend = _RecordingBackend(**{f"fail_{failure}_index": 0})
    with pytest.raises(RuntimeError, match=f"{failure} 0 failed"):
        _run_sft(tmp_path, backend, sample_count=5)

    assert ("submit_forward", 1, "cross_entropy") in backend.events
    assert backend.forward_index == 2
    assert not any(event[0] == "checkpoint" for event in backend.events)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("resume_with_optimizer", [None, False, True])
def test_tinker_setup_preserves_client_parameters_and_resume_modes(use_async, resume_with_optimizer):
    future = MagicMock(result_async=AsyncMock())
    training_client = MagicMock(
        load_state_async=AsyncMock(return_value=future),
        load_state_with_optimizer_async=AsyncMock(return_value=future),
    )
    training_client.load_state.return_value = future
    training_client.load_state_with_optimizer.return_value = future
    service_client = MagicMock(create_lora_training_client_async=AsyncMock(return_value=training_client))
    service_client.create_lora_training_client.return_value = training_client
    backend = TinkerBackend(service_client=service_client)
    resume_path = "tinker://checkpoint/path" if resume_with_optimizer is not None else None

    with (
        patch("ctm.backends.tinker.model_info.get_recommended_renderer_name", return_value="test-renderer"),
        patch(
            "ctm.backends.tinker.checkpoint_utils.add_renderer_name_to_user_metadata",
            side_effect=lambda metadata, name: metadata.update({"renderer": name}),
        ),
    ):
        kwargs = dict(
            model="test/model",
            lora=LoRAConfig(rank=4, seed=7),
            resume_from=resume_path,
            resume_with_optimizer=bool(resume_with_optimizer),
        )
        if use_async:
            asyncio.run(backend.setup_async(**kwargs))
            service_client.create_lora_training_client_async.assert_awaited_once()
            service_client.create_lora_training_client.assert_not_called()
            create = service_client.create_lora_training_client_async
        else:
            backend.setup(**kwargs)
            service_client.create_lora_training_client.assert_called_once()
            service_client.create_lora_training_client_async.assert_not_called()
            create = service_client.create_lora_training_client

    assert backend.model == "test/model"
    assert backend.training_client is training_client
    assert create.call_args.kwargs == dict(
        base_model="test/model",
        rank=4,
        seed=7,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
        user_metadata={"renderer": "test-renderer"},
    )
    load_name = "load_state_with_optimizer" if resume_with_optimizer else "load_state"
    if use_async:
        load_name += "_async"
    for name in ("load_state", "load_state_async", "load_state_with_optimizer", "load_state_with_optimizer_async"):
        load = getattr(training_client, name)
        if resume_path and name == load_name:
            if use_async:
                load.assert_awaited_once_with(resume_path)
                future.result_async.assert_awaited_once()
            else:
                load.assert_called_once_with(resume_path)
                future.result.assert_called_once()
        else:
            load.assert_not_called()


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("unsupported", [{"target_modules": ["q_proj"]}, {"alpha": 9}, {"dropout": 0.1}])
def test_unsupported_lora_is_rejected_before_service_creation(use_async, unsupported):
    backend = TinkerBackend()
    with patch("ctm.backends.tinker.tinker.ServiceClient", side_effect=AssertionError("service must stay lazy")):
        with pytest.raises(NotImplementedError):
            kwargs = dict(model="test/model", lora=LoRAConfig(rank=4, **unsupported))
            if use_async:
                asyncio.run(backend.setup_async(**kwargs))
            else:
                backend.setup(**kwargs)


@pytest.mark.parametrize(
    "sample_count,batch_size,accumulation,epochs,every,skip",
    [
        (1, 8, 2, 2, None, 0),
        (7, 2, 2, 3, 2, 1),
        (5, 1, 2, 2, 1, 0),
        (6, 2, 4, 2, None, 0),
    ],
)
def test_pipeline_matches_sequential_training_and_bounds_pending_work(
    tmp_path, sample_count, batch_size, accumulation, epochs, every, skip
):
    config = dict(
        sample_count=sample_count,
        batch_size=batch_size,
        gradient_accumulation_steps=accumulation,
        n_epochs=epochs,
        checkpoint=CheckpointConfig(save_every_n_steps=every, skip_near_final_steps=skip),
    )
    sequential = _RecordingBackend()
    sequential.training_submit_ahead = 0
    pipelined = _RecordingBackend()
    sequential_log = _run_sft(tmp_path, sequential, **config)
    pipelined_log = _run_sft(tmp_path, pipelined, **config)

    assert pipelined.batches == sequential.batches
    assert pipelined.optim_submissions == sequential.optim_submissions
    assert pipelined_log.log_metrics.call_args_list == sequential_log.log_metrics.call_args_list

    def submitted(backend):
        return [e for e in backend.events if e[0].startswith("submit_") or e[0] == "checkpoint"]

    assert submitted(pipelined) == submitted(sequential)
    outstanding = 0
    for event in pipelined.events:
        if event[0] == "submit_forward":
            outstanding += 1
        elif event[0] == "finish_forward":
            outstanding -= 1
        elif event[0] == "checkpoint":
            assert outstanding == 0
        assert 0 <= outstanding <= 2
    assert outstanding == 0


@pytest.mark.parametrize(
    "resume_path,override,expected",
    [
        (None, None, False),
        ("tinker://checkpoint/weights/1", None, True),
        ("tinker://checkpoint/sampler_weights/1", None, False),
        ("tinker://checkpoint/weights/1", False, False),
    ],
)
def test_sft_awaits_native_setup_with_resolved_resume_mode(tmp_path, resume_path, override, expected):
    class AsyncSetupBackend(_RecordingBackend):
        def setup(self, **kwargs):
            pytest.fail("synchronous setup must not be called")

        async def setup_async(self, **kwargs):
            self.events.append(("setup_async", kwargs["resume_from"], kwargs["resume_with_optimizer"]))

    backend = AsyncSetupBackend()
    _run_sft(tmp_path, backend, sample_count=1, resume_from=resume_path, resume_with_optimizer=override)
    assert backend.events[0] == ("setup_async", resume_path, expected)
