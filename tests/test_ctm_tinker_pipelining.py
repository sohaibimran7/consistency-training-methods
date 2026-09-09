"""Offline contracts for Tinker's native setup and SFT submit-ahead queue."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from tinker import types
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier

from ctm.backends.base import ForwardBackwardOutput
from ctm.backends.tinker import TinkerBackend
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.training.sft import SFTConfig, train_sft


class _Renderer:
    def build_supervised_example(self, _messages):
        return types.ModelInput.from_ints(tokens=[1, 2, 3]), torch.tensor([0.0, 1.0, 1.0])


class _RecordingBackend:
    """Backend seam mock that records submission and completion order."""

    renderer_source = "tinker"
    training_submit_ahead = 1

    def __init__(self, *, fail_forward_index: int | None = None):
        self.events: list[tuple] = []
        self.forward_index = 0
        self.last_forward_index: int | None = None
        self.optim_submissions: list[tuple[int, float]] = []
        self.fail_forward_index = fail_forward_index

    def setup(self, **kwargs):
        self.events.append(("setup", kwargs["resume_from"], kwargs["resume_with_optimizer"]))

    async def submit_forward_backward(self, datums, loss_fn):
        index = self.forward_index
        self.forward_index += 1
        self.last_forward_index = index
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


def _run_sft(tmp_path: Path, backend: _RecordingBackend, **config_overrides) -> None:
    data = _write_samples(tmp_path, config_overrides.pop("sample_count"))
    with (
        patch("ctm.training.sft.setup_logging", return_value=MagicMock()),
        patch("ctm.training.sft.get_renderer_and_tokenizer", return_value=(_Renderer(), object())),
        patch("ctm.training.sft.write_run_manifest"),
    ):
        asyncio.run(train_sft(data, config=_config(tmp_path, **config_overrides), backend=backend))


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


def test_submit_ahead_propagates_a_pending_failure(tmp_path):
    backend = _RecordingBackend(fail_forward_index=0)
    with pytest.raises(RuntimeError, match="forward 0 failed"):
        _run_sft(tmp_path, backend, sample_count=2)

    assert ("submit_forward", 1, "cross_entropy") in backend.events
    assert not any(event[0] == "checkpoint" for event in backend.events)


class _AsyncFuture:
    def __init__(self, events: list[tuple], mode: str, path: str):
        self.events = events
        self.mode = mode
        self.path = path

    async def result_async(self):
        self.events.append(("finish_load", self.mode, self.path))


class _AsyncTrainingClient:
    def __init__(self, events: list[tuple]):
        self.events = events

    async def load_state_async(self, path: str):
        self.events.append(("load", "weights", path))
        return _AsyncFuture(self.events, "weights", path)

    async def load_state_with_optimizer_async(self, path: str):
        self.events.append(("load", "optimizer", path))
        return _AsyncFuture(self.events, "optimizer", path)


class _AsyncServiceClient:
    def __init__(self, training_client: _AsyncTrainingClient, events: list[tuple]):
        self.training_client = training_client
        self.events = events
        self.kwargs = None

    async def create_lora_training_client_async(self, **kwargs):
        self.kwargs = kwargs
        self.events.append(("create", kwargs["base_model"]))
        return self.training_client


@pytest.mark.parametrize(
    ("resume_with_optimizer", "expected_mode"),
    [(False, "weights"), (True, "optimizer")],
)
def test_tinker_setup_async_resumes_each_supported_mode(resume_with_optimizer, expected_mode):
    events: list[tuple] = []
    training_client = _AsyncTrainingClient(events)
    service_client = _AsyncServiceClient(training_client, events)
    backend = TinkerBackend(service_client=service_client)
    resume_path = "tinker://checkpoint/path"

    with (
        patch("ctm.backends.tinker.model_info.get_recommended_renderer_name", return_value="test-renderer"),
        patch(
            "ctm.backends.tinker.checkpoint_utils.add_renderer_name_to_user_metadata",
            side_effect=lambda metadata, name: metadata.update({"renderer": name}),
        ),
    ):
        asyncio.run(
            backend.setup_async(
                model="test/model",
                lora=LoRAConfig(rank=4, seed=7),
                resume_from=resume_path,
                resume_with_optimizer=resume_with_optimizer,
            )
        )

    assert backend.model == "test/model"
    assert backend.training_client is training_client
    assert service_client.kwargs["rank"] == 4
    assert service_client.kwargs["seed"] == 7
    assert service_client.kwargs["user_metadata"] == {"renderer": "test-renderer"}
    assert events == [
        ("create", "test/model"),
        ("load", expected_mode, resume_path),
        ("finish_load", expected_mode, resume_path),
    ]
