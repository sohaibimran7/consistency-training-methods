"""Phase lifecycle regression tests for the OPCT training loop.

These are deliberately CPU-only scheduling tests.  The recording backend has
the same lifecycle contract as a sleeping rollout-worker topology, while its
small sampler/optimizer keeps the assertions about ordering independent of
CUDA, vLLM, and any paid service.
"""

from __future__ import annotations

import asyncio
import random
from unittest.mock import MagicMock, patch

import torch
from tinker import types

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.training.opct import OPCTConfig, OPCTGenerationConfig, OPCTTrainer


class _Tokenizer:
    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _Renderer:
    def build_generation_prompt(self, messages):
        content = str(messages[0]["content"])
        index = int(content.rsplit(" ", 1)[-1])
        return types.ModelInput.from_ints(tokens=[20 if "variant" in content else 10, index])

    def get_stop_sequences(self):
        return []


class _Sampler:
    def __init__(self, events, *, role: str):
        self.events = events
        self.role = role

    async def sample_batch(self, prompts, *, max_tokens, temperature, stop, num_samples):
        del max_tokens, temperature, stop
        self.events.append(("sample", self.role, tuple(tuple(prompt.to_ints()) for prompt in prompts)))
        return [
            [SampledSequence(tokens=[30], logprobs=[-0.5]) for _ in range(num_samples)]
            for _ in prompts
        ]

    async def score_completions(self, prompts, completion_tokens):
        self.events.append(
            (
                "score",
                self.role,
                tuple(tuple(prompt.to_ints()) for prompt in prompts),
                tuple(tuple(tokens) for tokens in completion_tokens),
            )
        )
        value = -1.0 if self.role == "reference" else -0.2
        return [[value for _ in tokens] for tokens in completion_tokens]


class _RecordingBackend:
    """Minimal backend that makes phase and async ordering observable."""

    renderer_source = "tinker"
    policy_samplers_are_snapshots = True

    def __init__(self, events, *, phase_shared: bool):
        self.events = events
        self.sampling_training_overlap_supported = not phase_shared
        self._phase_shared = phase_shared
        self.shutdown_calls = 0

    def setup(self, **_kwargs):
        return None

    def policy_sampler(self, _name):
        return _Sampler(self.events, role="student")

    def base_sampler(self):
        return _Sampler(self.events, role="reference")

    async def enter_rollout_phase(self):
        if not self._phase_shared:
            raise AssertionError("disjoint OPCT backend must not receive a rollout phase callback")
        self.events.append(("rollout_phase",))

    async def enter_training_phase(self):
        if not self._phase_shared:
            raise AssertionError("disjoint OPCT backend must not receive a training phase callback")
        self.events.append(("training_phase",))

    async def submit_forward_backward(self, datums, loss_fn):
        self.events.append(("fwd_submitted", loss_fn, len(datums)))
        lengths = [int(datum.loss_fn_inputs["target_tokens"].to_torch().numel()) for datum in datums]
        events = self.events

        class _Pending:
            async def result(self):
                events.append(("fwd_result_started",))
                # This exposes a scheduler yield point.  A rollout request
                # scheduled too early would be visible between the markers.
                await asyncio.sleep(0)
                events.append(("fwd_result_completed",))
                return ForwardBackwardOutput(
                    logprobs=[torch.zeros(length) for length in lengths],
                    metrics={"loss": 0.5},
                )

        return _Pending()

    async def submit_optim_step(self, *, learning_rate, adam):
        del adam
        self.events.append(("optim_submitted", learning_rate))
        events = self.events

        class _Pending:
            async def result(self):
                events.append(("optim_completed",))

        return _Pending()

    async def refresh_policy_sampler(self, name):
        self.events.append(("refresh_policy_sampler", name))
        return _Sampler(self.events, role="student")

    async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
        del log_dir, loop_state, kind
        self.events.append(("checkpoint_started", name))
        # A checkpoint must not leave an opportunity for the next rollout
        # wake on a phase-shared host.
        await asyncio.sleep(0)
        self.events.append(("checkpoint_completed", name))
        return {"sampler_path": f"fake://{name}", "state_path": None}

    def shutdown(self):
        self.shutdown_calls += 1


def _pair(index: int) -> dict:
    return {
        "reference_messages": [{"role": "user", "content": f"reference {index}"}],
        "variant_messages": [{"role": "user", "content": f"variant {index}"}],
    }


def _trainer(tmp_path, backend, *, accumulation: int, checkpoint: CheckpointConfig) -> OPCTTrainer:
    trainer = OPCTTrainer(
        config=OPCTConfig(
            experiment_name="phase-shared-opct",
            run_name="lifecycle",
            model="unit/model",
            lora=LoRAConfig(rank=4, seed=0),
            optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="constant"),
            generation=OPCTGenerationConfig(rollouts_per_prompt=1, max_new_tokens=8, temperature=0.7),
            batch_size=1,
            gradient_accumulation_steps=accumulation,
            checkpoint=checkpoint,
            log_base_dir=str(tmp_path / "logs"),
            rollout_log="none",
        ),
        backend=backend,
    )
    trainer.setup_done = True
    trainer.renderer = _Renderer()
    trainer.tokenizer = _Tokenizer()
    trainer.sampling_client = _Sampler(backend.events, role="student")
    trainer.reference_policy = _Sampler(backend.events, role="reference")
    return trainer


def _run(trainer: OPCTTrainer, samples: list[dict]) -> None:
    random.seed(0)
    with patch("ctm.training.opct.setup_logging", return_value=MagicMock()):
        asyncio.run(trainer.train(samples))


def test_phase_shared_opct_finishes_sampling_and_scoring_before_forward_backward(tmp_path):
    events = []
    backend = _RecordingBackend(events, phase_shared=True)
    trainer = _trainer(tmp_path, backend, accumulation=2, checkpoint=CheckpointConfig())

    _run(trainer, [_pair(0), _pair(1)])

    first_training = events.index(("training_phase",))
    first_fwd_start = events.index(("fwd_result_started",))
    first_fwd_done = events.index(("fwd_result_completed",))

    # The one unchanged-policy group does all generation and raw scoring while
    # vLLM is awake, then transitions once into coordinator work.
    assert events[0] == ("rollout_phase",)
    assert any(event[0] == "sample" for event in events[:first_training])
    assert {event[1] for event in events[:first_training] if event[0] == "score"} == {
        "student",
        "reference",
    }
    assert not any(event[0] in {"sample", "score"} for event in events[first_training:])
    assert first_training < first_fwd_start < first_fwd_done
    assert backend.shutdown_calls == 1


def test_phase_shared_opct_checkpoints_before_refresh_and_next_rollout(tmp_path):
    events = []
    backend = _RecordingBackend(events, phase_shared=True)
    trainer = _trainer(
        tmp_path,
        backend,
        accumulation=1,
        checkpoint=CheckpointConfig(save_every_n_steps=1, skip_near_final_steps=0),
    )

    _run(trainer, [_pair(0), _pair(1)])

    checkpoint_started = next(index for index, event in enumerate(events) if event[0] == "checkpoint_started")
    checkpoint_completed = next(index for index, event in enumerate(events) if event[0] == "checkpoint_completed")
    refresh = next(index for index, event in enumerate(events) if event[0] == "refresh_policy_sampler")
    next_rollout = next(
        index for index, event in enumerate(events) if index > refresh and event == ("rollout_phase",)
    )
    next_sample = next(
        index for index, event in enumerate(events) if index > next_rollout and event[0] == "sample"
    )

    assert any(event == ("training_phase",) for event in events[:checkpoint_started])
    assert checkpoint_started < checkpoint_completed < refresh < next_rollout < next_sample
    assert backend.shutdown_calls == 1


def test_disjoint_opct_keeps_historical_refresh_before_checkpoint_order(tmp_path):
    events = []
    backend = _RecordingBackend(events, phase_shared=False)
    trainer = _trainer(
        tmp_path,
        backend,
        accumulation=1,
        checkpoint=CheckpointConfig(save_every_n_steps=1, skip_near_final_steps=0),
    )

    _run(trainer, [_pair(0)])

    refresh = next(index for index, event in enumerate(events) if event[0] == "refresh_policy_sampler")
    first_checkpoint = next(index for index, event in enumerate(events) if event[0] == "checkpoint_started")
    assert refresh < first_checkpoint
    assert not any(event[0] in {"rollout_phase", "training_phase"} for event in events)
    assert backend.shutdown_calls == 1
