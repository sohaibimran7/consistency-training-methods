"""Offline tests for paired-prompt On-Policy Consistency Training."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch
from tinker import types

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.training.manifest import read_run_manifest
from ctm.training.opct import (
    OPCTConfig,
    OPCTGenerationConfig,
    OPCTTrainer,
    apply_reference_reverse_kl,
)


def _datum(sampled=(-1.0, -2.0)):
    return OPCTTrainer._create_datum(
        types.ModelInput.from_ints(tokens=[10, 11]),
        [20, 21],
        list(sampled),
    )


def test_reference_reverse_kl_aligns_only_action_tokens_and_discounts_future_signal():
    datum = _datum()
    apply_reference_reverse_kl(
        [datum],
        [[-2.0, -1.0]],
        kl_coef=2.0,
        kl_discount_factor=0.5,
    )

    mask = datum.loss_fn_inputs["mask"].to_torch() > 0
    advantages = datum.loss_fn_inputs["advantages"].to_torch()
    # reverse KL = [1, -1], raw signal = [-2, 2], discounted = [-1, 2]
    assert advantages[mask].tolist() == pytest.approx([-1.0, 2.0])


def test_reference_reverse_kl_is_zero_when_student_and_teacher_scores_match():
    datum = _datum(sampled=(-0.2, -0.3))
    metrics = apply_reference_reverse_kl(
        [datum],
        [[-0.2, -0.3]],
        kl_coef=1.0,
        kl_discount_factor=0.9,
    )
    assert metrics["teacher_kl"] == pytest.approx(0.0)
    assert torch.count_nonzero(datum.loss_fn_inputs["advantages"].to_torch()) == 0


def test_reference_reverse_kl_rejects_teacher_token_misalignment():
    with pytest.raises(ValueError, match="2 action tokens but 1 teacher logprobs"):
        apply_reference_reverse_kl(
            [_datum()],
            [[-1.0]],
            kl_coef=1.0,
            kl_discount_factor=0.0,
        )


class FakeRenderer:
    def build_generation_prompt(self, messages):
        is_variant = "variant" in messages[0]["content"]
        return types.ModelInput.from_ints(tokens=[20 if is_variant else 10])

    def get_stop_sequences(self):
        return []


class FakeSampler:
    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        assert prompt.to_ints() == [20]
        return [SampledSequence(tokens=[30, 31], logprobs=[-0.2, -0.3]) for _ in range(num_samples)]


class FakeBackend:
    renderer_source = "tinker"
    policy_samplers_are_snapshots = True

    def __init__(self):
        self.reference_calls = []
        self.fb_datums = []
        self.loss_fns = []
        self.optim_steps = 0
        self.refreshes = 0
        self.shutdowns = 0

    async def score_reference_completions(self, reference_prompts, completion_tokens):
        self.reference_calls.append(
            ([prompt.to_ints() for prompt in reference_prompts], [list(tokens) for tokens in completion_tokens])
        )
        return [[-1.2, -1.3] for _ in completion_tokens]

    async def submit_forward_backward(self, datums, loss_fn):
        self.fb_datums.append(list(datums))
        self.loss_fns.append(loss_fn)

        class Pending:
            async def result(self_inner):
                lengths = [len(d.loss_fn_inputs["target_tokens"].to_torch()) for d in datums]
                return ForwardBackwardOutput(
                    logprobs=[torch.zeros(length) for length in lengths],
                    metrics={"loss": 0.5},
                )

        return Pending()

    async def submit_optim_step(self, *, learning_rate, adam):
        self.optim_steps += 1

        class Pending:
            async def result(self_inner):
                return None

        return Pending()

    async def refresh_policy_sampler(self, name):
        self.refreshes += 1
        return FakeSampler()

    async def save_checkpoint(self, *, name, log_dir, loop_state, kind):
        return {"sampler_path": f"fake://{name}", "state_path": None}

    def shutdown(self):
        self.shutdowns += 1


def _pair(index):
    return {
        "reference_messages": [{"role": "user", "content": f"reference {index}"}],
        "variant_messages": [{"role": "user", "content": f"variant {index}"}],
    }


def test_opct_loop_samples_variant_scores_reference_and_refreshes_every_update(tmp_path):
    backend = FakeBackend()
    config = OPCTConfig(
        experiment_name="itest",
        run_name="opct",
        model="fake-model",
        lora=LoRAConfig(rank=4, seed=0),
        optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="constant"),
        generation=OPCTGenerationConfig(rollouts_per_prompt=2, max_new_tokens=8, temperature=0.7),
        batch_size=1,
        n_epochs=1,
        kl_coef=2.0,
        kl_discount_factor=0.0,
        checkpoint=CheckpointConfig(save_every_n_steps=10),
        log_base_dir=str(tmp_path / "logs"),
    )
    trainer = OPCTTrainer(config=config, backend=backend)
    trainer.setup_done = True
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = MagicMock()
    trainer.sampling_client = FakeSampler()

    with patch("ctm.training.opct.setup_logging") as setup_logging:
        logger = MagicMock()
        setup_logging.return_value = logger
        checkpoint = asyncio.run(trainer.train([_pair(0), _pair(1)]))

    assert checkpoint == "fake://itest_opct"
    assert backend.loss_fns == ["importance_sampling", "importance_sampling"]
    assert backend.optim_steps == backend.refreshes == 2
    assert backend.shutdowns == 1
    assert len(backend.reference_calls) == 2
    assert all(prompt == [10] for call, _ in backend.reference_calls for prompt in call)
    assert all(tokens == [30, 31] for _, completions in backend.reference_calls for tokens in completions)
    assert all(datum.model_input.to_ints()[0] == 20 for batch in backend.fb_datums for datum in batch)
    assert all(
        datum.loss_fn_inputs["advantages"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0].lt(0).all()
        for batch in backend.fb_datums
        for datum in batch
    )
    manifest = read_run_manifest(tmp_path / "logs" / "itest" / "opct")
    assert manifest["kind"] == "opct" and manifest["n_samples"] == 2


def test_opct_allows_same_prompt_field_for_self_distillation_control():
    config = OPCTConfig(reference_messages_field="messages", variant_messages_field="messages")
    sample = {"messages": [{"role": "user", "content": "same prompt"}]}
    from ctm.training.opct import validate_opct_samples

    assert validate_opct_samples([sample], config) == [sample]


def test_opct_cli_dry_run_validates_pairs_without_initializing_backend(tmp_path, capsys):
    from scripts.train_opct import main

    data = tmp_path / "pairs.jsonl"
    data.write_text(
        '{"reference_messages":[{"role":"user","content":"clean"}],'
        '"variant_messages":[{"role":"user","content":"variant"}]}\n',
        encoding="utf-8",
    )
    with patch("scripts.train_opct.build_backend") as build_backend:
        main(
            [
                "--model",
                "unit/model",
                "--data",
                str(data),
                "--experiment-name",
                "unit",
                "--run-name",
                "dry",
                "--dry-run",
            ]
        )
    assert not build_backend.called
    output = capsys.readouterr().out
    assert "Method: OPCT" in output
    assert "Policy refresh: after every optimizer update" in output
