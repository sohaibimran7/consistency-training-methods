"""Offline tests for paired-prompt On-Policy Consistency Training."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch
from tinker import types

from ctm.backends.base import ForwardBackwardOutput, SampledSequence
from ctm.core.config import AdamConfig, CheckpointConfig, LoRAConfig
from ctm.evals.analysis.rollouts import iter_rollouts
from ctm.training.manifest import read_run_manifest
from ctm.training.opct import (
    OPCTConfig,
    OPCTGenerationConfig,
    OPCTTrainer,
    apply_reference_reverse_kl,
)
from ctm.training.rollout_log import RolloutLogger


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
    # Tinker removes the mask before submitting the loss, so prompt advantages
    # themselves must remain zero even when future action signal is discounted.
    assert torch.count_nonzero(advantages[~mask]) == 0


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
    def __init__(self, backend):
        self.backend = backend

    async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
        assert prompt.to_ints() == [20]
        self.backend.temperatures.append(temperature)
        # These are scores under the temperature-adjusted behavior distribution.
        return [SampledSequence(tokens=[30, 31], logprobs=[-4.0, -5.0]) for _ in range(num_samples)]

    async def score_completions(self, prompts, completion_tokens):
        self.backend.student_score_calls.append(
            ([prompt.to_ints() for prompt in prompts], [list(tokens) for tokens in completion_tokens])
        )
        # Raw current-policy scores intentionally differ from behavior scores.
        return [[-0.2, -0.3] for _ in completion_tokens]


class FakeReferencePolicy:
    def __init__(self, backend):
        self.backend = backend

    async def score_completions(self, prompts, completion_tokens):
        self.backend.reference_calls.append(
            ([prompt.to_ints() for prompt in prompts], [list(tokens) for tokens in completion_tokens])
        )
        return [[-1.2, -1.3] for _ in completion_tokens]


class FakeTokenizer:
    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class FakeBackend:
    renderer_source = "tinker"
    policy_samplers_are_snapshots = True

    def __init__(self):
        self.reference_calls = []
        self.student_score_calls = []
        self.temperatures = []
        self.fb_datums = []
        self.loss_fns = []
        self.optim_steps = 0
        self.refreshes = 0
        self.shutdowns = 0

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
        return FakeSampler(self)

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
        generation=OPCTGenerationConfig(rollouts_per_prompt=2, max_new_tokens=8, temperature=0.4),
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
    trainer.sampling_client = FakeSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)

    with patch("ctm.training.opct.setup_logging") as setup_logging:
        logger = MagicMock()
        setup_logging.return_value = logger
        checkpoint = asyncio.run(trainer.train([_pair(0), _pair(1)]))

    assert checkpoint == "fake://itest_opct"
    assert backend.loss_fns == ["importance_sampling", "importance_sampling"]
    assert backend.optim_steps == backend.refreshes == 2
    assert backend.shutdowns == 1
    assert len(backend.reference_calls) == 2
    assert len(backend.student_score_calls) == 2
    assert backend.temperatures == [0.4, 0.4]
    assert all(prompt == [10] for call, _ in backend.reference_calls for prompt in call)
    assert all(prompt == [20] for call, _ in backend.student_score_calls for prompt in call)
    assert all(tokens == [30, 31] for _, completions in backend.reference_calls for tokens in completions)
    assert all(tokens == [30, 31] for _, completions in backend.student_score_calls for tokens in completions)
    assert all(datum.model_input.to_ints()[0] == 20 for batch in backend.fb_datums for datum in batch)
    # Importance sampling retains behavior scores, while the KL advantage uses
    # raw student (-0.2/-0.3) versus raw teacher (-1.2/-1.3) scores.
    assert all(
        datum.loss_fn_inputs["logprobs"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0].tolist()
        == pytest.approx([-4.0, -5.0])
        for batch in backend.fb_datums
        for datum in batch
    )
    assert all(
        datum.loss_fn_inputs["advantages"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0].tolist()
        == pytest.approx([-2.0, -2.0])
        for batch in backend.fb_datums
        for datum in batch
    )
    assert all(
        datum.loss_fn_inputs["advantages"].to_torch()[datum.loss_fn_inputs["mask"].to_torch() > 0].lt(0).all()
        for batch in backend.fb_datums
        for datum in batch
    )
    manifest = read_run_manifest(tmp_path / "logs" / "itest" / "opct")
    assert manifest["kind"] == "opct" and manifest["n_samples"] == 2


def test_opct_logs_and_skips_invalid_rollouts_while_training_valid_completion(tmp_path):
    backend = FakeBackend()

    class MixedSampler(FakeSampler):
        async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
            assert num_samples == 5
            return [
                SampledSequence(tokens=[], logprobs=[]),
                SampledSequence(tokens=[40], logprobs=None),
                SampledSequence(tokens=[41, 42], logprobs=[-1.0]),
                SampledSequence(tokens=[43], logprobs=[float("nan")]),
                SampledSequence(tokens=[30, 31], logprobs=[-4.0, -5.0]),
            ]

    trainer = OPCTTrainer(
        config=OPCTConfig(generation=OPCTGenerationConfig(rollouts_per_prompt=5)),
        backend=backend,
    )
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = FakeTokenizer()
    trainer.sampling_client = MixedSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)
    trainer._rollout_logger = RolloutLogger(tmp_path / "rollouts")

    datums, _, _ = asyncio.run(trainer._build_batch([(7, _pair(7))]))
    trainer._log_rollouts(step=1, epoch=0)

    assert len(datums) == 1
    records = list(iter_rollouts(tmp_path / "rollouts"))
    assert [record.skip_reason for record in records] == [
        "empty_completion",
        "missing_logprobs",
        "misaligned_logprobs",
        "non_finite_logprobs",
        None,
    ]
    assert [record.skipped_from_training for record in records] == [True, True, True, True, False]
    assert all(record.datapoint_idx == 7 for record in records)
    assert records[4].completion_text == "30 31"
    assert records[4].prompt_context == {"reference": "10", "variant": "20"}
    assert records[4].reward == pytest.approx(-1.0)
    assert records[4].advantage == pytest.approx(-1.0)


def test_opct_all_invalid_batch_logs_provenance_without_optimizer_update(tmp_path):
    backend = FakeBackend()

    class InvalidSampler(FakeSampler):
        async def sample(self, prompt, *, max_tokens, temperature, stop, num_samples):
            assert num_samples == 2
            return [
                SampledSequence(tokens=[], logprobs=None),
                SampledSequence(tokens=[40], logprobs=None),
            ]

        async def score_completions(self, prompts, completion_tokens):
            raise AssertionError("invalid completions must not be scored")

    config = OPCTConfig(
        experiment_name="itest",
        run_name="invalid",
        model="fake-model",
        generation=OPCTGenerationConfig(rollouts_per_prompt=2),
        batch_size=1,
        checkpoint=CheckpointConfig(save_every_n_steps=10),
        log_base_dir=str(tmp_path / "logs"),
    )
    trainer = OPCTTrainer(config=config, backend=backend)
    trainer.setup_done = True
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = FakeTokenizer()
    trainer.sampling_client = InvalidSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)

    with patch("ctm.training.opct.setup_logging") as setup_logging:
        setup_logging.return_value = MagicMock()
        checkpoint = asyncio.run(trainer.train([_pair(0)]))

    assert checkpoint == "fake://itest_invalid"
    assert backend.fb_datums == []
    assert backend.optim_steps == backend.refreshes == 0
    records = list(iter_rollouts(tmp_path / "logs" / "itest" / "invalid" / "rollouts"))
    assert [record.skip_reason for record in records] == ["empty_completion", "missing_logprobs"]


def test_opct_refuses_to_overwrite_existing_rollout_steps(tmp_path):
    rollout_dir = tmp_path / "existing-rollouts"
    rollout_dir.mkdir()
    (rollout_dir / "index.json").write_text('{"steps":[{"step":1}]}', encoding="utf-8")
    backend = FakeBackend()
    config = OPCTConfig(
        experiment_name="itest",
        run_name="warm-start",
        model="fake-model",
        rollout_dir=str(rollout_dir),
        log_base_dir=str(tmp_path / "logs"),
    )
    trainer = OPCTTrainer(config=config, backend=backend)
    trainer.setup_done = True

    with patch("ctm.training.opct.setup_logging") as setup_logging:
        setup_logging.return_value = MagicMock()
        with pytest.raises(FileExistsError, match="warm start"):
            asyncio.run(trainer.train([_pair(0)]))

    assert backend.fb_datums == []


def test_opct_rollout_write_failure_aborts_before_training_mutation(tmp_path):
    backend = FakeBackend()
    config = OPCTConfig(
        experiment_name="itest",
        run_name="write-failure",
        model="fake-model",
        generation=OPCTGenerationConfig(rollouts_per_prompt=1),
        batch_size=1,
        log_base_dir=str(tmp_path / "logs"),
    )
    trainer = OPCTTrainer(config=config, backend=backend)
    trainer.setup_done = True
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = FakeTokenizer()
    trainer.sampling_client = FakeSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)
    failing_rollout_logger = MagicMock()
    failing_rollout_logger.log_step.side_effect = OSError("disk full")

    with (
        patch("ctm.training.opct.setup_logging") as setup_logging,
        patch("ctm.training.opct.RolloutLogger", return_value=failing_rollout_logger),
    ):
        setup_logging.return_value = MagicMock()
        with pytest.raises(OSError, match="disk full"):
            asyncio.run(trainer.train([_pair(0)]))

    assert backend.fb_datums == []
    assert backend.optim_steps == 0


class SetupBackend:
    renderer_source = "tinker"

    def __init__(self, *, snapshots):
        self.policy_samplers_are_snapshots = snapshots
        self.setup_calls = []
        self.initial_policy = object()
        self.base_policy = object()
        self.base_calls = 0

    def setup(self, **kwargs):
        self.setup_calls.append(kwargs)

    def policy_sampler(self, name):
        return self.initial_policy

    def base_sampler(self):
        self.base_calls += 1
        return self.base_policy


def test_opct_resume_uses_exact_run_start_snapshot_as_teacher():
    backend = SetupBackend(snapshots=True)
    trainer = OPCTTrainer(
        config=OPCTConfig(model="unit/model"),
        backend=backend,
        resume_from="tinker://checkpoint",
    )

    with patch("ctm.training.opct.get_renderer_and_tokenizer", return_value=(FakeRenderer(), MagicMock())):
        trainer.setup()

    assert backend.setup_calls[0]["resume_from"] == "tinker://checkpoint"
    assert trainer.reference_policy is trainer.sampling_client is backend.initial_policy
    assert backend.base_calls == 0


def test_opct_resume_rejects_live_policy_handle_before_loading_checkpoint():
    backend = SetupBackend(snapshots=False)
    trainer = OPCTTrainer(
        config=OPCTConfig(model="unit/model"),
        backend=backend,
        resume_from="file://checkpoint",
    )

    with pytest.raises(NotImplementedError, match="immutable run-start policy handle"):
        trainer.setup()
    assert backend.setup_calls == []


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
