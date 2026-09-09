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
    discounted_future_sum,
)
from ctm.training.rollout_log import RolloutLogger


def test_opct_generation_config_accepts_explicit_eos_only_policy():
    config = OPCTGenerationConfig(max_new_tokens=None)
    assert config.max_new_tokens is None


def _datum(sampled=(-1.0, -2.0)):
    return OPCTTrainer._create_datum(
        types.ModelInput.from_ints(tokens=[10, 11]),
        [20, 21],
        list(sampled),
    )


def _discounted_future_sum_reference(values: torch.Tensor, discount: float) -> torch.Tensor:
    output = torch.empty_like(values)
    running = torch.zeros((), dtype=values.dtype, device=values.device)
    for index in range(len(values) - 1, -1, -1):
        running = values[index] + discount * running
        output[index] = running
    return output


def test_opct_generation_config_keeps_legacy_greedy_temperature():
    assert OPCTGenerationConfig(temperature=0.0).temperature == 0.0


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf"), True])
def test_opct_generation_rejects_invalid_temperatures(temperature):
    with pytest.raises(ValueError, match="temperature must be a finite non-negative number"):
        OPCTGenerationConfig(temperature=temperature)


@pytest.mark.parametrize("discount", [0.0, 0.5, 0.999, 1.0])
def test_discounted_future_sum_matches_recurrence_on_long_sequences(discount):
    values = torch.randn(16_385, generator=torch.Generator().manual_seed(1234))

    actual = discounted_future_sum(values, discount)
    expected = _discounted_future_sum_reference(values, discount)

    assert actual.dtype == values.dtype
    assert actual.device == values.device
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-4)


def test_discounted_future_sum_preserves_dtype_and_handles_empty_input():
    for dtype in (torch.float16, torch.float32, torch.float64):
        values = torch.tensor([1.0, -2.0, 3.0], dtype=dtype)
        actual = discounted_future_sum(values, 0.75)
        expected = _discounted_future_sum_reference(values, 0.75)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)

    empty = torch.empty(0, dtype=torch.float64)
    actual_empty = discounted_future_sum(empty, 0.9)
    assert actual_empty.shape == empty.shape
    assert actual_empty.dtype == empty.dtype


@pytest.mark.parametrize("discount", [-0.01, 1.01, float("nan")])
def test_discounted_future_sum_rejects_invalid_discounts(discount):
    with pytest.raises(ValueError, match=r"discount must be in \[0, 1\]"):
        discounted_future_sum(torch.ones(3), discount)


def test_discounted_future_sum_has_the_same_gradients_as_recurrence():
    actual_values = torch.randn(257, dtype=torch.float64, requires_grad=True)
    expected_values = actual_values.detach().clone().requires_grad_()
    weights = torch.randn(257, dtype=torch.float64)

    actual_loss = (discounted_future_sum(actual_values, 0.87) * weights).sum()
    expected_loss = (_discounted_future_sum_reference(expected_values, 0.87) * weights).sum()
    (actual_gradient,) = torch.autograd.grad(actual_loss, actual_values)
    (expected_gradient,) = torch.autograd.grad(expected_loss, expected_values)

    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)


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


def test_reference_reverse_kl_discount_does_not_cross_datum_boundaries():
    datums = [_datum(), _datum()]
    apply_reference_reverse_kl(
        datums,
        [[0.0, 0.0], [0.0, 0.0]],
        student_action_logprobs=[[1.0, 2.0], [10.0, 20.0]],
        kl_coef=1.0,
        kl_discount_factor=0.5,
    )

    action_advantages = []
    for datum in datums:
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        advantages = datum.loss_fn_inputs["advantages"].to_torch()
        action_advantages.append(advantages[mask].tolist())
        assert torch.count_nonzero(advantages[~mask]) == 0

    torch.testing.assert_close(
        torch.tensor(action_advantages),
        torch.tensor([[-2.0, -2.0], [-20.0, -20.0]]),
    )


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


class FusedFakeBackend(FakeBackend):
    supports_fused_opct_scoring = True

    def __init__(self, rollout_dir=None):
        super().__init__()
        self.fused_behavior_actions = []
        self.rollout_dir = rollout_dir
        self.records_before_optim = []

    async def submit_opct_forward_backward(
        self,
        datums,
        *,
        behavior_temperature,
        kl_coef,
        kl_discount_factor,
        loss_fn,
    ):
        del behavior_temperature, kl_coef, kl_discount_factor
        self.fb_datums.append(list(datums))
        self.loss_fns.append(loss_fn)
        raw_logprobs = []
        for datum in datums:
            mask = datum.loss_fn_inputs["mask"].to_torch().bool()
            behavior = datum.loss_fn_inputs["logprobs"].to_torch()
            self.fused_behavior_actions.append(behavior[mask].tolist())
            raw = torch.zeros_like(behavior)
            raw[mask] = torch.tensor(
                [-0.2 - 0.1 * index for index in range(int(mask.sum()))],
                dtype=raw.dtype,
            )
            raw_logprobs.append(raw)

        class Pending:
            async def result(self_inner):
                return ForwardBackwardOutput(
                    logprobs=raw_logprobs,
                    # KL metrics are deliberately deferred to the trainer's
                    # normalized raw-score reduction.
                    metrics={"loss": 0.75},
                )

        return Pending()

    async def submit_optim_step(self, *, learning_rate, adam):
        if self.rollout_dir is not None:
            self.records_before_optim = list(iter_rollouts(self.rollout_dir))
        return await super().submit_optim_step(learning_rate=learning_rate, adam=adam)


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


def test_fused_opct_skips_student_rescoring_but_retains_behavior_and_teacher_scores():
    backend = FusedFakeBackend()
    trainer = OPCTTrainer(
        config=OPCTConfig(
            generation=OPCTGenerationConfig(rollouts_per_prompt=2),
            rollout_log="none",
        ),
        backend=backend,
    )
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = FakeTokenizer()
    trainer.sampling_client = FakeSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)
    batch = [(0, _pair(0))]
    pairs = trainer._prepare_batch(batch)
    samples = [
        [
            SampledSequence(tokens=[30, 31], logprobs=[-4.0, -5.0]),
            SampledSequence(tokens=[30, 31], logprobs=[-6.0, -7.0]),
        ]
    ]

    results = asyncio.run(
        trainer._build_batch_group(
            [batch],
            prepared_pair_groups=[pairs],
            sampled_groups=[samples],
        )
    )

    datums, _, _, _ = results[0]
    assert backend.student_score_calls == []
    assert len(backend.reference_calls) == 1
    assert len(datums) == 2
    for datum, expected_behavior in zip(datums, ([-4.0, -5.0], [-6.0, -7.0])):
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        assert datum.loss_fn_inputs["logprobs"].to_torch()[mask].tolist() == pytest.approx(
            expected_behavior
        )
        assert datum.loss_fn_inputs["opct_teacher_logprobs"].to_torch().tolist() == pytest.approx(
            [-1.2, -1.3]
        )


def test_fused_opct_completes_metrics_and_persists_provenance_before_optimizer(tmp_path):
    rollout_dir = tmp_path / "logs" / "itest" / "fused" / "rollouts"
    backend = FusedFakeBackend(rollout_dir=rollout_dir)
    config = OPCTConfig(
        experiment_name="itest",
        run_name="fused",
        model="fake-model",
        optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="constant"),
        generation=OPCTGenerationConfig(rollouts_per_prompt=1, temperature=0.4),
        batch_size=1,
        n_epochs=1,
        kl_coef=2.0,
        kl_discount_factor=0.5,
        checkpoint=CheckpointConfig(save_every_n_steps=10),
        log_base_dir=str(tmp_path / "logs"),
    )
    trainer = OPCTTrainer(config=config, backend=backend)
    trainer.setup_done = True
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = FakeTokenizer()
    trainer.sampling_client = FakeSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)

    with patch("ctm.training.opct.setup_logging") as setup_logging:
        logger = MagicMock()
        setup_logging.return_value = logger
        asyncio.run(trainer.train([_pair(0)]))

    assert backend.student_score_calls == []
    assert len(backend.reference_calls) == 1
    assert backend.fused_behavior_actions == [[-4.0, -5.0]]
    assert backend.loss_fns == ["importance_sampling"]
    assert backend.optim_steps == 1

    # raw student [-0.2, -0.3] - teacher [-1.2, -1.3] = [1, 1].
    # With coef=2 and discount=.5, advantages are [-3, -2].
    assert len(backend.records_before_optim) == 1
    assert backend.records_before_optim[0].reward == pytest.approx(-2.0)
    assert backend.records_before_optim[0].advantage == pytest.approx(-2.5)
    datum = backend.fb_datums[0][0]
    mask = datum.loss_fn_inputs["mask"].to_torch() > 0
    assert datum.loss_fn_inputs["advantages"].to_torch()[mask].tolist() == pytest.approx([-3.0, -2.0])

    logged_metrics = next(
        call.args[0]
        for call in logger.log_metrics.call_args_list
        if "train/teacher_kl" in call.args[0]
    )
    assert logged_metrics["train/teacher_kl"] == pytest.approx(1.0)
    assert logged_metrics["train/student_entropy"] == pytest.approx(0.25)
    assert logged_metrics["train/teacher_cross_entropy"] == pytest.approx(1.25)
    assert logged_metrics["train/teacher_scored_tokens"] == 2.0
    assert logged_metrics["train/loss"] == pytest.approx(0.75)


def test_opct_batches_generation_and_scoring_across_unchanged_policy_accumulation_group(tmp_path):
    backend = FakeBackend()
    backend.batch_sample_calls = []

    class NativeBatchSampler(FakeSampler):
        async def sample(self, *args, **kwargs):
            raise AssertionError("native batch sampler should not receive per-prompt generation calls")

        async def sample_batch(self, prompts, *, max_tokens, temperature, stop, num_samples):
            self.backend.batch_sample_calls.append(
                {
                    "prompts": [prompt.to_ints() for prompt in prompts],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": stop,
                    "num_samples": num_samples,
                }
            )
            return [[SampledSequence(tokens=[30, 31], logprobs=[-4.0, -5.0]) for _ in range(num_samples)] for _ in prompts]

    config = OPCTConfig(
        experiment_name="itest",
        run_name="opct-batched-generation",
        model="fake-model",
        lora=LoRAConfig(rank=4, seed=0),
        optimizer=AdamConfig(learning_rate=1e-4, lr_schedule="constant"),
        generation=OPCTGenerationConfig(rollouts_per_prompt=4, max_new_tokens=8, temperature=0.4),
        batch_size=1,
        gradient_accumulation_steps=16,
        n_epochs=1,
        checkpoint=CheckpointConfig(save_every_n_steps=10),
        log_base_dir=str(tmp_path / "logs"),
        rollout_log="none",
    )
    trainer = OPCTTrainer(config=config, backend=backend)
    trainer.setup_done = True
    trainer.renderer = FakeRenderer()
    trainer.tokenizer = MagicMock()
    trainer.sampling_client = NativeBatchSampler(backend)
    trainer.reference_policy = FakeReferencePolicy(backend)

    with patch("ctm.training.opct.setup_logging") as setup_logging:
        setup_logging.return_value = MagicMock()
        asyncio.run(trainer.train([_pair(index) for index in range(16)]))

    assert backend.batch_sample_calls == [
        {
            "prompts": [[20]] * 16,
            "max_tokens": 8,
            "temperature": 0.4,
            "stop": [],
            "num_samples": 4,
        }
    ]
    assert [len(datums) for datums in backend.fb_datums] == [4] * 16
    assert len(backend.student_score_calls) == 1
    assert len(backend.reference_calls) == 1
    # The physical scorer batch is observable in the fake transport: 16
    # prompts x 4 rollouts, while all 16 per-prompt reductions stay separate.
    assert len(backend.student_score_calls[0][1]) == 64
    assert len(backend.reference_calls[0][1]) == 64
    assert backend.optim_steps == backend.refreshes == 1


def test_opct_grouped_scoring_matches_legacy_per_microbatch_reductions():
    class LengthAwareSampler(FakeSampler):
        async def score_completions(self, prompts, completion_tokens):
            self.backend.student_score_calls.append(([prompt.to_ints() for prompt in prompts], [list(tokens) for tokens in completion_tokens]))
            return [[-0.2 - 0.01 * index for index, _ in enumerate(tokens)] for tokens in completion_tokens]

    class LengthAwareReference(FakeReferencePolicy):
        async def score_completions(self, prompts, completion_tokens):
            self.backend.reference_calls.append(([prompt.to_ints() for prompt in prompts], [list(tokens) for tokens in completion_tokens]))
            return [[-1.2 - 0.02 * index for index, _ in enumerate(tokens)] for tokens in completion_tokens]

    def make_trainer():
        backend = FakeBackend()
        trainer = OPCTTrainer(
            config=OPCTConfig(
                generation=OPCTGenerationConfig(rollouts_per_prompt=2),
                batch_size=1,
                gradient_accumulation_steps=2,
                kl_coef=1.7,
                kl_discount_factor=0.6,
                rollout_log="none",
            ),
            backend=backend,
        )
        trainer.renderer = FakeRenderer()
        trainer.tokenizer = FakeTokenizer()
        trainer.sampling_client = LengthAwareSampler(backend)
        trainer.reference_policy = LengthAwareReference(backend)
        return trainer, backend

    batches = [[(0, _pair(0))], [(1, _pair(1))]]

    def samples():
        return [
            [
                [
                    SampledSequence(tokens=[30, 31], logprobs=[-4.0, -5.0]),
                    SampledSequence(tokens=[], logprobs=[]),
                ]
            ],
            [
                [
                    SampledSequence(tokens=[32], logprobs=[-3.0]),
                    SampledSequence(tokens=[33, 34, 35], logprobs=[-2.0, -2.5, -3.0]),
                ]
            ],
        ]

    grouped, grouped_backend = make_trainer()
    grouped_pairs = [grouped._prepare_batch(batch) for batch in batches]
    grouped_results = asyncio.run(
        grouped._build_batch_group(
            batches,
            prepared_pair_groups=grouped_pairs,
            sampled_groups=samples(),
        )
    )

    legacy, legacy_backend = make_trainer()

    async def build_legacy():
        output = []
        for batch, batch_samples in zip(batches, samples()):
            datums, metrics, lengths = await legacy._build_batch(
                batch,
                prepared_pairs=legacy._prepare_batch(batch),
                sampled_groups=batch_samples,
            )
            output.append((datums, metrics, lengths, list(legacy._pending_rollout_meta)))
        return output

    legacy_results = asyncio.run(build_legacy())

    assert len(grouped_backend.student_score_calls) == len(grouped_backend.reference_calls) == 1
    assert len(grouped_backend.student_score_calls[0][1]) == 3
    assert len(legacy_backend.student_score_calls) == len(legacy_backend.reference_calls) == 2
    for grouped_result, legacy_result in zip(grouped_results, legacy_results):
        grouped_datums, grouped_metrics, grouped_lengths, grouped_meta = grouped_result
        legacy_datums, legacy_metrics, legacy_lengths, legacy_meta = legacy_result
        assert grouped_metrics == pytest.approx(legacy_metrics)
        assert grouped_lengths == legacy_lengths
        assert grouped_meta == legacy_meta
        assert len(grouped_datums) == len(legacy_datums)
        for grouped_datum, legacy_datum in zip(grouped_datums, legacy_datums):
            assert grouped_datum.model_input.to_ints() == legacy_datum.model_input.to_ints()
            for key in ("target_tokens", "logprobs", "advantages", "mask"):
                torch.testing.assert_close(
                    grouped_datum.loss_fn_inputs[key].to_torch(),
                    legacy_datum.loss_fn_inputs[key].to_torch(),
                )


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


def test_opct_cli_accepts_common_rollout_worker_flags(tmp_path, capsys, monkeypatch):
    from scripts.train_opct import main

    data = tmp_path / "pairs.jsonl"
    data.write_text(
        '{"reference_messages":[{"role":"user","content":"clean"}],"variant_messages":[{"role":"user","content":"variant"}]}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,3,5")

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
                "parallel-dry",
                "--backend",
                "local",
                "--local-device",
                "cuda:0",
                "--local-rollout-gpus",
                "1,2",
                "--dry-run",
            ]
        )

    assert not build_backend.called
    output = capsys.readouterr().out
    assert "Rollout workers: 2 (logical 1 -> 3, logical 2 -> 5)" in output
    assert "logs/unit/parallel-dry/rollout_workers" in output
