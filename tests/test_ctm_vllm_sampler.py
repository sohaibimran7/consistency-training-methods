"""Contract tests for the vLLM sampler — no vllm installed.

The vllm API surface (LLM.generate / SamplingParams / TokensPrompt / LoRARequest)
is faked; what's under test is OUR side of the contract: request construction,
LoRA hot-reload versioning, base-vs-policy routing, and token/logprob extraction.
Real-engine behaviour is validated on a GPU box (tests there carry @pytest.mark.gpu).
"""

import asyncio
import math
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tinker import types

from ctm.backends.local.engine import HAS_PEFT, LocalBackend
from ctm.backends.local.vllm_sampler import VLLMSampler
from ctm.core.config import LoRAConfig


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokensPrompt:
    def __init__(self, prompt_token_ids):
        self.prompt_token_ids = prompt_token_ids


class FakeLoRARequest:
    def __init__(self, name, lora_int_id, path):
        self.name, self.lora_int_id, self.path = name, lora_int_id, path


def fake_api():
    return SimpleNamespace(
        LLM=None, SamplingParams=FakeSamplingParams, TokensPrompt=FakeTokensPrompt, LoRARequest=FakeLoRARequest
    )


class FakeEngine:
    """Records generate() calls; returns two completions with per-token logprobs."""

    def __init__(self, drop_logprob_for_token=None):
        self.calls = []
        self.drop = drop_logprob_for_token

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls.append(SimpleNamespace(prompts=prompts, params=params, lora_request=lora_request))

        def lp_dict(token, value):
            if token == self.drop:
                return {}  # sampled token missing its own logprob
            return {token: SimpleNamespace(logprob=value)}

        completions = [
            SimpleNamespace(
                token_ids=[7, 8],
                logprobs=[lp_dict(7, -0.5), lp_dict(8, -0.7)],
                finish_reason="stop",
                stop_reason=8,
            ),
            SimpleNamespace(
                token_ids=[9],
                logprobs=[lp_dict(9, -1.2)],
                finish_reason="stop",
                stop_reason=9,
            ),
        ]
        return [SimpleNamespace(outputs=completions) for _ in prompts]


class FakeScoringEngine:
    """Returns one prompt logprob mapping per supplied token position."""

    def __init__(self, *, malformed=None):
        self.calls = []
        self.malformed = malformed

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls.append(SimpleNamespace(prompts=prompts, params=params, lora_request=lora_request))
        outputs = []
        for prompt in prompts:
            tokens = list(prompt.prompt_token_ids)
            prompt_logprobs = [
                {token: SimpleNamespace(logprob=-(position + token / 100.0))} for position, token in enumerate(tokens)
            ]
            returned_tokens = list(tokens)
            if self.malformed == "tokens":
                returned_tokens[-1] += 1
            elif self.malformed == "missing_prompt_logprobs":
                prompt_logprobs = None
            elif self.malformed == "length":
                prompt_logprobs = prompt_logprobs[:-1]
            elif self.malformed == "missing_token":
                prompt_logprobs[-1] = {}
            elif self.malformed == "nonfinite":
                prompt_logprobs[-1][tokens[-1]].logprob = math.inf
            outputs.append(
                SimpleNamespace(
                    outputs=(
                        [SimpleNamespace(token_ids=[200001], logprobs=None, finish_reason="stop", stop_reason=200001)]
                        if params.kwargs.get("max_tokens") is None
                        else []
                    ),
                    prompt_token_ids=returned_tokens,
                    prompt_logprobs=prompt_logprobs,
                )
            )
        return outputs


def make_sampler(engine=None):
    return VLLMSampler(model="some/base", engine=engine or FakeEngine(), api=fake_api())


class TestVLLMSampler:
    def test_qwen35_lora_sampling_fails_before_a_noop_policy_can_be_used(self):
        sampler = VLLMSampler(model="Qwen/Qwen3.5-9B", engine=FakeEngine(), api=fake_api())
        with pytest.raises(ValueError, match="maps no `model.layers"):
            sampler.advance_policy("/tmp/qwen35-adapter")

    def test_qwen35_base_only_sampling_is_not_blocked(self):
        sampler = VLLMSampler(model="Qwen/Qwen3.5-9B", engine=FakeEngine(), api=fake_api())
        sampler.sample([1], max_tokens=1, temperature=0.0, stop=[], num_samples=1, use_base=True)

    def test_qwen35_policy_sampling_requires_a_published_compatibility_snapshot(self):
        sampler = VLLMSampler(model="Qwen/Qwen3.5-9B", engine=FakeEngine(), api=fake_api())
        with pytest.raises(RuntimeError, match="refusing to silently sample the frozen base"):
            sampler.sample([1], max_tokens=1, temperature=0.7, stop=[], num_samples=1, use_base=False)

    def test_engine_is_constructed_with_processed_logprobs_mode_pinned(self):
        constructed = []

        class RecordingLLM:
            def __init__(self, **kwargs):
                constructed.append(kwargs)

        api = fake_api()
        api.LLM = RecordingLLM
        sampler = VLLMSampler(model="some/base", api=api, gpu_memory_utilization=0.8)

        assert constructed == [
            {
                "model": "some/base",
                "enable_lora": True,
                "gpu_memory_utilization": 0.8,
                "max_lora_rank": 64,
                "logprobs_mode": "processed_logprobs",
            }
        ]
        sampler.shutdown()

    @pytest.mark.parametrize("mode", ["raw_logprobs", "raw_logits", None])
    def test_incompatible_logprobs_mode_fails_closed(self, mode):
        with pytest.raises(ValueError, match="requires logprobs_mode='processed_logprobs'"):
            VLLMSampler(
                model="some/base",
                engine=FakeEngine(),
                api=fake_api(),
                logprobs_mode=mode,
            )

    def test_policy_before_any_snapshot_uses_base(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.sample([1, 2], max_tokens=8, temperature=0.7, stop=[], num_samples=2, use_base=False)
        assert engine.calls[0].lora_request is None  # no adapter published yet

    def test_advance_policy_bumps_lora_request_id(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.advance_policy("/tmp/adapters/v1")
        s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False)
        s.advance_policy("/tmp/adapters/v2")
        s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False)
        first, second = engine.calls[0].lora_request, engine.calls[1].lora_request
        assert (first.lora_int_id, first.path) == (1, "/tmp/adapters/v1")
        assert (second.lora_int_id, second.path) == (2, "/tmp/adapters/v2")
        assert first.name != second.name  # unique id+name defeats vLLM's adapter cache

    def test_explicit_distributed_policy_version_is_used_and_must_increase(self):
        engine = FakeEngine()
        sampler = make_sampler(engine)
        sampler.advance_policy("/tmp/adapters/v7", version=7)
        sampler.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=False)

        request = engine.calls[0].lora_request
        assert (request.lora_int_id, request.path) == (7, "/tmp/adapters/v7")
        with pytest.raises(ValueError, match="must increase"):
            sampler.advance_policy("/tmp/adapters/stale", version=7)

    def test_base_sampling_never_attaches_adapter(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.advance_policy("/tmp/adapters/v1")
        s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=1, use_base=True)
        assert engine.calls[0].lora_request is None

    def test_request_params_and_prompt(self):
        engine = FakeEngine()
        s = make_sampler(engine)
        s.sample([5, 6, 7], max_tokens=32, temperature=0.7, stop=[2, "text-stop", 3], num_samples=4, use_base=False)
        call = engine.calls[0]
        assert call.prompts[0].prompt_token_ids == [5, 6, 7]
        assert call.params.kwargs["n"] == 4
        assert call.params.kwargs["max_tokens"] == 32
        assert call.params.kwargs["temperature"] == 0.7
        assert call.params.kwargs["stop_token_ids"] == [2, 3]  # non-int stops filtered
        assert call.params.kwargs["ignore_eos"] is False
        assert call.params.kwargs["logprobs"] == 0

    def test_no_cap_request_preserves_none_and_requires_stop_termination(self):
        engine = FakeEngine()
        sampler = make_sampler(engine)

        sequences = sampler.sample(
            [5, 6],
            max_tokens=None,
            temperature=1.0,
            stop=[200001, 200008],
            num_samples=2,
            use_base=True,
        )

        assert engine.calls[0].params.kwargs["max_tokens"] is None
        assert [sequence.tokens for sequence in sequences] == [[7, 8], [9]]

    def test_no_cap_request_rejects_context_length_termination(self):
        engine = FakeEngine()
        original_generate = engine.generate

        def length_terminated(*args, **kwargs):
            outputs = original_generate(*args, **kwargs)
            outputs[0].outputs[0].finish_reason = "length"
            outputs[0].outputs[0].stop_reason = None
            return outputs

        engine.generate = length_terminated
        sampler = make_sampler(engine)

        with pytest.raises(RuntimeError, match="did not terminate through an EOS/stop token"):
            sampler.sample(
                [5],
                max_tokens=None,
                temperature=1.0,
                stop=[200001, 200008],
                num_samples=1,
                use_base=True,
            )

    def test_ignore_eos_is_explicit_opt_in(self):
        engine = FakeEngine()
        sampler = make_sampler(engine)
        sampler.sample(
            [5],
            max_tokens=32,
            temperature=0.7,
            stop=[],
            num_samples=1,
            use_base=False,
            ignore_eos=True,
        )
        assert engine.calls[0].params.kwargs["ignore_eos"] is True

    def test_token_and_logprob_extraction(self):
        s = make_sampler()
        seqs = s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=2, use_base=False)
        assert [q.tokens for q in seqs] == [[7, 8], [9]]
        assert seqs[0].logprobs == pytest.approx([-0.5, -0.7])
        assert seqs[1].logprobs == pytest.approx([-1.2])

    def test_batch_uses_one_engine_call_and_preserves_prompt_order(self):
        engine = FakeEngine()
        sampler = make_sampler(engine)

        batches = sampler.sample_batch(
            [[1], [2, 3]],
            max_tokens=4,
            temperature=1.0,
            stop=[],
            num_samples=2,
            use_base=False,
        )

        assert len(engine.calls) == 1
        assert [prompt.prompt_token_ids for prompt in engine.calls[0].prompts] == [[1], [2, 3]]
        assert [[sequence.tokens for sequence in batch] for batch in batches] == [
            [[7, 8], [9]],
            [[7, 8], [9]],
        ]

    def test_completion_scoring_uses_prompt_logprobs_and_exact_combined_tokens(self):
        engine = FakeScoringEngine()
        sampler = make_sampler(engine)
        sampler.advance_policy("/tmp/adapters/v3", version=3)

        scores = sampler.score_completions(
            [[10, 11], [20]],
            [[7, 8], [9, 10, 11]],
            use_base=False,
        )

        call = engine.calls[0]
        assert [prompt.prompt_token_ids for prompt in call.prompts] == [
            [10, 11, 7, 8],
            [20, 9, 10, 11],
        ]
        assert call.params.kwargs == {
            "n": 1,
            "max_tokens": 1,
            "temperature": 0.0,
            "prompt_logprobs": 0,
        }
        assert call.lora_request.lora_int_id == 3
        assert scores[0] == pytest.approx([-2.07, -3.08])
        assert scores[1] == pytest.approx([-1.09, -2.1, -3.11])

        sampler.score_completions([[10]], [[7]], use_base=True)
        assert engine.calls[1].lora_request is None

    def test_uncapped_parity_scorer_preserves_none_and_requires_eos(self):
        engine = FakeScoringEngine()
        sampler = make_sampler(engine)

        scores = sampler.score_completions_uncapped_eos_tail(
            [[10, 11]],
            [[7, 8]],
            use_base=True,
        )

        assert scores[0] == pytest.approx([-2.07, -3.08])
        assert engine.calls[0].params.kwargs["max_tokens"] is None

    def test_uncapped_parity_scorer_rejects_non_eos_tail(self):
        engine = FakeScoringEngine()
        original_generate = engine.generate

        def length_terminated(*args, **kwargs):
            outputs = original_generate(*args, **kwargs)
            outputs[0].outputs[0].finish_reason = "length"
            outputs[0].outputs[0].stop_reason = None
            return outputs

        engine.generate = length_terminated
        sampler = make_sampler(engine)

        with pytest.raises(RuntimeError, match="did not terminate through an EOS/stop token"):
            sampler.score_completions_uncapped_eos_tail([[10]], [[7]], use_base=True)

    @pytest.mark.parametrize(
        ("malformed", "message"),
        [
            ("tokens", "misaligned"),
            ("missing_prompt_logprobs", "omitted prompt_logprobs"),
            ("length", "prompt-logprob position"),
            ("missing_token", "omitted token"),
            ("nonfinite", "non-finite logprob"),
        ],
    )
    def test_completion_scoring_rejects_malformed_or_nonfinite_engine_results(self, malformed, message):
        sampler = make_sampler(FakeScoringEngine(malformed=malformed))
        with pytest.raises(RuntimeError, match=message):
            sampler.score_completions([[1, 2]], [[3]], use_base=True)

    def test_completion_scoring_validates_batch_and_nonempty_sequences(self):
        sampler = make_sampler(FakeScoringEngine())
        with pytest.raises(ValueError, match="same length"):
            sampler.score_completions([[1]], [], use_base=True)
        with pytest.raises(ValueError, match="prompt 0 is empty"):
            sampler.score_completions([[]], [[1]], use_base=True)
        with pytest.raises(ValueError, match="completion 0 is empty"):
            sampler.score_completions([[1]], [[]], use_base=True)

        assert sampler.score_completions([], [], use_base=True) == []
        assert sampler.engine.calls == []

    def test_missing_sampled_logprob_marks_sequence_logprobless(self):
        s = make_sampler(FakeEngine(drop_logprob_for_token=8))
        seqs = s.sample([1], max_tokens=4, temperature=1.0, stop=[], num_samples=2, use_base=False)
        assert seqs[0].logprobs is None  # token 8's logprob missing → whole seq excluded downstream
        assert seqs[1].logprobs == pytest.approx([-1.2])

    def test_shutdown_releases_engine_and_is_idempotent(self):
        engine = FakeEngine()
        engine_ref = weakref.ref(engine)
        sampler = make_sampler(engine)
        del engine
        sampler.shutdown()
        assert sampler.engine is None
        assert engine_ref() is None

        sampler.shutdown()
        assert sampler.engine is None


@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
class TestLocalBackendVLLMWiring:
    def _backend(self, engine):
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=1))
        backend = LocalBackend(
            device="cpu",
            use_lora=True,
            model_instance=model,
            sampler="vllm",
            vllm_options={"engine": engine, "api": fake_api()},
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=2, seed=0))
        return backend

    def test_setup_defers_engine_boot(self):
        # SFT-family runs call setup() but never request a sampler — no engine
        # boot, no adapter snapshot, no vllm package requirement. No fake
        # engine/api is injected here: a boot inside setup() would hit vllm.
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=1, n_head=1))
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=model, sampler="vllm")
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=2, seed=0))
        assert backend._vllm is None
        assert backend._adapter_scratch is None

    def test_first_sampler_request_boots_and_publishes_adapter(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        backend.policy_sampler("p")
        assert backend._vllm.adapter_version == 1
        adapter_dir = Path(backend._vllm.adapter_dir)
        assert adapter_dir.exists() and any(adapter_dir.iterdir())  # real peft snapshot on disk

    def test_refresh_on_cold_engine_boots_and_publishes_once(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        asyncio.run(backend.refresh_policy_sampler("step1"))
        assert backend._vllm.adapter_version == 1  # lazy boot's publish, no double snapshot

    def test_refresh_snapshots_and_bumps_version(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        backend.policy_sampler("p")  # boot (v1), as the RL loop does at setup
        asyncio.run(backend.refresh_policy_sampler("step1"))
        assert backend._vllm.adapter_version == 2
        assert backend._vllm.adapter_dir.endswith("v2")

    def test_sampling_routes_through_vllm(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        policy = backend.policy_sampler("p")
        seqs = asyncio.run(
            policy.sample(
                types.ModelInput.from_ints(tokens=[1, 2]), max_tokens=4, temperature=1.0, stop=[], num_samples=2
            )
        )
        assert [q.tokens for q in seqs] == [[7, 8], [9]]
        assert engine.calls[-1].lora_request.lora_int_id == 1  # policy = current adapter

        base = backend.base_sampler()
        asyncio.run(
            base.sample(
                types.ModelInput.from_ints(tokens=[1, 2]), max_tokens=4, temperature=1.0, stop=[], num_samples=1
            )
        )
        assert engine.calls[-1].lora_request is None  # base = engine's frozen weights

    def test_shutdown_releases_started_vllm(self):
        engine = FakeEngine()
        backend = self._backend(engine)
        backend.policy_sampler("p")
        sampler = backend._vllm

        backend.shutdown()
        assert backend._vllm is None
        assert sampler.engine is None

        backend.shutdown()
        assert backend._vllm is None

    def test_vllm_requires_lora(self):
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            sampler="vllm",
            model_instance=torch.nn.Linear(2, 2),
            vllm_options={"engine": FakeEngine(), "api": fake_api()},
        )
        with pytest.raises(NotImplementedError):
            backend.setup(model="x", lora=LoRAConfig(rank=2))
