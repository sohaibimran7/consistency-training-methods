"""CPU tests for the LocalBackend (torch/transformers engine) with a tiny random GPT-2.

No downloads, no GPU, no peft: the model is randomly initialized from a config,
and LoRA-specific paths are exercised only when peft is installed (skip otherwise).
Uses full fine-tuning (use_lora=False) so the offline suite always covers the
loss/optimizer/sampler/checkpoint machinery.
"""

import asyncio

import pytest
import torch
from tinker import types
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from ctm.backends.local.engine import HAS_PEFT, LocalBackend, _lora_target_module_names
from ctm.core.config import AdamConfig, LoRAConfig

VOCAB = 128


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
        )
    )


def make_backend() -> LocalBackend:
    backend = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
    backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
    return backend


def sft_datum(tokens=(5, 6, 7, 8, 9), n_prompt=2):
    # weights align with the FULL token sequence; the helper drops weights[0] when
    # left-shifting targets, so pass len(tokens) weights.
    weights = torch.tensor([0.0] * n_prompt + [1.0] * (len(tokens) - n_prompt))
    return datum_from_model_input_weights(types.ModelInput.from_ints(tokens=list(tokens)), weights)


def rl_datum(backend, prompt=(5, 6), action=(7, 8), advantage=1.0):
    """RL datum whose sampled logprobs are the model's OWN current logprobs → ratio starts at 1."""
    seq = list(prompt) + list(action)
    probe = datum_from_model_input_weights(types.ModelInput.from_ints(tokens=list(seq)), torch.ones(len(seq)))
    with torch.no_grad():
        lp = backend._target_logprobs([probe])[0]
    action_logprobs = lp[len(prompt) - 1 :].tolist()  # logprobs of the action tokens
    transition = Transition(
        ob=types.ModelInput.from_ints(tokens=list(prompt)),
        ac=TokensWithLogprobs(tokens=list(action), maybe_logprobs=action_logprobs),
        reward=0.0,
        episode_done=True,
    )
    traj = Trajectory(transitions=[transition], final_ob=types.ModelInput.from_ints(tokens=[]))
    return trajectory_to_data(traj, traj_advantage=advantage)[0]


async def step(backend, datums, loss_fn, lr=1e-2):
    pending = await backend.submit_forward_backward(datums, loss_fn)
    out = await pending.result()
    opt = await backend.submit_optim_step(learning_rate=lr, adam=AdamConfig(learning_rate=lr))
    await opt.result()
    return out


class TestLocalSFT:
    def test_cross_entropy_loss_decreases(self):
        backend = make_backend()
        datums = [sft_datum(), sft_datum(tokens=(9, 8, 7, 6, 5))]

        async def run():
            first = await step(backend, datums, "cross_entropy")
            for _ in range(25):
                last = await step(backend, datums, "cross_entropy")
            return first, last

        first, last = asyncio.run(run())
        assert last.metrics["loss"] < first.metrics["loss"] * 0.7
        # per-datum logprob tensors match target lengths
        for out_lp, d in zip(first.logprobs, datums):
            assert out_lp.shape == d.loss_fn_inputs["target_tokens"].to_torch().shape

    def test_gradient_accumulation_two_phase(self):
        backend = make_backend()

        async def run():
            # two forward_backwards accumulate, one optim_step applies + zeroes
            await (await backend.submit_forward_backward([sft_datum()], "cross_entropy")).result()
            await (await backend.submit_forward_backward([sft_datum()], "cross_entropy")).result()
            params = [p for p in backend.model.parameters() if p.requires_grad]
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            await (await backend.submit_optim_step(learning_rate=1e-3, adam=AdamConfig())).result()
            assert all(p.grad is None for p in params)  # zero_grad(set_to_none=True)

        asyncio.run(run())


class TestLocalRL:
    @pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo"])
    def test_positive_advantage_increases_action_logprob(self, loss_fn):
        backend = make_backend()
        datum = rl_datum(backend, advantage=1.0)
        targets = datum.loss_fn_inputs["target_tokens"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch()

        def action_logprob():
            with torch.no_grad():
                lp = backend._target_logprobs([datum])[0]
            return float((lp * mask).sum())

        before = action_logprob()

        async def run():
            for _ in range(10):
                await step(backend, [datum], loss_fn, lr=5e-3)

        asyncio.run(run())
        assert action_logprob() > before
        assert targets.shape == mask.shape  # sanity: datum structure as expected

    def test_kl_penalty_requires_lora(self):
        backend = make_backend()  # full fine-tune → no frozen base available
        datum = rl_datum(backend)
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.incorporate_kl_penalty([datum], kl_coef=0.1, kl_discount_factor=0.0))
        with pytest.raises(NotImplementedError):
            backend.base_sampler()


class TestLocalSampler:
    def test_sample_shapes_and_logprobs(self):
        backend = make_backend()
        handle = backend.policy_sampler("test")
        seqs = asyncio.run(
            handle.sample(
                types.ModelInput.from_ints(tokens=[5, 6, 7]),
                max_tokens=6,
                temperature=1.0,
                stop=[],
                num_samples=3,
            )
        )
        assert len(seqs) == 3
        for s in seqs:
            assert 1 <= len(s.tokens) <= 6
            assert len(s.logprobs) == len(s.tokens)
            assert all(lp <= 0.0 for lp in s.logprobs)
            assert all(0 <= t < VOCAB for t in s.tokens)

    def test_stop_token_truncates(self):
        backend = make_backend()
        handle = backend.policy_sampler("test")
        stop = list(range(VOCAB))  # every token is a stop token → length-1 completions
        seqs = asyncio.run(
            handle.sample(
                types.ModelInput.from_ints(tokens=[5, 6]),
                max_tokens=8,
                temperature=1.0,
                stop=stop,
                num_samples=2,
            )
        )
        assert all(len(s.tokens) == 1 for s in seqs)

    def test_refresh_returns_live_policy(self):
        backend = make_backend()
        h = asyncio.run(backend.refresh_policy_sampler("x"))
        assert isinstance(h, type(backend.policy_sampler("x")))


class TestLocalCheckpoint:
    def test_save_and_resume_full_finetune(self, tmp_path):
        backend = make_backend()
        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        paths = asyncio.run(
            backend.save_checkpoint(name="run1_step1", log_dir=tmp_path, loop_state={"step": 1}, kind="both")
        )

        ckpt = tmp_path / "checkpoints" / "run1_step1"
        assert paths["sampler_path"] == f"file://{ckpt.resolve()}"
        assert paths["state_path"] is not None
        assert (ckpt / "weights.pt").exists()
        assert (ckpt / "optimizer.pt").exists()
        assert (ckpt / "manifest.json").exists()

        # A fresh backend (different random init) converges to the saved weights on load.
        torch.manual_seed(123)
        other = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
        other.setup(
            model="tiny-gpt2-test",
            lora=LoRAConfig(rank=4),
            resume_from=paths["sampler_path"],
            resume_with_optimizer=True,
        )
        for p1, p2 in zip(backend.model.state_dict().values(), other.model.state_dict().values()):
            assert torch.equal(p1, p2)

    def test_sampler_kind_skips_optimizer(self, tmp_path):
        backend = make_backend()
        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        paths = asyncio.run(backend.save_checkpoint(name="run1", log_dir=tmp_path, loop_state={}, kind="sampler"))
        assert paths["state_path"] is None
        assert not (tmp_path / "checkpoints" / "run1" / "optimizer.pt").exists()


@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
class TestLocalLoRA:
    def test_portable_component_flags_select_expected_modules(self):
        model = tiny_model()
        attention = _lora_target_module_names(
            model,
            LoRAConfig(train_mlp=False, train_attn=True, train_unembed=False),
        )
        mlp = _lora_target_module_names(
            model,
            LoRAConfig(train_mlp=True, train_attn=False, train_unembed=False),
        )
        unembed = _lora_target_module_names(
            model,
            LoRAConfig(train_mlp=False, train_attn=False, train_unembed=True),
        )

        assert attention and all("attn" in name for name in attention)
        assert mlp and all("attn" not in name and name != "lm_head" for name in mlp)
        assert unembed == ["lm_head"]

    def test_lora_wraps_and_trains(self):
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=tiny_model())
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4, seed=0))
        n_trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in backend.model.parameters())
        assert 0 < n_trainable < n_total * 0.5  # adapter-only training

        first = asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        assert first.metrics["loss"] > 0
