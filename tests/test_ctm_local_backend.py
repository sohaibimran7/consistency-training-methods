"""CPU tests for the LocalBackend (torch/transformers engine) with a tiny random GPT-2.

No downloads, no GPU, no peft: the model is randomly initialized from a config,
and LoRA-specific paths are exercised only when peft is installed (skip otherwise).
Uses full fine-tuning (use_lora=False) so the offline suite always covers the
loss/optimizer/sampler/checkpoint machinery.
"""

import asyncio
import copy
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from tinker import types
from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from ctm.backends.local import losses as local_losses
from ctm.backends.local.engine import (
    HAS_PEFT,
    FusedExpertLoRAWarning,
    LocalBackend,
    LocalSamplerHandle,
    _lora_target_module_names,
    _lora_target_parameter_names,
    _selected_hidden_logprobs,
    _selected_hidden_raw_and_temperature_logprobs,
    _selected_target_logprobs,
    _selected_token_components,
)
from ctm.core.config import AdamConfig, LoRAConfig

VOCAB = 128


class DeterministicTinyCausalLM(torch.nn.Module):
    """Small dropout-free model used for exact microbatch comparisons."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 12):
        super().__init__()
        torch.manual_seed(17)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, vocab_size)
        self.forward_calls = 0

    def forward(self, *, input_ids, attention_mask=None, use_cache=None):
        assert use_cache is False
        self.forward_calls += 1
        hidden = self.embedding(input_ids)
        positions = torch.arange(1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype)
        hidden = hidden.cumsum(dim=1) / positions[None, :, None]
        return SimpleNamespace(logits=self.projection(hidden))


def deterministic_backend(
    *,
    max_datums: int | None,
    max_tokens: int | None,
    keep_frozen_base: bool = False,
) -> LocalBackend:
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=DeterministicTinyCausalLM(),
        keep_frozen_base=keep_frozen_base,
        forward_microbatch_max_datums=max_datums,
        forward_microbatch_max_tokens=max_tokens,
    )
    backend.setup(model="deterministic-tiny", lora=LoRAConfig(rank=4))
    return backend


def test_selected_target_logprobs_matches_dense_value_and_gradient():
    torch.manual_seed(3)
    targets = torch.randint(0, 19, (11,))
    chunked_logits = torch.randn(11, 19, requires_grad=True)
    dense_logits = chunked_logits.detach().clone().requires_grad_(True)

    chunked = _selected_target_logprobs(chunked_logits, targets, chunk_size=3)
    dense = torch.log_softmax(dense_logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(chunked, dense)

    weights = torch.linspace(0.1, 1.1, 11)
    (chunked * weights).sum().backward()
    (dense * weights).sum().backward()
    torch.testing.assert_close(chunked_logits.grad, dense_logits.grad)


def test_selected_hidden_raw_and_temperature_scores_share_one_exact_head_pass():
    torch.manual_seed(31)
    hidden = torch.randn(13, 7, requires_grad=True)
    dense_hidden = hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 23, (13,))
    head = torch.nn.Linear(7, 23, bias=False)
    dense_head = copy.deepcopy(head)

    raw, behavior = _selected_hidden_raw_and_temperature_logprobs(
        hidden,
        targets,
        head,
        temperature=0.7,
        chunk_size=4,
    )
    dense_logits = dense_head(dense_hidden).float()
    expected_raw = torch.log_softmax(dense_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    expected_behavior = torch.log_softmax(dense_logits / 0.7, dim=-1).gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(raw, expected_raw)
    torch.testing.assert_close(behavior, expected_behavior)

    (raw.sum() + 0.3 * behavior.sum()).backward()
    (expected_raw.sum() + 0.3 * expected_behavior.sum()).backward()
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(head.weight.grad, dense_head.weight.grad)


def test_selected_hidden_logprobs_bounds_head_calls_and_matches_dense_gradient():
    torch.manual_seed(5)
    chunked_head = torch.nn.Linear(7, 19)
    dense_head = copy.deepcopy(chunked_head)
    chunked_hidden = torch.randn(11, 7, requires_grad=True)
    dense_hidden = chunked_hidden.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 19, (11,))
    head_chunk_sizes = []
    hook = chunked_head.register_forward_pre_hook(lambda _module, args: head_chunk_sizes.append(args[0].shape[0]))

    chunked = _selected_hidden_logprobs(chunked_hidden, targets, chunked_head, chunk_size=3)
    dense_logits = dense_head(dense_hidden)
    dense = torch.log_softmax(dense_logits.float(), dim=-1).gather(1, targets[:, None]).squeeze(1)
    torch.testing.assert_close(chunked, dense)

    weights = torch.linspace(0.1, 1.1, 11)
    (chunked * weights).sum().backward()
    (dense * weights).sum().backward()
    hook.remove()
    assert head_chunk_sizes and max(head_chunk_sizes) <= 3
    torch.testing.assert_close(chunked_hidden.grad, dense_hidden.grad)
    for chunked_parameter, dense_parameter in zip(chunked_head.parameters(), dense_head.parameters()):
        torch.testing.assert_close(chunked_parameter.grad, dense_parameter.grad)


def test_selected_hidden_logprobs_applies_muse_multiplier_before_softcap():
    torch.manual_seed(29)
    hidden = torch.randn(9, 5)
    targets = torch.randint(0, 17, (9,))
    head = torch.nn.Linear(5, 17, bias=False)

    actual = _selected_hidden_logprobs(
        hidden,
        targets,
        head,
        chunk_size=3,
        output_multiplier=0.19611613513818404,
        final_logit_softcapping=20.0,
        checkpoint_chunks=False,
    )
    dense_logits = head(hidden).float() * 0.19611613513818404
    dense_logits = 20.0 * torch.tanh(dense_logits / 20.0)
    expected = torch.log_softmax(dense_logits, dim=-1).gather(1, targets[:, None]).squeeze(1)

    torch.testing.assert_close(actual, expected)


def test_muse_text_only_detaches_vision_modules_but_keeps_language_path():
    class FakeMuse(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="muse_glimmer")
            self.model = torch.nn.Module()
            self.model.language_model = torch.nn.Linear(3, 3)
            self.model.vision_tower = torch.nn.Linear(3, 3)
            self.model.vision_adapter = torch.nn.Linear(3, 3)
            self.model.vision_projection = torch.nn.Linear(3, 3)
            self.model.perception_emb_norm = torch.nn.LayerNorm(3)

    model = FakeMuse()
    language_model = model.model.language_model
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=model,
        hf_language_model_only=True,
    )
    backend._detach_unused_multimodal_modules()

    assert model.model.language_model is language_model
    assert model.model.vision_tower is None
    assert model.model.vision_adapter is None
    assert model.model.vision_projection is None
    assert model.model.perception_emb_norm is None


def test_gemma4_unified_reference_uses_full_image_text_loader(monkeypatch):
    import transformers

    class FakeGemma4Unified(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(model_type="gemma4_unified")
            self.embedding = torch.nn.Embedding(VOCAB, 4)
            self.projection = torch.nn.Linear(4, VOCAB)

        def get_input_embeddings(self):
            return self.embedding

    class ImageTextLoader:
        calls: ClassVar[list[tuple[str, dict]]] = []

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            cls.calls.append((model_name, kwargs))
            return FakeGemma4Unified()

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda model_name: SimpleNamespace(model_type="gemma4_unified"),
    )
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", ImageTextLoader, raising=False)
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("Gemma 4 must not use the text-only CausalLM loader"),
    )

    backend = LocalBackend(device="cpu", use_lora=False)
    backend.setup(model="google/gemma-4-12B-it", lora=LoRAConfig(rank=4))

    assert isinstance(backend.model, FakeGemma4Unified)
    assert len(ImageTextLoader.calls) == 1
    model_name, load_kwargs = ImageTextLoader.calls[0]
    assert model_name == "google/gemma-4-12B-it"
    assert load_kwargs["torch_dtype"] == torch.float32
    assert load_kwargs["config"].model_type == "gemma4_unified"


@pytest.mark.parametrize(
    ("logits_shape", "targets_shape", "chunk_size"),
    [((2, 3, 4), (2,), 2), ((2, 4), (3,), 2), ((2, 4), (2,), 0)],
)
def test_selected_target_logprobs_rejects_invalid_shapes(logits_shape, targets_shape, chunk_size):
    with pytest.raises(ValueError):
        _selected_target_logprobs(torch.randn(logits_shape), torch.zeros(targets_shape, dtype=torch.long), chunk_size=chunk_size)


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


def deterministic_tiny_hf_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(23)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=VOCAB,
            n_positions=64,
            n_embd=24,
            n_layer=2,
            n_head=2,
            attn_pdrop=0.0,
            embd_pdrop=0.0,
            resid_pdrop=0.0,
        )
    )


def checkpointable_layer_model(layer_count: int = 4):
    class CheckpointLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.gradient_checkpointing = False

    class CheckpointBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(CheckpointLayer() for _ in range(layer_count))

    class CheckpointCausalLM(torch.nn.Module):
        base_model_prefix = "model"

        def __init__(self):
            super().__init__()
            self.model = CheckpointBackbone()
            self.config = SimpleNamespace(use_cache=True)
            self.gradient_checkpointing_kwargs = None

        @property
        def is_gradient_checkpointing(self):
            return any(layer.gradient_checkpointing for layer in self.model.layers)

        def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs
            for layer in self.model.layers:
                layer.gradient_checkpointing = True

    return CheckpointCausalLM()


def dense_datum_logprobs(model, datums):
    token_lists = [datum.model_input.to_ints() for datum in datums]
    max_len = max(len(tokens) for tokens in token_lists)
    input_ids = torch.zeros((len(datums), max_len), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for index, tokens in enumerate(token_lists):
        input_ids[index, : len(tokens)] = torch.tensor(tokens)
        attention_mask[index, : len(tokens)] = 1
    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return [torch.log_softmax(logits[index, : len(tokens)].float(), dim=-1).gather(1, datum.loss_fn_inputs["target_tokens"].to_torch().long()[:, None]).squeeze(1) for index, (tokens, datum) in enumerate(zip(token_lists, datums))]


def make_backend() -> LocalBackend:
    backend = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
    backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
    return backend


def test_local_hf_eos_only_generation_has_no_length_stopping_path(monkeypatch):
    backend = make_backend()

    def emit_eos(probabilities, *, num_samples):
        assert num_samples == 1
        return torch.full(
            (probabilities.shape[0], 1),
            3,
            dtype=torch.long,
            device=probabilities.device,
        )

    monkeypatch.setattr(torch, "multinomial", emit_eos)
    groups = backend._generate_batch(
        prompt_tokens_batch=[[1, 2], [4]],
        max_tokens=None,
        temperature=1.0,
        stop=[3],
        num_samples=2,
        use_base=False,
    )

    assert [[sequence.tokens for sequence in group] for group in groups] == [
        [[3], [3]],
        [[3], [3]],
    ]
    assert all(
        len(sequence.logprobs or []) == 1
        for group in groups
        for sequence in group
    )


def test_streaming_capped_sampling_stops_at_eos_or_exact_output_limit_without_retaining_scores(monkeypatch):
    backend = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model(), hf_streaming_sampling=True)
    backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
    emitted = iter(([3, 5, 6, 7], [5, 3, 6, 7], [5, 5, 6, 3], [5, 5, 6, 5]))

    def choose_tokens(probabilities, *, num_samples):
        assert num_samples == 1
        return torch.tensor(next(emitted), dtype=torch.long)[:, None]

    def forbidden_generate(**kwargs):
        raise AssertionError("streaming sampling must not retain model.generate output_scores")

    monkeypatch.setattr(torch, "multinomial", choose_tokens)
    monkeypatch.setattr(backend.model, "generate", forbidden_generate)
    groups = backend._generate_batch(prompt_tokens_batch=[[1, 2], [4]], max_tokens=4,
                                     temperature=1.0, stop=[3], num_samples=2, use_base=False)
    assert [[s.tokens for s in group] for group in groups] == [[[3], [5, 3]], [[6, 6, 6, 6], [7, 7, 3]]]
    assert all(len(s.tokens) == len(s.logprobs) for group in groups for s in group)
    assert next(emitted, None) is None


def test_device_map_constructor_guards():
    with pytest.raises(ValueError, match="max_memory requires device_map"):
        LocalBackend(max_memory={0: "45GiB"})
    with pytest.raises(ValueError, match="applies only when LocalBackend loads"):
        LocalBackend(model_instance=tiny_model(), device_map="auto")


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, float("nan"), float("inf"), True])
def test_local_backend_rejects_invalid_ppo_clip_epsilon(value):
    with pytest.raises(ValueError, match="ppo_clip_epsilon must be a finite number in \\(0, 1\\)"):
        LocalBackend(model_instance=DeterministicTinyCausalLM(), ppo_clip_epsilon=value)


def test_gradient_checkpointing_activates_and_preserves_loss_and_gradients():
    plain_model = deterministic_tiny_hf_model()
    checkpointed_model = copy.deepcopy(plain_model)
    plain = LocalBackend(device="cpu", use_lora=False, model_instance=plain_model)
    checkpointed = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=checkpointed_model,
        gradient_checkpointing=True,
    )
    plain.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
    checkpointed.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))

    datum = sft_datum(tokens=(5, 6, 7, 8, 9, 10, 11), n_prompt=2)
    plain_output = asyncio.run(forward_backward(plain, [datum], "cross_entropy"))
    checkpointed_output = asyncio.run(forward_backward(checkpointed, [datum], "cross_entropy"))

    assert checkpointed.gradient_checkpointing is True
    assert checkpointed.gradient_checkpointing_layers is None
    assert checkpointed_model.is_gradient_checkpointing is True
    assert checkpointed_model.config.use_cache is False
    assert checkpointed_output.metrics["loss"] == pytest.approx(plain_output.metrics["loss"])
    torch.testing.assert_close(checkpointed_output.logprobs[0], plain_output.logprobs[0])
    for plain_parameter, checkpointed_parameter in zip(plain_model.parameters(), checkpointed_model.parameters()):
        torch.testing.assert_close(checkpointed_parameter.grad, plain_parameter.grad)


def test_gradient_checkpointing_without_layer_limit_keeps_every_backbone_layer_enabled():
    model = checkpointable_layer_model()
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=model,
        gradient_checkpointing=True,
    )

    backend.setup(model="checkpointable", lora=LoRAConfig(rank=4))

    assert backend.gradient_checkpointing_layers is None
    assert model.gradient_checkpointing_kwargs == {"use_reentrant": False}
    assert [layer.gradient_checkpointing for layer in model.model.layers] == [True, True, True, True]


def test_selective_gradient_checkpointing_keeps_only_first_n_backbone_layers_enabled():
    model = checkpointable_layer_model()
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=model,
        gradient_checkpointing=True,
        gradient_checkpointing_layers=2,
    )

    backend.setup(model="checkpointable", lora=LoRAConfig(rank=4))

    assert backend.gradient_checkpointing_layers == 2
    assert model.is_gradient_checkpointing is True
    assert model.config.use_cache is False
    assert [layer.gradient_checkpointing for layer in model.model.layers] == [True, True, False, False]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_selective_gradient_checkpointing_requires_a_positive_integer(value):
    with pytest.raises(ValueError, match="must be a positive integer"):
        LocalBackend(gradient_checkpointing=True, gradient_checkpointing_layers=value)


def test_selective_gradient_checkpointing_requires_checkpointing_enabled():
    with pytest.raises(ValueError, match="requires gradient_checkpointing=True"):
        LocalBackend(gradient_checkpointing_layers=1)


def test_selective_gradient_checkpointing_rejects_layer_count_above_model_depth():
    model = checkpointable_layer_model(layer_count=3)
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=model,
        gradient_checkpointing=True,
        gradient_checkpointing_layers=4,
    )

    with pytest.raises(ValueError, match="exceeds the model's 3 backbone layers"):
        backend.setup(model="checkpointable", lora=LoRAConfig(rank=4))
    assert [layer.gradient_checkpointing for layer in model.model.layers] == [True, True, True]


def test_selective_gradient_checkpointing_rejects_unidentified_backbone_layers():
    model = tiny_model()
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=model,
        gradient_checkpointing=True,
        gradient_checkpointing_layers=1,
    )

    with pytest.raises(RuntimeError, match="could not identify .* backbone layers"):
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
    assert model.is_gradient_checkpointing is True


def test_device_map_is_forwarded_and_preserved(monkeypatch):
    class PretendShardedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(VOCAB, 8)
            self.projection = torch.nn.Linear(8, VOCAB)
            self.hf_device_map = {"embedding": 0, "projection": 1}

        def get_input_embeddings(self):
            return self.embedding

        def to(self, *args, **kwargs):
            raise AssertionError("setup must not collapse a device-mapped model with .to()")

    loaded = PretendShardedModel()
    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return loaded

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake_from_pretrained)
    backend = LocalBackend(
        use_lora=False,
        device_map="auto",
        max_memory={0: "45GiB", 1: "45GiB"},
    )
    backend.setup(model="pretend-sharded", lora=LoRAConfig(rank=4))

    assert captured == {
        "model_name": "pretend-sharded",
        "kwargs": {
            "torch_dtype": torch.float32,
            "device_map": "auto",
            "max_memory": {0: "45GiB", 1: "45GiB"},
        },
    }
    assert backend.model is loaded
    assert backend.device == "cpu"  # the fake embedding was deliberately left on CPU


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


async def forward_backward(backend, datums, loss_fn):
    return await (await backend.submit_forward_backward(datums, loss_fn)).result()


@pytest.mark.parametrize("loss_fn", ["importance_sampling", "ppo"])
def test_fused_opct_forward_backward_matches_explicit_single_pass_objective(loss_fn):
    fused = deterministic_backend(max_datums=None, max_tokens=None)
    reference = deterministic_backend(max_datums=None, max_tokens=None)
    fused_datum = rl_datum(fused, prompt=(5, 6), action=(7, 8, 9, 10), advantage=0.0)
    reference_datum = rl_datum(reference, prompt=(5, 6), action=(7, 8, 9, 10), advantage=0.0)
    teacher = torch.tensor([-2.1, -1.7, -2.4, -1.9])
    fused_datum.loss_fn_inputs["opct_teacher_logprobs"] = types.TensorData.from_torch(teacher)

    # Store processed generation scores that deliberately cannot be recovered
    # by applying the configured temperature to this model's current logits.
    behavior_offsets = torch.tensor([0.15, 0.25, 0.05, 0.10])
    for datum in (fused_datum, reference_datum):
        mask = datum.loss_fn_inputs["mask"].to_torch().bool()
        stored_behavior = datum.loss_fn_inputs["logprobs"].to_torch().clone()
        stored_behavior[mask] += behavior_offsets
        datum.loss_fn_inputs["logprobs"] = types.TensorData.from_torch(stored_behavior)
    with torch.no_grad():
        _, reconstructed_behavior = reference._opct_logprobs_batch(
            [reference_datum],
            behavior_temperature=0.7,
        )
    reference_mask = reference_datum.loss_fn_inputs["mask"].to_torch().bool()
    stored_reference_behavior = reference_datum.loss_fn_inputs["logprobs"].to_torch()
    assert not torch.allclose(
        stored_reference_behavior[reference_mask],
        reconstructed_behavior[0][reference_mask],
    )

    def reject_temperature_recomputation(*args, **kwargs):
        raise AssertionError("fused OPCT must use the behavior scores stored in the datum")

    fused._opct_logprobs_batch = reject_temperature_recomputation
    fused.model.forward_calls = 0
    reference.model.forward_calls = 0

    async def run_fused():
        pending = await fused.submit_opct_forward_backward(
            [fused_datum],
            behavior_temperature=0.7,
            kl_coef=1.3,
            kl_discount_factor=0.6,
            loss_fn=loss_fn,
        )
        return await pending.result()

    fused_output = asyncio.run(run_fused())

    raw = reference._target_logprobs_batch([reference_datum])[0]
    mask = reference_mask
    raw_action = raw[mask]
    behavior_action = stored_reference_behavior[mask]
    reverse_kl = raw_action.detach() - teacher
    action_signal = -1.3 * reverse_kl
    running = torch.zeros((), dtype=action_signal.dtype)
    discounted = torch.empty_like(action_signal)
    for index in range(len(action_signal) - 1, -1, -1):
        running = action_signal[index] + 0.6 * running
        discounted[index] = running
    ratio = torch.exp(raw_action - behavior_action.detach())
    if loss_fn == "ppo":
        clipped = torch.clamp(
            ratio,
            1.0 - reference.ppo_clip_epsilon,
            1.0 + reference.ppo_clip_epsilon,
        )
        surrogate = torch.minimum(ratio * discounted, clipped * discounted)
    else:
        surrogate = ratio * discounted
    expected_loss = -surrogate.sum() / mask.sum()
    expected_loss.backward()

    assert fused.model.forward_calls == 1
    assert fused_output.metrics["loss"] == pytest.approx(float(expected_loss.detach()), abs=1e-6)
    assert fused_output.metrics["teacher_kl"] == pytest.approx(float(reverse_kl.mean()), abs=1e-6)
    assert fused_output.metrics["teacher_scored_tokens"] == 4
    torch.testing.assert_close(fused_output.logprobs[0], raw.detach())
    for fused_parameter, reference_parameter in zip(fused.model.parameters(), reference.model.parameters()):
        torch.testing.assert_close(fused_parameter.grad, reference_parameter.grad)


def microbatch_test_datums(backend: LocalBackend, loss_fn: str):
    if loss_fn == "cross_entropy":
        return [
            sft_datum(tokens=(5, 6, 7, 8, 9, 10, 11, 12), n_prompt=2),
            sft_datum(tokens=(4, 3, 2, 1), n_prompt=1),
            sft_datum(tokens=(12, 11, 10, 9, 8, 7), n_prompt=3),
            sft_datum(tokens=(2, 4, 6, 8, 10), n_prompt=5),  # zero weight
        ]
    datums = [
        rl_datum(backend, prompt=(5, 6), action=(7, 8, 9, 10), advantage=1.25),
        rl_datum(backend, prompt=(3,), action=(4, 5), advantage=-0.5),
        rl_datum(backend, prompt=(9, 8, 7), action=(6, 5, 4), advantage=0.75),
        rl_datum(backend, prompt=(2, 3), action=(4,), advantage=2.0),
    ]
    # Exercise both sides of PPO clipping instead of leaving every ratio at 1.
    for datum, shift in zip(datums, (-0.5, 0.4, -0.3, 0.2)):
        sampled = datum.loss_fn_inputs["logprobs"].to_torch()
        datum.loss_fn_inputs["logprobs"] = types.TensorData.from_torch(sampled + shift)
    return datums


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_forward_microbatch_matches_unchunked_loss_gradients_logprobs_and_optimizer(loss_fn):
    unchunked = deterministic_backend(max_datums=None, max_tokens=None)
    chunked = deterministic_backend(max_datums=2, max_tokens=8)
    datums = microbatch_test_datums(unchunked, loss_fn)
    unchunked.model.forward_calls = 0
    chunked.model.forward_calls = 0

    unchunked_out = asyncio.run(forward_backward(unchunked, datums, loss_fn))
    chunked_out = asyncio.run(forward_backward(chunked, datums, loss_fn))

    assert unchunked.model.forward_calls == 1
    assert chunked.model.forward_calls > 1
    assert unchunked._gradient_accumulations == chunked._gradient_accumulations == 1
    assert chunked_out.metrics["loss"] == pytest.approx(unchunked_out.metrics["loss"], abs=1e-6)
    if loss_fn == "cross_entropy":
        expected_loss = local_losses.cross_entropy_loss(
            unchunked_out.logprobs,
            [datum.loss_fn_inputs["weights"].to_torch() for datum in datums],
        )
    else:
        loss_args = (
            unchunked_out.logprobs,
            [datum.loss_fn_inputs["logprobs"].to_torch() for datum in datums],
            [datum.loss_fn_inputs["advantages"].to_torch() for datum in datums],
            [datum.loss_fn_inputs["mask"].to_torch().float() for datum in datums],
        )
        expected_loss = local_losses.ppo_loss(*loss_args, clip_epsilon=unchunked.ppo_clip_epsilon) if loss_fn == "ppo" else local_losses.importance_sampling_loss(*loss_args)
    assert unchunked_out.metrics["loss"] == pytest.approx(float(expected_loss), abs=1e-6)
    assert len(chunked_out.logprobs) == len(datums)
    for chunked_logprobs, unchunked_logprobs, datum in zip(chunked_out.logprobs, unchunked_out.logprobs, datums):
        torch.testing.assert_close(chunked_logprobs, unchunked_logprobs)
        assert chunked_logprobs.shape == datum.loss_fn_inputs["target_tokens"].to_torch().shape

    unchunked_grads = {name: parameter.grad.clone() for name, parameter in unchunked.model.named_parameters()}
    chunked_grads = {name: parameter.grad.clone() for name, parameter in chunked.model.named_parameters()}
    assert unchunked_grads.keys() == chunked_grads.keys()
    for name, unchunked_grad in unchunked_grads.items():
        torch.testing.assert_close(chunked_grads[name], unchunked_grad, atol=1e-6, rtol=1e-5)

    # A second logical submission must count once on both backends even though
    # the chunked backend performs several model forwards for it.
    asyncio.run(forward_backward(unchunked, list(reversed(datums)), loss_fn))
    asyncio.run(forward_backward(chunked, list(reversed(datums)), loss_fn))
    assert unchunked._gradient_accumulations == chunked._gradient_accumulations == 2

    adam = AdamConfig(weight_decay=0.0, grad_clip_norm=0.0)

    async def optimize(backend):
        await (await backend.submit_optim_step(learning_rate=1e-3, adam=adam)).result()

    asyncio.run(optimize(unchunked))
    asyncio.run(optimize(chunked))
    assert unchunked._gradient_accumulations == chunked._gradient_accumulations == 0
    for chunked_parameter, unchunked_parameter in zip(chunked.model.parameters(), unchunked.model.parameters()):
        torch.testing.assert_close(chunked_parameter, unchunked_parameter, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_forward_microbatch_zero_denominator_is_finite(loss_fn):
    backend = deterministic_backend(max_datums=1, max_tokens=2)
    if loss_fn == "cross_entropy":
        datums = [sft_datum(tokens=(5, 6, 7), n_prompt=3)]
    else:
        datum = rl_datum(backend, prompt=(5,), action=(6, 7), advantage=1.0)
        mask = datum.loss_fn_inputs["mask"].to_torch()
        datum.loss_fn_inputs["mask"] = types.TensorData.from_torch(torch.zeros_like(mask))
        datums = [datum]

    out = asyncio.run(forward_backward(backend, datums, loss_fn))

    assert out.metrics["loss"] == pytest.approx(0.0)
    assert torch.isfinite(torch.tensor(out.metrics["loss"]))
    assert backend._gradient_accumulations == 1


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_forward_microbatch_empty_logical_batch_is_a_zero_loss_submission(loss_fn):
    backend = deterministic_backend(max_datums=1, max_tokens=2)

    out = asyncio.run(forward_backward(backend, [], loss_fn))

    assert out.logprobs == []
    assert out.metrics["loss"] == pytest.approx(0.0)
    assert backend.model.forward_calls == 0
    assert backend._gradient_accumulations == 1


def test_forward_microbatch_budgets_use_padded_footprint_and_validate_values():
    backend = deterministic_backend(max_datums=3, max_tokens=10)
    assert backend._forward_microbatches([9, 2, 2, 6, 1]) == [[0], [3], [1, 2, 4]]
    default_backend = LocalBackend(model_instance=DeterministicTinyCausalLM())
    assert default_backend._forward_microbatches([1] * 9) == [list(range(8)), [8]]
    for kwargs in (
        {"forward_microbatch_max_datums": 0},
        {"forward_microbatch_max_tokens": -1},
        {"forward_microbatch_max_tokens": True},
    ):
        with pytest.raises(ValueError, match="positive integer or None"):
            LocalBackend(model_instance=DeterministicTinyCausalLM(), **kwargs)
    for value in (0, -1, True, None):
        with pytest.raises(ValueError, match="target_logprob_chunk_size must be a positive integer"):
            LocalBackend(model_instance=DeterministicTinyCausalLM(), target_logprob_chunk_size=value)


@pytest.mark.parametrize("loss_fn", ["cross_entropy", "importance_sampling", "ppo"])
def test_hf_selected_token_path_matches_dense_loss_logprobs_and_gradients(loss_fn):
    selected_model = deterministic_tiny_hf_model()
    dense_model = copy.deepcopy(selected_model)
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=selected_model,
        forward_microbatch_max_datums=None,
        forward_microbatch_max_tokens=None,
        target_logprob_chunk_size=3,
    )
    backend.setup(model="deterministic-hf-tiny", lora=LoRAConfig(rank=4))
    assert _selected_token_components(backend.model) is not None
    datums = microbatch_test_datums(backend, loss_fn)

    top_level_forward_calls = []
    head_chunk_sizes = []
    model_hook = backend.model.register_forward_hook(lambda *_args: top_level_forward_calls.append(True))
    head_hook = backend.model.get_output_embeddings().register_forward_pre_hook(lambda _module, args: head_chunk_sizes.append(args[0].shape[0]))
    selected_output = asyncio.run(forward_backward(backend, datums, loss_fn))
    model_hook.remove()
    head_hook.remove()

    dense_logprobs = dense_datum_logprobs(dense_model, datums)
    if loss_fn == "cross_entropy":
        dense_loss = local_losses.cross_entropy_loss(
            dense_logprobs,
            [datum.loss_fn_inputs["weights"].to_torch() for datum in datums],
        )
    else:
        loss_args = (
            dense_logprobs,
            [datum.loss_fn_inputs["logprobs"].to_torch() for datum in datums],
            [datum.loss_fn_inputs["advantages"].to_torch() for datum in datums],
            [datum.loss_fn_inputs["mask"].to_torch().float() for datum in datums],
        )
        dense_loss = local_losses.ppo_loss(*loss_args, clip_epsilon=backend.ppo_clip_epsilon) if loss_fn == "ppo" else local_losses.importance_sampling_loss(*loss_args)
    dense_loss.backward()

    assert top_level_forward_calls == []
    assert head_chunk_sizes and max(head_chunk_sizes) <= 3
    assert selected_output.metrics["loss"] == pytest.approx(float(dense_loss.detach()), abs=1e-6)
    for selected, dense in zip(selected_output.logprobs, dense_logprobs):
        torch.testing.assert_close(selected, dense.detach(), atol=1e-6, rtol=1e-5)
    dense_gradients = dict(dense_model.named_parameters())
    for name, selected_parameter in backend.model.named_parameters():
        dense_gradient = dense_gradients[name].grad
        assert selected_parameter.grad is not None and dense_gradient is not None
        torch.testing.assert_close(selected_parameter.grad, dense_gradient, atol=1e-6, rtol=1e-5)


def test_completion_scoring_microbatches_mixed_lengths_and_restores_order_without_dense_logits():
    model = deterministic_tiny_hf_model()
    dense_model = copy.deepcopy(model).eval()
    backend = LocalBackend(
        device="cpu",
        use_lora=False,
        model_instance=model,
        forward_microbatch_max_datums=2,
        forward_microbatch_max_tokens=9,
        target_logprob_chunk_size=2,
    )
    backend.setup(model="deterministic-hf-tiny", lora=LoRAConfig(rank=4))
    prompts = [types.ModelInput.from_ints(tokens=tokens) for tokens in ([2], [3, 4, 5], [6, 7], [8, 9, 10, 11])]
    completions = [[12, 13, 14], [15], [16, 17, 18, 19], [20, 21]]

    backbone_batch_sizes = []
    head_chunk_sizes = []
    backbone_hook = backend.model.transformer.register_forward_pre_hook(
        lambda _module, args, kwargs: backbone_batch_sizes.append(kwargs["input_ids"].shape[0]),
        with_kwargs=True,
    )
    head_hook = backend.model.get_output_embeddings().register_forward_pre_hook(lambda _module, args: head_chunk_sizes.append(args[0].shape[0]))
    scored = backend._score_completions(prompts, completions, use_base=False)
    backbone_hook.remove()
    head_hook.remove()

    expected = []
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            prompt_tokens = prompt.to_ints()
            input_ids = torch.tensor([prompt_tokens + completion[:-1]])
            dense_logits = dense_model(input_ids=input_ids, use_cache=False).logits[0]
            positions = dense_logits[len(prompt_tokens) - 1 :]
            target = torch.tensor(completion)
            expected.append(torch.log_softmax(positions.float(), dim=-1).gather(1, target[:, None]).squeeze(1).tolist())

    assert len(backbone_batch_sizes) > 1
    assert max(backbone_batch_sizes) <= 2
    assert head_chunk_sizes and max(head_chunk_sizes) <= 2
    assert model.training
    for actual, dense in zip(scored, expected):
        assert actual == pytest.approx(dense, abs=1e-6)


def test_selected_token_path_falls_back_to_custom_causal_lm_forward_without_changing_logits():
    from transformers import GPT2LMHeadModel

    class CustomPostprocessedGPT2(GPT2LMHeadModel):
        def __init__(self, config):
            super().__init__(config)
            self.forward_calls = 0

        def forward(self, *args, **kwargs):
            self.forward_calls += 1
            output = super().forward(*args, **kwargs)
            bias = torch.linspace(-0.3, 0.3, output.logits.shape[-1], device=output.logits.device)
            output.logits = output.logits + bias
            return output

    template = deterministic_tiny_hf_model()
    model = CustomPostprocessedGPT2(template.config)
    model.load_state_dict(template.state_dict())
    backend = LocalBackend(device="cpu", use_lora=False, model_instance=model)
    backend.setup(model="custom-gpt2", lora=LoRAConfig(rank=4))
    assert _selected_token_components(backend.model) is None
    datum = sft_datum(tokens=(5, 6, 7, 8, 9), n_prompt=2)

    with torch.no_grad():
        actual = backend._target_logprobs([datum])[0]
        expected = dense_datum_logprobs(model, [datum])[0]

    assert model.forward_calls == 2
    torch.testing.assert_close(actual, expected)


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

    def test_selective_full_finetune_keeps_frozen_base(self):
        model = tiny_model()
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=model,
            full_finetune_modules=["attn"],
            keep_frozen_base=True,
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))

        trainable = [name for name, parameter in backend.model.named_parameters() if parameter.requires_grad]
        assert trainable and all("attn" in name for name in trainable)
        assert backend._frozen_base_model is not None
        assert all(not parameter.requires_grad for parameter in backend._frozen_base_model.parameters())
        assert all(torch.equal(initial[name], value) for name, value in backend._frozen_base_model.state_dict().items())

        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        assert all(torch.equal(initial[name], value) for name, value in backend._frozen_base_model.state_dict().items())


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

    def test_kl_base_scoring_microbatch_matches_unchunked_and_preserves_order(self):
        unchunked = deterministic_backend(max_datums=None, max_tokens=None, keep_frozen_base=True)
        chunked = deterministic_backend(max_datums=2, max_tokens=8, keep_frozen_base=True)
        with torch.no_grad():
            offset = torch.linspace(-0.2, 0.2, unchunked.model.projection.bias.numel())
            unchunked.model.projection.bias.add_(offset)
            chunked.model.projection.bias.add_(offset)

        datums = microbatch_test_datums(unchunked, "importance_sampling")
        chunked_datums = copy.deepcopy(datums)
        assert unchunked._frozen_base_model is not None
        assert chunked._frozen_base_model is not None
        unchunked._frozen_base_model.forward_calls = 0
        chunked._frozen_base_model.forward_calls = 0

        unchunked_metrics = asyncio.run(unchunked.incorporate_kl_penalty(datums, kl_coef=0.3, kl_discount_factor=0.7))
        chunked_metrics = asyncio.run(chunked.incorporate_kl_penalty(chunked_datums, kl_coef=0.3, kl_discount_factor=0.7))

        assert unchunked._frozen_base_model.forward_calls == 1
        assert chunked._frozen_base_model.forward_calls > 1
        assert chunked_metrics["kl_policy_base"] == pytest.approx(unchunked_metrics["kl_policy_base"], abs=1e-6)
        for chunked_datum, unchunked_datum in zip(chunked_datums, datums):
            torch.testing.assert_close(
                chunked_datum.loss_fn_inputs["advantages"].to_torch(),
                unchunked_datum.loss_fn_inputs["advantages"].to_torch(),
                atol=1e-6,
                rtol=1e-5,
            )

    def test_policy_handle_scoring_matches_raw_frozen_base_logits(self):
        model = tiny_model()
        backend = LocalBackend(
            device="cpu",
            use_lora=False,
            model_instance=model,
            keep_frozen_base=True,
        )
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
        prompt = types.ModelInput.from_ints(tokens=[5, 6])
        completion = [7, 8]

        scored = asyncio.run(backend.base_sampler().score_completions([prompt], [completion]))[0]
        current_scored = asyncio.run(backend.policy_sampler("current").score_completions([prompt], [completion]))[0]
        frozen = backend._frozen_base_model
        assert frozen is not None
        with torch.no_grad():
            logits = frozen(input_ids=torch.tensor([[5, 6, 7, 8]])).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)
            expected = [float(logprobs[0, 1, 7]), float(logprobs[0, 2, 8])]
        assert scored == pytest.approx(expected)
        assert current_scored == pytest.approx(expected)


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

    def test_concurrent_calls_are_coalesced_into_one_backend_batch(self):
        class BatchBackend:
            def __init__(self):
                self.calls = []

            def _sample_batch(self, **kwargs):
                self.calls.append(kwargs)
                return [[] for _ in kwargs["prompt_tokens_batch"]]

        backend = BatchBackend()
        handle = LocalSamplerHandle(backend, use_base=True)

        async def sample_all():
            return await asyncio.gather(
                *(
                    handle.sample(
                        types.ModelInput.from_ints(tokens=[token]),
                        max_tokens=8,
                        temperature=0.7,
                        stop=[0],
                        num_samples=1,
                    )
                    for token in (1, 2, 3, 4)
                )
            )

        assert asyncio.run(sample_all()) == [[], [], [], []]
        assert len(backend.calls) == 1
        assert backend.calls[0]["prompt_tokens_batch"] == [[1], [2], [3], [4]]
        assert backend.calls[0]["use_base"] is True


class TestLocalCheckpoint:
    def test_save_and_resume_full_finetune(self, tmp_path):
        backend = make_backend()
        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        paths = asyncio.run(backend.save_checkpoint(name="run1_step1", log_dir=tmp_path, loop_state={"step": 1}, kind="both"))

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

    def test_resumed_segment_without_optimizer_step_keeps_optimizer_state(self, tmp_path):
        backend = make_backend()
        asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        parent = asyncio.run(backend.save_checkpoint(name="parent", log_dir=tmp_path, loop_state={}, kind="both"))

        resumed = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
        resumed.setup(
            model="tiny-gpt2-test",
            lora=LoRAConfig(rank=4),
            resume_from=parent["sampler_path"],
            resume_with_optimizer=True,
        )
        # Zero-signal segment: no optimizer step, so the optimizer is never built.
        child = asyncio.run(resumed.save_checkpoint(name="child", log_dir=tmp_path, loop_state={}, kind="both"))

        assert child["state_path"] is not None
        saved = torch.load(tmp_path / "checkpoints" / "child" / "optimizer.pt")
        expected = torch.load(tmp_path / "checkpoints" / "parent" / "optimizer.pt")
        assert saved["param_groups"] == expected["param_groups"]
        assert saved["state"].keys() == expected["state"].keys()
        for key, value in expected["state"].items():
            for name, tensor in value.items():
                assert torch.equal(saved["state"][key][name], tensor)

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

    def test_exact_target_modules_alpha_and_dropout(self):
        model = tiny_model()
        targets = _lora_target_module_names(
            model,
            LoRAConfig(target_modules=["c_attn"], rank=4, alpha=12, dropout=0.15),
        )
        assert targets and all(name.endswith(".c_attn") for name in targets)

        backend = LocalBackend(device="cpu", use_lora=True, model_instance=model)
        backend.setup(
            model="tiny-gpt2-test",
            lora=LoRAConfig(target_modules=["c_attn"], rank=4, alpha=12, dropout=0.15),
        )
        config = backend.model.peft_config["default"]
        assert config.lora_alpha == 12
        assert config.lora_dropout == 0.15

    def test_selected_token_path_preserves_active_and_disabled_adapter_context_and_gradients(self):
        backend = LocalBackend(
            device="cpu",
            use_lora=True,
            model_instance=deterministic_tiny_hf_model(),
            forward_microbatch_max_datums=None,
            forward_microbatch_max_tokens=None,
            target_logprob_chunk_size=2,
        )
        backend.setup(
            model="deterministic-hf-tiny",
            lora=LoRAConfig(rank=2, target_modules=["c_attn", "lm_head"], dropout=0.0, seed=7),
        )
        assert _selected_token_components(backend.model) is not None
        with torch.no_grad():
            for name, parameter in backend.model.named_parameters():
                if ".lora_B." in name:
                    parameter.normal_(mean=0.0, std=0.03)

        datums = [
            sft_datum(tokens=(5, 6, 7, 8, 9, 10), n_prompt=2),
            sft_datum(tokens=(3, 4, 5), n_prompt=1),
        ]
        with torch.no_grad():
            selected_adapter = backend._target_logprobs(datums)
            dense_adapter = dense_datum_logprobs(backend.model, datums)
            selected_base = backend._target_logprobs(datums, use_base=True)
            with backend._base_ctx():
                dense_base = dense_datum_logprobs(backend.model, datums)

        for selected, dense in zip(selected_adapter, dense_adapter):
            torch.testing.assert_close(selected, dense, atol=1e-6, rtol=1e-5)
        for selected, dense in zip(selected_base, dense_base):
            torch.testing.assert_close(selected, dense, atol=1e-6, rtol=1e-5)
        assert any(not torch.allclose(adapter, base) for adapter, base in zip(selected_adapter, selected_base))

        backend.model.zero_grad(set_to_none=True)
        outer_forward_calls = []
        head_chunk_sizes = []
        model_hook = backend.model.register_forward_hook(lambda *_args: outer_forward_calls.append(True))
        head_hook = backend.model.get_output_embeddings().register_forward_pre_hook(lambda _module, args: head_chunk_sizes.append(args[0].shape[0]))
        asyncio.run(forward_backward(backend, datums, "cross_entropy"))
        model_hook.remove()
        head_hook.remove()

        assert outer_forward_calls == []
        assert head_chunk_sizes and max(head_chunk_sizes) <= 2
        attention_gradient = sum(float(parameter.grad.abs().sum()) for name, parameter in backend.model.named_parameters() if "attn.c_attn" in name and "lora_" in name and parameter.grad is not None)
        head_gradient = sum(float(parameter.grad.abs().sum()) for name, parameter in backend.model.named_parameters() if "lm_head" in name and "lora_" in name and parameter.grad is not None)
        assert attention_gradient > 0
        assert head_gradient > 0

    def test_fused_moe_expert_parameters_are_selected_for_mlp_lora(self):
        class Experts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 4, 8))
                self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4))

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = Experts()

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                self.self_attn = torch.nn.Linear(4, 4)

        model = torch.nn.ModuleList([Block()])
        config = LoRAConfig(train_mlp=True, train_attn=True, train_unembed=False)

        assert _lora_target_module_names(model, config) == ["0.self_attn"]
        assert _lora_target_parameter_names(model, config) == [
            "0.mlp.experts.gate_up_proj",
            "0.mlp.experts.down_proj",
        ]

    def test_tiny_gpt_oss_wraps_attention_and_fused_expert_parameters(self):
        from transformers import GptOssConfig, GptOssForCausalLM

        model = GptOssForCausalLM(
            GptOssConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
                num_local_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=64,
            )
        )
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=model)
        with pytest.warns(FusedExpertLoRAWarning, match="fused 3-D expert parameter"):
            backend.setup(
                model="tiny-gpt-oss",
                lora=LoRAConfig(rank=2, alpha=4, train_mlp=True, train_attn=True, train_unembed=False),
            )
        peft_config = backend.model.peft_config["default"]

        assert any(name.endswith("self_attn.q_proj") for name in peft_config.target_modules)
        assert sorted(peft_config.target_parameters) == [
            "model.layers.0.mlp.experts.down_proj",
            "model.layers.0.mlp.experts.gate_up_proj",
        ]

    def test_lora_wraps_and_trains(self):
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=tiny_model())
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4, seed=0))
        n_trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in backend.model.parameters())
        assert 0 < n_trainable < n_total * 0.5  # adapter-only training

        first = asyncio.run(step(backend, [sft_datum()], "cross_entropy"))
        assert first.metrics["loss"] > 0
