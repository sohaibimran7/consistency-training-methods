"""CPU tests for the internal-consistency training path (ACT / AttCT / MLPCT).

Covers the ported losses (ctm/backends/local/consistency_losses.py), the MLP
hook manager, the paired-datum adapter (ctm/training/consistency_data.py), and
the LocalBackend consistency loss_fns — all offline: losses run on synthetic
output objects, the backend on a tiny random GPT-2, the adapter on an in-memory
word-level tokenizer.
"""

import asyncio
import math
from types import SimpleNamespace

import pytest
import torch
from tinker import types

from ctm.backends.local.consistency_losses import (
    ActivationConsistencyLoss,
    JSDAttentionConsistencyLoss,
    MLPConsistencyLoss,
)
from ctm.backends.local.engine import HAS_PEFT, LocalBackend
from ctm.backends.local.mlp_hooks import MLPHookManager, find_mlp_down_proj_modules
from ctm.core.config import AdamConfig, LoRAConfig
from ctm.training.consistency_data import (
    build_consistency_datum,
    build_consistency_datums,
    find_content_token_boundary,
    longest_matching_suffix_len,
)

VOCAB = 128


def tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=32, n_layer=2, n_head=2))


# ── synthetic paired outputs ─────────────────────────────────────────────────
# Clean sequence length 6, biased length 8 with the clean content at positions
# [2:8) — i.e. start_index=2, clean_start_index=0, clean_len=6, match_len=6.

IDX = dict(start_index=2, clean_start_index=0, clean_len=6, match_len=6)


def paired_attentions(n_layers=2, heads=2, perturb=0.0):
    torch.manual_seed(1)
    clean_atts, adv_atts = [], []
    for _ in range(n_layers):
        adv = torch.softmax(torch.randn(1, heads, 8, 8), dim=-1)
        clean = adv[:, :, 2:8, 2:8].clone()
        if perturb:
            adv = torch.softmax(torch.log(adv) + perturb * torch.randn_like(adv), dim=-1)
        clean_atts.append(clean)
        adv_atts.append(adv.requires_grad_(True))
    return SimpleNamespace(attentions=tuple(clean_atts)), SimpleNamespace(attentions=tuple(adv_atts))


def paired_hidden_states(n_layers=2, dim=4, offset=0.0, mismatch_embedding=False):
    """hidden_states tuples of length n_layers+1 (index 0 = input embedding)."""
    torch.manual_seed(2)
    clean_hs, adv_hs = [], []
    for layer in range(n_layers + 1):
        adv = torch.randn(1, 8, dim)
        clean = adv[:, 2:8, :].clone()
        if layer == 0 and mismatch_embedding:
            clean = clean + 100.0
        if offset and layer > 0:
            adv = adv + offset
        clean_hs.append(clean)
        adv_hs.append(adv.requires_grad_(True))
    return SimpleNamespace(hidden_states=tuple(clean_hs)), SimpleNamespace(hidden_states=tuple(adv_hs))


class TestJSDAttentionConsistencyLoss:
    def test_zero_when_aligned_windows_match(self):
        clean, adv = paired_attentions()
        out = JSDAttentionConsistencyLoss()(clean, adv, **IDX)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_and_differentiable_when_windows_differ(self):
        clean, adv = paired_attentions(perturb=0.5)
        out = JSDAttentionConsistencyLoss()(clean, adv, **IDX)
        assert out["loss"].item() > 1e-4
        assert out["loss"].item() <= math.log(2) + 1e-6  # JSD is bounded
        out["loss"].backward()
        assert adv.attentions[0].grad is not None
        assert len(out["layer_losses"]) == 2

    def test_layer_selection_last_half(self):
        clean, adv = paired_attentions()
        out = JSDAttentionConsistencyLoss(layer_selection="last_half")(clean, adv, **IDX)
        assert len(out["layer_losses"]) == 1

    def test_requires_attentions(self):
        with pytest.raises(ValueError, match="output_attentions"):
            JSDAttentionConsistencyLoss()(SimpleNamespace(attentions=None), SimpleNamespace(attentions=None), **IDX)


class TestActivationConsistencyLoss:
    def test_zero_on_matching_suffix_ignores_embedding_layer(self):
        # Embedding layer (hidden_states[0]) deliberately mismatched: Eq. 1 skips it.
        clean, adv = paired_hidden_states(mismatch_embedding=True)
        out = ActivationConsistencyLoss()(clean, adv, **IDX)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-9)
        assert out["num_layers_used"] == 2

    def test_paper_scale_sums_over_hidden_dim(self):
        # adv = clean + 1 everywhere → per-position ||diff||² = dim → loss = dim.
        dim = 4
        clean, adv = paired_hidden_states(dim=dim, offset=1.0)
        out = ActivationConsistencyLoss()(clean, adv, **IDX)
        assert out["loss"].item() == pytest.approx(dim, rel=1e-5)
        out["loss"].backward()
        assert adv.hidden_states[1].grad is not None
        # Clean side is the stop-gradient reference.
        assert clean.hidden_states[1].grad is None

    def test_match_len_window_beats_content_window(self):
        # match_len=3 compares only the last 3 positions of both sequences.
        clean, adv = paired_hidden_states()
        out = ActivationConsistencyLoss()(clean, adv, start_index=0, clean_start_index=0, clean_len=6, match_len=3)
        assert out["match_len"] == 3
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-9)


class TestMLPConsistencyLoss:
    def test_zero_when_states_match(self):
        torch.manual_seed(3)
        adv_states = [torch.randn(1, 8, 5, requires_grad=True) for _ in range(2)]
        clean_states = [s[:, 2:8, :].detach().clone() for s in adv_states]
        out = MLPConsistencyLoss()(None, None, **IDX, clean_mlp_states=clean_states, adv_mlp_states=adv_states)
        assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_and_differentiable_on_mismatch(self):
        torch.manual_seed(4)
        adv_states = [torch.randn(1, 8, 5, requires_grad=True) for _ in range(2)]
        clean_states = [torch.randn(1, 6, 5) for _ in range(2)]
        out = MLPConsistencyLoss()(None, None, **IDX, clean_mlp_states=clean_states, adv_mlp_states=adv_states)
        assert out["loss"].item() > 1e-3
        out["loss"].backward()
        assert adv_states[0].grad is not None

    def test_requires_states(self):
        with pytest.raises(ValueError, match="clean_mlp_states"):
            MLPConsistencyLoss()(None, None, **IDX)


class TestMLPHooks:
    def test_finds_gpt2_mlp_projections_in_order(self):
        model = tiny_model()
        modules = find_mlp_down_proj_modules(model)
        assert [name for name, _ in modules] == [
            "transformer.h.0.mlp.c_proj",
            "transformer.h.1.mlp.c_proj",
        ]

    @pytest.mark.parametrize("variant,dim", [("hidden", 128), ("output", 32)])  # GPT-2 MLP inner = 4×n_embd
    def test_captures_states_per_layer(self, variant, dim):
        model = tiny_model()
        mgr = MLPHookManager(model, variant=variant).install()
        model(input_ids=torch.tensor([[5, 6, 7]]))
        states = mgr.get_states()
        assert len(states) == 2
        assert all(s.shape == (1, 3, dim) for s in states)
        mgr.clear()
        with pytest.raises(RuntimeError, match="no layers fired"):
            mgr.get_states()
        mgr.remove()


# ── paired-datum adapter ─────────────────────────────────────────────────────


def word_tokenizer():
    """In-memory word-level fast tokenizer (offset mapping, no chat template, no downloads)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = "what is two plus two ? my professor says the answer five . [UNK]".split()
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")


CLEAN = "what is two plus two ?"
BIASED = "my professor says the answer is five . what is two plus two ?"


def sample(biased=BIASED, clean=CLEAN):
    return {
        "biased_messages": [{"role": "user", "content": biased}],
        "unbiased_messages": [{"role": "user", "content": clean}],
    }


class TestConsistencyData:
    def test_longest_matching_suffix(self):
        assert longest_matching_suffix_len([1, 2, 3], [9, 2, 3]) == 2
        assert longest_matching_suffix_len([1, 2], [1, 2]) == 2
        assert longest_matching_suffix_len([1], [2]) == 0

    def test_boundary_finding_with_offsets(self):
        tok = word_tokenizer()
        ids, start, length = find_content_token_boundary(BIASED, CLEAN, tok)
        assert (start, length) == (8, 6)
        assert len(ids) == 14

    def test_build_datum_alignment(self):
        d = build_consistency_datum(word_tokenizer(), sample())
        assert len(d.model_input.to_ints()) == 14
        get = lambda k: int(d.loss_fn_inputs[k].to_torch()[0])  # noqa: E731
        assert get("start_index") == 8
        assert get("clean_start_index") == 0
        assert get("clean_len") == 6
        assert get("match_len") == 6  # biased ends with the full clean prompt
        assert d.loss_fn_inputs["clean_tokens"].to_torch().shape == (6,)

    def test_trailing_assistant_message_is_dropped(self):
        s = sample()
        s["biased_messages"] = s["biased_messages"] + [{"role": "assistant", "content": "five"}]
        s["unbiased_messages"] = s["unbiased_messages"] + [{"role": "assistant", "content": "four"}]
        d = build_consistency_datum(word_tokenizer(), s)
        assert len(d.model_input.to_ints()) == 14  # prompt-only: assistant turn excluded

    def test_unalignable_and_malformed_samples_are_skipped(self):
        tok = word_tokenizer()
        with pytest.raises(ValueError, match="not found"):
            build_consistency_datum(tok, sample(biased="says the answer is five ."))
        with pytest.raises(ValueError, match="biased_messages"):
            build_consistency_datum(tok, {"messages": []})
        datums, skipped = build_consistency_datums(tok, [sample(), sample(biased="the answer is five .")])
        assert len(datums) == 1 and skipped == 1


# ── LocalBackend integration ─────────────────────────────────────────────────

CONSISTENCY_LOSS_FNS = ["activation_consistency", "attention_consistency", "mlp_consistency"]


def consistency_datum(clean=(5, 6, 7, 8, 9), prefix=(3, 4)):
    biased = list(prefix) + list(clean)

    def scalar(v):
        return types.TensorData.from_torch(torch.tensor([v], dtype=torch.long))

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=biased),
        loss_fn_inputs={
            "clean_tokens": types.TensorData.from_torch(torch.tensor(list(clean), dtype=torch.long)),
            "start_index": scalar(len(prefix)),
            "clean_start_index": scalar(0),
            "clean_len": scalar(len(clean)),
            "match_len": scalar(len(clean)),
        },
    )


async def step(backend, datums, loss_fn, lr=1e-2):
    pending = await backend.submit_forward_backward(datums, loss_fn)
    out = await pending.result()
    opt = await backend.submit_optim_step(learning_rate=lr, adam=AdamConfig(learning_rate=lr))
    await opt.result()
    return out


class TestLocalBackendConsistency:
    def test_full_finetune_has_no_frozen_base(self):
        backend = LocalBackend(device="cpu", use_lora=False, model_instance=tiny_model())
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4))
        with pytest.raises(NotImplementedError):
            asyncio.run(backend.submit_forward_backward([consistency_datum()], "activation_consistency"))

    def test_train_sft_rejects_tinker_backend(self, tmp_path):
        from ctm.training.sft import SFTConfig, train_sft

        with pytest.raises(ValueError, match="local backend"):
            asyncio.run(train_sft(tmp_path / "pairs.jsonl", config=SFTConfig(method="act")))


@pytest.mark.skipif(not HAS_PEFT, reason="peft not installed")
class TestLocalBackendConsistencyLoRA:
    def make_backend(self):
        backend = LocalBackend(device="cpu", use_lora=True, model_instance=tiny_model())
        backend.setup(model="tiny-gpt2-test", lora=LoRAConfig(rank=4, seed=0))
        return backend

    @pytest.mark.parametrize("loss_fn", CONSISTENCY_LOSS_FNS)
    def test_loss_finite_and_grads_reach_adapter(self, loss_fn):
        backend = self.make_backend()
        datums = [consistency_datum(), consistency_datum(clean=(9, 8, 7, 6), prefix=(2, 3, 4))]

        async def run():
            pending = await backend.submit_forward_backward(datums, loss_fn)
            out = await pending.result()
            grads = [p.grad for p in backend.model.parameters() if p.requires_grad]
            assert any(g is not None and g.abs().sum() > 0 for g in grads)
            await (await backend.submit_optim_step(learning_rate=1e-3, adam=AdamConfig())).result()
            return out

        out = asyncio.run(run())
        assert math.isfinite(out.metrics["loss"])
        assert out.metrics["loss"] >= 0
        assert out.logprobs == []

    def test_activation_consistency_loss_decreases(self):
        # The biased prompt shifts the clean content by two positions, so absolute
        # position embeddings guarantee a nonzero starting loss; training the
        # adapter against the frozen base should shrink it.
        backend = self.make_backend()
        datum = consistency_datum()

        async def run():
            first = await step(backend, [datum], "activation_consistency")
            for _ in range(15):
                last = await step(backend, [datum], "activation_consistency")
            return first, last

        first, last = asyncio.run(run())
        assert first.metrics["loss"] > 0
        assert last.metrics["loss"] < first.metrics["loss"]

    def test_mlp_hooks_are_removed_after_step(self):
        backend = self.make_backend()
        asyncio.run(step(backend, [consistency_datum()], "mlp_consistency"))
        assert backend._mlp_hooks is not None
        assert not backend._mlp_hooks._handles  # removed in the finally block
