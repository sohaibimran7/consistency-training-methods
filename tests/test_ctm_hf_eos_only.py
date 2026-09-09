"""Fail-closed contract tests for uncapped native-HF EOS-only evaluation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ctm.evals import hf_eos_kernel, hf_eos_only


def _set_exact_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_eos_only.RUNTIME_ENV, hf_eos_only.RUNTIME_ENV_VALUE)
    monkeypatch.setenv(hf_eos_only.EXPECTED_INSPECT_ENV, "inspect-test")
    monkeypatch.setenv(hf_eos_only.EXPECTED_TRANSFORMERS_ENV, "transformers-test")
    versions = {"inspect-ai": "inspect-test", "transformers": "transformers-test"}
    monkeypatch.setattr(hf_eos_only.importlib.metadata, "version", versions.__getitem__)


def test_runtime_policy_requires_activation_and_exact_versions(monkeypatch):
    with pytest.raises(hf_eos_only.HFEOSError, match=hf_eos_only.RUNTIME_ENV):
        hf_eos_only.runtime_policy()

    _set_exact_environment(monkeypatch)
    policy = hf_eos_only.runtime_policy()
    assert policy["schema"] == hf_eos_only.RUNTIME_SCHEMA
    assert policy["output_token_cap"] is None
    assert policy["termination"] == "model_eos_only"
    assert policy["standard_transformers_generate"] == "not-invoked"

    monkeypatch.setenv(hf_eos_only.EXPECTED_TRANSFORMERS_ENV, "different")
    with pytest.raises(hf_eos_only.HFEOSError, match="Transformers"):
        hf_eos_only.runtime_policy()


def test_generate_rejects_any_effective_token_cap_before_sampling(monkeypatch):
    _set_exact_environment(monkeypatch)
    provider = SimpleNamespace(do_sample=True)

    with pytest.raises(Exception, match="token cap|token-cap|must be absent"):
        asyncio.run(
            hf_eos_only._generate(
                provider,
                input=[],
                tools=[],
                tool_choice=None,
                config={"temperature": 1.0, "max_tokens": 500},
            )
        )


def test_eos_only_sampler_unions_tokenizer_model_text_and_generation_eos_ids():
    """A tokenizer EOS must not hide a model's chat end-of-turn marker."""

    tokenizer = SimpleNamespace(eos_token_id=1)
    model = SimpleNamespace(
        config=SimpleNamespace(
            eos_token_id=[1, 106],
            text_config=SimpleNamespace(eos_token_id=(50, 106)),
        ),
        generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]),
    )

    assert hf_eos_kernel.eos_ids(tokenizer, model) == {1, 50, 106}


def test_saved_gemma_suppression_uses_standard_processor_before_sampling_warpers_and_returns_eos(monkeypatch):
    """Retain Gemma's saved multimodal-token suppression in the direct loop.

    The first model step ranks a saved suppressed token above EOS. Suppression
    must run before top-k so the direct loop selects EOS and exits without
    calling ``model.generate``.
    """

    torch = pytest.importorskip("torch")
    import transformers.generation as generation
    from transformers.generation import SuppressTokensLogitsProcessor

    class SavedGemmaGenerationConfig:
        eos_token_id = [1, 106, 50]
        pad_token_id = 0

        def to_diff_dict(self):
            return {
                "bos_token_id": 2,
                "do_sample": True,
                "eos_token_id": [1, 106, 50],
                "pad_token_id": 0,
                "suppress_tokens": [258883, 258882],
                "temperature": 1.0,
                "top_k": 64,
                "top_p": 0.95,
                "transformers_version": "5.15.1",
            }

    class GemmaLikeModel:
        config = SimpleNamespace(eos_token_id=[1, 106, 50])

        def __init__(self):
            self.generation_config = SavedGemmaGenerationConfig()
            self.calls = 0
            self.forward_kwargs = []

        def generate(self, *_args, **_kwargs):
            raise AssertionError("the uncapped loop must not call model.generate")

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            past_key_values,
            use_cache,
            return_dict,
            logits_to_keep=None,
        ):
            self.calls += 1
            self.forward_kwargs.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "return_dict": return_dict,
                    "logits_to_keep": logits_to_keep,
                }
            )
            logits = torch.full((1, 1, 258884), -1000.0)
            logits[0, 0, 258883] = 100.0
            logits[0, 0, 258882] = 98.0
            logits[0, 0, 1] = 99.0
            return SimpleNamespace(logits=logits, past_key_values=object())

        __call__ = forward

    model = GemmaLikeModel()
    saved = hf_eos_kernel.saved_generation_config(model)
    processor = hf_eos_kernel.saved_logits_processor(saved, device="cpu")
    assert isinstance(processor, SuppressTokensLogitsProcessor)
    assert processor.suppress_tokens.tolist() == [258883, 258882]

    original = torch.full((1, 258884), -1000.0)
    original[0, 258883] = 100.0
    original[0, 258882] = 98.0
    original[0, 1] = 99.0
    suppressed = SuppressTokensLogitsProcessor([258883, 258882], device="cpu")(
        torch.tensor([[2]]), original
    )
    assert torch.isneginf(suppressed[0, 258883])
    assert torch.isneginf(suppressed[0, 258882])
    assert int(torch.argmax(suppressed, dim=-1).item()) == 1

    order = []

    def recording_warper(name, original):
        class RecordingWarper:
            def __init__(self, *args, **kwargs):
                self.delegate = original(*args, **kwargs)

            def __call__(self, input_ids, scores):
                order.append((name, bool(torch.isneginf(scores[0, 258883]).item())))
                return self.delegate(input_ids, scores)

        return RecordingWarper

    monkeypatch.setattr(
        generation,
        "TemperatureLogitsWarper",
        recording_warper("temperature", generation.TemperatureLogitsWarper),
    )
    monkeypatch.setattr(
        generation,
        "TopKLogitsWarper",
        recording_warper("top_k", generation.TopKLogitsWarper),
    )
    monkeypatch.setattr(
        generation,
        "TopPLogitsWarper",
        recording_warper("top_p", generation.TopPLogitsWarper),
    )

    result = hf_eos_kernel.eos_only_model_generate(
        model,
        input_ids=torch.tensor([[2]]),
        attention_mask=torch.tensor([[1]]),
        tokenizer=SimpleNamespace(eos_token_id=1, pad_token_id=1),
        config={"temperature": 0.5, "top_k": 1, "top_p": 0.95},
        do_sample=True,
        return_dict_in_generate=True,
        output_logits=False,
        output_hidden_states=False,
    )

    assert result.sequences.tolist() == [[2, 1]]
    assert model.calls == 1
    assert model.forward_kwargs[0]["logits_to_keep"] == 1
    assert order == [("temperature", True), ("top_k", True), ("top_p", True)]


@pytest.mark.parametrize(
    ("field", "value"),
    [("bad_words_ids", [[1, 2]]), ("max_length", 32)],
)
def test_eos_only_sampler_fails_closed_for_unreviewed_saved_generation_processors(field, value):
    """Do not silently claim generic GenerationConfig parity."""

    torch = pytest.importorskip("torch")

    class UnsupportedSavedGenerationConfig:
        eos_token_id = 1

        def to_diff_dict(self):
            return {field: value}

    class Model:
        config = SimpleNamespace(eos_token_id=1)
        generation_config = UnsupportedSavedGenerationConfig()

        def __call__(self, **_kwargs):  # pragma: no cover - must fail before model execution
            raise AssertionError("unreviewed saved generation config must fail before model execution")

    with pytest.raises(hf_eos_kernel.HFEOSError, match=field):
        hf_eos_kernel.eos_only_model_generate(
            Model(),
            input_ids=torch.tensor([[2]]),
            attention_mask=torch.tensor([[1]]),
            tokenizer=SimpleNamespace(eos_token_id=1, pad_token_id=1),
            config={"temperature": 1.0, "top_k": 0, "top_p": 1.0},
            do_sample=True,
            return_dict_in_generate=True,
            output_logits=False,
            output_hidden_states=False,
        )
