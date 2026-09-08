"""Fail-closed contract tests for uncapped native-HF EOS-only evaluation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ctm.evals import hf_eos_only


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

    from experiments.elephant_aita_ntaflip.no_cap_hf import _eos_ids

    tokenizer = SimpleNamespace(eos_token_id=1)
    model = SimpleNamespace(
        config=SimpleNamespace(
            eos_token_id=[1, 106],
            text_config=SimpleNamespace(eos_token_id=(50, 106)),
        ),
        generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]),
    )

    assert _eos_ids(tokenizer, model) == {1, 50, 106}
