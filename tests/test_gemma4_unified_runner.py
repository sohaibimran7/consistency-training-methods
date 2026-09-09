"""Focused contract for direct Gemma 4 unified model resolution."""

from __future__ import annotations

import pytest

from ctm.evals import local_model, runner


def test_direct_hf_gemma4_marker_uses_the_processor_preserving_bridge(monkeypatch):
    captured = {}
    expected = object()

    monkeypatch.setattr(
        runner,
        "effective_provider_generation_config",
        lambda value, **_kwargs: dict(value or {"temperature": 0.0}),
    )

    def fake_bridge(model, *, model_args, generation_config):
        captured["model"] = model
        captured["model_args"] = dict(model_args)
        captured["generation_config"] = dict(generation_config)
        return expected

    monkeypatch.setattr(local_model, "gemma4_unified_hf_model", fake_bridge)

    result = runner.resolve_eval_model(
        model="hf//pinned/gemma-snapshot",
        model_args={
            "device": "cuda:0",
            "dtype": "bfloat16",
            "do_sample": True,
            "gemma4_unified_processor": True,
        },
        generation_config={"temperature": 1.0, "top_p": 0.95},
    )

    assert result is expected
    assert captured == {
        "model": "hf//pinned/gemma-snapshot",
        "model_args": {
            "device": "cuda:0",
            "dtype": "bfloat16",
            "do_sample": True,
            "gemma4_unified_processor": True,
        },
        "generation_config": {"temperature": 1.0, "top_p": 0.95},
    }


def test_direct_hf_gemma4_marker_forbids_language_model_only(monkeypatch):
    monkeypatch.setattr(
        runner,
        "effective_provider_generation_config",
        lambda value, **_kwargs: dict(value or {"temperature": 0.0}),
    )
    with pytest.raises(ValueError, match="full conditional-generation wrapper"):
        runner.resolve_eval_model(
            model="hf/google/gemma-4-12B-it",
            model_args={
                "gemma4_unified_processor": True,
                "hf_language_model_only": True,
            },
        )
