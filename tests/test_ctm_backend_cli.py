"""Tests for the shared --backend CLI plumbing (ctm.backends.cli)."""

import argparse
import asyncio

import pytest
import torch
from tinker import types

from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.backends.local.engine import LocalBackend
from ctm.backends.tinker import TinkerBackend
from ctm.core.config import LoRAConfig


def parse(argv):
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    return parser.parse_args(argv)


class TestBackendCLI:
    def test_default_is_tinker_and_builds_tinker_backend(self):
        args = parse([])
        assert args.backend == "tinker"
        backend = build_backend(args)
        assert isinstance(backend, TinkerBackend)
        assert backend.renderer_source == "tinker"
        assert "tinker" in describe_backend(args)

    def test_local_builds_localbackend_with_options(self):
        args = parse(
            [
                "--backend",
                "local",
                "--local-device",
                "cpu",
                "--local-dtype",
                "float32",
                "--local-sampler",
                "hf",
                "--local-gpu-mem-util",
                "0.3",
            ]
        )
        backend = build_backend(args)
        assert isinstance(backend, LocalBackend)
        assert backend.device == "cpu"
        assert backend.dtype == torch.float32
        assert backend.sampler == "hf"
        assert backend.use_lora is True
        assert backend.renderer_source == "hf"
        assert backend.vllm_options == {"gpu_memory_utilization": 0.3}
        assert "local" in describe_backend(args) and "hf" in describe_backend(args)

    def test_local_defaults_to_vllm_lora(self):
        args = parse(["--backend", "local", "--local-device", "cpu"])
        backend = build_backend(args)
        assert backend.sampler == "vllm"
        assert backend.dtype == torch.bfloat16
        assert backend.use_lora is True

    def test_full_finetune_flag(self):
        args = parse(
            [
                "--backend",
                "local",
                "--local-device",
                "cpu",
                "--local-sampler",
                "hf",
                "--local-full-finetune",
                "--local-trainable-modules",
                "self_attn",
            ]
        )
        backend = build_backend(args)
        assert backend.use_lora is False
        assert backend.full_finetune_modules == ["self_attn"]
        assert "full-finetune" in describe_backend(args)

    def test_trainable_modules_require_full_finetune(self):
        args = parse(["--backend", "local", "--local-trainable-modules", "self_attn"])
        with pytest.raises(ValueError, match="requires --local-full-finetune"):
            build_backend(args)


def test_tinker_training_run_records_renderer_metadata(monkeypatch):
    class Service:
        def create_lora_training_client(self, **kwargs):
            self.kwargs = kwargs
            return object()

    service = Service()
    monkeypatch.setattr("ctm.backends.tinker.model_info.get_recommended_renderer_name", lambda _: "unit-renderer")
    TinkerBackend(service).setup(
        model="unit/model",
        lora=LoRAConfig(rank=4, train_mlp=False, train_attn=True, train_unembed=False, seed=9),
    )
    assert service.kwargs["user_metadata"] == {"renderer_name": "unit-renderer"}
    assert {key: service.kwargs[key] for key in ("rank", "train_mlp", "train_attn", "train_unembed", "seed")} == {
        "rank": 4,
        "train_mlp": False,
        "train_attn": True,
        "train_unembed": False,
        "seed": 9,
    }


def test_tinker_scores_only_completion_tokens_under_reference_prompt():
    class SamplingClient:
        async def compute_logprobs_async(self, model_input):
            assert model_input.to_ints() == [1, 2, 3, 4]
            return [None, -0.1, -0.2, -0.3]

    backend = TinkerBackend()
    backend.model = "unit/model"
    backend._base_sampling_client = SamplingClient()
    scored = asyncio.run(
        backend.score_reference_completions(
            [types.ModelInput.from_ints(tokens=[1, 2])],
            [[3, 4]],
        )
    )
    assert scored == [[-0.2, -0.3]]
