"""Tests for the shared --backend CLI plumbing (ctm.backends.cli)."""

import argparse

import torch

from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.backends.local.engine import LocalBackend


def parse(argv):
    parser = argparse.ArgumentParser()
    add_backend_args(parser)
    return parser.parse_args(argv)


class TestBackendCLI:
    def test_default_is_tinker_and_builds_none(self):
        args = parse([])
        assert args.backend == "tinker"
        assert build_backend(args) is None  # loop default = TinkerBackend
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
        assert backend.vllm_options == {"gpu_memory_utilization": 0.3}
        assert "local" in describe_backend(args) and "hf" in describe_backend(args)

    def test_local_defaults_to_vllm_lora(self):
        args = parse(["--backend", "local", "--local-device", "cpu"])
        backend = build_backend(args)
        assert backend.sampler == "vllm"
        assert backend.dtype == torch.bfloat16
        assert backend.use_lora is True

    def test_full_finetune_flag(self):
        args = parse(["--backend", "local", "--local-device", "cpu", "--local-sampler", "hf", "--local-full-finetune"])
        backend = build_backend(args)
        assert backend.use_lora is False
        assert "full-finetune" in describe_backend(args)
