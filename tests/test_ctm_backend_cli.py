"""Tests for the shared --backend CLI plumbing (ctm.backends.cli)."""

import argparse
import asyncio
from types import SimpleNamespace

import pytest
import torch
from tinker import types

from ctm.backends.cli import add_backend_args, build_backend, build_base_generation_backend, describe_backend
from ctm.backends.local.engine import LocalBackend
from ctm.backends.local.rollout_workers import FrozenBaseVLLMBackend, RolloutParallelBackend
from ctm.backends.local.vllm_sampler import VLLMSampler
from ctm.backends.tinker import TinkerBackend, TinkerSamplerHandle
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
                "--local-vllm-language-model-only",
                "--local-hf-language-model-only",
                "--local-vllm-max-num-seqs",
                "256",
                "--local-vllm-max-num-batched-tokens",
                "4096",
                "--local-vllm-max-model-len",
                "32768",
                "--local-vllm-gdn-prefill-backend",
                "triton",
            ]
        )
        backend = build_backend(args)
        assert isinstance(backend, LocalBackend)
        assert backend.device == "cpu"
        assert backend.dtype == torch.float32
        assert backend.sampler == "hf"
        assert backend.use_lora is True
        assert backend.hf_language_model_only is True
        assert backend.renderer_source == "hf"
        assert backend.vllm_options == {
            "gpu_memory_utilization": 0.3,
            "dtype": "float32",
            "language_model_only": True,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096,
            "max_model_len": 32768,
            "gdn_prefill_backend": "triton",
        }
        assert backend.forward_microbatch_max_datums == 8
        assert backend.forward_microbatch_max_tokens == 2048
        assert backend.target_logprob_chunk_size == 32
        assert backend.gradient_checkpointing is False
        assert backend.gradient_checkpointing_layers is None
        assert backend.ppo_clip_epsilon == pytest.approx(0.2)
        assert "local" in describe_backend(args) and "hf" in describe_backend(args)

    def test_local_defaults_to_vllm_lora(self):
        args = parse(["--backend", "local", "--local-device", "cpu"])
        backend = build_backend(args)
        assert backend.sampler == "vllm"
        assert backend.dtype == torch.bfloat16
        assert backend.use_lora is True

    def test_local_vllm_language_model_only_is_opt_in(self):
        backend = build_backend(parse(["--backend", "local", "--local-device", "cpu"]))
        assert backend.vllm_options == {"gpu_memory_utilization": 0.45, "dtype": "bfloat16"}
        assert backend.hf_language_model_only is False

    def test_local_ppo_clip_epsilon_is_forwarded_to_the_training_backend(self):
        backend = build_backend(
            parse(
                [
                    "--backend",
                    "local",
                    "--local-device",
                    "cpu",
                    "--local-ppo-clip-epsilon",
                    "0.125",
                ]
            )
        )

        assert backend.ppo_clip_epsilon == pytest.approx(0.125)

    @pytest.mark.parametrize("value", ["0", "-0.1", "1", "nan", "inf"])
    def test_local_ppo_clip_epsilon_must_be_a_finite_proper_clip_range(self, value):
        with pytest.raises(ValueError, match="--local-ppo-clip-epsilon must be finite and in \\(0, 1\\)"):
            build_backend(parse(["--backend", "local", "--local-ppo-clip-epsilon", value]))

    def test_local_vllm_max_num_seqs_must_be_positive(self):
        args = parse(["--backend", "local", "--local-vllm-max-num-seqs", "0"])
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_backend(args)

    def test_local_vllm_generation_config_and_eager_reach_worker_options(self):
        from ctm.backends.cli import _vllm_options
        args = parse(["--backend", "local", "--local-vllm-generation-config", "vllm", "--local-vllm-enforce-eager"])
        options = _vllm_options(args, worker=True)
        assert options["generation_config"] == "vllm" and options["enforce_eager"] is True
        default = _vllm_options(parse(["--backend", "local"]), worker=True)
        assert "generation_config" not in default and "enforce_eager" not in default

    def test_local_vllm_max_num_batched_tokens_must_be_positive(self):
        args = parse(["--backend", "local", "--local-vllm-max-num-batched-tokens", "0"])
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_backend(args)

    def test_local_vllm_max_model_len_must_be_positive(self):
        args = parse(["--backend", "local", "--local-vllm-max-model-len", "0"])
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_backend(args)

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("--local-gpu-mem-util", "0"),
            ("--local-gpu-mem-util", "1.01"),
            ("--local-rollout-gpu-mem-util", "nan"),
            ("--local-rollout-gpu-mem-util", "-0.1"),
        ],
    )
    def test_vllm_memory_utilization_must_be_in_unit_interval(self, option, value):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            build_backend(parse(["--backend", "local", option, value]))

    @pytest.mark.parametrize(
        "option",
        [
            "--local-forward-microbatch-max-datums",
            "--local-forward-microbatch-max-tokens",
            "--local-target-logprob-chunk-size",
        ],
    )
    def test_forward_microbatch_limits_must_be_positive(self, option):
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_backend(parse(["--backend", "local", option, "0"]))

    def test_forward_microbatch_limits_are_execution_knobs(self):
        backend = build_backend(
            parse(
                [
                    "--backend",
                    "local",
                    "--local-forward-microbatch-max-datums",
                    "3",
                    "--local-forward-microbatch-max-tokens",
                    "777",
                    "--local-target-logprob-chunk-size",
                    "96",
                ]
            )
        )
        assert backend.forward_microbatch_max_datums == 3
        assert backend.forward_microbatch_max_tokens == 777
        assert backend.target_logprob_chunk_size == 96

    def test_gradient_checkpointing_is_an_execution_knob(self):
        backend = build_backend(parse(["--backend", "local", "--local-gradient-checkpointing"]))
        assert backend.gradient_checkpointing is True
        assert backend.gradient_checkpointing_layers is None

    def test_selective_gradient_checkpointing_is_forwarded(self):
        backend = build_backend(
            parse(
                [
                    "--backend",
                    "local",
                    "--local-gradient-checkpointing",
                    "--local-gradient-checkpointing-layers",
                    "16",
                ]
            )
        )

        assert backend.gradient_checkpointing is True
        assert backend.gradient_checkpointing_layers == 16

    def test_selective_gradient_checkpointing_requires_checkpointing_flag(self):
        args = parse(["--backend", "local", "--local-gradient-checkpointing-layers", "16"])
        with pytest.raises(ValueError, match="requires --local-gradient-checkpointing"):
            build_backend(args)

    def test_selective_gradient_checkpointing_layer_count_must_be_positive(self):
        args = parse(
            [
                "--backend",
                "local",
                "--local-gradient-checkpointing",
                "--local-gradient-checkpointing-layers",
                "0",
            ]
        )
        with pytest.raises(ValueError, match="must be a positive integer"):
            build_backend(args)

    def test_local_device_map_and_memory_cap(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        args = parse(
            [
                "--backend",
                "local",
                "--local-device-map",
                "auto",
                "--local-max-memory-per-gpu",
                "45GiB",
            ]
        )
        backend = build_backend(args)

        assert backend.device_map == "auto"
        assert backend.max_memory == {0: "45GiB", 1: "45GiB"}
        assert "device_map=auto" in describe_backend(args)

    def test_local_memory_cap_requires_device_map(self):
        args = parse(["--backend", "local", "--local-max-memory-per-gpu", "45GiB"])
        with pytest.raises(ValueError, match="requires --local-device-map"):
            build_backend(args)

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

    def test_common_builder_wraps_local_backend_with_rollout_workers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7,9")
        status_dir = tmp_path / "worker-status"
        parity_attestation = tmp_path / "qwen35-rollout-worker-parity-attestation.json"
        muse_parity_attestation = tmp_path / "muse-glimmer-rollout-worker-parity-attestation.json"
        args = parse(
            [
                "--backend",
                "local",
                "--local-device",
                "cuda:0",
                "--local-rollout-gpus",
                "1,2",
                "--local-rollout-seed-base",
                "42",
                "--local-gpu-mem-util",
                "0.34",
                "--local-rollout-gpu-mem-util",
                "0.9",
                "--local-vllm-max-num-batched-tokens",
                "8192",
                "--local-rollout-status-dir",
                str(status_dir),
                "--local-qwen35-rollout-parity-attestation",
                str(parity_attestation),
                "--local-muse-rollout-parity-attestation",
                str(muse_parity_attestation),
            ]
        )

        backend = build_backend(args)

        assert isinstance(backend, RolloutParallelBackend)
        assert isinstance(backend.training_backend, LocalBackend)
        assert [(gpu.logical_index, gpu.device_token) for gpu in backend.gpus] == [(1, "7"), (2, "9")]
        assert backend.status_dir == status_dir
        assert backend.training_backend.vllm_options["gpu_memory_utilization"] == 0.34
        assert backend.worker_vllm_options["gpu_memory_utilization"] == 0.9
        assert backend.training_backend.vllm_options["dtype"] == "bfloat16"
        assert backend.worker_vllm_options["dtype"] == "bfloat16"
        assert backend.training_backend.vllm_options["max_num_batched_tokens"] == 8192
        assert backend.worker_vllm_options["max_num_batched_tokens"] == 8192
        assert backend.worker_vllm_options["seed"] == 42
        assert backend.qwen35_rollout_parity_attestation == parity_attestation
        assert backend.muse_rollout_parity_attestation == muse_parity_attestation
        assert "rollout_workers=1,2" in describe_backend(args)

    def test_base_generation_uses_every_explicit_gpu_without_a_model_coordinator(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
        args = parse(
            [
                "--backend",
                "local",
                "--local-gpu-mem-util",
                "0.34",
                "--local-rollout-gpu-mem-util",
                "0.9",
                "--local-rollout-gpus",
                "0,1",
                "--local-rollout-seed-base",
                "42",
            ]
        )

        backend = build_base_generation_backend(
            args,
            model="unit/model",
            default_status_dir=tmp_path / "workers",
        )

        assert isinstance(backend, FrozenBaseVLLMBackend)
        assert [(gpu.logical_index, gpu.device_token) for gpu in backend.gpus] == [(0, "GPU-a"), (1, "GPU-b")]
        assert backend.engine_kwargs["gpu_memory_utilization"] == 0.9
        assert backend.engine_kwargs["dtype"] == "bfloat16"
        assert backend.engine_kwargs["seed"] == 42
        assert not hasattr(backend, "model")

    def test_rollout_worker_llm_receives_the_explicit_local_dtype(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
        backend = build_backend(
            parse(
                [
                    "--backend",
                    "local",
                    "--local-device",
                    "cuda:0",
                    "--local-dtype",
                    "bfloat16",
                    "--local-rollout-gpus",
                    "1",
                    "--local-rollout-status-dir",
                    str(tmp_path / "worker-status"),
                ]
            )
        )
        assert isinstance(backend, RolloutParallelBackend)

        constructed = []

        class RecordingLLM:
            def __init__(self, **kwargs):
                constructed.append(kwargs)

        api = SimpleNamespace(LLM=RecordingLLM, SamplingParams=object, TokensPrompt=object, LoRARequest=object)
        sampler = VLLMSampler(
            model="unit/model",
            enable_lora=False,
            api=api,
            **backend.worker_vllm_options,
        )

        assert constructed[0]["dtype"] == "bfloat16"
        sampler.shutdown()

    @pytest.mark.parametrize("seed", ["-1", str(2**31)])
    def test_rollout_seed_base_must_be_in_supported_range(self, seed):
        args = parse(["--backend", "local", "--local-rollout-seed-base", seed])
        with pytest.raises(ValueError, match="local-rollout-seed-base"):
            build_backend(args)

    def test_rollout_status_override_requires_worker_gpu_list(self, tmp_path):
        args = parse(["--local-rollout-status-dir", str(tmp_path)])
        with pytest.raises(ValueError, match="requires --local-rollout-gpus"):
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


def test_tinker_policy_handle_scores_only_completion_tokens_with_raw_policy():
    class SamplingClient:
        async def compute_logprobs_async(self, model_input):
            assert model_input.to_ints() == [1, 2, 3, 4]
            return [None, -0.1, -0.2, -0.3]

    scored = asyncio.run(
        TinkerSamplerHandle(SamplingClient()).score_completions(
            [types.ModelInput.from_ints(tokens=[1, 2])],
            [[3, 4]],
        )
    )
    assert scored == [[-0.2, -0.3]]
