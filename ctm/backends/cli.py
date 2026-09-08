"""Shared CLI plumbing for backend selection.

Used by scripts/train_rlct.py and train_bct.py (and any future
training entry point): one flag group, one builder, no per-script duplication.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ctm.backends.base import TrainingBackend
from ctm.backends.local.losses import PPO_CLIP_EPSILON

if TYPE_CHECKING:
    from ctm.backends.local.phase_shared import PhaseSharedTopology
    from ctm.backends.local.rollout_workers import RolloutGPU

_DTYPES = ("float32", "bfloat16", "float16")
_DEFAULT_FORWARD_MICROBATCH_MAX_DATUMS = 8
_DEFAULT_FORWARD_MICROBATCH_MAX_TOKENS = 2048
_DEFAULT_TARGET_LOGPROB_CHUNK_SIZE = 32


@dataclass(frozen=True)
class RolloutParallelCLIConfig:
    """Resolved execution-only rollout-worker configuration."""

    gpus: tuple[RolloutGPU, ...]
    status_dir: Path
    start_timeout_seconds: float
    request_timeout_seconds: float


@dataclass(frozen=True)
class PhaseSharedCLIConfig:
    """Resolved, opt-in phase-shared local execution configuration.

    This intentionally describes *logical* GPU indices relative to the
    command's explicit ``CUDA_VISIBLE_DEVICES`` allocation.  The first
    training GPU is rank zero and the only adapter publisher; it is not
    assumed to be logical GPU zero.
    """

    topology: PhaseSharedTopology
    coordinator_device: str
    rollout: RolloutParallelCLIConfig
    replica_start_timeout_seconds: float
    replica_command_timeout_seconds: float
    replica_shutdown_timeout_seconds: float


def _phase_shared_requested(args: argparse.Namespace) -> bool:
    """Whether the caller explicitly selected the phase-shared topology."""

    return bool(getattr(args, "local_phase_shared", False))


def add_backend_args(
    parser: argparse.ArgumentParser,
    *,
    enable_phase_shared: bool = False,
) -> None:
    """Add common backend arguments to ``parser``.

    Phase sharing requires explicit lifecycle scheduling in the owning training
    loop. Keep its shared-GPU/sleep surface disabled unless the caller has
    implemented and tested its sleep/wake boundaries. Dedicated rollout
    workers remain available: they stay resident on GPUs distinct from a
    training coordinator and do not require that lifecycle.
    """

    g = parser.add_argument_group("backend", "Training compute backend selection")
    g.add_argument(
        "--backend",
        default="tinker",
        choices=["tinker", "local"],
        help="'tinker' (managed service, default) or 'local' (in-process torch/PEFT on this machine's GPUs — the Isambard/Vast.ai path)",
    )
    g.add_argument("--local-device", default=None, help="LocalBackend device (default: cuda if available, else cpu)")
    g.add_argument("--local-dtype", default="bfloat16", choices=list(_DTYPES), help="LocalBackend model dtype")
    g.add_argument(
        "--local-sampler",
        default="vllm",
        choices=["vllm", "hf"],
        help="Rollout engine for --backend local: 'vllm' (fast, LoRA hot-reload; production) or 'hf' (model.generate; correct but slow — debugging). The engine boots lazily on first rollout, so non-sampling (SFT) runs never touch it",
    )
    g.add_argument(
        "--local-ppo-clip-epsilon",
        type=float,
        default=PPO_CLIP_EPSILON,
        help="PPO surrogate clip epsilon for LocalBackend (standard default: 0.2)",
    )
    g.add_argument(
        "--local-gpu-mem-util",
        type=float,
        default=0.45,
        help="vLLM gpu_memory_utilization — leave headroom for the training model when colocated on one GPU",
    )
    g.add_argument(
        "--local-rollout-gpu-mem-util",
        type=float,
        default=None,
        help="Dedicated rollout/base-only worker vLLM gpu_memory_utilization. Defaults to --local-gpu-mem-util for compatibility; dedicated worker GPUs can safely use a larger value",
    )
    g.add_argument(
        "--local-vllm-language-model-only",
        action="store_true",
        help="Pass language_model_only=True to vLLM. Use this for hybrid vision-language checkpoints such as Qwen3.5 when the experiment is text-only, so vLLM skips the vision encoder and multimodal profiling.",
    )
    g.add_argument(
        "--local-hf-language-model-only",
        action="store_true",
        help=(
            "Detach supported multimodal vision/connector modules after HF loading while "
            "retaining canonical text-model parameter names. Required for text-only Muse training."
        ),
    )
    g.add_argument(
        "--local-vllm-max-num-seqs",
        type=int,
        default=None,
        help="Optional vLLM scheduler sequence cap. Hybrid/Mamba models may require a value below their available cache-block count for CUDA graph capture.",
    )
    g.add_argument(
        "--local-vllm-max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM scheduler token-budget cap per iteration (max_num_batched_tokens).",
    )
    g.add_argument(
        "--local-vllm-max-model-len",
        type=int,
        default=None,
        help="Optional vLLM context cap. Set this to the experiment's actual prompt-plus-generation bound so hybrid models do not reserve cache for an unused advertised context window.",
    )
    g.add_argument(
        "--local-vllm-gdn-prefill-backend",
        choices=["flashinfer", "triton"],
        default=None,
        help=(
            "Optional vLLM gated-delta-network prefill backend. Use 'triton' on "
            "runtime-only GH200 systems where FlashInfer cannot JIT its GDN kernel."
        ),
    )
    if enable_phase_shared:
        g.add_argument(
            "--local-vllm-sleep-during-training",
            action="store_true",
            help=(
                "Enable vLLM sleep mode and use explicit rollout/training phase barriers. "
                "This is required before a rollout GPU can be reused by a colocated trainer"
            ),
        )
        g.add_argument(
            "--local-phase-shared",
            action="store_true",
            help=(
                "Opt in to exact replicated local LoRA training: vLLM rollout workers sleep, "
                "then one HF/PEFT replica per --local-training-gpus performs the same globally "
                "normalised update. Requires --backend local, --local-sampler vllm, and an "
                "explicit CUDA_VISIBLE_DEVICES allocation."
            ),
        )
        g.add_argument(
            "--local-training-gpus",
            default=None,
            help=(
                "Comma-separated logical GPU indices used by replicated local training, or 'all'. "
                "Only with --local-phase-shared; defaults to every CUDA_VISIBLE_DEVICES entry. "
                "The first listed GPU is rank 0 and publishes adapters."
            ),
        )
    else:
        parser.set_defaults(
            local_vllm_sleep_during_training=False,
            local_phase_shared=False,
            local_training_gpus=None,
        )
    g.add_argument(
        "--local-device-map",
        default=None,
        help="Shard the training model across this process's GPUs with Transformers/Accelerate, for example 'auto'. This changes placement only, not the training method",
    )
    g.add_argument(
        "--local-max-memory-per-gpu",
        default=None,
        help="Per-GPU model-placement cap, for example '45GiB'. Leave activation headroom. Requires --local-device-map",
    )
    g.add_argument(
        "--local-forward-microbatch-max-datums",
        type=int,
        default=_DEFAULT_FORWARD_MICROBATCH_MAX_DATUMS,
        help="Maximum datums in one internal LocalBackend training/scoring forward",
    )
    g.add_argument(
        "--local-forward-microbatch-max-tokens",
        type=int,
        default=_DEFAULT_FORWARD_MICROBATCH_MAX_TOKENS,
        help="Maximum padded input-token slots in one internal LocalBackend training/scoring forward",
    )
    g.add_argument(
        "--local-target-logprob-chunk-size",
        type=int,
        default=_DEFAULT_TARGET_LOGPROB_CHUNK_SIZE,
        help="Selected target-token positions processed by the LM head per exact logprob workspace chunk",
    )
    g.add_argument(
        "--local-gradient-checkpointing",
        action="store_true",
        help="Recompute backbone activations during backward so long individual sequences fit in GPU memory",
    )
    g.add_argument(
        "--local-gradient-checkpointing-layers",
        type=int,
        default=None,
        help="Checkpoint only the first N backbone layers. Requires --local-gradient-checkpointing; omit to checkpoint every layer",
    )
    g.add_argument(
        "--local-full-finetune",
        action="store_true",
        help="Disable LoRA and train ordinary model parameters; incompatible with --local-sampler vllm",
    )
    g.add_argument(
        "--local-trainable-modules",
        nargs="+",
        metavar="SELECTOR",
        help="With --local-full-finetune, train only matching parameter groups. Selectors may be dotted components (self_attn) or full-name globs (*.self_attn.*). Omit to train everything",
    )
    g.add_argument(
        "--local-rollout-gpus",
        default=None,
        help=(
            "Dedicated rollout-worker logical GPU indices relative to CUDA_VISIBLE_DEVICES. "
            "They exclude the training coordinator by default; when phase sharing is explicitly "
            "enabled they may overlap --local-training-gpus."
        ),
    )
    g.add_argument(
        "--local-rollout-status-dir",
        default=None,
        help="Persistent worker status/adapter/IPC directory base (default: logs/EXPERIMENT/RUN/rollout_workers)",
    )
    g.add_argument(
        "--local-rollout-start-timeout-seconds",
        type=float,
        default=1800.0,
        help="Timeout for each rollout worker's vLLM engine startup",
    )
    g.add_argument(
        "--local-rollout-request-timeout-seconds",
        type=float,
        default=7200.0,
        help="Timeout for one rollout or adapter-refresh worker command",
    )
    g.add_argument(
        "--local-rollout-seed-base",
        type=int,
        default=None,
        help=(
            "Base vLLM engine seed for rollout workers. Worker i receives base+i, so independent "
            "engines cannot share vLLM's default RNG stream. If omitted, one random base is drawn "
            "and recorded when each pool is created"
        ),
    )
    g.add_argument(
        "--local-qwen35-rollout-parity-attestation",
        default=None,
        help=(
            "Optional path to the fixed-token Qwen3.5 rollout-worker parity attestation. "
            "When supplied, it overrides the default attestation path under --local-rollout-status-dir."
        ),
    )
    g.add_argument(
        "--local-muse-rollout-parity-attestation",
        default=None,
        help=(
            "Path to the required uncapped EOS-only HF/PEFT↔vLLM LoRA parity attestation "
            "for Muse Glimmer rollout workers. Muse production runs fail closed without it."
        ),
    )
    if enable_phase_shared:
        g.add_argument(
            "--local-replica-start-timeout-seconds",
            type=float,
            default=None,
            help=(
                "Phase-shared replicated-trainer startup/rendezvous timeout. Defaults to "
                "--local-rollout-start-timeout-seconds."
            ),
        )
        g.add_argument(
            "--local-replica-command-timeout-seconds",
            type=float,
            default=None,
            help=(
                "Phase-shared replicated-trainer command timeout. Defaults to "
                "--local-rollout-request-timeout-seconds."
            ),
        )
        g.add_argument(
            "--local-replica-shutdown-timeout-seconds",
            type=float,
            default=30.0,
            help="Phase-shared replicated-trainer per-process shutdown timeout.",
        )
    else:
        parser.set_defaults(
            local_replica_start_timeout_seconds=None,
            local_replica_command_timeout_seconds=None,
            local_replica_shutdown_timeout_seconds=30.0,
        )


def _validate_local_execution_args(args: argparse.Namespace) -> None:
    ppo_clip_epsilon = getattr(args, "local_ppo_clip_epsilon", PPO_CLIP_EPSILON)
    if (
        isinstance(ppo_clip_epsilon, bool)
        or not isinstance(ppo_clip_epsilon, (int, float))
        or not math.isfinite(float(ppo_clip_epsilon))
        or not 0 < ppo_clip_epsilon < 1
    ):
        raise ValueError("--local-ppo-clip-epsilon must be finite and in (0, 1)")
    for option, value in (
        ("--local-gpu-mem-util", args.local_gpu_mem_util),
        ("--local-rollout-gpu-mem-util", args.local_rollout_gpu_mem_util),
    ):
        if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"{option} must be finite and in (0, 1]")
    for option, value in (
        ("--local-vllm-max-num-seqs", args.local_vllm_max_num_seqs),
        ("--local-vllm-max-num-batched-tokens", args.local_vllm_max_num_batched_tokens),
        ("--local-vllm-max-model-len", args.local_vllm_max_model_len),
        ("--local-forward-microbatch-max-datums", args.local_forward_microbatch_max_datums),
        ("--local-forward-microbatch-max-tokens", args.local_forward_microbatch_max_tokens),
        ("--local-target-logprob-chunk-size", args.local_target_logprob_chunk_size),
        ("--local-gradient-checkpointing-layers", args.local_gradient_checkpointing_layers),
    ):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{option} must be a positive integer")
    if args.local_gradient_checkpointing_layers is not None and not args.local_gradient_checkpointing:
        raise ValueError("--local-gradient-checkpointing-layers requires --local-gradient-checkpointing")
    if args.local_trainable_modules and not args.local_full_finetune:
        raise ValueError("--local-trainable-modules requires --local-full-finetune")
    if getattr(args, "local_vllm_sleep_during_training", False):
        if args.backend != "local":
            raise ValueError("--local-vllm-sleep-during-training requires --backend local")
        if args.local_sampler != "vllm":
            raise ValueError("--local-vllm-sleep-during-training requires --local-sampler vllm")
    phase_shared = _phase_shared_requested(args)
    if phase_shared:
        if args.backend != "local":
            raise ValueError("--local-phase-shared requires --backend local")
        if args.local_sampler != "vllm":
            raise ValueError("--local-phase-shared requires --local-sampler vllm")
        if args.local_full_finetune:
            raise ValueError("--local-phase-shared requires LoRA; it is incompatible with --local-full-finetune")
        if args.local_device_map or args.local_max_memory_per_gpu:
            raise ValueError(
                "--local-phase-shared replicates one full trainer per selected GPU; "
                "it is incompatible with --local-device-map and --local-max-memory-per-gpu"
            )
        if args.local_trainable_modules:
            raise ValueError("--local-phase-shared requires LoRA and cannot use --local-trainable-modules")
    elif getattr(args, "local_training_gpus", None) is not None:
        raise ValueError("--local-training-gpus requires --local-phase-shared")

    for option, value in (
        ("--local-rollout-start-timeout-seconds", getattr(args, "local_rollout_start_timeout_seconds", None)),
        ("--local-rollout-request-timeout-seconds", getattr(args, "local_rollout_request_timeout_seconds", None)),
        ("--local-replica-start-timeout-seconds", getattr(args, "local_replica_start_timeout_seconds", None)),
        ("--local-replica-command-timeout-seconds", getattr(args, "local_replica_command_timeout_seconds", None)),
        ("--local-replica-shutdown-timeout-seconds", getattr(args, "local_replica_shutdown_timeout_seconds", None)),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0
        ):
            raise ValueError(f"{option} must be finite and positive")
    rollout_seed_base = getattr(args, "local_rollout_seed_base", None)
    if rollout_seed_base is not None and (
        isinstance(rollout_seed_base, bool)
        or not isinstance(rollout_seed_base, int)
        or not 0 <= rollout_seed_base <= 2**31 - 1
    ):
        raise ValueError("--local-rollout-seed-base must be an integer in [0, 2147483647]")


def _vllm_options(args: argparse.Namespace, *, worker: bool) -> dict[str, object]:
    """Build coordinator or dedicated-worker vLLM options after validation."""

    _validate_local_execution_args(args)
    memory_utilization = args.local_gpu_mem_util
    if worker and args.local_rollout_gpu_mem_util is not None:
        memory_utilization = args.local_rollout_gpu_mem_util
    return {
        "gpu_memory_utilization": memory_utilization,
        # Keep every rollout vLLM instance on the same explicitly selected
        # precision as its local trainer.  In particular, do not let a worker
        # silently inherit vLLM's mutable ``dtype='auto'`` resolution.
        "dtype": args.local_dtype,
        **(
            {"enable_sleep_mode": True}
            if getattr(args, "local_vllm_sleep_during_training", False) or _phase_shared_requested(args)
            else {}
        ),
        **({"language_model_only": True} if args.local_vllm_language_model_only else {}),
        **({"max_num_seqs": args.local_vllm_max_num_seqs} if args.local_vllm_max_num_seqs is not None else {}),
        **(
            {"max_num_batched_tokens": args.local_vllm_max_num_batched_tokens}
            if args.local_vllm_max_num_batched_tokens is not None
            else {}
        ),
        **({"max_model_len": args.local_vllm_max_model_len} if args.local_vllm_max_model_len is not None else {}),
        **(
            {"gdn_prefill_backend": args.local_vllm_gdn_prefill_backend}
            if args.local_vllm_gdn_prefill_backend is not None
            else {}
        ),
        **(
            {"seed": args.local_rollout_seed_base}
            if worker and args.local_rollout_seed_base is not None
            else {}
        ),
    }


def resolve_phase_shared_args(args: argparse.Namespace) -> PhaseSharedCLIConfig | None:
    """Resolve the opt-in phase-shared topology without probing CUDA.

    Unlike the historical rollout-worker resolver, this intentionally permits
    a rollout worker and a replicated trainer to name the same logical GPU.
    Their residency is protected by the vLLM sleep/wake barrier, not by a
    static disjoint-device rule.  The resolver is CPU-only, which lets launch
    scripts attest the exact topology before any model or vLLM engine starts.
    """

    _validate_local_execution_args(args)
    if not _phase_shared_requested(args):
        return None

    from ctm.backends.local.phase_shared import resolve_phase_shared_topology
    from ctm.backends.local.rollout_workers import RolloutGPU

    topology = resolve_phase_shared_topology(
        train_gpus_spec=getattr(args, "local_training_gpus", None),
        rollout_gpus_spec=getattr(args, "local_rollout_gpus", None),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allow_overlap=True,
    )
    coordinator_device = f"cuda:{topology.coordinator.logical_index}"
    requested_device = getattr(args, "local_device", None)
    if requested_device is not None and str(requested_device).strip() != coordinator_device:
        raise ValueError(
            "--local-phase-shared chooses rank 0 from the first --local-training-gpus entry; "
            f"--local-device must be omitted or equal {coordinator_device!r}, got {requested_device!r}"
        )

    # Populate the effective coordinator for pre-run metadata in existing
    # scripts as well as for the actual LocalBackend.  This is not an implicit
    # placement decision: it is the deterministic first entry of the already
    # validated topology above.
    args.local_device = coordinator_device

    status_override = getattr(args, "local_rollout_status_dir", None)
    if status_override is not None and not str(status_override).strip():
        raise ValueError("--local-rollout-status-dir must be a non-empty path")
    experiment_name = getattr(args, "experiment_name", None)
    run_name = getattr(args, "run_name", None)
    if status_override is None and (not experiment_name or not run_name):
        raise ValueError(
            "--local-phase-shared needs --local-rollout-status-dir when the command has no experiment/run name"
        )
    status_dir = Path(status_override or Path("logs") / experiment_name / run_name / "rollout_workers")

    start_timeout = float(args.local_rollout_start_timeout_seconds)
    request_timeout = float(args.local_rollout_request_timeout_seconds)
    if not math.isfinite(start_timeout) or start_timeout <= 0:
        raise ValueError("--local-rollout-start-timeout-seconds must be finite and positive")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("--local-rollout-request-timeout-seconds must be finite and positive")
    replica_start_timeout = float(
        start_timeout
        if getattr(args, "local_replica_start_timeout_seconds", None) is None
        else args.local_replica_start_timeout_seconds
    )
    replica_command_timeout = float(
        request_timeout
        if getattr(args, "local_replica_command_timeout_seconds", None) is None
        else args.local_replica_command_timeout_seconds
    )
    replica_shutdown_timeout = float(getattr(args, "local_replica_shutdown_timeout_seconds", 30.0))

    rollout = RolloutParallelCLIConfig(
        gpus=tuple(RolloutGPU(gpu.logical_index, gpu.device_token) for gpu in topology.rollout_gpus),
        status_dir=status_dir,
        start_timeout_seconds=start_timeout,
        request_timeout_seconds=request_timeout,
    )
    return PhaseSharedCLIConfig(
        topology=topology,
        coordinator_device=coordinator_device,
        rollout=rollout,
        replica_start_timeout_seconds=replica_start_timeout,
        replica_command_timeout_seconds=replica_command_timeout,
        replica_shutdown_timeout_seconds=replica_shutdown_timeout,
    )


def resolve_rollout_parallel_args(args: argparse.Namespace) -> RolloutParallelCLIConfig | None:
    """Validate and resolve common rollout-worker CLI arguments.

    GPU indices are logical within the command's inherited allocation.  The
    resolver never probes CUDA and never permits a worker outside
    ``CUDA_VISIBLE_DEVICES``.
    """

    _validate_local_execution_args(args)
    phase_shared = resolve_phase_shared_args(args)
    if phase_shared is not None:
        return phase_shared.rollout
    spec = getattr(args, "local_rollout_gpus", None)
    status_override = getattr(args, "local_rollout_status_dir", None)
    if spec is None:
        if status_override is not None:
            raise ValueError("--local-rollout-status-dir requires --local-rollout-gpus")
        return None
    if args.backend != "local":
        raise ValueError("--local-rollout-gpus requires --backend local")
    if args.local_sampler != "vllm":
        raise ValueError("--local-rollout-gpus requires --local-sampler vllm")
    if args.local_full_finetune:
        raise ValueError("--local-rollout-gpus requires LoRA; it is incompatible with --local-full-finetune")
    if args.local_device_map:
        raise ValueError("--local-rollout-gpus requires a single coordinator device, not --local-device-map")
    start_timeout = float(args.local_rollout_start_timeout_seconds)
    request_timeout = float(args.local_rollout_request_timeout_seconds)
    if not math.isfinite(start_timeout) or start_timeout <= 0:
        raise ValueError("--local-rollout-start-timeout-seconds must be finite and positive")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("--local-rollout-request-timeout-seconds must be finite and positive")
    if status_override is not None and not str(status_override).strip():
        raise ValueError("--local-rollout-status-dir must be a non-empty path")

    from ctm.backends.local.rollout_workers import resolve_rollout_gpus

    gpus = resolve_rollout_gpus(
        spec,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        coordinator_device=args.local_device,
    )
    experiment_name = getattr(args, "experiment_name", None)
    run_name = getattr(args, "run_name", None)
    if status_override is None and (not experiment_name or not run_name):
        raise ValueError("parallel rollout needs --local-rollout-status-dir when the command has no experiment/run name")
    status_dir = Path(status_override or Path("logs") / experiment_name / run_name / "rollout_workers")
    return RolloutParallelCLIConfig(
        gpus=gpus,
        status_dir=status_dir,
        start_timeout_seconds=start_timeout,
        request_timeout_seconds=request_timeout,
    )


def resolve_base_generation_parallel_args(
    args: argparse.Namespace,
    *,
    default_status_dir: str | Path,
) -> RolloutParallelCLIConfig | None:
    """Resolve coordinator-free workers for frozen-base target generation."""

    _validate_local_execution_args(args)
    if _phase_shared_requested(args):
        raise ValueError("--local-phase-shared is a training topology and cannot be used for frozen-base generation")
    spec = getattr(args, "local_rollout_gpus", None)
    status_override = getattr(args, "local_rollout_status_dir", None)
    if spec is None:
        if status_override is not None:
            raise ValueError("--local-rollout-status-dir requires --local-rollout-gpus")
        return None
    if args.backend != "local":
        raise ValueError("base-only --local-rollout-gpus requires --backend local")
    if args.local_sampler != "vllm":
        raise ValueError("base-only --local-rollout-gpus requires --local-sampler vllm")
    if args.local_full_finetune:
        raise ValueError("base-only target generation is incompatible with --local-full-finetune")
    if args.local_device_map or args.local_max_memory_per_gpu:
        raise ValueError("base-only target generation does not load a training model or use --local-device-map")
    start_timeout = float(args.local_rollout_start_timeout_seconds)
    request_timeout = float(args.local_rollout_request_timeout_seconds)
    if not math.isfinite(start_timeout) or start_timeout <= 0:
        raise ValueError("--local-rollout-start-timeout-seconds must be finite and positive")
    if not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError("--local-rollout-request-timeout-seconds must be finite and positive")
    if status_override is not None and not str(status_override).strip():
        raise ValueError("--local-rollout-status-dir must be a non-empty path")

    from ctm.backends.local.rollout_workers import resolve_base_only_gpus

    gpus = resolve_base_only_gpus(
        spec,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    return RolloutParallelCLIConfig(
        gpus=gpus,
        status_dir=Path(status_override or default_status_dir),
        start_timeout_seconds=start_timeout,
        request_timeout_seconds=request_timeout,
    )


def build_base_generation_backend(
    args: argparse.Namespace,
    *,
    model: str,
    default_status_dir: str | Path,
) -> object:
    """Build a sampler backend specialized for immutable target generation.

    Local vLLM uses no Transformers/PEFT training model. Other backend/sampler
    combinations keep their existing implementation as a compatibility path.
    """

    _validate_local_execution_args(args)
    if _phase_shared_requested(args):
        raise ValueError("--local-phase-shared is a training topology and cannot be used for frozen-base generation")
    if args.backend != "local" or args.local_sampler != "vllm":
        if getattr(args, "local_rollout_gpus", None) is not None:
            raise ValueError("base-only rollout workers require --backend local --local-sampler vllm")
        return build_backend(args)
    if args.local_full_finetune:
        raise ValueError("base-only target generation is incompatible with --local-full-finetune")
    if args.local_device_map or args.local_max_memory_per_gpu:
        raise ValueError("base-only target generation does not load a training model or use --local-device-map")

    parallel = resolve_base_generation_parallel_args(args, default_status_dir=default_status_dir)
    from ctm.backends.local.rollout_workers import FrozenBaseVLLMBackend

    return FrozenBaseVLLMBackend(
        model=model,
        engine_kwargs=_vllm_options(args, worker=True),
        gpus=() if parallel is None else parallel.gpus,
        status_dir=None if parallel is None else parallel.status_dir,
        start_timeout_seconds=(args.local_rollout_start_timeout_seconds if parallel is None else parallel.start_timeout_seconds),
        request_timeout_seconds=(args.local_rollout_request_timeout_seconds if parallel is None else parallel.request_timeout_seconds),
    )


def build_backend(
    args: argparse.Namespace,
    *,
    consistency_loss_options: dict | None = None,
    requires_frozen_base: bool = False,
) -> TrainingBackend:
    """Build the concrete backend explicitly selected by the CLI."""

    phase_shared = resolve_phase_shared_args(args)
    rollout_parallel = phase_shared.rollout if phase_shared is not None else resolve_rollout_parallel_args(args)

    if args.backend == "tinker":
        from ctm.backends.tinker import TinkerBackend

        return TinkerBackend()
    import torch

    from ctm.backends.local.engine import LocalBackend

    if args.local_max_memory_per_gpu and not args.local_device_map:
        raise ValueError("--local-max-memory-per-gpu requires --local-device-map")
    _validate_local_execution_args(args)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.local_dtype]
    max_memory = None
    if args.local_max_memory_per_gpu:
        max_memory = {index: args.local_max_memory_per_gpu for index in range(torch.cuda.device_count())}
    # Keep the rank-zero and child constructor inputs literally identical
    # except for the replicated wrapper's per-rank ``device`` and gradient
    # reducer injection.  In particular, do not capture a live model, sampler,
    # callback, or CUDA object here: child replicas are spawned later and must
    # reconstruct the same portable LocalBackend recipe independently.
    local_backend_kwargs = {
        "dtype": dtype,
        "use_lora": not args.local_full_finetune,
        "sampler": args.local_sampler,
        "ppo_clip_epsilon": getattr(args, "local_ppo_clip_epsilon", PPO_CLIP_EPSILON),
        "device_map": args.local_device_map,
        "max_memory": max_memory,
        "vllm_options": _vllm_options(args, worker=False),
        "consistency_loss_options": consistency_loss_options,
        "full_finetune_modules": args.local_trainable_modules,
        "keep_frozen_base": requires_frozen_base and args.local_full_finetune,
        "hf_language_model_only": args.local_hf_language_model_only,
        "gradient_checkpointing": args.local_gradient_checkpointing,
        "gradient_checkpointing_layers": args.local_gradient_checkpointing_layers,
        "forward_microbatch_max_datums": args.local_forward_microbatch_max_datums,
        "forward_microbatch_max_tokens": args.local_forward_microbatch_max_tokens,
        "target_logprob_chunk_size": args.local_target_logprob_chunk_size,
    }
    backend = LocalBackend(
        device=(phase_shared.coordinator_device if phase_shared is not None else args.local_device),
        **local_backend_kwargs,
    )
    if phase_shared is None and rollout_parallel is None:
        return backend

    from ctm.backends.local.rollout_workers import RolloutParallelBackend

    if phase_shared is not None:
        from ctm.backends.local.replicated import LocalBackendConstructorSpec, ReplicatedTrainingBackend

        # ``LocalBackendConstructorSpec`` validates picklability immediately,
        # before a long run starts or a child process has touched a GPU.  It
        # deliberately contains the same kwargs passed to rank zero above;
        # ReplicatedTrainingBackend injects each child rank's explicit device
        # and distributed gradient SUM hook itself.
        backend = ReplicatedTrainingBackend(
            backend,
            topology=phase_shared.topology,
            child_backend_spec=LocalBackendConstructorSpec(LocalBackend, kwargs=local_backend_kwargs),
            start_timeout_seconds=phase_shared.replica_start_timeout_seconds,
            command_timeout_seconds=phase_shared.replica_command_timeout_seconds,
            shutdown_timeout_seconds=phase_shared.replica_shutdown_timeout_seconds,
        )

    assert rollout_parallel is not None
    return RolloutParallelBackend(
        backend,
        gpus=rollout_parallel.gpus,
        status_dir=rollout_parallel.status_dir,
        worker_vllm_options=_vllm_options(args, worker=True),
        start_timeout_seconds=rollout_parallel.start_timeout_seconds,
        request_timeout_seconds=rollout_parallel.request_timeout_seconds,
        qwen35_rollout_parity_attestation=getattr(args, "local_qwen35_rollout_parity_attestation", None),
        muse_rollout_parity_attestation=getattr(args, "local_muse_rollout_parity_attestation", None),
    )


def describe_backend(args: argparse.Namespace) -> str:
    """One-line summary for the pre-run config printout."""
    if args.backend == "tinker":
        return "tinker (managed service)"
    phase_shared = resolve_phase_shared_args(args)
    if phase_shared is not None:
        topology = phase_shared.topology
        train = ",".join(str(gpu.logical_index) for gpu in topology.train_gpus)
        rollout = ",".join(str(gpu.logical_index) for gpu in topology.rollout_gpus)
        overlap = ",".join(str(gpu.logical_index) for gpu in topology.overlap) or "none"
        worker_memory = getattr(args, "local_rollout_gpu_mem_util", None) or args.local_gpu_mem_util
        return (
            "local phase-shared ("
            f"rank0={phase_shared.coordinator_device}, train=[{train}], rollout=[{rollout}], "
            f"overlap=[{overlap}], rollout_status={phase_shared.rollout.status_dir}, "
            f"rollout_timeouts={phase_shared.rollout.start_timeout_seconds:g}/{phase_shared.rollout.request_timeout_seconds:g}s, "
            f"replica_timeouts={phase_shared.replica_start_timeout_seconds:g}/"
            f"{phase_shared.replica_command_timeout_seconds:g}/"
            f"{phase_shared.replica_shutdown_timeout_seconds:g}s, "
            f"{args.local_dtype}, sampler=vllm, worker_gpu_mem={worker_memory}, "
            "vllm_sleep=training_phase, LoRA)"
        )
    device_map = getattr(args, "local_device_map", None)
    rollout_gpus = getattr(args, "local_rollout_gpus", None)
    placement = f", device_map={device_map}" if device_map else ""
    worker_memory = getattr(args, "local_rollout_gpu_mem_util", None)
    rollout = f", rollout_workers={rollout_gpus}, worker_gpu_mem={worker_memory or args.local_gpu_mem_util}" if rollout_gpus is not None else ""
    sleep = ", vllm_sleep=training_phase" if getattr(args, "local_vllm_sleep_during_training", False) else ""
    return f"local ({args.local_device or 'auto'}, {args.local_dtype}, sampler={args.local_sampler}{placement}{rollout}{sleep}, {'full-finetune' if args.local_full_finetune else 'LoRA'}{f', modules={args.local_trainable_modules}' if args.local_trainable_modules else ''})"


def describe_base_generation_backend(args: argparse.Namespace) -> str:
    """Stable progress identity text for frozen-base target generation."""

    if args.backend != "local" or args.local_sampler != "vllm":
        return describe_backend(args)
    workers = getattr(args, "local_rollout_gpus", None)
    memory = getattr(args, "local_rollout_gpu_mem_util", None) or args.local_gpu_mem_util
    placement = f"logical_workers={workers}" if workers is not None else "in_process"
    return f"local frozen-base vllm ({placement}, gpu_memory_utilization={memory})"
