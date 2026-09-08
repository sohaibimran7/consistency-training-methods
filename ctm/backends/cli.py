"""Shared command-line plumbing for local and managed training backends.

The local phase-shared path is deliberately opt-in.  It keeps vLLM rollout
workers and HF/PEFT training replicas on the same explicitly allocated GPUs,
with a sleep/wake boundary between those two phases.  The topology resolver is
CPU-only so a launch script can inspect the exact plan before it starts a model
or a worker process.
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
_DEFAULT_FORWARD_MICROBATCH_MAX_TOKENS = 2_048
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
    """Resolved phase-shared local execution configuration."""

    topology: PhaseSharedTopology
    coordinator_device: str
    rollout: RolloutParallelCLIConfig
    replica_start_timeout_seconds: float
    replica_command_timeout_seconds: float
    replica_shutdown_timeout_seconds: float


def _phase_shared_requested(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "local_phase_shared", False))


def add_backend_args(
    parser: argparse.ArgumentParser,
    *,
    enable_phase_shared: bool = False,
) -> None:
    """Add common backend and execution arguments to ``parser``.

    Phase sharing requires explicit lifecycle scheduling in the owning
    training loop.  Keep its command-line surface disabled unless that caller
    has implemented and tested those boundaries; merely composing the backend
    wrappers is not sufficient.
    """

    group = parser.add_argument_group("backend", "Training compute backend selection")
    group.add_argument(
        "--backend",
        default="tinker",
        choices=["tinker", "local"],
        help="'tinker' (managed service, default) or 'local' (torch/PEFT on the allocated GPUs)",
    )
    group.add_argument("--local-device", default=None, help="LocalBackend device (default: cuda if available)")
    group.add_argument("--local-dtype", default="bfloat16", choices=list(_DTYPES), help="Local model dtype")
    group.add_argument(
        "--local-sampler",
        default="vllm",
        choices=["vllm", "hf"],
        help="Local rollout engine: vLLM (production) or HF generate (diagnostic)",
    )
    group.add_argument(
        "--local-gpu-mem-util",
        type=float,
        default=0.45,
        help="vLLM GPU-memory fraction; leave training headroom when colocated",
    )
    group.add_argument(
        "--local-rollout-gpu-mem-util",
        type=float,
        default=None,
        help="vLLM memory fraction for dedicated rollout workers (defaults to --local-gpu-mem-util)",
    )
    group.add_argument(
        "--local-vllm-max-num-seqs",
        type=int,
        default=None,
        help="Optional vLLM scheduler sequence cap",
    )
    group.add_argument(
        "--local-vllm-max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional vLLM scheduler token-budget cap per iteration",
    )
    group.add_argument(
        "--local-vllm-max-model-len",
        type=int,
        default=None,
        help="Optional vLLM context cap",
    )
    if enable_phase_shared:
        group.add_argument(
            "--local-vllm-sleep-during-training",
            action="store_true",
            help="Enable vLLM sleep mode so its engine releases memory during training",
        )
        group.add_argument(
            "--local-phase-shared",
            action="store_true",
            help=(
                "Run exact replicated LoRA training on the same GPUs as vLLM rollouts. Requires local/vLLM execution and an explicit CUDA_VISIBLE_DEVICES allocation."
            ),
        )
        group.add_argument(
            "--local-training-gpus",
            default=None,
            help=(
                "Logical training GPU indices relative to CUDA_VISIBLE_DEVICES, or 'all'. Only with --local-phase-shared; defaults to every visible GPU."
            ),
        )
    else:
        parser.set_defaults(
            local_vllm_sleep_during_training=False,
            local_phase_shared=False,
            local_training_gpus=None,
        )
    if enable_phase_shared:
        group.add_argument(
            "--local-rollout-gpus",
            default=None,
            help=(
                "Logical vLLM worker GPU indices relative to CUDA_VISIBLE_DEVICES. With phase sharing these may overlap training GPUs and default to all visible GPUs."
            ),
        )
        group.add_argument(
            "--local-rollout-status-dir",
            default=None,
            help="Persistent rollout worker status, adapter, and IPC directory",
        )
        group.add_argument(
            "--local-rollout-start-timeout-seconds",
            type=float,
            default=1_800.0,
            help="Timeout for each vLLM rollout worker startup",
        )
        group.add_argument(
            "--local-rollout-request-timeout-seconds",
            type=float,
            default=7_200.0,
            help="Timeout for one rollout or adapter refresh command",
        )
        group.add_argument(
            "--local-replica-start-timeout-seconds",
            type=float,
            default=None,
            help="Replicated trainer startup/rendezvous timeout (defaults to rollout startup timeout)",
        )
        group.add_argument(
            "--local-replica-command-timeout-seconds",
            type=float,
            default=None,
            help="Replicated trainer command timeout (defaults to rollout request timeout)",
        )
        group.add_argument(
            "--local-replica-shutdown-timeout-seconds",
            type=float,
            default=30.0,
            help="Replicated trainer shutdown timeout",
        )
        group.add_argument(
            "--local-rollout-seed-base",
            type=int,
            default=None,
            help="Base vLLM engine seed; each rollout worker receives a distinct derived seed",
        )
    else:
        parser.set_defaults(
            local_rollout_gpus=None,
            local_rollout_status_dir=None,
            local_rollout_start_timeout_seconds=1_800.0,
            local_rollout_request_timeout_seconds=7_200.0,
            local_replica_start_timeout_seconds=None,
            local_replica_command_timeout_seconds=None,
            local_replica_shutdown_timeout_seconds=30.0,
            local_rollout_seed_base=None,
        )
    group.add_argument(
        "--local-forward-microbatch-max-datums",
        type=int,
        default=_DEFAULT_FORWARD_MICROBATCH_MAX_DATUMS,
        help="Maximum datums in one internal local training/scoring forward",
    )
    group.add_argument(
        "--local-forward-microbatch-max-tokens",
        type=int,
        default=_DEFAULT_FORWARD_MICROBATCH_MAX_TOKENS,
        help="Maximum padded token slots in one internal local training/scoring forward",
    )
    group.add_argument(
        "--local-target-logprob-chunk-size",
        type=int,
        default=_DEFAULT_TARGET_LOGPROB_CHUNK_SIZE,
        help="Selected target-token positions processed per exact logprob workspace chunk",
    )
    group.add_argument(
        "--local-gradient-checkpointing",
        action="store_true",
        help="Recompute backbone activations during backward to reduce memory use",
    )
    group.add_argument(
        "--local-gradient-checkpointing-layers",
        type=int,
        default=None,
        help="Checkpoint only the first N backbone layers (requires --local-gradient-checkpointing)",
    )
    group.add_argument(
        "--local-full-finetune",
        action="store_true",
        help="Disable LoRA and train ordinary model parameters; incompatible with phase sharing",
    )
    group.add_argument(
        "--local-trainable-modules",
        nargs="+",
        metavar="SELECTOR",
        help="With full fine-tuning, train only matching parameter groups",
    )


def _validate_positive_int(value: object, *, option: str, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{option} must be a positive integer")


def _validate_positive_seconds(value: object, *, option: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{option} must be finite and positive")


def _validate_local_execution_args(args: argparse.Namespace) -> None:
    """Reject impossible local-mode combinations before any CUDA access."""

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
        _validate_positive_int(value, option=option, allow_none=True)
    if args.local_gradient_checkpointing_layers is not None and not args.local_gradient_checkpointing:
        raise ValueError("--local-gradient-checkpointing-layers requires --local-gradient-checkpointing")
    if args.local_trainable_modules and not args.local_full_finetune:
        raise ValueError("--local-trainable-modules requires --local-full-finetune")
    if args.local_vllm_sleep_during_training:
        if args.backend != "local":
            raise ValueError("--local-vllm-sleep-during-training requires --backend local")
        if args.local_sampler != "vllm":
            raise ValueError("--local-vllm-sleep-during-training requires --local-sampler vllm")
    if _phase_shared_requested(args):
        if args.backend != "local":
            raise ValueError("--local-phase-shared requires --backend local")
        if args.local_sampler != "vllm":
            raise ValueError("--local-phase-shared requires --local-sampler vllm")
        if args.local_full_finetune:
            raise ValueError("--local-phase-shared requires LoRA; it is incompatible with --local-full-finetune")
        if args.local_trainable_modules:
            raise ValueError("--local-phase-shared requires LoRA and cannot use --local-trainable-modules")
    elif args.local_training_gpus is not None:
        raise ValueError("--local-training-gpus requires --local-phase-shared")

    for option, value in (
        ("--local-rollout-start-timeout-seconds", args.local_rollout_start_timeout_seconds),
        ("--local-rollout-request-timeout-seconds", args.local_rollout_request_timeout_seconds),
        ("--local-replica-shutdown-timeout-seconds", args.local_replica_shutdown_timeout_seconds),
    ):
        _validate_positive_seconds(value, option=option)
    for option, value in (
        ("--local-replica-start-timeout-seconds", args.local_replica_start_timeout_seconds),
        ("--local-replica-command-timeout-seconds", args.local_replica_command_timeout_seconds),
    ):
        if value is not None:
            _validate_positive_seconds(value, option=option)
    if args.local_rollout_seed_base is not None and (
        isinstance(args.local_rollout_seed_base, bool)
        or not isinstance(args.local_rollout_seed_base, int)
        or not 0 <= args.local_rollout_seed_base <= 2**31 - 1
    ):
        raise ValueError("--local-rollout-seed-base must be an integer in [0, 2147483647]")


def _vllm_options(args: argparse.Namespace, *, worker: bool) -> dict[str, object]:
    """Build vLLM options shared by the coordinator and worker pool."""

    memory_utilization = args.local_gpu_mem_util
    if worker and args.local_rollout_gpu_mem_util is not None:
        memory_utilization = args.local_rollout_gpu_mem_util
    options: dict[str, object] = {"gpu_memory_utilization": memory_utilization}
    if args.local_vllm_sleep_during_training or _phase_shared_requested(args):
        options["enable_sleep_mode"] = True
    for option, value in (
        ("max_num_seqs", args.local_vllm_max_num_seqs),
        ("max_num_batched_tokens", args.local_vllm_max_num_batched_tokens),
        ("max_model_len", args.local_vllm_max_model_len),
    ):
        if value is not None:
            options[option] = value
    if worker and args.local_rollout_seed_base is not None:
        options["seed"] = args.local_rollout_seed_base
    return options


def resolve_phase_shared_args(args: argparse.Namespace) -> PhaseSharedCLIConfig | None:
    """Resolve phase-shared GPU topology without probing CUDA."""

    _validate_local_execution_args(args)
    if not _phase_shared_requested(args):
        return None

    from ctm.backends.local.phase_shared import resolve_phase_shared_topology
    from ctm.backends.local.rollout_workers import RolloutGPU

    topology = resolve_phase_shared_topology(
        train_gpus_spec=args.local_training_gpus,
        rollout_gpus_spec=args.local_rollout_gpus,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allow_overlap=True,
    )
    coordinator_device = f"cuda:{topology.coordinator.logical_index}"
    if args.local_device is not None and str(args.local_device).strip() != coordinator_device:
        raise ValueError(
            f"--local-phase-shared chooses rank 0 from the first --local-training-gpus entry; --local-device must be omitted or equal {coordinator_device!r}, got {args.local_device!r}"
        )
    # This is deterministic metadata derived from the selected topology, not
    # implicit CUDA discovery.  Existing launchers use it in their run record.
    args.local_device = coordinator_device

    status_override = args.local_rollout_status_dir
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
    replica_start_timeout = float(
        start_timeout if args.local_replica_start_timeout_seconds is None else args.local_replica_start_timeout_seconds
    )
    replica_command_timeout = float(
        request_timeout
        if args.local_replica_command_timeout_seconds is None
        else args.local_replica_command_timeout_seconds
    )
    return PhaseSharedCLIConfig(
        topology=topology,
        coordinator_device=coordinator_device,
        rollout=RolloutParallelCLIConfig(
            gpus=tuple(RolloutGPU(gpu.logical_index, gpu.device_token) for gpu in topology.rollout_gpus),
            status_dir=status_dir,
            start_timeout_seconds=start_timeout,
            request_timeout_seconds=request_timeout,
        ),
        replica_start_timeout_seconds=replica_start_timeout,
        replica_command_timeout_seconds=replica_command_timeout,
        replica_shutdown_timeout_seconds=float(args.local_replica_shutdown_timeout_seconds),
    )


def resolve_rollout_parallel_args(args: argparse.Namespace) -> RolloutParallelCLIConfig | None:
    """Resolve the historical dedicated rollout-worker mode or phase-sharing pool."""

    _validate_local_execution_args(args)
    phase_shared = resolve_phase_shared_args(args)
    if phase_shared is not None:
        return phase_shared.rollout

    spec = args.local_rollout_gpus
    status_override = args.local_rollout_status_dir
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
        raise ValueError(
            "parallel rollout needs --local-rollout-status-dir when the command has no experiment/run name"
        )
    return RolloutParallelCLIConfig(
        gpus=gpus,
        status_dir=Path(status_override or Path("logs") / experiment_name / run_name / "rollout_workers"),
        start_timeout_seconds=float(args.local_rollout_start_timeout_seconds),
        request_timeout_seconds=float(args.local_rollout_request_timeout_seconds),
    )


def build_backend(
    args: argparse.Namespace,
    *,
    consistency_loss_options: dict | None = None,
    requires_frozen_base: bool = False,
) -> TrainingBackend:
    """Build the explicit backend, adding worker/replica wrappers when requested."""

    _validate_local_execution_args(args)
    if args.backend == "tinker":
        from ctm.backends.tinker import TinkerBackend

        return TinkerBackend()

    import torch

    from ctm.backends.local.engine import LocalBackend

    phase_shared = resolve_phase_shared_args(args)
    rollout_parallel = phase_shared.rollout if phase_shared is not None else resolve_rollout_parallel_args(args)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.local_dtype]
    local_backend_kwargs = {
        "dtype": dtype,
        "use_lora": not args.local_full_finetune,
        "sampler": args.local_sampler,
        "ppo_clip_epsilon": PPO_CLIP_EPSILON,
        "vllm_options": _vllm_options(args, worker=False),
        "consistency_loss_options": consistency_loss_options,
        "full_finetune_modules": args.local_trainable_modules,
        "keep_frozen_base": requires_frozen_base and args.local_full_finetune,
        "gradient_checkpointing": args.local_gradient_checkpointing,
        "gradient_checkpointing_layers": args.local_gradient_checkpointing_layers,
        "forward_microbatch_max_datums": args.local_forward_microbatch_max_datums,
        "forward_microbatch_max_tokens": args.local_forward_microbatch_max_tokens,
        "target_logprob_chunk_size": args.local_target_logprob_chunk_size,
    }
    backend = LocalBackend(
        device=phase_shared.coordinator_device if phase_shared is not None else args.local_device,
        **local_backend_kwargs,
    )
    if phase_shared is None and rollout_parallel is None:
        return backend

    from ctm.backends.local.rollout_workers import RolloutParallelBackend

    if phase_shared is not None:
        from ctm.backends.local.replicated import LocalBackendConstructorSpec, ReplicatedTrainingBackend

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
    )


def describe_backend(args: argparse.Namespace) -> str:
    """Return a stable, concise summary for the pre-run log."""

    if args.backend == "tinker":
        return "tinker (managed service)"
    phase_shared = resolve_phase_shared_args(args)
    if phase_shared is not None:
        topology = phase_shared.topology
        train = ",".join(str(gpu.logical_index) for gpu in topology.train_gpus)
        rollout = ",".join(str(gpu.logical_index) for gpu in topology.rollout_gpus)
        overlap = ",".join(str(gpu.logical_index) for gpu in topology.overlap) or "none"
        return (
            "local phase-shared ("
            f"rank0={phase_shared.coordinator_device}, train=[{train}], rollout=[{rollout}], "
            f"overlap=[{overlap}], rollout_status={phase_shared.rollout.status_dir}, "
            f"rollout_timeouts={phase_shared.rollout.start_timeout_seconds:g}/"
            f"{phase_shared.rollout.request_timeout_seconds:g}s, "
            f"replica_timeouts={phase_shared.replica_start_timeout_seconds:g}/"
            f"{phase_shared.replica_command_timeout_seconds:g}/"
            f"{phase_shared.replica_shutdown_timeout_seconds:g}s, "
            f"{args.local_dtype}, sampler=vllm, vllm_sleep=training_phase, LoRA)"
        )
    workers = args.local_rollout_gpus
    worker_summary = f", rollout_workers={workers}" if workers is not None else ""
    sleep = ", vllm_sleep=training_phase" if args.local_vllm_sleep_during_training else ""
    mode = "full-finetune" if args.local_full_finetune else "LoRA"
    modules = f", modules={args.local_trainable_modules}" if args.local_trainable_modules else ""
    return f"local ({args.local_device or 'auto'}, {args.local_dtype}, sampler={args.local_sampler}{worker_summary}{sleep}, {mode}{modules})"
