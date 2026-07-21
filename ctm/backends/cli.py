"""Shared CLI plumbing for backend selection.

Used by scripts/train_rlct.py and train_bct.py (and any future
training entry point): one flag group, one builder, no per-script duplication.
"""

import argparse

from ctm.backends.base import TrainingBackend

_DTYPES = ("float32", "bfloat16", "float16")


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("backend", "Training compute backend selection")
    g.add_argument(
        "--backend",
        default="tinker",
        choices=["tinker", "local"],
        help="'tinker' (managed service, default) or 'local' (in-process torch/PEFT "
        "on this machine's GPUs — the Isambard/Vast.ai path)",
    )
    g.add_argument("--local-device", default=None, help="LocalBackend device (default: cuda if available, else cpu)")
    g.add_argument("--local-dtype", default="bfloat16", choices=list(_DTYPES), help="LocalBackend model dtype")
    g.add_argument(
        "--local-sampler",
        default="vllm",
        choices=["vllm", "hf"],
        help="Rollout engine for --backend local: 'vllm' (fast, LoRA hot-reload; "
        "production) or 'hf' (model.generate; correct but slow — debugging). "
        "The engine boots lazily on first rollout, so non-sampling (SFT) runs never touch it",
    )
    g.add_argument(
        "--local-gpu-mem-util",
        type=float,
        default=0.45,
        help="vLLM gpu_memory_utilization — leave headroom for the training model " "when colocated on one GPU",
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
        help="With --local-full-finetune, train only matching parameter groups. Selectors may be "
        "dotted components (self_attn) or full-name globs (*.self_attn.*). Omit to train everything",
    )


def build_backend(
    args: argparse.Namespace,
    *,
    consistency_loss_options: dict | None = None,
    requires_frozen_base: bool = False,
) -> TrainingBackend:
    """Build the concrete backend explicitly selected by the CLI."""

    if args.backend == "tinker":
        from ctm.backends.tinker import TinkerBackend

        return TinkerBackend()
    import torch

    from ctm.backends.local.engine import LocalBackend

    if args.local_trainable_modules and not args.local_full_finetune:
        raise ValueError("--local-trainable-modules requires --local-full-finetune")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.local_dtype]
    return LocalBackend(
        device=args.local_device,
        dtype=dtype,
        use_lora=not args.local_full_finetune,
        sampler=args.local_sampler,
        vllm_options={"gpu_memory_utilization": args.local_gpu_mem_util},
        consistency_loss_options=consistency_loss_options,
        full_finetune_modules=args.local_trainable_modules,
        keep_frozen_base=requires_frozen_base and args.local_full_finetune,
    )


def describe_backend(args: argparse.Namespace) -> str:
    """One-line summary for the pre-run config printout."""
    if args.backend == "tinker":
        return "tinker (managed service)"
    return (
        f"local ({args.local_device or 'auto'}, {args.local_dtype}, "
        f"sampler={args.local_sampler}, "
        f"{'full-finetune' if args.local_full_finetune else 'LoRA'}"
        f"{f', modules={args.local_trainable_modules}' if args.local_trainable_modules else ''})"
    )
