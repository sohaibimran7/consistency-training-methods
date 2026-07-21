"""
Unified SFT-family training script (BCT + internal-consistency methods).

Trains on arbitrary data files with flexible mixing and hyperparameters.

Methods (--method):
    bct    — supervised cross-entropy on {"messages": [...]} rows (default; any backend)
    act    — activation (residual stream) consistency  ┐ paired-prompt rows
    attct  — attention (JSD) consistency                │ {"variant_messages", "reference_messages"};
    mlpct  — MLP post-activation consistency            ┘ --backend local + LoRA only

Usage:
    # List available models
    python scripts/train_bct.py --list-models

    # BCT training on Llama (all file types)
    python scripts/train_bct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data instruct.jsonl:10 bct_cot.jsonl:5 bct_non_cot.jsonl:5 \
        --experiment-name bct_llama

    # With interleaving for mixed batches
    python scripts/train_bct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data instruct.jsonl:10 bct_cot.jsonl:5 bct_non_cot.jsonl:5 \
        --interleave \
        --experiment-name bct_llama_interleaved

    # Attention consistency training on a local GPU (no rollouts in SFT → hf sampler)
    python scripts/train_bct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --method attct \
        --data pairs.jsonl \
        --backend local --local-sampler hf \
        --experiment-name attct_llama
"""

import argparse
import json
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

import asyncio  # noqa: E402

import tinker  # noqa: E402

from ctm.training.sft import METHOD_LOSS_FNS, SFTConfig, train_sft  # noqa: E402
from ctm.artifacts import plain_file_identity  # noqa: E402
from ctm.core.config import (  # noqa: E402
    AdamConfig,
    CheckpointConfig,
    resolve_lora_config,
)
from ctm.backends.cli import add_backend_args, build_backend, describe_backend  # noqa: E402
from ctm.cli_safety import parse_json_object, reject_inline_secrets  # noqa: E402
from ctm.training.consistency_losses import create_consistency_loss  # noqa: E402


def list_available_models() -> list[str]:
    """Query Tinker API for available models."""
    service_client = tinker.ServiceClient()
    capabilities = service_client.get_server_capabilities()
    return [model.model_name for model in capabilities.supported_models if model.model_name is not None]


def print_available_models() -> None:
    """Print available models from Tinker API."""
    print("Querying Tinker API for available models...")
    print()
    try:
        models = list_available_models()
        print("Available models:")
        print("-" * 50)
        for model in sorted(models):
            print(f"  {model}")
        print("-" * 50)
        print(f"Total: {len(models)} models")
    except Exception as e:
        print(f"Error querying models: {e}")
        print()
        print("Common models (may not all be available):")
        print("  meta-llama/Llama-3.1-8B-Instruct")
        print("  meta-llama/Llama-3.1-70B-Instruct")
        print("  gpt-oss-120b")


def parse_file_spec(spec: str) -> tuple[Path, int | None]:
    """Parse file:limit spec. Returns (path, limit) where limit may be None."""
    if ":" in spec:
        path_str, limit_str = spec.rsplit(":", 1)
        return Path(path_str), int(limit_str)
    return Path(spec), None


def load_and_combine(
    file_specs: list[tuple[Path, int | None]],
    interleave: bool,
) -> list[dict]:
    """Load samples from multiple files and combine them."""
    all_file_samples: list[list[dict]] = []
    for path, limit in file_specs:
        samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                samples.append(json.loads(line))
        print(f"  {path.name}: {len(samples)} samples")
        all_file_samples.append(samples)

    if not interleave:
        return [s for samples in all_file_samples for s in samples]

    # Round-robin interleave
    result: list[dict] = []
    iterators = [iter(samples) for samples in all_file_samples]
    while iterators:
        for it in list(iterators):
            try:
                result.append(next(it))
            except StopIteration:
                iterators.remove(it)
    return result


def resolve_optimizer_config(
    raw: dict,
    *,
    learning_rate: float | None,
    lr_schedule: str | None,
) -> AdamConfig:
    """Build the shared Adam configuration, with scalar CLI overrides."""

    unknown = sorted(set(raw) - set(AdamConfig.model_fields))
    if unknown:
        raise ValueError(f"optimizer_config has unknown field(s): {unknown}")
    values = dict(raw)
    if learning_rate is not None:
        values["learning_rate"] = learning_rate
    if lr_schedule is not None:
        values["lr_schedule"] = lr_schedule
    return AdamConfig(**values)


def main():
    parser = argparse.ArgumentParser(
        description="Train BCT, ACT, AttCT, or MLPCT with an explicit compute backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--model", help="Model name")
    parser.add_argument(
        "--method",
        default="bct",
        choices=["bct", "act", "attct", "mlpct"],
        help="Training method: bct (SFT cross-entropy on messages, default) or "
        "act/attct/mlpct (activation/attention/MLP consistency on "
        "variant_messages/reference_messages prompt pairs; requires --backend local)",
    )

    # Data
    parser.add_argument("--data", nargs="+", metavar="FILE[:N]", help="Data files with optional sample limits")
    parser.add_argument(
        "--data-manifest",
        nargs="+",
        type=Path,
        help="Optional provenance manifest(s) associated with --data; paths and hashes are recorded",
    )
    parser.add_argument("--interleave", action="store_true", help="Round-robin interleave samples across files")
    parser.add_argument(
        "--reference-messages-field",
        default="reference_messages",
        help="Reference-prompt field for act/attct/mlpct rows",
    )
    parser.add_argument(
        "--variant-messages-field",
        default="variant_messages",
        help="Variant-prompt field for act/attct/mlpct rows",
    )
    parser.add_argument(
        "--alignment-text-field",
        help="Optional row field containing the exact shared text region to align; "
        "otherwise the complete reference user message must occur in the variant",
    )
    parser.add_argument(
        "--method-config",
        help="JSON object or JSON file containing ACT, AttCT, or MLPCT loss options",
    )

    # Naming
    parser.add_argument("--experiment-name", default="sft_experiment")
    parser.add_argument("--run-name", default="default")
    parser.add_argument("--wandb-project", help="Explicitly enable W&B logging to this project")

    # Hyperparameters
    parser.add_argument(
        "--lora-config",
        help="JSON object or JSON file with rank, alpha, dropout, target_modules, portable component flags, and seed",
    )
    parser.add_argument(
        "--optimizer-config",
        help="JSON object or JSON file with Adam learning_rate, lr_schedule, betas, epsilon, weight decay, and clipping",
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: auto-detect from model)")
    parser.add_argument(
        "--lr-schedule",
        default=None,
        choices=["constant", "linear", "cosine"],
        help="Override optimizer_config.lr_schedule (effective default: linear)",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Accumulate this many microbatches before each optimizer step",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=None, help="Override lora_config.rank (effective default: 8)")
    parser.add_argument("--seed", type=int, default=None, help="Override lora_config.seed")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Save full optimizer state for intermediate checkpoints (for resuming)",
    )
    parser.add_argument(
        "--skip-near-final", type=int, default=0, help="Skip intermediate checkpoints within N steps of final"
    )

    # Resume from checkpoint
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Checkpoint to load before training (tinker://... or file://... for the matching backend)",
    )
    parser.add_argument(
        "--resume-with-optimizer",
        dest="resume_with_optimizer",
        action="store_const",
        const=True,
        default=None,
        help="Force restoring optimizer state on resume (default: infer from URI)",
    )
    parser.add_argument(
        "--no-resume-optimizer",
        dest="resume_with_optimizer",
        action="store_const",
        const=False,
        help="Force weights-only resume (optimizer resets)",
    )

    # Backend
    add_backend_args(parser)

    # Execution
    parser.add_argument("-y", "--yes", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print_available_models()
        return

    if not args.model:
        parser.error("--model is required unless using --list-models")
    if not args.data:
        parser.error("--data is required")
    if args.method != "bct" and args.backend != "local":
        parser.error(
            f"--method {args.method} needs paired forward passes with internal activations "
            "(Tinker doesn't expose them) — pass --backend local"
        )
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be >= 1")
    try:
        method_config = parse_json_object(args.method_config, label="method_config")
        raw_lora_config = parse_json_object(args.lora_config, label="lora_config")
        raw_optimizer_config = parse_json_object(args.optimizer_config, label="optimizer_config")
        reject_inline_secrets(method_config, path="method_config")
        reject_inline_secrets(raw_lora_config, path="lora_config")
        reject_inline_secrets(raw_optimizer_config, path="optimizer_config")
        if args.method == "bct" and method_config:
            raise ValueError("--method-config applies only to act, attct, and mlpct")
        if args.method != "bct":
            create_consistency_loss(METHOD_LOSS_FNS[args.method], method_config)
        lora_config = resolve_lora_config(raw_lora_config, rank=args.lora_rank, seed=args.seed)
        optimizer_config = resolve_optimizer_config(
            raw_optimizer_config,
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
        )
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    # Parse and validate files
    file_specs = [parse_file_spec(spec) for spec in args.data]
    for path, _ in file_specs:
        if not path.exists():
            parser.error(f"File not found: {path}")
    for path in args.data_manifest or []:
        if not path.is_file():
            parser.error(f"Data manifest not found: {path}")

    # Load samples
    print("Loading samples...")
    all_samples = load_and_combine(file_specs, args.interleave)
    n_samples = len(all_samples)
    print(f"Total: {n_samples} samples")

    # If multiple files or limits, write combined data to temp file
    if len(file_specs) > 1 or any(limit is not None for _, limit in file_specs):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for sample in all_samples:
            tmp.write(json.dumps(sample) + "\n")
        tmp.close()
        data_path = Path(tmp.name)
        print(f"Combined data written to {data_path}")
    else:
        data_path = file_specs[0][0]

    # Build config
    config = SFTConfig(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        method=args.method,
        model=args.model,
        lora=lora_config,
        optimizer=optimizer_config,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint=CheckpointConfig(
            save_every_n_steps=args.save_every,
            save_state=args.save_state,
            skip_near_final_steps=args.skip_near_final,
        ),
        reference_messages_field=args.reference_messages_field,
        variant_messages_field=args.variant_messages_field,
        alignment_text_field=args.alignment_text_field,
        method_config=method_config,
        run_metadata={
            "data_files": [str(p) for p, _ in file_specs],
            "data_artifacts": [plain_file_identity(path) for path, _ in file_specs],
            "data_manifests": [plain_file_identity(path) for path in (args.data_manifest or [])],
            "interleave": args.interleave,
            "backend": args.backend,
            "method": args.method,
        },
    )

    # Print summary (ceiling division matches the actual batch loop in train_sft)
    n_microbatches = (n_samples + args.batch_size - 1) // args.batch_size
    n_steps = (n_microbatches + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
    n_steps *= args.epochs
    n_ckpts = n_steps // args.save_every
    print()
    print(f"Model: {config.model}")
    print(f"Method: {config.method}")
    print(f"Backend: {describe_backend(args)}")
    print(f"Experiment: {config.experiment_name} / {config.run_name}")
    seed_str = f", seed={config.lora.seed}" if config.lora.seed is not None else ""
    learning_rate = config.optimizer.learning_rate
    print(
        f"Hyperparams: lr={learning_rate if learning_rate is not None else 'auto'}, "
        f"schedule={config.optimizer.lr_schedule}, microbatch={args.batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps}, "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps}, "
        f"epochs={args.epochs}, lora_rank={config.lora.rank}{seed_str}"
    )
    print(
        "LoRA: "
        f"rank={config.lora.rank}, alpha={config.lora.resolved_alpha}, dropout={config.lora.dropout}, "
        f"targets={config.lora.target_modules or {'mlp': config.lora.train_mlp, 'attn': config.lora.train_attn, 'unembed': config.lora.train_unembed}}"
    )
    print(f"Steps: {n_steps}, checkpoints: ~{n_ckpts} intermediate + 1 final")
    if args.save_state:
        print("Checkpoints: intermediate + final save full state (resumable)")
    else:
        print("Final checkpoint: sampler weights only (pass --save-state for a resumable full-state checkpoint)")
    if args.resume_from:
        print(f"Resuming from: {args.resume_from}")

    if not args.yes:
        if input("\nProceed? (y/n): ").lower() != "y":
            print("Cancelled.")
            return

    # Train
    final_checkpoint = asyncio.run(
        train_sft(
            data_path,
            config,
            resume_from=args.resume_from,
            resume_with_optimizer=args.resume_with_optimizer,
            backend=build_backend(
                args,
                consistency_loss_options=method_config,
                requires_frozen_base=args.method != "bct",
            ),
        )
    )
    print(f"\nDone! Final checkpoint: {final_checkpoint}")
    print(f"CTM_FINAL_CHECKPOINT={final_checkpoint}")


if __name__ == "__main__":
    main()
