"""Train On-Policy Consistency Training (OPCT) on paired prompt JSONL.

Each row supplies a frozen-teacher reference prompt and a student variant
prompt.  The student samples online from the variant; an immutable snapshot of
the run-start policy scores that continuation under the reference prompt.

Example:
    python scripts/train_opct.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --data paired-prompts.jsonl \
        --reference-messages-field unbiased_messages \
        --variant-messages-field biased_messages \
        --rollouts-per-prompt 4 --kl-coef 2.0 \
        --experiment-name opct --run-name main
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from ctm.artifacts import plain_file_identity, verify_data_manifest_bindings
from ctm.backends.cli import add_backend_args, build_backend, describe_backend
from ctm.cli_safety import parse_json_object, reject_inline_secrets
from ctm.core.config import AdamConfig, CheckpointConfig, resolve_lora_config
from ctm.training.opct import OPCTConfig, OPCTGenerationConfig, OPCTTrainer, validate_opct_samples


def parse_file_spec(spec: str) -> tuple[Path, int | None]:
    """Parse ``FILE[:N]`` while allowing colons inside the file path."""

    if ":" in spec:
        path_text, possible_limit = spec.rsplit(":", 1)
        try:
            return Path(path_text), int(possible_limit)
        except ValueError:
            pass
    return Path(spec), None


def load_and_combine(file_specs: list[tuple[Path, int | None]], *, interleave: bool) -> list[dict]:
    groups: list[list[dict]] = []
    for path, limit in file_specs:
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if limit is not None and len(rows) >= limit:
                    break
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: each row must be a JSON object")
                rows.append(value)
        print(f"  {path.name}: {len(rows)} pairs")
        groups.append(rows)

    if not interleave:
        return [row for group in groups for row in group]
    output: list[dict] = []
    iterators = [iter(group) for group in groups]
    while iterators:
        for iterator in list(iterators):
            try:
                output.append(next(iterator))
            except StopIteration:
                iterators.remove(iterator)
    return output


def resolve_optimizer_config(
    raw: dict,
    *,
    learning_rate: float | None,
    lr_schedule: str | None,
) -> AdamConfig:
    unknown = sorted(set(raw) - set(AdamConfig.model_fields))
    if unknown:
        raise ValueError(f"optimizer_config has unknown field(s): {unknown}")
    values = {"lr_schedule": "constant", **raw}
    if learning_rate is not None:
        values["learning_rate"] = learning_rate
    if lr_schedule is not None:
        values["lr_schedule"] = lr_schedule
    return AdamConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="On-Policy Consistency Training on paired clean/variant prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Base model used to initialize a fresh student and its run-start teacher",
    )
    parser.add_argument("--data", nargs="+", required=True, metavar="FILE[:N]")
    parser.add_argument(
        "--data-manifest",
        nargs="+",
        type=Path,
        help="Optional one-per-file manifest(s); each must bind the exact corresponding --data bytes",
    )
    parser.add_argument("--interleave", action="store_true", help="Round-robin rows from multiple data files")
    parser.add_argument("--reference-messages-field", default="reference_messages")
    parser.add_argument("--variant-messages-field", default="variant_messages")

    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--wandb-project")

    parser.add_argument("--lora-config", help="JSON object or file with the shared LoRA configuration")
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--optimizer-config", help="JSON object or file with the shared Adam configuration")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=["constant", "linear", "cosine"], default=None)

    parser.add_argument("--rollouts-per-prompt", type=int, default=4, help="Student rollouts k for each prompt pair")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--kl-coef", type=float, default=1.0, help="Reverse-KL coefficient lambda")
    parser.add_argument(
        "--kl-discount-factor",
        type=float,
        default=0.0,
        help="Future-token discount gamma for reverse-KL credit",
    )
    parser.add_argument(
        "--loss-fn",
        choices=["importance_sampling", "ppo"],
        default="importance_sampling",
        help="Policy-gradient loss applied to the reverse-KL token advantages",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Prompt pairs per microbatch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--skip-near-final", type=int, default=0)
    parser.add_argument(
        "--rollout-log",
        choices=["all", "none"],
        default="all",
        help="Persist every sampled student completion, including invalid/skipped samples",
    )
    parser.add_argument(
        "--rollout-dir",
        help="Rollout output directory (default: logs/EXPERIMENT/RUN/rollouts)",
    )
    parser.add_argument(
        "--resume-from",
        help="Tinker student checkpoint to load and freeze as this run's teacher",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-with-optimizer",
        dest="resume_with_optimizer",
        action="store_const",
        const=True,
        default=None,
        help="Force optimizer-state restoration from --resume-from (default: infer from URI)",
    )
    resume_group.add_argument(
        "--no-resume-optimizer",
        dest="resume_with_optimizer",
        action="store_const",
        const=False,
        help="Load only student weights from --resume-from",
    )

    add_backend_args(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-y", "--yes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        file_specs = [parse_file_spec(value) for value in args.data]
        for path, limit in file_specs:
            if not path.is_file():
                raise ValueError(f"data file not found: {path}")
            if limit is not None and limit < 1:
                raise ValueError(f"data limit must be positive: {path}:{limit}")
        for path in args.data_manifest or []:
            if not path.is_file():
                raise ValueError(f"data manifest not found: {path}")
        if args.data_manifest:
            verify_data_manifest_bindings(
                [path for path, _limit in file_specs],
                args.data_manifest,
            )

        raw_lora = parse_json_object(args.lora_config, label="lora_config")
        raw_optimizer = parse_json_object(args.optimizer_config, label="optimizer_config")
        reject_inline_secrets(raw_lora, path="lora_config")
        reject_inline_secrets(raw_optimizer, path="optimizer_config")
        lora = resolve_lora_config(raw_lora, rank=args.lora_rank, seed=args.seed)
        optimizer = resolve_optimizer_config(
            raw_optimizer,
            learning_rate=args.lr,
            lr_schedule=args.lr_schedule,
        )
        if args.lr is not None and args.lr <= 0:
            raise ValueError("--lr must be positive")
        if args.checkpoint_every < 1:
            raise ValueError("--checkpoint-every must be positive")
        if args.skip_near_final < 0:
            raise ValueError("--skip-near-final must be non-negative")
        if args.resume_with_optimizer is not None and not args.resume_from:
            raise ValueError("--resume-with-optimizer/--no-resume-optimizer requires --resume-from")
        if args.resume_from and args.backend == "local":
            raise ValueError(
                "OPCT --resume-from is unavailable with --backend local because its policy handle is live; "
                "use Tinker or start a fresh local run so the teacher remains immutable"
            )
        if args.rollout_dir is not None and not args.rollout_dir.strip():
            raise ValueError("--rollout-dir must be a non-empty path")
        if args.rollout_log == "none" and args.rollout_dir is not None:
            raise ValueError("--rollout-dir requires --rollout-log all")

        print("Loading prompt pairs...")
        samples = load_and_combine(file_specs, interleave=args.interleave)
        config = OPCTConfig(
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            wandb_project=args.wandb_project,
            model=args.model,
            lora=lora,
            optimizer=optimizer,
            generation=OPCTGenerationConfig(
                rollouts_per_prompt=args.rollouts_per_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            ),
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kl_coef=args.kl_coef,
            kl_discount_factor=args.kl_discount_factor,
            loss_fn=args.loss_fn,
            checkpoint=CheckpointConfig(
                save_every_n_steps=args.checkpoint_every,
                save_state=args.save_state,
                skip_near_final_steps=args.skip_near_final,
            ),
            rollout_log=args.rollout_log,
            rollout_dir=args.rollout_dir,
            reference_messages_field=args.reference_messages_field,
            variant_messages_field=args.variant_messages_field,
            run_metadata={
                "data_files": [str(path) for path, _ in file_specs],
                "data_artifacts": [plain_file_identity(path) for path, _ in file_specs],
                "data_manifests": [plain_file_identity(path) for path in (args.data_manifest or [])],
                "interleave": args.interleave,
                "backend": args.backend,
                "method": "opct",
                "teacher_policy": "run_start",
                "resume_from": args.resume_from,
                "resume_with_optimizer": args.resume_with_optimizer,
            },
        )
        validate_opct_samples(samples, config)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    microbatches = (len(samples) + config.batch_size - 1) // config.batch_size
    steps = (
        (microbatches + config.gradient_accumulation_steps - 1) // config.gradient_accumulation_steps * config.n_epochs
    )
    print()
    print(f"Model: {config.model}")
    print("Method: OPCT (paired output consistency)")
    print(f"Backend: {describe_backend(args)}")
    print(f"Experiment: {config.experiment_name} / {config.run_name}")
    print(f"Prompt pairs: {len(samples)}")
    print(
        f"Hyperparams: k={config.generation.rollouts_per_prompt}, lambda={config.kl_coef}, "
        f"gamma={config.kl_discount_factor}, temperature={config.generation.temperature}, "
        f"batch={config.batch_size}, grad_accum={config.gradient_accumulation_steps}, "
        f"epochs={config.n_epochs}, optimizer_steps={steps}"
    )
    print(f"Pair fields: teacher={config.reference_messages_field!r}, " f"student={config.variant_messages_field!r}")
    teacher_source = (
        f"run-start Tinker checkpoint ({args.resume_from})" if args.resume_from else "run-start base policy"
    )
    print(f"Frozen teacher: {teacher_source}")
    if config.rollout_log == "all":
        rollout_dir = config.rollout_dir or f"logs/{config.experiment_name}/{config.run_name}/rollouts"
        print(f"Rollout log: all sampled completions -> {rollout_dir}")
    else:
        print("Rollout log: disabled")
    if args.resume_from:
        optimizer_resume = "auto" if args.resume_with_optimizer is None else str(args.resume_with_optimizer).lower()
        print(f"Student checkpoint: {args.resume_from} (optimizer restore: {optimizer_resume})")
    print("Policy refresh: after every optimizer update (fully on-policy)")
    if args.dry_run:
        print("Dry run complete; no backend was initialized.")
        return
    if not args.yes and input("Proceed with OPCT training? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    trainer = OPCTTrainer(
        config=config,
        backend=build_backend(args, requires_frozen_base=True),
        resume_from=args.resume_from,
        resume_with_optimizer=args.resume_with_optimizer,
    )
    final_checkpoint = asyncio.run(trainer.train(samples))
    print(f"CTM_FINAL_CHECKPOINT={final_checkpoint}")


if __name__ == "__main__":
    main()
