"""Run an upstream Inspect task factory against one model/checkpoint."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from ctm.cli_safety import reject_inline_secrets
from ctm.evals.runner import (
    normalize_generation_config,
    parse_json_object,
    run_task_evals,
    validate_tinker_generation_config,
)


def _command_summary(argv: list[str]) -> str:
    return "python scripts/run_evals.py " + " ".join(shlex.quote(arg) for arg in argv)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run an upstream Inspect task factory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task-factory",
        required=True,
        help="Import path in module:callable form, e.g. benchmark.tasks:create_tasks",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Inspect model spec, e.g. openai/gpt-4.1-mini")
    model_group.add_argument("--tinker-base-model", help="Untrained base model sampled through Tinker")
    model_group.add_argument("--tinker-checkpoint", help="Saved tinker:// sampler-weights path")
    model_group.add_argument("--local-checkpoint", help="Saved file:// LocalBackend LoRA checkpoint")
    parser.add_argument("--base-model", help="Optional expected base model; verified against checkpoint metadata")
    parser.add_argument(
        "--renderer-name", help="Optional expected Tinker renderer; verified against checkpoint metadata"
    )
    parser.add_argument(
        "--task-args",
        help="Inline JSON object or JSON file passed to the task factory",
    )
    parser.add_argument("--model-args", help="Inline JSON object or JSON file passed to Inspect get_model")
    parser.add_argument(
        "--metadata",
        help="Inline JSON object or JSON file recorded in every Inspect eval log",
    )
    parser.add_argument(
        "--generation-config",
        help="Inline JSON object or JSON file parsed as Inspect GenerateConfig for either model path",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Preserve structured reasoning in Tinker model outputs",
    )
    parser.add_argument("--log-dir", default="logs/evals")
    parser.add_argument(
        "--limit",
        type=int,
        help="Source-sample cap per task; epochs, solvers, and graders can still make multiple model calls",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and print the run without constructing tasks or models"
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Run after printing the exact command")
    args = parser.parse_args(argv)

    if (args.model or args.tinker_base_model) and args.base_model:
        parser.error("--base-model applies only to saved checkpoints")
    if args.model and args.renderer_name:
        parser.error("--renderer-name applies only to Tinker models")
    if args.model and args.include_reasoning:
        parser.error("--include-reasoning applies only to Tinker models")
    if args.local_checkpoint and args.renderer_name:
        parser.error("--renderer-name applies only to Tinker models")
    if args.local_checkpoint and args.include_reasoning:
        parser.error("--include-reasoning applies only to Tinker models")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be >= 1")

    try:
        task_args = parse_json_object(args.task_args, label="task_args")
        model_args = parse_json_object(args.model_args, label="model_args")
        metadata = parse_json_object(args.metadata, label="metadata")
        generation_config = parse_json_object(args.generation_config, label="generation_config")
        for label, value in (
            ("task_args", task_args),
            ("model_args", model_args),
            ("metadata", metadata),
            ("generation_config", generation_config),
        ):
            reject_inline_secrets(value, path=label)
        generation_config = normalize_generation_config(generation_config)
        if (args.tinker_base_model or args.tinker_checkpoint) and model_args:
            raise ValueError("--model-args applies only to ordinary Inspect providers and local checkpoints")
        if args.tinker_base_model or args.tinker_checkpoint:
            validate_tinker_generation_config(generation_config)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    print("\nExact eval command:")
    print(f"  {_command_summary(effective_argv)}")
    print("\nResolved parameters:")
    print(f"  task_factory={args.task_factory}")
    print(f"  model={args.model or args.tinker_base_model or args.tinker_checkpoint or args.local_checkpoint}")
    if args.base_model:
        print(f"  base_model={args.base_model}")
    if args.renderer_name:
        print(f"  renderer_name={args.renderer_name}")
    print(f"  task_args={task_args}")
    print(f"  model_args={model_args}")
    print(f"  metadata={metadata}")
    print(f"  generation_config={generation_config}")
    print(f"  log_dir={args.log_dir}, limit={args.limit}, epochs={args.epochs}")
    # Upstream task construction can materialize missing datasets, so it remains
    # behind the user's confirmation.
    print(
        "  preflight_samples=deferred (upstream task construction can materialize datasets; "
        "use --limit to bound source samples per task)"
    )

    if args.dry_run:
        print("Dry run complete; no task or model was constructed.")
        return

    if not args.yes and input("\nProceed with eval? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    logs = run_task_evals(
        args.task_factory,
        model=args.model,
        tinker_base_model_name=args.tinker_base_model,
        tinker_checkpoint=args.tinker_checkpoint,
        local_checkpoint=args.local_checkpoint,
        base_model=args.base_model,
        renderer_name=args.renderer_name,
        task_args=task_args,
        model_args=model_args,
        generation_config=generation_config,
        metadata=metadata,
        include_reasoning=args.include_reasoning,
        log_dir=args.log_dir,
        limit=args.limit,
        epochs=args.epochs,
    )
    failed = [log for log in logs if getattr(log, "status", None) not in (None, "success")]
    if failed:
        raise SystemExit(f"{len(failed)}/{len(logs)} eval logs did not complete successfully")


if __name__ == "__main__":
    main()
