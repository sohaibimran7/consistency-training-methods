"""Run an upstream Inspect task factory against one model/checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from ctm.cli_safety import reject_inline_secrets
from ctm.evals.local_model import (
    PERSISTENT_VLLM_CHILD_METADATA_ENV,
    PersistentNativeVLLMServer,
    native_vllm_request_identity,
)
from ctm.evals.runner import (
    build_tasks,
    effective_provider_generation_config,
    normalize_task_indices,
    parse_json_object,
    resolve_eval_model,
    run_task_evals,
    validate_tinker_generation_config,
)


def _command_summary(argv: list[str]) -> str:
    return "python scripts/run_evals.py " + " ".join(shlex.quote(arg) for arg in argv)


def _isolated_child_argv(argv: list[str], task_index: int) -> list[str]:
    """Build one non-recursive child invocation from the user's original argv."""

    child_argv: list[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        if token in {"--isolate-tasks", "--persistent-vllm-server"}:
            continue
        if token == "--task-index":
            skip_value = True
            continue
        if token.startswith("--task-index="):
            continue
        child_argv.append(token)
    if "--yes" not in child_argv and "-y" not in child_argv:
        child_argv.append("--yes")
    return [*child_argv, "--task-index", str(task_index)]


def _persistent_resume_argv(argv: list[str], task_index: int) -> list[str]:
    """Build a one-task parent invocation that retains persistent server ownership."""

    resume_argv: list[str] = []
    skip_value = False
    for token in argv:
        if skip_value:
            skip_value = False
            continue
        if token == "--task-index":
            skip_value = True
            continue
        if token.startswith("--task-index="):
            continue
        resume_argv.append(token)
    if "--yes" not in resume_argv and "-y" not in resume_argv:
        resume_argv.append("--yes")
    return [*resume_argv, "--task-index", str(task_index)]


def _persistent_child_metadata() -> dict[str, object] | None:
    """Read parent-server provenance injected into an isolated child."""

    raw = os.environ.get(PERSISTENT_VLLM_CHILD_METADATA_ENV)
    if raw is None:
        return None
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {PERSISTENT_VLLM_CHILD_METADATA_ENV}") from exc
    if not isinstance(metadata, dict) or metadata.get("mode") != "parent_owned_external":
        raise ValueError(f"invalid {PERSISTENT_VLLM_CHILD_METADATA_ENV} payload")
    return metadata


def _validate_persistent_vllm_mode(args: argparse.Namespace, model_args: dict[str, object]) -> None:
    """Fail closed unless the parent can exclusively own a native vLLM server."""

    if not args.persistent_vllm_server:
        return
    if not args.isolate_tasks:
        raise ValueError("--persistent-vllm-server requires --isolate-tasks")
    if args.model is not None:
        if not args.model.startswith("vllm/"):
            raise ValueError("--persistent-vllm-server requires a native vllm/... Inspect model")
    elif args.local_checkpoint is not None:
        if model_args.get("provider", "hf") != "vllm":
            raise ValueError("--persistent-vllm-server with --local-checkpoint requires --model-args with provider='vllm'")
    else:
        raise ValueError("--persistent-vllm-server supports only native vLLM models or local checkpoints")

    lifecycle_options = sorted(set(model_args) & {"api_key", "base_url", "lazy_init", "port"})
    if lifecycle_options:
        raise ValueError(f"--persistent-vllm-server owns these vLLM lifecycle option(s); remove them from --model-args: {lifecycle_options}")
    if os.environ.get("VLLM_BASE_URL"):
        raise ValueError("--persistent-vllm-server requires VLLM_BASE_URL to be unset in the parent")


def _validate_persistent_child_environment(
    args: argparse.Namespace,
    model_args: dict[str, object],
    metadata: dict[str, object] | None,
) -> None:
    """Verify that an isolated child targets its parent's exact external model identity."""

    if metadata is None:
        return
    if args.isolate_tasks or args.persistent_vllm_server:
        raise ValueError("parent-owned vLLM child metadata is valid only in a non-recursive task child")
    if not os.environ.get("VLLM_BASE_URL") or not os.environ.get("VLLM_API_KEY"):
        raise ValueError("parent-owned vLLM child is missing VLLM_BASE_URL or VLLM_API_KEY")
    if metadata.get("base_url") != os.environ["VLLM_BASE_URL"]:
        raise ValueError("parent-owned vLLM child endpoint does not match its metadata")
    base_model, adapter = native_vllm_request_identity(
        model=args.model,
        local_checkpoint=args.local_checkpoint,
        model_args=model_args,
    )
    if metadata.get("base_model") != base_model or metadata.get("adapter") != adapter:
        raise ValueError("parent-owned vLLM child model or adapter identity does not match its metadata")
    if metadata.get("served_model") != (adapter or base_model):
        raise ValueError("parent-owned vLLM child served-model identity does not match its metadata")


def _eval_log_names(log_dir: str) -> set[str]:
    """Snapshot complete Inspect log names without reading sample payloads."""

    from inspect_ai.log import list_eval_logs

    return {info.name for info in list_eval_logs(log_dir, formats=["eval"], recursive=False)}


def _successful_isolated_log(
    log_dir: str,
    *,
    previous_logs: set[str],
    task_index: int,
    task_count: int,
) -> str | None:
    """Return the new success artifact proving a child finished before a late crash."""

    from inspect_ai.log import list_eval_logs, read_eval_log

    matches: list[str] = []
    for info in list_eval_logs(log_dir, formats=["eval"], recursive=False):
        if info.name in previous_logs:
            continue
        try:
            log = read_eval_log(info, header_only=True)
        except Exception:  # A partial/corrupt artifact is never proof of success.
            continue
        metadata = log.eval.metadata or {}
        if log.status == "success" and metadata.get("task_indices") == [task_index] and metadata.get("task_count") == task_count:
            matches.append(info.name)
    return matches[0] if len(matches) == 1 else None


def _run_isolated_tasks(
    argv: list[str],
    *,
    log_dir: str,
    task_count: int,
    task_indices: list[int] | None,
    persistent_server: PersistentNativeVLLMServer | None = None,
) -> None:
    """Run selected tasks serially in fresh interpreters and retain each normal Inspect log."""

    selected = normalize_task_indices(task_indices, task_count=task_count)
    for task_index in selected:
        if persistent_server is not None:
            persistent_server.assert_healthy()
        previous_logs = _eval_log_names(log_dir)
        child_argv = _isolated_child_argv(argv, task_index)
        command = [sys.executable, str(Path(__file__).resolve()), *child_argv]
        print(f"\nIsolated task {task_index}/{task_count}:")
        print(f"  {shlex.join(command)}", flush=True)
        run_options: dict[str, object] = {"cwd": PROJECT_ROOT, "check": False}
        if persistent_server is not None:
            run_options["env"] = persistent_server.child_environment()
        completed = subprocess.run(command, **run_options)
        if completed.returncode:
            health_error: RuntimeError | None = None
            if persistent_server is not None:
                try:
                    persistent_server.assert_healthy()
                except RuntimeError as exc:
                    health_error = exc
            successful_log = _successful_isolated_log(
                log_dir,
                previous_logs=previous_logs,
                task_index=task_index,
                task_count=task_count,
            )
            if successful_log is not None and health_error is None:
                print(
                    f"WARNING: isolated task {task_index}/{task_count} exited with status {completed.returncode} after writing a verified successful log; continuing: {successful_log}",
                    flush=True,
                )
                continue
            resume_argv = _persistent_resume_argv(argv, task_index) if persistent_server is not None else _isolated_child_argv(argv, task_index)
            server_diagnostic = f" The parent-owned server is unhealthy: {health_error}." if health_error else ""
            raise SystemExit(f"isolated task {task_index}/{task_count} exited with status {completed.returncode}; completed tasks kept their .eval logs.{server_diagnostic} Resume this task with:\n  {_command_summary(resume_argv)}")


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
    parser.add_argument("--renderer-name", help="Optional expected Tinker renderer; verified against checkpoint metadata")
    parser.add_argument(
        "--task-args",
        help="Inline JSON object or JSON file passed to the task factory",
    )
    parser.add_argument("--model-args", help="Inline JSON object or JSON file passed to Inspect get_model")
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
        "--max-tasks",
        type=int,
        help="Maximum number of task definitions Inspect may execute concurrently",
    )
    parser.add_argument(
        "--task-index",
        dest="task_indices",
        action="append",
        type=int,
        help="Run one 1-based task-factory position; repeat to select several tasks",
    )
    parser.add_argument(
        "--isolate-tasks",
        action="store_true",
        help="Run each selected task in a fresh child process to isolate model-server and asyncio lifecycle state",
    )
    parser.add_argument(
        "--persistent-vllm-server",
        action="store_true",
        help=("With --isolate-tasks, keep one parent-owned native vLLM server for this condition while each task still runs in a fresh child interpreter"),
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
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("--max-tasks must be >= 1")
    if args.task_indices is not None and any(index < 1 for index in args.task_indices):
        parser.error("--task-index must be >= 1")

    try:
        task_args = parse_json_object(args.task_args, label="task_args")
        model_args = parse_json_object(args.model_args, label="model_args")
        generation_config = parse_json_object(args.generation_config, label="generation_config")
        for label, value in (
            ("task_args", task_args),
            ("model_args", model_args),
            ("generation_config", generation_config),
        ):
            reject_inline_secrets(value, path=label)
        generation_config = effective_provider_generation_config(
            generation_config,
            model=args.model,
            local_checkpoint=args.local_checkpoint,
            model_args=model_args,
        )
        if (args.tinker_base_model or args.tinker_checkpoint) and model_args:
            raise ValueError("--model-args applies only to ordinary Inspect providers and local checkpoints")
        if args.tinker_base_model or args.tinker_checkpoint:
            validate_tinker_generation_config(generation_config)
        _validate_persistent_vllm_mode(args, model_args)
        persistent_child_metadata = _persistent_child_metadata()
        _validate_persistent_child_environment(args, model_args, persistent_child_metadata)
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
    print(f"  generation_config={generation_config}")
    print(f"  log_dir={args.log_dir}, limit={args.limit}, epochs={args.epochs}, max_tasks={args.max_tasks}")
    print(f"  task_indices={args.task_indices or 'all'}, isolate_tasks={args.isolate_tasks}, persistent_vllm_server={args.persistent_vllm_server}")
    # Upstream task construction can materialize missing datasets, so it remains
    # behind the user's confirmation.
    print("  preflight_samples=deferred (upstream task construction can materialize datasets; use --limit to bound source samples per task)")

    if not args.yes and input("\nProceed with eval? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    if args.isolate_tasks:
        tasks = build_tasks(args.task_factory, task_args=task_args)
        try:
            if args.persistent_vllm_server:
                parent_model = resolve_eval_model(
                    model=args.model,
                    tinker_base_model_name=args.tinker_base_model,
                    tinker_checkpoint=args.tinker_checkpoint,
                    local_checkpoint=args.local_checkpoint,
                    base_model=args.base_model,
                    renderer_name=args.renderer_name,
                    model_args=model_args,
                    generation_config=generation_config,
                    include_reasoning=args.include_reasoning,
                )
                server = PersistentNativeVLLMServer.start(
                    parent_model,
                    log_dir=args.log_dir,
                    source_metadata={
                        "model": args.model,
                        "local_checkpoint": args.local_checkpoint,
                        "base_model": args.base_model,
                        "model_args": model_args,
                        "generation_config": generation_config,
                    },
                )
                with server:
                    _run_isolated_tasks(
                        effective_argv,
                        log_dir=args.log_dir,
                        task_count=len(tasks),
                        task_indices=args.task_indices,
                        persistent_server=server,
                    )
            else:
                _run_isolated_tasks(
                    effective_argv,
                    log_dir=args.log_dir,
                    task_count=len(tasks),
                    task_indices=args.task_indices,
                )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
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
        include_reasoning=args.include_reasoning,
        log_dir=args.log_dir,
        limit=args.limit,
        epochs=args.epochs,
        max_tasks=args.max_tasks,
        task_indices=args.task_indices,
        metadata=({"persistent_vllm_server": persistent_child_metadata} if persistent_child_metadata is not None else None),
    )
    failed = [log for log in logs if getattr(log, "status", None) not in (None, "success")]
    if failed:
        raise SystemExit(f"{len(failed)}/{len(logs)} eval logs did not complete successfully")


if __name__ == "__main__":
    main()
