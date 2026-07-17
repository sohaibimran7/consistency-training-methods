"""Run independently configured experiment stages from YAML.

The runner deliberately knows nothing about settings or benchmarks. Each stage
is an argv list plus an argument map, so training and evaluation can use
different packages, datasets, splits, and metrics.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ctm.cli_safety import reject_inline_secrets

STAGE_ORDER = ("data_generation", "data_preparation", "training", "evaluation", "analysis")
STAGE_ALIASES = {
    "data_gen": "data_generation",
    "datagen": "data_generation",
    "data_prep": "data_preparation",
    "dataprep": "data_preparation",
    "train": "training",
    "eval": "evaluation",
    "analyze": "analysis",
    "viz": "analysis",
}
_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_.-]*)\}")
CHECKPOINT_MARKER = "CTM_FINAL_CHECKPOINT="
_RESERVED_CONTEXT = {"python", "project_root", "experiment", "checkpoint", "training_data"}
OUTPUT_SCHEMA_VERSION = 1


class ExperimentConfigError(ValueError):
    """A YAML experiment config cannot be translated to commands."""


def load_experiment(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text())
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentConfigError("experiment config must be a YAML object")
    if not isinstance(value.get("name"), str) or not value["name"].strip():
        raise ExperimentConfigError("experiment config needs a non-empty name")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value["name"]):
        raise ExperimentConfigError(
            "experiment name must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens"
        )
    if not any(stage in value for stage in STAGE_ORDER):
        raise ExperimentConfigError(f"experiment config needs at least one stage: {list(STAGE_ORDER)}")
    variables = value.get("variables", {})
    if not isinstance(variables, Mapping) or any(not isinstance(key, str) for key in variables):
        raise ExperimentConfigError("experiment variables must be an object with string keys")
    conflicts = sorted(set(variables) & _RESERVED_CONTEXT)
    if conflicts:
        raise ExperimentConfigError(f"experiment variables use reserved names: {conflicts}")
    return value


def _canonical_stage(value: str) -> str:
    stage = STAGE_ALIASES.get(value, value)
    if stage not in STAGE_ORDER:
        raise ExperimentConfigError(f"unknown stage {value!r}; expected one of {list(STAGE_ORDER)}")
    return stage


def select_stages(
    config: Mapping[str, Any],
    *,
    stages: Sequence[str] | None = None,
    start_from: str | None = None,
) -> list[str]:
    present = [stage for stage in STAGE_ORDER if stage in config]
    if stages is not None and start_from is not None:
        raise ExperimentConfigError("pass --stages or --start-from, not both")
    if stages is not None:
        requested = [_canonical_stage(stage) for stage in stages]
        missing = [stage for stage in requested if stage not in present]
        if missing:
            raise ExperimentConfigError(f"requested stage(s) absent from config: {missing}")
        return [stage for stage in STAGE_ORDER if stage in requested]
    if start_from is not None:
        first = _canonical_stage(start_from)
        return [stage for stage in present if STAGE_ORDER.index(stage) >= STAGE_ORDER.index(first)]
    return present


def _entries(config: Mapping[str, Any], stage: str) -> list[Mapping[str, Any]]:
    value = config[stage]
    entries = value if isinstance(value, list) else [value]
    if not entries or any(not isinstance(entry, Mapping) for entry in entries):
        raise ExperimentConfigError(f"{stage} must be a command object or non-empty list of command objects")
    return list(entries)


def _render_string(value: str, context: Mapping[str, Any], *, strict: bool) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        replacement = context.get(key)
        if replacement is None:
            if strict:
                raise ExperimentConfigError(f"unresolved ${{{key}}} placeholder")
            return match.group(0)
        return str(replacement)

    return _PLACEHOLDER.sub(replace, value)


def _render(value: Any, context: Mapping[str, Any], *, strict: bool) -> Any:
    if isinstance(value, str):
        return _render_string(value, context, strict=strict)
    if isinstance(value, Mapping):
        return {str(key): _render(item, context, strict=strict) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, context, strict=strict) for item in value]
    return value


def _flag(key: str) -> str:
    return key if key.startswith("-") else "--" + key.replace("_", "-")


def _argument_tokens(args: Any) -> list[str]:
    if args is None:
        return []
    if isinstance(args, list):
        if any(not isinstance(value, (str, int, float)) for value in args):
            raise ExperimentConfigError("list-form args must contain only scalar argv tokens")
        return [str(value) for value in args]
    if not isinstance(args, Mapping):
        raise ExperimentConfigError("command args must be an object or argv-token list")

    tokens: list[str] = []
    for key, value in args.items():
        if not isinstance(key, str):
            raise ExperimentConfigError(
                f"argument keys must be strings; quote YAML 1.1 words such as 'yes' (got {key!r})"
            )
        flag = _flag(str(key))
        if value is None or value is False:
            continue
        tokens.append(flag)
        if value is True:
            continue
        if isinstance(value, Mapping):
            tokens.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
        elif isinstance(value, list):
            tokens.extend(
                (
                    json.dumps(item, sort_keys=True, separators=(",", ":"))
                    if isinstance(item, (Mapping, list))
                    else str(item)
                )
                for item in value
            )
        else:
            tokens.append(str(value))
    return tokens


def command_argv(spec: Mapping[str, Any], context: Mapping[str, Any], *, strict: bool = True) -> list[str]:
    rendered = _render(spec, context, strict=strict)
    command = rendered.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(token, str) for token in command):
        raise ExperimentConfigError("each command needs a non-empty string-list command")
    unknown = sorted(set(rendered) - {"name", "command", "args"})
    if unknown:
        raise ExperimentConfigError(f"unknown command field(s): {unknown}")
    return [*command, *_argument_tokens(rendered.get("args"))]


def _uses_placeholder(value: Any, name: str) -> bool:
    if isinstance(value, str):
        return any(match.group(1) == name for match in _PLACEHOLDER.finditer(value))
    if isinstance(value, Mapping):
        return any(_uses_placeholder(item, name) for item in value.values())
    if isinstance(value, list):
        return any(_uses_placeholder(item, name) for item in value)
    return False


def validate_checkpoint_ownership(config: Mapping[str, Any], selected_stages: Sequence[str]) -> None:
    """Reject the ambiguous last-training-checkpoint convention.

    A single training command may publish ``${checkpoint}`` for later stages.
    With multiple selected training commands there is no declared owner, so any
    selected command using that placeholder is rejected until named outputs exist.
    """

    n_training = len(_entries(config, "training")) if "training" in selected_stages else 0
    if n_training <= 1:
        return
    consumers = []
    for stage in selected_stages:
        for index, spec in enumerate(_entries(config, stage), start=1):
            if _uses_placeholder(spec, "checkpoint"):
                consumers.append(str(spec.get("name") or f"{stage}-{index}"))
    if consumers:
        raise ExperimentConfigError(
            f"ambiguous ${{checkpoint}} ownership: the selected plan has {n_training} training commands "
            f"and the placeholder is used by {consumers}. Split the runs or pass explicit checkpoint "
            "values; named stage outputs are not implemented."
        )


def selected_stages_use_placeholder(config: Mapping[str, Any], selected_stages: Sequence[str], name: str) -> bool:
    return any(_uses_placeholder(spec, name) for stage in selected_stages for spec in _entries(config, stage))


def planned_commands(
    config: Mapping[str, Any],
    selected_stages: Sequence[str],
    context: Mapping[str, Any],
    *,
    strict: bool,
) -> list[tuple[str, str, list[str]]]:
    validate_checkpoint_ownership(config, selected_stages)
    planned = []
    for stage in selected_stages:
        names: set[str] = set()
        for index, spec in enumerate(_entries(config, stage), start=1):
            name = str(spec.get("name") or f"{stage}-{index}")
            if name in names:
                raise ExperimentConfigError(f"duplicate {stage} command name {name!r}")
            if stage == "training" and not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", name):
                raise ExperimentConfigError(
                    f"training command name {name!r} must contain only letters, digits, underscores, and hyphens"
                )
            names.add(name)
            planned.append((stage, name, command_argv(spec, context, strict=strict)))
    return planned


def initial_context(
    config: Mapping[str, Any],
    *,
    checkpoint: str | None = None,
    training_data: str | None = None,
) -> dict[str, Any]:
    """Build the placeholder context from YAML variables and CLI-wide inputs."""

    variables = config.get("variables", {})
    if not isinstance(variables, Mapping):
        raise ExperimentConfigError("experiment variables must be an object")
    conflicts = sorted(set(variables) & _RESERVED_CONTEXT)
    if conflicts:
        raise ExperimentConfigError(f"experiment variables use reserved names: {conflicts}")
    return {
        **dict(variables),
        "python": sys.executable,
        "project_root": PROJECT_ROOT,
        "experiment": config["name"],
        "checkpoint": checkpoint or config.get("checkpoint"),
        "training_data": training_data,
    }


def output_state_path(config: Mapping[str, Any]) -> Path:
    """Return the structured output path for one experiment."""

    return PROJECT_ROOT / "logs" / "experiments" / str(config["name"]) / "outputs.json"


def load_output_context(config: Mapping[str, Any]) -> dict[str, str]:
    """Load previously published named training checkpoints, if present."""

    path = output_state_path(config)
    if not path.exists():
        return {}
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError(f"invalid experiment output state {path}: {exc}") from exc
    if state.get("schema_version") != OUTPUT_SCHEMA_VERSION or state.get("experiment") != config["name"]:
        raise ExperimentConfigError(f"experiment output state does not match {config['name']!r}: {path}")
    checkpoints = state.get("training_checkpoints", {})
    if not isinstance(checkpoints, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str) or not value for name, value in checkpoints.items()
    ):
        raise ExperimentConfigError(f"invalid training_checkpoints in {path}")
    context = {f"training.{name}.checkpoint": value for name, value in checkpoints.items()}
    if len(checkpoints) == 1:
        context["checkpoint"] = next(iter(checkpoints.values()))
    return context


def save_training_checkpoint(config: Mapping[str, Any], name: str, checkpoint: str) -> None:
    """Publish one named checkpoint atomically for later experiment stages."""

    path = output_state_path(config)
    existing = load_output_context(config)
    checkpoints = {
        key.removeprefix("training.").removesuffix(".checkpoint"): value
        for key, value in existing.items()
        if key.startswith("training.") and key.endswith(".checkpoint")
    }
    checkpoints[name] = checkpoint
    state = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "experiment": config["name"],
        "training_checkpoints": checkpoints,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_command(argv: Sequence[str]) -> str | None:
    """Stream one subprocess and return a checkpoint announced by training."""

    process = subprocess.Popen(
        list(argv),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    checkpoint = None
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        stripped = line.strip()
        if stripped.startswith(CHECKPOINT_MARKER):
            announced = stripped.removeprefix(CHECKPOINT_MARKER).strip()
            if not announced:
                raise ExperimentConfigError("training emitted an empty final-checkpoint marker")
            checkpoint = announced
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(argv))
    return checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent experiment stages from YAML")
    parser.add_argument("config", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stages", help="Comma-separated stages to run")
    selection.add_argument("--start-from", help="Skip configured stages before this stage")
    parser.add_argument("--checkpoint", help="Value for ${checkpoint}; overrides the YAML checkpoint")
    parser.add_argument("--training-data", help="Required value for ${training_data} when selected commands use it")
    parser.add_argument("--dry-run", action="store_true", help="Print the config and commands without executing")
    parser.add_argument("-y", "--yes", action="store_true", help="Execute after printing the plan")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_experiment(args.config)
        reject_inline_secrets(config, path="experiment")
        stages = select_stages(
            config,
            stages=[part.strip() for part in args.stages.split(",") if part.strip()] if args.stages else None,
            start_from=args.start_from,
        )
        context = initial_context(config, checkpoint=args.checkpoint, training_data=args.training_data)
        if "training" not in stages:
            context.update(load_output_context(config))
        explicit_checkpoint = args.checkpoint or config.get("checkpoint")
        if explicit_checkpoint:
            context["checkpoint"] = explicit_checkpoint
        if selected_stages_use_placeholder(config, stages, "training_data") and not args.training_data:
            raise ExperimentConfigError("selected commands use ${training_data}; pass --training-data PATH")
        preview = planned_commands(config, stages, context, strict=False)
    except (ExperimentConfigError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print("\nExperiment config:")
    print(yaml.safe_dump(config, sort_keys=False).rstrip())
    print("\nCommands:")
    for stage, name, command in preview:
        print(f"  [{stage}:{name}] {shlex.join(command)}")
    if args.dry_run:
        print("\nDry run complete.")
        return
    if not args.yes and input("\nProceed with these stages? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    try:
        for stage in stages:
            for index, spec in enumerate(_entries(config, stage), start=1):
                name = str(spec.get("name") or f"{stage}-{index}")
                command = command_argv(spec, context, strict=True)
                print(f"\n[{stage}:{name}] {shlex.join(command)}")
                checkpoint = run_command(command)
                if checkpoint:
                    context["checkpoint"] = checkpoint
                    if stage == "training":
                        context[f"training.{name}.checkpoint"] = checkpoint
                        save_training_checkpoint(config, name, checkpoint)
    except (ExperimentConfigError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
