"""Run independently configured experiment stages from YAML.

The runner deliberately knows nothing about settings or benchmarks. Each stage
is an argv list plus an argument map, so training and evaluation can use
different packages, datasets, splits, and metrics.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from ctm.cli_safety import reject_inline_secrets
from ctm.importing import load_callable

STAGE_ORDER = ("data_generation", "data_preparation", "training", "evaluation", "analysis", "rendering")
STAGE_ALIASES = {
    "data_gen": "data_generation",
    "datagen": "data_generation",
    "data_prep": "data_preparation",
    "dataprep": "data_preparation",
    "train": "training",
    "eval": "evaluation",
    "analyze": "analysis",
    "render": "rendering",
    "viz": "rendering",
}
_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_.-]*)\}")
CHECKPOINT_MARKER = "CTM_FINAL_CHECKPOINT="
_RESERVED_CONTEXT = {"python", "project_root", "experiment", "checkpoint", "training_data"}
OUTPUT_SCHEMA_VERSION = 1
GPU_STAGES = frozenset({"data_preparation", "training", "evaluation"})


class ExperimentConfigError(ValueError):
    """A YAML experiment config cannot be translated to commands."""


def _validate_experiment_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError("experiment config needs a non-empty name")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ExperimentConfigError("experiment name must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens")
    return value


def _validate_target_name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentConfigError("execution target must be a non-empty string")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ExperimentConfigError("execution target must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens")
    return value


def load_experiment_source(path: str | Path) -> dict[str, Any]:
    """Read an authored experiment file without expanding a factory."""

    source = Path(path)
    try:
        value = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid YAML in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentConfigError("experiment config must be a YAML object")
    _validate_experiment_name(value.get("name"))
    return value


def _validate_topology_profile(value: str | None) -> str | None:
    """Validate an explicit, launcher-selected logical GPU topology profile."""

    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
        raise ExperimentConfigError(
            "topology profile must start with a letter or digit and contain only letters, digits, dots, underscores, and hyphens"
        )
    return value


def compile_experiment(
    source: Mapping[str, Any],
    *,
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Expand an optional ``module:callable`` factory into an execution plan.

    ``topology_profile`` is intentionally an explicit compiler input instead
    of an ambient environment setting.  Protected launchers pass it again to
    the target attestation, which binds the resulting logical GPU bundle and
    replays the same compilation before starting the child.
    """

    name = _validate_experiment_name(source.get("name"))
    topology_profile = _validate_topology_profile(topology_profile)
    factory_spec = source.get("experiment_factory")
    if factory_spec is None:
        if topology_profile is not None:
            raise ExperimentConfigError("--topology-profile requires a factory experiment that defines selectable profiles")
        value = dict(source)
    else:
        unknown = sorted(set(source) - {"name", "experiment_factory", "spec"})
        if unknown:
            raise ExperimentConfigError(f"factory experiment has unknown top-level field(s): {unknown}")
        spec = source.get("spec")
        if not isinstance(spec, Mapping):
            raise ExperimentConfigError("factory experiment needs a spec object")
        try:
            factory = load_callable(factory_spec, label="experiment_factory")
            expanded = factory(name=name, spec=dict(spec), topology_profile=topology_profile)
        except (TypeError, ValueError) as exc:
            raise ExperimentConfigError(f"experiment factory failed: {exc}") from exc
        if not isinstance(expanded, Mapping):
            raise ExperimentConfigError("experiment factory must return an object")
        value = dict(expanded)
        if value.get("name") != name:
            raise ExperimentConfigError("experiment factory must preserve the authored experiment name")

    if not any(stage in value for stage in STAGE_ORDER):
        raise ExperimentConfigError(f"experiment config needs at least one stage: {list(STAGE_ORDER)}")
    variables = value.get("variables", {})
    if not isinstance(variables, Mapping) or any(not isinstance(key, str) for key in variables):
        raise ExperimentConfigError("experiment variables must be an object with string keys")
    conflicts = sorted(set(variables) & _RESERVED_CONTEXT)
    if conflicts:
        raise ExperimentConfigError(f"experiment variables use reserved names: {conflicts}")
    validate_training_backend_consistency(value)
    return value


def _declared_backend(entry: Mapping[str, Any]) -> str | None:
    """Return an explicitly declared training backend from either args form."""

    args = entry.get("args")
    if isinstance(args, Mapping):
        backend = args.get("backend")
        return str(backend) if backend is not None else None
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        tokens = [str(token) for token in args]
        if "--backend" in tokens:
            index = tokens.index("--backend")
            if index + 1 >= len(tokens):
                raise ExperimentConfigError("training command --backend needs a value")
            return tokens[index + 1]
    return None


def validate_training_backend_consistency(config: Mapping[str, Any]) -> None:
    """Reject plans that would compare training runs from different backends.

    Tinker and local execution do not promise byte-identical prompt rendering or
    optimizer semantics. Keeping one backend per experiment makes the backend an
    execution detail instead of an uncontrolled scientific variable.
    """

    if "training" not in config:
        return
    raw_entries = config["training"]
    entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
    if not entries or any(not isinstance(entry, Mapping) for entry in entries):
        return  # The normal stage validator will report the structural error.

    declared: dict[str, list[str]] = {}
    undeclared: list[str] = []
    for index, entry in enumerate(entries, start=1):
        name = str(entry.get("name") or f"training-{index}")
        backend = _declared_backend(entry)
        if backend is None:
            undeclared.append(name)
        else:
            declared.setdefault(backend, []).append(name)

    if declared and undeclared:
        raise ExperimentConfigError(f"training commands must all declare --backend when any command does; missing for {undeclared}")
    if len(declared) > 1:
        details = ", ".join(f"{backend}={names}" for backend, names in sorted(declared.items()))
        raise ExperimentConfigError(f"an experiment cannot mix training backends because its results would not be directly comparable; split this plan into one backend per experiment ({details})")


def load_experiment(
    path: str | Path,
    *,
    topology_profile: str | None = None,
) -> dict[str, Any]:
    """Read and compile either a direct plan or a concise experiment spec."""

    return compile_experiment(load_experiment_source(path), topology_profile=topology_profile)


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


def _entries(
    config: Mapping[str, Any],
    stage: str,
    *,
    target: str | None = None,
) -> list[Mapping[str, Any]]:
    value = config[stage]
    entries = value if isinstance(value, list) else [value]
    if not entries or any(not isinstance(entry, Mapping) for entry in entries):
        raise ExperimentConfigError(f"{stage} must be a command object or non-empty list of command objects")
    for entry in entries:
        entry_target = entry.get("target")
        if entry_target is not None:
            try:
                _validate_target_name(entry_target)
            except ExperimentConfigError as exc:
                raise ExperimentConfigError(f"{stage} command has invalid target: {exc}") from exc
    if target is None:
        return list(entries)
    _validate_target_name(target)
    return [entry for entry in entries if entry.get("target") == target]


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
            raise ExperimentConfigError(f"argument keys must be strings; quote YAML 1.1 words such as 'yes' (got {key!r})")
        flag = _flag(str(key))
        if value is None or value is False:
            continue
        tokens.append(flag)
        if value is True:
            continue
        if isinstance(value, Mapping):
            tokens.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
        elif isinstance(value, list):
            tokens.extend((json.dumps(item, sort_keys=True, separators=(",", ":")) if isinstance(item, (Mapping, list)) else str(item)) for item in value)
        else:
            tokens.append(str(value))
    return tokens


def command_argv(spec: Mapping[str, Any], context: Mapping[str, Any], *, strict: bool = True) -> list[str]:
    rendered = _render(spec, context, strict=strict)
    command = rendered.get("command")
    if not isinstance(command, list) or not command or any(not isinstance(token, str) for token in command):
        raise ExperimentConfigError("each command needs a non-empty string-list command")
    unknown = sorted(set(rendered) - {"name", "target", "resource", "gpu_count", "command", "args"})
    if unknown:
        raise ExperimentConfigError(f"unknown command field(s): {unknown}")
    return [*command, *_argument_tokens(rendered.get("args"))]


def command_resource(spec: Mapping[str, Any], stage: str) -> str:
    """Return the execution resource requested by a command."""

    resource = spec.get("resource", "gpu" if stage in GPU_STAGES else "cpu")
    if resource not in {"cpu", "gpu"}:
        raise ExperimentConfigError(f"{stage} command resource must be 'cpu' or 'gpu'; got {resource!r}")
    return str(resource)


def command_gpu_count(spec: Mapping[str, Any], stage: str) -> int:
    """Return the number of exclusively assigned GPUs requested by a command."""

    resource = command_resource(spec, stage)
    value = spec.get("gpu_count")
    if value is None:
        return 1 if resource == "gpu" else 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentConfigError(f"{stage} command gpu_count must be a positive integer")
    if resource != "gpu":
        raise ExperimentConfigError(f"{stage} command gpu_count applies only to resource: gpu")
    if value < 1:
        raise ExperimentConfigError(f"{stage} command gpu_count must be a positive integer")
    return value


def _uses_placeholder(value: Any, name: str) -> bool:
    if isinstance(value, str):
        return any(match.group(1) == name for match in _PLACEHOLDER.finditer(value))
    if isinstance(value, Mapping):
        return any(_uses_placeholder(item, name) for item in value.values())
    if isinstance(value, list):
        return any(_uses_placeholder(item, name) for item in value)
    return False


def validate_checkpoint_ownership(
    config: Mapping[str, Any],
    selected_stages: Sequence[str],
    *,
    target: str | None = None,
) -> None:
    """Reject the ambiguous last-training-checkpoint convention.

    A single training command may publish ``${checkpoint}`` for later stages.
    With multiple selected training commands there is no declared owner, so any
    selected command using that placeholder is rejected until named outputs exist.
    """

    n_training = len(_entries(config, "training", target=target)) if "training" in selected_stages else 0
    if n_training <= 1:
        return
    consumers = []
    for stage in selected_stages:
        for index, spec in enumerate(_entries(config, stage, target=target), start=1):
            if _uses_placeholder(spec, "checkpoint"):
                consumers.append(str(spec.get("name") or f"{stage}-{index}"))
    if consumers:
        raise ExperimentConfigError(f"ambiguous ${{checkpoint}} ownership: the selected plan has {n_training} training commands and the placeholder is used by {consumers}. Split the runs or pass explicit checkpoint values; named stage outputs are not implemented.")


def selected_stages_use_placeholder(
    config: Mapping[str, Any],
    selected_stages: Sequence[str],
    name: str,
    *,
    target: str | None = None,
) -> bool:
    return any(_uses_placeholder(spec, name) for stage in selected_stages for spec in _entries(config, stage, target=target))


def planned_commands(
    config: Mapping[str, Any],
    selected_stages: Sequence[str],
    context: Mapping[str, Any],
    *,
    strict: bool,
    target: str | None = None,
) -> list[tuple[str, str, list[str]]]:
    validate_checkpoint_ownership(config, selected_stages, target=target)
    planned = []
    for stage in selected_stages:
        names: set[str] = set()
        for index, spec in enumerate(_entries(config, stage, target=target), start=1):
            name = str(spec.get("name") or f"{stage}-{index}")
            command_gpu_count(spec, stage)  # validate resource allocation in previews as well as execution
            if name in names:
                raise ExperimentConfigError(f"duplicate {stage} command name {name!r}")
            if stage == "training" and not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_-]*", name):
                raise ExperimentConfigError(f"training command name {name!r} must contain only letters, digits, underscores, and hyphens")
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


def output_state_path(config: Mapping[str, Any], *, target: str | None = None) -> Path:
    """Return the canonical or target-scoped structured output path."""

    root = PROJECT_ROOT / "logs" / "experiments" / str(config["name"])
    if target is not None:
        root = root / "targets" / _validate_target_name(target)
    return root / "outputs.json"


def resolved_plan_path(config: Mapping[str, Any], *, target: str | None = None) -> Path:
    """Return the immutable expanded-plan path for one experiment/target."""

    state_path = output_state_path(config) if target is None else output_state_path(config, target=target)
    return state_path.with_name("resolved-plan.yaml")


def _resolved_plan(config: Mapping[str, Any], *, target: str | None) -> dict[str, Any]:
    if target is None:
        return dict(config)
    plan = {key: value for key, value in config.items() if key not in STAGE_ORDER}
    for stage in STAGE_ORDER:
        if stage not in config:
            continue
        entries = _entries(config, stage, target=target)
        if entries:
            plan[stage] = [dict(entry) for entry in entries]
    return plan


def resolved_plan_text(config: Mapping[str, Any], *, target: str | None = None) -> str:
    """Serialize the complete or target-scoped command plan deterministically."""

    return yaml.safe_dump(_resolved_plan(config, target=target), sort_keys=False).rstrip() + "\n"


def validate_resolved_plan(
    config: Mapping[str, Any],
    *,
    target: str | None = None,
) -> tuple[Path, str]:
    """Reject reuse of an experiment/target name for a different expanded plan."""

    path = resolved_plan_path(config) if target is None else resolved_plan_path(config, target=target)
    content = resolved_plan_text(config, target=target)
    digest = hashlib.sha256(content.encode()).hexdigest()
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ExperimentConfigError(f"resolved plan differs from {path}. Use a new experiment name, or move the existing experiment log directory to an archive before rerunning.")
    return path, digest


def save_resolved_plan(
    config: Mapping[str, Any],
    *,
    target: str | None = None,
) -> tuple[Path, str]:
    """Persist the expanded plan atomically after run approval."""

    path, digest = validate_resolved_plan(config, target=target)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".yaml.tmp")
        temporary.write_text(resolved_plan_text(config, target=target), encoding="utf-8")
        temporary.replace(path)
    return path, digest


def _load_output_state(config: Mapping[str, Any], *, target: str | None = None) -> tuple[Path, dict[str, Any]] | None:
    path = output_state_path(config) if target is None else output_state_path(config, target=target)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError(f"invalid experiment output state {path}: {exc}") from exc
    if state.get("schema_version") != OUTPUT_SCHEMA_VERSION or state.get("experiment") != config["name"]:
        raise ExperimentConfigError(f"experiment output state does not match {config['name']!r}: {path}")
    if state.get("execution_target") != target:
        raise ExperimentConfigError(f"experiment output state target does not match {target!r}: {path}")
    checkpoints = state.get("training_checkpoints", {})
    if not isinstance(checkpoints, Mapping) or any(not isinstance(name, str) or not isinstance(value, str) or not value for name, value in checkpoints.items()):
        raise ExperimentConfigError(f"invalid training_checkpoints in {path}")
    return path, state


def load_output_context(config: Mapping[str, Any], *, target: str | None = None) -> dict[str, str]:
    """Load previously published named training checkpoints, if present."""

    loaded = _load_output_state(config, target=target)
    if loaded is None:
        return {}
    _, state = loaded
    checkpoints = state["training_checkpoints"]
    context = {f"training.{name}.checkpoint": value for name, value in checkpoints.items()}
    if len(checkpoints) == 1:
        context["checkpoint"] = next(iter(checkpoints.values()))
    return context


def validated_completed_training(config: Mapping[str, Any], *, target: str | None = None) -> dict[str, str]:
    """Return completed commands whose published checkpoints still validate."""

    loaded = _load_output_state(config, target=target)
    if loaded is None:
        return {}
    _, state = loaded
    declared_names = {str(entry.get("name") or f"training-{index}") for index, entry in enumerate(_entries(config, "training", target=target), start=1)}
    completed = dict(state["training_checkpoints"])
    stale = sorted(set(completed) - declared_names)
    if stale:
        raise ExperimentConfigError(f"completed training state contains command(s) outside the selected plan: {stale}")
    for checkpoint in completed.values():
        _checkpoint_artifact_manifest(checkpoint)
    return completed


def save_training_checkpoint(
    config: Mapping[str, Any],
    name: str,
    checkpoint: str,
    *,
    target: str | None = None,
) -> None:
    """Publish one named checkpoint atomically within one execution scope."""

    path = output_state_path(config) if target is None else output_state_path(config, target=target)
    existing = load_output_context(config, target=target)
    checkpoints = {key.removeprefix("training.").removesuffix(".checkpoint"): value for key, value in existing.items() if key.startswith("training.") and key.endswith(".checkpoint")}
    checkpoints[name] = checkpoint
    state = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "experiment": config["name"],
        "training_checkpoints": checkpoints,
        **({"execution_target": target} if target is not None else {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _publication_manifest(config: Mapping[str, Any]) -> tuple[str, dict[str, list[str]]]:
    publication = config.get("training_output_publication")
    if not isinstance(publication, Mapping):
        raise ExperimentConfigError("experiment does not declare training_output_publication")
    owner = publication.get("owner")
    targets = publication.get("targets")
    if not isinstance(owner, str) or not owner:
        raise ExperimentConfigError("training_output_publication.owner must be a non-empty string")
    _validate_target_name(owner)
    if not isinstance(targets, list) or not targets or any(not isinstance(target, str) for target in targets) or len(targets) != len(set(targets)):
        raise ExperimentConfigError("training_output_publication.targets must be a non-empty unique string list")
    declared_targets = [_validate_target_name(target) for target in targets]

    names_by_target: dict[str, list[str]] = {target: [] for target in declared_targets}
    for index, entry in enumerate(_entries(config, "training"), start=1):
        name = str(entry.get("name") or f"training-{index}")
        target = entry.get("target")
        if target not in names_by_target:
            raise ExperimentConfigError(f"training command {name!r} has target {target!r}, outside training_output_publication.targets")
        names_by_target[target].append(name)
    empty = [target for target, names in names_by_target.items() if not names]
    if empty:
        raise ExperimentConfigError(f"training_output_publication has target(s) without training commands: {empty}")
    return owner, names_by_target


def _checkpoint_artifact_manifest(checkpoint: str) -> dict[str, Any]:
    """Validate and hash a local checkpoint after cross-node synchronization."""

    if not checkpoint.startswith("file://"):
        return {"checkpoint": checkpoint, "storage": "remote"}

    directory = Path(checkpoint.removeprefix("file://")).resolve()
    if not directory.is_dir():
        raise ExperimentConfigError(f"local checkpoint directory is missing after synchronization: {directory}")

    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentConfigError(f"local checkpoint has no manifest.json: {directory}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError(f"invalid local checkpoint manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, Mapping) or manifest.get("backend") != "local":
        raise ExperimentConfigError(f"local checkpoint manifest is not a LocalBackend checkpoint: {manifest_path}")

    if manifest.get("lora") is True:
        required = [directory / "adapter_config.json"]
        adapter_weights = [directory / name for name in ("adapter_model.safetensors", "adapter_model.bin")]
        if not any(path.is_file() for path in adapter_weights):
            raise ExperimentConfigError(f"local LoRA checkpoint has no adapter weights: {directory}")
    elif manifest.get("lora") is False:
        required = [directory / "weights.pt"]
    else:
        raise ExperimentConfigError(f"local checkpoint manifest has invalid lora flag: {manifest_path}")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ExperimentConfigError(f"local checkpoint is incomplete; missing files: {missing}")

    files: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ExperimentConfigError(f"local checkpoint contains a symlink and cannot be integrity-hashed: {path}")
        if not path.is_file():
            continue
        payload = path.read_bytes()
        files.append(
            {
                "path": str(path.relative_to(directory)),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not files:
        raise ExperimentConfigError(f"local checkpoint contains no files: {directory}")
    return {
        "checkpoint": checkpoint,
        "storage": "local",
        "backend": "local",
        "model": manifest.get("model"),
        "lora": manifest["lora"],
        "files": files,
    }


def publish_training_outputs(config: Mapping[str, Any]) -> tuple[Path, str]:
    """Merge completed target states once, from the declared publication owner."""

    owner, names_by_target = _publication_manifest(config)
    checkpoints: dict[str, str] = {}
    checkpoint_artifacts: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    for target, expected_names in names_by_target.items():
        loaded = _load_output_state(config, target=target)
        if loaded is None:
            raise ExperimentConfigError(f"training target {target!r} has no output state")
        path, state = loaded
        target_checkpoints = dict(state["training_checkpoints"])
        expected = set(expected_names)
        actual = set(target_checkpoints)
        if actual != expected:
            missing, unexpected = sorted(expected - actual), sorted(actual - expected)
            raise ExperimentConfigError(f"training target {target!r} checkpoint set is incomplete or stale: missing={missing}, unexpected={unexpected}")
        checkpoints.update(target_checkpoints)
        for name, checkpoint in target_checkpoints.items():
            checkpoint_artifacts[name] = _checkpoint_artifact_manifest(checkpoint)
        sources.append(
            {
                "target": target,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "training_commands": expected_names,
            }
        )

    state = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "experiment": config["name"],
        "training_checkpoints": checkpoints,
        "checkpoint_artifacts": checkpoint_artifacts,
        "publication": {
            "owner": owner,
            "resolved_plan_sha256": hashlib.sha256(resolved_plan_text(config).encode()).hexdigest(),
            "sources": sources,
        },
    }
    path = output_state_path(config)
    serialized = json.dumps(state, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(serialized.encode()).hexdigest()
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise ExperimentConfigError(f"canonical experiment output state already exists and differs: {path}. Move the existing experiment log directory to an archive before publishing a different state.")
        return path, digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return path, digest


def run_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    label: str | None = None,
) -> str | None:
    """Stream one subprocess and return a checkpoint announced by training."""

    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=dict(env) if env is not None else None,
    )
    checkpoint = None
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{label}] {line}" if label else line, end="", flush=True)
        stripped = line.strip()
        if stripped.startswith(CHECKPOINT_MARKER):
            announced = stripped.removeprefix(CHECKPOINT_MARKER).strip()
            if not announced:
                raise ExperimentConfigError("training emitted an empty final-checkpoint marker")
            checkpoint = announced
    return_code = process.wait()
    elapsed = time.monotonic() - started
    print(
        "CTM_COMMAND_TIMING="
        + json.dumps(
            {
                "label": label,
                "elapsed_seconds": round(elapsed, 6),
                "return_code": return_code,
                "status": "passed" if return_code == 0 else "failed",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(argv))
    return checkpoint


def _missing_executables(preview: Sequence[tuple[str, str, list[str]]]) -> dict[str, str]:
    """Map each command executable absent from PATH to the first command needing it."""

    missing: dict[str, str] = {}
    for stage, name, command in preview:
        executable = command[0]
        if "${" not in executable and executable not in missing and shutil.which(executable) is None:
            missing[executable] = f"{stage}:{name}"
    return missing


def _parse_gpus(value: str | None) -> list[str]:
    if value is None:
        return []
    gpus = [part.strip() for part in value.split(",") if part.strip()]
    if not gpus:
        raise ExperimentConfigError("--gpus needs at least one comma-separated GPU id")
    if len(gpus) != len(set(gpus)):
        raise ExperimentConfigError("--gpus cannot contain duplicate GPU ids")
    if any(not re.fullmatch(r"\d+", gpu) for gpu in gpus):
        raise ExperimentConfigError("--gpus accepts numeric GPU ids such as 0,1,2,3")
    return gpus


def _with_onpolicy_target_attestation(
    command: Sequence[str],
    *,
    stage: str,
    attestation: Path | None,
) -> list[str]:
    """Append the immutable target sidecar to the one protected training child."""

    output = list(command)
    if attestation is not None and stage == "training":
        output.extend(("--onpolicy-target-attestation", str(attestation)))
    return output


def _run_stage_parallel(
    config: Mapping[str, Any],
    stage: str,
    context: dict[str, Any],
    *,
    target: str | None,
    parallel: int,
    gpus: Sequence[str],
    skip_completed: frozenset[str] = frozenset(),
    onpolicy_target_attestation: Path | None = None,
) -> None:
    """Run independent commands concurrently with exclusive one-or-more-GPU bundles."""

    work = []
    for index, spec in enumerate(_entries(config, stage, target=target), start=1):
        name = str(spec.get("name") or f"{stage}-{index}")
        if stage == "training" and name in skip_completed:
            print(f"\n[{stage}:{name}] SKIPPED: validated completed checkpoint", flush=True)
            continue
        work.append(
            (
                name,
                _with_onpolicy_target_attestation(
                    command_argv(spec, context, strict=True),
                    stage=stage,
                    attestation=onpolicy_target_attestation,
                ),
                command_resource(spec, stage),
                command_gpu_count(spec, stage),
            )
        )
    if not work:
        return

    needs_gpu = any(resource == "gpu" for _, _, resource, _ in work)
    if needs_gpu and not gpus:
        raise ExperimentConfigError(f"parallel {stage} execution includes GPU commands; pass --gpus with the visible GPU ids")
    largest_bundle = max((gpu_count for _, _, _, gpu_count in work), default=0)
    if largest_bundle > len(gpus):
        raise ExperimentConfigError(f"parallel {stage} command requests {largest_bundle} GPU(s), but only {len(gpus)} were supplied")
    gpu_order = {gpu: index for index, gpu in enumerate(gpus)}
    available_gpus = list(gpus)
    gpu_condition = threading.Condition()

    def acquire_gpus(count: int) -> list[str]:
        if count == 0:
            return []
        with gpu_condition:
            while len(available_gpus) < count:
                gpu_condition.wait()
            selected = available_gpus[:count]
            del available_gpus[:count]
            return selected

    def release_gpus(selected: Sequence[str]) -> None:
        if not selected:
            return
        with gpu_condition:
            available_gpus.extend(selected)
            available_gpus.sort(key=gpu_order.__getitem__)
            gpu_condition.notify_all()

    def run_one(item: tuple[str, list[str], str, int]) -> tuple[str, str | None]:
        name, command, resource, gpu_count = item
        assigned_gpus: list[str] = []
        child_env = None
        try:
            if resource == "gpu":
                assigned_gpus = acquire_gpus(gpu_count)
                child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(assigned_gpus)}
            suffix = f" gpu={','.join(assigned_gpus)}" if assigned_gpus else " cpu"
            label = f"{stage}:{name}{suffix}"
            print(f"\n[{label}] {shlex.join(command)}", flush=True)
            return name, run_command(command, env=child_env, label=label)
        finally:
            release_gpus(assigned_gpus)

    max_workers = min(parallel, len(work))
    failures: list[Exception] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, item): item[0] for item in work}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                completed_name, checkpoint = future.result()
            except Exception as exc:  # wait for in-flight jobs, then stop at the stage barrier
                failures.append(exc)
                print(f"\n[{stage}:{name}] FAILED: {exc}", file=sys.stderr, flush=True)
                continue
            if checkpoint:
                context["checkpoint"] = checkpoint
                if stage == "training":
                    context[f"training.{completed_name}.checkpoint"] = checkpoint
                    save_training_checkpoint(config, completed_name, checkpoint, target=target)
    if failures:
        raise failures[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run independent experiment stages from YAML")
    parser.add_argument("config", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--stages", help="Comma-separated stages to run")
    selection.add_argument("--start-from", help="Skip configured stages before this stage")
    parser.add_argument("--checkpoint", help="Value for ${checkpoint}; overrides the YAML checkpoint")
    parser.add_argument("--training-data", help="Required value for ${training_data} when selected commands use it")
    parser.add_argument(
        "--target",
        help="Run only command entries whose optional target field exactly matches this value",
    )
    parser.add_argument(
        "--topology-profile",
        help=(
            "Explicit factory-defined logical GPU topology profile. Protected launchers bind this profile "
            "into their immutable target attestation."
        ),
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum independent commands per stage (analysis remains ordered)",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated physical GPU ids; parallel commands get gpu_count exclusive ids each (default: 1)",
    )
    parser.add_argument(
        "--publish-training-outputs",
        action="store_true",
        help="As the declared publication owner, atomically merge completed target-scoped training outputs",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Validate and skip already completed training commands; interrupted commands restart normally",
    )
    parser.add_argument(
        "--onpolicy-target-attestation",
        type=Path,
        help=(
            "Immutable Qwen3.5 on-policy target sidecar. Requires one selected training target; "
            "the runner revalidates the recompiled child argv and passes this exact path to the child."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the config and commands without executing")
    parser.add_argument("-y", "--yes", action="store_true", help="Execute after printing the plan")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    onpolicy_target_attestation: Path | None = None
    try:
        source_config = load_experiment_source(args.config)
        reject_inline_secrets(source_config, path="experiment specification")
        config = compile_experiment(source_config, topology_profile=args.topology_profile)
        reject_inline_secrets(config, path="experiment")
        if args.publish_training_outputs:
            incompatible = []
            if args.stages or args.start_from:
                incompatible.append("--stages/--start-from")
            if args.target:
                incompatible.append("--target")
            if args.parallel != 1:
                incompatible.append("--parallel")
            if args.gpus:
                incompatible.append("--gpus")
            if args.checkpoint or args.training_data:
                incompatible.append("--checkpoint/--training-data")
            if args.dry_run:
                incompatible.append("--dry-run")
            if args.resume_completed:
                incompatible.append("--resume-completed")
            if args.onpolicy_target_attestation is not None:
                incompatible.append("--onpolicy-target-attestation")
            if incompatible:
                raise ExperimentConfigError("--publish-training-outputs cannot be combined with " + ", ".join(incompatible))
            publication_owner, publication_targets = _publication_manifest(config)
            plan_path, plan_digest = validate_resolved_plan(config)
            stages: list[str] = []
            preview: list[tuple[str, str, list[str]]] = []
            gpus: list[str] = []
            completed_training: dict[str, str] = {}
        else:
            publication_owner, publication_targets = None, None
            if args.parallel < 1:
                raise ExperimentConfigError("--parallel must be at least 1")
            gpus = _parse_gpus(args.gpus)
            if gpus and args.parallel == 1:
                raise ExperimentConfigError("--gpus only applies with --parallel greater than 1; sequential runs inherit the ambient CUDA environment")
            stages = select_stages(
                config,
                stages=[part.strip() for part in args.stages.split(",") if part.strip()] if args.stages else None,
                start_from=args.start_from,
            )
            if args.parallel > 1 and not gpus and not args.dry_run:
                for stage in stages:
                    if stage != "analysis" and any(command_resource(spec, stage) == "gpu" for spec in _entries(config, stage, target=args.target)):
                        raise ExperimentConfigError(f"parallel {stage} execution includes GPU commands; pass --gpus with the visible GPU ids")
            if args.parallel > 1 and gpus:
                for stage in stages:
                    if stage == "analysis":
                        continue
                    largest_bundle = max(
                        (command_gpu_count(spec, stage) for spec in _entries(config, stage, target=args.target)),
                        default=0,
                    )
                    if largest_bundle > len(gpus):
                        raise ExperimentConfigError(f"parallel {stage} command requests {largest_bundle} GPU(s), but only {len(gpus)} were supplied")
            context = initial_context(config, checkpoint=args.checkpoint, training_data=args.training_data)
            target_has_training = "training" in stages and bool(_entries(config, "training", target=args.target))
            completed_training = {}
            if args.resume_completed:
                if "training" not in stages:
                    raise ExperimentConfigError("--resume-completed requires the training stage to be selected")
                completed_training = validated_completed_training(config, target=args.target)
                context.update(load_output_context(config, target=args.target))
            elif not target_has_training:
                context.update(load_output_context(config))
            explicit_checkpoint = args.checkpoint or config.get("checkpoint")
            if explicit_checkpoint:
                context["checkpoint"] = explicit_checkpoint
            if selected_stages_use_placeholder(config, stages, "training_data", target=args.target) and not args.training_data:
                raise ExperimentConfigError("selected commands use ${training_data}; pass --training-data PATH")
            preview = planned_commands(config, stages, context, strict=False, target=args.target)
            if completed_training:
                preview = [item for item in preview if not (item[0] == "training" and item[1] in completed_training)]
            if args.target and not preview and not completed_training:
                raise ExperimentConfigError(f"no commands select target {args.target!r}")
            if args.onpolicy_target_attestation is not None:
                if args.target is None:
                    raise ExperimentConfigError("--onpolicy-target-attestation requires --target")
                if stages != ["training"]:
                    raise ExperimentConfigError("--onpolicy-target-attestation requires exactly --stages training")
                if args.resume_completed:
                    raise ExperimentConfigError("--onpolicy-target-attestation cannot be combined with --resume-completed")
                onpolicy_target_attestation = args.onpolicy_target_attestation.resolve()
                if not onpolicy_target_attestation.is_file():
                    raise ExperimentConfigError(
                        f"on-policy target attestation is missing: {onpolicy_target_attestation}"
                    )
                exact = planned_commands(config, stages, context, strict=True, target=args.target)
                if len(exact) != 1 or exact[0][0] != "training":
                    raise ExperimentConfigError(
                        "--onpolicy-target-attestation requires exactly one compiled selected training command"
                    )
                from experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight import (
                    validate_onpolicy_target_attestation_for_runner,
                )

                base_command = exact[0][2]
                validate_onpolicy_target_attestation_for_runner(
                    attestation=onpolicy_target_attestation,
                    plan=args.config,
                    target=args.target,
                    child_argv=[
                        *base_command[1:],
                        "--onpolicy-target-attestation",
                        str(onpolicy_target_attestation),
                    ],
                    interpreter=base_command[0],
                    # Parallel launch gives the one protected child the full
                    # selected bundle. Bind that exact inherited allocation,
                    # including logical GPU 0 (the coordinator), before any
                    # subprocess is allowed to start.
                    cuda_visible_devices=(
                        ",".join(gpus) if args.parallel > 1 else os.environ.get("CUDA_VISIBLE_DEVICES")
                    ),
                    topology_profile=args.topology_profile,
                )
                preview = [
                    (
                        stage,
                        name,
                        _with_onpolicy_target_attestation(
                            command,
                            stage=stage,
                            attestation=onpolicy_target_attestation,
                        ),
                    )
                    for stage, name, command in preview
                ]
            if not args.dry_run:
                missing = _missing_executables(preview)
                if missing:
                    details = ", ".join(f"{executable!r} (needed by {where})" for executable, where in missing.items())
                    raise ExperimentConfigError(f"selected commands need executables not on PATH: {details}; install them or narrow the run with --stages/--start-from")
            plan_path, plan_digest = validate_resolved_plan(config, target=args.target)
    except (ExperimentConfigError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    print("\nExperiment specification:")
    print(yaml.safe_dump(source_config, sort_keys=False).rstrip())
    print(f"\nResolved plan: {plan_path} (sha256:{plan_digest})")
    if args.publish_training_outputs:
        assert publication_owner is not None and publication_targets is not None
        print(f"\nTraining-output publication owner: {publication_owner}")
        for target_name, command_names in publication_targets.items():
            print(f"  {target_name}: {', '.join(command_names)}")
        if not args.yes and input("\nPublish these target outputs? [y/N] ").strip().lower() != "y":
            print("Aborted.")
            return
        try:
            saved_plan_path, saved_plan_digest = save_resolved_plan(config)
            print(f"\nSaved resolved plan: {saved_plan_path} (sha256:{saved_plan_digest})")
            state_path, state_digest = publish_training_outputs(config)
            print(f"Published canonical training outputs: {state_path} (sha256:{state_digest})")
        except (ExperimentConfigError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
        return
    if args.target:
        print(f"\nExecution target: {args.target}")
    if args.parallel > 1:
        gpu_text = ",".join(gpus) if gpus else "none (CPU stages only)"
        print(f"\nParallel execution: up to {args.parallel} commands; GPUs: {gpu_text}")
        print("Stage barriers are preserved; analysis commands remain ordered.")
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
        saved_plan_path, saved_plan_digest = save_resolved_plan(config, target=args.target)
        print(f"\nSaved resolved plan: {saved_plan_path} (sha256:{saved_plan_digest})")
        for stage in stages:
            if args.parallel > 1 and stage != "analysis":
                _run_stage_parallel(
                    config,
                    stage,
                    context,
                    target=args.target,
                    parallel=args.parallel,
                    gpus=gpus,
                    skip_completed=frozenset(completed_training),
                    onpolicy_target_attestation=onpolicy_target_attestation,
                )
                continue
            for index, spec in enumerate(_entries(config, stage, target=args.target), start=1):
                name = str(spec.get("name") or f"{stage}-{index}")
                if stage == "training" and name in completed_training:
                    print(f"\n[{stage}:{name}] SKIPPED: validated completed checkpoint", flush=True)
                    continue
                command = _with_onpolicy_target_attestation(
                    command_argv(spec, context, strict=True),
                    stage=stage,
                    attestation=onpolicy_target_attestation,
                )
                print(f"\n[{stage}:{name}] {shlex.join(command)}")
                checkpoint = run_command(command)
                if checkpoint:
                    context["checkpoint"] = checkpoint
                    if stage == "training":
                        context[f"training.{name}.checkpoint"] = checkpoint
                        save_training_checkpoint(config, name, checkpoint, target=args.target)
    except (ExperimentConfigError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
