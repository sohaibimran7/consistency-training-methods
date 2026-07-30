"""Thin, policy-free constructors for Inspect datasets and tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import import_module
from typing import Any

_RESERVED_TASK_OPTIONS = frozenset({"dataset", "name", "scorer", "solver"})


def build_memory_dataset(
    samples: Iterable[Any],
    *,
    name: str | None = None,
    location: str | None = None,
    shuffled: bool = False,
) -> Any:
    """Build an Inspect ``MemoryDataset`` from caller-created Samples."""

    _validate_optional_text(name, field="dataset name")
    _validate_optional_text(location, field="dataset location")
    if not isinstance(shuffled, bool):
        raise TypeError("dataset shuffled must be a bool")
    if isinstance(samples, (str, bytes, bytearray, Mapping)):
        raise TypeError("samples must be an iterable of Inspect Sample objects")
    try:
        materialized = list(samples)
    except TypeError as exc:
        raise TypeError("samples must be an iterable of Inspect Sample objects") from exc

    try:
        dataset_module = import_module("inspect_ai.dataset")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Inspect dataset construction requires the optional 'inspect-ai' package") from exc
    memory_dataset = getattr(dataset_module, "MemoryDataset", None)
    if memory_dataset is None:
        raise ImportError("inspect_ai.dataset does not provide MemoryDataset")
    return memory_dataset(samples=materialized, name=name, location=location, shuffled=shuffled)


def build_inspect_task(
    samples: Iterable[Any],
    *,
    scorer: Any,
    solver: Any,
    task_name: str | None = None,
    dataset_name: str | None = None,
    dataset_location: str | None = None,
    dataset_shuffled: bool = False,
    task_options: Mapping[str, Any] | None = None,
) -> Any:
    """Build an Inspect ``Task`` without selecting benchmark policy.

    Samples, scorer, solver, and any optional task metadata/configuration are
    supplied by the caller. This helper only creates ``MemoryDataset`` and
    ``Task`` instances and does not transform, select, score, or tag examples.
    """

    _validate_optional_text(task_name, field="task name")
    if scorer is None:
        raise ValueError("scorer must be supplied explicitly")
    if solver is None:
        raise ValueError("solver must be supplied explicitly")
    if task_options is None:
        options: dict[str, Any] = {}
    elif not isinstance(task_options, Mapping):
        raise TypeError("task_options must be a mapping")
    else:
        options = dict(task_options)
    conflicts = sorted(_RESERVED_TASK_OPTIONS.intersection(options))
    if conflicts:
        raise ValueError(f"task_options cannot override explicit constructor fields: {conflicts}")

    dataset = build_memory_dataset(
        samples,
        name=dataset_name,
        location=dataset_location,
        shuffled=dataset_shuffled,
    )
    try:
        inspect_ai = import_module("inspect_ai")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Inspect task construction requires the optional 'inspect-ai' package") from exc
    task = getattr(inspect_ai, "Task", None)
    if task is None:
        raise ImportError("inspect_ai does not provide Task")
    return task(
        dataset=dataset,
        scorer=scorer,
        solver=solver,
        name=task_name,
        **options,
    )


def _validate_optional_text(value: str | None, *, field: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{field} must be None or a non-empty string")


__all__ = ["build_inspect_task", "build_memory_dataset"]
