#!/usr/bin/env python3
"""Validate and print the fixed two-host Vast.ai Figure 6 GPU plan."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True, slots=True)
class Assignment:
    """One model server's physical GPU allocation and loopback port."""

    model_key: str
    gpu_ids: tuple[int, ...]
    port: int

    @property
    def gpu_list(self) -> str:
        return ",".join(str(gpu_id) for gpu_id in self.gpu_ids)


HOST_PLANS: dict[str, tuple[Assignment, ...]] = {
    "A": (
        Assignment("qwen36", (0,), 8100),
        Assignment("qwen32", (1,), 8101),
        Assignment("qwen_mo_mid", (2,), 8102),
        Assignment("qwen_mo_post", (3,), 8103),
        Assignment("llama33", (4, 5), 8104),
        Assignment("llama_mo_mid", (6, 7), 8105),
    ),
    "B": (Assignment("llama_mo_post", (0, 1), 8106),),
}

HOST_GPU_COUNTS = {"A": 8, "B": 2}


def load_models(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the pinned serving registry as model-keyed mappings."""

    registry = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not isinstance(registry.get("models"), dict):
        raise ValueError("model registry must contain a models mapping")
    return registry["models"]


def validate_plan(
    host: str,
    assignments: Sequence[Assignment],
    models: Mapping[str, Mapping[str, Any]],
    *,
    detected_gpu_count: int,
) -> None:
    """Reject plan drift, TP mismatches, port reuse, and GPU overlap/gaps."""

    if host not in HOST_GPU_COUNTS:
        raise ValueError(f"unknown host {host!r}; choose A or B")
    expected_gpu_count = HOST_GPU_COUNTS[host]
    if detected_gpu_count != expected_gpu_count:
        raise ValueError(
            f"host {host} requires exactly {expected_gpu_count} visible GPUs; detected {detected_gpu_count}"
        )
    if tuple(assignments) != HOST_PLANS[host]:
        raise ValueError(f"host {host} assignments do not match the fixed Figure 6 plan")

    seen_models: set[str] = set()
    seen_gpus: set[int] = set()
    seen_ports: set[int] = set()
    for assignment in assignments:
        if assignment.model_key in seen_models:
            raise ValueError(f"host {host} repeats model {assignment.model_key}")
        seen_models.add(assignment.model_key)
        if assignment.model_key not in models:
            raise ValueError(f"host {host} references unknown model {assignment.model_key}")
        model = models[assignment.model_key]
        tensor_parallel_size = model.get("tensor_parallel_size")
        if tensor_parallel_size != len(assignment.gpu_ids):
            raise ValueError(
                f"{assignment.model_key} requires TP={tensor_parallel_size}; plan assigns {len(assignment.gpu_ids)} GPUs"
            )
        if not assignment.gpu_ids or any(
            isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0 for gpu_id in assignment.gpu_ids
        ):
            raise ValueError(f"{assignment.model_key} has an invalid GPU list")
        overlap = seen_gpus.intersection(assignment.gpu_ids)
        if overlap:
            raise ValueError(f"host {host} GPU assignment overlaps at {sorted(overlap)}")
        if len(set(assignment.gpu_ids)) != len(assignment.gpu_ids):
            raise ValueError(f"{assignment.model_key} repeats a GPU in its assignment")
        seen_gpus.update(assignment.gpu_ids)
        if (
            isinstance(assignment.port, bool)
            or not isinstance(assignment.port, int)
            or not 1024 <= assignment.port <= 65535
        ):
            raise ValueError(f"{assignment.model_key} has invalid port {assignment.port!r}")
        if assignment.port in seen_ports:
            raise ValueError(f"host {host} repeats localhost port {assignment.port}")
        seen_ports.add(assignment.port)

    expected_gpus = set(range(expected_gpu_count))
    if seen_gpus != expected_gpus:
        raise ValueError(
            f"host {host} must assign physical GPUs {sorted(expected_gpus)} exactly once; got {sorted(seen_gpus)}"
        )


def validate_global_plan(models: Mapping[str, Mapping[str, Any]]) -> None:
    """Ensure the two host plans partition all seven models and all ports are unique."""

    assignments = [assignment for plan in HOST_PLANS.values() for assignment in plan]
    model_keys = [assignment.model_key for assignment in assignments]
    if len(model_keys) != len(set(model_keys)):
        raise ValueError("a model is assigned to more than one Vast host")
    if set(model_keys) != set(models):
        raise ValueError(
            f"Vast host plans must partition the model registry; plan={sorted(model_keys)}, registry={sorted(models)}"
        )
    ports = [assignment.port for assignment in assignments]
    if len(ports) != len(set(ports)):
        raise ValueError("Vast model localhost ports must be globally unique")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--host", choices=sorted(HOST_PLANS), required=True)
    parser.add_argument("--detected-gpu-count", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        models = load_models(args.models)
        validate_global_plan(models)
        assignments = HOST_PLANS[args.host]
        validate_plan(args.host, assignments, models, detected_gpu_count=args.detected_gpu_count)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    for assignment in assignments:
        print(f"{assignment.model_key}\t{assignment.gpu_list}\t{assignment.port}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
