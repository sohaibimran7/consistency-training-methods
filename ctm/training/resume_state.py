"""Strict local-RL checkpoint state for sequential data-scaling continuations.

The ordinary checkpoint manifest is deliberately backend-neutral. A local
on-policy continuation needs more than adapter weights, however: it must know
which optimizer update comes next and restore the coordinator's random-number
state before it shuffles the next frozen data segment. This module owns that
small, JSON-only state boundary.

It does *not* claim to serialize a vLLM engine's private sampling RNG. The
dedicated rollout workers expose no supported snapshot/restore API for that
state. Callers that need a bitwise-identical uninterrupted rollout stream must
therefore fail closed rather than treating this as an exact on-policy resume.
Sequential extensions are labelled ``optimizer_data_segment`` instead.
"""

from __future__ import annotations

import base64
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RL_LOOP_STATE_SCHEMA = "ctm.rl_loop_state.v1"
RUNTIME_RNG_SCHEMA = "ctm.rl_runtime_rng.v1"


def _as_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (str, int, float)) or value is None:
        return value
    raise TypeError(f"unsupported RNG-state value: {type(value).__name__}")


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _bytes_to_base64(value: Any) -> str:
    """Encode a CPU torch RNG tensor without persisting a pickle blob."""

    raw = bytes(value.detach().cpu().tolist())
    return base64.b64encode(raw).decode("ascii")


def _base64_to_uint8_tensor(value: Any):
    if not isinstance(value, str) or not value:
        raise ValueError("torch RNG state must be a non-empty base64 string")
    try:
        raw = base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("torch RNG state is not valid base64") from exc
    if not raw:
        raise ValueError("torch RNG state decodes to no bytes")
    import torch

    return torch.tensor(list(raw), dtype=torch.uint8)


def capture_runtime_rng_state() -> dict[str, Any]:
    """Capture coordinator RNG state in a portable, JSON-only form.

    The CUDA field is intentionally *one coordinator device*, not every
    visible GPU. Rollout workers are independent processes and their private
    vLLM streams are not serializable through PyTorch's CUDA RNG API.
    """

    state: dict[str, Any] = {
        "schema": RUNTIME_RNG_SCHEMA,
        "python_random_state": _to_jsonable(random.getstate()),
    }
    try:
        import torch

        state["torch_cpu_rng_state_base64"] = _bytes_to_base64(torch.get_rng_state())
        if torch.cuda.is_available():
            device = int(torch.cuda.current_device())
            state["torch_cuda_coordinator_device"] = device
            state["torch_cuda_rng_state_base64"] = _bytes_to_base64(torch.cuda.get_rng_state(device))
    except (ImportError, RuntimeError):
        # Tinker-only callers may not have a local torch/CUDA runtime. The
        # state remains useful for Python shuffling but is not strict enough
        # for a protected local sequential continuation.
        pass
    return state


def restore_runtime_rng_state(state: Mapping[str, Any], *, require_torch: bool) -> None:
    """Restore a state emitted by :func:`capture_runtime_rng_state`.

    Call after model/adapter restoration so construction does not consume the
    coordinator-side RNG state needed for the next shuffle.
    """

    if not isinstance(state, Mapping):
        raise ValueError("runtime_rng must be an object")
    if state.get("schema") != RUNTIME_RNG_SCHEMA:
        raise ValueError("runtime_rng has an unsupported schema")
    python_state = state.get("python_random_state")
    if not isinstance(python_state, list):
        raise ValueError("runtime_rng.python_random_state must be a JSON list")
    try:
        random.setstate(_tuple_tree(python_state))
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime_rng.python_random_state is invalid") from exc

    cpu = state.get("torch_cpu_rng_state_base64")
    if cpu is None:
        if require_torch:
            raise ValueError("strict local continuation requires torch_cpu_rng_state_base64")
        return
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - protected local runs ship torch
        raise ValueError("checkpoint contains torch RNG state but torch is unavailable") from exc
    torch.set_rng_state(_base64_to_uint8_tensor(cpu))

    cuda = state.get("torch_cuda_rng_state_base64")
    coordinator = state.get("torch_cuda_coordinator_device")
    if cuda is None and coordinator is None:
        return
    if not torch.cuda.is_available():
        raise ValueError("checkpoint contains coordinator CUDA RNG state but CUDA is unavailable")
    coordinator_index = _as_nonnegative_int(coordinator, label="runtime_rng.torch_cuda_coordinator_device")
    if coordinator_index >= torch.cuda.device_count():
        raise ValueError(
            "checkpoint coordinator CUDA RNG device is outside the current visible allocation: "
            f"{coordinator_index} >= {torch.cuda.device_count()}"
        )
    torch.cuda.set_rng_state(_base64_to_uint8_tensor(cuda), device=coordinator_index)


@dataclass(frozen=True, slots=True)
class RLResumeState:
    """Validated progress recovered from a completed local-RL checkpoint."""

    global_step: int
    optimizer_step: int
    completed_epochs: int
    runtime_rng: dict[str, Any]
    checkpoint_dir: Path


def _checkpoint_directory(resume_from: str | Path) -> Path:
    raw = str(resume_from)
    path = Path(raw[len("file://") :] if raw.startswith("file://") else raw).resolve()
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint must be a regular directory: {resume_from}")
    return path


def load_strict_local_rl_resume_state(resume_from: str | Path) -> RLResumeState:
    """Load a completed local-RL state or reject an ambiguous checkpoint.

    A data-scaling boundary must occur after an optimizer update. In
    particular, a checkpoint with partially accumulated gradients cannot be
    safely continued because gradients are intentionally not serialized.
    """

    checkpoint_dir = _checkpoint_directory(resume_from)
    manifest_path = checkpoint_dir / "manifest.json"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    for path, label in ((manifest_path, "checkpoint manifest"), (optimizer_path, "optimizer state")):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"strict local continuation requires a regular {label}: {path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"checkpoint manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    if document.get("backend") != "local" or document.get("kind") != "both":
        raise ValueError("strict local continuation requires a local kind='both' checkpoint")
    loop = document.get("loop_state")
    if not isinstance(loop, Mapping):
        raise ValueError("strict local continuation requires checkpoint loop_state")
    if loop.get("schema") != RL_LOOP_STATE_SCHEMA:
        raise ValueError(
            "strict local continuation requires a checkpoint written with "
            f"{RL_LOOP_STATE_SCHEMA}; this checkpoint cannot prove its loop/RNG state"
        )
    global_step = _as_nonnegative_int(loop.get("global_step"), label="checkpoint loop_state.global_step")
    optimizer_step = _as_nonnegative_int(loop.get("optimizer_step"), label="checkpoint loop_state.optimizer_step")
    completed_epochs = _as_nonnegative_int(loop.get("completed_epochs"), label="checkpoint loop_state.completed_epochs")
    if loop.get("step") != global_step:
        raise ValueError("checkpoint loop_state.step must equal global_step")
    if loop.get("accumulated_grads") != 0:
        raise ValueError("strict local continuation refuses a checkpoint with accumulated gradients")
    if not loop.get("final"):
        raise ValueError("strict local continuation requires a final segment checkpoint")
    runtime_rng = loop.get("runtime_rng")
    if not isinstance(runtime_rng, dict):
        raise ValueError("strict local continuation requires runtime_rng state")
    # Validate the complete payload without perturbing the caller's live RNG.
    if runtime_rng.get("schema") != RUNTIME_RNG_SCHEMA:
        raise ValueError("checkpoint runtime_rng has an unsupported schema")
    if not isinstance(runtime_rng.get("python_random_state"), list):
        raise ValueError("checkpoint runtime_rng lacks Python state")
    if not isinstance(runtime_rng.get("torch_cpu_rng_state_base64"), str):
        raise ValueError("checkpoint runtime_rng lacks CPU torch state")
    if not isinstance(runtime_rng.get("torch_cuda_rng_state_base64"), str):
        raise ValueError("checkpoint runtime_rng lacks coordinator CUDA state")
    if not isinstance(runtime_rng.get("torch_cuda_coordinator_device"), int):
        raise ValueError("checkpoint runtime_rng lacks coordinator CUDA device")
    return RLResumeState(
        global_step=global_step,
        optimizer_step=optimizer_step,
        completed_epochs=completed_epochs,
        runtime_rng=dict(runtime_rng),
        checkpoint_dir=checkpoint_dir,
    )


__all__ = [
    "RL_LOOP_STATE_SCHEMA",
    "RUNTIME_RNG_SCHEMA",
    "RLResumeState",
    "capture_runtime_rng_state",
    "load_strict_local_rl_resume_state",
    "restore_runtime_rng_state",
]
