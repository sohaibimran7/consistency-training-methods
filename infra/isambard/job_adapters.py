"""Build fail-closed controller requests for the maintained Isambard jobs.

The shared controller deliberately stores a self-contained shell script rather
than a path into a mutable checkout.  These adapters capture the maintained
wrapper body at enqueue time, remove its scheduler directives, and put the
allocation shape in ``resources``.  They do not submit jobs or inspect remote
state.

Only output locations that the underlying wrapper can actually control are
accepted.  In particular, the r5 continuation writes beneath ``REPO_DIR``;
it cannot be redirected to an arbitrary output directory.  The interactive
r5 profile below is therefore a bounded hardware diagnostic, not a second
training continuation.  The established isolated-duplicate workflow remains
an explicit, separately prepared operation until it has a controller-native
preparation contract.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Optional

RMCT_R5_SEGMENT = "rmct_r5_segment"
RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC = "rmct_r5_interactive_gpu_diagnostic"
GEMMA_MAIN_16GPU = "gemma_main_16gpu"
GEMMA_SMOKE = "gemma_smoke"
GEMMA_EOS_DEBUG = "gemma_eos_debug"

PROFILES = frozenset(
    {
        RMCT_R5_SEGMENT,
        RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC,
        GEMMA_MAIN_16GPU,
        GEMMA_SMOKE,
        GEMMA_EOS_DEBUG,
    }
)

_SBATCH = re.compile(r"^\s*#SBATCH\b")
_ENVIRONMENT_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_CREDENTIAL_LIKE = re.compile(r"PASSWORD|PASSWD|SECRET|TOKEN|API_KEY|PRIVATE_KEY|CREDENTIAL")

_RMCT_SEGMENT_WRAPPER = Path("infra/isambard/run_qwen35_rmct_convergence_r5_patience_segment.sbatch")
_GEMMA_MAIN_WRAPPER = Path("infra/isambard/run_gemma4_12b_base_two_bias_evals_16gpu.sbatch")
_GEMMA_SMOKE_WRAPPER = Path("infra/isambard/run_gemma4_12b_base_two_bias_smoke.sbatch")
_GEMMA_DEBUG_WRAPPER = Path("infra/isambard/debug_gemma_eos.sbatch")

_GEMMA_CAMPAIGN_NAME = "gemma4-12b-base-two-bias-50x21-16gpu-v1"


class AdapterError(ValueError):
    """A requested adapter profile would misrepresent its wrapper contract."""


def _remote_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AdapterError(f"{field} must be a non-empty remote path")
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) == "/" or ".." in path.parts or "//" in value or value.endswith("/"):
        raise AdapterError(f"{field} must be a normalised, non-root absolute remote path")
    normal = str(path)
    if normal != value:
        raise AdapterError(f"{field} must already be normalised")
    return normal


def _under(path: str, parent: str, *, field: str) -> None:
    child = PurePosixPath(path)
    root = PurePosixPath(parent)
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise AdapterError(f"{field} must be inside remote_dir") from exc


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AdapterError(f"{field} must be non-empty text")
    return value


def _positive_minutes(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AdapterError("minutes must be a positive integer")
    return value


def _environment(value: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise AdapterError("env must be an object containing non-secret configuration values")
    result: dict[str, str] = {}
    for name, item in value.items():
        if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
            raise AdapterError("env contains an invalid environment variable name")
        if _CREDENTIAL_LIKE.search(name):
            raise AdapterError(f"env must not contain credential-like value {name}")
        if not isinstance(item, str) or not item or "\x00" in item:
            raise AdapterError(f"env value for {name} must be non-empty text")
        result[name] = item
    return result


def _allow_environment(env: dict[str, str], *, allowed: set[str], required: set[str]) -> dict[str, str]:
    unexpected = sorted(set(env) - allowed)
    if unexpected:
        raise AdapterError(f"unsupported configuration for this profile: {', '.join(unexpected)}")
    missing = sorted(required - set(env))
    if missing:
        raise AdapterError(f"missing required configuration for this profile: {', '.join(missing)}")
    return {name: env[name] for name in sorted(env)}


def _absolute_configuration(env: Mapping[str, str], names: set[str]) -> None:
    for name in names.intersection(env):
        _remote_path(env[name], field=f"env.{name}")


def _capture_wrapper(checkout: Path, relative_path: Path, *, injection: str = "") -> str:
    path = checkout / relative_path
    if not path.is_file() or path.is_symlink():
        raise AdapterError(f"required maintained wrapper is missing or linked: {path}")
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AdapterError(f"could not read maintained wrapper: {path}") from exc
    if not source.startswith("#!"):
        raise AdapterError(f"maintained wrapper has no shell shebang: {path}")
    script = "".join(line for line in source.splitlines(keepends=True) if not _SBATCH.match(line))
    if injection:
        marker = "set -euo pipefail\n"
        if marker not in script:
            raise AdapterError(f"maintained wrapper has no safe injection point: {path}")
        script = script.replace(marker, marker + injection, 1)
    if not script.endswith("\n"):
        script += "\n"
    return script


def _resources(
    *,
    nodes: int,
    gpus: int,
    minutes: int,
    memory_mb: int,
    cpus_per_task: Optional[int] = None,
    cpus_per_gpu: Optional[int] = None,
    gpus_per_node: Optional[int] = None,
) -> dict[str, int]:
    if cpus_per_task is not None and cpus_per_gpu is not None:
        raise AssertionError("a wrapper may choose only one CPU allocation shape")
    result = {"nodes": nodes, "gpus": gpus, "minutes": minutes, "memory_mb": memory_mb}
    if cpus_per_task is not None:
        result["cpus_per_task"] = cpus_per_task
    if cpus_per_gpu is not None:
        result["cpus_per_gpu"] = cpus_per_gpu
    if gpus_per_node is not None:
        result["gpus_per_node"] = gpus_per_node
    return result


def _request(
    *,
    request_id: str,
    owner: str,
    mode: str,
    output_roots: list[str],
    script: str,
    resources: dict[str, int],
    remote_dir: str,
    env: dict[str, str],
) -> dict[str, Any]:
    return {
        "id": _text(request_id, field="request_id"),
        "owner": _text(owner, field="owner"),
        "mode": mode,
        "output_roots": output_roots,
        "script": script,
        "resources": resources,
        "remote_dir": remote_dir,
        "env": env,
    }


def _require_mode(profile: str, mode: str, allowed: set[str]) -> None:
    if mode not in {"interactive", "batch"}:
        raise AdapterError("mode must be either 'interactive' or 'batch'")
    if mode not in allowed:
        supported = " or ".join(sorted(allowed))
        raise AdapterError(f"{profile} must use {supported} mode")


def _require_minutes(profile: str, minutes: int, expected: int) -> None:
    if minutes != expected:
        raise AdapterError(f"{profile} requires {expected} minutes to preserve the maintained wrapper contract")


def _require_at_most_minutes(profile: str, minutes: int, maximum: int) -> None:
    if minutes > maximum:
        raise AdapterError(f"{profile} allows at most {maximum} minutes")


def _validate_rmct_r5_segment_index(segment_index: int) -> None:
    if not 11 <= segment_index <= 31:
        raise AdapterError("CTM_RMCT_SEGMENT_INDEX must be in [11, 31]")


def _build_rmct_segment(
    *,
    request_id: str,
    owner: str,
    checkout: Path,
    remote_dir: str,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict[str, Any]:
    _require_mode(RMCT_R5_SEGMENT, mode, {"batch"})
    _require_minutes(RMCT_R5_SEGMENT, minutes, 12 * 60)
    supplied = _allow_environment(
        env,
        allowed={"SCRATCHDIR", "CTM_RMCT_SEGMENT_INDEX"},
        required={"SCRATCHDIR", "CTM_RMCT_SEGMENT_INDEX"},
    )
    _absolute_configuration(supplied, {"SCRATCHDIR"})
    try:
        index = int(supplied["CTM_RMCT_SEGMENT_INDEX"])
    except ValueError as exc:
        raise AdapterError("CTM_RMCT_SEGMENT_INDEX must be an integer") from exc
    _validate_rmct_r5_segment_index(index)
    if output_root != remote_dir:
        raise AdapterError(
            "rmct_r5_segment writes its continuation, checkpoint, and parity-adjacent evidence below REPO_DIR; "
            "output_root must equal remote_dir"
        )
    return _request(
        request_id=request_id,
        owner=owner,
        mode=mode,
        output_roots=[remote_dir],
        script=_capture_wrapper(checkout, _RMCT_SEGMENT_WRAPPER),
        resources=_resources(
            nodes=1,
            gpus=4,
            minutes=minutes,
            memory_mb=200 * 1024,
            cpus_per_gpu=16,
        ),
        remote_dir=remote_dir,
        env={"REPO_DIR": remote_dir, **supplied},
    )


def _rmct_diagnostic_script() -> str:
    """A small four-GPU probe that does no model import, preparation, or training."""

    return """#!/usr/bin/env bash
# Controller-owned non-training RMCT hardware diagnostic.
set -euo pipefail
umask 077

: "${CTM_RMCT_DIAGNOSTIC_OUTPUT_ROOT:?CTM_RMCT_DIAGNOSTIC_OUTPUT_ROOT is required}"
mkdir -p "$CTM_RMCT_DIAGNOSTIC_OUTPUT_ROOT"
exec srun --nodes=1 --ntasks=1 --gpus=4 --cpus-per-task=64 --exact \\
    bash -c '
        set -euo pipefail
        visible=${CUDA_VISIBLE_DEVICES:-}
        IFS=, read -r -a devices <<< "$visible"
        if [[ ${#devices[@]} -ne 4 ]]; then
            echo "ERROR: expected exactly four Slurm-visible GPUs, got: $visible" >&2
            exit 2
        fi
        for device in "${devices[@]}"; do
            if [[ -z "$device" || "$device" == "-1" || "$device" == "NoDevFiles" ]]; then
                echo "ERROR: invalid Slurm-visible GPU: $device" >&2
                exit 2
            fi
        done
        output="$CTM_RMCT_DIAGNOSTIC_OUTPUT_ROOT/hardware-${SLURM_JOB_ID:?SLURM_JOB_ID is required}.txt"
        if [[ -e "$output" ]]; then
            echo "ERROR: diagnostic output already exists: $output" >&2
            exit 2
        fi
        {
            printf "slurm_job_id=%s\\n" "$SLURM_JOB_ID"
            printf "cuda_visible_devices=%s\\n" "$visible"
            nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader
        } | tee "$output"
    '
"""


def _build_rmct_diagnostic(
    *,
    request_id: str,
    owner: str,
    remote_dir: str,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict[str, Any]:
    _require_mode(RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC, mode, {"interactive"})
    _require_at_most_minutes(RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC, minutes, 30)
    supplied = _allow_environment(env, allowed=set(), required=set())
    return _request(
        request_id=request_id,
        owner=owner,
        mode=mode,
        output_roots=[output_root],
        script=_rmct_diagnostic_script(),
        resources=_resources(
            nodes=1,
            gpus=4,
            minutes=minutes,
            memory_mb=200 * 1024,
            cpus_per_task=64,
        ),
        remote_dir=remote_dir,
        env={"CTM_RMCT_DIAGNOSTIC_OUTPUT_ROOT": output_root, **supplied},
    )


def _gemma_environment(env: dict[str, str], *, remote_dir: str, output_root: str) -> dict[str, str]:
    supplied = _allow_environment(
        env,
        allowed={
            "CTM_GEMMA4_12B_SNAPSHOT",
            "CTM_GEMMA4_EVAL_PYTHON",
            "CTM_GEMMA4_EVAL_SOURCE_STAGE2_MANIFEST",
            "CTM_GEMMA4_EVAL_STAGE2_ARTIFACT_ROOT",
        },
        required={"CTM_GEMMA4_12B_SNAPSHOT"},
    )
    _absolute_configuration(
        supplied,
        {
            "CTM_GEMMA4_12B_SNAPSHOT",
            "CTM_GEMMA4_EVAL_PYTHON",
            "CTM_GEMMA4_EVAL_SOURCE_STAGE2_MANIFEST",
            "CTM_GEMMA4_EVAL_STAGE2_ARTIFACT_ROOT",
        },
    )
    if PurePosixPath(output_root).name != _GEMMA_CAMPAIGN_NAME:
        raise AdapterError(f"Gemma campaign output_root must be named {_GEMMA_CAMPAIGN_NAME!r}")
    parent = str(PurePosixPath(output_root).parent)
    return {
        "REPO_DIR": remote_dir,
        "CTM_GEMMA4_EVAL_PARENT_ROOT": parent,
        "CTM_GEMMA4_EVAL_CAMPAIGN_ROOT": output_root,
        **supplied,
    }


def _build_gemma_main(
    *,
    request_id: str,
    owner: str,
    checkout: Path,
    remote_dir: str,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict[str, Any]:
    _require_mode(GEMMA_MAIN_16GPU, mode, {"batch"})
    _require_minutes(GEMMA_MAIN_16GPU, minutes, 12 * 60)
    return _request(
        request_id=request_id,
        owner=owner,
        mode=mode,
        output_roots=[output_root],
        script=_capture_wrapper(checkout, _GEMMA_MAIN_WRAPPER),
        resources=_resources(
            nodes=4,
            gpus=16,
            minutes=minutes,
            memory_mb=400 * 1024,
            cpus_per_gpu=16,
            gpus_per_node=4,
        ),
        remote_dir=remote_dir,
        env=_gemma_environment(env, remote_dir=remote_dir, output_root=output_root),
    )


def _build_gemma_smoke(
    *,
    request_id: str,
    owner: str,
    checkout: Path,
    remote_dir: str,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict[str, Any]:
    _require_mode(GEMMA_SMOKE, mode, {"interactive", "batch"})
    _require_minutes(GEMMA_SMOKE, minutes, 2 * 60)
    return _request(
        request_id=request_id,
        owner=owner,
        mode=mode,
        output_roots=[output_root],
        script=_capture_wrapper(checkout, _GEMMA_SMOKE_WRAPPER),
        resources=_resources(
            nodes=1,
            gpus=1,
            minutes=minutes,
            memory_mb=96 * 1024,
            cpus_per_gpu=16,
            gpus_per_node=1,
        ),
        remote_dir=remote_dir,
        env=_gemma_environment(env, remote_dir=remote_dir, output_root=output_root),
    )


def _gemma_debug_environment(env: dict[str, str], *, remote_dir: str, output_root: str) -> tuple[dict[str, str], bool]:
    supplied = _allow_environment(
        env,
        allowed={
            "CTM_GEMMA4_12B_SNAPSHOT",
            "CTM_GEMMA4_EOS_DEBUG_MANIFEST",
            "CTM_GEMMA4_EOS_DEBUG_RANK",
            "CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS",
            "CTM_GEMMA4_EOS_DEBUG_TASK_INDEX",
            "CTM_GEMMA4_EOS_DEBUG_MEMORY_EVERY",
            "CTM_GEMMA4_EVAL_PYTHON",
        },
        required={"CTM_GEMMA4_12B_SNAPSHOT", "CTM_GEMMA4_EOS_DEBUG_MANIFEST"},
    )
    _absolute_configuration(
        supplied, {"CTM_GEMMA4_12B_SNAPSHOT", "CTM_GEMMA4_EOS_DEBUG_MANIFEST", "CTM_GEMMA4_EVAL_PYTHON"}
    )
    for name, default, lower, upper in (
        ("CTM_GEMMA4_EOS_DEBUG_RANK", "0", 0, 15),
        ("CTM_GEMMA4_EOS_DEBUG_TASK_INDEX", "3", 1, 21),
        ("CTM_GEMMA4_EOS_DEBUG_MEMORY_EVERY", "256", 1, None),
    ):
        if name == "CTM_GEMMA4_EOS_DEBUG_RANK" and "CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS" in supplied:
            continue
        raw = supplied.get(name, default)
        try:
            number = int(raw)
        except ValueError as exc:
            raise AdapterError(f"{name} must be an integer") from exc
        if number < lower or (upper is not None and number > upper):
            bounded = f"[{lower}, {upper}]" if upper is not None else f">= {lower}"
            raise AdapterError(f"{name} must be {bounded}")
        supplied[name] = str(number)
    use_question_ids = "CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS" in supplied
    if use_question_ids:
        try:
            question_ids = json.loads(supplied["CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS"])
        except json.JSONDecodeError as exc:
            raise AdapterError("CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS must be a JSON array of question IDs") from exc
        if (
            not isinstance(question_ids, list)
            or not question_ids
            or any(not isinstance(item, str) or not item for item in question_ids)
            or len(question_ids) != len(set(question_ids))
        ):
            raise AdapterError("CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS must be a non-empty JSON array of distinct strings")
        if "CTM_GEMMA4_EOS_DEBUG_RANK" in supplied:
            raise AdapterError("select either CTM_GEMMA4_EOS_DEBUG_RANK or CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS")
    return (
        {
            "REPO_DIR": remote_dir,
            "CTM_GEMMA4_EOS_DEBUG_LOG_DIR": output_root,
            **supplied,
        },
        use_question_ids,
    )


def _gemma_debug_arguments(*, use_question_ids: bool) -> str:
    selector = (
        '    --question-ids "$CTM_GEMMA4_EOS_DEBUG_QUESTION_IDS" \\\n'
        if use_question_ids
        else '    --rank "$CTM_GEMMA4_EOS_DEBUG_RANK" \\\n'
    )
    return (
        """\
# The maintained debug wrapper forwards "$@". Bind its documented CLI inputs
# from the immutable request rather than accepting scheduler-side positional input.
set -- \\
    --manifest "$CTM_GEMMA4_EOS_DEBUG_MANIFEST" \\
    --model-snapshot "$CTM_GEMMA4_12B_SNAPSHOT" \\
    --log-dir "$CTM_GEMMA4_EOS_DEBUG_LOG_DIR" \\
    --task-index "$CTM_GEMMA4_EOS_DEBUG_TASK_INDEX" \\
"""
        + selector
        + """    --memory-every "$CTM_GEMMA4_EOS_DEBUG_MEMORY_EVERY"
"""
    )


def _build_gemma_debug(
    *,
    request_id: str,
    owner: str,
    checkout: Path,
    remote_dir: str,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict[str, Any]:
    _require_mode(GEMMA_EOS_DEBUG, mode, {"interactive", "batch"})
    _require_minutes(GEMMA_EOS_DEBUG, minutes, 30)
    runtime_env, use_question_ids = _gemma_debug_environment(env, remote_dir=remote_dir, output_root=output_root)
    return _request(
        request_id=request_id,
        owner=owner,
        mode=mode,
        output_roots=[output_root],
        script=_capture_wrapper(
            checkout,
            _GEMMA_DEBUG_WRAPPER,
            injection=_gemma_debug_arguments(use_question_ids=use_question_ids),
        ),
        resources=_resources(
            nodes=1,
            gpus=1,
            minutes=minutes,
            memory_mb=96 * 1024,
            cpus_per_gpu=16,
            gpus_per_node=1,
        ),
        remote_dir=remote_dir,
        env=runtime_env,
    )


def build_request(
    profile: str,
    *,
    request_id: str,
    owner: str,
    checkout: Path,
    remote_dir: str,
    output_root: str,
    mode: str,
    minutes: int,
    env: dict[str, str],
) -> dict[str, Any]:
    """Build one immutable controller request from a maintained profile.

    ``checkout`` is local and is used only to capture the maintained shell
    wrapper.  ``remote_dir`` and ``output_root`` are already-normalised
    Isambard paths; they are deliberately never resolved on the laptop.
    """

    if profile not in PROFILES:
        raise AdapterError(f"unknown Isambard job profile: {profile!r}")
    local_checkout = Path(checkout).expanduser()
    if not local_checkout.is_dir() or local_checkout.is_symlink():
        raise AdapterError(f"checkout must be a regular local directory: {local_checkout}")
    remote = _remote_path(remote_dir, field="remote_dir")
    output = _remote_path(output_root, field="output_root")
    _under(output, remote, field="output_root")
    requested_minutes = _positive_minutes(minutes)
    configuration = _environment(env)

    common = {
        "request_id": request_id,
        "owner": owner,
        "checkout": local_checkout,
        "remote_dir": remote,
        "output_root": output,
        "mode": mode,
        "minutes": requested_minutes,
        "env": configuration,
    }
    if profile == RMCT_R5_SEGMENT:
        return _build_rmct_segment(**common)
    if profile == RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC:
        common.pop("checkout")
        return _build_rmct_diagnostic(**common)
    if profile == GEMMA_MAIN_16GPU:
        return _build_gemma_main(**common)
    if profile == GEMMA_SMOKE:
        return _build_gemma_smoke(**common)
    return _build_gemma_debug(**common)


__all__ = [
    "AdapterError",
    "GEMMA_EOS_DEBUG",
    "GEMMA_MAIN_16GPU",
    "GEMMA_SMOKE",
    "PROFILES",
    "RMCT_R5_INTERACTIVE_GPU_DIAGNOSTIC",
    "RMCT_R5_SEGMENT",
    "build_request",
]
