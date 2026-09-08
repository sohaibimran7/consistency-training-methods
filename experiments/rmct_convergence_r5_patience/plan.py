"""Exact no-output-cap RMCT continuation from the sealed r4 step-176 state.

The optimizer, data, rollout counts, loss, and four-GH200 topology are inherited
from r4.  The two deliberate changes are isolated here:

* generation is EOS-only and passes ``--no-max-new-tokens``;
* convergence is owned by the separately attested eight-window patience guard.

The new namespace prevents capped r4 samples from being confused with this
uncapped continuation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.rmct_convergence_r4_recovery import plan as r4


MODEL = r4.MODEL
BASE_SNAPSHOT = r4.BASE_SNAPSHOT
CONDITION_NAME = r4.CONDITION_NAME
TOPOLOGY_PROFILE = r4.TOPOLOGY_PROFILE
GPU_COUNT = r4.GPU_COUNT
UPDATES_PER_SEGMENT = r4.UPDATES_PER_SEGMENT
TOTAL_SEGMENTS = r4.TOTAL_SEGMENTS
HARD_CAP_OPTIMIZER_STEPS = r4.HARD_CAP_OPTIMIZER_STEPS
PARENT_SEGMENT_INDEX = 10
START_SEGMENT_INDEX = PARENT_SEGMENT_INDEX + 1
PARENT_RUN_PREFIX = r4.RUN_PREFIX
RUN_PREFIX = "rmct-convergence-gcall-r2-mb40960-uncapped-patience-r5"
SEGMENT_SCHEMA = "rmct-convergence-uncapped-patience-r5-segment-v1"
COMMAND_SCHEMA = "rmct-convergence-uncapped-patience-r5-command-v1"
LOCAL_FORWARD_MICROBATCH_MAX_TOKENS = r4.RECOVERY_FORWARD_MICROBATCH_MAX_TOKENS

_GENERATION_CAP_KEYS = {
    "max_new_tokens",
    "max_tokens",
    "max_output_tokens",
    "max_completion_tokens",
}


class PlanError(ValueError):
    """The r5 continuation cannot be constructed without contract drift."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not START_SEGMENT_INDEX <= value < TOTAL_SEGMENTS:
        raise PlanError(f"r5 segment_index must be in [{START_SEGMENT_INDEX}, {TOTAL_SEGMENTS - 1}]")
    return value


def run_name(segment_index: int) -> str:
    return f"{RUN_PREFIX}-s{_index(segment_index) + 1:03d}"


def parent_run_name() -> str:
    return r4.run_name(PARENT_SEGMENT_INDEX)


def parent_checkpoint_path(repository: str | Path) -> Path:
    return r4.final_checkpoint_path(repository, PARENT_SEGMENT_INDEX)


def final_checkpoint_path(repository: str | Path, segment_index: int) -> Path:
    root = Path(repository).resolve()
    run = run_name(segment_index)
    return root / "logs" / CONDITION_NAME / run / "checkpoints" / f"{CONDITION_NAME}_{run}"


def _parent(repository: str | Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    if index == START_SEGMENT_INDEX:
        checkpoint = parent_checkpoint_path(repository)
        prefix = PARENT_RUN_PREFIX
        previous = PARENT_SEGMENT_INDEX
        run = parent_run_name()
        kind = "sealed_r4_terminal_checkpoint_superseded_by_patience_amendment"
    else:
        previous = index - 1
        checkpoint = final_checkpoint_path(repository, previous)
        prefix = RUN_PREFIX
        run = run_name(previous)
        kind = "sealed_r5_patience_checkpoint"
    return {
        "kind": kind,
        "condition": CONDITION_NAME,
        "run_prefix": prefix,
        "segment_index": previous,
        "run_name": run,
        "optimizer_step": (previous + 1) * UPDATES_PER_SEGMENT,
        "uri": f"file://{checkpoint}",
        "expected_kind": "both",
        "expected_final": True,
        "resume_with_optimizer": True,
        "resume_state_required": True,
    }


def segment_record(repository: str | Path, segment_index: int) -> dict[str, Any]:
    index = _index(segment_index)
    return {
        "schema": SEGMENT_SCHEMA,
        "condition": CONDITION_NAME,
        "run_prefix": RUN_PREFIX,
        "segment_index": index,
        "run_name": run_name(index),
        "optimizer_step_start": index * UPDATES_PER_SEGMENT + 1,
        "optimizer_step_end": (index + 1) * UPDATES_PER_SEGMENT,
        "optimizer_steps": UPDATES_PER_SEGMENT,
        "parent": _parent(repository, index),
        "generation": {"output_token_cap": None, "termination": "model_eos_only"},
    }


def _assert_no_generation_cap(value: Any, *, path: str = "args") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _GENERATION_CAP_KEYS:
                raise PlanError(f"generation cap key is forbidden at {path}.{key}")
            _assert_no_generation_cap(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_generation_cap(item, path=f"{path}[{index}]")


def segment_args(repository: str | Path, segment_index: int, *, model_snapshot: str | Path) -> dict[str, Any]:
    index = _index(segment_index)
    args = r4.segment_args(repository, index, model_path=model_snapshot)
    args["run_name"] = run_name(index)
    args.pop("max_new_tokens", None)
    args["no_max_new_tokens"] = True
    parent = _parent(repository, index)
    args.update(
        {
            "resume_from": parent["uri"],
            "resume_with_optimizer": True,
            "resume_state_required": True,
        }
    )
    _assert_no_generation_cap(args)
    if args.get("no_max_new_tokens") is not True:
        raise PlanError("r5 must explicitly select --no-max-new-tokens")
    return args


def command_attestation(
    repository: str | Path, segment_index: int, *, model_snapshot: str | Path
) -> dict[str, Any]:
    root = Path(repository).resolve()
    snapshot = Path(model_snapshot).resolve()
    if snapshot.is_symlink() or not snapshot.is_dir() or snapshot.name != BASE_SNAPSHOT or not (snapshot / "config.json").is_file():
        raise PlanError(f"pinned offline Qwen snapshot is invalid: {snapshot}")
    args = segment_args(root, segment_index, model_snapshot=snapshot)
    from scripts.run_experiment import _argument_tokens

    argv = [sys.executable, str((root / "scripts" / "train_rlct.py").resolve()), *_argument_tokens(args)]
    forbidden = {"--max-new-tokens", "--max-tokens", "--max-output-tokens", "--max-completion-tokens"}
    if forbidden.intersection(argv) or "--no-max-new-tokens" not in argv:
        raise PlanError("r5 argv must be EOS-only and contain no output-token cap")
    source = Path(__file__).resolve()
    return {
        "schema": COMMAND_SCHEMA,
        "condition": CONDITION_NAME,
        "run_prefix": RUN_PREFIX,
        "logical_segment_index": _index(segment_index),
        "plan": {"path": str(source), "sha256": _sha256(source)},
        "model": {"repo_id": MODEL, "revision": BASE_SNAPSHOT, "snapshot_path": str(snapshot)},
        "segment": segment_record(root, segment_index),
        "argv": argv,
        "environment_contract": {
            "cuda_visible_devices_preserved": True,
            "topology_profile": TOPOLOGY_PROFILE,
            "phase_shared": True,
            "gradient_checkpointing_layers": "all",
            "local_forward_microbatch_max_datums": 8,
            "local_forward_microbatch_max_tokens": LOCAL_FORWARD_MICROBATCH_MAX_TOKENS,
            "local_target_logprob_chunk_size": 2048,
            "output_token_cap": None,
            "generation_termination": "model_eos_only",
        },
    }


def write_command(path: str | Path, document: Mapping[str, Any]) -> str:
    target = Path(path).resolve()
    payload = _canonical(document)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise PlanError(f"refusing to overwrite different r5 command attestation: {target}")
        return "resumed"
    return "written"


def execute(
    repository: str | Path,
    segment_index: int,
    *,
    model_snapshot: str | Path,
    command_output: str | Path,
    yes: bool,
) -> int:
    if not yes:
        raise PlanError("r5 execution requires explicit yes=True")
    document = command_attestation(repository, segment_index, model_snapshot=model_snapshot)
    write_command(command_output, document)
    process = subprocess.run(document["argv"], cwd=Path(repository).resolve(), env=dict(os.environ), check=False)
    return int(process.returncode)


__all__ = [
    "BASE_SNAPSHOT",
    "COMMAND_SCHEMA",
    "CONDITION_NAME",
    "HARD_CAP_OPTIMIZER_STEPS",
    "MODEL",
    "PARENT_SEGMENT_INDEX",
    "RUN_PREFIX",
    "START_SEGMENT_INDEX",
    "TOTAL_SEGMENTS",
    "UPDATES_PER_SEGMENT",
    "PlanError",
    "command_attestation",
    "execute",
    "final_checkpoint_path",
    "parent_checkpoint_path",
    "parent_run_name",
    "run_name",
    "segment_args",
    "segment_record",
    "write_command",
]
