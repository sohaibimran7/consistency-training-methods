"""Pinned evaluation settings for the Qwen3.5 no-CoT IID diagnostic.

This module contains no launcher and makes no model/API call.  It gives a
future scheduler one importable, testable boundary instead of copying the
historical CoT task factory or decode settings into an ad-hoc shell command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments.stage1_iid_diagnostic_none.prepare import PROMPT_STYLE, validate_manifest

BASE_MODEL = "Qwen/Qwen3.5-9B"
TASK_FACTORY = "experiments.stage1_iid_diagnostic_none.tasks:diagnostic_tasks"
TASK_COUNT = 4
SPLITS = ("train_eval", "heldout_in_domain")

# ``prompt_style: none`` intentionally does not pass Qwen's
# ``enable_thinking=false`` template flag.  It removes the explicit user CoT
# suffix while retaining the reasoning model's native template behaviour.
GENERATION_CONFIG: dict[str, Any] = {
    "extra_body": {"top_k": 20},
    "max_tokens": 20480,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}
VLLM_MODEL_ARGS: dict[str, Any] = {
    "provider": "vllm",
    "gpu_memory_utilization": 0.9,
    "language_model_only": True,
    "max_model_len": 32768,
    "max_num_seqs": 256,
}
NATIVE_VLLM_MODEL_ARGS: dict[str, Any] = {
    name: value for name, value in VLLM_MODEL_ARGS.items() if name != "provider"
}


def task_args(*, manifest: str | Path, split: str, unbiased_log: str | Path) -> dict[str, Any]:
    """Return one fail-closed task-factory argument object.

    Validation occurs before a scheduler starts an evaluator.  The task
    factory repeats the source verification in case a file changes between
    planning and task construction.
    """

    if split not in SPLITS:
        raise ValueError(f"unsupported no-CoT diagnostic split: {split!r}")
    log = str(unbiased_log)
    if not log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    manifest_path = Path(manifest).resolve()
    validate_manifest(manifest_path, verify_source=True)
    return {
        "include_bias_acknowledged": False,
        "manifest": str(manifest_path),
        "prompt_style": PROMPT_STYLE,
        "split": split,
        "unbiased_log": log,
    }


__all__ = [
    "BASE_MODEL",
    "GENERATION_CONFIG",
    "NATIVE_VLLM_MODEL_ARGS",
    "SPLITS",
    "TASK_COUNT",
    "TASK_FACTORY",
    "VLLM_MODEL_ARGS",
    "task_args",
]
