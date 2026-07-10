"""Run bookkeeping shared by the SFT and RL loops: log paths, git state, LR fallback.

Canonical home of helpers previously in ``cot_transparency.apis.tinker.common``
(which now re-exports from here).
"""

import logging
import subprocess
from typing import Optional

from tinker_cookbook import hyperparam_utils

_log = logging.getLogger(__name__)


class SafeFileWrapper:
    """Wraps a file object to silently handle BrokenPipeError.

    When running as a background process, the parent may close its pipe
    (e.g., Claude Code session refresh). tqdm and print calls then raise
    BrokenPipeError. This wrapper swallows those errors so training can
    continue uninterrupted.
    """
    def __init__(self, fp):
        self._fp = fp

    def write(self, s):
        try:
            return self._fp.write(s)
        except BrokenPipeError:
            return 0

    def flush(self):
        try:
            self._fp.flush()
        except BrokenPipeError:
            pass

    def __getattr__(self, name):
        return getattr(self._fp, name)


def build_checkpoint_name(
    experiment_name: str,
    run_name: str,
    step: Optional[int] = None,
) -> str:
    """
    Build checkpoint name from experiment and run names.

    Examples:
        - Final: "bct_debug_control"
        - Intermediate: "bct_debug_control_step100"
    """
    base = f"{experiment_name}_{run_name}"
    return f"{base}_step{step}" if step is not None else base


def build_log_dir(base_dir: str, experiment_name: str, run_name: str) -> str:
    """
    Build log directory path.

    Example: "logs/bct_debug/control/"
    """
    return f"{base_dir}/{experiment_name}/{run_name}"


def get_git_state() -> dict:
    """Capture current git state for reproducibility logging.

    Returns a dict with commit SHA, branch, dirty flag, changed files list,
    and the full diff of uncommitted changes (truncated to 50k chars).
    Degrades gracefully if not in a git repo.
    """
    def _run(args: list[str]) -> str:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        if not sha:
            return {"git_error": "not a git repository"}
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty_files = _run(["git", "status", "--short"])
        diff = _run(["git", "diff"])
        max_diff = 50_000
        return {
            "git_sha": sha,
            "git_branch": branch,
            "git_dirty": len(dirty_files) > 0,
            "git_dirty_files": dirty_files,
            "git_diff": diff[:max_diff] + ("\n... (truncated)" if len(diff) > max_diff else ""),
        }
    except Exception as e:
        return {"git_error": str(e)}


def warn_if_dirty(git_state: dict) -> None:
    """Print a prominent warning if the git working tree is dirty."""
    if git_state.get("git_dirty"):
        files = git_state.get("git_dirty_files", "")
        n_files = len([l for l in files.splitlines() if l.strip()])
        print(
            f"\n{'='*60}\n"
            f"WARNING: Git working tree is DIRTY ({n_files} file(s) changed)\n"
            f"Commit: {git_state.get('git_sha', 'unknown')}\n"
            f"Branch: {git_state.get('git_branch', 'unknown')}\n"
            f"Changed files:\n{files}\n"
            f"The diff is logged to WandB for reproducibility.\n"
            f"{'='*60}\n"
        )


def get_recommended_lr(model: str, is_lora: bool = True, fallback: float = 1e-4) -> float:
    """
    Get recommended learning rate for a model using Tinker's hyperparam_utils.

    Falls back to default if model not in hyperparam_utils.
    """
    try:
        return hyperparam_utils.get_lr(model, is_lora=is_lora)
    except Exception as e:
        # get_lr raises ConfigurationError (a ValueError subclass, NOT in the old
        # KeyError/AssertionError/NotImplementedError/OSError tuple) for any model that is
        # neither Llama/Qwen nor in its explicit list — so an arbitrary base model with no
        # explicit --lr used to crash LR resolution instead of falling back. The documented
        # contract is "fall back for unknown models", so catch broadly and warn.
        _log.warning(
            "get_recommended_lr(%s) failed (%s: %s); using fallback lr=%g",
            model, type(e).__name__, e, fallback,
        )
        return fallback
