"""Checkpoint scheduling shared by the SFT and RL loops, backend-injected.

Same schedule logic as the legacy ``cot_transparency.apis.tinker.common`` helpers,
but persisting through ``TrainingBackend.save_checkpoint`` instead of a raw
tinker training client.
"""

from typing import Optional

from ctm.backends.base import TrainingBackend
from ctm.training.run_utils import build_checkpoint_name


def _checkpoint_path(paths: dict) -> str:
    return paths.get("sampler_path") or paths.get("state_path")


async def save_intermediate_checkpoint(
    backend: TrainingBackend,
    *,
    experiment_name: str,
    run_name: str,
    checkpoint_cfg,
    global_step: int,
    total_steps: int,
    epoch: int,
    log_dir,
    checkpoint_paths: list,
    logger,
) -> Optional[str]:
    """Save an intermediate checkpoint when the schedule fires (and we're not near final).

    Returns the checkpoint path, or None if nothing was saved. Appends to `checkpoint_paths`
    and logs `{"checkpoint": path}` as a side effect, matching the prior inline behaviour.
    """
    steps_remaining = total_steps - global_step
    near_final = steps_remaining <= checkpoint_cfg.skip_near_final_steps
    if not (checkpoint_cfg.save_every_n_steps
            and global_step % checkpoint_cfg.save_every_n_steps == 0
            and not near_final):
        return None
    name = build_checkpoint_name(experiment_name, run_name, step=global_step)
    kind = "both" if checkpoint_cfg.save_state else "sampler"
    paths = await backend.save_checkpoint(
        name=name,
        log_dir=str(log_dir),
        loop_state={"epoch": epoch, "step": global_step},
        kind=kind,
    )
    path = _checkpoint_path(paths)
    checkpoint_paths.append(path)
    logger.log_metrics({"checkpoint": path}, step=global_step)
    return path


async def finalize_checkpoint(
    backend: TrainingBackend,
    *,
    experiment_name: str,
    run_name: str,
    n_epochs: int,
    save_state: bool,
    global_step: int,
    log_dir,
    checkpoint_paths: list,
    logger,
) -> str:
    """Save the final (no step-suffix) checkpoint, log it, and close the logger. Returns the path."""
    final_name = build_checkpoint_name(experiment_name, run_name)
    kind = "both" if save_state else "sampler"
    paths = await backend.save_checkpoint(
        name=final_name,
        log_dir=str(log_dir),
        loop_state={"epoch": n_epochs, "step": global_step, "final": True},
        kind=kind,
    )
    final_path = _checkpoint_path(paths)
    checkpoint_paths.append(final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")
    logger.log_metrics({"final_checkpoint": final_path}, step=global_step)
    logger.log_hparams({"final_checkpoint": final_path, "all_checkpoints": checkpoint_paths})
    logger.close()
    return final_path
