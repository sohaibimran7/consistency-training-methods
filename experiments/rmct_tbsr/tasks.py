"""Inspect task factories for the exact RMCT training and HLE populations."""

from __future__ import annotations

import hashlib
from pathlib import Path

from experiments.rmct_tbsr.constants import (
    DEFAULT_HLE_DIR,
    HLE_BIASES,
    HLE_FILE_SHA256,
    HLE_FILES,
    TRAINING_DATASETS,
    TRAINING_SPLIT,
)
from experiments.rmct_tbsr.prepare import validate_training_manifest


def training_tasks(
    manifest: str,
    unbiased_log: str,
    prompt_style: str = "none",
    split: str = TRAINING_SPLIT,
):
    """Build clean LogiQA/HellaSwag tasks, then their matched biased tasks."""

    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty local Inspect log location")
    if prompt_style != "none":
        raise ValueError("the RMCT training population uses prompt_style='none'")
    if split != TRAINING_SPLIT:
        raise ValueError(f"the RMCT training split must be {TRAINING_SPLIT!r}")
    artifact = validate_training_manifest(manifest)

    from experiments.switch_gate.tasks import switch_gate_biased, switch_gate_unbiased

    clean = [
        switch_gate_unbiased(
            frozen_file=str(artifact.path),
            dataset=dataset,
            prompt_style=prompt_style,
            split=split,
        )
        for dataset in TRAINING_DATASETS
    ]
    biased = [
        switch_gate_biased(
            frozen_file=str(artifact.path),
            dataset=dataset,
            unbiased_log=unbiased_log,
            bias_type="wrong_argument",
            prompt_style=prompt_style,
            split=split,
        )
        for dataset in TRAINING_DATASETS
    ]
    return [*clean, *biased]


def hle_tasks(unbiased_log: str, hle_dir: str = DEFAULT_HLE_DIR):
    """Build clean HLE and the six RMCT biases in the paper's fixed order."""

    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty local Inspect log location")
    root = Path(hle_dir)
    paths = {name: root / filename for name, filename in HLE_FILES.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen RMCT HLE file does not exist: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != HLE_FILE_SHA256[name]:
            raise ValueError(f"{path}: SHA-256 mismatch for exact RMCT HLE {name!r} input")

    from experiments.switch_gate.tasks import hle_tasks as switch_gate_hle_tasks

    return switch_gate_hle_tasks(
        unbiased_file=str(paths["unbiased"]),
        bias_files={bias: str(paths[bias]) for bias in HLE_BIASES},
        unbiased_log=unbiased_log,
    )


__all__ = ["hle_tasks", "training_tasks"]
