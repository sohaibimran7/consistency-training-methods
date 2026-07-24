"""Read/filter RL rollouts persisted by ``ctm.training.rollout_log.RolloutLogger``.

Typical use:

    from ctm.evals.analysis.rollouts import iter_rollouts, load_index

    load_index("logs/rl_wfs/run1/rollouts")                      # cheap step overview
    for r in iter_rollouts("logs/rl_wfs/run1/rollouts",
                           steps=[12], perturbation_idx=1, trait=1.0):
        print(r.advantage, r.completion_text[:200])
"""

import json
from pathlib import Path
from typing import Iterator, Optional, Sequence

import zstandard

from ctm.core.types import RolloutRecord
from ctm.training.rollout_log import INDEX_NAME


def load_index(directory: str | Path) -> list[dict]:
    """Per-step summaries (step, file, counts, mean rates) without decompressing anything."""
    path = Path(directory) / INDEX_NAME
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("steps", [])


def iter_rollouts(
    directory: str | Path,
    *,
    steps: Optional[Sequence[int]] = None,
    perturbation_idx: Optional[int] = None,
    role: Optional[str] = None,
    trait: Optional[float] = None,
    datapoint_idx: Optional[int] = None,
    skipped_from_training: Optional[bool] = None,
    skip_reason: Optional[str] = None,
) -> Iterator[RolloutRecord]:
    """Stream records, decompressing only the step files that match ``steps``."""
    directory = Path(directory)
    wanted = set(steps) if steps is not None else None
    for entry in load_index(directory):
        if wanted is not None and entry["step"] not in wanted:
            continue
        raw = zstandard.ZstdDecompressor().decompress((directory / entry["file"]).read_bytes())
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = RolloutRecord.model_validate_json(line)
            if perturbation_idx is not None and record.perturbation_idx != perturbation_idx:
                continue
            if role is not None and record.role != role:
                continue
            if trait is not None and record.trait_value != trait:
                continue
            if datapoint_idx is not None and record.datapoint_idx != datapoint_idx:
                continue
            if skipped_from_training is not None and record.skipped_from_training != skipped_from_training:
                continue
            if skip_reason is not None and record.skip_reason != skip_reason:
                continue
            yield record
