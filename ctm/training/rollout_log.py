"""Persist RL gradient rollouts for later inspection.

Layout (under the run's log dir by default, ``experiments/<name>/rollouts/`` later):

    <dir>/
      index.json            # {"steps": [{"step", "file", "n_records", "p_ref_mean", ...}]}
      step_000001.jsonl.zst # one RolloutRecord per line (zstd-compressed JSONL)

Modes (``RLConfig.rollout_log``):
    "none"     — off.
    "gradient" — every gradient-receiving rollout (consistency + anchor) each step.
                 Rate-estimation-only rollouts are freed early by the loop for
                 memory and are not currently retained.

Read side: ``ctm.evals.analysis.rollouts.iter_rollouts`` / ``load_index``.
"""

import json
from pathlib import Path
from typing import Iterable, Optional

import zstandard

from ctm.core.types import RolloutRecord

INDEX_NAME = "index.json"


def step_filename(step: int) -> str:
    return f"step_{step:06d}.jsonl.zst"


class RolloutLogger:
    """Writes one compressed JSONL file of RolloutRecords per training step.

    The index is rewritten after each step (it is small) so a partially
    completed / crashed run still has a consistent view of what was captured.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._index: list[dict] = []
        index_path = self.directory / INDEX_NAME
        if index_path.exists():  # resumed run: extend the existing index
            try:
                self._index = json.loads(index_path.read_text()).get("steps", [])
            except (json.JSONDecodeError, OSError):
                self._index = []

    def log_step(self, records: Iterable[RolloutRecord]) -> Optional[Path]:
        records = list(records)
        if not records:
            return None
        step = records[0].step
        path = self.directory / step_filename(step)
        payload = "".join(r.model_dump_json() + "\n" for r in records)
        with open(path, "wb") as f:
            f.write(zstandard.ZstdCompressor().compress(payload.encode("utf-8")))

        p_hats = [r.p_hat for r in records if r.p_hat is not None]
        self._index = [e for e in self._index if e.get("step") != step]  # resume overwrite
        self._index.append({
            "step": step,
            "file": path.name,
            "n_records": len(records),
            "n_train": sum(1 for r in records if r.role == "train"),
            "n_anchor": sum(1 for r in records if r.role == "anchor"),
            "p_ref_mean": sum(r.p_ref for r in records) / len(records),
            "p_hat_mean": (sum(p_hats) / len(p_hats)) if p_hats else None,
            "trait_mean": sum(r.trait_value for r in records) / len(records),
        })
        self._index.sort(key=lambda e: e["step"])
        (self.directory / INDEX_NAME).write_text(json.dumps({"steps": self._index}, indent=1))
        return path
