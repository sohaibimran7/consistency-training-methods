"""Training-run provenance manifest.

Written into the run's log dir at train start by BOTH loops. This is the seed of
the phase-2 "checkpoints own their provenance" design: the manifest travels with
the run, so eval/analysis can classify in-domain vs cross-domain automatically
instead of relying on hand-maintained model_registry fields.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ctm.cli_safety import redact_secrets
from ctm.training.run_utils import get_git_state

MANIFEST_NAME = "manifest.json"


def config_hash(config_dump: dict) -> str:
    """Stable hash of a config dump (order-independent)."""
    canonical = json.dumps(config_dump, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def write_run_manifest(
    log_dir: str | Path,
    *,
    kind: str,  # "rl" | "sft"
    model: str,
    backend: Any,  # TrainingBackend instance (class name is recorded)
    config_dump: dict,  # full pydantic model_dump of the run config
    extra: Optional[dict] = None,
) -> Path:
    git = get_git_state()
    git.pop("git_diff", None)  # the full diff already goes to WandB; keep the manifest small
    redacted_config = redact_secrets(config_dump)
    redacted_extra = redact_secrets(extra or {})
    manifest = {
        "kind": kind,
        "model": model,
        "backend": type(backend).__name__,
        "config_hash": config_hash(redacted_config),
        "config": redacted_config,
        "git": git,
        "written_at": datetime.now(timezone.utc).isoformat(),
        **redacted_extra,
    }
    path = Path(log_dir) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return path


def read_run_manifest(log_dir: str | Path) -> Optional[dict]:
    path = Path(log_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
