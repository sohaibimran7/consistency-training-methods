"""Model-registry entries from run manifests — provenance, not hand-bookkeeping.

Every training run writes ``manifest.json`` into its log dir
(ctm.training.manifest). This module turns those into
``sycophancy_eval_inspect/model_registry.json``-style entry *suggestions*, so
"which biases was this checkpoint trained on" comes from the run itself instead
of a hand-maintained field.

Suggestions are meant to be merged (styling like color/hatch stays human-chosen):

    from ctm.evals.analysis.registry import scan_manifests, registry_suggestions
    print(json.dumps(registry_suggestions("logs"), indent=2))
"""

import json
from pathlib import Path
from typing import Optional

from ctm.training.manifest import MANIFEST_NAME


def manifest_registry_entry(manifest: dict) -> dict:
    """A registry-entry suggestion for one run manifest."""
    cfg = manifest.get("config", {})
    meta = cfg.get("run_metadata", {}) or {}
    return {
        "display_name": cfg.get("run_name") or cfg.get("experiment_name", "unknown"),
        "training_type": manifest.get("kind"),                # "rl" | "sft"
        "base_model": manifest.get("model"),
        "backend": manifest.get("backend"),
        "setting": meta.get("setting"),
        "training_biases": meta.get("bias_types", []),
        "datasets": meta.get("datasets", []),
        "prompt_style": meta.get("prompt_style"),
        "control": meta.get("control", False),
        "config_hash": manifest.get("config_hash"),
        "written_at": manifest.get("written_at"),
    }


def scan_manifests(logs_root: str | Path) -> dict[str, dict]:
    """All run manifests under a logs tree, keyed by '<experiment>/<run>' dir."""
    logs_root = Path(logs_root)
    found: dict[str, dict] = {}
    for path in sorted(logs_root.glob(f"**/{MANIFEST_NAME}")):
        try:
            manifest = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "kind" not in manifest or "config" not in manifest:
            continue  # not a run manifest (e.g. a local-checkpoint manifest)
        found[str(path.parent.relative_to(logs_root))] = manifest
    return found


def registry_suggestions(logs_root: str | Path, setting: Optional[str] = None) -> dict[str, dict]:
    """Registry-entry suggestions for every run under ``logs_root``.

    Args:
        setting: only include runs whose run_metadata.setting matches (None = all).
    """
    out: dict[str, dict] = {}
    for run_dir, manifest in scan_manifests(logs_root).items():
        entry = manifest_registry_entry(manifest)
        if setting is not None and entry.get("setting") != setting:
            continue
        out[run_dir] = entry
    return out
