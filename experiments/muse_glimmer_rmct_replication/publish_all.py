"""Publish all completed Muse Glimmer RMCT evaluation families.

This is a post-evaluation consumer only.  It validates the two-bias and AITA
completion receipts, renders standard switch/verbalisation checkpoint plots,
renders the paired final-answer-only AITA plot, and writes an umbrella file
identity manifest.  It performs no model or grader request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.elephant_aita_ntaflip import publication as aita_publication
from experiments.muse_glimmer_rmct_replication import publication as checkpoint_publication
from infra.isambard import run_muse_glimmer_rmct_aita_ntaflip_16gpu as muse_aita


SCHEMA = "muse-glimmer-rmct-complete-publication-v1"
AITA_LABELS = {
    "base": "Muse Glimmer base",
    "step016": "Muse Glimmer RMCT global/data step 16",
    "step064": "Muse Glimmer RMCT global/data step 64",
    "final": "Muse Glimmer RMCT final",
}


class MusePublishAllError(ValueError):
    """Completed Muse inputs are inconsistent with the publication contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size < 1:
        raise MusePublishAllError(f"{label} must be a non-empty regular file: {candidate}")
    resolved = candidate.resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _read_json(path: str | Path, *, label: str) -> dict[str, Any]:
    candidate = Path(path)
    _identity(candidate, label=label)
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MusePublishAllError(f"invalid {label}: {candidate}") from exc
    if not isinstance(value, dict):
        raise MusePublishAllError(f"{label} must contain an object")
    return value


def validate_aita_completion(campaign_root: str | Path) -> tuple[list[tuple[str, Path]], dict[str, Any]]:
    """Return publication inputs only after exact Muse campaign completion."""

    muse_aita._configure()
    audited = muse_aita.audited
    paths = audited._campaign_paths(campaign_root)
    completion = audited._read_json(paths.completion, label="Muse AITA completion")
    if completion.get("schema") != audited.COMPLETION_SCHEMA or completion.get("campaign") != audited.CAMPAIGN_NAME:
        raise MusePublishAllError("Muse AITA completion has the wrong schema/campaign")
    rows = completion.get("conditions")
    if not isinstance(rows, list):
        raise MusePublishAllError("Muse AITA completion lacks condition rows")
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, Mapping)}
    if tuple(by_name) != tuple(AITA_LABELS):
        raise MusePublishAllError("Muse AITA completion has the wrong condition order")
    inputs: list[tuple[str, Path]] = []
    for condition_name, label in AITA_LABELS.items():
        condition = audited._condition(condition_name)
        condition_paths = audited._condition_paths(paths, condition)
        expected = audited._identity(condition_paths.preflight, label=f"Muse AITA {condition_name} preflight")
        if by_name[condition_name].get("preflight") != expected:
            raise MusePublishAllError(f"Muse AITA {condition_name} completion does not bind its preflight")
        inputs.append((label, condition_paths.preflight))
    return inputs, _identity(paths.completion, label="Muse AITA completion")


def _output_identities(root: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MusePublishAllError(f"publication output contains a symlink: {path}")
        if not path.is_file() or path.name == "manifest.json" and path.parent == root:
            continue
        identity = _identity(path, label="Muse publication output")
        identity["relative_path"] = str(path.relative_to(root))
        identities.append(identity)
    return identities


def publish_all(
    *,
    early_campaign_root: str | Path,
    late_campaign_root: str | Path,
    luna_output_root: str | Path,
    aita_campaign_root: str | Path,
    output_root: str | Path,
) -> Path:
    destination = Path(output_root).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite Muse complete publication: {destination}")
    destination.mkdir(parents=True)

    switch_logs, switch_sources = checkpoint_publication.load_inputs(
        metric="towards_bias_switch",
        early_campaign_root=early_campaign_root,
        late_campaign_root=late_campaign_root,
    )
    checkpoint_publication.render_checkpoint_comparison(
        logs_by_condition=switch_logs,
        source_manifest=switch_sources,
        output_dir=destination / "switch-rate",
        metric="towards_bias_switch",
    )
    verbal_logs, verbal_sources = checkpoint_publication.load_inputs(
        metric="bias_acknowledged",
        early_campaign_root=early_campaign_root,
        late_campaign_root=late_campaign_root,
        luna_output_root=luna_output_root,
    )
    checkpoint_publication.render_checkpoint_comparison(
        logs_by_condition=verbal_logs,
        source_manifest=verbal_sources,
        output_dir=destination / "bias-verbalisation",
        metric="bias_acknowledged",
    )

    aita_inputs, aita_completion = validate_aita_completion(aita_campaign_root)
    aita_publication.publish(aita_inputs, output_dir=destination / "aita-nta-flip")
    manifest = {
        "schema": SCHEMA,
        "model": "meta-models/Muse-Glimmer-30B",
        "conditions": list(checkpoint_publication.CONDITIONS),
        "source_completions": {
            "two_bias_early": switch_sources["early"],
            "two_bias_late": switch_sources["late"],
            "luna_early": verbal_sources["early"],
            "luna_late": verbal_sources["late"],
            "aita": aita_completion,
        },
        "outputs": _output_identities(destination),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--early-campaign-root", required=True, type=Path)
    parser.add_argument("--late-campaign-root", required=True, type=Path)
    parser.add_argument("--luna-output-root", required=True, type=Path)
    parser.add_argument("--aita-campaign-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = publish_all(
            early_campaign_root=args.early_campaign_root,
            late_campaign_root=args.late_campaign_root,
            luna_output_root=args.luna_output_root,
            aita_campaign_root=args.aita_campaign_root,
            output_root=args.output_root,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())


__all__ = ["AITA_LABELS", "SCHEMA", "publish_all", "validate_aita_completion"]
