"""Deploy the frozen Stage 2 JSONL substrate without changing its bytes.

The canonical manifest contains workstation-absolute artifact paths.  This
module is the narrow, write-once deployment boundary for a copied Isambard
tree: it substitutes *only* the fourteen artifact ``path`` values and records
the source-manifest digest plus every byte identity.  It never copies or
modifies a JSONL file itself.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


DEPLOYMENT_SCHEMA = "rmct-two-bias-stage2-deployment-v1"
# The canonical workstation manifest is copied to Isambard with its original
# absolute paths intact. Those paths are expected to be nonexistent there, so
# source validation is byte/payload-layout based; only the deployed output is
# passed to the legacy path-opening validator.
CANONICAL_SOURCE_MANIFEST_SHA256 = "147b1739ad407058e71385f473d8f050e4689166f9a97122c40de506372d7850"
_HEX = frozenset("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} must be a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return document


def _artifact_entries(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    populations = document.get("populations")
    if not isinstance(populations, Mapping) or set(populations) != {"in_domain", "hle"}:
        raise ValueError("Stage 2 deployment source has invalid populations")
    entries: dict[str, dict[str, Any]] = {}
    for population in ("in_domain", "hle"):
        population_record = populations[population]
        if not isinstance(population_record, Mapping):
            raise ValueError(f"Stage 2 deployment source has invalid {population} population")
        artifacts = population_record.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"Stage 2 deployment source has invalid {population} artifacts")
        for bias, entry in artifacts.items():
            if not isinstance(bias, str) or not isinstance(entry, Mapping):
                raise ValueError("Stage 2 deployment source has malformed artifact entry")
            entries[f"{population}/{bias}"] = dict(entry)
    if len(entries) != 14:
        raise ValueError(f"Stage 2 deployment source must contain 14 artifacts, got {len(entries)}")
    return entries


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"refusing to overwrite differing deployment manifest: {path}")
        return "resumed"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(f"deployment manifest appeared and differs: {path}") from None
            return "resumed"
    finally:
        temporary.unlink(missing_ok=True)
    return "written"


def materialize_deployment_manifest(
    source_manifest: str | Path,
    artifact_root: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Publish a manifest whose artifact paths point at verified copied JSONLs.

    ``artifact_root`` mirrors the source manifest's directory layout, e.g.
    its ``in_domain/`` and ``hle/`` directories.  The caller must copy the
    JSONLs beforehand; this function validates every source and destination
    hash, performs no file copy, and writes only ``output``.
    """

    source_path = Path(source_manifest).expanduser().resolve()
    root = Path(artifact_root).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(f"source Stage 2 manifest must be a regular file: {source_path}")
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"artifact_root must be a regular directory: {root}")
    if destination == source_path:
        raise ValueError("deployment manifest output must differ from source manifest")

    source_digest = _sha256_file(source_path)
    if source_digest != CANONICAL_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "source Stage 2 manifest does not byte-match the pinned canonical "
            f"manifest: {source_path}"
        )
    # Do not call legacy validate_manifest here: its canonical artifact paths
    # point at the workstation and must be allowed to be absent on Isambard.
    # The pinned manifest digest plus the checks below bind its unchanged
    # layout and all declared JSONL identities before any path rewrite.
    source = _read_object(source_path, label="source Stage 2 manifest")
    if "deployment_provenance" in source:
        raise ValueError("source manifest must be the canonical un-deployed Stage 2 manifest")
    deployed = copy.deepcopy(source)
    source_entries = _artifact_entries(source)
    # Validate the copy's layout too, but mutate the actual nested entries
    # below rather than the detached dictionaries returned by the helper.
    _artifact_entries(deployed)
    copied: dict[str, dict[str, Any]] = {}
    for key in sorted(source_entries):
        source_entry = source_entries[key]
        population, _, bias = key.partition("/")
        deployed_entry = deployed["populations"][population]["artifacts"][bias]
        if not isinstance(deployed_entry, dict):  # defensive after deep copy / source validation
            raise ValueError(f"deployed Stage 2 artifact entry is malformed: {key}")
        raw_source = source_entry.get("path")
        expected_sha = source_entry.get("content_sha256")
        expected_size = source_entry.get("byte_count")
        if not isinstance(raw_source, str) or not raw_source or not _is_sha256(expected_sha):
            raise ValueError(f"source Stage 2 artifact has incomplete identity: {key}")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 1:
            raise ValueError(f"source Stage 2 artifact has invalid byte count: {key}")
        # Never resolve/open the canonical source path. It deliberately names
        # a different host. The immutable manifest hash fixes it as provenance
        # while the labelled population/bias entry fixes the copied layout.
        # The canonical pin authenticates this basename too.  In particular,
        # the clean/unbiased artifact is named ``clean.jsonl``, not
        # ``unbiased.jsonl``; do not invent a filename from a scientific key.
        source_name = Path(raw_source).name
        if source_name in {"", ".", ".."} or Path(source_name).suffix != ".jsonl":
            raise ValueError(f"source Stage 2 artifact has an unsafe immutable filename: {key}")
        target = (root / key.partition("/")[0] / source_name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:  # pragma: no cover - relative paths above make this defensive
            raise ValueError(f"deployed Stage 2 artifact escapes artifact_root: {target}") from exc
        if target.is_symlink() or not target.is_file():
            raise FileNotFoundError(f"deployed Stage 2 artifact must be a regular copied JSONL: {target}")
        if _sha256_file(target) != expected_sha or target.stat().st_size != expected_size:
            raise ValueError(f"deployed Stage 2 artifact bytes differ from source identity: {target}")
        deployed_entry["path"] = str(target)
        copied[key] = {
            "source_path": raw_source,
            "deployed_path": str(target),
            "sha256": expected_sha,
            "byte_count": expected_size,
        }

    deployed["deployment_provenance"] = {
        "schema": DEPLOYMENT_SCHEMA,
        "source_manifest": {"path": str(source_path), "sha256": source_digest},
        "artifact_root": str(root),
        "copied_artifacts": copied,
        "path_rewrite_policy": "only_population_artifact_paths_rewritten_jsonl_bytes_unchanged",
    }
    status = _write_once(destination, _canonical_json(deployed))
    # Re-run the existing verifier over the *deployed* manifest. It opens the
    # copied JSONLs at their new paths and verifies their old immutable hashes.
    from experiments.stage2_ood_hle.materialize import validate_manifest

    validate_manifest(destination)
    return {
        "schema": DEPLOYMENT_SCHEMA,
        "status": status,
        "manifest_path": str(destination),
        "manifest_sha256": _sha256_file(destination),
        "source_manifest_sha256": source_digest,
        "artifact_root": str(root),
        "artifacts": copied,
    }


def validate_deployment_manifest(manifest: str | Path) -> dict[str, Any] | None:
    """Validate optional deployment provenance after normal Stage 2 validation."""

    from experiments.stage2_ood_hle.materialize import validate_manifest

    path = Path(manifest).expanduser().resolve()
    validate_manifest(path)
    document = _read_object(path, label="Stage 2 manifest")
    provenance = document.get("deployment_provenance")
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "schema",
        "source_manifest",
        "artifact_root",
        "copied_artifacts",
        "path_rewrite_policy",
    }:
        raise ValueError("deployment manifest has malformed deployment provenance")
    source = provenance.get("source_manifest")
    if (
        provenance.get("schema") != DEPLOYMENT_SCHEMA
        or not isinstance(source, Mapping)
        or set(source) != {"path", "sha256"}
        or not isinstance(source.get("path"), str)
        or not Path(source["path"]).is_absolute()
        or source.get("sha256") != CANONICAL_SOURCE_MANIFEST_SHA256
        or not isinstance(provenance.get("artifact_root"), str)
        or not Path(provenance["artifact_root"]).is_absolute()
        or provenance.get("path_rewrite_policy")
        != "only_population_artifact_paths_rewritten_jsonl_bytes_unchanged"
    ):
        raise ValueError("deployment manifest does not bind its canonical source identity")
    copied = provenance.get("copied_artifacts")
    if not isinstance(copied, Mapping):
        raise ValueError("deployment manifest has no copied-artifact provenance")
    entries = _artifact_entries(document)
    if set(copied) != set(entries):
        raise ValueError("deployment manifest copied-artifact set differs from Stage 2 artifacts")
    root = Path(provenance["artifact_root"]).resolve()
    for key, entry in entries.items():
        record = copied[key]
        if not isinstance(record, Mapping) or set(record) != {"source_path", "deployed_path", "sha256", "byte_count"}:
            raise ValueError(f"deployment manifest has malformed copied-artifact record: {key}")
        artifact_path = Path(str(entry.get("path", ""))).resolve()
        if (
            record.get("deployed_path") != str(artifact_path)
            or record.get("sha256") != entry.get("content_sha256")
            or record.get("byte_count") != entry.get("byte_count")
            or not isinstance(record.get("source_path"), str)
            or not Path(record["source_path"]).is_absolute()
        ):
            raise ValueError(f"deployment manifest copied-artifact identity conflicts with Stage 2 artifact: {key}")
        try:
            artifact_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"deployment manifest artifact escapes artifact_root: {key}") from exc
    return dict(provenance)


__all__ = [
    "CANONICAL_SOURCE_MANIFEST_SHA256",
    "DEPLOYMENT_SCHEMA",
    "materialize_deployment_manifest",
    "validate_deployment_manifest",
]
