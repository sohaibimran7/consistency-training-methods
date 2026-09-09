"""Verify a deduplicated CTM recovery tar and optionally restore selected files.

Only regular files from the original inventory can be restored. Symlink
targets were not captured. Restoration always uses a new destination and
retains partial files if verification fails; it never overwrites live data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


def _relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"unsafe artifact path: {value!r}")
    if path.as_posix() != value or value == ".":
        raise ValueError(f"noncanonical artifact path: {value!r}")
    return value


def recover(
    archive: str | Path,
    *,
    root: str | None = None,
    prefix: str | None = None,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Verify every blob, inventory binding and declared missing identity."""

    if destination is not None and root is None:
        raise ValueError("restoration requires an exact inventory root")
    if prefix is not None:
        prefix = _relative(prefix.rstrip("/"))
    target = Path(destination) if destination is not None else None
    if target is not None and (target.exists() or target.is_symlink()):
        raise FileExistsError(f"restore destination must be new: {target}")
    seen: dict[str, int] = {}
    expected: dict[str, int] = {}
    selected: dict[str, list[str]] = {}
    restored: list[str] = []
    roots: list[str] = []
    inventory = None
    inventory_digest = None
    manifest = None
    members: set[str] = set()
    with tarfile.open(archive, "r|") as tar:
        for member in tar:
            if not member.isfile() or member.name in members:
                raise ValueError(f"unexpected or duplicate recovery member: {member.name}")
            members.add(member.name)
            stream = tar.extractfile(member)
            assert stream is not None
            if member.name == "inventory.json":
                if inventory is not None or len(members) != 1:
                    raise ValueError("inventory must be the first recovery member")
                payload = stream.read()
                inventory_digest = hashlib.sha256(payload).hexdigest()
                inventory = json.loads(payload)
                roots = [entry["root"] for entry in inventory["roots"]]
                if len(roots) != len(set(roots)) or (root is not None and root not in roots):
                    raise ValueError("inventory roots are duplicated or requested root is absent")
                for location in inventory["roots"]:
                    paths: set[str] = set()
                    for entry in location["files"]:
                        relative = _relative(entry["path"])
                        if relative in paths:
                            raise ValueError(f"duplicate inventoried path: {relative}")
                        paths.add(relative)
                        digest = entry.get("sha256")
                        if digest is None:
                            continue
                        if entry.get("kind") != "file" or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                            raise ValueError("invalid regular artifact identity")
                        size = entry["size_bytes"]
                        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                            raise ValueError("invalid artifact size")
                        if digest in expected and expected[digest] != size:
                            raise ValueError("one digest has inconsistent inventoried sizes")
                        expected[digest] = size
                        if location["root"] == root and (prefix is None or relative == prefix or relative.startswith(prefix + "/")):
                            selected.setdefault(digest, []).append(relative)
                if target is not None:
                    if not selected:
                        raise ValueError("selection contains no inventoried regular files")
                    target.mkdir(parents=True, exist_ok=False)
            elif member.name == "manifest.json":
                if inventory is None:
                    raise ValueError("completion manifest precedes the inventory")
                manifest = json.loads(stream.read())
            elif re.fullmatch(r"blobs/[0-9a-f]{64}", member.name):
                if inventory is None or manifest is not None:
                    raise ValueError("blob outside the inventory/manifest envelope")
                digest = member.name.split("/")[1]
                if digest not in expected or member.size != expected[digest]:
                    raise ValueError(f"blob is not bound to the inventory: {digest}")
                hasher = hashlib.sha256()
                handles = []
                try:
                    if target is not None:
                        for relative in selected.get(digest, []):
                            path = target / relative
                            path.parent.mkdir(parents=True, exist_ok=True)
                            handles.append(path.open("xb"))
                    while chunk := stream.read(1024 * 1024):
                        hasher.update(chunk)
                        for handle in handles:
                            handle.write(chunk)
                finally:
                    for handle in handles:
                        handle.close()
                if hasher.hexdigest() != digest:
                    raise ValueError(f"artifact blob digest mismatch: {digest}; partial restoration retained")
                seen[digest] = member.size
                if target is not None:
                    restored.extend(selected.get(digest, []))
            else:
                raise ValueError(f"unexpected recovery member: {member.name}")
    if not isinstance(manifest, dict) or manifest.get("schema") != "ctm-artifact-recovery-v1":
        raise ValueError("archive has no supported completion manifest")
    missing = [entry["sha256"] for entry in manifest["unavailable_hashes"]]
    if len(missing) != len(set(missing)) or set(missing) != set(expected) - set(seen):
        raise ValueError("missing identities do not match the completion manifest")
    if manifest["inventory_sha256"] != inventory_digest:
        raise ValueError("inventory does not match the completion manifest")
    if manifest["unique_files"] != len(seen) or manifest["unique_bytes"] != sum(seen.values()):
        raise ValueError("archive contents do not match the completion counts")
    return {
        "schema": "ctm-artifact-recovery-verification-v1",
        "inventory_sha256": inventory_digest,
        "verified_files": len(seen),
        "verified_bytes": sum(seen.values()),
        "unavailable_hashes": missing,
        "roots": roots,
        "restored_files": sorted(restored),
        "selected_unavailable_hashes": sorted(set(selected) - set(seen)),
        "limitations": "Symlink targets and original file modes were not captured. Restored files use current process defaults.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--root", help="Exact root string from the original inventory")
    parser.add_argument("--prefix", help="Optional relative directory/file within that root")
    parser.add_argument("--destination", type=Path, help="New directory; omit to verify only")
    args = parser.parse_args()
    try:
        report = recover(args.archive, root=args.root, prefix=args.prefix, destination=args.destination)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(1, f"Recovery failed: {exc}. Any partial destination is retained.\n")
    print(json.dumps(report, indent=2))
    if report["selected_unavailable_hashes"]:
        parser.exit(2, "Selected artifact identities were unavailable when this backup was created.\n")


if __name__ == "__main__":
    main()
