import hashlib
import io
import json
import tarfile

import pytest

from scripts.restore_artifacts import recover


def _archive(path, *, corrupt=False, unavailable=False):
    payload = b"historical result\n"
    digest = hashlib.sha256(payload).hexdigest()
    inventory = {
        "roots": [
            {
                "root": "/original/repo",
                "files": [
                    {"path": name, "kind": "file", "sha256": digest, "size_bytes": len(payload)}
                    for name in ("artifacts/result.txt", "logs/copy.txt")
                ],
            }
        ]
    }
    raw_inventory = json.dumps(inventory).encode()
    manifest = {
        "schema": "ctm-artifact-recovery-v1",
        "inventory_sha256": hashlib.sha256(raw_inventory).hexdigest(),
        "unique_files": 0 if unavailable else 1,
        "unique_bytes": 0 if unavailable else len(payload),
        "unavailable_hashes": [{"sha256": digest}] if unavailable else [],
    }
    with tarfile.open(path, "w") as archive:
        def add(name, content):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        add("inventory.json", raw_inventory)
        if not unavailable:
            add(f"blobs/{digest}", b"X" * len(payload) if corrupt else payload)
        add("manifest.json", json.dumps(manifest).encode())
    return payload


def test_recovery_verifies_deduplicated_contents_and_restores_selected_copy(tmp_path):
    archive = tmp_path / "backup.tar"
    payload = _archive(archive)
    destination = tmp_path / "restored"
    report = recover(archive, root="/original/repo", prefix="artifacts", destination=destination)
    assert report["verified_files"] == 1
    assert report["restored_files"] == ["artifacts/result.txt"]
    assert (destination / "artifacts/result.txt").read_bytes() == payload
    assert not (destination / "logs").exists()
    with pytest.raises(FileExistsError):
        recover(archive, root="/original/repo", destination=destination)


def test_recovery_refuses_corrupted_bytes_and_reports_unavailable_selection(tmp_path):
    corrupt = tmp_path / "corrupt.tar"
    _archive(corrupt, corrupt=True)
    destination = tmp_path / "partial"
    with pytest.raises(ValueError, match="digest mismatch"):
        recover(corrupt, root="/original/repo", destination=destination)
    assert destination.exists()  # Failed restoration is retained for inspection.
    missing = tmp_path / "missing.tar"
    _archive(missing, unavailable=True)
    report = recover(missing, root="/original/repo", destination=tmp_path / "missing-selection")
    assert report["verified_files"] == 0
    assert len(report["selected_unavailable_hashes"]) == 1
