"""Focused tests for portable source and runtime provenance."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import ctm.provenance as provenance
from ctm.provenance import (
    RuntimeProvenanceError,
    SourceSnapshotError,
    capture_runtime_identity,
    capture_source_snapshot,
    default_source_root,
    restore_source_snapshot,
    verify_runtime_identity,
    verify_source_snapshot,
)


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        os.chmod(path, 0o755)


def _capture(source_root: Path, run_dir: Path) -> dict:
    return capture_source_snapshot(
        source_root,
        bundle_dir=run_dir / "provenance" / "source",
        bundle_root=run_dir,
    )


def _git(args: list[str], *, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class _FixtureDistribution:
    def __init__(self, name: str, version: str, direct_url: dict[str, object] | None = None) -> None:
        self.metadata = {"Name": name}
        self.version = version
        self._direct_url = None if direct_url is None else json.dumps(direct_url)

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return self._direct_url


def test_default_source_root_is_anchored_to_the_package_not_ambient_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert (default_source_root() / "ctm" / "provenance.py").is_file()


def test_runtime_identity_records_only_runtime_and_distribution_metadata() -> None:
    runtime = capture_runtime_identity()
    assert runtime["python"]["implementation"]
    assert runtime["python"]["version"]
    assert runtime["platform"]["system"]
    assert runtime["distributions"]
    assert verify_runtime_identity(runtime)["runtime_sha256"] == runtime["runtime_sha256"]

    tampered = dict(runtime)
    tampered["platform"] = {**runtime["platform"], "machine": "tampered"}
    with pytest.raises(RuntimeProvenanceError, match="integrity digest"):
        verify_runtime_identity(tampered, actual=runtime)


def test_runtime_vcs_direct_url_is_sanitised_and_commit_changes_identity(monkeypatch) -> None:
    def capture_for_commit(commit_id: str) -> dict:
        distribution = _FixtureDistribution(
            "mcq-bias",
            "0.1.0",
            {
                "url": "git+https://user:secret-token@github.com/example/mcq-bias.git?access_token=secret#fragment",
                "vcs_info": {"vcs": "git", "commit_id": commit_id},
            },
        )
        monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: [distribution])
        return capture_runtime_identity()

    first = capture_for_commit("a" * 40)
    second = capture_for_commit("b" * 40)
    assert first["distributions"] == [
        {
            "name": "mcq-bias",
            "version": "0.1.0",
            "direct_url": {
                "kind": "vcs",
                "vcs": "git",
                "commit_id": "a" * 40,
                "url": "https://github.com/example/mcq-bias.git",
            },
        }
    ]
    assert first["runtime_sha256"] != second["runtime_sha256"]
    serialised = json.dumps(first)
    assert "secret-token" not in serialised
    assert "access_token" not in serialised
    assert "fragment" not in serialised
    assert verify_runtime_identity(first, actual=first)["runtime_sha256"] == first["runtime_sha256"]


def test_runtime_editable_local_direct_url_is_nonportable_without_local_path(monkeypatch) -> None:
    distribution = _FixtureDistribution(
        "mcq-bias",
        "0.1.0",
        {"url": "file:///private/example/mcq-bias", "dir_info": {"editable": True}},
    )
    monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: [distribution])

    runtime = capture_runtime_identity()

    assert runtime["distributions"][0]["direct_url"] == {
        "kind": "local",
        "editable": True,
        "portable": False,
    }
    assert "/private/example/mcq-bias" not in json.dumps(runtime)


def test_runtime_inventory_preserves_duplicate_distribution_versions_and_marks_ambiguity(monkeypatch) -> None:
    def capture_for(distributions: list[_FixtureDistribution]) -> dict:
        monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: distributions)
        return capture_runtime_identity()

    first = capture_for(
        [
            _FixtureDistribution("debugpy", "1.9.0"),
            _FixtureDistribution("alpha", "1.0.0"),
            _FixtureDistribution("DebugPy", "1.8.0"),
            _FixtureDistribution("debugpy", "1.8.0"),
        ]
    )
    changed = capture_for(
        [
            _FixtureDistribution("debugpy", "1.10.0"),
            _FixtureDistribution("alpha", "1.0.0"),
            _FixtureDistribution("DebugPy", "1.8.0"),
            _FixtureDistribution("debugpy", "1.9.0"),
        ]
    )

    assert first["distributions"] == [
        {"name": "alpha", "version": "1.0.0"},
        {"name": "debugpy", "version": "1.8.0"},
        {"name": "debugpy", "version": "1.9.0"},
    ]
    assert first["distribution_name_ambiguities"] == [
        {"name": "debugpy", "distinct_identity_count": 2}
    ]
    assert first["runtime_sha256"] != changed["runtime_sha256"]
    with pytest.raises(RuntimeProvenanceError, match="runtime identity mismatch"):
        verify_runtime_identity(first, actual=changed)


def test_dirty_and_untracked_source_restores_exact_bytes_modes_and_safe_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "pkg" / "runner.py", "print('base')\n", executable=True)
    _git(["init"], cwd=source)
    _git(["config", "user.email", "tests@example.invalid"], cwd=source)
    _git(["config", "user.name", "CTM tests"], cwd=source)
    _git(["add", "pkg/runner.py"], cwd=source)
    _git(["commit", "-m", "base"], cwd=source)

    _write(source / "pkg" / "runner.py", "print('dirty executable')\n", executable=True)
    _write(source / "pkg" / "new_module.py", "VALUE = 'untracked'\n")
    os.symlink("../pkg/runner.py", source / "pkg" / "runner_link.py")

    snapshot = _capture(source, tmp_path / "run")
    assert snapshot["git"]["available"] is True
    assert snapshot["git"]["dirty"] is True
    assert snapshot["scope"]["status"] == "complete_for_declared_scope"
    runner = next(entry for entry in snapshot["entries"] if entry["path"] == "pkg/runner.py")
    assert runner["mode"] & 0o111
    assert {entry["path"] for entry in snapshot["entries"]} >= {
        "pkg",
        "pkg/new_module.py",
        "pkg/runner.py",
        "pkg/runner_link.py",
    }

    restored = restore_source_snapshot(snapshot, tmp_path / "restored", bundle_root=tmp_path / "run")
    assert (restored / "pkg" / "runner.py").read_text(encoding="utf-8") == "print('dirty executable')\n"
    assert (restored / "pkg" / "new_module.py").read_text(encoding="utf-8") == "VALUE = 'untracked'\n"
    assert (restored / "pkg" / "runner_link.py").is_symlink()
    assert os.readlink(restored / "pkg" / "runner_link.py") == "../pkg/runner.py"
    assert (restored / "pkg" / "runner.py").stat().st_mode & 0o111
    verify_source_snapshot(snapshot, bundle_root=tmp_path / "run", source_root=restored)


def test_source_export_without_git_is_complete_for_its_explicit_scope(tmp_path: Path) -> None:
    source = tmp_path / "export"
    _write(source / "pyproject.toml", "[project]\nname = 'example'\n")
    _write(source / "src" / "example.py", "ANSWER = 42\n")

    snapshot = _capture(source, tmp_path / "run")
    assert snapshot["git"] == {"available": False, "reason": "no_git_metadata_in_source_root"}
    assert snapshot["source_sha256"]
    repeated = _capture(source, tmp_path / "repeat-run")
    assert repeated["source_sha256"] == snapshot["source_sha256"]
    assert (tmp_path / "run" / snapshot["bundle"]["path"]).read_bytes() == (
        tmp_path / "repeat-run" / repeated["bundle"]["path"]
    ).read_bytes()
    restored = restore_source_snapshot(snapshot, tmp_path / "restored", bundle_root=tmp_path / "run")
    assert (restored / "pyproject.toml").is_file()
    assert (restored / "src" / "example.py").read_text(encoding="utf-8") == "ANSWER = 42\n"


def test_credentials_outputs_and_archived_environments_are_excluded_and_reported(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write(source / "pyproject.toml", "[project]\nname = 'example'\n")
    _write(source / "requirements.txt", "pytest==8.0\n")
    _write(source / "active.py", "ACTIVE = True\n")
    _write(source / ".env", "TOKEN=must-not-enter-the-bundle\n")
    _write(source / "credentials.json", "{\"token\": \"must-not-enter-the-bundle\"}\n")
    _write(source / "secrets" / "provider.txt", "must-not-enter-the-bundle\n")
    _write(source / "weights.pt", "model-bytes")
    _write(source / "_archive" / "environments" / "old" / "site-packages" / "package.py", "OLD = True\n")
    _write(source / "environments" / "old" / "site-packages" / "package.py", "OLD = True\n")
    _write(source / ".venv-muse-cu129" / "pyvenv.cfg", "home = /python\n")
    _write(source / ".venv-muse-cu129" / "bin" / "python", "interpreter metadata\n")
    _write(source / ".venv-muse-cu129" / "root-metadata.json", "must-not-enter-the-bundle\n")
    _write(source / ".venv-offline" / "bin" / "activate", "activation metadata\n")
    _write(source / "copied-runtime" / "pyvenv.cfg", "home = /python\n")
    _write(source / "copied-runtime" / "bin" / "python", "interpreter metadata\n")
    _write(source / "copied-runtime" / "root-metadata.json", "must-not-enter-the-bundle\n")
    _write(source / "environments" / "old" / "sitepackages" / "package.py", "OLD = True\n")

    snapshot = _capture(source, tmp_path / "run")
    included = {entry["path"] for entry in snapshot["entries"]}
    assert {"pyproject.toml", "requirements.txt", "active.py"} <= included
    assert ".env" not in included
    assert "credentials.json" not in included
    assert not any(path.startswith("secrets/") for path in included)
    assert "weights.pt" not in included
    assert not any(path.startswith("_archive/") for path in included)
    assert not any(path.startswith("environments/old/site-packages/") for path in included)
    assert not any(path == ".venv-muse-cu129" or path.startswith(".venv-muse-cu129/") for path in included)
    assert not any(path == ".venv-offline" or path.startswith(".venv-offline/") for path in included)
    assert not any(path == "copied-runtime" or path.startswith("copied-runtime/") for path in included)
    assert not any(path.startswith("environments/old/sitepackages/") for path in included)
    exclusions = {(entry["path"], entry["reason"]) for entry in snapshot["exclusions"]}
    assert (".env", "environment_or_credentials") in exclusions
    assert ("credentials.json", "credentials") in exclusions
    assert ("secrets", "credentials") in exclusions
    assert ("weights.pt", "model_or_checkpoint_output") in exclusions
    assert ("_archive", "archived_material") in exclusions
    assert ("environments/old/site-packages", "environment") in exclusions
    assert (".venv-muse-cu129", "environment") in exclusions
    assert (".venv-offline", "environment") in exclusions
    assert ("copied-runtime", "environment") in exclusions
    assert ("environments/old/sitepackages", "environment") in exclusions
    bundle = tmp_path / "run" / snapshot["bundle"]["path"]
    assert b"must-not-enter-the-bundle" not in bundle.read_bytes()


def test_unsafe_symlink_and_bundle_tampering_fail_closed(tmp_path: Path) -> None:
    outside_secret = tmp_path / "outside-secret.txt"
    outside_secret.write_text("must-not-read", encoding="utf-8")
    unsafe_source = tmp_path / "unsafe-source"
    _write(unsafe_source / "active.py", "ACTIVE = True\n")
    os.symlink("../outside-secret.txt", unsafe_source / "escape")
    with pytest.raises(SourceSnapshotError, match="escapes the source root"):
        _capture(unsafe_source, tmp_path / "unsafe-run")

    source = tmp_path / "source"
    _write(source / "active.py", "ACTIVE = True\n")
    snapshot = _capture(source, tmp_path / "run")
    bundle = tmp_path / "run" / snapshot["bundle"]["path"]
    with bundle.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(SourceSnapshotError, match="digest mismatch"):
        verify_source_snapshot(snapshot, bundle_root=tmp_path / "run")


def test_failed_source_bundle_publish_archives_its_partial_file(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    _write(source / "active.py", "ACTIVE = True\n")
    run = tmp_path / "run"

    def fail_bundle_link(_source_path: str | Path, _bundle_path: str | Path) -> None:
        raise OSError("injected source-bundle link failure")

    monkeypatch.setattr(provenance.os, "link", fail_bundle_link)
    with pytest.raises(SourceSnapshotError, match="cannot publish source bundle"):
        _capture(source, run)

    partials = list((run / "provenance" / "source" / "_archive" / "failed-provenance-writes").glob("*.partial.*"))
    assert len(partials) == 1
    assert partials[0].read_bytes()
