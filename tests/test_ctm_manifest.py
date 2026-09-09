"""Tests for immutable training-run provenance manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ctm.training.manifest as manifest_module
from ctm.identity import sha256_bytes
from ctm.provenance import SourceSnapshotError
from ctm.training.manifest import (
    ATTEMPT_MANIFEST_DIRECTORY,
    RunManifestError,
    config_hash,
    read_attempt_manifest,
    read_run_manifest,
    verify_run_manifest,
    write_run_manifest,
)


class DummyBackend:
    pass


def _source_root(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "training.py").write_text("print('training')\n", encoding="utf-8")
    (source / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    return source


class TestRunManifest:
    def test_default_source_root_is_not_derived_from_ambient_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        run = tmp_path / "run"
        path = write_run_manifest(
            run,
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={"seed": 1},
            attempt_id="default-root",
        )
        manifest = read_run_manifest(run)
        assert manifest is not None
        assert any(entry["path"] == "ctm/provenance.py" for entry in manifest["source_snapshot"]["entries"])
        assert verify_run_manifest(path)["attempt_id"] == "default-root"

    def test_write_and_read_current_manifest_with_retained_source_bundle(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        config = {"experiment_name": "e", "run_name": "r", "model": "some/model"}
        path = write_run_manifest(
            tmp_path / "run",
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump=config,
            extra={"n_datapoints": 12},
            source_root=source,
            attempt_id="first-attempt",
        )

        assert path.name == "manifest.json"
        manifest = read_run_manifest(tmp_path / "run")
        assert manifest is not None
        assert manifest["kind"] == "rl"
        assert manifest["model"] == "some/model"
        assert manifest["backend"] == "DummyBackend"
        assert manifest["n_datapoints"] == 12
        assert manifest["config"]["experiment_name"] == "e"
        assert len(manifest["config_hash"]) == 16
        assert "git_sha" in manifest["git"] or "git_error" in manifest["git"]
        assert "git_diff" not in manifest["git"]
        assert manifest["source_identity"] == manifest["source_snapshot"]["source_sha256"]
        assert manifest["environment_identity"] == manifest["environment"]["runtime_sha256"]
        assert (tmp_path / "run" / manifest["source_snapshot"]["bundle"]["path"]).is_file()
        immutable = tmp_path / "run" / ATTEMPT_MANIFEST_DIRECTORY / "first-attempt.json"
        assert immutable.read_bytes() == path.read_bytes()
        assert read_attempt_manifest(tmp_path / "run", "first-attempt") == manifest
        assert verify_run_manifest(path, source_root=source)["attempt_id"] == "first-attempt"

    def test_input_identity_changes_and_attempt_records_are_immutable(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        run = tmp_path / "run"
        write_run_manifest(
            run,
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={"seed": 1},
            source_root=source,
            attempt_id="attempt-one",
        )
        first_immutable = run / ATTEMPT_MANIFEST_DIRECTORY / "attempt-one.json"
        first_bytes = first_immutable.read_bytes()
        first_manifest = read_run_manifest(run)
        assert first_manifest is not None

        second = write_run_manifest(
            run,
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={"seed": 2},
            source_root=source,
            attempt_id="attempt-two",
        )
        second_manifest = read_run_manifest(run)
        assert second_manifest is not None
        assert first_manifest["input_identity"] != second_manifest["input_identity"]
        assert first_immutable.read_bytes() == first_bytes
        assert second.read_bytes() == (run / ATTEMPT_MANIFEST_DIRECTORY / "attempt-two.json").read_bytes()
        assert read_attempt_manifest(run, "attempt-one") == first_manifest
        with pytest.raises(FileExistsError, match="immutable training attempt manifest"):
            write_run_manifest(
                run,
                kind="rl",
                model="some/model",
                backend=DummyBackend(),
                config_dump={"seed": 1},
                source_root=source,
                attempt_id="attempt-one",
            )

    def test_legacy_current_manifest_bytes_survive_migration_and_retry(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        run = tmp_path / "run"
        run.mkdir()
        legacy_bytes = b'{\r\n "kind": "rl",\r\n "model": "old/model"\r\n}\r\n'
        (run / "manifest.json").write_bytes(legacy_bytes)
        preserved = run / ATTEMPT_MANIFEST_DIRECTORY / f"legacy-{sha256_bytes(legacy_bytes)}.json"

        write_run_manifest(
            run,
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={"seed": 1},
            source_root=source,
            attempt_id="migrated-attempt",
        )
        first_current = (run / "manifest.json").read_bytes()
        assert preserved.read_bytes() == legacy_bytes
        assert b"manifest_schema" not in preserved.read_bytes()
        assert first_current == (run / ATTEMPT_MANIFEST_DIRECTORY / "migrated-attempt.json").read_bytes()

        write_run_manifest(
            run,
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={"seed": 2},
            source_root=source,
            attempt_id="retry-attempt",
        )
        assert preserved.read_bytes() == legacy_bytes
        assert list((run / ATTEMPT_MANIFEST_DIRECTORY).glob("legacy-*.json")) == [preserved]
        assert (run / "manifest.json").read_bytes() == (run / ATTEMPT_MANIFEST_DIRECTORY / "retry-attempt.json").read_bytes()

    def test_current_manifest_symlink_is_rejected_without_reading_its_target(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        run = tmp_path / "run"
        run.mkdir()
        target = tmp_path / "outside-manifest.json"
        target_bytes = b'{"legacy": "outside"}\n'
        target.write_bytes(target_bytes)
        os.symlink(target, run / "manifest.json")

        with pytest.raises(RunManifestError, match="must not be a symlink"):
            write_run_manifest(
                run,
                kind="rl",
                model="some/model",
                backend=DummyBackend(),
                config_dump={"seed": 1},
                source_root=source,
                attempt_id="symlinked-current",
            )

        assert target.read_bytes() == target_bytes
        assert not (run / ATTEMPT_MANIFEST_DIRECTORY).exists()

    def test_failed_current_manifest_write_archives_its_partial_file(self, tmp_path: Path, monkeypatch) -> None:
        source = _source_root(tmp_path)
        run = tmp_path / "run"
        current = run / "manifest.json"
        original_replace = manifest_module.os.replace

        def fail_only_current_replace(source_path: str | Path, destination: str | Path) -> None:
            if Path(destination) == current:
                raise OSError("injected current-manifest replace failure")
            original_replace(source_path, destination)

        monkeypatch.setattr(manifest_module.os, "replace", fail_only_current_replace)
        with pytest.raises(RunManifestError, match="partial retained at"):
            write_run_manifest(
                run,
                kind="rl",
                model="some/model",
                backend=DummyBackend(),
                config_dump={"seed": 1},
                source_root=source,
                attempt_id="current-write-failure",
            )

        partials = list((run / "_archive" / "failed-provenance-writes").glob("*.partial.*"))
        assert len(partials) == 1
        assert partials[0].read_bytes().startswith(b"{")
        assert not current.exists()

    def test_training_manifest_recursively_redacts_secrets(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        write_run_manifest(
            tmp_path / "run",
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={
                "run_metadata": {
                    "grader": {"api_key": "must-not-persist"},
                    "generation": {"extra_headers": {"Authorization": "must-not-persist"}},
                }
            },
            extra={"auth_token": "must-not-persist"},
            source_root=source,
            attempt_id="redacted",
        )
        manifest = read_run_manifest(tmp_path / "run")
        assert manifest is not None
        assert manifest["config"]["run_metadata"]["grader"]["api_key"] == "<redacted>"
        assert manifest["config"]["run_metadata"]["generation"]["extra_headers"] == "<redacted>"
        assert manifest["auth_token"] == "<redacted>"

    def test_manifest_verification_detects_tampering(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        path = write_run_manifest(
            tmp_path / "run",
            kind="rl",
            model="some/model",
            backend=DummyBackend(),
            config_dump={"seed": 1},
            source_root=source,
            attempt_id="tamper-target",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["model"] = "changed/model"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with pytest.raises(RunManifestError, match="input disagrees|integrity digest"):
            verify_run_manifest(path)

    def test_failed_source_capture_creates_no_valid_manifest(self, tmp_path: Path) -> None:
        source = _source_root(tmp_path)
        secret = tmp_path / "secret.txt"
        secret.write_text("not source", encoding="utf-8")
        os.symlink("../secret.txt", source / "unsafe-link")
        run = tmp_path / "run"
        with pytest.raises(SourceSnapshotError, match="escapes the source root"):
            write_run_manifest(
                run,
                kind="rl",
                model="some/model",
                backend=DummyBackend(),
                config_dump={"seed": 1},
                source_root=source,
                attempt_id="failed",
            )
        assert not (run / "manifest.json").exists()
        assert not (run / ATTEMPT_MANIFEST_DIRECTORY).exists()

    def test_hash_stable_and_config_sensitive(self) -> None:
        a = {"experiment_name": "e", "run_name": "r", "optimizer": {"lr": 0.0001}}
        b = {"optimizer": {"lr": 0.0001}, "run_name": "r", "experiment_name": "e"}
        c = {"experiment_name": "e", "run_name": "r", "optimizer": {"lr": 0.42}}
        assert config_hash(a) == config_hash(b)
        assert config_hash(a) != config_hash(c)

    def test_hash_retains_historical_json_byte_contract_for_unicode_and_whitespace(self) -> None:
        config = {
            "z": "  spaced\nsnowman ☃  ",
            "alpha": {"line": " x \t", "list": ["é", "  "]},
        }
        assert config_hash(config) == "e42c8b393f506896"

    def test_read_missing_returns_none(self, tmp_path: Path) -> None:
        assert read_run_manifest(tmp_path) is None
        assert read_attempt_manifest(tmp_path, "absent") is None
