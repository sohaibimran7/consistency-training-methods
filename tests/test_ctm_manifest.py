"""Tests for training-run provenance manifests (ctm.training.manifest)."""

from ctm.training.manifest import config_hash, read_run_manifest, write_run_manifest
from ctm.training.rl import RLConfig


class DummyBackend:
    pass


class TestRunManifest:
    def test_write_and_read(self, tmp_path):
        cfg = RLConfig(experiment_name="e", run_name="r", model="some/model")
        path = write_run_manifest(
            tmp_path,
            kind="rl",
            model=cfg.model,
            backend=DummyBackend(),
            config_dump=cfg.model_dump(),
            extra={"n_datapoints": 12},
        )
        assert path.name == "manifest.json"
        m = read_run_manifest(tmp_path)
        assert m["kind"] == "rl"
        assert m["model"] == "some/model"
        assert m["backend"] == "DummyBackend"
        assert m["n_datapoints"] == 12
        assert m["config"]["experiment_name"] == "e"
        assert len(m["config_hash"]) == 16
        assert "git_sha" in m["git"] or "git_error" in m["git"]
        assert "git_diff" not in m["git"]  # kept small; the diff goes to WandB

    def test_hash_stable_and_config_sensitive(self):
        a = RLConfig(experiment_name="e", run_name="r").model_dump()
        b = RLConfig(experiment_name="e", run_name="r").model_dump()
        c = RLConfig(experiment_name="e", run_name="r", kl_coef=0.42).model_dump()
        assert config_hash(a) == config_hash(b)
        assert config_hash(a) != config_hash(c)

    def test_training_manifest_recursively_redacts_secrets(self, tmp_path):
        write_run_manifest(
            tmp_path,
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
        )
        manifest = read_run_manifest(tmp_path)
        assert manifest["config"]["run_metadata"]["grader"]["api_key"] == "<redacted>"
        assert manifest["config"]["run_metadata"]["generation"]["extra_headers"] == "<redacted>"
        assert manifest["auth_token"] == "<redacted>"

    def test_read_missing_returns_none(self, tmp_path):
        assert read_run_manifest(tmp_path) is None
