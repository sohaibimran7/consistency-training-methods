from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "infra/isambard/prepare_qwen35_rmct_r5_interactive_duplicate.py"


def _load_helper():
    name = "rmct_r5_interactive_duplicate_for_test"
    spec = importlib.util.spec_from_file_location(name, HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_helper_exposes_prepare_validate_status_smoke_and_one_window_run() -> None:
    helper = _load_helper()
    parser = helper._parser()
    source = Path("/remote/frozen-source")
    duplicate = Path("/remote/r5-interactive-copy")
    runtime = source / ".venv/bin/python"

    prepare = parser.parse_args(
        [
            "prepare",
            "--source-repository",
            str(source),
            "--duplicate-repository",
            str(duplicate),
            "--runtime-python",
            str(runtime),
        ]
    )
    assert prepare.command == "prepare"
    status = parser.parse_args(
        ["status", "--source-repository", str(source), "--duplicate-repository", str(duplicate)]
    )
    assert status.command == "status"
    run = parser.parse_args(
        [
            "run",
            "--source-repository",
            str(source),
            "--duplicate-repository",
            str(duplicate),
            "--runtime-python",
            str(runtime),
            "--model-snapshot",
            "/scratch/snapshot/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "--yes",
        ]
    )
    assert run.max_segments == 1

    with pytest.raises(helper.DuplicateError, match="one full r5 window"):
        helper.run_windows(
            source_repository=source,
            duplicate_repository=duplicate,
            runtime_python=runtime,
            model_snapshot="/scratch/snapshot/c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            max_segments=2,
            minimum_remaining_seconds=0,
            yes=True,
        )


def test_smoke_preserves_production_rollouts_and_removes_every_generation_cap(monkeypatch, tmp_path: Path) -> None:
    helper = _load_helper()
    snapshot = tmp_path / helper.r5.BASE_SNAPSHOT
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_segment_args(root, index, *, model_snapshot):
        seen["root"] = root
        seen["index"] = index
        seen["snapshot"] = model_snapshot
        return {
            "run_name": "production-name",
            "n_datapoints": 32,
            "load_config": {"n_datapoints": 32, "segment_index": index},
            "checkpoint_every": 16,
            "n_ref_rollouts": 96,
            "n_train_rollouts": 96,
            "n_consistency_rollouts": 96,
            "n_anchor_rollouts": 96,
            "local_phase_shared": True,
            "local_training_gpus": "all",
            "local_rollout_gpus": "all",
            "no_max_new_tokens": True,
        }

    monkeypatch.setattr(helper.r5, "segment_args", fake_segment_args)
    args = helper.smoke_args(tmp_path, snapshot=snapshot)

    assert seen["index"] == helper.r5.START_SEGMENT_INDEX
    assert args["run_name"] == helper._smoke_run_name()
    assert args["n_datapoints"] == 2
    assert args["load_config"] == {"n_datapoints": 2, "segment_index": helper.r5.START_SEGMENT_INDEX}
    assert args["checkpoint_every"] == 1
    assert args["setting_factory"] == helper.SMOKE_SETTING_FACTORY
    assert args["n_ref_rollouts"] == args["n_train_rollouts"] == 96
    assert args["n_consistency_rollouts"] == args["n_anchor_rollouts"] == 96
    assert args["no_max_new_tokens"] is True
    assert not {"max_new_tokens", "max_tokens", "max_output_tokens", "max_completion_tokens"}.intersection(args)


def test_one_batch_smoke_factory_delegates_to_verified_32_datum_parent_then_returns_two(monkeypatch) -> None:
    helper = _load_helper()
    calls: list[dict[str, object]] = []
    source_rows = [{"question_id": f"q{index}", "nested": {"index": index}} for index in range(32)]

    class Base:
        name = "shared-qid-two-bias"

        def load_datapoints(self, **kwargs):
            calls.append(kwargs)
            return source_rows

        def perturbations(self):
            return [lambda _: {"messages": [{"role": "user", "content": "x"}]}] * 3

        def training_perturbation_indices(self):
            return [1, 2]

        def trait_classifier(self):
            return lambda *_: 0.0

        def answer_parser(self):
            return lambda _: "A"

        def run_metadata(self):
            return {"base": True}

        def training_artifact_identity(self):
            return [{"artifact": "canonical"}]

    monkeypatch.setattr(helper, "create_shared_qid_two_bias_setting", lambda **kwargs: Base())
    setting = helper.create_one_batch_smoke_setting(data_path="data", manifest_path="manifest")
    rows = setting.load_datapoints(n_datapoints=2, segment_index=11)

    assert calls == [{"n_datapoints": helper.SEGMENT_DATAPOINTS, "segment_index": 11}]
    assert [row["question_id"] for row in rows] == ["q0", "q1"]
    rows[0]["nested"]["index"] = -1
    assert source_rows[0]["nested"]["index"] == 0
    smoke_metadata = setting.run_metadata()["interactive_one_batch_smoke"]
    assert smoke_metadata["parent_segment_datapoints"] == helper.SEGMENT_DATAPOINTS
    assert smoke_metadata["selected_datapoints"] == 2
    with pytest.raises(ValueError, match="n_datapoints=2"):
        setting.load_datapoints(n_datapoints=32, segment_index=11)


def test_snapshot_accepts_canonical_hf_blob_symlink_but_rejects_external_target(tmp_path: Path) -> None:
    helper = _load_helper()
    model_root = tmp_path / "hub" / "models--Qwen--Qwen3.5-9B"
    snapshot = model_root / "snapshots" / helper.r5.BASE_SNAPSHOT
    blobs = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "config-blob"
    blob.write_text("{}\n", encoding="utf-8")
    os.symlink("../../blobs/config-blob", snapshot / "config.json")
    assert helper._snapshot(snapshot) == snapshot

    external = tmp_path / "external-config.json"
    external.write_text("{}\n", encoding="utf-8")
    (snapshot / "config.json").unlink()
    os.symlink(str(external), snapshot / "config.json")
    with pytest.raises(helper.DuplicateError, match="outside its canonical HF cache"):
        helper._snapshot(snapshot)


def test_copy_regular_tree_copies_bytes_and_rejects_links(tmp_path: Path) -> None:
    helper = _load_helper()
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    (source / "nested" / "module.py").write_text("x = 1\n", encoding="utf-8")
    destination = tmp_path / "copy"
    copied = helper._copy_regular_tree(source, destination, label="test input")
    assert copied == {"nested/module.py": {"sha256": helper._sha256(destination / "nested/module.py"), "size_bytes": 6}}

    linked = tmp_path / "linked-source"
    linked.mkdir()
    os.symlink(source / "nested" / "module.py", linked / "link.py")
    with pytest.raises(helper.DuplicateError, match="linked or non-regular"):
        helper._copy_regular_tree(linked, tmp_path / "linked-copy", label="linked test input")


def test_static_contract_never_submits_or_cancels_and_uses_read_only_parity_resume() -> None:
    text = HELPER.read_text(encoding="utf-8")
    assert "subprocess.run(command" in text
    assert '"--resume"]' in text
    assert "fresh_worker_parity_probe_forbidden" in text
    assert "queue_chain_cancelled\": False" in text
    assert '"sbatch"' not in text
    assert "scancel" not in text
    assert "--partition=" not in text
    assert "--qos=" not in text
    assert "--reservation=interactive" in text
    assert "--no-max-new-tokens" in text
