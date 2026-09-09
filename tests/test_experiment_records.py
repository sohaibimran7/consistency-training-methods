"""Focused coverage for immutable generic experiment lifecycle records."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ctm.experiments import records


def _attempt(tmp_path: Path, *, name: str = "unit") -> records.ExperimentAttempt:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text(f"name: {name}\nanalysis:\n  command: [python, task.py]\n", encoding="utf-8")
    storage = tmp_path / "records"
    resolved_plan = storage / "resolved-plan.yaml"
    resolved_plan.parent.mkdir(exist_ok=True)
    resolved_plan.write_text(f"name: {name}\nanalysis:\n  command: [python, task.py]\n", encoding="utf-8")
    return records.create_attempt_record(
        storage / "attempts",
        experiment_name=name,
        runner_argv=["scripts/run_experiment.py", str(config), "--yes"],
        source_yaml=config,
        resolved_plan_text=f"name: {name}\nanalysis:\n  command: [python, task.py]\n",
        resolved_plan_path=resolved_plan,
        source_root=source,
        bundle_root=storage,
        bundle_directory=storage / "source-bundles",
        working_directory=tmp_path,
        selected_stages=["analysis"],
    )


def test_successful_subprocess_records_exact_argv_source_runtime_and_lineage(tmp_path):
    attempt = _attempt(tmp_path)
    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"id": 1}\n', encoding="utf-8")
    output_path = tmp_path / "output.txt"
    argv = [
        sys.executable,
        "-c",
        "from pathlib import Path; Path('output.txt').write_text('done', encoding='utf-8')",
        "--data-manifest",
        str(input_path),
    ]
    command = records.start_command_record(
        attempt,
        stage="analysis",
        name="write-output",
        argv=argv,
        cuda_visible_devices=["2"],
        declared_inputs=[input_path],
        declared_outputs=[output_path],
        working_directory=tmp_path,
    )
    result = subprocess.run(argv, cwd=tmp_path, check=False)
    assert result.returncode == 0
    records.complete_command_record(command, outcome="succeeded", return_code=result.returncode)
    records.complete_attempt_record(attempt, outcome="succeeded")

    verified_attempt = records.verify_attempt_record(
        attempt,
        source_root=tmp_path / "source",
        allowed_path_roots=[tmp_path],
    )
    verified_command = records.verify_command_record(command, allowed_path_roots=[tmp_path])
    document = verified_attempt["attempt"]
    start = verified_command["command"]
    terminal = verified_command["lifecycle"]["event"]

    assert verified_attempt["lifecycle"]["status"] == "succeeded"
    assert verified_command["lifecycle"]["status"] == "succeeded"
    assert document["runner"]["runtime_identity_scope"] == (
        "parent_runner_process_only; command child environments are not inferred or claimed by this record"
    )
    assert document["plan"]["source_yaml"]["content_sha256"]
    assert document["provenance"]["source_snapshot"]["bundle"]["path"].startswith("source-bundles/")
    assert (attempt.bundle_root / document["provenance"]["source_snapshot"]["bundle"]["path"]).is_file()
    assert start["argv"] == argv
    assert start["cuda_placement"] == {"mode": "runner_assigned", "visible_devices": ["2"]}
    assert start["input_lineage"]["inferred_inputs"]
    assert terminal["output_lineage"]["declared_outputs"][0]["content_sha256"]


def test_failure_and_missing_terminal_remain_distinct(tmp_path):
    attempt = _attempt(tmp_path)
    incomplete = records.start_command_record(
        attempt,
        stage="analysis",
        name="never-finished",
        argv=[sys.executable, "-c", "pass"],
        working_directory=tmp_path,
    )
    assert records.verify_command_record(incomplete)["lifecycle"]["status"] == "incomplete"

    failing = records.start_command_record(
        attempt,
        stage="analysis",
        name="fails",
        argv=[sys.executable, "-c", "import sys; sys.exit(7)"],
        working_directory=tmp_path,
    )
    result = subprocess.run([sys.executable, "-c", "import sys; sys.exit(7)"], cwd=tmp_path, check=False)
    assert result.returncode == 7
    records.complete_command_record(failing, outcome="failed", return_code=result.returncode)
    records.complete_attempt_record(attempt, outcome="failed")

    assert records.verify_command_record(failing)["lifecycle"]["status"] == "failed"
    assert records.verify_attempt_record(attempt)["lifecycle"]["status"] == "failed"
    with pytest.raises(records.ExperimentRecordError, match="immutable terminal event"):
        records.complete_command_record(failing, outcome="succeeded", return_code=0)


def test_retries_are_separate_immutable_attempts(tmp_path):
    first = _attempt(tmp_path, name="retry")
    before = (first.directory / "attempt.json").read_bytes()
    second = _attempt(tmp_path, name="retry")

    assert first.attempt_id != second.attempt_id
    assert first.directory != second.directory
    assert (first.directory / "attempt.json").read_bytes() == before
    assert records.list_attempt_records(tmp_path / "records" / "attempts") == sorted(
        [first.directory, second.directory], key=lambda path: path.name
    )


def test_attempt_retains_yaml_and_resolved_plan_bytes_across_retry_and_original_mutation(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
    external_config = tmp_path / "external-config" / "experiment.yaml"
    external_config.parent.mkdir()
    storage = tmp_path / "records"
    shared_plan = storage / "resolved-plan.yaml"
    storage.mkdir()
    original_yaml = b"name: retained-first\nanalysis:\n  command: [python, first.py]\n"
    original_plan = "name: retained-first\nanalysis:\n  command: [python, first.py]\n"
    external_config.write_bytes(original_yaml)
    shared_plan.write_text(original_plan, encoding="utf-8")

    first = records.create_attempt_record(
        storage / "attempts",
        experiment_name="retained-first",
        runner_argv=["scripts/run_experiment.py", str(external_config), "--yes"],
        source_yaml=external_config,
        resolved_plan_text=original_plan,
        resolved_plan_path=shared_plan,
        source_root=source_root,
        bundle_root=storage,
        bundle_directory=storage / "source-bundles",
        working_directory=tmp_path,
    )
    first_document = json.loads((first.directory / "attempt.json").read_text(encoding="utf-8"))
    retained_yaml = first.directory / first_document["plan"]["source_yaml"]["retained_copy"]["path"]
    retained_plan = first.directory / first_document["plan"]["resolved_selected_plan"]["retained_copy"]["path"]
    assert retained_yaml.read_bytes() == original_yaml
    assert retained_plan.read_bytes() == original_plan.encode("utf-8")

    retry_yaml = b"name: retained-retry\nanalysis:\n  command: [python, retry.py]\n"
    retry_plan = "name: retained-retry\nanalysis:\n  command: [python, retry.py]\n"
    external_config.write_bytes(retry_yaml)
    shared_plan.write_text(retry_plan, encoding="utf-8")
    second = records.create_attempt_record(
        storage / "attempts",
        experiment_name="retained-retry",
        runner_argv=["scripts/run_experiment.py", str(external_config), "--yes"],
        source_yaml=external_config,
        resolved_plan_text=retry_plan,
        resolved_plan_path=shared_plan,
        source_root=source_root,
        bundle_root=storage,
        bundle_directory=storage / "source-bundles",
        working_directory=tmp_path,
    )
    assert second.directory != first.directory

    external_config.unlink()
    assert records.verify_attempt_record(first, verify_runtime=False)["lifecycle"]["status"] == "incomplete"
    assert retained_yaml.read_bytes() == original_yaml
    assert retained_plan.read_bytes() == original_plan.encode("utf-8")

    # The retained bytes can restore an external authored config and mutable
    # shared plan; only this explicit allowed-root mode rechecks those paths.
    external_config.write_bytes(retained_yaml.read_bytes())
    shared_plan.write_bytes(retained_plan.read_bytes())
    records.verify_attempt_record(first, verify_runtime=False, allowed_path_roots=[tmp_path])

    retained_plan.write_bytes(b"forged resolved plan\n")
    with pytest.raises(records.ExperimentRecordError, match="retained resolved selected plan copy bytes"):
        records.verify_attempt_record(first, verify_runtime=False)


def test_attempt_rejects_symlinked_source_yaml(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "package.py").write_text("VALUE = 1\n", encoding="utf-8")
    actual_config = tmp_path / "actual.yaml"
    actual_config.write_text("name: linked\n", encoding="utf-8")
    linked_config = tmp_path / "linked.yaml"
    linked_config.symlink_to(actual_config)

    with pytest.raises(records.ExperimentRecordError, match="source YAML must be a regular file without symlinks"):
        records.create_attempt_record(
            tmp_path / "records" / "attempts",
            experiment_name="linked",
            runner_argv=["scripts/run_experiment.py", str(linked_config), "--yes"],
            source_yaml=linked_config,
            resolved_plan_text="name: linked\n",
            source_root=source_root,
            bundle_root=tmp_path / "records",
            bundle_directory=tmp_path / "records" / "source-bundles",
            working_directory=tmp_path,
        )


def test_parallel_commands_have_independent_uuid_records_and_one_attempt_capture(tmp_path, monkeypatch):
    calls = 0
    original_capture = records.capture_source_snapshot

    def capture_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_capture(*args, **kwargs)

    monkeypatch.setattr(records, "capture_source_snapshot", capture_once)
    attempt = _attempt(tmp_path, name="parallel")

    def run_one(index: int) -> records.ExperimentCommand:
        output = tmp_path / f"output-{index}.txt"
        argv = [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path('output-{index}.txt').write_text('ok', encoding='utf-8')",
        ]
        command = records.start_command_record(
            attempt,
            stage="analysis",
            name=f"command-{index}",
            argv=argv,
            declared_outputs=[output],
            working_directory=tmp_path,
        )
        result = subprocess.run(argv, cwd=tmp_path, check=False)
        records.complete_command_record(command, outcome="succeeded", return_code=result.returncode)
        return command

    with ThreadPoolExecutor(max_workers=4) as executor:
        commands = list(executor.map(run_one, range(4)))
    records.complete_attempt_record(attempt, outcome="succeeded")

    assert calls == 1
    assert len({command.command_id for command in commands}) == 4
    assert len(list((attempt.directory / "commands").iterdir())) == 4
    assert all(records.verify_command_record(command, allowed_path_roots=[tmp_path])["lifecycle"]["status"] == "succeeded" for command in commands)


def test_verifier_detects_tampered_input_output_and_immutable_record(tmp_path):
    attempt = _attempt(tmp_path, name="tamper")
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    input_path.write_text("before", encoding="utf-8")
    output_path.write_text("after", encoding="utf-8")
    command = records.start_command_record(
        attempt,
        stage="analysis",
        name="tamperable",
        argv=[sys.executable, "-c", "pass", "--data-manifest", str(input_path)],
        declared_inputs=[input_path],
        declared_outputs=[output_path],
        working_directory=tmp_path,
    )
    records.complete_command_record(command, outcome="succeeded", return_code=0)
    input_path.write_text("changed", encoding="utf-8")
    with pytest.raises(records.ExperimentRecordError, match="captured local reference changed"):
        records.verify_command_record(command, allowed_path_roots=[tmp_path])

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("name: tamper\nanalysis: []\n", encoding="utf-8")
    with pytest.raises(records.ExperimentRecordError, match="captured local reference changed"):
        records.verify_attempt_record(attempt, verify_runtime=False, allowed_path_roots=[tmp_path])

    attempt_path = attempt.directory / "attempt.json"
    tampered = json.loads(attempt_path.read_text(encoding="utf-8"))
    tampered["experiment"]["name"] = "forged"
    attempt_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(records.ExperimentRecordError, match="record_sha256"):
        records.verify_attempt_record(attempt, verify_runtime=False)


def test_symlink_and_credential_uri_are_explicitly_unverified_without_secret_text(tmp_path):
    target = tmp_path / "target.txt"
    target.write_text("data", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    linked_directory_target = tmp_path / "linked-directory-target"
    linked_directory_target.mkdir()
    (linked_directory_target / "nested.txt").write_text("data", encoding="utf-8")
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(linked_directory_target, target_is_directory=True)

    link_identity = records.capture_path_identity(link, working_directory=tmp_path)
    nested_link_identity = records.capture_path_identity(linked_directory / "nested.txt", working_directory=tmp_path)
    uri_identity = records.capture_path_identity(
        "https://username:password@example.test/source?token=secret#fragment",
        working_directory=tmp_path,
    )

    assert link_identity["status"] == "unverified_symlink"
    assert nested_link_identity["status"] == "unverified_symlink_ancestor"
    assert uri_identity["status"] == "unverified_external_uri"
    serialized = json.dumps(uri_identity, sort_keys=True)
    assert "password" not in serialized
    assert "secret" not in serialized


def test_failed_immutable_publish_archives_partial_bytes(monkeypatch, tmp_path):
    def fail_link(*args, **kwargs):
        raise OSError("simulated link failure")

    monkeypatch.setattr(records.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated link failure"):
        records._publish_immutable_json(tmp_path / "record.json", {"value": "forensic bytes"})

    archived = list((tmp_path / "_archive" / "failed-experiment-record-writes").iterdir())
    assert len(archived) == 1
    assert archived[0].read_text(encoding="utf-8") == '{"value":"forensic bytes"}\n'
