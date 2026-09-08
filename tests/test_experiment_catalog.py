"""Offline tests for the validated experiment catalogue API and CLI."""

from __future__ import annotations

import copy
import json

import pytest

from ctm.experiments.catalog import (
    CatalogValidationError,
    catalog_json_schema,
    find_experiment,
    render_catalog_markdown,
    validate_catalog,
)
from scripts.experiment_catalog import main


def _repo_location(path: str, *, availability: str = "available") -> dict[str, str]:
    location = {"kind": "repo_path", "path": path, "availability": availability}
    if availability != "available":
        location["availability_note"] = "recorded from a historical checkout"
    return location


def _external_location(uri: str, *, availability: str = "available") -> dict[str, str]:
    location = {"kind": "external", "uri": uri, "availability": availability}
    if availability != "available":
        location["availability_note"] = "historical storage is no longer reachable"
    return location


def _reference(role: str, location: dict[str, str], **extra: str) -> dict[str, object]:
    return {"role": role, "location": location, **extra}


def _experiment(identifier: str, *, status: str = "complete") -> dict[str, object]:
    return {
        "id": identifier,
        "question": "Does the intervention improve the pre-registered metric?",
        "status": status,
        "protocol_refs": [_reference("canonical_plan", _repo_location("experiments/example/plan.py"))],
        "source_snapshots": [
            _reference(
                "source_tree",
                _external_location("ssh://isambard.example/projects/example/repo", availability="unreachable"),
                revision="abc123",
            )
        ],
        "environment_refs": [_reference("requirements", _repo_location("requirements.txt"))],
        "artifacts": [
            _reference(
                "accepted_result",
                _external_location("s3://ctm-results/example/summary.json", availability="missing"),
                sha256="a" * 64,
            )
        ],
        "lineage": {"supersedes": [], "derived_from": []},
        "owner": "ctm-research",
        "current_task_refs": [_reference("issue", _external_location("https://example.test/tasks/42"))],
        "completion_criteria": ["Publish a paired aggregate with the registered denominator."],
        "evidence_notes": ["The signed run record and review note accepted this result."],
    }


def _catalog(*experiments: dict[str, object]) -> dict[str, object]:
    return {"schema_version": 1, "experiments": list(experiments)}


def test_valid_catalog_can_be_found_and_rendered_without_resolving_locations():
    catalog = _catalog(_experiment("rmct.example"))

    validated = validate_catalog(catalog)
    assert find_experiment(validated, "rmct.example").owner == "ctm-research"

    rendered = render_catalog_markdown(validated)
    assert "# Experiment catalogue" in rendered
    assert "`experiments/example/plan.py` (repository path; recorded availability: available)" in rendered
    assert "`requirements.txt` (repository path; recorded availability: available)" in rendered
    assert "ssh://isambard.example/projects/example/repo" in rendered
    assert "recorded availability: unreachable" in rendered
    assert "recorded availability: missing" in rendered
    assert "**Recorded status:** `complete`" in rendered


def test_json_schema_is_generated_from_the_validation_model():
    schema = catalog_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == 1
    assert "ExperimentRecord" in schema["$defs"]


def test_cli_show_finds_an_id_and_prints_exact_recorded_locations(tmp_path, capsys):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog(_experiment("rmct.example"))), encoding="utf-8")

    assert main(["show", str(catalog_path), "rmct.example"]) == 0
    output = capsys.readouterr().out
    assert "# `rmct.example`" in output
    assert "`experiments/example/plan.py`" in output
    assert "`s3://ctm-results/example/summary.json`" in output
    assert "historical storage is no longer reachable" in output


def test_cli_validate_list_and_render_are_offline(tmp_path, capsys):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog(_experiment("rmct.example"))), encoding="utf-8")

    assert main(["validate", str(catalog_path)]) == 0
    assert "Valid catalogue: 1 experiment(s)." in capsys.readouterr().out
    assert main(["list", str(catalog_path)]) == 0
    assert capsys.readouterr().out.startswith("rmct.example\tcomplete\t")
    assert main(["render", str(catalog_path)]) == 0
    assert "# Experiment catalogue" in capsys.readouterr().out


def test_duplicate_ids_are_rejected():
    with pytest.raises(CatalogValidationError, match="duplicate id"):
        validate_catalog(_catalog(_experiment("duplicate"), _experiment("duplicate")))


def test_bad_lineage_is_rejected_for_unknown_ids_and_cycles():
    unknown = _experiment("newer", status="active")
    unknown["lineage"] = {"supersedes": ["not-recorded"], "derived_from": []}
    with pytest.raises(CatalogValidationError, match="unknown experiment"):
        validate_catalog(_catalog(unknown))

    first = _experiment("first", status="active")
    second = _experiment("second", status="active")
    first["lineage"] = {"supersedes": [], "derived_from": ["second"]}
    second["lineage"] = {"supersedes": [], "derived_from": ["first"]}
    with pytest.raises(CatalogValidationError, match="lineage contains a cycle"):
        validate_catalog(_catalog(first, second))


def test_complete_requires_explicit_result_and_evidence_not_path_presence():
    no_result = _experiment("no-result")
    no_result["artifacts"] = [_reference("checkpoint", _repo_location("artifacts/checkpoint"))]
    with pytest.raises(CatalogValidationError, match="artifact role"):
        validate_catalog(_catalog(no_result))

    no_evidence = _experiment("no-evidence")
    no_evidence["evidence_notes"] = []
    with pytest.raises(CatalogValidationError, match="evidence_notes must not be empty"):
        validate_catalog(_catalog(no_evidence))

    recorded_missing = _experiment("historical-complete")
    recorded_missing["artifacts"] = [
        _reference(
            "result",
            _external_location("s3://ctm-results/historical/summary.json", availability="missing"),
        )
    ]
    validate_catalog(_catalog(recorded_missing))


def test_locations_must_distinguish_repo_paths_from_explicit_external_uris():
    absolute_repo_path = _experiment("bad-repo-path")
    absolute_repo_path["protocol_refs"] = [_reference("plan", _repo_location("/tmp/plan.py"))]
    with pytest.raises(CatalogValidationError, match="relative POSIX repository path"):
        validate_catalog(_catalog(absolute_repo_path))

    implicit_external_path = copy.deepcopy(_experiment("bad-external-uri"))
    implicit_external_path["artifacts"] = [
        _reference(
            "accepted_result",
            {"kind": "external", "uri": "/projects/remote/result.json", "availability": "available"},
        )
    ]
    with pytest.raises(CatalogValidationError, match="explicit URI"):
        validate_catalog(_catalog(implicit_external_path))
