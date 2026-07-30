from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from ctm.artifacts import ArtifactManifestError, artifact_manifest_path
from ctm_data.adapters.irpan_2510_27062.artifacts import (
    producer_identity,
    read_artifact,
    require_local_artifact,
    write_artifact,
)
from ctm_data.adapters.irpan_2510_27062.reconstruction import reconstruction_config, reconstruction_ledger
from ctm_data.adapters.irpan_2510_27062.schema import (
    RecordSchemaError,
    make_derived_record,
    make_source_record,
    validate_record,
)
from ctm_data.adapters.irpan_2510_27062.source_registry import SOURCES, require_source


def _source_row() -> dict:
    return make_source_record(
        record_type="source_prompt",
        source="harmbench",
        source_key="fixture-1",
        payload={"prompt": "Describe a fictional safety audit."},
        metadata={"fixture": True},
    )


def test_source_record_identity_is_normalized_and_content_bound() -> None:
    first = make_source_record(
        record_type="source_prompt",
        source="harmbench",
        source_key=" fixture-1 ",
        payload={"prompt": "Line one\r\nLine two"},
    )
    second = make_source_record(
        record_type="source_prompt",
        source="harmbench",
        source_key="fixture-1",
        payload={"prompt": "Line one\nLine two"},
    )
    assert first == second
    tampered = {**first, "payload": {"prompt": "Different"}}
    with pytest.raises(RecordSchemaError, match="digest mismatch"):
        validate_record(tampered)


def test_derived_record_requires_explicit_lineage() -> None:
    source = _source_row()
    derived = make_derived_record(
        record_type="wrapper_candidate",
        example_id=source["example_id"],
        source=source["source"],
        source_key=source["source_key"],
        payload={"clean_prompt": source["payload"]["prompt"], "wrapped_prompt": "Audit context: ..."},
        parent_hashes=[source["content_sha256"]],
    )
    assert validate_record(derived)["parent_hashes"] == [source["content_sha256"]]
    with pytest.raises(RecordSchemaError, match="at least one"):
        make_derived_record(
            record_type="wrapper_candidate",
            example_id=source["example_id"],
            source="harmbench",
            source_key="fixture-1",
            payload={},
            parent_hashes=[],
        )


def test_artifact_round_trip_parent_manifest_and_no_overwrite(tmp_path: Path) -> None:
    producer = producer_identity("foundation-test", __file__)
    source_path = tmp_path / "source.jsonl"
    source_manifest = write_artifact(
        source_path,
        [_source_row()],
        artifact_kind="normalized_source",
        role="training",
        producer=producer,
        config={"fixture": 1},
    )
    rows, verified = read_artifact(source_path, expected_kind="normalized_source")
    assert rows == [_source_row()]
    assert verified["content_sha256"] == source_manifest["content_sha256"]

    derived = make_derived_record(
        record_type="wrapper_candidate",
        example_id=rows[0]["example_id"],
        source="harmbench",
        source_key="fixture-1",
        payload={"clean_prompt": rows[0]["payload"]["prompt"], "wrapped_prompt": "Fictional audit wrapper."},
        parent_hashes=[rows[0]["content_sha256"]],
    )
    derived_path = tmp_path / "derived.jsonl"
    manifest = write_artifact(
        derived_path,
        [derived],
        artifact_kind="wrapper_candidates",
        role="training",
        producer=producer,
        config={"catalog": "fixture"},
        parent_artifacts=[source_path],
    )
    assert manifest["provenance"]["parent_artifacts"][0]["content_sha256"] == source_manifest["content_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_artifact(
            derived_path,
            [derived],
            artifact_kind="wrapper_candidates",
            role="training",
            producer=producer,
            config={},
        )


def test_artifact_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_artifact(
        path,
        [_source_row()],
        artifact_kind="normalized_source",
        role="training",
        producer=producer_identity("foundation-test", __file__),
        config={},
    )
    path.write_text(json.dumps(_source_row()) + "\n " + "\n", encoding="utf-8")
    with pytest.raises(ArtifactManifestError, match="digest mismatch"):
        read_artifact(path)


def test_source_registry_covers_every_paper_dataset_and_gate() -> None:
    assert set(SOURCES) == {
        "arc",
        "openbookqa",
        "bbh",
        "mmlu",
        "harmbench",
        "or_bench",
        "clearharm",
        "wildguardtest",
        "xstest",
        "wildjailbreak",
    }
    wildjailbreak = require_source("wildjailbreak")
    assert wildjailbreak.local_only
    assert "gated" in wildjailbreak.access
    assert len(wildjailbreak.revision or "") == 40
    assert "gated" in require_source("wildguardtest").access
    with pytest.raises(KeyError, match="unknown paper source"):
        require_source("not_a_source")


def test_missing_local_export_is_actionable_and_does_not_use_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fail_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    with pytest.raises(FileNotFoundError, match="never download"):
        require_local_artifact(
            tmp_path / "wildjailbreak.jsonl",
            source_key="wildjailbreak",
            acquisition_url=require_source("wildjailbreak").official_url,
        )


def test_reconstruction_ledger_keeps_facts_and_defaults_separate() -> None:
    config = reconstruction_config({"bootstrap_replicates": 25})
    assert config["bootstrap_replicates"] == 25
    ledger = reconstruction_ledger({"bootstrap_replicates": 25})
    assert ledger["reconstruction_choices"]["bootstrap_replicates"]["paper_status"] == "paper-unspecified"
    assert ledger["reconstruction_choices"]["bootstrap_replicates"]["selected"] == 25
    assert "jailbreak_final_sources" in ledger["paper_facts"]
    with pytest.raises(KeyError, match="unknown reconstruction"):
        reconstruction_config({"invented": True})


def test_manifest_sidecar_path_is_outside_payload() -> None:
    path = Path("example.jsonl")
    assert artifact_manifest_path(path).name == "example.jsonl.manifest.json"
