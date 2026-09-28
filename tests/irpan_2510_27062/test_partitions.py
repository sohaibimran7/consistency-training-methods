from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctm.artifacts import ArtifactManifestError, artifact_manifest_path
from scripts.irpan_2510_27062.artifacts import producer_identity, read_artifact, write_artifact
from scripts.irpan_2510_27062.partitions import (
    FINAL_EVAL,
    HARM_BENCH_PARTITION_NAMESPACE,
    HARM_BENCH_PARTITION_RULE,
    HARM_BENCH_PARTITION_SEED,
    PAPER_UNSPECIFIED_RECONSTRUCTION,
    PARTITION_REGISTRY,
    TRAINING,
    VALIDATION,
    PartitionError,
    assign_harmbench_partition,
    harmbench_partition_provenance,
    partition_harmbench_ids,
    require_partition,
    stable_source_identity,
    verify_disjoint_ids,
)
from scripts.irpan_2510_27062.schema import make_source_record


def _row() -> dict:
    return make_source_record(
        record_type="fixture_source",
        source="arc",
        source_key="fixture-1",
        payload={"prompt": "Which fixture is stable?"},
    )


def _write(path: Path, *, role: str = TRAINING, provenance: dict | None = None) -> dict:
    return write_artifact(
        path,
        [_row()],
        artifact_kind="fixture_source",
        role=role,
        producer=producer_identity("partition-test", __file__),
        config={"fixture": True},
        provenance=provenance,
    )


def _rewrite_manifest(path: Path, update) -> None:
    sidecar = artifact_manifest_path(path)
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    update(manifest)
    sidecar.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_write_requires_explicit_closed_role_and_reserves_it(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="role"):
        write_artifact(  # type: ignore[call-arg]
            tmp_path / "missing-role.jsonl",
            [_row()],
            artifact_kind="fixture_source",
            producer=producer_identity("partition-test", __file__),
            config={},
        )
    with pytest.raises(ArtifactManifestError, match="unknown artifact role"):
        _write(tmp_path / "unknown-role.jsonl", role="testing")
    with pytest.raises(ArtifactManifestError, match="reserved fields.*role"):
        _write(tmp_path / "override-role.jsonl", provenance={"role": VALIDATION})


def test_read_rejects_unknown_mismatched_and_missing_roles(tmp_path: Path) -> None:
    mismatch_path = tmp_path / "mismatch.jsonl"
    _write(mismatch_path)
    with pytest.raises(ArtifactManifestError, match="artifact role is 'training', expected 'validation'"):
        read_artifact(mismatch_path, expected_role=VALIDATION)
    with pytest.raises(ArtifactManifestError, match="unknown artifact role 'testing'"):
        read_artifact(mismatch_path, expected_role="testing")

    missing_path = tmp_path / "missing.jsonl"
    _write(missing_path)
    _rewrite_manifest(missing_path, lambda manifest: manifest["provenance"].pop("role"))
    with pytest.raises(ArtifactManifestError, match=r"no provenance\.role"):
        read_artifact(missing_path)

    unknown_path = tmp_path / "unknown.jsonl"
    _write(unknown_path)
    _rewrite_manifest(unknown_path, lambda manifest: manifest["provenance"].__setitem__("role", "testing"))
    with pytest.raises(ArtifactManifestError, match="unknown artifact role 'testing'"):
        read_artifact(unknown_path)


def test_role_tampering_fails_expected_role_and_manifest_integrity(tmp_path: Path) -> None:
    path = tmp_path / "tampered-role.jsonl"
    _write(path, role=TRAINING)
    _rewrite_manifest(path, lambda manifest: manifest["provenance"].__setitem__("role", VALIDATION))

    with pytest.raises(ArtifactManifestError, match="artifact role is 'validation', expected 'training'"):
        read_artifact(path, expected_role=TRAINING)
    with pytest.raises(ArtifactManifestError, match="manifest integrity digest mismatch"):
        read_artifact(path, expected_role=VALIDATION)


def test_partition_registry_has_every_exact_paper_route() -> None:
    expected = {
        "arc": {TRAINING},
        "openbookqa": {TRAINING},
        "bbh": {TRAINING},
        "mmlu": {VALIDATION, FINAL_EVAL},
        "harmbench": {TRAINING, VALIDATION},
        "or_bench": {VALIDATION},
        "clearharm": {FINAL_EVAL},
        "wildguardtest": {FINAL_EVAL},
        "xstest": {FINAL_EVAL},
        "wildjailbreak": {FINAL_EVAL},
    }
    assert {source: set(partitions) for source, partitions in PARTITION_REGISTRY.items()} == expected
    assert sum(len(partitions) for partitions in PARTITION_REGISTRY.values()) == 12
    assert require_partition("mmlu", role=VALIDATION).paper_route == "sycophancy_validation_helpfulness_and_resistance"
    assert require_partition("mmlu", role=VALIDATION).paper_status == PAPER_UNSPECIFIED_RECONSTRUCTION
    assert require_partition("mmlu", role=FINAL_EVAL).paper_route == "sycophancy_held_out_reporting"
    assert "disjoint from the validation artifact" in require_partition("mmlu", role=FINAL_EVAL).notes
    assert require_partition("harmbench", TRAINING).paper_status == PAPER_UNSPECIFIED_RECONSTRUCTION
    assert require_partition("harmbench", VALIDATION).paper_status == PAPER_UNSPECIFIED_RECONSTRUCTION
    with pytest.raises(PartitionError, match="multiple registered partitions"):
        require_partition("harmbench")
    with pytest.raises(PartitionError, match="multiple registered partitions"):
        require_partition("mmlu")
    with pytest.raises(PartitionError, match="has role 'training', not 'validation'"):
        require_partition("harmbench", TRAINING, role=VALIDATION)


def test_harmbench_partition_is_exact_deterministic_and_order_invariant() -> None:
    ids = [f"hb-{index}" for index in range(12)]
    expected = {
        TRAINING: ("hb-0", "hb-10", "hb-11", "hb-2", "hb-3", "hb-4", "hb-6", "hb-7", "hb-8", "hb-9"),
        VALIDATION: ("hb-1", "hb-5"),
    }
    assert partition_harmbench_ids(ids) == expected
    assert partition_harmbench_ids(reversed(ids)) == expected
    for partition, partition_ids in expected.items():
        assert all(assign_harmbench_partition(example_id) == partition for example_id in partition_ids)

    provenance = harmbench_partition_provenance(VALIDATION)
    assert provenance == {
        "source": "harmbench",
        "partition": VALIDATION,
        "role": VALIDATION,
        "paper_status": PAPER_UNSPECIFIED_RECONSTRUCTION,
        "namespace": HARM_BENCH_PARTITION_NAMESPACE,
        "seed": HARM_BENCH_PARTITION_SEED,
        "rule": HARM_BENCH_PARTITION_RULE,
        "modulus": 5,
        "validation_buckets": [0],
    }


def test_harmbench_partitions_are_disjoint_and_explicit_conflicts_fail() -> None:
    partitions = partition_harmbench_ids(f"example-{index}" for index in range(100))
    assert partitions[TRAINING]
    assert partitions[VALIDATION]
    assert set(partitions[TRAINING]).isdisjoint(partitions[VALIDATION])
    assert verify_disjoint_ids(partitions[TRAINING], partitions[VALIDATION]) is None

    with pytest.raises(PartitionError, match="overlap.*same-id"):
        verify_disjoint_ids(["train-only", "same-id"], ["validation-only", "same-id"])
    with pytest.raises(PartitionError, match="conflicting with configured partition 'training'"):
        assign_harmbench_partition("hb-5", configured_partition=TRAINING)
    assert assign_harmbench_partition("hb-5", configured_partition=VALIDATION) == VALIDATION


def test_harmbench_identity_ignores_generic_ids_and_closes_cross_split_leakage() -> None:
    training_export = {"id": "train-row-17", "BehaviorID": "shared-behavior", "Behavior": "Prompt."}
    validation_export = {"id": "validation-row-99", "BehaviorID": "shared-behavior", "Behavior": "Prompt."}
    training_id = stable_source_identity("harmbench", training_export)
    validation_id = stable_source_identity("harmbench", validation_export)
    assert training_id == validation_id == "shared-behavior"
    assert assign_harmbench_partition(training_id) == assign_harmbench_partition(validation_id)
    with pytest.raises(PartitionError, match="overlap.*shared-behavior"):
        verify_disjoint_ids([training_id], [validation_id])

    with pytest.raises(PartitionError, match="conflicting harmbench stable identity fields"):
        stable_source_identity(
            "harmbench",
            {"BehaviorID": "one", "behavior_id": "two", "id": "ignored"},
        )
