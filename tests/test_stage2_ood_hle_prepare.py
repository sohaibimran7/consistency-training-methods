"""Contract tests for immutable Stage 2 OOD-HLE 2×2 preparation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.stage1_iid_diagnostic_none import prepare as iid_prepare
from experiments.stage2_ood_hle import materialize as prepare
from experiments.stage2_ood_hle.prepare import load_bias_contract


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )


def _wrong_message(record, argument: str) -> list[dict[str, str]]:
    from mcq_bias.pipeline.records import instruction_suffix

    return [{"role": "user", "content": f"{argument}\n{record.parsed_input()}{instruction_suffix('none')}"}]


def _in_domain_rows() -> list[dict]:
    from mcq_bias.pipeline.records import MCQRecord, instruction_suffix

    rows: list[dict] = []
    for index in range(prepare.IN_DOMAIN_ROWS):
        dataset = "logiqa" if index < prepare.IN_DOMAIN_ROWS // 2 else "hellaswag"
        record = MCQRecord(
            question=f"In-domain {dataset} question {index}?",
            options=[f"{dataset} option {index}-{letter}" for letter in "ABCD"],
            ground_truth_idx=index % 4,
            dataset=dataset,
        )
        argument = f"Stored wrong argument {index}."
        rows.append(
            {
                "question": record.parsed_input(),
                "question_id": record.question_id,
                "source_dataset": dataset,
                "prompt_style": "none",
                "unbiased_messages": [
                    {"role": "user", "content": record.parsed_input() + instruction_suffix("none")}
                ],
                "biased_messages": _wrong_message(record, argument),
                "bias_type": "wrong_argument",
                "ground_truth": record.ground_truth,
                "biased_option": record.biased_option,
                "biasing_text": argument,
            }
        )
    return rows


def _hle_rows_by_variant() -> dict[str, list[dict]]:
    from mcq_bias.pipeline.injectors import default_injectors
    from mcq_bias.pipeline.records import MCQRecord

    records = [
        MCQRecord(
            question=f"HLE question {index}?",
            options=[f"HLE option {index}-{letter}" for letter in "ABCD"],
            ground_truth_idx=index % 4,
            dataset=prepare.HLE_DATASET,
        )
        for index in range(prepare.HLE_ROWS)
    ]
    clean = [
        {
            "question": record.question,
            "question_id": record.question_id,
            "source_dataset": prepare.HLE_DATASET,
            "prompt_style": "none",
            "unbiased_messages": record.unbiased_messages("none"),
            "ground_truth": record.ground_truth,
        }
        for record in records
    ]
    variants: dict[str, list[dict]] = {"unbiased": clean}
    injectors = default_injectors(records)
    for bias_type in prepare.HLE_BIASES:
        rows: list[dict] = []
        for index, record in enumerate(records):
            if bias_type == prepare.TRAINING_BIAS:
                result_messages = _wrong_message(record, f"HLE wrong argument {index}.")
                biasing_text = f"HLE wrong argument {index}."
                biased_option = record.biased_option
            else:
                result = injectors[bias_type].inject(record, "none")
                assert result is not None
                result_messages = result.messages
                biasing_text = result.biasing_text
                biased_option = result.biased_option
            rows.append(
                {
                    **clean[index],
                    "bias_type": bias_type,
                    "biased_messages": result_messages,
                    "biased_option": biased_option,
                    "biasing_text": biasing_text,
                }
            )
        variants[bias_type] = rows
    return variants


def _install_fake_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict, dict[str, Path], dict[str, str]]:
    """Create syntactically real frozen inputs without any network or model."""

    source_rows = _in_domain_rows()
    source_path = tmp_path / "stage1-heldout-in-domain.jsonl"
    source_payload = _jsonl(source_rows)
    source_path.write_bytes(source_payload)
    counts = Counter(row["source_dataset"] for row in source_rows)
    iid_document = {
        "kind": iid_prepare.MANIFEST_KIND,
        "schema_version": iid_prepare.SCHEMA_VERSION,
        "splits": {
            "heldout_in_domain": {
                "path": str(source_path.resolve()),
                "content_sha256": _sha256(source_payload),
                "row_count": len(source_rows),
                "counts_by_dataset": dict(counts),
                "question_ids": [row["question_id"] for row in source_rows],
            }
        },
    }
    iid_manifest = tmp_path / "stage1-iid.manifest.json"
    iid_manifest.write_text(json.dumps(iid_document, sort_keys=True), encoding="utf-8")
    calls: list[bool] = []

    def fake_validate_manifest(path: str | Path, *, verify_source: bool = False) -> dict:
        assert Path(path).resolve() == iid_manifest.resolve()
        calls.append(verify_source)
        return iid_document

    monkeypatch.setattr(iid_prepare, "validate_manifest", fake_validate_manifest)

    source_root = tmp_path / "frozen-hle-source"
    hle_dir = source_root / "hle-eval"
    hle_dir.mkdir(parents=True)
    export_payload = _jsonl([{"source_row": index} for index in range(prepare.HLE_SOURCE_EXPORT_ROWS)])
    export_path = source_root / prepare.HLE_SOURCE_EXPORT_FILENAME
    export_path.write_bytes(export_payload)
    monkeypatch.setattr(prepare, "HLE_SOURCE_EXPORT_SHA256", _sha256(export_payload))
    source_manifest = {
        "kind": "hle_text_multiple_choice_export",
        "output": {
            "content_sha256": prepare.HLE_SOURCE_EXPORT_SHA256,
            "path": "/lus/lfs1aip2/projects/a5v/sohaib.a5v/ctm-rmct-full-20260725/"
            "artifacts/rmct-hle-gpt-oss-20b/data/hle-text-mc.jsonl",
        },
        "row_count": prepare.HLE_SOURCE_EXPORT_ROWS,
        "schema_version": 1,
        "selection": prepare.HLE_SOURCE_SELECTION,
        "source": {
            "dataset": prepare.HLE_SOURCE_DATASET_REPOSITORY,
            "revision": prepare.HLE_SOURCE_REVISION,
            "split": prepare.HLE_SOURCE_SPLIT,
        },
        "written_at": "2026-07-25T19:18:51.853748+00:00",
    }
    source_manifest_payload = (json.dumps(source_manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    source_manifest_path = source_root / prepare.HLE_SOURCE_MANIFEST_FILENAME
    source_manifest_path.write_bytes(source_manifest_payload)
    monkeypatch.setattr(prepare, "HLE_SOURCE_MANIFEST_SHA256", _sha256(source_manifest_payload))

    suite = _hle_rows_by_variant()
    hle_paths: dict[str, Path] = {}
    hle_hashes: dict[str, str] = {}
    for name, filename in prepare.HLE_FILES.items():
        path = hle_dir / filename
        payload = _jsonl(suite[name])
        path.write_bytes(payload)
        hle_paths[name] = path
        hle_hashes[name] = _sha256(payload)
    monkeypatch.setattr(prepare, "HLE_FILE_SHA256", hle_hashes)
    assert calls == []
    return iid_manifest, hle_dir, iid_document, hle_paths, hle_hashes


def test_prepare_builds_immutable_four_regime_matrix_without_mutating_sources(tmp_path, monkeypatch):
    iid_manifest, hle_dir, iid_document, hle_paths, _ = _install_fake_inputs(tmp_path, monkeypatch)
    source_path = Path(iid_document["splits"]["heldout_in_domain"]["path"])
    source_before = source_path.read_bytes()
    hle_before = {name: path.read_bytes() for name, path in hle_paths.items()}
    output = tmp_path / "stage2-ood-hle"

    manifest = prepare.prepare_ood_hle_2x2(
        iid_manifest=iid_manifest,
        hle_dir=hle_dir,
        output_dir=output,
    )

    assert source_path.read_bytes() == source_before
    assert {name: path.read_bytes() for name, path in hle_paths.items()} == hle_before
    assert manifest["kind"] == prepare.MANIFEST_KIND
    assert manifest["regime_order"] == list(prepare.REGIMES)
    assert manifest["training_contract"]["headline_iid_split"] == "heldout_in_domain"
    assert manifest["canonical_hle_source"]["source"]["revision"] == prepare.HLE_SOURCE_REVISION
    training_bias, heldout_biases, _ = load_bias_contract(output / "manifest.json")
    assert training_bias == prepare.TRAINING_BIAS
    assert heldout_biases == prepare.HELDOUT_BIASES
    assert {
        name: value["nominal_biased_samples"] for name, value in manifest["regimes"].items()
    } == {
        prepare.IID: 200,
        prepare.HELDOUT_DATASET: 100,
        prepare.HELDOUT_BIAS: 1000,
        prepare.HELDOUT_DATASET_AND_BIAS: 500,
    }

    in_domain = manifest["populations"]["in_domain"]["artifacts"]
    assert Path(in_domain["wrong_argument"]["path"]).read_bytes() == source_before
    clean_source = in_domain["unbiased"]["source"]
    assert clean_source["source_iid_heldout"]["content_sha256"] == _sha256(source_before)
    expected_targets = {
        row["question_id"]: row["biased_option"]
        for row in _read_rows(Path(in_domain["wrong_argument"]["path"]))
    }
    for bias_type in prepare.HLE_BIASES:
        rows = _read_rows(Path(in_domain[bias_type]["path"]))
        assert {row["question_id"]: row["biased_option"] for row in rows} == expected_targets
    for name, original in hle_before.items():
        assert Path(manifest["populations"]["hle"]["artifacts"][name]["path"]).read_bytes() == original

    assert prepare.validate_manifest(output / "manifest.json") == manifest
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare.prepare_ood_hle_2x2(iid_manifest=iid_manifest, hle_dir=hle_dir, output_dir=output)
    assert (output / "manifest.json").read_bytes() == (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def test_prepare_rejects_in_domain_target_that_is_not_reconstructed_from_frozen_question(tmp_path, monkeypatch):
    iid_manifest, hle_dir, iid_document, _, _ = _install_fake_inputs(tmp_path, monkeypatch)
    source_path = Path(iid_document["splits"]["heldout_in_domain"]["path"])
    rows = _read_rows(source_path)
    row = rows[0]
    row["biased_option"] = next(
        option for option in "ABCD" if option not in {row["ground_truth"], row["biased_option"]}
    )
    payload = _jsonl(rows)
    source_path.write_bytes(payload)
    iid_document["splits"]["heldout_in_domain"]["content_sha256"] = _sha256(payload)

    with pytest.raises(ValueError, match="reconstructed biased target differs"):
        prepare.prepare_ood_hle_2x2(
            iid_manifest=iid_manifest,
            hle_dir=hle_dir,
            output_dir=tmp_path / "bad-in-domain",
        )


def test_prepare_rejects_hle_variant_with_mismatching_target(tmp_path, monkeypatch):
    iid_manifest, hle_dir, _, hle_paths, hle_hashes = _install_fake_inputs(tmp_path, monkeypatch)
    changed = _read_rows(hle_paths["suggested_answer"])
    changed[0]["biased_option"] = next(
        option
        for option in "ABCD"
        if option not in {changed[0]["ground_truth"], changed[0]["biased_option"]}
    )
    changed_payload = _jsonl(changed)
    hle_paths["suggested_answer"].write_bytes(changed_payload)
    hle_hashes["suggested_answer"] = _sha256(changed_payload)

    with pytest.raises(ValueError, match="biased target differs"):
        prepare.prepare_ood_hle_2x2(
            iid_manifest=iid_manifest,
            hle_dir=hle_dir,
            output_dir=tmp_path / "bad-hle",
        )


def test_validate_manifest_rejects_tampered_published_artifact(tmp_path, monkeypatch):
    iid_manifest, hle_dir, _, _, _ = _install_fake_inputs(tmp_path, monkeypatch)
    output = tmp_path / "stage2-ood-hle"
    manifest = prepare.prepare_ood_hle_2x2(iid_manifest=iid_manifest, hle_dir=hle_dir, output_dir=output)
    artifact = Path(manifest["populations"]["in_domain"]["artifacts"]["suggested_answer"]["path"])
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="artifact bytes differ"):
        prepare.validate_manifest(output / "manifest.json")


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
