import hashlib
import json
from pathlib import Path

import pytest

import experiments.rmct_tbsr.prepare as prepare_module
from experiments.rmct_tbsr.constants import (
    AUTHORITATIVE_TRAINING_COUNTS,
    AUTHORITATIVE_TRAINING_ROWS,
    SOURCE_ROWS,
    TRAINING_COUNTS,
    TRAINING_ROWS,
)


def _row(dataset: str, index: int) -> dict:
    question_id = f"{dataset}-{index:04d}"
    return {
        "question": f"Question {question_id}",
        "question_id": question_id,
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": "clean"}],
        "biased_messages": [{"role": "user", "content": "biased"}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": "frozen distractor",
    }


def _source_rows() -> list[dict]:
    prefix = [
        *(_row("logiqa", index) for index in range(AUTHORITATIVE_TRAINING_COUNTS["logiqa"])),
        *(_row("hellaswag", index) for index in range(AUTHORITATIVE_TRAINING_COUNTS["hellaswag"])),
    ]
    tail = [_row("hellaswag", 10_000 + index) for index in range(SOURCE_ROWS - AUTHORITATIVE_TRAINING_ROWS)]
    return [*prefix, *tail]


def _write_source(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    return payload


def test_prepare_freezes_balanced_hash_ranked_subset_and_validates_all_identities(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    source_payload = _write_source(source, _source_rows())
    output = tmp_path / "training.jsonl"
    manifest_path = tmp_path / "training.manifest.json"
    source_hash = hashlib.sha256(source_payload).hexdigest()

    manifest = prepare_module.prepare_training_population(
        source,
        output,
        manifest_path,
        expected_source_sha256=source_hash,
    )

    output_rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in output_rows] == manifest["training"]["question_ids"]
    assert manifest["training"]["rows"] == TRAINING_ROWS
    assert manifest["training"]["counts_by_dataset"] == TRAINING_COUNTS
    assert len(manifest["training"]["question_ids"]) == TRAINING_ROWS
    assert manifest["source"]["rows"] == SOURCE_ROWS
    assert manifest["source"]["authoritative_prefix_rows"] == AUTHORITATIVE_TRAINING_ROWS
    assert manifest["selection"]["seed"] == 20260729

    monkeypatch.setattr(prepare_module, "SOURCE_SHA256", source_hash)
    monkeypatch.setattr(
        prepare_module,
        "TRAINING_SHA256",
        hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    artifact = prepare_module.validate_training_manifest(manifest_path)
    assert artifact.path == output.resolve()
    assert len(artifact.rows) == TRAINING_ROWS
    assert artifact.training_identity["content_sha256"] == manifest["training"]["content_sha256"]


def test_prepare_refuses_overwrite_and_preserves_published_bytes(tmp_path):
    source = tmp_path / "source.jsonl"
    payload = _write_source(source, _source_rows())
    output = tmp_path / "training.jsonl"
    manifest = tmp_path / "training.manifest.json"
    digest = hashlib.sha256(payload).hexdigest()
    prepare_module.prepare_training_population(source, output, manifest, expected_source_sha256=digest)
    before = (output.read_bytes(), manifest.read_bytes())

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_module.prepare_training_population(source, output, manifest, expected_source_sha256=digest)

    assert (output.read_bytes(), manifest.read_bytes()) == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "question_id": rows[0]["question_id"]}), "duplicate"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "biased_option": "A"}), "distractor"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "source_dataset": "hellaswag"}), "dataset counts"),
    ],
)
def test_prepare_fails_closed_on_population_defects(tmp_path, mutation, message):
    rows = _source_rows()
    mutation(rows)
    source = tmp_path / "source.jsonl"
    payload = _write_source(source, rows)

    with pytest.raises(ValueError, match=message):
        prepare_module.prepare_training_population(
            source,
            tmp_path / "training.jsonl",
            tmp_path / "manifest.json",
            expected_source_sha256=hashlib.sha256(payload).hexdigest(),
        )
