import hashlib
import json
from pathlib import Path

import pytest

from experiments.switch_gate.prepare import (
    CONFIRMATION_SIZES,
    SELECTION_SEED,
    prepare_switch_gate,
)


def _row(dataset: str, index: int) -> dict:
    question_id = f"{dataset}-{index:04d}"
    return {
        "question": f"Question {question_id}?",
        "question_id": question_id,
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": f"clean {question_id}"}],
        "biased_messages": [{"role": "user", "content": f"biased {question_id}"}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": "a frozen distractor argument",
    }


def _source_rows() -> list[dict]:
    # Contents stay tiny even though the authoritative cardinalities are real.
    return [*(_row("logiqa", index) for index in range(472)), *(_row("hellaswag", index) for index in range(1576))]


def _write_jsonl(path: Path, rows: list[dict]) -> bytes:
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    return payload


def _read_ids(path: str) -> list[str]:
    return [json.loads(line)["question_id"] for line in Path(path).read_text().splitlines() if line.strip()]


def _rank(question_id: str) -> bytes:
    return hashlib.sha256(f"{SELECTION_SEED}{question_id}".encode()).digest()


def test_prepare_freezes_stratified_disjoint_nested_splits(tmp_path):
    source = tmp_path / "source.jsonl"
    source_payload = _write_jsonl(source, _source_rows())
    output_dir = tmp_path / "prepared"
    manifest_path = tmp_path / "split-manifest.json"

    manifest = prepare_switch_gate(source, output_dir, manifest_path)

    assert json.loads(manifest_path.read_text()) == manifest
    assert manifest["schema_version"] == 1
    assert manifest["source"]["content_sha256"] == hashlib.sha256(source_payload).hexdigest()
    assert manifest["source"]["authoritative_prefix_counts_by_dataset"] == {
        "logiqa": 472,
        "hellaswag": 1576,
    }
    assert manifest["selection"]["seed"] == "20260729"
    assert manifest["screen"]["row_count"] == 100
    assert manifest["screen"]["counts_by_dataset"] == {"logiqa": 23, "hellaswag": 77}

    expected_logiqa = sorted((f"logiqa-{index:04d}" for index in range(472)), key=_rank)[:23]
    expected_hellaswag = sorted((f"hellaswag-{index:04d}" for index in range(1576)), key=_rank)[:77]
    assert set(manifest["screen"]["question_ids"]) == set(expected_logiqa + expected_hellaswag)

    screen_path = Path(manifest["screen"]["path"])
    assert _read_ids(str(screen_path)) == manifest["screen"]["question_ids"]
    assert hashlib.sha256(screen_path.read_bytes()).hexdigest() == manifest["screen"]["content_sha256"]

    expected_counts = {
        600: {"logiqa": 138, "hellaswag": 462},
        800: {"logiqa": 184, "hellaswag": 616},
        1000: {"logiqa": 230, "hellaswag": 770},
        1200: {"logiqa": 277, "hellaswag": 923},
        1600: {"logiqa": 369, "hellaswag": 1231},
        1948: {"logiqa": 449, "hellaswag": 1499},
    }
    confirmation_sets = []
    for size in CONFIRMATION_SIZES:
        entry = manifest["confirmation"][f"confirmation-n{size}"]
        path = Path(entry["path"])
        assert entry["row_count"] == size
        assert entry["counts_by_dataset"] == expected_counts[size]
        assert _read_ids(str(path)) == entry["question_ids"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["content_sha256"]
        confirmation_sets.append(set(entry["question_ids"]))

    assert all(left < right for left, right in zip(confirmation_sets, confirmation_sets[1:]))
    screen_ids = set(manifest["screen"]["question_ids"])
    assert screen_ids.isdisjoint(confirmation_sets[-1])
    assert len(screen_ids | confirmation_sets[-1]) == 2048
    assert all(manifest["assertions"].values())


def test_prepare_refuses_to_overwrite_any_published_output(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, _source_rows())
    output_dir = tmp_path / "prepared"
    manifest_path = tmp_path / "split-manifest.json"
    prepare_switch_gate(source, output_dir, manifest_path)
    before = {path: path.read_bytes() for path in [manifest_path, *output_dir.glob("*.jsonl")]}

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_switch_gate(source, output_dir, manifest_path)

    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("defect", ["wrong_count", "duplicate_id"])
def test_prepare_validates_authoritative_counts_and_unique_ids(tmp_path, defect):
    rows = _source_rows()
    if defect == "wrong_count":
        rows[0]["source_dataset"] = "hellaswag"
        match = "dataset counts"
    else:
        rows[1]["question_id"] = rows[0]["question_id"]
        match = "duplicate question_id"
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, rows)

    with pytest.raises(ValueError, match=match):
        prepare_switch_gate(source, tmp_path / "prepared", tmp_path / "manifest.json")

    assert not (tmp_path / "manifest.json").exists()
