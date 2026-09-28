import hashlib
import json
import sys
import types
from pathlib import Path

import experiments.rmct_tbsr.prepare as prepare_module
import experiments.rmct_tbsr.tasks as task_module
from experiments.rmct_tbsr.constants import (
    AUTHORITATIVE_TRAINING_COUNTS,
    AUTHORITATIVE_TRAINING_ROWS,
    HLE_BIASES,
    HLE_FILES,
    SOURCE_ROWS,
    TRAINING_SPLIT,
)


def _row(dataset: str, index: int) -> dict:
    return {
        "question": "q",
        "question_id": f"{dataset}-{index}",
        "source_dataset": dataset,
        "prompt_style": "none",
        "unbiased_messages": [{"role": "user", "content": "clean"}],
        "biased_messages": [{"role": "user", "content": "biased"}],
        "bias_type": "wrong_argument",
        "ground_truth": "A",
        "biased_option": "B",
        "biasing_text": "bias",
    }


def _training_manifest(tmp_path: Path, monkeypatch) -> Path:
    rows = [
        *(_row("logiqa", index) for index in range(AUTHORITATIVE_TRAINING_COUNTS["logiqa"])),
        *(_row("hellaswag", index) for index in range(AUTHORITATIVE_TRAINING_COUNTS["hellaswag"])),
        *(_row("hellaswag", 10_000 + index) for index in range(SOURCE_ROWS - AUTHORITATIVE_TRAINING_ROWS)),
    ]
    source = tmp_path / "source.jsonl"
    source_payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    source.write_bytes(source_payload)
    source_hash = hashlib.sha256(source_payload).hexdigest()
    output = tmp_path / "training.jsonl"
    manifest = tmp_path / "manifest.json"
    prepare_module.prepare_training_population(source, output, manifest, expected_source_sha256=source_hash)
    monkeypatch.setattr(prepare_module, "SOURCE_SHA256", source_hash)
    monkeypatch.setattr(prepare_module, "TRAINING_SHA256", hashlib.sha256(output.read_bytes()).hexdigest())
    return manifest


def test_training_factory_constructs_exact_four_ordered_scorer_compatible_tasks(tmp_path, monkeypatch):
    manifest = _training_manifest(tmp_path, monkeypatch)
    calls: list[tuple[str, dict]] = []
    switch_tasks = types.ModuleType("experiments.switch_gate.tasks")

    def clean(**kwargs):
        calls.append(("clean", kwargs))
        return {"kind": "clean", **kwargs}

    def biased(**kwargs):
        calls.append(("biased", kwargs))
        return {"kind": "biased", "include_bias_acknowledged": False, **kwargs}

    switch_tasks.switch_gate_unbiased = clean
    switch_tasks.switch_gate_biased = biased
    monkeypatch.setitem(sys.modules, "experiments.switch_gate.tasks", switch_tasks)

    tasks = task_module.training_tasks(str(manifest), "logs/rmct-tbsr/20260730/training")

    assert [task["kind"] for task in tasks] == ["clean", "clean", "biased", "biased"]
    assert [task["dataset"] for task in tasks] == ["logiqa", "hellaswag", "logiqa", "hellaswag"]
    assert all(task["split"] == TRAINING_SPLIT for task in tasks)
    assert all(task["prompt_style"] == "none" for task in tasks)
    assert all(task["bias_type"] == "wrong_argument" for task in tasks[2:])
    assert all(task["unbiased_log"] == "logs/rmct-tbsr/20260730/training" for task in tasks[2:])


def test_hle_factory_verifies_files_and_preserves_paper_bias_order(tmp_path, monkeypatch):
    hashes = {}
    for name, filename in HLE_FILES.items():
        path = tmp_path / filename
        path.write_text(name, encoding="utf-8")
        hashes[name] = hashlib.sha256(name.encode()).hexdigest()
    monkeypatch.setattr(task_module, "HLE_FILE_SHA256", hashes)

    captured = {}
    switch_tasks = types.ModuleType("experiments.switch_gate.tasks")

    def hle_tasks(**kwargs):
        captured.update(kwargs)
        return ["clean", *kwargs["bias_files"]]

    switch_tasks.hle_tasks = hle_tasks
    monkeypatch.setitem(sys.modules, "experiments.switch_gate.tasks", switch_tasks)

    tasks = task_module.hle_tasks("logs/rmct-tbsr/20260730/hle", str(tmp_path))

    assert tasks == ["clean", *HLE_BIASES]
    assert list(captured["bias_files"]) == list(HLE_BIASES)
    assert captured["unbiased_log"] == "logs/rmct-tbsr/20260730/hle"
