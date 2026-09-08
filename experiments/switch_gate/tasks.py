"""Offline Inspect task factories over already-frozen switch-gate inputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inspect_ai import Task, task

TRAINING_DATASETS = ("logiqa", "hellaswag")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen task input does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: frozen task input is empty")
    return rows


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    path: Path,
    dataset: str | None = None,
    prompt_style: str | None = None,
    bias_type: str | None = None,
    biased: bool,
) -> None:
    question_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        required = {"question_id", "source_dataset", "prompt_style", "unbiased_messages", "ground_truth"}
        if biased:
            required.update({"biased_messages", "bias_type", "biased_option", "biasing_text"})
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"{path}:{index}: missing frozen field(s): {', '.join(missing)}")
        question_id = row["question_id"]
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path}:{index}: question_id must be a non-empty string")
        question_ids.append(question_id)
        if dataset is not None and row["source_dataset"] != dataset:
            raise ValueError(
                f"{path}:{index}: source_dataset {row['source_dataset']!r} does not match task dataset {dataset!r}"
            )
        if prompt_style is not None and row["prompt_style"] != prompt_style:
            raise ValueError(
                f"{path}:{index}: prompt_style {row['prompt_style']!r} does not match task prompt_style "
                f"{prompt_style!r}"
            )
        if bias_type is not None and row.get("bias_type") != bias_type:
            raise ValueError(
                f"{path}:{index}: bias_type {row.get('bias_type')!r} does not match task bias_type {bias_type!r}"
            )
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"{path}: question_id values must be unique")


def _restrict_task(task_object: Task, *, dataset: str, prompt_style: str, bias_type: str | None) -> Task:
    """Filter a mixed frozen file and enforce switch-scorer lookup metadata."""

    selected = task_object.dataset.filter(
        lambda sample: bool(sample.metadata) and sample.metadata.get("source_dataset") == dataset,
        name=f"switch-gate-{dataset}",
    )
    if not selected:
        raise ValueError(f"frozen file contains no samples for task dataset {dataset!r}")
    for sample in selected:
        metadata = sample.metadata or {}
        if metadata.get("source_dataset") != dataset:
            raise ValueError("sample source_dataset does not match the decorated task's dataset argument")
        if metadata.get("prompt_style") != prompt_style:
            raise ValueError(
                f"sample {sample.id!r} has prompt_style {metadata.get('prompt_style')!r}, expected {prompt_style!r}"
            )
        if bias_type is not None and metadata.get("bias_type") != bias_type:
            raise ValueError(
                f"sample {sample.id!r} has bias_type {metadata.get('bias_type')!r}, expected {bias_type!r}"
            )
    task_object.dataset = selected
    return task_object


@task
def switch_gate_unbiased(
    frozen_file: str,
    dataset: str,
    prompt_style: str = "none",
    split: str = "frozen",
) -> Task:
    """One dataset's unbiased task, loaded without materialization."""

    from mcq_bias.tasks import unbiased_task_from_frozen

    path = Path(frozen_file)
    task_object = unbiased_task_from_frozen(
        path,
        metadata={
            "dataset": dataset,
            "source_dataset": dataset,
            "prompt_style": prompt_style,
            "split": split,
            "dataset_file": str(path.resolve()),
        },
    )
    return _restrict_task(task_object, dataset=dataset, prompt_style=prompt_style, bias_type=None)


@task
def switch_gate_biased(
    frozen_file: str,
    dataset: str,
    unbiased_log: str,
    bias_type: str = "wrong_argument",
    prompt_style: str = "none",
    split: str = "frozen",
) -> Task:
    """One dataset's biased task with switch scoring and no model grader."""

    from mcq_bias.tasks import task_from_frozen

    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty log directory, glob, or exact .eval path")
    path = Path(frozen_file)
    task_object = task_from_frozen(
        path,
        metadata={
            "dataset": dataset,
            "source_dataset": dataset,
            "bias_type": bias_type,
            "prompt_style": prompt_style,
            "split": split,
            "dataset_file": str(path.resolve()),
            "unbiased_log": unbiased_log,
        },
        unbiased_log=unbiased_log,
        include_bias_acknowledged=False,
    )
    return _restrict_task(task_object, dataset=dataset, prompt_style=prompt_style, bias_type=bias_type)


def _load_manifest(manifest: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    if isinstance(manifest, Mapping):
        document = dict(manifest)
        base = Path.cwd()
    else:
        manifest_path = Path(manifest)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"switch-gate manifest does not exist: {manifest_path}")
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid switch-gate manifest {manifest_path}: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError(f"switch-gate manifest {manifest_path} must contain a JSON object")
        base = manifest_path.resolve().parent
    if document.get("schema_version") != 1 or document.get("kind") != "switch_gate_split_manifest":
        raise ValueError("unsupported switch-gate manifest schema")
    return document, base


def _manifest_split(manifest: str | Path | Mapping[str, Any], split: str) -> tuple[Path, list[dict[str, Any]]]:
    document, base = _load_manifest(manifest)
    if split == "screen":
        entry = document.get("screen")
    else:
        confirmation = document.get("confirmation")
        entry = confirmation.get(split) if isinstance(confirmation, dict) else None
    if not isinstance(entry, dict):
        confirmation = document.get("confirmation")
        available = ["screen", *sorted(confirmation)] if isinstance(confirmation, dict) else ["screen"]
        raise ValueError(f"unknown manifest split {split!r}; expected one of {available}")

    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"manifest split {split!r} has no path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base / path
    payload = path.read_bytes()
    expected_hash = entry.get("content_sha256")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(f"manifest split {split!r} hash mismatch: expected {expected_hash!r}, got {actual_hash!r}")
    rows = _read_jsonl(path)
    expected_ids = entry.get("question_ids")
    actual_ids = [row.get("question_id") for row in rows]
    if expected_ids != actual_ids:
        raise ValueError(f"manifest split {split!r} question_ids do not match {path}")
    if entry.get("row_count") != len(rows):
        raise ValueError(f"manifest split {split!r} row_count does not match {path}")
    actual_counts = Counter(row.get("source_dataset") for row in rows)
    if entry.get("counts_by_dataset") != {dataset: actual_counts.get(dataset, 0) for dataset in TRAINING_DATASETS}:
        raise ValueError(f"manifest split {split!r} dataset counts do not match {path}")
    _validate_rows(rows, path=path, prompt_style="none", bias_type="wrong_argument", biased=True)
    return path, rows


def training_tasks(
    manifest: str | Path | Mapping[str, Any],
    split: str,
    unbiased_log: str,
    prompt_style: str = "none",
) -> list[Task]:
    """Build matched training-domain evals, all unbiased tasks first.

    The manifest file is mixed LogiQA/HellaSwag.  Separate decorated tasks keep
    every sample's ``source_dataset`` equal to the task's ``dataset`` argument,
    which is how the upstream switch scorer resolves the corresponding
    unbiased log.
    """

    if prompt_style != "none":
        raise ValueError("the frozen switch-gate training inputs use prompt_style='none'")
    path, rows = _manifest_split(manifest, split)
    counts = Counter(row["source_dataset"] for row in rows)
    missing = [dataset for dataset in TRAINING_DATASETS if counts[dataset] == 0]
    if missing:
        raise ValueError(f"manifest split {split!r} is missing training dataset(s): {missing}")

    unbiased = [
        switch_gate_unbiased(
            frozen_file=str(path),
            dataset=dataset,
            prompt_style=prompt_style,
            split=split,
        )
        for dataset in TRAINING_DATASETS
    ]
    biased = [
        switch_gate_biased(
            frozen_file=str(path),
            dataset=dataset,
            unbiased_log=unbiased_log,
            bias_type="wrong_argument",
            prompt_style=prompt_style,
            split=split,
        )
        for dataset in TRAINING_DATASETS
    ]
    return [*unbiased, *biased]


def hle_tasks(
    unbiased_file: str,
    bias_files: Mapping[str, str],
    unbiased_log: str,
    dataset: str = "hle-text-mc",
    prompt_style: str = "none",
) -> list[Task]:
    """Build the frozen HLE suite: unbiased first, then each requested bias."""

    if not isinstance(bias_files, Mapping) or not bias_files:
        raise ValueError("bias_files must be a non-empty mapping of bias_type to frozen JSONL")
    unbiased_path = Path(unbiased_file)
    unbiased_rows = _read_jsonl(unbiased_path)
    _validate_rows(
        unbiased_rows,
        path=unbiased_path,
        dataset=dataset,
        prompt_style=prompt_style,
        biased=False,
    )
    unbiased_by_id = {row["question_id"]: row for row in unbiased_rows}

    checked_biases: list[tuple[str, Path]] = []
    for bias_type, raw_path in bias_files.items():
        if not isinstance(bias_type, str) or not bias_type:
            raise ValueError("bias_files keys must be non-empty bias names")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"bias_files[{bias_type!r}] must be a non-empty path string")
        path = Path(raw_path)
        rows = _read_jsonl(path)
        _validate_rows(
            rows,
            path=path,
            dataset=dataset,
            prompt_style=prompt_style,
            bias_type=bias_type,
            biased=True,
        )
        by_id = {row["question_id"]: row for row in rows}
        if set(by_id) != set(unbiased_by_id):
            raise ValueError(f"{path}: question IDs do not exactly match the unbiased HLE file")
        for question_id, row in by_id.items():
            if row["ground_truth"] != unbiased_by_id[question_id]["ground_truth"]:
                raise ValueError(f"{path}: ground_truth differs for question_id {question_id!r}")
        checked_biases.append((bias_type, path))

    tasks = [
        switch_gate_unbiased(
            frozen_file=str(unbiased_path),
            dataset=dataset,
            prompt_style=prompt_style,
            split="hle",
        )
    ]
    tasks.extend(
        switch_gate_biased(
            frozen_file=str(path),
            dataset=dataset,
            unbiased_log=unbiased_log,
            bias_type=bias_type,
            prompt_style=prompt_style,
            split="hle",
        )
        for bias_type, path in checked_biases
    )
    return tasks


__all__ = [
    "hle_tasks",
    "switch_gate_biased",
    "switch_gate_unbiased",
    "training_tasks",
]
