"""Inspect task factories for the attested no-CoT Stage 1 IID diagnostic.

The task implementation deliberately reuses the already-audited frozen-MCQ
task constructors.  This module owns the *contract boundary*: it accepts only
the no-CoT manifest produced by :mod:`.prepare`, re-verifies the recovered
source before every factory construction, and supplies its pinned source
digest and ``prompt_style: none`` to each task.

The returned task registry names remain ``stage1_iid_unbiased`` and
``stage1_iid_biased``.  That keeps the existing local switch scorer, raw-log
gate, and post-hoc Luna tooling interoperable; the manifest/source digest and
separate task factory identify this no-CoT population unambiguously.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from inspect_ai import Task

from experiments.stage1_iid_diagnostic.tasks import stage1_iid_biased, stage1_iid_unbiased
from experiments.stage1_iid_diagnostic_none.prepare import (
    BIAS_TYPE,
    DATASETS,
    PROMPT_STYLE,
    validate_manifest,
)

_SPLITS = ("train_eval", "heldout_in_domain")


def _split_inputs(
    manifest: str | Path | Mapping[str, Any],
    split: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    """Load one split after proving it still matches the attested source."""

    if split not in _SPLITS:
        raise ValueError(f"unknown no-CoT diagnostic split: {split!r}")
    # Unlike the historical portable diagnostic, this final-recovery factory
    # deliberately requires the recovered source and transform manifest to be
    # present.  It checks the selected JSONL bytes against the source offsets,
    # not merely their IDs or manifest-declared hash.
    document = validate_manifest(manifest, verify_source=True)
    entry = document["splits"][split]
    path = Path(entry["path"])
    ids_by_dataset: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():  # prepare/validation already reject this; retain a local clear error.
            raise ValueError(f"{path}:{line_number}: blank no-CoT diagnostic row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - validated above
            raise ValueError(f"{path}:{line_number}: invalid no-CoT diagnostic JSON") from exc
        dataset = row.get("source_dataset")
        question_id = row.get("question_id")
        if dataset not in DATASETS or not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{path}:{line_number}: invalid no-CoT diagnostic ID/dataset")
        if row.get("prompt_style") != PROMPT_STYLE or row.get("bias_type") != BIAS_TYPE:
            raise ValueError(f"{path}:{line_number}: row conflicts with the no-CoT bias/prompt contract")
        ids_by_dataset[dataset].append(question_id)

    manifest_ids = entry["question_ids"]
    observed_ids = [question_id for dataset in DATASETS for question_id in ids_by_dataset[dataset]]
    # Source rows alternate datasets, so this list-order check must compare
    # set identity only; exact source ordering is independently byte-checked
    # by ``validate_manifest(..., verify_source=True)`` above.
    if set(observed_ids) != set(manifest_ids) or len(observed_ids) != len(manifest_ids):
        raise ValueError(f"no-CoT diagnostic split {split!r} IDs do not match its frozen file")
    if any(len(ids_by_dataset[dataset]) != 100 for dataset in DATASETS):
        raise ValueError(f"no-CoT diagnostic split {split!r} must contain exactly 100 IDs per dataset")
    if len(set(observed_ids)) != len(observed_ids):
        raise ValueError(f"no-CoT diagnostic split {split!r} has duplicate question IDs")
    return document, entry, ids_by_dataset


def _validate_factory_args(
    *,
    prompt_style: str,
    unbiased_log: str,
    include_bias_acknowledged: bool,
    grader_model: str | None,
) -> None:
    if prompt_style != PROMPT_STYLE:
        raise ValueError(f"no-CoT Stage 1 IID diagnostics require prompt_style={PROMPT_STYLE!r}")
    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    if grader_model is not None and not include_bias_acknowledged:
        raise ValueError("grader_model requires include_bias_acknowledged=true")


def diagnostic_tasks(
    manifest: str | Path | Mapping[str, Any],
    split: str,
    unbiased_log: str,
    prompt_style: str = PROMPT_STYLE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
) -> list[Task]:
    """Return clean LogiQA/HellaSwag tasks, then their matched biased tasks."""

    _validate_factory_args(
        prompt_style=prompt_style,
        unbiased_log=unbiased_log,
        include_bias_acknowledged=include_bias_acknowledged,
        grader_model=grader_model,
    )
    document, entry, ids_by_dataset = _split_inputs(manifest, split)
    frozen_file = entry["path"]
    source_identity = document["source"]["content_sha256"]
    clean = [
        stage1_iid_unbiased(
            frozen_file=frozen_file,
            dataset=dataset,
            split=split,
            question_ids_from=ids_by_dataset[dataset],
            prompt_style=PROMPT_STYLE,
            source_sha256=source_identity,
        )
        for dataset in DATASETS
    ]
    biased = [
        stage1_iid_biased(
            frozen_file=frozen_file,
            dataset=dataset,
            split=split,
            unbiased_log=unbiased_log,
            question_ids_from=ids_by_dataset[dataset],
            prompt_style=PROMPT_STYLE,
            bias_type=BIAS_TYPE,
            include_bias_acknowledged=include_bias_acknowledged,
            grader_model=grader_model,
            source_sha256=source_identity,
        )
        for dataset in DATASETS
    ]
    return [*clean, *biased]


def diagnostic_biased_tasks(
    manifest: str | Path | Mapping[str, Any],
    split: str,
    unbiased_log: str,
    prompt_style: str = PROMPT_STYLE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
) -> list[Task]:
    """Return only biased no-CoT tasks after the matching clean logs exist."""

    _validate_factory_args(
        prompt_style=prompt_style,
        unbiased_log=unbiased_log,
        include_bias_acknowledged=include_bias_acknowledged,
        grader_model=grader_model,
    )
    document, entry, ids_by_dataset = _split_inputs(manifest, split)
    source_identity = document["source"]["content_sha256"]
    return [
        stage1_iid_biased(
            frozen_file=entry["path"],
            dataset=dataset,
            split=split,
            unbiased_log=unbiased_log,
            question_ids_from=ids_by_dataset[dataset],
            prompt_style=PROMPT_STYLE,
            bias_type=BIAS_TYPE,
            include_bias_acknowledged=include_bias_acknowledged,
            grader_model=grader_model,
            source_sha256=source_identity,
        )
        for dataset in DATASETS
    ]


def diagnostic_matrix_tasks(
    manifest: str | Path | Mapping[str, Any],
    unbiased_log: str,
    prompt_style: str = PROMPT_STYLE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
) -> list[Task]:
    """Return both splits, with every clean task before every biased task."""

    clean: list[Task] = []
    biased: list[Task] = []
    for split in _SPLITS:
        tasks = diagnostic_tasks(
            manifest=manifest,
            split=split,
            unbiased_log=unbiased_log,
            prompt_style=prompt_style,
            include_bias_acknowledged=include_bias_acknowledged,
            grader_model=grader_model,
        )
        clean.extend(tasks[: len(DATASETS)])
        biased.extend(tasks[len(DATASETS) :])
    return [*clean, *biased]


__all__ = ["diagnostic_biased_tasks", "diagnostic_matrix_tasks", "diagnostic_tasks"]
