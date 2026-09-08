"""Inspect task factories over the frozen Stage 1 IID diagnostic splits."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Mapping

from inspect_ai import Task, task

from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat
from experiments.stage1_iid_diagnostic.prepare import (
    BIAS_TYPE,
    DATASETS,
    PROMPT_STYLE,
    validate_manifest,
)


def _restrict_task(
    task_object: Task,
    *,
    dataset: str,
    split: str,
    prompt_style: str,
    bias_type: str | None,
    question_ids: list[str],
) -> Task:
    """Restrict an upstream task to exactly the manifest-selected IDs.

    ``mcq_bias.task_from_frozen`` accepts ``question_ids_from`` for pairing its
    switch scorer, but it does not promise to apply that list to the task
    dataset.  Filtering here makes the evaluated population explicit for both
    the native split files and a larger alternate prompt rendering.
    """

    requested = [str(question_id) for question_id in question_ids]
    wanted = set(requested)
    if not wanted or len(wanted) != len(requested):
        raise ValueError(f"diagnostic split {split!r}/{dataset!r} has invalid question IDs")
    selected = task_object.dataset.filter(
        lambda sample: (
            bool(sample.metadata)
            and sample.metadata.get("source_dataset") == dataset
            and str(sample.id) in wanted
        ),
        name=f"stage1-iid-{split}-{dataset}-{bias_type or 'unbiased'}",
    )
    selected_ids: set[str] = set()
    for sample in selected:
        metadata = sample.metadata or {}
        sample_id = str(sample.id)
        if sample_id in selected_ids:
            raise ValueError(f"diagnostic task has duplicate requested question_id {sample_id!r}")
        selected_ids.add(sample_id)
        if metadata.get("source_dataset") != dataset or sample_id not in wanted:
            raise ValueError(f"sample {sample.id!r} conflicts with the diagnostic selection")
        if metadata.get("prompt_style") != prompt_style:
            raise ValueError(f"sample {sample.id!r} prompt_style does not match {prompt_style!r}")
        if bias_type is not None and metadata.get("bias_type") != bias_type:
            raise ValueError(f"sample {sample.id!r} bias_type does not match {bias_type!r}")
    if selected_ids != wanted:
        missing = sorted(wanted - selected_ids)
        unexpected = sorted(selected_ids - wanted)
        raise ValueError(
            f"diagnostic split {split!r}/{dataset!r} does not exactly match requested IDs: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    task_object.dataset = selected
    return task_object


@task
def stage1_iid_unbiased(
    frozen_file: str,
    dataset: str,
    split: str,
    question_ids_from: list[str],
    prompt_style: str = PROMPT_STYLE,
    source_sha256: str | None = None,
) -> Task:
    """Load one dataset's clean prompts from a verified diagnostic split."""

    # The clean task includes the optional options-considered scorer.  Install
    # the shared compatibility wrapper here too, rather than relying on a
    # later biased task in the factory to have been constructed first.
    install_conditional_nan_compat()
    from mcq_bias.tasks import unbiased_task_from_frozen

    path = Path(frozen_file)
    task_object = unbiased_task_from_frozen(
        path,
        metadata={
            "dataset": dataset,
            "source_dataset": dataset,
            "bias_type": None,
            "prompt_style": prompt_style,
            "split": split,
            "dataset_file": str(path.resolve()),
            "source_sha256": source_sha256,
            "source_identity_digest": source_sha256,
            "question_ids_from": question_ids_from,
        },
    )
    return _restrict_task(
        task_object,
        dataset=dataset,
        split=split,
        prompt_style=prompt_style,
        bias_type=None,
        question_ids=question_ids_from,
    )


@task
def stage1_iid_biased(
    frozen_file: str,
    dataset: str,
    split: str,
    unbiased_log: str,
    question_ids_from: list[str],
    variant_file: str | None = None,
    prompt_style: str = PROMPT_STYLE,
    bias_type: str = BIAS_TYPE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
    source_sha256: str | None = None,
) -> Task:
    """Load one dataset's biased prompts and pair them with its clean log.

    The acknowledgement grader is opt-in.  The default task uses only local,
    deterministic MCQ and switch scorers and creates no grader client.
    """

    install_conditional_nan_compat()
    from mcq_bias.tasks import task_from_frozen

    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    if grader_model is not None and not include_bias_acknowledged:
        raise ValueError("grader_model requires include_bias_acknowledged=true")
    path = Path(variant_file) if variant_file is not None else Path(frozen_file)
    if variant_file is not None and not path.is_file():
        raise FileNotFoundError(f"diagnostic variant_file does not exist: {path}")
    kwargs: dict[str, Any] = {
        "metadata": {
            "dataset": dataset,
            "source_dataset": dataset,
            "bias_type": bias_type,
            "prompt_style": prompt_style,
            "split": split,
            "dataset_file": str(path.resolve()),
            "variant_file": str(path.resolve()) if variant_file is not None else None,
            "unbiased_log": unbiased_log,
            "include_bias_acknowledged": include_bias_acknowledged,
            "grader_model": grader_model,
            "source_sha256": source_sha256,
            "source_identity_digest": source_sha256,
            "question_ids_from": question_ids_from,
        },
        "unbiased_log": unbiased_log,
        "grader_model": grader_model,
        "include_bias_acknowledged": include_bias_acknowledged,
        "question_ids_from": question_ids_from,
    }
    # Older frozen mcq-bias releases use task metadata for these identities.
    # Newer releases expose them as switch-scorer arguments, which makes the
    # directory lookup additionally reject a mismatched source digest.  Keep
    # this wrapper runnable against the repository-pinned release while using
    # the stronger check whenever the installed package supports it.
    upstream_parameters = inspect.signature(task_from_frozen).parameters
    if "source_dataset" in upstream_parameters:
        kwargs["source_dataset"] = dataset
    if "source_identity_digest" in upstream_parameters:
        kwargs["source_identity_digest"] = source_sha256
    task_object = task_from_frozen(path, **kwargs)
    return _restrict_task(
        task_object,
        dataset=dataset,
        split=split,
        prompt_style=prompt_style,
        bias_type=bias_type,
        question_ids=question_ids_from,
    )


def _split_inputs(
    manifest: str | Path | Mapping[str, Any],
    split: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[str]]]:
    """Load one frozen split and prove its IDs before constructing tasks."""

    document = validate_manifest(manifest)
    entry = document["splits"][split]
    ids_by_dataset: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    with Path(entry["path"]).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ids_by_dataset[row["source_dataset"]].append(row["question_id"])
    if set().union(*map(set, ids_by_dataset.values())) != set(entry["question_ids"]):
        raise ValueError(f"diagnostic split {split!r} IDs do not match its frozen file")
    if any(len(ids_by_dataset[dataset]) != 100 for dataset in DATASETS):
        raise ValueError(f"diagnostic split {split!r} must contain exactly 100 IDs per dataset")
    return document, entry, ids_by_dataset


def _validate_variant_file(
    variant_file: str | Path | None,
    *,
    ids_by_dataset: Mapping[str, list[str]],
) -> str | None:
    """Ensure an alternate biased file contains each requested sample once.

    The canonical-pair diagnostic evaluates a derived prompt file while pairing
    against clean outputs from the frozen original split.  Do not let a missing
    or duplicate row silently turn that into a different population.
    """

    if variant_file is None:
        return None
    path = Path(variant_file)
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic variant_file does not exist: {path}")
    wanted = {
        (dataset, question_id)
        for dataset, question_ids in ids_by_dataset.items()
        for question_id in question_ids
    }
    found: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON variant row") from exc
            key = (row.get("source_dataset"), row.get("question_id"))
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"{path}: duplicate requested variant row {key!r}")
            messages = row.get("biased_messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: requested variant row has no biased_messages")
            found.add(key)
    missing = sorted(wanted - found)
    if missing:
        raise ValueError(f"{path}: variant_file is missing {len(missing)} requested row(s), first={missing[0]!r}")
    return str(path.resolve())


def diagnostic_tasks(
    manifest: str | Path | Mapping[str, Any],
    split: str,
    unbiased_log: str,
    prompt_style: str = PROMPT_STYLE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
    variant_file: str | Path | None = None,
) -> list[Task]:
    """Return clean LogiQA/HellaSwag tasks, then their matched biased tasks."""

    if prompt_style != PROMPT_STYLE:
        raise ValueError(f"Stage 1 IID diagnostics require prompt_style={PROMPT_STYLE!r}")
    if split not in {"train_eval", "heldout_in_domain"}:
        raise ValueError("split must be train_eval or heldout_in_domain")
    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    if grader_model is not None and not include_bias_acknowledged:
        raise ValueError("grader_model requires include_bias_acknowledged=true")

    document, entry, ids_by_dataset = _split_inputs(manifest, split)
    frozen_file = entry["path"]
    # This identity is recorded in every task header without making the task
    # dependent on the manifest's filesystem location.
    manifest_identity = document["source"]["content_sha256"]
    resolved_variant_file = _validate_variant_file(variant_file, ids_by_dataset=ids_by_dataset)
    clean = [
        stage1_iid_unbiased(
            frozen_file=frozen_file,
            dataset=dataset,
            split=split,
            question_ids_from=ids_by_dataset[dataset],
            prompt_style=prompt_style,
            source_sha256=manifest_identity,
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
            variant_file=resolved_variant_file,
            prompt_style=prompt_style,
            bias_type=BIAS_TYPE,
            include_bias_acknowledged=include_bias_acknowledged,
            grader_model=grader_model,
            source_sha256=manifest_identity,
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
    variant_file: str | Path | None = None,
) -> list[Task]:
    """Return only the two biased tasks, using already-generated clean logs.

    This is used for a second prompt rendering (the canonical consistency
    wrapper) without regenerating the clean reference answers or calling a
    model-based acknowledgement grader.
    """

    if prompt_style != PROMPT_STYLE:
        raise ValueError(f"Stage 1 IID diagnostics require prompt_style={PROMPT_STYLE!r}")
    if split not in {"train_eval", "heldout_in_domain"}:
        raise ValueError("split must be train_eval or heldout_in_domain")
    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    if grader_model is not None and not include_bias_acknowledged:
        raise ValueError("grader_model requires include_bias_acknowledged=true")

    document, entry, ids_by_dataset = _split_inputs(manifest, split)
    resolved_variant_file = _validate_variant_file(variant_file, ids_by_dataset=ids_by_dataset)
    manifest_identity = document["source"]["content_sha256"]
    return [
        stage1_iid_biased(
            frozen_file=entry["path"],
            dataset=dataset,
            split=split,
            unbiased_log=unbiased_log,
            question_ids_from=ids_by_dataset[dataset],
            variant_file=resolved_variant_file,
            prompt_style=prompt_style,
            bias_type=BIAS_TYPE,
            include_bias_acknowledged=include_bias_acknowledged,
            grader_model=grader_model,
            source_sha256=manifest_identity,
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
    """Return both splits while loading a checkpoint only once.

    All four clean tasks precede all four biased tasks so the local switch
    scorer can resolve each paired clean result from ``unbiased_log``.  The
    question IDs are disjoint across splits, which keeps pairing unambiguous.
    """

    clean: list[Task] = []
    biased: list[Task] = []
    for split in ("train_eval", "heldout_in_domain"):
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


__all__ = [
    "diagnostic_matrix_tasks",
    "diagnostic_biased_tasks",
    "diagnostic_tasks",
    "stage1_iid_biased",
    "stage1_iid_unbiased",
]
