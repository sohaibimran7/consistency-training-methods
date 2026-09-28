"""Inspect task factories for the immutable Stage 2 HLE 2×2 OOD suite.

The factory deliberately constructs a *fresh, fully paired* four-cell matrix.
Each population has one clean generation shared by all six bias variants, so
the IID and held-out-bias columns differ only in the prompt intervention—not
in which sampled clean answer their switch metric is conditioned on.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inspect_ai import Task, task

from ctm_data.adapters.mcq_bias.scorer_compat import install_conditional_nan_compat
from experiments.stage2_ood_hle.materialize import (
    HELDOUT_BIAS,
    HELDOUT_BIASES,
    HELDOUT_DATASET,
    HELDOUT_DATASET_AND_BIAS,
    IID,
    IN_DOMAIN_DATASETS,
    PROMPT_STYLE,
    REGIMES,
    TRAINING_BIAS,
    validate_manifest,
)


@dataclass(frozen=True, slots=True)
class OODTaskSpec:
    """One exact task cell, including the clean population identity it uses."""

    kind: str
    regime: str
    population: str
    dataset: str
    bias_type: str | None
    frozen_file: str
    question_ids: tuple[str, ...]
    source_identity_digest: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_artifact(entry: Mapping[str, Any], *, label: str) -> tuple[Path, list[dict[str, Any]]]:
    """Reload an already-validated frozen artifact without a TOCTOU gap."""

    raw_path = entry.get("path")
    expected_digest = entry.get("content_sha256")
    expected_ids = entry.get("question_ids")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}: manifest artifact has no path")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError(f"{label}: manifest artifact has an invalid SHA-256")
    if not isinstance(expected_ids, list) or not all(isinstance(item, str) and item for item in expected_ids):
        raise ValueError(f"{label}: manifest artifact has invalid question IDs")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: frozen artifact does not exist: {path}")
    payload = path.read_bytes()
    if _sha256(payload) != expected_digest:
        raise ValueError(f"{label}: frozen artifact bytes changed after manifest validation")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{label}: blank row {line_number} is not permitted")
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{label}: invalid JSON at row {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label}: row {line_number} must be a JSON object")
        rows.append(row)
    ids = [row.get("question_id") for row in rows]
    if ids != expected_ids or len(ids) != len(set(ids)):
        raise ValueError(f"{label}: rows no longer match the manifest question-ID identity")
    return path, rows


def _restrict_task(
    task_object: Task,
    *,
    dataset: str,
    regime: str,
    prompt_style: str,
    bias_type: str | None,
    question_ids: Sequence[str],
) -> Task:
    """Restrict a mixed frozen file to one exact dataset-level task cell."""

    requested = [str(question_id) for question_id in question_ids]
    wanted = set(requested)
    if not wanted or len(wanted) != len(requested):
        raise ValueError(f"invalid OOD task IDs for {regime}/{dataset}")
    selected = task_object.dataset.filter(
        lambda sample: (
            bool(sample.metadata)
            and sample.metadata.get("source_dataset") == dataset
            and str(sample.id) in wanted
        ),
        name=f"stage2-ood-{regime}-{dataset}-{bias_type or 'unbiased'}",
    )
    observed: set[str] = set()
    for sample in selected:
        metadata = sample.metadata or {}
        sample_id = str(sample.id)
        if sample_id in observed:
            raise ValueError(f"OOD task has duplicate requested question_id {sample_id!r}")
        observed.add(sample_id)
        if metadata.get("source_dataset") != dataset or sample_id not in wanted:
            raise ValueError(f"OOD task sample {sample_id!r} conflicts with its frozen selection")
        if metadata.get("prompt_style") != prompt_style:
            raise ValueError(f"OOD task sample {sample_id!r} has a conflicting prompt style")
        if bias_type is not None and metadata.get("bias_type") != bias_type:
            raise ValueError(f"OOD task sample {sample_id!r} has a conflicting bias type")
    if observed != wanted:
        missing = sorted(wanted - observed)
        unexpected = sorted(observed - wanted)
        raise ValueError(
            f"OOD task {regime}/{dataset} does not exactly match its frozen IDs: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    task_object.dataset = selected
    return task_object


@task
def stage2_ood_unbiased(
    frozen_file: str,
    dataset: str,
    regime: str,
    population: str,
    question_ids_from: list[str],
    source_identity_digest: str,
    prompt_style: str = PROMPT_STYLE,
) -> Task:
    """Generate one population's clean answers for all its bias variants."""

    if prompt_style != PROMPT_STYLE:
        raise ValueError(f"Stage 2 OOD requires prompt_style={PROMPT_STYLE!r}")
    install_conditional_nan_compat()
    from mcq_bias.tasks import unbiased_task_from_frozen

    path = Path(frozen_file).resolve()
    task_object = unbiased_task_from_frozen(
        path,
        metadata={
            "dataset": dataset,
            "source_dataset": dataset,
            "regime": regime,
            "population": population,
            "bias_type": None,
            "prompt_style": prompt_style,
            "dataset_file": str(path),
            "source_identity_digest": source_identity_digest,
            "question_ids_from": question_ids_from,
        },
    )
    return _restrict_task(
        task_object,
        dataset=dataset,
        regime=regime,
        prompt_style=prompt_style,
        bias_type=None,
        question_ids=question_ids_from,
    )


@task
def stage2_ood_biased(
    frozen_file: str,
    dataset: str,
    regime: str,
    population: str,
    bias_type: str,
    unbiased_log: str,
    question_ids_from: list[str],
    source_identity_digest: str,
    prompt_style: str = PROMPT_STYLE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
) -> Task:
    """Generate one frozen biased prompt variant and attach paired switch scoring."""

    if prompt_style != PROMPT_STYLE:
        raise ValueError(f"Stage 2 OOD requires prompt_style={PROMPT_STYLE!r}")
    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    if grader_model is not None and not include_bias_acknowledged:
        raise ValueError("grader_model requires include_bias_acknowledged=true")
    install_conditional_nan_compat()
    from mcq_bias.tasks import task_from_frozen

    path = Path(frozen_file).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen OOD biased artifact does not exist: {path}")
    kwargs: dict[str, Any] = {
        "metadata": {
            "dataset": dataset,
            "source_dataset": dataset,
            "regime": regime,
            "population": population,
            "bias_type": bias_type,
            "prompt_style": prompt_style,
            "dataset_file": str(path),
            "unbiased_log": unbiased_log,
            "include_bias_acknowledged": include_bias_acknowledged,
            "grader_model": grader_model,
            "source_identity_digest": source_identity_digest,
            "question_ids_from": question_ids_from,
        },
        "unbiased_log": unbiased_log,
        "grader_model": grader_model,
        "include_bias_acknowledged": include_bias_acknowledged,
        "question_ids_from": question_ids_from,
    }
    # The pinned mcq-bias package supports these stronger directory-matching
    # arguments; retain compatibility with its older public signature too.
    parameters = inspect.signature(task_from_frozen).parameters
    if "source_dataset" in parameters:
        kwargs["source_dataset"] = dataset
    if "source_identity_digest" in parameters:
        kwargs["source_identity_digest"] = source_identity_digest
    task_object = task_from_frozen(path, **kwargs)
    return _restrict_task(
        task_object,
        dataset=dataset,
        regime=regime,
        prompt_style=prompt_style,
        bias_type=bias_type,
        question_ids=question_ids_from,
    )


def _population_specs(
    document: Mapping[str, Any],
    *,
    population: str,
    regime: str,
    bias_type: str | None,
    kind: str,
) -> list[OODTaskSpec]:
    populations = document["populations"]
    population_entry = populations[population]
    artifacts = population_entry["artifacts"]
    artifact_name = "unbiased" if bias_type is None else bias_type
    artifact_entry = artifacts[artifact_name]
    path, rows = _read_artifact(artifact_entry, label=f"{population}/{artifact_name}")
    clean_digest = artifacts["unbiased"]["content_sha256"]
    if not isinstance(clean_digest, str) or len(clean_digest) != 64:
        raise ValueError(f"{population}: missing clean population identity digest")
    source_identity = f"stage2-ood-hle-2x2:{clean_digest}"
    if population == "in_domain":
        expected_datasets = IN_DOMAIN_DATASETS
    elif population == "hle":
        dataset = population_entry.get("source_dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("HLE population has no source_dataset")
        expected_datasets = (dataset,)
    else:  # pragma: no cover - manifest validation prevents this
        raise ValueError(f"unknown OOD population {population!r}")

    specs: list[OODTaskSpec] = []
    for dataset in expected_datasets:
        ids = tuple(str(row.get("question_id", "")) for row in rows if row.get("source_dataset") == dataset)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{population}/{artifact_name}/{dataset}: invalid frozen question IDs")
        specs.append(
            OODTaskSpec(
                kind=kind,
                regime=regime,
                population=population,
                dataset=dataset,
                bias_type=bias_type,
                frozen_file=str(path),
                question_ids=ids,
                source_identity_digest=source_identity,
            )
        )
    return specs


def ood_task_specs(manifest: str | Path | Mapping[str, Any]) -> list[OODTaskSpec]:
    """Return the exact 3 clean + 18 biased tasks in stable execution order.

    The first three tasks form the two shared clean-reference populations.  A
    paired scorer may safely wait for these logs while the later biased tasks
    generate, but retaining this order makes launch manifests and preflight
    reports easy to audit.
    """

    if isinstance(manifest, Mapping):
        document = dict(manifest)
        # The validator deliberately requires a file-backed artifact; a
        # mapping is useful only after a caller has validated it separately.
        raise TypeError("OOD task specs require a file-backed immutable manifest")
    document = validate_manifest(manifest)
    # JSON manifests are written with sorted keys for byte-stable hashing, so
    # dictionary iteration is not the display/execution order.  The frozen
    # explicit list is the immutable order contract.
    if document.get("regime_order") != list(REGIMES):
        raise ValueError("Stage 2 OOD regimes are not in the required headline order")

    clean = [
        *_population_specs(document, population="in_domain", regime=IID, bias_type=None, kind="unbiased"),
        *_population_specs(
            document, population="hle", regime=HELDOUT_DATASET, bias_type=None, kind="unbiased"
        ),
    ]
    biased = [
        *_population_specs(
            document, population="in_domain", regime=IID, bias_type=TRAINING_BIAS, kind="biased"
        ),
        *_population_specs(
            document, population="hle", regime=HELDOUT_DATASET, bias_type=TRAINING_BIAS, kind="biased"
        ),
    ]
    for bias_type in HELDOUT_BIASES:
        biased.extend(
            _population_specs(
                document,
                population="in_domain",
                regime=HELDOUT_BIAS,
                bias_type=bias_type,
                kind="biased",
            )
        )
    for bias_type in HELDOUT_BIASES:
        biased.extend(
            _population_specs(
                document,
                population="hle",
                regime=HELDOUT_DATASET_AND_BIAS,
                bias_type=bias_type,
                kind="biased",
            )
        )
    specs = [*clean, *biased]
    if len(specs) != 21 or sum(spec.kind == "unbiased" for spec in specs) != 3:
        raise ValueError("Stage 2 OOD task matrix must contain exactly 3 clean and 18 biased tasks")
    return specs


def ood_tasks(
    manifest: str | Path,
    unbiased_log: str,
    prompt_style: str = PROMPT_STYLE,
    include_bias_acknowledged: bool = False,
    grader_model: str | None = None,
    hf_eos_only_no_token_cap: bool = False,
) -> list[Task]:
    """Build the full fresh 2×2 OOD evaluation matrix.

    All six variants in each population deliberately point to one shared
    ``unbiased_log`` root.  The upstream switch scorer waits for the matching
    clean task if necessary, which permits the model-generation jobs to be
    launched concurrently without sending anything to a model-based grader.
    """

    if prompt_style != PROMPT_STYLE:
        raise ValueError(f"Stage 2 OOD requires prompt_style={PROMPT_STYLE!r}")
    if not unbiased_log:
        raise ValueError("unbiased_log must be a non-empty path, directory, or glob")
    if grader_model is not None and not include_bias_acknowledged:
        raise ValueError("grader_model requires include_bias_acknowledged=true")
    if not isinstance(hf_eos_only_no_token_cap, bool):
        raise ValueError("hf_eos_only_no_token_cap must be boolean")
    if hf_eos_only_no_token_cap:
        from ctm.evals.hf_eos_only import install_native_hf_eos_only_sampling

        install_native_hf_eos_only_sampling()
    tasks: list[Task] = []
    for spec in ood_task_specs(manifest):
        args = {
            "frozen_file": spec.frozen_file,
            "dataset": spec.dataset,
            "regime": spec.regime,
            "population": spec.population,
            "question_ids_from": list(spec.question_ids),
            "source_identity_digest": spec.source_identity_digest,
            "prompt_style": prompt_style,
        }
        if spec.kind == "unbiased":
            tasks.append(stage2_ood_unbiased(**args))
        else:
            assert spec.bias_type is not None
            tasks.append(
                stage2_ood_biased(
                    **args,
                    bias_type=spec.bias_type,
                    unbiased_log=unbiased_log,
                    include_bias_acknowledged=include_bias_acknowledged,
                    grader_model=grader_model,
                )
            )
    return tasks


__all__ = [
    "OODTaskSpec",
    "ood_task_specs",
    "ood_tasks",
    "stage2_ood_biased",
    "stage2_ood_unbiased",
]
