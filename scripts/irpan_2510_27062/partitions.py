"""Closed artifact roles and deterministic paper-dataset partitions.

The paper names the datasets and their broad uses, but it does not publish an
exact HarmBench train/validation split.  That split is therefore a checked-in
reconstruction: stable example IDs are assigned by one fixed hash rule.  The
registry also makes the model-selection boundary explicit. MMLU has separate
validation and final-evaluation artifacts so the harmonic-mean selection route
cannot consume the final reporting rows.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

from scripts.irpan_2510_27062.schema import normalize_text, sha256_json

TRAINING = "training"
VALIDATION = "validation"
FINAL_EVAL = "final_eval"

ArtifactRole: TypeAlias = Literal["training", "validation", "final_eval"]
ARTIFACT_ROLES = frozenset({TRAINING, VALIDATION, FINAL_EVAL})

PAPER_REPORTED = "paper-reported"
PAPER_UNSPECIFIED_RECONSTRUCTION = "paper-unspecified reconstruction"


class PartitionError(ValueError):
    """A source partition, artifact role, or partition ID set is invalid."""


@dataclass(frozen=True, slots=True)
class PartitionSpec:
    """One source route and its immutable artifact-role boundary."""

    source: str
    partition: ArtifactRole
    role: ArtifactRole
    paper_route: str
    paper_status: str
    notes: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _spec(
    source: str,
    role: ArtifactRole,
    paper_route: str,
    *,
    paper_status: str = PAPER_REPORTED,
    notes: str = "",
) -> PartitionSpec:
    return PartitionSpec(
        source=source,
        partition=role,
        role=role,
        paper_route=paper_route,
        paper_status=paper_status,
        notes=notes,
    )


_PARTITIONS: dict[str, dict[str, PartitionSpec]] = {
    "arc": {TRAINING: _spec("arc", TRAINING, "sycophancy_train")},
    "openbookqa": {TRAINING: _spec("openbookqa", TRAINING, "sycophancy_train")},
    "bbh": {TRAINING: _spec("bbh", TRAINING, "sycophancy_train")},
    "mmlu": {
        VALIDATION: _spec(
            "mmlu",
            VALIDATION,
            "sycophancy_validation_helpfulness_and_resistance",
            paper_status=PAPER_UNSPECIFIED_RECONSTRUCTION,
            notes="The paper reports MMLU-based selection but does not publish the exact validation membership.",
        ),
        FINAL_EVAL: _spec(
            "mmlu",
            FINAL_EVAL,
            "sycophancy_held_out_reporting",
            notes="Held-out clean and wrong-suggestion reporting; disjoint from the validation artifact.",
        ),
    },
    "harmbench": {
        TRAINING: _spec(
            "harmbench",
            TRAINING,
            "jailbreak_train",
            paper_status=PAPER_UNSPECIFIED_RECONSTRUCTION,
            notes="Exact train/validation membership is reconstructed by the fixed stable-ID hash rule.",
        ),
        VALIDATION: _spec(
            "harmbench",
            VALIDATION,
            "jailbreak_validation_safety",
            paper_status=PAPER_UNSPECIFIED_RECONSTRUCTION,
            notes="Exact train/validation membership is reconstructed by the fixed stable-ID hash rule.",
        ),
    },
    "or_bench": {VALIDATION: _spec("or_bench", VALIDATION, "jailbreak_validation_helpfulness")},
    "clearharm": {FINAL_EVAL: _spec("clearharm", FINAL_EVAL, "jailbreak_final_safety")},
    "wildguardtest": {FINAL_EVAL: _spec("wildguardtest", FINAL_EVAL, "jailbreak_final_safety")},
    "xstest": {FINAL_EVAL: _spec("xstest", FINAL_EVAL, "jailbreak_final_helpfulness")},
    "wildjailbreak": {FINAL_EVAL: _spec("wildjailbreak", FINAL_EVAL, "jailbreak_final_helpfulness")},
}

PARTITION_REGISTRY: Mapping[str, Mapping[str, PartitionSpec]] = MappingProxyType(
    {source: MappingProxyType(partitions) for source, partitions in _PARTITIONS.items()}
)

HARM_BENCH_PARTITION_NAMESPACE = "irpan_2510_27062:harmbench_train_validation:reconstruction_v1"
HARM_BENCH_PARTITION_SEED = 251_027_063
HARM_BENCH_PARTITION_MODULUS = 5
HARM_BENCH_VALIDATION_BUCKETS = (0,)
HARM_BENCH_PARTITION_RULE = (
    "sha256(canonical_json({namespace, seed, example_id})) modulo 5; "
    "bucket 0 is validation and buckets 1-4 are training"
)

HARM_BENCH_IDENTITY_FIELD = "BehaviorID"
SOURCE_IDENTITY_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # Generic export fields such as ``id`` are deliberately excluded.  A
        # HarmBench behavior must hash identically in the training and
        # validation loaders even when an export happens to carry both.
        "harmbench": (HARM_BENCH_IDENTITY_FIELD, "behavior_id"),
        "or_bench": ("id", "ID", "prompt_id", "index"),
        "clearharm": ("id", "ID", "prompt_id", "index"),
        "wildguardtest": ("id", "ID", "prompt_id", "index"),
        "xstest": ("id", "ID", "prompt_id", "index"),
        "wildjailbreak": ("id", "ID", "prompt_id", "index"),
    }
)


def extract_source_identity(
    source: str,
    row: Mapping[str, Any],
    *,
    prompt: str | None = None,
    allow_prompt_fallback: bool = False,
) -> tuple[str, str]:
    """Return one source-specific stable identity and the field that supplied it.

    The checked-in field order defines priority for each source.  HarmBench is
    intentionally stricter: its spelling-compatible BehaviorID aliases must
    agree, and generic ``id`` fields are never considered.  This single
    extractor is shared by the train and evaluation normalizers.
    """

    if not isinstance(source, str) or source not in SOURCE_IDENTITY_FIELDS:
        raise PartitionError(f"unknown identity source {source!r}; choose one of {sorted(SOURCE_IDENTITY_FIELDS)}")
    if not isinstance(row, Mapping):
        raise PartitionError(f"{source} source row must be an object")

    present: list[tuple[str, str]] = []
    for field in SOURCE_IDENTITY_FIELDS[source]:
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            continue
        normalized = normalize_text(str(value))
        if normalized:
            present.append((field, normalized))
    distinct = {value for _field, value in present}
    if source == "harmbench" and len(distinct) > 1:
        details = {field: value for field, value in present}
        raise PartitionError(f"conflicting {source} stable identity fields: {details}")
    if present:
        # Field order is source-specific and stable; aliases that agree do not
        # change the selected identity.
        return present[0][1], present[0][0]

    if source == "harmbench" and not allow_prompt_fallback:
        raise PartitionError(
            "HarmBench rows require a non-empty BehaviorID identity; generic id fields are not partition keys"
        )
    if not allow_prompt_fallback:
        raise PartitionError(f"{source} row has no recognized stable identity field")
    if not isinstance(prompt, str) or not normalize_text(prompt):
        raise PartitionError(f"{source} identity fallback requires a non-empty normalized prompt")
    prompt_identity = "prompt-sha256:" + sha256(normalize_text(prompt).encode("utf-8")).hexdigest()
    return prompt_identity, "prompt_sha256"


def stable_source_identity(
    source: str,
    row: Mapping[str, Any],
    *,
    prompt: str | None = None,
    allow_prompt_fallback: bool = False,
) -> str:
    """Return only the stable identity from :func:`extract_source_identity`."""

    identity, _field = extract_source_identity(
        source,
        row,
        prompt=prompt,
        allow_prompt_fallback=allow_prompt_fallback,
    )
    return identity


def require_artifact_role(role: str) -> ArtifactRole:
    """Return a closed-vocabulary role or fail without coercion."""

    if not isinstance(role, str) or role not in ARTIFACT_ROLES:
        raise PartitionError(f"unknown artifact role {role!r}; choose one of {sorted(ARTIFACT_ROLES)}")
    return role  # type: ignore[return-value]


def require_partition(
    source: str,
    partition: str | None = None,
    *,
    role: str | None = None,
) -> PartitionSpec:
    """Resolve one configured source partition and reject ambiguity/conflicts.

    ``partition`` and ``role`` may both be supplied, but they must name the same
    registered boundary.  HarmBench deliberately cannot be resolved without an
    explicit partition or role because it has both training and validation
    routes.
    """

    if not isinstance(source, str) or source not in PARTITION_REGISTRY:
        raise PartitionError(f"unknown partition source {source!r}; choose one of {sorted(PARTITION_REGISTRY)}")
    source_partitions = PARTITION_REGISTRY[source]
    resolved_role = require_artifact_role(role) if role is not None else None
    if partition is not None:
        if not isinstance(partition, str) or partition not in source_partitions:
            raise PartitionError(
                f"unknown partition {partition!r} for {source!r}; choose one of {sorted(source_partitions)}"
            )
        spec = source_partitions[partition]
        if resolved_role is not None and spec.role != resolved_role:
            raise PartitionError(
                f"configured partition {partition!r} for {source!r} has role {spec.role!r}, not {resolved_role!r}"
            )
        return spec
    if resolved_role is not None:
        matches = [spec for spec in source_partitions.values() if spec.role == resolved_role]
        if len(matches) != 1:
            raise PartitionError(f"{source!r} has no registered partition for role {resolved_role!r}")
        return matches[0]
    if len(source_partitions) != 1:
        raise PartitionError(
            f"{source!r} has multiple registered partitions; configure one of {sorted(source_partitions)}"
        )
    return next(iter(source_partitions.values()))


def artifact_role_for_source(source: str, partition: str | None = None) -> ArtifactRole:
    """Return the registered role for an unambiguous/configured source route."""

    return require_partition(source, partition).role


def partition_registry_payload() -> dict[str, dict[str, dict[str, str]]]:
    """Return a JSON-safe snapshot of every checked-in source partition."""

    return {
        source: {partition: spec.as_dict() for partition, spec in partitions.items()}
        for source, partitions in PARTITION_REGISTRY.items()
    }


def assign_harmbench_partition(
    example_id: str,
    *,
    configured_partition: str | None = None,
    configured_role: str | None = None,
) -> ArtifactRole:
    """Assign one stable example ID by the fixed reconstruction rule.

    When a caller already configured a partition or role, this function treats
    that declaration as a constraint.  It raises on conflict instead of
    silently repartitioning the example.
    """

    stable_id = _stable_example_id(example_id)
    digest = sha256_json(
        {
            "namespace": HARM_BENCH_PARTITION_NAMESPACE,
            "seed": HARM_BENCH_PARTITION_SEED,
            "example_id": stable_id,
        }
    )
    bucket = int(digest, 16) % HARM_BENCH_PARTITION_MODULUS
    assigned: ArtifactRole = VALIDATION if bucket in HARM_BENCH_VALIDATION_BUCKETS else TRAINING
    if configured_partition is not None or configured_role is not None:
        configured = require_partition(
            "harmbench",
            configured_partition,
            role=configured_role,
        )
        if configured.partition != assigned:
            raise PartitionError(
                f"HarmBench example {stable_id!r} hashes to {assigned!r}, "
                f"conflicting with configured partition {configured.partition!r}"
            )
    return assigned


def partition_harmbench_ids(
    example_ids: Iterable[str],
    *,
    configured_partition: str | None = None,
    configured_role: str | None = None,
) -> dict[str, tuple[str, ...]]:
    """Partition stable IDs into sorted, duplicate-free training/validation sets."""

    buckets: dict[str, list[str]] = {TRAINING: [], VALIDATION: []}
    seen: set[str] = set()
    for raw_id in example_ids:
        stable_id = _stable_example_id(raw_id)
        if stable_id in seen:
            raise PartitionError(f"duplicate HarmBench example ID {stable_id!r}")
        seen.add(stable_id)
        assigned = assign_harmbench_partition(
            stable_id,
            configured_partition=configured_partition,
            configured_role=configured_role,
        )
        buckets[assigned].append(stable_id)
    result = {partition: tuple(sorted(ids)) for partition, ids in buckets.items()}
    verify_disjoint_ids(result[TRAINING], result[VALIDATION])
    return result


def verify_disjoint_ids(training_ids: Iterable[str], validation_ids: Iterable[str]) -> None:
    """Reject duplicates within or overlap across explicit partition ID sets."""

    training = _validated_id_set(training_ids, partition=TRAINING)
    validation = _validated_id_set(validation_ids, partition=VALIDATION)
    overlap = sorted(training & validation)
    if overlap:
        raise PartitionError(f"training and validation example IDs overlap: {overlap}")


def harmbench_partition_provenance(partition: str) -> dict[str, object]:
    """Return the exact reconstruction contract to record in artifact provenance."""

    spec = require_partition("harmbench", partition)
    return {
        "source": "harmbench",
        "partition": spec.partition,
        "role": spec.role,
        "paper_status": spec.paper_status,
        "namespace": HARM_BENCH_PARTITION_NAMESPACE,
        "seed": HARM_BENCH_PARTITION_SEED,
        "rule": HARM_BENCH_PARTITION_RULE,
        "modulus": HARM_BENCH_PARTITION_MODULUS,
        "validation_buckets": list(HARM_BENCH_VALIDATION_BUCKETS),
    }


def _stable_example_id(value: str) -> str:
    if not isinstance(value, str) or not normalize_text(value):
        raise PartitionError("example IDs must be non-empty strings")
    return normalize_text(value)


def _validated_id_set(values: Iterable[str], *, partition: str) -> set[str]:
    ids: list[str] = [_stable_example_id(value) for value in values]
    if len(ids) != len(set(ids)):
        raise PartitionError(f"{partition} example IDs contain duplicates")
    return set(ids)


__all__ = [
    "ARTIFACT_ROLES",
    "FINAL_EVAL",
    "HARM_BENCH_IDENTITY_FIELD",
    "HARM_BENCH_PARTITION_MODULUS",
    "HARM_BENCH_PARTITION_NAMESPACE",
    "HARM_BENCH_PARTITION_RULE",
    "HARM_BENCH_PARTITION_SEED",
    "HARM_BENCH_VALIDATION_BUCKETS",
    "PAPER_REPORTED",
    "PAPER_UNSPECIFIED_RECONSTRUCTION",
    "PARTITION_REGISTRY",
    "SOURCE_IDENTITY_FIELDS",
    "TRAINING",
    "VALIDATION",
    "ArtifactRole",
    "PartitionError",
    "PartitionSpec",
    "artifact_role_for_source",
    "assign_harmbench_partition",
    "extract_source_identity",
    "harmbench_partition_provenance",
    "partition_harmbench_ids",
    "partition_registry_payload",
    "require_artifact_role",
    "require_partition",
    "stable_source_identity",
    "verify_disjoint_ids",
]
