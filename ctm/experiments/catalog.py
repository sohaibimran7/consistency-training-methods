"""A small validated index for CTM experiment provenance.

The catalogue records declared locations; it never scans a repository, walks
archived environments, contacts remote storage, or infers completion from a
path. Pydantic models are the single source of truth for both validation and
the generated JSON Schema. Cross-entry lineage remains a small semantic check.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, get_args
from urllib.parse import urlparse

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _stable_identifier(value: str) -> str:
    _non_blank(value)
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise ValueError("must use lowercase letters, digits, '.', '_' or '-' and start with a letter or digit")
    return value


def _relative_repo_path(value: str) -> str:
    _non_blank(value)
    if value.startswith("/") or "\\" in value:
        raise ValueError("must be a relative POSIX repository path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("must not contain empty, '.' or '..' path segments")
    return value


def _external_uri(value: str) -> str:
    _non_blank(value)
    if "\\" in value or re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError("must be an explicit URI, not a local filesystem path")
    parsed = urlparse(value)
    if not parsed.scheme or (not parsed.netloc and not parsed.path):
        raise ValueError("must be an explicit URI with a scheme")
    return value


NonEmptyText: TypeAlias = Annotated[str, Field(min_length=1), AfterValidator(_non_blank)]
Identifier: TypeAlias = Annotated[
    str,
    Field(pattern=_IDENTIFIER_RE.pattern),
    AfterValidator(_stable_identifier),
]
Sha256Digest: TypeAlias = Annotated[str, Field(pattern=_SHA256_RE.pattern)]
RepositoryPath: TypeAlias = Annotated[str, Field(min_length=1), AfterValidator(_relative_repo_path)]
ExternalURI: TypeAlias = Annotated[str, Field(min_length=1), AfterValidator(_external_uri)]

SchemaVersion: TypeAlias = Literal[1]
ExperimentStatus: TypeAlias = Literal[
    "planned",
    "active",
    "complete",
    "diagnostic_only",
    "invalidated",
    "superseded",
]
LocationAvailability: TypeAlias = Literal["available", "missing", "unreachable", "unknown"]

CATALOG_SCHEMA_VERSION = get_args(SchemaVersion)[0]
"""The version accepted by :class:`ExperimentCatalog`."""

EXPERIMENT_STATUSES = frozenset(get_args(ExperimentStatus))
LOCATION_AVAILABILITY = frozenset(get_args(LocationAvailability))
_RESULT_ARTIFACT_ROLES = frozenset({"result", "accepted_result"})


class CatalogError(ValueError):
    """Base error for catalogue loading, validation, and lookup."""


class CatalogValidationError(CatalogError):
    """A catalogue document does not satisfy the versioned contract."""


class ExperimentNotFoundError(CatalogError):
    """No experiment has the requested stable identifier."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Location(_StrictModel):
    availability: LocationAvailability
    availability_note: NonEmptyText | None = None

    @model_validator(mode="after")
    def _explain_unavailable_location(self) -> _Location:
        if self.availability != "available" and self.availability_note is None:
            raise ValueError("availability_note is required unless availability is 'available'")
        return self


class RepositoryLocation(_Location):
    """A path to a checked-in file, always relative to the repository root."""

    kind: Literal["repo_path"]
    path: RepositoryPath


class ExternalLocation(_Location):
    """An explicit URI outside the repository, with recorded availability."""

    kind: Literal["external"]
    uri: ExternalURI


CatalogLocation: TypeAlias = Annotated[RepositoryLocation | ExternalLocation, Field(discriminator="kind")]


class CatalogReference(_StrictModel):
    """A named protocol, source, environment, task, or artifact reference."""

    role: Identifier
    location: CatalogLocation
    sha256: Sha256Digest | None = None
    revision: NonEmptyText | None = None
    note: NonEmptyText | None = None


class Lineage(_StrictModel):
    supersedes: list[Identifier]
    derived_from: list[Identifier]

    @field_validator("supersedes", "derived_from")
    @classmethod
    def _unique_identifiers(cls, identifiers: list[str]) -> list[str]:
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("must not contain duplicate ids")
        return identifiers

    @model_validator(mode="after")
    def _disjoint_relationships(self) -> Lineage:
        overlap = set(self.supersedes).intersection(self.derived_from)
        if overlap:
            raise ValueError(f"cannot both supersede and derive from {sorted(overlap)!r}")
        return self


class ExperimentRecord(_StrictModel):
    """One stable experiment identity and its recorded provenance."""

    id: Identifier
    question: NonEmptyText
    status: ExperimentStatus
    protocol_refs: list[CatalogReference]
    source_snapshots: list[CatalogReference]
    environment_refs: list[CatalogReference]
    artifacts: list[CatalogReference]
    lineage: Lineage
    owner: NonEmptyText
    current_task_refs: list[CatalogReference]
    completion_criteria: list[NonEmptyText]
    evidence_notes: list[NonEmptyText]

    @model_validator(mode="after")
    def _complete_has_recorded_evidence(self) -> ExperimentRecord:
        if self.status != "complete":
            return self
        missing = [
            name
            for name, records in (
                ("protocol_refs", self.protocol_refs),
                ("source_snapshots", self.source_snapshots),
                ("environment_refs", self.environment_refs),
                ("completion_criteria", self.completion_criteria),
                ("evidence_notes", self.evidence_notes),
            )
            if not records
        ]
        if missing:
            fields = ", ".join(f"{name} must not be empty" for name in missing)
            raise ValueError(f"complete entries require non-empty fields: {fields}")
        if not any(artifact.role in _RESULT_ARTIFACT_ROLES for artifact in self.artifacts):
            allowed = ", ".join(sorted(_RESULT_ARTIFACT_ROLES))
            raise ValueError(f"complete entries require an artifact role of: {allowed}")
        return self


class ExperimentCatalog(_StrictModel):
    """The JSON document root and cross-record lineage validation."""

    schema_version: SchemaVersion
    experiments: list[ExperimentRecord]

    @model_validator(mode="after")
    def _valid_lineage(self) -> ExperimentCatalog:
        by_id: dict[str, ExperimentRecord] = {}
        for experiment in self.experiments:
            if experiment.id in by_id:
                raise ValueError(f"duplicate id {experiment.id!r}")
            by_id[experiment.id] = experiment

        graph: dict[str, set[str]] = {identifier: set() for identifier in by_id}
        superseded_by: dict[str, set[str]] = {identifier: set() for identifier in by_id}
        for identifier, experiment in by_id.items():
            for relation, targets in (
                ("supersedes", experiment.lineage.supersedes),
                ("derived_from", experiment.lineage.derived_from),
            ):
                for target in targets:
                    if target == identifier:
                        raise ValueError(f"experiment {identifier!r} cannot {relation} itself")
                    if target not in by_id:
                        raise ValueError(f"experiment {identifier!r} {relation} unknown experiment {target!r}")
                    graph[identifier].add(target)
                    if relation == "supersedes":
                        superseded_by[target].add(identifier)

        for identifier, experiment in by_id.items():
            if experiment.status == "superseded" and not superseded_by[identifier]:
                raise ValueError(f"experiment {identifier!r} is superseded but has no recorded successor")

        visiting: list[str] = []
        visited: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in visited:
                return
            if identifier in visiting:
                start = visiting.index(identifier)
                cycle = " -> ".join([*visiting[start:], identifier])
                raise ValueError(f"lineage contains a cycle: {cycle}")
            visiting.append(identifier)
            for target in sorted(graph[identifier]):
                visit(target)
            visiting.pop()
            visited.add(identifier)

        for identifier in sorted(graph):
            visit(identifier)
        return self


CatalogInput: TypeAlias = Mapping[str, Any] | ExperimentCatalog


def catalog_json_schema() -> dict[str, Any]:
    """Return JSON Schema generated from the same Pydantic models used to validate."""

    return ExperimentCatalog.model_json_schema()


def validate_catalog(document: CatalogInput) -> ExperimentCatalog:
    """Validate a parsed catalogue offline and return its typed model."""

    if isinstance(document, ExperimentCatalog):
        return document
    try:
        return ExperimentCatalog.model_validate(document)
    except ValidationError as exc:
        raise CatalogValidationError(_format_validation_error(exc)) from exc


def load_catalog(path: str | Path) -> ExperimentCatalog:
    """Read and validate one JSON catalogue document from *path*."""

    target = Path(path)
    try:
        payload = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogValidationError(f"cannot read catalogue {target}: {exc}") from exc
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CatalogValidationError(f"invalid JSON in catalogue {target}: {exc}") from exc
    return validate_catalog(document)


def list_experiments(catalog: CatalogInput) -> tuple[ExperimentRecord, ...]:
    """Return validated experiments in their recorded order."""

    return tuple(validate_catalog(catalog).experiments)


def find_experiment(catalog: CatalogInput, experiment_id: str) -> ExperimentRecord:
    """Return one validated experiment by its stable identifier."""

    if not isinstance(experiment_id, str) or not experiment_id:
        raise ExperimentNotFoundError("experiment id must be a non-empty string")
    for experiment in validate_catalog(catalog).experiments:
        if experiment.id == experiment_id:
            return experiment
    raise ExperimentNotFoundError(f"no experiment with id {experiment_id!r}")


def render_catalog_markdown(catalog: CatalogInput) -> str:
    """Render every validated entry as readable Markdown, including locations."""

    validated = validate_catalog(catalog)
    lines = [
        "# Experiment catalogue",
        "",
        f"Schema version: {validated.schema_version}",
        "",
        f"Recorded experiments: {len(validated.experiments)}",
    ]
    if validated.experiments:
        lines.extend(["", "| ID | Status | Question |", "| --- | --- | --- |"])
        for experiment in validated.experiments:
            lines.append(
                "| "
                + " | ".join(
                    (
                        _table_cell(experiment.id),
                        _table_cell(experiment.status),
                        _table_cell(experiment.question),
                    )
                )
                + " |"
            )
        for experiment in validated.experiments:
            lines.extend(["", _render_experiment_markdown(experiment, heading_level=2)])
    return "\n".join(lines).rstrip() + "\n"


def render_experiment_markdown(catalog: CatalogInput, experiment_id: str) -> str:
    """Render one experiment, including every declared exact location."""

    return _render_experiment_markdown(find_experiment(catalog, experiment_id), heading_level=1) + "\n"


def _render_experiment_markdown(experiment: ExperimentRecord, *, heading_level: int) -> str:
    heading = "#" * heading_level
    subgroup_heading = "#" * (heading_level + 1)
    lines = [
        f"{heading} `{experiment.id}`",
        "",
        f"- **Question:** {_markdown_text(experiment.question)}",
        f"- **Recorded status:** `{experiment.status}`",
        f"- **Owner:** {_markdown_text(experiment.owner)}",
        "",
        f"{subgroup_heading} Protocol references",
        *_render_references(experiment.protocol_refs),
        "",
        f"{subgroup_heading} Source snapshots",
        *_render_references(experiment.source_snapshots),
        "",
        f"{subgroup_heading} Environment references",
        *_render_references(experiment.environment_refs),
        "",
        f"{subgroup_heading} Artifacts",
        *_render_references(experiment.artifacts),
        "",
        f"{subgroup_heading} Lineage",
        f"- **Supersedes:** {_render_identifiers(experiment.lineage.supersedes)}",
        f"- **Derived from:** {_render_identifiers(experiment.lineage.derived_from)}",
        "",
        f"{subgroup_heading} Current task references",
        *_render_references(experiment.current_task_refs),
        "",
        f"{subgroup_heading} Completion criteria",
        *_render_text_list(experiment.completion_criteria),
        "",
        f"{subgroup_heading} Evidence notes",
        *_render_text_list(experiment.evidence_notes),
    ]
    return "\n".join(lines)


def _render_references(references: list[CatalogReference]) -> list[str]:
    if not references:
        return ["- None recorded."]
    lines: list[str] = []
    for reference in references:
        details = [_render_location(reference.location)]
        if reference.revision is not None:
            details.append(f"revision `{_inline_code(reference.revision)}`")
        if reference.sha256 is not None:
            details.append(f"SHA-256 `{reference.sha256}`")
        if reference.note is not None:
            details.append(_markdown_text(reference.note))
        lines.append(f"- `{reference.role}`: " + "; ".join(details))
    return lines


def _render_location(location: CatalogLocation) -> str:
    if location.kind == "repo_path":
        raw_location = location.path
        kind = "repository path"
    else:
        raw_location = location.uri
        kind = "external URI"
    detail = f"{kind}; recorded availability: {location.availability}"
    if location.availability_note is not None:
        detail += f" ({_markdown_text(location.availability_note)})"
    return f"`{_inline_code(raw_location)}` ({detail})"


def _render_identifiers(identifiers: list[str]) -> str:
    return ", ".join(f"`{identifier}`" for identifier in identifiers) if identifiers else "None recorded."


def _render_text_list(items: list[str]) -> list[str]:
    return [f"- {_markdown_text(item)}" for item in items] if items else ["- None recorded."]


def _format_validation_error(error: ValidationError) -> str:
    messages = []
    for item in error.errors(include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "catalogue"
        messages.append(f"{location}: {item['msg']}")
    return "; ".join(messages)


def _table_cell(value: str) -> str:
    return " ".join(value.split()).replace("|", "\\|")


def _markdown_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")


def _inline_code(value: str) -> str:
    return value.replace("`", "\\`")


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "EXPERIMENT_STATUSES",
    "LOCATION_AVAILABILITY",
    "CatalogError",
    "CatalogLocation",
    "CatalogReference",
    "CatalogValidationError",
    "ExperimentCatalog",
    "ExperimentNotFoundError",
    "ExperimentRecord",
    "ExternalLocation",
    "Lineage",
    "RepositoryLocation",
    "catalog_json_schema",
    "find_experiment",
    "list_experiments",
    "load_catalog",
    "render_catalog_markdown",
    "render_experiment_markdown",
    "validate_catalog",
]
