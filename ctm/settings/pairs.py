"""Reusable Setting support for immutable reference/variant prompt pairs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ctm.artifacts import (
    artifact_identity,
    artifact_selection_identity,
    read_verified_jsonl_artifact,
)
from ctm.pairs import canonical_pair_row
from ctm.training.refusal import RefusalJudge
from ctm.training.refusal.judge import (
    DEFAULT_REFUSAL_MODEL,
    CompletionCallback,
    normalize_refusal_judge_options,
)

PAIR_ARTIFACT_SCHEMA = "ctm.prompt_pairs"
PAIR_ARTIFACT_SCHEMA_VERSION = 1


class PairArtifactError(ValueError):
    """A prompt-pair artifact or its selected prefix is malformed."""


def load_pair_artifact(
    path: str | Path,
    *,
    n_datapoints: int | None = None,
    expected_schema: str = PAIR_ARTIFACT_SCHEMA,
    expected_schema_version: int = PAIR_ARTIFACT_SCHEMA_VERSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and load a deterministic prefix of a prompt-pair artifact."""

    if n_datapoints is not None and (
        not isinstance(n_datapoints, int) or isinstance(n_datapoints, bool) or n_datapoints < 1
    ):
        raise PairArtifactError("n_datapoints must be a positive integer or None")
    target = Path(path)
    decoded_rows, manifest = read_verified_jsonl_artifact(
        target,
        expected_schema=expected_schema,
        expected_schema_version=expected_schema_version,
        row_validator=canonical_pair_row,
    )
    rows: list[dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for line_number, row in enumerate(decoded_rows, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise PairArtifactError(f"{target}:{line_number}: pair_id must be a non-empty string")
        if pair_id in seen_pair_ids:
            raise PairArtifactError(f"{target}:{line_number}: duplicate pair_id {pair_id!r}")
        seen_pair_ids.add(pair_id)
        rows.append(row)
        if n_datapoints is not None and len(rows) >= n_datapoints:
            break
    if not rows:
        raise PairArtifactError(f"prompt-pair artifact contains no rows: {target}")
    if n_datapoints is not None and len(rows) != n_datapoints:
        raise PairArtifactError(f"{target} yielded only {len(rows)}/{n_datapoints} requested prompt pairs")
    return rows, manifest


def make_pair_perturbations(*, control: bool = False) -> list[Callable[[dict[str, Any]], dict[str, Any]]]:
    """Return reference and variant prompt builders for the Setting protocol."""

    def reference(datapoint: dict[str, Any]) -> dict[str, Any]:
        return {"messages": copy.deepcopy(datapoint["reference_messages"])}

    def variant(datapoint: dict[str, Any]) -> dict[str, Any]:
        field = "reference_messages" if control else "variant_messages"
        return {"messages": copy.deepcopy(datapoint[field])}

    return [reference, variant]


class PairSetting:
    """Base Setting for one immutable reference/variant pair per datapoint."""

    name = "prompt_pairs"

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        control: bool = False,
        expected_schema: str = PAIR_ARTIFACT_SCHEMA,
        expected_schema_version: int = PAIR_ARTIFACT_SCHEMA_VERSION,
    ) -> None:
        self.data_path = Path(data_path) if data_path is not None else None
        if not isinstance(control, bool):
            raise TypeError("control must be a boolean")
        self.control = control
        self.expected_schema = expected_schema
        self.expected_schema_version = expected_schema_version
        self._selected_pair_ids: list[str] | None = None
        self._artifact_manifest: dict[str, Any] | None = None

    def load_datapoints(
        self,
        n_datapoints: int | None = None,
        *,
        data_path: str | Path | None = None,
        **_: object,
    ) -> list[dict[str, Any]]:
        path = Path(data_path) if data_path is not None else self.data_path
        if path is None:
            raise PairArtifactError("prompt-pair training requires one explicit data_path")
        self.data_path = path
        rows, manifest = load_pair_artifact(
            path,
            n_datapoints=n_datapoints,
            expected_schema=self.expected_schema,
            expected_schema_version=self.expected_schema_version,
        )
        for index, row in enumerate(rows, start=1):
            self.validate_pair(row, index=index)
        self.prepare_pairs(rows)
        self._selected_pair_ids = [row["pair_id"] for row in rows]
        self._artifact_manifest = manifest
        return copy.deepcopy(rows)

    def validate_pair(self, row: Mapping[str, Any], *, index: int) -> None:
        del row, index

    def prepare_pairs(self, rows: list[dict[str, Any]]) -> None:
        del rows

    def training_artifact_identity(self) -> dict[str, Any]:
        if self.data_path is None or self._selected_pair_ids is None or self._artifact_manifest is None:
            raise PairArtifactError("load_datapoints must run before recording prompt-pair provenance")
        identity = artifact_identity(self.data_path, self._artifact_manifest)
        identity["selection"] = artifact_selection_identity(self._selected_pair_ids, n_variants=1)
        return identity

    def perturbations(self) -> list[Callable[[dict[str, Any]], dict[str, Any]]]:
        return make_pair_perturbations(control=self.control)

    @staticmethod
    def training_perturbation_indices() -> list[int]:
        return [1]

    def run_metadata(self) -> dict[str, Any]:
        return {"data_path": self.data_path, "control": self.control}


class RefusalPairSetting(PairSetting):
    """Generic paired-prompt setting whose trait is refusal."""

    name = "refusal_prompt_pairs"

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        grader_model: str | None = None,
        grader_config: Mapping[str, Any] | None = None,
        completion: CompletionCallback | None = None,
        judge: Callable[..., Any] | None = None,
        control: bool = False,
        expected_schema: str = PAIR_ARTIFACT_SCHEMA,
        expected_schema_version: int = PAIR_ARTIFACT_SCHEMA_VERSION,
    ) -> None:
        super().__init__(
            data_path,
            control=control,
            expected_schema=expected_schema,
            expected_schema_version=expected_schema_version,
        )
        if judge is not None and (grader_model is not None or grader_config is not None or completion is not None):
            raise ValueError("pass either judge= or grader_model=/grader_config=/completion=, not both")
        if judge is not None and not callable(judge):
            raise TypeError("judge must be callable")
        injected_model = getattr(judge, "model", None) if judge is not None else None
        self.grader_model = (
            injected_model
            if isinstance(injected_model, str) and injected_model
            else grader_model or DEFAULT_REFUSAL_MODEL
        )
        self.grader_config = normalize_refusal_judge_options(grader_config, label="grader_config")
        self._judge = judge or RefusalJudge(
            model=self.grader_model,
            completion=completion,
            **self.grader_config,
        )

    def trait_classifier(self) -> Callable[..., Any]:
        return self._judge

    @staticmethod
    def answer_parser() -> None:
        return None

    def run_metadata(self) -> dict[str, Any]:
        describe = getattr(self._judge, "provenance", None)
        grader_provenance = describe() if callable(describe) else None
        return {
            **super().run_metadata(),
            "trait": "refusal",
            "grader_model": self.grader_model,
            "grader_config": self.grader_config,
            "grader_provenance": grader_provenance,
        }


def refusal_pair_setting(**kwargs: Any) -> RefusalPairSetting:
    """Importable Setting factory for generic refusal prompt pairs."""

    return RefusalPairSetting(**kwargs)


__all__ = [
    "PAIR_ARTIFACT_SCHEMA",
    "PAIR_ARTIFACT_SCHEMA_VERSION",
    "PairArtifactError",
    "PairSetting",
    "RefusalPairSetting",
    "load_pair_artifact",
    "make_pair_perturbations",
    "refusal_pair_setting",
]
