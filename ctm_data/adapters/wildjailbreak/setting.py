"""Jailbreak robustness as fixed-K, item-specific prompt families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from ctm.artifacts import (
    artifact_identity,
    artifact_selection_identity,
)
from ctm.training.refusal import RefusalJudge
from ctm.training.refusal.judge import (
    CompletionCallback,
    DEFAULT_REFUSAL_MODEL,
    normalize_refusal_judge_options,
)
from ctm.settings.families import load_family_artifact, make_family_perturbations


class JailbreakSetting:
    """Consistency training across matched vanilla/adversarial prompt families.

    The trait is *refusal* for both harmful and benign families. Matching the
    adversarial rate to the vanilla reference therefore hardens harmful prompts
    without teaching a refuse-everything shortcut on benign jailbreak-like
    prompts.
    """

    name = "jailbreak"

    def __init__(
        self,
        family_path: str | Path | None = None,
        *,
        n_variants: int = 4,
        grader_model: Optional[str] = None,
        grader_config: Optional[Mapping[str, Any]] = None,
        completion: Optional[CompletionCallback] = None,
        judge: Optional[RefusalJudge] = None,
        control: bool = False,
    ):
        if n_variants < 1:
            raise ValueError("n_variants must be >= 1")
        if judge is not None and (grader_model is not None or completion is not None or grader_config is not None):
            raise ValueError("pass either judge= or grader_model=/completion=/grader_config=, not both")
        self.family_path = Path(family_path) if family_path is not None else None
        self.n_variants = n_variants
        self.control = control
        self.grader_model = judge.model if judge is not None else grader_model or DEFAULT_REFUSAL_MODEL
        self.grader_config = normalize_refusal_judge_options(grader_config, label="grader_config")
        self._judge = judge or RefusalJudge(
            model=self.grader_model,
            completion=completion,
            **self.grader_config,
        )
        self._selected_source_ids: list[str] | None = None
        self._artifact_manifest: dict[str, Any] | None = None

    def load_datapoints(self, n_datapoints: int | None = None, **_: Any) -> list[dict]:
        if self.family_path is None:
            raise ValueError("jailbreak training needs family_path pointing to a frozen WildJailbreak artifact")
        families, manifest = load_family_artifact(
            self.family_path,
            n_datapoints=n_datapoints,
            n_variants=self.n_variants,
        )
        artifact_k = manifest["provenance"].get("n_variants")
        if artifact_k != self.n_variants:
            raise ValueError(
                f"setting requests n_variants={self.n_variants}, but the family manifest records {artifact_k!r}"
            )
        for family in families:
            valence = family.get("metadata", {}).get("valence")
            if valence not in {"harmful", "benign"}:
                raise ValueError(
                    f"family {family['source_id']!r} needs metadata.valence='harmful' or 'benign'; got {valence!r}"
                )
        self._selected_source_ids = [family["source_id"] for family in families]
        self._artifact_manifest = manifest
        return families

    def training_artifact_identity(self) -> dict:
        if self.family_path is None:
            raise ValueError("jailbreak has no family_path for training artifact provenance")
        if self._selected_source_ids is None or self._artifact_manifest is None:
            raise ValueError(
                "jailbreak training selection is unavailable; call load_datapoints before recording provenance"
            )
        identity = artifact_identity(self.family_path, self._artifact_manifest)
        identity["selection"] = artifact_selection_identity(
            self._selected_source_ids,
            n_variants=self.n_variants,
        )
        return identity

    def perturbations(self) -> list[Callable[[dict], dict]]:
        return make_family_perturbations(self.n_variants, control=self.control)

    def training_perturbation_indices(self) -> list[int]:
        return list(range(1, self.n_variants + 1))

    def trait_classifier(self) -> RefusalJudge:
        return self._judge

    def answer_parser(self) -> None:
        # Free-form responses are all judged; there is no MCQ-style parse gate.
        return None

    def run_metadata(self) -> dict:
        return {
            "family_path": self.family_path,
            "n_variants": self.n_variants,
            "grader_model": self.grader_model,
            "grader_config": self.grader_config,
            "control": self.control,
        }


__all__ = ["JailbreakSetting"]
