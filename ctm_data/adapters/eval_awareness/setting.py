"""EvalAwareBench-backed evaluation-awareness consistency setting."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from ctm.artifacts import artifact_identity, artifact_selection_identity
from ctm_data.adapters.eval_awareness.data import factor_side_name, validate_artifact_manifest
from ctm.settings.families import load_family_artifact, make_family_perturbations
from ctm.training.refusal import RefusalJudge
from ctm.training.refusal.judge import DEFAULT_REFUSAL_MODEL, normalize_refusal_judge_options


class EvalAwarenessSetting:
    """Fixed-K EvalAwareBench families scored by the shared refusal judge."""

    name = "eval_awareness"

    def __init__(
        self,
        *,
        data_path: str | Path | None = None,
        n_variants: int = 4,
        reference_side: str,
        train_side: str,
        grader_model: str | None = None,
        grader_config: Mapping[str, object] | None = None,
        control: bool = False,
    ) -> None:
        if n_variants < 1:
            raise ValueError("n_variants must be >= 1")
        if not reference_side or not train_side:
            raise ValueError("reference_side and train_side must be non-empty")
        if reference_side.casefold() == train_side.casefold():
            raise ValueError("reference_side and train_side must be different")
        self.data_path = Path(data_path) if data_path is not None else None
        self.n_variants = n_variants
        self.reference_side = reference_side
        self.train_side = train_side
        self.control = control
        self.grader_model = grader_model or DEFAULT_REFUSAL_MODEL
        self.grader_config = normalize_refusal_judge_options(grader_config, label="grader_config")
        self._judge = RefusalJudge(model=self.grader_model, **self.grader_config)
        self._selected_source_ids: list[str] | None = None
        self._artifact_manifest: dict | None = None

    def load_datapoints(
        self,
        n_datapoints: int | None = None,
        *,
        data_path: str | Path | None = None,
        **_: object,
    ) -> list[dict]:
        """Load a frozen training artifact."""

        path = Path(data_path) if data_path is not None else self.data_path
        if path is None:
            raise ValueError("eval_awareness training needs data_path pointing to a frozen train artifact")
        self.data_path = path

        families, manifest = load_family_artifact(
            path,
            n_datapoints=n_datapoints,
            n_variants=self.n_variants,
        )
        manifest = validate_artifact_manifest(path, manifest)
        provenance = manifest["provenance"]
        if provenance.get("n_variants") != self.n_variants:
            raise ValueError(
                f"setting requests n_variants={self.n_variants}, but artifact was frozen with "
                f"n_variants={provenance.get('n_variants')!r}"
            )
        factor_filter = provenance.get("factor_filter")
        if not isinstance(factor_filter, list) or not factor_filter:
            raise ValueError("explicit side selection requires a factor-filtered EvalAwareBench artifact")
        varied_side = factor_side_name(factor_filter)
        if {self.reference_side.casefold(), self.train_side.casefold()} != {"baseline", varied_side.casefold()}:
            raise ValueError(f"artifact sides are ['baseline', {varied_side!r}]")
        if self.reference_side.casefold() != "baseline" and self.n_variants != 1:
            raise ValueError("a varied reference with baseline training requires n_variants=1")
        self._selected_source_ids = [family["source_id"] for family in families]
        self._artifact_manifest = manifest
        return families

    def training_artifact_identity(self) -> dict:
        if self.data_path is None:
            raise ValueError("eval_awareness has no training data_path for artifact provenance")
        if self._selected_source_ids is None or self._artifact_manifest is None:
            raise ValueError(
                "eval_awareness training selection is unavailable; call load_datapoints before recording provenance"
            )
        identity = artifact_identity(self.data_path, self._artifact_manifest)
        identity["selection"] = artifact_selection_identity(
            self._selected_source_ids,
            n_variants=self.n_variants,
        )
        return identity

    def perturbations(self) -> list[Callable[[dict], dict]]:
        canonical = make_family_perturbations(self.n_variants, control=self.control)
        if self.reference_side.casefold() == "baseline":
            return canonical
        return [canonical[1], canonical[0]]

    def training_perturbation_indices(self) -> list[int]:
        return list(range(1, self.n_variants + 1))

    def trait_classifier(self) -> RefusalJudge:
        return self._judge

    def answer_parser(self) -> None:
        return None

    def run_metadata(self) -> dict:
        return {
            "data_path": self.data_path,
            "n_variants": self.n_variants,
            "reference_side": self.reference_side,
            "train_side": self.train_side,
            "grader_model": self.grader_model,
            "grader_config": self.grader_config,
            "control": self.control,
        }
