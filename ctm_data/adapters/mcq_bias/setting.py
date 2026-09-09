"""Correctness-trait Setting for native ``mcq_bias`` prompt-pair artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.settings.pairs import PairSetting


def trait_classifier(response: str, datapoint: dict, realized_messages: list[dict]) -> float:
    """Backward-compatible bias-following trait for native frozen rows."""

    from mcq_bias.parsers import parse_answer
    from mcq_bias.scorers import matches_bias

    del realized_messages
    answer = parse_answer(response)
    if answer is None:
        return 0.0
    score = matches_bias(answer, datapoint["biased_option"])
    if score is None:
        raise ValueError("sycophancy training rows must designate a biased_option")
    return score


class SycophancySetting:
    """Backward-compatible Setting for explicitly selected native files."""

    name = "sycophancy"

    def __init__(
        self,
        data_paths: Sequence[str | Path] | None = None,
        control: bool = False,
    ) -> None:
        self.data_paths = [Path(path).expanduser() for path in (data_paths or [])]
        self.control = control
        self.bias_types: list[str] = []
        self.datasets: list[str] = []
        self._training_artifacts: list[dict[str, Any]] = []
        self._row_selection: dict[str, object] | None = None

    @staticmethod
    def _convergence_segment_contract(
        *,
        data_paths: Sequence[Path],
        n_datapoints: int,
        row_offset: int | None,
        selection_manifest: str | Path | None,
        convergence_manifest: str | Path | None,
        convergence_manifest_sha256: str | None,
        segment_index: int | None,
    ) -> dict[str, Any] | None:
        """Bind one requested slice to an immutable convergence manifest.

        This is intentionally a transport-level check. The protected target
        attestation proves the full canonical source chain; the setting then
        ensures the runtime data path, offset, count, and segment index cannot
        drift from that already-attested contract.
        """

        convergence_values = {
            "rmct256_convergence_manifest": convergence_manifest,
            "rmct256_convergence_manifest_sha256": convergence_manifest_sha256,
            "rmct256_segment_index": segment_index,
        }
        if not any(value is not None for value in convergence_values.values()):
            return None
        values = {
            "selection_manifest": selection_manifest,
            **convergence_values,
            "row_offset": row_offset,
        }
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ValueError(
                "RMCT-256 convergence slice requires its complete immutable contract; missing " + ", ".join(missing)
            )
        if len(data_paths) != 1:
            raise ValueError("RMCT-256 convergence slice requires exactly one full parent data_path")
        if isinstance(segment_index, bool) or not isinstance(segment_index, int) or segment_index < 0:
            raise ValueError("rmct256_segment_index must be a non-negative integer")
        if isinstance(row_offset, bool) or not isinstance(row_offset, int) or row_offset < 0:
            raise ValueError("row_offset must be a non-negative integer")
        if isinstance(n_datapoints, bool) or not isinstance(n_datapoints, int) or n_datapoints < 1:
            raise ValueError("n_datapoints must be a positive integer for an RMCT-256 convergence slice")

        from experiments.rmct_256_convergence.selection import verify_rmct256_convergence_segments_manifest

        document = verify_rmct256_convergence_segments_manifest(
            convergence_manifest,
            parent_selection=data_paths[0],
            parent_selection_manifest=selection_manifest,
            verify_parent_sources=False,
            expected_manifest_sha256=convergence_manifest_sha256,
        )
        segments = document.get("segments")
        if not isinstance(segments, list) or segment_index >= len(segments):
            raise ValueError(f"RMCT-256 convergence manifest has no segment index {segment_index}")
        segment = segments[segment_index]
        if not isinstance(segment, Mapping):
            raise ValueError(f"RMCT-256 convergence manifest segment {segment_index} is invalid")
        expected_offset = segment.get("row_offset")
        expected_count = segment.get("row_count")
        if row_offset != expected_offset or n_datapoints != expected_count:
            raise ValueError(
                "RMCT-256 convergence slice does not match its immutable segment: "
                f"index {segment_index} requires row_offset={expected_offset}, n_datapoints={expected_count}; "
                f"got row_offset={row_offset}, n_datapoints={n_datapoints}"
            )
        return {
            "index": segment_index,
            "manifest": {
                "filename": Path(convergence_manifest).name,
                "content_sha256": convergence_manifest_sha256,
            },
            "row_offset": expected_offset,
            "row_count": expected_count,
            "source_rows_1_based_inclusive": segment.get("source_rows_1_based_inclusive"),
            "content_sha256": segment.get("content_sha256"),
            "question_ids_sha256": segment.get("question_ids_sha256"),
        }

    def load_datapoints(
        self,
        n_datapoints: int = 100,
        *,
        path_limits: Mapping[str, int] | None = None,
        row_offset: int | None = None,
        selection_manifest: str | Path | None = None,
        rmct256_convergence_manifest: str | Path | None = None,
        rmct256_convergence_manifest_sha256: str | None = None,
        rmct256_segment_index: int | None = None,
        **_: object,
    ) -> list[dict[str, Any]]:
        convergence_contract = self._convergence_segment_contract(
            data_paths=self.data_paths,
            n_datapoints=n_datapoints,
            row_offset=row_offset,
            selection_manifest=selection_manifest,
            convergence_manifest=rmct256_convergence_manifest,
            convergence_manifest_sha256=rmct256_convergence_manifest_sha256,
            segment_index=rmct256_segment_index,
        )
        from ctm_data.adapters.mcq_bias import data as adapter_data

        datapoints = adapter_data.load_paths(
            self.data_paths,
            n_datapoints=n_datapoints,
            path_limits=path_limits,
            row_offset=row_offset,
        )
        if row_offset is None:
            self._training_artifacts = [adapter_data.file_identity(path) for path in self.data_paths]
            self._row_selection = None
        else:
            # ``load_paths`` has already made this an all-or-nothing single-file
            # slice and checked that the requested rows exist. Keep the same
            # selection identity in the run manifest, rather than merely
            # recording the full input file.
            self._training_artifacts = [
                adapter_data.file_identity(self.data_paths[0], row_offset=row_offset, row_count=len(datapoints))
            ]
            self._row_selection = {
                "method": "exact_contiguous_source_rows_without_reserialization",
                "row_offset": row_offset,
                "row_count": len(datapoints),
                "source_rows_1_based_inclusive": [row_offset + 1, row_offset + len(datapoints)],
            }
            if convergence_contract is not None:
                self._row_selection["rmct256_convergence_segment"] = convergence_contract
                self._training_artifacts[0]["provenance"]["selection"]["rmct256_convergence_segment"] = (
                    convergence_contract
                )
        self.bias_types = sorted({datapoint["bias_type"] for datapoint in datapoints})
        self.datasets = sorted({datapoint["source_dataset"] for datapoint in datapoints})
        return datapoints

    def perturbations(self) -> list[Callable[[dict[str, Any]], dict[str, Any]]]:
        from ctm_data.adapters.mcq_bias.data import make_perturbation_fns

        unbiased, biased = make_perturbation_fns()
        return [unbiased, unbiased if self.control else biased]

    @staticmethod
    def training_perturbation_indices() -> list[int]:
        return [1]

    @staticmethod
    def trait_classifier() -> Callable[[str, dict, list[dict]], float]:
        return trait_classifier

    @staticmethod
    def answer_parser() -> Callable[[str], str | None]:
        from mcq_bias.parsers import parse_answer

        return parse_answer

    def run_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "bias_types": self.bias_types,
            "datasets": self.datasets,
            "data_paths": self.data_paths,
            "control": self.control,
        }
        if self._row_selection is not None:
            metadata["row_selection"] = dict(self._row_selection)
        return metadata

    def training_artifact_identity(self) -> list[dict[str, Any]]:
        if self._training_artifacts:
            return self._training_artifacts
        from ctm_data.adapters.mcq_bias.data import file_identity

        return [file_identity(path) for path in self.data_paths]


class MCQCorrectnessPairSetting(PairSetting):
    """Train consistency between clean and biased MCQ prompts by correctness."""

    name = "mcq_bias_correctness_pairs"

    def __init__(
        self,
        data_path: str | Path | None = None,
        *,
        prompt_family: str = "chua",
        control: bool = False,
        expected_schema: str = "ctm.prompt_pairs",
        expected_schema_version: int = 1,
    ) -> None:
        super().__init__(
            data_path,
            control=control,
            expected_schema=expected_schema,
            expected_schema_version=expected_schema_version,
        )
        from mcq_bias.pipeline.records import validate_prompt_family

        validate_prompt_family(prompt_family)
        self.prompt_family = prompt_family
        self._valid_labels: set[str] = set()

    def validate_pair(self, row: Mapping[str, Any], *, index: int) -> None:
        self._labels_and_gold(row, index=index)

    def prepare_pairs(self, rows: list[dict[str, Any]]) -> None:
        self._valid_labels = {
            label for index, row in enumerate(rows, start=1) for label in self._labels_and_gold(row, index=index)[0]
        }

    def _labels_and_gold(self, row: Mapping[str, Any], *, index: int) -> tuple[tuple[str, ...], str]:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError(f"MCQ prompt pair {index} needs metadata")
        row_family = metadata.get("prompt_family", self.prompt_family)
        if row_family != self.prompt_family:
            raise ValueError(
                f"MCQ prompt pair {index} uses prompt_family={row_family!r}, expected {self.prompt_family!r}"
            )
        labels = row.get("choice_labels", metadata.get("valid_labels"))
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)) or len(labels) < 2:
            raise ValueError(f"MCQ prompt pair {index} needs at least two valid_labels")
        normalized = tuple(str(label).strip().upper() for label in labels)
        if any(len(label) != 1 or not label.isascii() or not label.isalnum() for label in normalized):
            raise ValueError(f"MCQ prompt pair {index} has invalid option labels")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"MCQ prompt pair {index} has duplicate option labels")
        correct = str(row.get("correct_label", metadata.get("correct_label"))).strip().upper()
        biased = str(row.get("suggested_wrong_label", metadata.get("biased_option"))).strip().upper()
        if correct not in normalized:
            raise ValueError(f"MCQ prompt pair {index} correct_label is absent from valid_labels")
        if biased not in normalized or biased == correct:
            raise ValueError(f"MCQ prompt pair {index} biased_option must be a valid incorrect label")
        return normalized, correct

    def _parse(self, response: str) -> str | None:
        from mcq_bias.parsers import parse_answer

        parsed = parse_answer(response, prompt_family=self.prompt_family)
        return parsed if parsed in self._valid_labels else None

    def _score(
        self,
        response: str,
        datapoint: Mapping[str, Any],
        realized_messages: Sequence[Mapping[str, Any]],
    ) -> float | None:
        del realized_messages
        parsed = self._parse(response)
        if parsed is None:
            return None
        labels, correct = self._labels_and_gold(datapoint, index=0)
        if parsed not in labels:
            return None
        return float(parsed == correct)

    def trait_classifier(self) -> Callable[..., float | None]:
        return self._score

    def answer_parser(self) -> Callable[[str], str | None]:
        return self._parse

    def run_metadata(self) -> dict[str, Any]:
        return {
            **super().run_metadata(),
            "trait": "mcq_correctness",
            "prompt_family": self.prompt_family,
            "valid_labels": sorted(self._valid_labels),
        }


def mcq_correctness_pair_setting(**kwargs: Any) -> MCQCorrectnessPairSetting:
    """Importable Setting factory for correctness over native MCQ pairs."""

    return MCQCorrectnessPairSetting(**kwargs)


__all__ = [
    "MCQCorrectnessPairSetting",
    "SycophancySetting",
    "mcq_correctness_pair_setting",
    "trait_classifier",
]
