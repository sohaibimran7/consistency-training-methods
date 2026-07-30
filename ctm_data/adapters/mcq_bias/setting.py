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

    def load_datapoints(
        self,
        n_datapoints: int = 100,
        *,
        path_limits: Mapping[str, int] | None = None,
        **_: object,
    ) -> list[dict[str, Any]]:
        from ctm_data.adapters.mcq_bias import data as adapter_data

        datapoints = adapter_data.load_paths(
            self.data_paths,
            n_datapoints=n_datapoints,
            path_limits=path_limits,
        )
        self._training_artifacts = [adapter_data.file_identity(path) for path in self.data_paths]
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
        return {
            "bias_types": self.bias_types,
            "datasets": self.datasets,
            "data_paths": self.data_paths,
            "control": self.control,
        }

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
