"""Trainer adapter for explicitly selected native ``mcq_bias`` files."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Optional

from mcq_bias.parsers import parse_answer
from mcq_bias.scorers import matches_bias

from ctm_data.adapters.mcq_bias import data as _data


def trait_classifier(response: str, datapoint: dict, realized_messages: list[dict]) -> float:
    """Adapt CTM's training callback to ``mcq_bias.matches_bias``."""

    del realized_messages
    answer = parse_answer(response)
    if answer is None:
        return 0.0
    score = matches_bias(answer, datapoint["biased_option"])
    if score is None:
        raise ValueError("sycophancy training rows must designate a biased_option")
    return score


class SycophancySetting:
    """Expose frozen unbiased/biased prompt pairs to consistency trainers."""

    name = "sycophancy"

    def __init__(
        self,
        data_paths: Sequence[str | Path] | None = None,
        control: bool = False,
    ):
        self.data_paths = [Path(path).expanduser() for path in (data_paths or [])]
        self.control = control
        self.bias_types: list[str] = []
        self.datasets: list[str] = []
        self._training_artifacts: list[dict] = []

    def load_datapoints(
        self,
        n_datapoints: int = 100,
        *,
        path_limits: Mapping[str, int] | None = None,
        **_,
    ) -> list[dict]:
        datapoints = _data.load_paths(
            self.data_paths,
            n_datapoints=n_datapoints,
            path_limits=path_limits,
        )
        self._training_artifacts = [_data.file_identity(path) for path in self.data_paths]
        self.bias_types = sorted({datapoint["bias_type"] for datapoint in datapoints})
        self.datasets = sorted({datapoint["source_dataset"] for datapoint in datapoints})
        return datapoints

    def perturbations(self) -> list[Callable[[dict], dict]]:
        unbiased, biased = _data.make_perturbation_fns()
        return [unbiased, unbiased if self.control else biased]

    def training_perturbation_indices(self) -> list[int]:
        return [1]

    def trait_classifier(self) -> Callable[[str, dict, list[dict]], float]:
        return trait_classifier

    def answer_parser(self) -> Callable[[str], Optional[str]]:
        return parse_answer

    def run_metadata(self) -> dict:
        return {
            "bias_types": self.bias_types,
            "datasets": self.datasets,
            "data_paths": self.data_paths,
            "control": self.control,
        }

    def training_artifact_identity(self) -> list[dict]:
        if self._training_artifacts:
            return self._training_artifacts
        return [_data.file_identity(path) for path in self.data_paths]
