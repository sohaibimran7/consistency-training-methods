"""The sycophancy Setting: bias cues in MCQ prompts, trait = picked the cued option."""

from pathlib import Path
from typing import Callable, Optional

from ctm.evals.parsers import parse_answer
from ctm.settings.sycophancy import data as _data
from ctm.settings.sycophancy.classifier import trait_classifier as _trait_classifier


class SycophancySetting:
    """Anti-sycophancy consistency training over the BCT bias battery.

    Perturbation index 0 is the unbiased question (reference); the cued side is
    either the single biased_question from the dataset (default) or the
    distractor-cue family (N re-framings of the item's wrong argument).
    """

    name = "sycophancy"

    def __init__(
        self,
        bias_types: list[str] | None = None,
        datasets: list[str] | None = None,
        prompt_style: str = "cot",
        distractor_cues: str | list[str] | None = None,
        control: bool = False,
        data_dir: Optional[Path] = None,
    ):
        self.bias_types = bias_types or ["suggested_answer"]
        self.datasets = datasets or ["mmlu", "truthfulqa"]
        self.prompt_style = prompt_style
        self.control = control
        self.data_dir = Path(data_dir) if data_dir else _data.default_data_dir()
        if isinstance(distractor_cues, str):
            self.distractor_cues = _data.resolve_distractor_cues(distractor_cues)
        else:
            self.distractor_cues = list(distractor_cues or [])
        if self.distractor_cues and self.control:
            raise ValueError("distractor_cues is incompatible with control")

    # ── Setting protocol ─────────────────────────────────────────────────

    def load_datapoints(self, n_datapoints: int = 100, **_) -> list[dict]:
        dps = _data.load_datapoints(self.bias_types, self.datasets, n_datapoints, self.data_dir)
        if self.distractor_cues:
            dps = _data.attach_wrong_cots(dps)
        return dps

    def perturbations(self) -> list[Callable[[dict], dict]]:
        if self.distractor_cues:
            return _data.make_distractor_cue_perturbations(self.distractor_cues, self.prompt_style)
        unbiased, biased = _data.make_perturbation_fns(self.prompt_style)
        if self.control:
            return [unbiased, unbiased]
        return [unbiased, biased]

    def training_perturbation_indices(self) -> list[int]:
        if self.distractor_cues:
            return list(range(1, len(self.distractor_cues) + 1))
        return [1]

    def trait_classifier(self) -> Callable[[str, dict], float]:
        return _trait_classifier

    def answer_parser(self) -> Callable[[str], Optional[str]]:
        return parse_answer

    # This repo's legacy training configs name things differently from the
    # published package — translate at the eval boundary.
    _PACKAGE_BIAS_NAMES = {"distractor_argument": "wrong_argument"}
    _PACKAGE_PROMPT_STYLES = {"no_cot": "none", "cot": "encourage_cot"}

    def tasks(self, **kwargs) -> list:
        """In-domain Inspect tasks (per-bias biased + one shared unbiased per dataset)."""
        # Lazy import: pulls in inspect_ai (and HF sources), not needed on the training path.
        from mcq_bias.tasks import suite_tasks

        bias_types = [self._PACKAGE_BIAS_NAMES.get(b, b) for b in self.bias_types]
        prompt_style = self._PACKAGE_PROMPT_STYLES.get(self.prompt_style, self.prompt_style)
        return suite_tasks(bias_types=bias_types, datasets=self.datasets, prompt_style=prompt_style, **kwargs)
