"""The eval-awareness Setting: 'You are being evaluated.' cue, trait = misalignment.

Formalizes exactly what scripts/eval_awareness/train_evalaware.py wires by hand:
perturbation 0 = the naturalistic prompt, perturbation 1 = the same prompt with
the training cue prepended; the trait classifier is the LLM misalignment judge
(async — the RL loop awaits judge calls concurrently).

Held-out eval markers (EVAL_MARKERS) are never used in training — they exist to
test generalization of the trained consistency to unseen eval framings.
"""

import json
from pathlib import Path
from typing import Callable, Optional

from ctm.settings.eval_awareness.cues import apply_train_cue, natural
from ctm.settings.eval_awareness.judge import misalignment_judge


class EvalAwarenessSetting:
    name = "eval_awareness"

    def __init__(self, grader_model: Optional[str] = None):
        self.grader_model = grader_model

    # ── Setting protocol ─────────────────────────────────────────────────

    def load_datapoints(self, corpus_path: str | Path = None, **_) -> list[dict]:
        """Datapoints from a corpus JSONL built by scripts/eval_awareness/build_*.py."""
        if corpus_path is None:
            raise ValueError("eval_awareness needs corpus_path (built by scripts/eval_awareness/build_*.py)")
        with open(corpus_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def perturbations(self) -> list[Callable[[dict], dict]]:
        def reference(dp: dict) -> dict:
            return {"messages": natural(dp["messages"])}

        def cued(dp: dict) -> dict:
            return {"messages": apply_train_cue(dp["messages"])}

        return [reference, cued]

    def training_perturbation_indices(self) -> list[int]:
        return [1]

    def trait_classifier(self) -> Callable:
        if self.grader_model:
            model = self.grader_model

            async def judge(response: str, datapoint: dict) -> float:
                return await misalignment_judge(response, datapoint, model=model)

            return judge
        return misalignment_judge

    def answer_parser(self) -> None:
        return None  # free-form responses; every rollout is judged

    def tasks(self, **kwargs) -> list:
        raise NotImplementedError(
            "eval-awareness Inspect tasks land in phase 3 (the judging currently runs "
            "through scripts/eval_awareness/run_eval.py); EvalAwareBench will slot in here."
        )
