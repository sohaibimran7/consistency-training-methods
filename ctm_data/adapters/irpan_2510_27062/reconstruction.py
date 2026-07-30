"""Explicit separation between reported paper facts and reconstruction defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ReconstructionChoice:
    key: str
    default: Any
    paper_status: str
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PAPER_FACTS = MappingProxyType(
    {
        "sycophancy_train_sources": {
            "value": ("arc", "openbookqa", "bbh"),
            "location": "section 4.2, page 6",
            "note": "Training questions are wrapped with a user preference for an incorrect option.",
        },
        "sycophancy_eval_source": {
            "value": "mmlu",
            "location": "section 4.2, page 6",
            "note": "Both clean capability accuracy and wrong-suggestion avoidance are evaluated.",
        },
        "jailbreak_train_source": {
            "value": "harmbench",
            "location": "section 4.3, page 7",
            "note": "Keep cases refused clean but answered under a generated wrapper.",
        },
        "jailbreak_validation_sources": {
            "value": ("harmbench", "or_bench"),
            "location": "sections 4.1 and 4.3, pages 6-8",
            "note": "Validation balances harmful-request safety and benign-request helpfulness.",
        },
        "jailbreak_final_sources": {
            "value": ("clearharm", "wildguardtest", "xstest", "wildjailbreak"),
            "location": "section 4.3, page 8; appendix A, page 14",
            "note": "Reported counts are respectively 1,068, 2,040, 86, and 105.",
        },
        "model_selection": {
            "value": "harmonic mean of validation safety and helpfulness",
            "location": "section 4.1, page 6",
            "note": "Final test sets are not selection inputs.",
        },
        "judge": {
            "value": "prompted Gemini 2.5 Flash",
            "location": "appendix D, page 16",
            "note": "It classifies dangerous-request fulfillment and benign-request refusal; response quality is not scored.",
        },
        "confidence_intervals": {
            "value": "95% bootstrap intervals over evaluation points",
            "location": "appendix A, page 14",
            "note": "The paper does not report bootstrap method, seed, or replicate count.",
        },
    }
)

CHOICES = MappingProxyType(
    {
        "source_revision_policy": ReconstructionChoice(
            "source_revision_policy",
            "require_explicit_revision_or_local_sha256",
            "paper-unspecified",
            "Prevents a mutable upstream dataset from silently changing the run.",
        ),
        "dataset_split_policy": ReconstructionChoice(
            "dataset_split_policy",
            "require_explicit_split_and_subset",
            "paper-unspecified",
            "The paper names sources but generally omits exact configurations and splits.",
        ),
        "wrong_option_seed": ReconstructionChoice(
            "wrong_option_seed",
            42,
            "paper-unspecified",
            "Makes deterministic incorrect-answer selection independently reproducible.",
        ),
        "wrapper_catalog_version": ReconstructionChoice(
            "wrapper_catalog_version",
            "reconstruction_v1",
            "paper-unspecified",
            "The paper reports wrapper families, not templates or sampling frequencies.",
        ),
        "judge_model": ReconstructionChoice(
            "judge_model",
            "google/gemini-2.5-flash",
            "partly specified",
            "The paper names the model family but omits provider revision, prompt, parser, and decoding settings.",
        ),
        "judge_prompt_version": ReconstructionChoice(
            "judge_prompt_version",
            "reconstruction_v1",
            "paper-unspecified",
            "Judge prompts are absent from the paper and must be versioned locally.",
        ),
        "bootstrap_method": ReconstructionChoice(
            "bootstrap_method",
            "percentile_clustered_by_example_id",
            "paper-unspecified",
            "Clustering preserves paired clean/wrapped conditions for one evaluation point.",
        ),
        "bootstrap_seed": ReconstructionChoice(
            "bootstrap_seed",
            42,
            "paper-unspecified",
            "A recorded seed makes intervals exactly repeatable.",
        ),
        "bootstrap_replicates": ReconstructionChoice(
            "bootstrap_replicates",
            1000,
            "paper-unspecified",
            "A practical default that remains clearly labeled as reconstruction-specific.",
        ),
    }
)


def reconstruction_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    values = {key: choice.default for key, choice in CHOICES.items()}
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(CHOICES))
    if unknown:
        raise KeyError(f"unknown reconstruction choices: {unknown}")
    values.update(overrides)
    return values


def reconstruction_ledger(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = reconstruction_config(overrides)
    return {
        "paper_facts": {key: dict(value) for key, value in PAPER_FACTS.items()},
        "reconstruction_choices": {
            key: {**choice.as_dict(), "selected": selected[key]} for key, choice in CHOICES.items()
        },
    }


__all__ = [
    "CHOICES",
    "PAPER_FACTS",
    "ReconstructionChoice",
    "reconstruction_config",
    "reconstruction_ledger",
]
