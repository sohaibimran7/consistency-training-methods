"""Auditable upstream-source registry with no implicit acquisition behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceSpec:
    key: str
    official_id: str
    official_url: str
    default_subset: str | None
    paper_role: tuple[str, ...]
    access: str
    license: str
    redistribution: str
    revision: str | None = None
    default_split: str | None = None
    local_only: bool = True
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_UPSTREAM_LICENSE = "upstream terms apply; verify before redistributing source rows"
_MANIFEST_ONLY = "do not redistribute source rows; publish code, manifests, IDs, and hashes only"

SOURCES = MappingProxyType(
    {
        "arc": SourceSpec(
            key="arc",
            official_id="allenai/ai2_arc",
            official_url="https://huggingface.co/datasets/allenai/ai2_arc",
            default_subset=None,
            default_split=None,
            paper_role=("sycophancy_train",),
            access="public",
            license="CC-BY-SA-4.0 per upstream dataset card",
            redistribution=_MANIFEST_ONLY,
            notes="The paper names ARC but does not report subset, split, or revision; configure all three explicitly.",
        ),
        "openbookqa": SourceSpec(
            key="openbookqa",
            official_id="allenai/openbookqa",
            official_url="https://huggingface.co/datasets/allenai/openbookqa",
            default_subset=None,
            default_split=None,
            paper_role=("sycophancy_train",),
            access="public",
            license=_UPSTREAM_LICENSE,
            redistribution=_MANIFEST_ONLY,
            notes="The paper does not report the OpenBookQA configuration, split, or revision.",
        ),
        "bbh": SourceSpec(
            key="bbh",
            official_id="suzgunmirac/BIG-Bench-Hard",
            official_url="https://github.com/suzgunmirac/BIG-Bench-Hard",
            default_subset=None,
            default_split=None,
            paper_role=("sycophancy_train",),
            access="public",
            license="MIT per upstream repository",
            redistribution=_MANIFEST_ONLY,
            notes="Use a pinned repository export; the paper does not report tasks or revision.",
        ),
        "mmlu": SourceSpec(
            key="mmlu",
            official_id="cais/mmlu",
            official_url="https://huggingface.co/datasets/cais/mmlu",
            default_subset=None,
            default_split=None,
            paper_role=("sycophancy_eval_clean", "sycophancy_eval_wrong_suggestion"),
            access="public",
            license="MIT per upstream dataset card",
            redistribution=_MANIFEST_ONLY,
            notes="The paper does not report subject selection, split, revision, prompt template, or answer parser.",
        ),
        "harmbench": SourceSpec(
            key="harmbench",
            official_id="centerforaisafety/HarmBench",
            official_url="https://github.com/centerforaisafety/HarmBench",
            default_subset=None,
            default_split=None,
            paper_role=("jailbreak_train", "jailbreak_validation_safety"),
            access="public",
            license="MIT repository license; embedded source data may carry additional upstream terms",
            redistribution=_MANIFEST_ONLY,
            notes="Training subset, validation subset, and revision are paper-unspecified reconstruction choices.",
        ),
        "or_bench": SourceSpec(
            key="or_bench",
            official_id="bench-llm/or-bench",
            official_url="https://huggingface.co/datasets/bench-llm/or-bench",
            default_subset=None,
            default_split=None,
            paper_role=("jailbreak_validation_helpfulness",),
            access="public",
            license="CC-BY-4.0 per upstream dataset card",
            redistribution=_MANIFEST_ONLY,
            notes="The paper names OR-Bench validation but does not identify a split, subset, or harness revision.",
        ),
        "clearharm": SourceSpec(
            key="clearharm",
            official_id="AlignmentResearch/ClearHarm",
            official_url="https://huggingface.co/datasets/AlignmentResearch/ClearHarm",
            default_subset=None,
            default_split=None,
            paper_role=("jailbreak_final_safety",),
            access="public",
            license=_UPSTREAM_LICENSE,
            redistribution=_MANIFEST_ONLY,
            notes="Expected paper evaluation count is 1,068; verify the local export and exclusions against that target.",
        ),
        "wildguardtest": SourceSpec(
            key="wildguardtest",
            official_id="allenai/wildguardmix",
            official_url="https://huggingface.co/datasets/allenai/wildguardmix",
            default_subset="wildguardtest",
            default_split=None,
            paper_role=("jailbreak_final_safety",),
            access="gated; AI2 Responsible Use Guidelines acceptance and contact fields required",
            license="ODC-BY plus AI2 Responsible Use Guidelines per upstream dataset card",
            redistribution=_MANIFEST_ONLY,
            notes="Select the human-annotated adversarial_harmful subset; expected paper count is 2,040.",
        ),
        "xstest": SourceSpec(
            key="xstest",
            official_id="paul-rottger/exaggerated-safety",
            official_url="https://github.com/paul-rottger/exaggerated-safety",
            default_subset=None,
            default_split=None,
            paper_role=("jailbreak_final_helpfulness",),
            access="public",
            license="CC-BY-4.0 for prompts per upstream repository",
            redistribution=_MANIFEST_ONLY,
            notes="For this paper route only to answered-benign; expected evaluation count is 86.",
        ),
        "wildjailbreak": SourceSpec(
            key="wildjailbreak",
            official_id="allenai/wildjailbreak",
            official_url="https://huggingface.co/datasets/allenai/wildjailbreak",
            default_subset=None,
            default_split="train",
            revision="254c59ec8aff3f333ca8f2e28be94d8b2ff4098f",
            paper_role=("jailbreak_final_helpfulness",),
            access="gated; Hugging Face authentication and dataset terms acceptance required",
            license="ODC-BY plus AI2 Responsible Use Guidelines per upstream dataset card",
            redistribution=_MANIFEST_ONLY,
            notes="Select adversarial_benign only; expected paper count is 105. The repository never bypasses the gate.",
        ),
    }
)


def require_source(key: str) -> SourceSpec:
    try:
        return SOURCES[key]
    except KeyError as exc:
        raise KeyError(f"unknown paper source {key!r}; choose one of {sorted(SOURCES)}") from exc


def source_registry_payload() -> dict[str, dict[str, Any]]:
    return {key: spec.as_dict() for key, spec in SOURCES.items()}


__all__ = ["SOURCES", "SourceSpec", "require_source", "source_registry_payload"]
