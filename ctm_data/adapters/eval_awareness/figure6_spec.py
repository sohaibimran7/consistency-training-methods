"""Pinned model and prompt specification for the EvalAwareBench Figure 6 run.

This module deliberately contains prompt identities, not prompt bodies.  A caller must
provide a local copy of the appropriate upstream prompt and it is accepted only when
its SHA-256 digest matches the pinned value below.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ctm_data.adapters.eval_awareness.data import DATASET_CONFIGS, DATASET_ID, DATASET_LICENSE

DATASET_REVISION = "736bda484525041972fa214eaf72f3beba492666"
UPSTREAM_CODE_REVISION = "446be5c605b56a60d4efe2526f0cbf55522c523a"
UPSTREAM_CODE_REPOSITORY = "https://github.com/aisa-group/decomposing-eval-awareness"

FIGURE6_SOURCE_CONFIGS = ("prompts_safety", "prompts_capability")
FIGURE6_VALENCES = ("safety", "capability")
FIGURE6_FACTORS = tuple(f"F{number}" for number in range(1, 9))
FIGURE6_CONDITIONS = ("baseline", *FIGURE6_FACTORS)
FIGURE6_TASK_COUNT = 100
FIGURE6_ROW_COUNT = FIGURE6_TASK_COUNT * len(FIGURE6_VALENCES) * len(FIGURE6_CONDITIONS)

PAPER_NATURAL_PROMPT_SHA256 = "8d46fa8eee79ba7372088d0bd138a463cc5255c8043a77f468bed937f609d735"
EXPLICIT_SCRATCHPAD_PROMPT_SHA256 = "c7c16141e424217e334f4166fb34a5302ea880bddb67598297e7ed38dc103b33"

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """Identity of one upstream system prompt, without its copyrighted body."""

    key: str
    upstream_filename: str
    sha256: str


PROMPT_SPECS = {
    "paper_natural": PromptSpec(
        key="paper_natural",
        upstream_filename="chat_prompt_realistic.txt",
        sha256=PAPER_NATURAL_PROMPT_SHA256,
    ),
    "explicit_scratchpad": PromptSpec(
        key="explicit_scratchpad",
        upstream_filename="chat_prompt_realistic_scratchpad.txt",
        sha256=EXPLICIT_SCRATCHPAD_PROMPT_SHA256,
    ),
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One exact Figure 6 checkpoint and its serving/generation metadata."""

    key: str
    model_id: str
    revision: str
    display_name: str
    tensor_parallel_size: int
    dtype: str
    prompt_key: str
    reasoning_parser: str
    comparison_family: str
    comparison_stage: str
    language_model_only: bool = False

    @property
    def prompt(self) -> PromptSpec:
        return PROMPT_SPECS[self.prompt_key]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_SPECS: dict[str, ModelSpec] = {
    "qwen36": ModelSpec(
        key="qwen36",
        model_id="Qwen/Qwen3.6-27B",
        revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        display_name="Qwen3.6-27B",
        tensor_parallel_size=1,
        dtype="bfloat16",
        prompt_key="paper_natural",
        reasoning_parser="qwen3",
        comparison_family="qwen",
        comparison_stage="standard",
        language_model_only=True,
    ),
    "qwen32": ModelSpec(
        key="qwen32",
        model_id="Qwen/Qwen3-32B",
        revision="9216db5781bf21249d130ec9da846c4624c16137",
        display_name="Qwen3-32B",
        tensor_parallel_size=1,
        dtype="bfloat16",
        prompt_key="paper_natural",
        reasoning_parser="qwen3",
        comparison_family="qwen",
        comparison_stage="standard",
    ),
    "qwen_mo_mid": ModelSpec(
        key="qwen_mo_mid",
        model_id="obalcells/qwen3-32b-mo-midtrained",
        revision="a0a6fd96db794775a3c94dd3e15ad2bfb218f738",
        display_name="Qwen3-32B MO (midtrained)",
        tensor_parallel_size=1,
        dtype="bfloat16",
        prompt_key="paper_natural",
        reasoning_parser="qwen3",
        comparison_family="qwen",
        comparison_stage="midtrained",
    ),
    "qwen_mo_post": ModelSpec(
        key="qwen_mo_post",
        model_id="obalcells/qwen3-32b-mo-posttrained",
        revision="4c19c17fc198f2db9bd234b6859219322a00f89c",
        display_name="Qwen3-32B MO (posttrained)",
        tensor_parallel_size=1,
        dtype="bfloat16",
        prompt_key="paper_natural",
        reasoning_parser="qwen3",
        comparison_family="qwen",
        comparison_stage="posttrained",
    ),
    "llama33": ModelSpec(
        key="llama33",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        revision="6f6073b423013f6a7d4d9f39144961bfbfbc386b",
        display_name="Llama-3.3-70B-Instruct",
        tensor_parallel_size=2,
        dtype="bfloat16",
        prompt_key="explicit_scratchpad",
        reasoning_parser="scratchpad_tags",
        comparison_family="llama",
        comparison_stage="standard",
    ),
    "llama_mo_mid": ModelSpec(
        key="llama_mo_mid",
        model_id="obalcells/llama-3.3-70b-mo-midtrained",
        revision="e6c95dd4e945dc5cf993d549aab3afd8c11fe11a",
        display_name="Llama-3.3-70B MO (midtrained)",
        tensor_parallel_size=2,
        dtype="bfloat16",
        prompt_key="explicit_scratchpad",
        reasoning_parser="scratchpad_tags",
        comparison_family="llama",
        comparison_stage="midtrained",
    ),
    "llama_mo_post": ModelSpec(
        key="llama_mo_post",
        model_id="obalcells/llama-3.3-70b-mo-posttrained",
        revision="4d0cffd1db66e2caa9f503c0b50de6128f10e089",
        display_name="Llama-3.3-70B MO (posttrained)",
        tensor_parallel_size=2,
        dtype="bfloat16",
        prompt_key="explicit_scratchpad",
        reasoning_parser="scratchpad_tags",
        comparison_family="llama",
        comparison_stage="posttrained",
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    """Return a pinned model spec with a useful error for unknown keys."""

    try:
        return MODEL_SPECS[model_key]
    except KeyError as exc:
        raise ValueError(f"unknown Figure 6 model key {model_key!r}; choose one of {sorted(MODEL_SPECS)}") from exc


def sha256_file(path: str | Path) -> str:
    """Hash a file without interpreting or redistributing its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_prompt(path: str | Path, expected_sha256: str) -> str:
    """Read a local prompt only when it has the explicitly expected digest."""

    if _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected prompt SHA-256 must be 64 lowercase hexadecimal characters")
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing system prompt: {prompt_path}")
    actual_sha256 = sha256_file(prompt_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"system prompt SHA-256 mismatch for {prompt_path}: expected {expected_sha256}, got {actual_sha256}"
        )
    prompt = prompt_path.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError(f"system prompt must not be empty: {prompt_path}")
    return prompt.strip()


def load_verified_model_prompt(model_key: str, path: str | Path) -> tuple[str, PromptSpec]:
    """Load the prompt pinned for ``model_key`` from a caller-supplied path."""

    model = get_model_spec(model_key)
    prompt_spec = model.prompt
    return load_verified_prompt(path, prompt_spec.sha256), prompt_spec


def _validate_registry() -> None:
    if tuple(FIGURE6_SOURCE_CONFIGS) != tuple(config for config in DATASET_CONFIGS if config != "prompts"):
        raise RuntimeError("Figure 6 source configs drifted from the EvalAwareBench adapter")
    if len(MODEL_SPECS) != 7 or any(key != model.key for key, model in MODEL_SPECS.items()):
        raise RuntimeError("Figure 6 model registry must contain seven uniquely keyed models")
    for model in MODEL_SPECS.values():
        if _REVISION_RE.fullmatch(model.revision) is None:
            raise RuntimeError(f"model {model.key} does not have an immutable 40-hex revision")
        if model.prompt_key not in PROMPT_SPECS:
            raise RuntimeError(f"model {model.key} names an unknown prompt")
    for prompt in PROMPT_SPECS.values():
        if _SHA256_RE.fullmatch(prompt.sha256) is None:
            raise RuntimeError(f"prompt {prompt.key} does not have a valid SHA-256")


_validate_registry()

__all__ = [
    "DATASET_ID",
    "DATASET_LICENSE",
    "DATASET_REVISION",
    "EXPLICIT_SCRATCHPAD_PROMPT_SHA256",
    "FIGURE6_CONDITIONS",
    "FIGURE6_FACTORS",
    "FIGURE6_ROW_COUNT",
    "FIGURE6_SOURCE_CONFIGS",
    "FIGURE6_TASK_COUNT",
    "FIGURE6_VALENCES",
    "MODEL_SPECS",
    "ModelSpec",
    "PAPER_NATURAL_PROMPT_SHA256",
    "PROMPT_SPECS",
    "PromptSpec",
    "UPSTREAM_CODE_REPOSITORY",
    "UPSTREAM_CODE_REVISION",
    "get_model_spec",
    "load_verified_model_prompt",
    "load_verified_prompt",
    "sha256_file",
]
