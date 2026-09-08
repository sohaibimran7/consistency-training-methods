"""Inspect task factory for the staged ELEPHANT AITA-NTA-FLIP benchmark.

Each task is one of exactly four deterministic shards.  A shard contains both
perspectives for every selected AITA ID, so distributing it across GPUs never
turns the paired moral-sycophancy outcome into an accidental cross-worker
join.  There is intentionally no model-based scorer: the completed EvalLogs
are parsed and scored locally by :mod:`.preflight`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Keep manifest/preflight tooling importable on non-evaluator hosts.
    from inspect_ai import Task, task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import GenerateConfig

    _INSPECT_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised by lightweight desktop installs
    # The generic runner imports Inspect before invoking the factory, but
    # allowing this module to import without it lets local custody/audit tools
    # inspect ``task_specs`` without pulling in the evaluator runtime.
    Task = Any  # type: ignore[misc,assignment]
    Sample = Any  # type: ignore[misc,assignment]
    GenerateConfig = Any  # type: ignore[misc,assignment]
    _INSPECT_IMPORT_ERROR = exc

    def task(function: Any = None, **_kwargs: Any) -> Any:
        def decorate(candidate: Any) -> Any:
            return candidate

        return decorate(function) if function is not None else decorate

from .prepare import (
    BENCHMARK,
    CONCURRENCY_CONFIG,
    GENERATION_CONFIG,
    MANIFEST_SCHEMA,
    NO_TOKEN_CAP_POLICY,
    NUM_SHARDS,
    PROMPT_SUFFIX,
    RUNTIME_GENERATION_CONFIG,
    assert_no_token_cap_mapping,
    load_frozen_pairs,
    manifest_sha256,
)


TASK_NAME = "aita_nta_flip_shard"
PERSPECTIVES = ("flipped", "original")


@dataclass(frozen=True, slots=True)
class AITATaskSpec:
    """One immutable, pair-preserving evaluation shard."""

    manifest_path: str
    manifest_sha256: str
    shard_index: int
    n_shards: int
    pair_ids: tuple[str, ...]
    pair_count: int
    generation_count: int
    pair_artifact_sha256: str
    manifest_schema: str
    no_token_cap_policy: dict[str, Any] | None


def _require_inspect() -> None:
    if _INSPECT_IMPORT_ERROR is not None:
        raise RuntimeError("Inspect AI is required to construct an AITA-NTA-FLIP evaluation task") from _INSPECT_IMPORT_ERROR


def _require_shard_index(value: int, *, n_shards: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= n_shards:
        raise ValueError(f"shard_index must be an integer in [0, {n_shards}), got {value!r}")
    return value


def _require_n_shards(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != NUM_SHARDS:
        raise ValueError(f"AITA-NTA-FLIP requires exactly {NUM_SHARDS} pair-preserving shards")
    return value


def task_specs(manifest: str | Path, n_shards: int = NUM_SHARDS) -> list[AITATaskSpec]:
    """Replay the manifest's full frozen four-shard topology.

    The task factory reloads this information independently on each worker;
    it never accepts a caller-provided subset of pair IDs.
    """

    _require_n_shards(n_shards)
    document, manifest_path, _ = load_frozen_pairs(manifest)
    manifest_identity = manifest_sha256(manifest_path)
    artifact = document["pair_artifact"]
    manifest_schema = document.get("schema")
    if not isinstance(manifest_schema, str):
        raise ValueError("AITA-NTA manifest schema must be a string")
    no_token_cap_policy = document.get("no_token_cap_policy")
    if manifest_schema == MANIFEST_SCHEMA:
        if no_token_cap_policy != NO_TOKEN_CAP_POLICY:
            raise ValueError("AITA-NTA r005 task requires the exact no-token-cap policy")
        assert_no_token_cap_mapping(document.get("sampling_config"), label="AITA-NTA r005 task sampling_config")
        assert_no_token_cap_mapping(document.get("generation_config"), label="AITA-NTA r005 task generation_config")
    elif no_token_cap_policy is not None:
        raise ValueError("legacy AITA-NTA manifests must not declare an r005 no-token-cap policy")
    result: list[AITATaskSpec] = []
    for entry in document["shards"]:
        if not isinstance(entry, Mapping):  # validated by load_frozen_pairs; retain a direct caller guard.
            raise ValueError("AITA-NTA manifest shard must be an object")
        shard_index = _require_shard_index(entry["shard_index"], n_shards=n_shards)
        pair_ids = entry["pair_ids"]
        if not isinstance(pair_ids, list) or any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids):
            raise ValueError("AITA-NTA manifest shard has invalid pair IDs")
        result.append(
            AITATaskSpec(
                manifest_path=str(manifest_path),
                manifest_sha256=manifest_identity,
                shard_index=shard_index,
                n_shards=n_shards,
                pair_ids=tuple(pair_ids),
                pair_count=int(entry["pair_count"]),
                generation_count=int(entry["generation_count"]),
                pair_artifact_sha256=str(artifact["content_sha256"]),
                manifest_schema=manifest_schema,
                no_token_cap_policy=dict(no_token_cap_policy) if isinstance(no_token_cap_policy, Mapping) else None,
            )
        )
    if [spec.shard_index for spec in result] != list(range(n_shards)):
        raise ValueError("AITA-NTA manifest shard ordering is not canonical")
    return result


def _sample_id(pair_id: str, perspective: str) -> str:
    if perspective not in PERSPECTIVES:
        raise ValueError(f"unknown AITA perspective {perspective!r}")
    return f"{pair_id}::{perspective}"


def _prompt(post: str) -> str:
    if not isinstance(post, str) or not post:
        raise ValueError("AITA post must be a non-empty string")
    # Do not strip or rewrite posts: the source text plus this literal suffix
    # is the paper's constrained binary-judgement protocol.
    return post + PROMPT_SUFFIX


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _task_metadata(spec: AITATaskSpec, *, eos_only_runtime: str = "qwen_r005") -> dict[str, Any]:
    metadata = {
        "benchmark": BENCHMARK,
        "schema": spec.manifest_schema,
        "manifest_sha256": spec.manifest_sha256,
        "pair_artifact_sha256": spec.pair_artifact_sha256,
        "shard_index": spec.shard_index,
        "n_shards": spec.n_shards,
        "pair_count": spec.pair_count,
        "generation_count": spec.generation_count,
        "prompt_suffix": PROMPT_SUFFIX,
        "system_prompt": None,
        # Keep the paper sampling protocol distinct from execution-only
        # batching, while attesting the exact union passed to Inspect.
        "sampling_config": dict(GENERATION_CONFIG),
        "concurrency_config": dict(CONCURRENCY_CONFIG),
        "generation_config": dict(RUNTIME_GENERATION_CONFIG),
    }
    if spec.no_token_cap_policy is not None:
        # The manifest policy describes the prohibited-cap contract; the
        # process-local policy attests the only supported implementation of
        # that contract, and the Qwen policy attests template-level thinking
        # suppression.  All three are checked again from the EvalLog header.
        metadata["no_token_cap_policy"] = dict(spec.no_token_cap_policy)
        if eos_only_runtime == "qwen_r005":
            from .no_cap_hf import QWEN_THINKING_POLICY, runtime_policy

            metadata.update(
                {
                    "qwen_thinking_policy": dict(QWEN_THINKING_POLICY),
                    "no_token_cap_runtime_policy": runtime_policy(),
                }
            )
        elif eos_only_runtime == "generic":
            from ctm.evals.hf_eos_only import runtime_policy

            metadata["no_token_cap_runtime_policy"] = runtime_policy()
        else:
            raise ValueError("eos_only_runtime must be 'qwen_r005' or 'generic'")
    return metadata


def _samples_for_spec(spec: AITATaskSpec, rows: Sequence[Mapping[str, str]]) -> list[Sample]:
    _require_inspect()
    rows_by_id = {row.get("pair_id"): row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("verified AITA-NTA staged rows unexpectedly have duplicate pair IDs")
    selected: list[Sample] = []
    for pair_id in spec.pair_ids:
        row = rows_by_id.get(pair_id)
        if row is None:
            raise ValueError(f"AITA-NTA manifest shard references a missing pair ID {pair_id!r}")
        for perspective in PERSPECTIVES:
            field = "flipped_post" if perspective == "flipped" else "original_post"
            prompt = _prompt(str(row[field]))
            selected.append(
                Sample(
                    id=_sample_id(pair_id, perspective),
                    input=prompt,
                    metadata={
                        "benchmark": BENCHMARK,
                        "schema": spec.manifest_schema,
                        "manifest_sha256": spec.manifest_sha256,
                        "pair_artifact_sha256": spec.pair_artifact_sha256,
                        "pair_id": pair_id,
                        "perspective": perspective,
                        "shard_index": spec.shard_index,
                        "n_shards": spec.n_shards,
                        "prompt_suffix": PROMPT_SUFFIX,
                        "system_prompt": None,
                        "prompt_sha256": _prompt_sha256(prompt),
                        **(
                            {
                                "no_token_cap_policy": dict(spec.no_token_cap_policy),
                            }
                            if spec.no_token_cap_policy is not None
                            else {}
                        ),
                    },
                )
            )
    if len(selected) != spec.generation_count:
        raise RuntimeError("internal AITA-NTA task construction produced the wrong generation count")
    return selected


@task(name=TASK_NAME)
def aita_nta_flip_shard(
    manifest: str,
    shard_index: int,
    n_shards: int = NUM_SHARDS,
    eos_only_runtime: str = "qwen_r005",
) -> Task:
    """Return one full, pair-preserving AITA-NTA-FLIP Inspect task shard.

    This is directly compatible with the generic runner's
    ``--task-factory experiments.elephant_aita_ntaflip.tasks:aita_nta_flip_shard``
    interface.  ``Task.config`` duplicates the frozen generation protocol as
    a local safety net; production launchers should pass the same dictionary
    (including the execution-only concurrency cap) through their standard
    ``--generation-config`` argument.  The r005 task installs its EOS-only
    native-HF path before model construction, so omitted token limits cannot
    be replaced by an Inspect or Transformers default.
    """

    _require_inspect()
    _require_n_shards(n_shards)
    index = _require_shard_index(shard_index, n_shards=n_shards)
    specs = task_specs(manifest, n_shards=n_shards)
    spec = specs[index]
    document, _, rows = load_frozen_pairs(manifest)
    # This is redundant with task_specs' validation but protects against a
    # source artifact changing between the two reads in a caller that has
    # ignored the immutable staging requirement.
    if document["pair_artifact"]["content_sha256"] != spec.pair_artifact_sha256:
        raise ValueError("AITA-NTA staged pair artifact changed during task construction")
    if eos_only_runtime not in {"qwen_r005", "generic"}:
        raise ValueError("eos_only_runtime must be 'qwen_r005' or 'generic'")
    if spec.no_token_cap_policy is not None and eos_only_runtime == "qwen_r005":
        # This must happen before scripts/run_evals.py resolves either a base
        # HF model or a local PEFT model.  Inspect 0.3.258 otherwise inserts a
        # hidden max_tokens default when the task invokes the model.
        from .no_cap_hf import install_native_hf_eos_only_sampling, runtime_policy

        installed_policy = install_native_hf_eos_only_sampling()
        if installed_policy != runtime_policy():
            raise RuntimeError("AITA r005 no-token-cap runtime policy differs from the frozen manifest")
    elif spec.no_token_cap_policy is not None:
        from ctm.evals.hf_eos_only import install_native_hf_eos_only_sampling

        install_native_hf_eos_only_sampling()
    metadata = _task_metadata(spec, eos_only_runtime=eos_only_runtime)
    return Task(
        dataset=_samples_for_spec(spec, rows),
        config=GenerateConfig(**RUNTIME_GENERATION_CONFIG, system_message=None),
        scorer=None,
        metadata=metadata,
    )


__all__ = [
    "AITATaskSpec",
    "PERSPECTIVES",
    "TASK_NAME",
    "aita_nta_flip_shard",
    "task_specs",
]
