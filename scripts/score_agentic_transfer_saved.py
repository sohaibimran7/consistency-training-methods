#!/usr/bin/env python3
"""Score frozen pilot Inspect logs without running any target-model solver.

With the original profile, `available` calls AM native + awareness judges and
IE awareness only; it never retries the unavailable direct IE judge. Explicit
`luna_all` applies Luna to both native rubrics and awareness, including IE.
Factory construction is offline; scoring itself can make paid judge calls and
requires approval. Output logs are private copies, not score-only exports.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from inspect_ai import score as inspect_score
from inspect_ai._util.registry import registry_unqualified_name
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from pydantic_core import to_jsonable_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agentic_transfer_tasks import (
    AWARENESS_CONFIG,
    AWARENESS_GRADER,
    build_pilot_tasks,
    transfer_native,
)


class SavedScoringError(ValueError):
    """Static errors deliberately omit source content and paths."""


def require(condition: bool) -> None:
    if not condition:
        raise SavedScoringError("invalid_frozen_pilot_scoring_contract")


def digest(value) -> str:
    payload = json.dumps(to_jsonable_python(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def canonical_source_input(value):
    """Compare typed source input without fresh top-level ChatMessage IDs.

    Preserve every other message field, including tool-call IDs, nested content
    IDs and metadata. This is comparison-only; never rewrite a frozen message.
    Strings remain distinct from single-message lists.
    """
    if isinstance(value, str):
        return value
    require(isinstance(value, list))
    message_types = (ChatMessageSystem, ChatMessageUser, ChatMessageAssistant, ChatMessageTool)
    require(all(isinstance(message, message_types) for message in value))
    return [message.model_dump(mode="json", exclude={"id"}) for message in value]


def canonical_target(value) -> list[str]:
    """Match Inspect Target's scalar-to-singleton-list normalization exactly."""
    require(isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value)))
    return [value] if isinstance(value, str) else list(value)


def native_judge_kwargs(judge_profile: str) -> dict:
    require(judge_profile in {"original", "luna_all"})
    return (
        {}
        if judge_profile == "original"
        else {"grader_model": AWARENESS_GRADER, "grader_config": dict(AWARENESS_CONFIG)}
    )


def judge_profile_contract(judge_profile: str) -> dict:
    overrides = native_judge_kwargs(judge_profile)
    return {
        "judge_profile": judge_profile,
        "native_override": overrides or None,
        "awareness_model": AWARENESS_GRADER,
        "awareness_config": dict(AWARENESS_CONFIG),
        "native_rubric_policy": "unchanged_native_scorer_and_original_source_input",
    }


def apply_judge_profile(copied, judge_profile: str) -> dict:
    """Re-key only the copied scoring envelope after original-contract validation."""
    contract = judge_profile_contract(judge_profile)
    mappings = {}
    if judge_profile == "luna_all":
        for sample in copied.samples or []:
            original = dict(sample.metadata["agentic_transfer"])
            native_id = digest(
                {
                    "source_native_scorer_id": original["native_scorer_id"],
                    "judge_profile": judge_profile,
                    **native_judge_kwargs(judge_profile),
                }
            )
            block_id = digest(
                {"source_comparison_block_id": original["comparison_block_id"], "native_scorer_id": native_id}
            )
            config_id = digest(
                {"source_configuration_id": original["configuration_id"], "comparison_block_id": block_id}
            )
            replacement = {
                **original,
                "native_scorer_id": native_id,
                "comparison_block_id": block_id,
                "configuration_id": config_id,
            }
            sample.metadata = {
                **sample.metadata,
                "agentic_transfer_original": original,
                "agentic_transfer": replacement,
            }
            mapping = {
                "old": {key: original[key] for key in ("native_scorer_id", "comparison_block_id", "configuration_id")},
                "new": {
                    key: replacement[key] for key in ("native_scorer_id", "comparison_block_id", "configuration_id")
                },
                "awareness_judge_id_unchanged": original["awareness_judge_id"],
            }
            mappings[digest(mapping)] = mapping
    record = {
        **contract,
        "envelope_id_mappings": [mappings[key] for key in sorted(mappings)],
        "original_envelopes_retained": judge_profile == "luna_all",
    }
    copied.eval.metadata = {**(copied.eval.metadata or {}), "agentic_transfer_judge_profile": record}
    return record


def file_digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class SavedOutputOnlyModel(Model):
    """Active scoring context with no target transport and a hard generation stop."""

    def __init__(self):
        mock = get_model("mockllm/agentic-transfer-saved-output-only")
        super().__init__(mock.api, GenerateConfig())
        self.forbidden_generation_attempts = 0

    async def generate(self, *args, **kwargs):
        self.forbidden_generation_attempts += 1
        raise SavedScoringError("target_generation_forbidden_during_saved_scoring")


def frozen_payload_digest(log) -> str:
    """Exclude only scorer-owned metadata/events/scores; hash target evidence."""
    return digest(
        [
            {
                key: getattr(sample, key)
                for key in ("id", "epoch", "input", "target", "choices", "messages", "output", "error")
            }
            for sample in log.samples or []
        ]
    )


def select_scorers(task, benchmark: str, mode: str, judge_profile: str = "original") -> tuple[list, list[str]]:
    require(mode in {"available", "all"})
    require(benchmark in {"agentic_misalignment", "instrumental_eval"})
    scorers = {registry_unqualified_name(s): s for s in task.scorer or []}
    require(set(scorers) == {"transfer_native", "transfer_awareness"})
    overrides = native_judge_kwargs(judge_profile)
    if overrides:
        scorers["transfer_native"] = transfer_native(benchmark, **overrides)
    names = (
        ["transfer_awareness"]
        if mode == "available" and benchmark == "instrumental_eval" and judge_profile == "original"
        else ["transfer_native", "transfer_awareness"]
    )
    return [scorers[name] for name in names], names


def reconstruct_log(log, *, tasks: list, mode: str, judge_profile: str = "original"):
    """Restore pinned source/task metadata on a copy; never apply factor solvers."""
    require(log.status in {"success", "error", "cancelled"})
    require(bool(log.samples))
    lookup = {}
    for task in tasks:
        for source in task.dataset:
            envelope = source.metadata["agentic_transfer"]
            key = (envelope["benchmark"], envelope["factor_id"], envelope["scenario_id"])
            require(key not in lookup)
            lookup[key] = (task, source)
    copied = log.model_copy(deep=True)
    matched_task = None
    benchmark = None
    trial_keys = []
    for sample in copied.samples:
        envelope = sample.metadata.get("agentic_transfer")
        require(isinstance(envelope, dict))
        key = (envelope.get("benchmark"), envelope.get("factor_id"), envelope.get("scenario_id"))
        require(key in lookup)
        task, source = lookup[key]
        require(matched_task is None or matched_task is task)
        matched_task, benchmark = task, key[0]
        require(envelope == source.metadata["agentic_transfer"])
        require(sample.id == source.id and type(sample.epoch) is int and sample.epoch >= 1)
        # Sample.input is the native source input. F8 exists only in messages;
        # reapplying its solver here would unblind the native rubric input.
        require(digest(canonical_source_input(sample.input)) == digest(canonical_source_input(source.input)))
        require(canonical_target(sample.target) == canonical_target(source.target))
        restored = {**(task.metadata or {}), **source.metadata}
        for name, value in restored.items():
            require(name not in sample.metadata or digest(sample.metadata[name]) == digest(value))
        sample.metadata = {**sample.metadata, **restored}
        trial_keys.append((envelope["model_role"], *key, sample.epoch))
    require(matched_task is not None and log.eval.task == matched_task.name)
    require(len(set(trial_keys)) == len(trial_keys))
    # The original factory/envelope/source checks above must precede overrides.
    apply_judge_profile(copied, judge_profile)
    selected, names = select_scorers(matched_task, benchmark, mode, judge_profile)
    return copied, selected, names, trial_keys


def score_saved_directory(
    input_dir: Path,
    output_dir: Path,
    dataset_dir: Path,
    *,
    mode: str,
    log_pattern: str = "*.eval",
    judge_profile: str = "original",
    task_builder=build_pilot_tasks,
) -> dict:
    """Preflight every frozen log before making the first approved judge call."""
    require(mode in {"available", "all"})
    profile_contract = judge_profile_contract(judge_profile)
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    require(input_dir.is_dir() and not output_dir.exists())
    require(not output_dir.is_relative_to(input_dir))
    require(isinstance(log_pattern, str) and bool(log_pattern.strip()))
    require(not Path(log_pattern).is_absolute() and ".." not in Path(log_pattern).parts)
    # Freeze the complete selected list before preflight or any judge calls.
    # Nonmatching, still-running logs belong to a separate scoring batch.
    paths = sorted(path for path in input_dir.rglob(log_pattern) if path.is_file() and path.suffix == ".eval")
    require(bool(paths))
    require(all(path.resolve().is_relative_to(input_dir) for path in paths))
    prepared, task_cache, all_trials, all_digests = [], {}, set(), set()
    for path in paths:
        source_sha = file_digest(path)
        require(source_sha not in all_digests)
        all_digests.add(source_sha)
        log = read_eval_log(path, resolve_attachments=True)
        require(bool(log.samples))
        roles = {sample.metadata.get("agentic_transfer", {}).get("model_role") for sample in log.samples}
        require(len(roles) == 1 and next(iter(roles)) in {"base", "mo_mid", "mo_post"})
        role = next(iter(roles))
        if role not in task_cache:
            # Model/scorer construction is offline, including IE's native model.
            # Only select_scorers controls which judges can actually be invoked.
            task_cache[role] = task_builder(dataset_dir=str(dataset_dir), model_role=role)
        copied, scorers, names, trial_keys = reconstruct_log(
            log, tasks=task_cache[role], mode=mode, judge_profile=judge_profile
        )
        require(not all_trials.intersection(trial_keys))
        all_trials.update(trial_keys)
        require(file_digest(path) == source_sha)
        prepared.append((path, source_sha, copied, scorers, names))
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for path, source_sha, copied, scorers, names in prepared:
        require(file_digest(path) == source_sha)
        before = frozen_payload_digest(copied)
        guard = SavedOutputOnlyModel()
        result = inspect_score(
            copied,
            scorers=scorers,
            metrics=[],
            action="overwrite",
            copy=True,
            model=guard,
            model_roles={role: guard for role in (copied.eval.model_roles or {})},
            display="none",
        )
        require(guard.forbidden_generation_attempts == 0)
        require(frozen_payload_digest(copied) == before and frozen_payload_digest(result) == before)
        require(result.eval.model == copied.eval.model and file_digest(path) == source_sha)
        skipped = ["transfer_native"] if names == ["transfer_awareness"] else []
        score_counts = Counter()
        for sample in result.samples or []:
            require(set(sample.scores or {}) == set(names))
            for name, score in (sample.scores or {}).items():
                key = "awareness_status" if name == "transfer_awareness" else "score_status"
                score_counts[f"{name}:{(score.metadata or {}).get(key, 'missing_status')}"] += 1
        record = {
            "schema_version": 1,
            "mode": mode,
            "log_pattern": log_pattern,
            "judge_profile_contract": copied.eval.metadata["agentic_transfer_judge_profile"],
            "status": "partial_scoring" if skipped else "requested_scorers_applied",
            "source_log_sha256": source_sha,
            "frozen_target_evidence_sha256": before,
            "target_generation_calls": 0,
            "applied_scorers": names,
            "skipped_scorers": skipped,
            "skip_reason": "native_IE_direct_OpenAI_authentication_unavailable_not_retried" if skipped else None,
            "sample_count": len(result.samples or []),
            "score_status_counts": dict(sorted(score_counts.items())),
            "scoring_script_sha256": file_digest(Path(__file__)),
        }
        # EvalSpec.metadata survives .eval serialization; EvalLog.metadata is a
        # transient reader field in the installed Inspect version.
        result.eval.metadata = {**(result.eval.metadata or {}), "agentic_transfer_posthoc": record}
        output = output_dir / f"{source_sha}.eval"
        require(not output.exists())
        write_eval_log(result, output, format="eval")
        record = {**record, "output_log_sha256": file_digest(output)}
        with (output_dir / f"{source_sha}.scoring.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        records.append(record)
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "log_pattern": log_pattern,
        "judge_profile_contract": profile_contract,
        "status": (
            "completed_partial_scoring" if any(r["skipped_scorers"] for r in records) else "completed_requested_scoring"
        ),
        "target_generation_calls": 0,
        "source_log_count": len(records),
        "sample_count": sum(r["sample_count"] for r in records),
        "records": records,
    }
    with (output_dir / "scoring-manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("available", "all"), default="available")
    parser.add_argument(
        "--judge-profile",
        choices=("original", "luna_all"),
        default="original",
        help="luna_all replaces both native judges with the existing Luna awareness configuration",
    )
    parser.add_argument(
        "--log-pattern", default="*.eval", help="Relative glob under input-dir; matching logs must all be complete"
    )
    args = parser.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        manifest = score_saved_directory(
            args.input_dir,
            args.output_dir,
            args.dataset_dir,
            mode=args.mode,
            log_pattern=args.log_pattern,
            judge_profile=args.judge_profile,
        )
    except Exception:
        # Preserve completed copies on failure; never echo private logs or paths.
        print(json.dumps({"error_type": "AgenticTransferSavedScoringError", "partial_outputs_may_exist": True}))
        return 2
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "mode",
                    "log_pattern",
                    "source_log_count",
                    "sample_count",
                    "target_generation_calls",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
