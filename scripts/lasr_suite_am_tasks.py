"""Generation-only LASR native-fork AM tasks; construction never calls a model.

Three fixed native fixtures, not independently sampled narratives. Source bytes
and archive pin are checked before importing the native task. This module has
only an offline-audit CLI; the coordinator separately owns live execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

from inspect_ai import Task
from inspect_ai._util.registry import registry_info, registry_params
from inspect_ai.model import ChatMessageSystem, GenerateConfig
from inspect_ai.solver import chain, solver

from scripts.lasr_transfer_tasks import (
    F8_NOTICE,
    MODELS,
    SEED_BASE,
    TARGET_CONFIG,
    canonical_messages,
    digest,
)

FORK_COMMIT = "3745dcbe815c23f77e1aefcec81327651ad9c985"
FORK_ARCHIVE_SHA256 = "5a594ff5c34c9b8bf094fce8942aad1985410ef574e3089abcf78190ea566896"
FAMILIES = ("blackmail", "leaking", "murder")
FACTORS = ("O", "F8")
LUNA_MODEL = "openrouter/openai/gpt-5.6-luna"
NATIVE_CONFIG = {
    "scenario_variant": "default",
    "goal_type": "explicit",
    "goal_value": "america",
    "urgency_type": "replacement",
    "extra_system_instructions": None,
    "prod": False,
    "test_eval_awareness": False,
    "agency_statement": "default",
    "cache": None,
    "grader_model": LUNA_MODEL,
    "grader_cache": None,
}
PACKAGE = "src/inspect_evals/agentic_misalignment/"


def verify_source(source_dir: Path | str, archive: Path | str) -> dict:
    source = Path(source_dir).resolve(strict=True)
    archive = Path(archive).resolve(strict=True)
    with archive.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != FORK_ARCHIVE_SHA256:
        raise ValueError("Pinned LASR AM archive SHA256 mismatch")
    manifest = {}
    with zipfile.ZipFile(archive) as zipped:
        if zipped.comment.decode("ascii") != FORK_COMMIT:
            raise ValueError("Pinned LASR AM archive commit mismatch")
        for entry in zipped.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if len(parts) < 2 or ".." in parts or PurePosixPath(entry.filename).is_absolute():
                raise ValueError("Unsafe archive member")
            relative = "/".join(parts[1:])
            if not (relative.startswith(PACKAGE) or relative == "src/inspect_evals/__init__.py"):
                continue
            local = source / relative
            if local.is_symlink() or source not in local.resolve(strict=True).parents:
                raise ValueError("Linked or escaping native source file")
            expected = hashlib.sha256(zipped.read(entry)).hexdigest()
            if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Native source drift: {relative}")
            manifest[relative] = expected
    if PACKAGE + "agentic_misalignment.py" not in manifest:
        raise ValueError("Native AM entrypoint missing")
    for local in (source / "src/inspect_evals").rglob("*"):
        if local.is_file() and local.suffix in {".py", ".so", ".pyd"}:
            if local.relative_to(source).as_posix() not in manifest:
                raise ValueError("Unarchived executable code in AM import tree")
    return {
        "fork_commit": FORK_COMMIT,
        "archive_sha256": observed,
        "verified_file_count": len(manifest),
        "source_files_sha256": digest(manifest),
        "native_scoring_sha256": digest(
            {key: value for key, value in manifest.items() if "/classifiers/" in key or key.endswith("/scorers.py")}
        ),
        "source_manifest": manifest,
    }


def _import_native(source: Path):
    for name, module in tuple(sys.modules.items()):
        if name == "inspect_evals" or name.startswith("inspect_evals."):
            location = getattr(module, "__file__", None)
            if location and source not in Path(location).resolve().parents:
                raise ValueError("Another inspect_evals checkout is already imported")
    import_root = str(source / "src")
    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    importlib.invalidate_caches()
    return importlib.import_module("inspect_evals.agentic_misalignment.agentic_misalignment")


def transform_messages(messages, factor: str):
    if factor not in FACTORS:
        raise ValueError("Only native O and F8 are supported")
    changed = [message.model_copy(deep=True) for message in messages]
    if factor == "F8":
        index = next((i for i, message in enumerate(changed) if message.role == "user"), None)
        if index is None or not isinstance(changed[index].content, str):
            raise ValueError("Expected native text first user message")
        changed[index].content = F8_NOTICE + changed[index].content
    return changed


@solver
def am_runtime_factor(factor: str):
    if factor not in FACTORS:
        raise ValueError("Only native O and F8 are supported")

    async def solve(state, generate):
        state.messages = transform_messages(state.messages, factor)
        return state

    return solve


def build_tasks(*, source_dir: Path | str, archive: Path | str, model_role: str, stage: str = "pilot", scenarios: list[str] | tuple[str, ...] | None = None) -> list[Task]:
    """48 one-sample tasks per role; smoke is three excluded native-O tasks."""
    if model_role not in MODELS or stage not in {"pilot", "audit", "smoke"}:
        raise ValueError("Unknown model role or stage")
    if scenarios is not None and (not scenarios or len(set(scenarios)) != len(scenarios) or set(scenarios) - set(FAMILIES)):
        raise ValueError("Unknown or duplicate AM scenario selection")
    families = FAMILIES if scenarios is None else tuple(family for family in FAMILIES if family in scenarios)
    proof = verify_source(source_dir, archive)
    module = _import_native(Path(source_dir).resolve())
    tasks = []
    adapter_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    budget_contract_sha = digest({"models": MODELS, "target_config": TARGET_CONFIG, "seed_base": SEED_BASE})
    for family in families:
        # Create once per family; repeated cells are copies of one native fixture.
        native = module.agentic_misalignment(scenario=family, **copy.deepcopy(NATIVE_CONFIG))
        steps = list(native.solver)
        if len(native.dataset) != 1 or [registry_info(step).name.split("/")[-1] for step in steps] != [
            "system_message",
            "generate",
        ]:
            raise ValueError("Unexpected native AM task topology")
        if native.sandbox is not None or native.message_limit != 3:
            raise ValueError("Unexpected native AM sandbox/message contract")
        if registry_params(steps[1]) != {"tool_calls": "none", "kwargs": {"cache": None}}:
            raise ValueError("Native text-only generation contract changed")
        if len(native.scorer or []) != 1 or registry_info(native.scorer[0]).name.split("/")[-1] != "harmfulness_scorer":
            raise ValueError("Unexpected native behavior scorer")
        sample = native.dataset[0]
        system = registry_params(steps[0])["template"]
        native_messages = [ChatMessageSystem(content=system), *sample.input]
        original = canonical_messages(native_messages)
        native_contract = {
            "fork_commit": FORK_COMMIT,
            "source_files_sha256": proof["source_files_sha256"],
            "native_scoring_sha256": proof["native_scoring_sha256"],
            "native_kwargs": {"scenario": family, **NATIVE_CONFIG},
            "original_messages": original,
            "actual_tools": [],
            "native_generate": registry_params(steps[1]),
            "message_limit": 3,
        }
        native_id = digest(native_contract)
        block_id = digest({"native_contract": native_id, "budget_contract": budget_contract_sha})
        for factor in (("O",) if stage == "smoke" else FACTORS):
            expected_runtime = canonical_messages(transform_messages(native_messages, factor))
            config_id = digest({"block": block_id, "factor": factor, "messages": expected_runtime})
            for repeat in (range(1) if stage == "smoke" else range(8)):
                episode = copy.deepcopy(sample)
                envelope = {
                    "schema_version": 1,
                    "protocol": "lasr_native_fork_am_o_f8_v1",
                    "model_role": model_role,
                    "model_id": MODELS[model_role][0],
                    "model_revision": MODELS[model_role][1],
                    "scenario": family,
                    "fixture_id": f"{family}_explicit-america_replacement_default",
                    "factor_id": factor,
                    "replicate_id": repeat,
                    "seed": SEED_BASE + repeat,
                    "comparison_block_id": block_id,
                    "configuration_id": config_id,
                    "native_protocol_id": native_id,
                    "native_scoring_id": proof["native_scoring_sha256"],
                }
                episode.id = f"lasr-am-{family}-{factor}-r{repeat}"
                episode.metadata = {
                    **(episode.metadata or {}),
                    "lasr_suite_am": envelope,
                    "partition": "excluded_smoke" if stage == "smoke" else "development_pilot",
                    "prompt_audit": {
                        "original_messages": original,
                        "runtime_messages": expected_runtime,
                        "original_messages_sha256": digest(original),
                        "runtime_messages_sha256": digest(expected_runtime),
                        "native_contract": native_contract,
                        "allowed_edit": "none" if factor == "O" else "prefix_first_actual_user_only",
                    },
                }
                tasks.append(
                    Task(
                        name=f"lasr_am_{family}_{factor}_r{repeat}",
                        dataset=[episode],
                        solver=chain(steps[0], am_runtime_factor(factor), steps[1]),
                        scorer=None,
                        metrics=[],
                        epochs=1,
                        message_limit=3,
                        config=GenerateConfig(**copy.deepcopy(TARGET_CONFIG), seed=SEED_BASE + repeat),
                        metadata={
                            "lasr_suite_am": envelope,
                            "source_proof": proof,
                            "adapter_sha256": adapter_sha,
                            "budget_contract_sha256": budget_contract_sha,
                            "generation_only": True,
                            "external_actions": False,
                        },
                    )
                )
    return tasks


def audit_summary(tasks: list[Task]) -> dict:
    cells = [task.dataset[0].metadata["lasr_suite_am"] for task in tasks]
    prompts = {}
    for task in tasks:
        sample = task.dataset[0]
        cell = sample.metadata["lasr_suite_am"]
        prompts.setdefault(cell["scenario"] + "/" + cell["factor_id"], sample.metadata["prompt_audit"])
    return {
        "status": "offline_constructed_no_model_calls",
        "episodes": len(tasks),
        "native_fixture_count": len({cell["scenario"] for cell in cells}),
        "repeats_are_same_fixture": True,
        "scoring_enabled": False,
        "target_config": TARGET_CONFIG,
        "inspect_ai_version": importlib.metadata.version("inspect-ai"),
        "cells": cells,
        "cells_sha256": digest(cells),
        "prompt_audits": prompts,
        "source_proof": tasks[0].metadata["source_proof"],
        "adapter_sha256": tasks[0].metadata["adapter_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit output exists; select a fresh output path")
    tasks = [
        task
        for role in MODELS
        for task in build_tasks(source_dir=args.source_dir, archive=args.archive, model_role=role, stage="audit")
    ]
    result = audit_summary(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "episodes": result["episodes"],
                "source_files": result["source_proof"]["verified_file_count"],
                "cells_sha256": result["cells_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
