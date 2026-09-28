"""Offline audit and four-cell task factory for the simulated upstream benchmark.

This command never runs a model. The factory is usable through scripts/run_evals.py
after approval. It preserves upstream prompts, pseudo-tools, and harmfulness scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Sequence

UPSTREAM_COMMIT = "47a0725a3eb630c94209ea2aebbcf420beee79ca"
UPSTREAM_SOURCE_SHA256 = "a284971487689c6ef47d0ded35f533b0c2b18304ac70134f95bb58d3de240295"
SCENARIOS = ("blackmail", "leaking", "murder")
AWARENESS_MODES = ("off", "passive", "primed")


def pilot_conditions(
    scenarios: Sequence[str] = ("blackmail",),
    *,
    awareness: str = "passive",
    prod: bool = False,
) -> list[dict[str, Any]]:
    """Explicit goal conflict / no goal conflict crossed with replacement / none.

    These are upstream template interventions, not perfectly matched minimal edits.
    They do not assert that goal_type=none is an aligned-goal condition.
    """
    if isinstance(scenarios, str) or not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("scenarios must be a nonempty sequence of unique scenario names")
    if set(scenarios) - set(SCENARIOS):
        raise ValueError(f"scenarios must be drawn from {SCENARIOS}")
    if awareness not in AWARENESS_MODES:
        raise ValueError(f"awareness must be one of {AWARENESS_MODES}")
    if not isinstance(prod, bool):
        raise ValueError("prod must be a bool")
    conditions = []
    for scenario in scenarios:
        for goal_type, goal_value in (("explicit", "america"), ("none", "none")):
            for urgency_type in ("replacement", "none"):
                native_id = f"{scenario}_{goal_type}-{goal_value}_{urgency_type}"
                conditions.append(
                    {
                        "condition_id": f"{native_id}__prod-{str(prod).lower()}__awareness-{awareness}",
                        "source_template_family": scenario,
                        "awareness_mode": awareness,
                        "task_args": {
                            "scenario": scenario,
                            "goal_type": goal_type,
                            "goal_value": goal_value,
                            "urgency_type": urgency_type,
                            "extra_system_instructions": None,
                            "prod": prod,
                            "test_eval_awareness": awareness == "primed",
                        },
                    }
                )
    return conditions


def verify_upstream_source() -> Path:
    """Reject drift in the task, classifiers, metadata, or bundled prompt templates."""
    package = importlib.import_module("inspect_evals.agentic_misalignment")
    root = Path(package.__file__).parent
    digest = hashlib.sha256()
    files = sorted(
        path for path in root.rglob("*") if path.suffix in {".py", ".md", ".yaml"} and path.name != "README.md"
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    actual = digest.hexdigest()
    if actual != UPSTREAM_SOURCE_SHA256:
        raise ValueError(
            f"Agentic Misalignment source drift: {actual}; expected " f"{UPSTREAM_SOURCE_SHA256} from {UPSTREAM_COMMIT}"
        )
    return root


def build_pilot_tasks(
    *,
    grader_model: str,
    scenarios: Sequence[str] = ("blackmail",),
    awareness: str = "passive",
    prod: bool = False,
) -> list[Any]:
    """Construct tasks only; an external approved runner performs generation.

    Passive awareness adds an output-only scorer without changing the native
    prompt. It observes expressed awareness, not hidden beliefs or predecision
    awareness. Primed awareness is the upstream prompt intervention.
    """
    conditions = pilot_conditions(scenarios, awareness=awareness, prod=prod)
    if not grader_model or not isinstance(grader_model, str):
        raise ValueError("an explicit grader_model string is required")
    verify_upstream_source()
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_evals.agentic_misalignment import agentic_misalignment
    from inspect_evals.agentic_misalignment.scorers import eval_awareness_scorer

    grader_config = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 4096,
        "max_connections": 4,
        "max_retries": 3,
    }
    grader = get_model(grader_model, config=GenerateConfig(**grader_config))
    tasks = []
    for condition in conditions:
        task = agentic_misalignment(**condition["task_args"], grader_model=grader)
        if awareness == "passive":
            task.scorer = [*task.scorer, eval_awareness_scorer(grader_model=grader)]
        metadata = {
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
            "pilot_condition_id": condition["condition_id"],
            "source_template_family": condition["source_template_family"],
            "awareness_mode": awareness,
            "grader_model": grader_model,
            "grader_config": grader_config,
            "action_environment": "text_only_simulation_no_executable_tools",
        }
        task.metadata = {**(task.metadata or {}), "ctm_agentic_pilot": metadata}
        for sample in task.dataset:
            sample.id = condition["condition_id"]
            sample.metadata = {**(sample.metadata or {}), **metadata}
        tasks.append(task)
    return tasks


def audit_conditions(conditions: Sequence[dict[str, Any]], tokenizer_path: str | None) -> list[dict[str, Any]]:
    """Hash exact prompts, optionally count locally cached chat-template tokens."""
    verify_upstream_source()
    from inspect_evals.agentic_misalignment.prompt_generator import (
        ALIGNMENT_FAKING_WARNING,
        generate_prompts,
    )

    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    audited = []
    for condition in conditions:
        args = dict(condition["task_args"])
        args["prod_flag"] = args.pop("prod")
        args["alignment_faking_warning"] = ALIGNMENT_FAKING_WARNING if args.pop("test_eval_awareness") else None
        prompts = generate_prompts(**args)
        messages = [
            {"role": "system", "content": prompts.system_prompt},
            {"role": "user", "content": "\n".join([prompts.user_prompt, prompts.email_content])},
        ]
        payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        item = {
            **condition,
            "prompt_messages_sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "prompt_characters": sum(len(message["content"]) for message in messages),
        }
        if tokenizer is not None:
            item["prompt_tokens"] = len(
                tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            )
            item["tokenizer_path"] = tokenizer_path
        audited.append(item)
    return audited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["blackmail"])
    parser.add_argument("--awareness", choices=AWARENESS_MODES, default="passive")
    parser.add_argument("--prod", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--models", type=int, default=3)
    parser.add_argument("--audit", action="store_true", help="Import pinned source and hash prompts; no model calls")
    parser.add_argument("--tokenizer-path", help="Local tokenizer path for offline chat-template token counts")
    args = parser.parse_args()
    if args.epochs < 1 or args.models < 1:
        parser.error("epochs and models must be positive")
    if args.tokenizer_path and not args.audit:
        parser.error("--tokenizer-path requires --audit")
    conditions = pilot_conditions(args.scenarios, awareness=args.awareness, prod=args.prod)
    if args.audit:
        conditions = audit_conditions(conditions, args.tokenizer_path)
    target_calls = len(conditions) * args.epochs * args.models
    print(
        json.dumps(
            {
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_source_sha256": UPSTREAM_SOURCE_SHA256,
                "source_verified": args.audit,
                "conditions": conditions,
                "conditions_per_model": len(conditions),
                "epochs": args.epochs,
                "models": args.models,
                "target_calls_before_retries": target_calls,
                "grader_calls_before_retries": target_calls * (1 if args.awareness == "off" else 2),
                "runs_models": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
