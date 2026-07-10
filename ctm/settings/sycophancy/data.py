"""Sycophancy setting: training datapoints and perturbation builders.

Canonical home of the data/perturbation logic that previously lived inline in
``scripts/tinker_training/train_rl.py`` (which now imports from here). Datapoint
schema: dicts with ``biased_question`` / ``unbiased_question`` message lists,
``biased_option``, ``ground_truth``, ``bias_name`` — as produced by the
dataset_dumps build scripts.
"""

import json
from pathlib import Path

from mcq_bias.pipeline.records import ANSWER_FORMAT_INSTRUCTION, COT_INSTRUCTION

from ctm.settings.sycophancy.cues import (
    DISTRACTOR_CUES,
    apply_distractor_cue,
    cue_keys as all_distractor_cue_keys,
    extract_wrong_cot,
    split_cues,
)

# ── no_cot conversion for legacy cot-baked data ──────────────────────────────
# The published package builds each prompt style natively and never converts;
# this converter exists ONLY for the legacy assets (dataset_dumps), whose
# stored messages have the CoT instructions baked in. It dies with them.

NO_COT_SUFFIX = ANSWER_FORMAT_INSTRUCTION

# The exact CoT chunks legacy formatters embedded, longest/most-specific first.
# wrong_few_shot reflowed the standard instruction across lines; post_hoc used
# its own format sentence. Frozen legacy shapes.
_LEGACY_COT_CHUNKS = (
    COT_INSTRUCTION,
    COT_INSTRUCTION.lstrip("\n"),  # distractor_fact placed it after a single newline
    "\nPlease think step by step and then give your answer in the format "
    '"Therefore, the best answer is: (X).".\n'
    "It's very important that you stick to this format.",  # wrong_few_shot reflow
    ' Then give your answer in the format "Therefore, the best answer is: (X)." '
    "It's very important that you stick to this format.",  # post_hoc format sentence
)
_LEGACY_COT_TRAILERS = ("\n\nLet's think step by step:", "\nLet's think step by step:")


def strip_cot_from_message(content: str) -> str:
    """Convert legacy cot-baked prompt content to no_cot.

    Removes the known instruction chunks in place rather than truncating, so
    content that legitimately follows the instruction (e.g. a suggested_answer
    anchor seeded to the final position) survives the conversion.
    """
    out = content
    for chunk in _LEGACY_COT_CHUNKS:
        out = out.replace(chunk, "")
    for trailer in _LEGACY_COT_TRAILERS:
        if out.endswith(trailer):
            out = out[: -len(trailer)]
            break
    if out == content:
        return content
    return out.rstrip("\n") + NO_COT_SUFFIX


def default_data_dir() -> Path:
    """dataset_dumps/test at the repo root (this file: ctm/settings/sycophancy/data.py)."""
    return Path(__file__).parent.parent.parent.parent / "dataset_dumps" / "test"


def load_datapoints(bias_types: list[str], datasets: list[str], n_datapoints: int, data_dir: Path) -> list[dict]:
    """Load and concatenate datapoints from all bias_type x dataset combinations.

    Args:
        n_datapoints: Total number of datapoints to load, split evenly across
            all bias_type x dataset combinations.
    """
    n_combos = len(bias_types) * len(datasets)
    per_combo = n_datapoints // n_combos if n_combos > 0 else n_datapoints
    datapoints = []
    missing, short = [], []
    for bias_type in bias_types:
        for dataset in datasets:
            path = data_dir / bias_type / f"{dataset}_{bias_type}.jsonl"
            if not path.exists():
                print(f"  Warning: {path} not found, skipping")
                missing.append(f"{dataset}/{bias_type}")
                continue
            loaded = []
            with open(path) as f:
                for line in f:
                    loaded.append(json.loads(line))
                    if len(loaded) >= per_combo:
                        break
            datapoints.extend(loaded)
            if len(loaded) < per_combo:
                short.append(f"{dataset}/{bias_type} ({len(loaded)}/{per_combo})")
            print(f"  Loaded {len(loaded)} from {path.name}")
    # Surface silent dataset skew: floored per_combo, missing files, and short combos all
    # change the trained mix vs bias_types/datasets intent without an error otherwise.
    if missing or short or len(datapoints) != n_datapoints:
        print(f"  ⚠️  Loaded {len(datapoints)}/{n_datapoints} requested datapoints across {n_combos} combo(s).")
        if missing:
            print(f"      Missing files (combo skipped entirely): {', '.join(missing)}")
        if short:
            print(f"      Short combos (fewer rows than per-combo target {per_combo}): {', '.join(short)}")
        print("      The trained mix may differ from intent; pass matching n_datapoints/datasets.")
    return datapoints


def _apply_prompt_style(messages: list[dict], prompt_style: str) -> list[dict]:
    """Strip CoT instructions from user messages if prompt_style is 'no_cot'."""
    if prompt_style != "no_cot":
        return messages
    out = []
    for m in messages:
        if m.get("role") == "user":
            out.append({**m, "content": strip_cot_from_message(m["content"])})
        else:
            out.append(m)
    return out


def make_perturbation_fns(prompt_style: str):
    def unbiased_perturbation(datapoint: dict) -> dict:
        return {"messages": _apply_prompt_style(datapoint["unbiased_question"], prompt_style)}

    def biased_perturbation(datapoint: dict) -> dict:
        return {"messages": _apply_prompt_style(datapoint["biased_question"], prompt_style)}

    return unbiased_perturbation, biased_perturbation


def resolve_distractor_cues(spec: str | None) -> list[str]:
    """Resolve a cue spec into a list of cue keys.

    'none'/'' -> [] (single-cue mode); 'all'/'train'/'holdout' -> registry splits;
    otherwise a comma-separated list of explicit cue keys.
    """
    if spec in (None, "", "none"):
        return []
    if spec == "all":
        return all_distractor_cue_keys()
    if spec == "train":
        return split_cues()[0]
    if spec == "holdout":
        return split_cues()[1]
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in DISTRACTOR_CUES]
    if unknown:
        raise SystemExit(f"Unknown distractor cue(s): {unknown}. Known: {all_distractor_cue_keys()}")
    return keys


def make_distractor_cue_perturbations(cues: list[str], prompt_style: str):
    """[unbiased (ref, idx 0)] + one perturbation per distractor cue (idx 1..N).

    Each cue re-frames the datapoint's pre-extracted wrong argument (dp['_wrong_cot'])
    around its unbiased question, so the same argument is presented many ways.
    """

    def unbiased(dp: dict) -> dict:
        return {"messages": _apply_prompt_style(dp["unbiased_question"], prompt_style)}

    def make_cue(key: str):
        def cue_pert(dp: dict) -> dict:
            # Strip CoT on the bare question FIRST, then wrap. Transforming the wrapped
            # blob would truncate at the CoT phrase now sitting inside <question>,
            # deleting </question> and (for question-first cues) the entire <argument> block.
            base = _apply_prompt_style(dp["unbiased_question"], prompt_style)
            return {"messages": apply_distractor_cue(base, dp["_wrong_cot"], key)}

        return cue_pert

    return [unbiased] + [make_cue(k) for k in cues]


def attach_wrong_cots(datapoints: list[dict]) -> list[dict]:
    """Keep only datapoints whose biased_question carries an extractable wrong argument,
    stashing it on dp['_wrong_cot'] for the cue perturbations."""
    kept = []
    for dp in datapoints:
        wc = extract_wrong_cot(dp.get("biased_question") or [])
        if wc:
            dp["_wrong_cot"] = wc
            kept.append(dp)
    return kept
