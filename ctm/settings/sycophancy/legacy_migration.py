"""One-time conversion of this repo's legacy wrong-argument sources into the
mcq_bias package's canonical per-model stores.

This module belongs to the RESEARCH repo, not the published eval package: it
reads legacy assets that only exist here (the Gemma-4 dump dir) and writes the package's canonical files
(mcq_bias/data/wrong_arguments/<model_slug>.jsonl). Idempotent; once the
canonical files are committed, this is only needed again on branches that
predate them.
"""

import json
import re
from pathlib import Path
from typing import Optional

from mcq_bias.pipeline.records import COT_INSTRUCTION, COT_TRAILER
from mcq_bias.pipeline.wrong_arguments import (
    DEFAULT_ARGUMENT_MODEL,
    WrongArgumentStore,
    append_arguments,
    arguments_path,
    model_slug,
)

# Mining helpers for the legacy dump format (cot-baked biased prompts).
_ARGUMENT_RE = re.compile(r"<argument>\n(.*?)\n</argument>", re.DOTALL)
_SUFFIX = COT_INSTRUCTION + COT_TRAILER


def _this_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def migrate_legacy_sources(repo_root: Optional[Path] = None, data_dir: Optional[Path] = None) -> dict[str, int]:
    """Convert the legacy argument sources into canonical per-model stores.

    - dataset_dumps/test/distractor_argument_g4/  → DEFAULT_ARGUMENT_MODEL's store
      (the set the Gemma-4 run produced; arguments mined from the biased prompts;
      "distractor_argument" is the legacy name of the wrong_argument bias)

    (The old gpt-3.5 TaskOutput asset is deliberately NOT migrated — those
    arguments are excluded from the published package.)

    Reads under ``repo_root`` (this repo); writes under ``data_dir`` (default:
    the package's data dir). Questions already present in a target store are
    skipped. Returns {model_slug: n_migrated}.
    """
    root = repo_root or _this_repo_root()
    migrated: dict[str, int] = {}
    stores: dict[str, WrongArgumentStore] = {}
    pending: dict[str, list[dict]] = {}

    def stage(
        model_name: str, question_id: Optional[str], parsed: Optional[str], argument: str, biased_option: str = ""
    ) -> None:
        slug = model_slug(model_name)
        if slug not in stores:
            stores[slug] = WrongArgumentStore.for_model(model_name, data_dir)
            pending[slug] = []
        probe_hit = (question_id and stores[slug]._by_id.get(question_id)) or (
            parsed and stores[slug]._by_parsed.get(parsed)
        )
        if probe_hit:
            return
        stores[slug].add(argument, parsed=parsed, question_id=question_id)
        pending[slug].append(
            {
                "question_id": question_id,
                "parsed_input": parsed,
                "wrong_argument": argument,
                "biased_option": biased_option,
                "model": model_name,
            }
        )

    # Legacy dump rows keep their legacy keys (biased_question / unbiased_question /
    # original_question_hash) — this reader is the translation boundary.
    g4_dir = root / "dataset_dumps/test/distractor_argument_g4"
    if g4_dir.is_dir():
        for path in sorted(g4_dir.glob("*.jsonl")):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    biased = rec.get("biased_question") or []
                    if not biased:
                        continue
                    m = _ARGUMENT_RE.search(biased[0].get("content", ""))
                    if not m:
                        continue
                    unb = (rec.get("unbiased_question") or [{}])[0].get("content", "")
                    parsed = unb[: -len(_SUFFIX)] if unb.endswith(_SUFFIX) else None
                    stage(
                        DEFAULT_ARGUMENT_MODEL,
                        rec.get("original_question_hash"),
                        parsed,
                        m.group(1),
                        rec.get("biased_option", ""),
                    )

    for slug, rows in pending.items():
        if rows:
            append_arguments(arguments_path(slug, data_dir), rows)
        migrated[slug] = len(rows)
    return migrated
