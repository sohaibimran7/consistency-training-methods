#!/usr/bin/env python3
"""Freeze explicitly selected WildJailbreak rows for consistency training."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ctm.settings.families import (
    FAMILY_SCHEMA_VERSION,
    select_fixed_variants,
    stable_digest,
    write_frozen_artifact,
)

WILDJAILBREAK_DATASET = "allenai/wildjailbreak"
WILDJAILBREAK_REVISION = "254c59ec8aff3f333ca8f2e28be94d8b2ff4098f"
WILDJAILBREAK_LICENSE = "ODC-BY"


def _text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_tactics(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid tactics literal: {value[:120]!r}") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("tactics must be a list")
    tactics = sorted({str(tactic).strip() for tactic in value if str(tactic).strip()})
    if not tactics:
        raise ValueError("adversarial rows need at least one tactic")
    return tactics


def _valence(data_type: str) -> str:
    if data_type == "adversarial_harmful":
        return "harmful"
    if data_type == "adversarial_benign":
        return "benign"
    raise ValueError(f"unsupported WildJailbreak data_type: {data_type!r}")


def build_wildjailbreak_families(
    rows: Iterable[Mapping[str, Any]],
    *,
    n_variants: int = 4,
    seed: str = "42",
) -> list[dict[str, Any]]:
    """Group exactly the supplied adversarial rows; CTM creates no holdout."""

    if n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        data_type = _text(row, "data_type")
        valence = _valence(data_type)
        vanilla = _text(row, "vanilla")
        adversarial = _text(row, "adversarial")
        tactics = _parse_tactics(row.get("tactics"))
        source_id = "wildjailbreak:" + stable_digest(vanilla, seed="wildjailbreak-base")[:24]
        group = groups.setdefault(
            source_id,
            {"vanilla": vanilla, "valence": valence, "data_type": data_type, "variants": []},
        )
        if group["vanilla"] != vanilla or group["valence"] != valence:
            raise ValueError(f"conflicting rows for {source_id}")
        variant_id = (
            "wildjailbreak-variant:"
            + stable_digest(f"{adversarial}\0{'|'.join(tactics)}", seed="wildjailbreak-variant")[:24]
        )
        group["variants"].append(
            {
                "variant_id": variant_id,
                "messages": [{"role": "user", "content": adversarial}],
                "axes": {"tactics": tactics},
            }
        )

    families = []
    incomplete: list[str] = []
    for source_id, group in groups.items():
        available = len({variant["variant_id"] for variant in group["variants"]})
        if available < n_variants:
            incomplete.append(f"{source_id} ({group['valence']}: {available}/{n_variants} variants)")
            continue
        families.append(
            {
                "schema_version": FAMILY_SCHEMA_VERSION,
                "source_id": source_id,
                "source": WILDJAILBREAK_DATASET,
                "reference_messages": [{"role": "user", "content": group["vanilla"]}],
                "variants": select_fixed_variants(
                    group["variants"],
                    source_id=source_id,
                    n_variants=n_variants,
                    seed=seed,
                ),
                "metadata": {"valence": group["valence"], "source_data_type": group["data_type"]},
            }
        )
    if incomplete:
        examples = ", ".join(sorted(incomplete)[:5])
        suffix = "" if len(incomplete) <= 5 else f", and {len(incomplete) - 5} more"
        raise ValueError(
            f"selected WildJailbreak rows contain {len(incomplete)} incomplete prompt family/families: "
            f"{examples}{suffix}"
        )
    families.sort(key=lambda family: stable_digest(family["source_id"], seed=seed))
    if not families:
        raise ValueError("selected WildJailbreak rows produced no training families")
    return families


def read_jsonl(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path_like in paths:
        path = Path(path_like)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: each row must be an object")
                rows.append(row)
    if not rows:
        raise ValueError("source JSONL files contained no rows")
    return rows


def materialize_wildjailbreak(
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    n_variants: int = 4,
    seed: str = "42",
    source_revision: str = WILDJAILBREAK_REVISION,
    source_license: str = WILDJAILBREAK_LICENSE,
) -> dict[str, Any]:
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows).encode()
    return write_frozen_artifact(
        output_path,
        build_wildjailbreak_families(rows, n_variants=n_variants, seed=seed),
        provenance={
            "dataset": WILDJAILBREAK_DATASET,
            "revision": source_revision,
            "source_license": source_license,
            "source_row_count": len(rows),
            "source_rows_sha256": hashlib.sha256(payload).hexdigest(),
            "n_variants": n_variants,
            "seed": seed,
            "completion_fields_used": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-variants", type=int, default=4)
    parser.add_argument("--seed", default="42")
    parser.add_argument("--source-revision", default=WILDJAILBREAK_REVISION)
    parser.add_argument("--source-license", default=WILDJAILBREAK_LICENSE)
    args = parser.parse_args(argv)
    manifest = materialize_wildjailbreak(
        read_jsonl(args.input_jsonl),
        args.output,
        n_variants=args.n_variants,
        seed=args.seed,
        source_revision=args.source_revision,
        source_license=args.source_license,
    )
    print(f"Wrote {manifest['row_count']} training families to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
