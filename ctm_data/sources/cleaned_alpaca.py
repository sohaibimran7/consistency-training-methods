"""Export a deterministic prompt subset from the paper-cited Cleaned Alpaca data.

The original responses are deliberately ignored. CTM samples fresh responses
from the experiment's frozen base model before training, matching the paper's
instruction-data construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ctm.artifacts import write_atomic_bytes

SOURCE_REVISION = "d03c782bffd50ceeb1f4ef3c020129229ec4698c"
SOURCE_URL = f"https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/{SOURCE_REVISION}/alpaca_data_cleaned.json"
SOURCE_SHA256 = "bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b"


def select_prompt_rows(records: Sequence[Any], *, count: int, seed: str) -> list[dict[str, Any]]:
    """Validate the source and return a deterministic shuffled prompt prefix."""

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive integer")
    validated: list[tuple[int, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Cleaned Alpaca row {index + 1} must be an object")
        instruction = record.get("instruction")
        input_text = record.get("input", "")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"Cleaned Alpaca row {index + 1} has no instruction")
        if not isinstance(input_text, str):
            raise ValueError(f"Cleaned Alpaca row {index + 1} input must be a string")
        content = instruction.strip()
        if input_text.strip():
            content += f"\n\nInput:\n{input_text.strip()}"
        validated.append((index, content))
    if len(validated) < count:
        raise ValueError(f"Cleaned Alpaca contains only {len(validated)}/{count} requested prompts")

    random.Random(seed).shuffle(validated)
    rows = []
    for source_index, content in validated[:count]:
        messages = [{"role": "user", "content": content}]
        rows.append(
            {
                "source_id": f"cleaned-alpaca-{source_index}",
                "reference_messages": messages,
                "variant_messages": messages,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export deterministic Cleaned Alpaca prompts for fresh base-model target sampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--seed", default="42")
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args(argv)

    if args.count < 1:
        parser.error("--count must be >= 1")
    if args.output.resolve() == args.manifest_output.resolve():
        parser.error("--output and --manifest-output must differ")
    existing = [str(path) for path in (args.output, args.manifest_output) if path.exists()]
    if existing:
        parser.error(f"refusing to overwrite existing output(s): {existing}")

    print("\nCleaned Alpaca export plan:")
    print(f"  source={SOURCE_URL}")
    print(f"  source_sha256={SOURCE_SHA256}")
    print(f"  count={args.count}, seed={args.seed!r}")
    print(f"  output={args.output}")
    print(f"  manifest={args.manifest_output}")
    print("  Original dataset responses are ignored; CTM will sample fresh targets.")
    if not args.yes and input("\nProceed? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        return

    with urllib.request.urlopen(SOURCE_URL) as response:  # noqa: S310 - immutable HTTPS URL above
        source_bytes = response.read()
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != SOURCE_SHA256:
        raise ValueError(f"Cleaned Alpaca source hash changed: expected {SOURCE_SHA256}, got {actual_hash}")
    records = json.loads(source_bytes)
    if not isinstance(records, list):
        raise ValueError("Cleaned Alpaca source must contain a JSON array")
    rows = select_prompt_rows(records, count=args.count, seed=args.seed)
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    manifest = {
        "schema_version": 1,
        "kind": "cleaned_alpaca_prompt_selection",
        "written_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "content_sha256": SOURCE_SHA256,
            "license": "CC BY-NC 4.0",
        },
        "selection": {"count": args.count, "seed": args.seed, "method": "random.Random(seed).shuffle prefix"},
        "original_responses_used": False,
        "row_count": len(rows),
        "output": {
            "path": str(args.output.resolve()),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        },
    }
    write_atomic_bytes(args.output, payload)
    write_atomic_bytes(
        args.manifest_output,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"Exported {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
