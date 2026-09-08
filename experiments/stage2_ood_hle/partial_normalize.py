"""Normalize one partial Stage 2 Inspect root in an isolated Python process.

Inspect EvalLogs retain full model transcripts.  The partial plotting command
uses this tiny child process once per condition so those transcripts are freed
before the next condition is opened.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from experiments.stage2_ood_hle.analyze import load_inspect_runs, parse_runs

SCHEMA = "stage2-ood-hle-partial-normalized-v1"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, metavar="CONDITION=LOCAL_LUNA_LOGS")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        runs = parse_runs([args.run])
        observations, sources = load_inspect_runs(runs, expected_prompt_style="none")
        payload = {
            "schema": SCHEMA,
            "observations": [asdict(row) for row in observations],
            "sources": sources,
        }
        destination = args.output.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"normalized {len(observations)} observations")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
