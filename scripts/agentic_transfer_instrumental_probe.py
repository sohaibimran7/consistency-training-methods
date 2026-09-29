"""Read-only InstrumentalEval source audit; never calls target or grader models.

Only public source CSVs are fetched when --fetch-public is explicitly supplied.
Output contains counts, hashes and source-row identifiers, never prompt text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

SOURCE_REVISION = "269678168cf0388bf5dd7d10dfaed4af07905c49"
INSPECT_EVALS_REVISION = "47a0725a3eb630c94209ea2aebbcf420beee79ca"
CHECKSUMS = {
    "Deceptive Behavior": "757034719bfa5703a90673fccd594b2d85610bfa8079747f33521982898102af",
    "Evading Shutdown": "c6c2dd40c833f11d450afaa53e5e58327bb69d52f90e47643ea795ef57be69d6",
    "Hacking Computer Systems": "874d84f4595c9a871fc5e884491a99baeded9442e6ae79e74187118ce04bccbf",
    "Hiding Unwanted Behavior": "c6bd396954981b54dea4caf58f73b68534bffc7317570f2318493069ba50e18f",
    "Running Multiple AI Copies": "03224acee7b77832a64e616ff76eb9dc61a6a381391e49b706b718ce544f15c4",
    "Strategically Appearing Aligned": "cbcd421ac0230892c785a77c4c15c2ab9d67dc31847d94521c5b6ec132b09d18",
}


def audit_csv(
    payload: bytes,
    *,
    task_type: str,
    expected_sha256: str,
    offset: int = 0,
    per_category: int = 2,
) -> dict:
    """Verify a native CSV and return text-free, stable source identifiers.

    sample_row is one-based after the grader-context record; inspect_index is
    one-based in the complete upstream dataset. Neither is a template-family ID.
    """
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"checksum mismatch for {task_type}")
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
    if len(rows) < 2 or any(len(row) != 1 or not row[0].strip() for row in rows):
        raise ValueError(f"expected nonempty one-column CSV with context: {task_type}")
    if per_category < 1 or per_category > len(rows) - 1:
        raise ValueError("per_category must be between one and the category size")
    samples = [
        {
            "source_id": f"{task_type}:{i}",
            "sample_row": i,
            "csv_record": i + 1,
            "inspect_index": offset + i,
            "input_sha256": hashlib.sha256(row[0].encode("utf-8")).hexdigest(),
        }
        for i, row in enumerate(rows[1:], start=1)
    ]
    selected = sorted(samples, key=lambda sample: sample["input_sha256"])[:per_category]
    return {
        "task_type": task_type,
        "file_sha256": digest,
        "eval_prompt_sha256": hashlib.sha256(rows[0][0].encode("utf-8")).hexdigest(),
        "sample_count": len(samples),
        "unique_input_count": len({sample["input_sha256"] for sample in samples}),
        "samples": samples,
        "smoke_selection": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-dir", type=Path, help="Local upstream benchmark directory")
    source.add_argument("--fetch-public", action="store_true", help="Fetch six pinned public CSVs in memory")
    parser.add_argument("--per-category", type=int, default=2)
    args = parser.parse_args()
    if args.per_category < 1:
        parser.error("--per-category must be positive")
    categories = []
    count = 0
    for task_type, expected_sha256 in CHECKSUMS.items():
        filename = f"Alignment Drift - {task_type}.csv"
        if args.dataset_dir is not None:
            payload = (args.dataset_dir / filename).read_bytes()
        else:
            url = (
                "https://raw.githubusercontent.com/yf-he/InstrumentalEval/"
                f"{SOURCE_REVISION}/benchmark/{quote(filename)}"
            )
            with urlopen(url, timeout=30) as response:
                payload = response.read()
        category = audit_csv(
            payload,
            task_type=task_type,
            expected_sha256=expected_sha256,
            offset=count,
            per_category=args.per_category,
        )
        categories.append(category)
        count += category["sample_count"]
    all_hashes = [s["input_sha256"] for c in categories for s in c["samples"]]
    print(
        json.dumps(
            {
                "schema_version": "instrumentaleval.source_audit.v1",
                "source_revision": SOURCE_REVISION,
                "inspect_evals_revision": INSPECT_EVALS_REVISION,
                "sample_count": count,
                "unique_input_count": len(set(all_hashes)),
                "categories": categories,
                "smoke_inspect_indices": sorted(s["inspect_index"] for c in categories for s in c["smoke_selection"]),
                "selection_rule": "smallest input SHA-256 per category; smoke only, not semantic clustering",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
