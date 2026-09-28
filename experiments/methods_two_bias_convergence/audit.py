"""Check every frozen prompt pair with the pinned production tokenizer, without generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plan


def audit(repository: Path, output: Path, *, native_diagnostic: bool = False) -> dict:
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from ctm.training.consistency_data import build_consistency_datums_with_audit
    snapshot = snapshot_download(plan.MODEL, revision=plan.REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    pool = plan.ordered_pool(repository)
    lengths = []
    failures = []
    aligned_rows = 0
    by_bias = {bias: {"rows": 0, "unaligned": 0, "suffix_failed": 0} for bias in plan.BIASES}
    for start in range(0, len(pool), 2):
        pairs = plan.paired_rows(pool[start:start + 2], method=None if native_diagnostic else "act")
        for pair in pairs:
            datums, alignment = build_consistency_datums_with_audit(tokenizer, [pair])
            aligned_rows += len(datums)
            counts = by_bias[pair["bias"]]
            counts["rows"] += 1
            counts["unaligned"] += int(not datums)
            counts["suffix_failed"] += int(not all(alignment["checks"].values()))
            if not all(alignment["checks"].values()):
                failures.append({"question_id": pair["question_id"], "bias": pair["bias"],
                                 "checks": alignment["checks"], "error": alignment["first_error"]})
            lengths.extend(len(datum.model_input.to_ints()) for datum in datums)
    result = {"schema": "ctm-expanded-all-pair-token-alignment-v1", "model": plan.MODEL,
              "revision": plan.REVISION, "data_sha256": plan.POOL_SHA, "unique_qids": len(pool),
              "pair_transform": "native_shared_pool" if native_diagnostic else plan.INTERNAL_PAIR_TRANSFORM,
              "paired_rows": len(pool) * 2, "unaligned_rows": len(pool) * 2 - aligned_rows,
              "full_reference_suffix_alignment": not failures, "prompt_tokens_min": min(lengths),
              "prompt_tokens_max": max(lengths), "generation_performed": False,
              "by_bias": by_bias, "failures": failures}
    plan.immutable_json(output, result)
    return result


def main() -> None:
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-diagnostic", action="store_true", help="Audit original placement for comparison, not internal-method training")
    args = parser.parse_args()
    result = audit(args.repository.resolve(), args.output.resolve(), native_diagnostic=args.native_diagnostic)
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}, sort_keys=True))
    if not result["full_reference_suffix_alignment"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
