"""Write the immutable target-to-training-child Qwen3.5 on-policy contract.

This follows the recovered-source and worker-parity gates.  It is deliberately
separate from the source-preflight CLI: the target contract is only meaningful
after a real worker-parity sidecar exists, and it binds the exact final child
argv that ``scripts/run_experiment.py`` will launch.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from experiments.rmct_paper_vast_dense_models.stage1.onpolicy_recovery_preflight import (
    attest_onpolicy_target,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--source-attestation", required=True, type=Path)
    parser.add_argument("--worker-parity-attestation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--worker-gpus", required=True)
    parser.add_argument("--worker-gpu-mem-util", required=True, type=float)
    parser.add_argument("--worker-max-model-len", required=True, type=int)
    parser.add_argument("--worker-max-num-seqs", required=True, type=int)
    parser.add_argument("--worker-max-num-batched-tokens", required=True, type=int)
    parser.add_argument("--worker-seed-base", required=True, type=int)
    parser.add_argument("--worker-gdn-prefill-backend", choices=["flashinfer", "triton"])
    parser.add_argument("--target-logprob-chunk-size", required=True, type=int)
    parser.add_argument(
        "--topology-profile",
        help="Explicit compiler topology profile to bind alongside the logical coordinator/worker layout",
    )
    parser.add_argument(
        "--rmct256-selection",
        type=Path,
        help="immutable selected RMCT-256 JSONL; required when the compiled target declares selection_manifest",
    )
    parser.add_argument(
        "--rmct256-selection-manifest",
        type=Path,
        help="content-addressed RMCT-256 selection manifest",
    )
    parser.add_argument("--rmct256-canonical-source", type=Path, help="verified canonical n=2048 RMCT source")
    parser.add_argument(
        "--rmct256-canonical-source-manifest",
        type=Path,
        help="immutable canonical n=2048 source manifest",
    )
    parser.add_argument(
        "--rmct256-original64-reference-manifest",
        type=Path,
        help="historical RMCT original64 reference manifest",
    )
    parser.add_argument(
        "--rmct256-stage2-manifest",
        type=Path,
        help="immutable Stage-2 OOD/HLE manifest for zero-overlap verification",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        result = attest_onpolicy_target(
            plan=args.plan,
            target=args.target,
            source=args.source,
            source_manifest=args.source_manifest,
            source_attestation=args.source_attestation,
            worker_parity_attestation=args.worker_parity_attestation,
            output=args.output,
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            worker_gpus=args.worker_gpus,
            worker_gpu_mem_util=args.worker_gpu_mem_util,
            worker_max_model_len=args.worker_max_model_len,
            worker_max_num_seqs=args.worker_max_num_seqs,
            worker_max_num_batched_tokens=args.worker_max_num_batched_tokens,
            target_logprob_chunk_size=args.target_logprob_chunk_size,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            worker_gdn_prefill_backend=args.worker_gdn_prefill_backend,
            worker_seed_base=args.worker_seed_base,
            topology_profile=args.topology_profile,
            rmct256_selection=args.rmct256_selection,
            rmct256_selection_manifest=args.rmct256_selection_manifest,
            rmct256_canonical_source=args.rmct256_canonical_source,
            rmct256_canonical_source_manifest=args.rmct256_canonical_source_manifest,
            rmct256_original64_reference_manifest=args.rmct256_original64_reference_manifest,
            rmct256_stage2_manifest=args.rmct256_stage2_manifest,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        "QWEN35_ONPOLICY_TARGET_ATTESTATION="
        + json.dumps(
            {
                "status": result["status"],
                "path": result["path"],
                "sha256": result["sha256"],
                "target": result["attestation"]["target"],
                "source": result["attestation"]["source"],
                **(
                    {"rmct256_selection": result["attestation"]["rmct256_selection"]["selection"]}
                    if "rmct256_selection" in result["attestation"]
                    else {}
                ),
                "worker_parity_attestation": result["attestation"]["worker_parity_attestation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
