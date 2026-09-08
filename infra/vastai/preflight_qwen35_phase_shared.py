#!/usr/bin/env python3
"""Fail-closed, non-production preflight for phase-shared Qwen3.5 training.

This is deliberately *not* a training launcher.  It creates a fresh evidence
directory and checks the exact resource choreography proposed for a local
phase-shared RMCT/OPCT run:

* one explicit CUDA_VISIBLE_DEVICES allocation is resolved into arbitrary
  trainer/rollout logical GPU sets (including 2-, 4-, and 8-GPU contracts);
* a vLLM worker and an HF/PEFT trainer really coexist on every selected GPU;
* every worker releases GPU memory with level-1 sleep and wakes again;
* the Qwen3.5 translated LoRA snapshot has a non-zero, HF-consistent effect
  on every worker after a real update; and
* one fixed cross-entropy update agrees with a single-rank reference before a
  conservative rank-zero packing-capacity sweep is recorded.

All writes stay beneath --output-dir.  The directory must not already exist,
so an incomplete preflight remains available for diagnosis and cannot be
silently overwritten.  The harness never reads experiment credentials,
datasets, checkpoints, or production output directories.

Example (four logical GPUs, all used in both phases)::

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
      python infra/vastai/preflight_qwen35_phase_shared.py \
        --output-dir /workspace/ctm-phase-shared-preflight-$(date -u +%Y%m%dT%H%M%SZ)

Run ``--dry-run`` first on a host without CUDA to inspect the resolved,
hash-bound-free execution contract.  For a full 2/4/8 topology-contract sweep
on one host, provide eight visible GPUs and ``--require-full-topology-contract-sweep``.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "qwen35-phase-shared-preflight-v1"
MODEL = "Qwen/Qwen3.5-9B"
# This is the immutable base snapshot already bound by the RMCT-256
# convergence contracts.  Keeping the phase-shared preflight on the same
# commit prevents a moving ``main`` ref from changing the model between the
# single-rank reference and concurrently started replicas/workers.
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
HF_SNAPSHOT_READINESS_SCHEMA = "qwen35-hf-snapshot-readiness-v1"
REQUIRED_PACKING_BUDGETS = (20_480, 40_960, 49_152)


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _finite_positive(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def _unit_interval(raw: str) -> float:
    value = _finite_positive(raw)
    if value > 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return value


def _checkpoint_layers(raw: str) -> str | int:
    value = raw.strip().lower()
    if value == "all":
        return "all"
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("gradient checkpoint layers must be 'all' or a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("gradient checkpoint layers must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL, help=f"Fixed preflight model (must be {MODEL!r})")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Fresh directory for all non-production evidence"
    )
    parser.add_argument(
        "--training-gpus",
        default="all",
        help="Logical trainer GPUs relative to CUDA_VISIBLE_DEVICES; defaults to all visible GPUs",
    )
    parser.add_argument(
        "--rollout-gpus",
        default=None,
        help=(
            "Logical vLLM worker GPUs relative to CUDA_VISIBLE_DEVICES. Defaults to --training-gpus; this preflight intentionally requires the two sets to be identical."
        ),
    )
    parser.add_argument(
        "--topology-contract-sizes",
        type=_positive_int,
        nargs="+",
        default=[2, 4, 8],
        help="Pure topology contracts to validate against this explicit allocation (default: 2 4 8)",
    )
    parser.add_argument(
        "--require-full-topology-contract-sweep",
        action="store_true",
        help="Fail unless the allocation is large enough to validate every requested topology-contract size",
    )
    parser.add_argument("--worker-gpu-mem-util", type=_unit_interval, default=0.35)
    parser.add_argument(
        "--worker-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help=(
            "Exact vLLM dtype for the cold coordinator and every rollout worker. "
            "The worker-parity attestation binds this value, so it must match the production CLI."
        ),
    )
    parser.add_argument("--worker-max-model-len", type=_positive_int, default=32_768)
    parser.add_argument("--worker-max-num-seqs", type=_positive_int, default=64)
    parser.add_argument("--worker-max-num-batched-tokens", type=_positive_int, default=8_192)
    parser.add_argument(
        "--worker-gdn-prefill-backend",
        choices=["flashinfer", "triton"],
        default=None,
        help=(
            "Optional vLLM gated-delta-network prefill backend. Use 'triton' on runtime-only GH200 systems where FlashInfer cannot JIT its GDN kernel."
        ),
    )
    parser.add_argument("--worker-seed-base", type=int, default=42)
    parser.add_argument("--target-logprob-chunk-size", type=_positive_int, default=2_048)
    parser.add_argument("--forward-microbatch-max-datums", type=_positive_int, default=8)
    parser.add_argument(
        "--gradient-checkpoint-layers",
        type=_checkpoint_layers,
        default="all",
        help="'all' (default) or the first N backbone layers; recorded in every evidence artifact",
    )
    parser.add_argument("--lora-rank", type=_positive_int, default=8)
    parser.add_argument("--lora-alpha", type=_positive_int, default=16)
    parser.add_argument("--lora-seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=_finite_positive, default=1e-4)
    parser.add_argument(
        "--parity-datums-per-rank",
        type=_positive_int,
        default=2,
        help="Fixed update datums per replicated rank; total is this value times world size",
    )
    parser.add_argument("--parity-sequence-tokens", type=_positive_int, default=512)
    parser.add_argument(
        "--packing-budgets",
        type=_positive_int,
        nargs="+",
        default=list(REQUIRED_PACKING_BUDGETS),
        help="Per-rank physical padded-token budgets; must include 20480, 40960, and 49152",
    )
    parser.add_argument(
        "--packing-probe-datums",
        type=_positive_int,
        default=2,
        help="Equal-length synthetic datums used for each rank-zero packing-capacity measurement",
    )
    parser.add_argument("--rollout-probe-max-tokens", type=_positive_int, default=32)
    parser.add_argument(
        "--effect-candidate-count",
        type=_positive_int,
        default=4,
        help=(
            "Fixed repeated-token cross-entropy suffix candidates used to find a measurable post-update " "LoRA effect"
        ),
    )
    parser.add_argument(
        "--effect-probe-completion-tokens",
        type=_positive_int,
        default=256,
        help=(
            "Suffix length scored by the update-aligned LoRA-effect probe. It must be shorter than "
            "--parity-sequence-tokens (default: 256)."
        ),
    )
    parser.add_argument("--adapter-effect-min", type=_finite_positive, default=1e-5)
    parser.add_argument("--effect-cosine-min", type=_unit_interval, default=0.90)
    parser.add_argument(
        "--effect-norm-ratio-min",
        type=_finite_positive,
        default=0.80,
        help="Minimum worker/HF LoRA-effect L2 norm ratio for aggregate and every worker (default: 0.80)",
    )
    parser.add_argument(
        "--effect-norm-ratio-max",
        type=_finite_positive,
        default=1.25,
        help="Maximum worker/HF LoRA-effect L2 norm ratio for aggregate and every worker (default: 1.25)",
    )
    parser.add_argument(
        "--effect-relative-l2-max",
        type=_finite_positive,
        default=0.25,
        help="Maximum worker-minus-HF effect L2 residual divided by HF effect L2 norm (default: 0.25)",
    )
    parser.add_argument("--wake-score-max-abs", type=_finite_positive, default=1e-5)
    parser.add_argument("--loss-parity-atol", type=_finite_positive, default=1e-4)
    parser.add_argument(
        "--loss-parity-rtol",
        type=_unit_interval,
        default=5e-4,
        help=(
            "Relative fixed-update loss tolerance for BF16 batch-shape drift. It is accepted only together with the adapter-state parity gate when --loss-parity-atol is exceeded (default: 5e-4)."
        ),
    )
    parser.add_argument(
        "--parameter-parity-atol",
        type=_finite_positive,
        default=None,
        help=(
            "Maximum raw-adapter discrepancy in the robust post-Adam envelope. Defaults to 2.05 times --learning-rate; this leaves bounded BF16/reduction headroom around the first AdamW step."
        ),
    )
    parser.add_argument(
        "--parameter-parity-delta-cosine-min",
        type=_unit_interval,
        default=0.9995,
        help=(
            "Monitoring-only minimum cosine similarity for reference/replicated post-Adam adapter updates "
            "pending cross-allocation calibration (default: 0.9995)."
        ),
    )
    parser.add_argument(
        "--parameter-parity-delta-relative-l2-max",
        type=_unit_interval,
        default=0.005,
        help=(
            "Diagnostic-only target for adapter-update relative L2 error (default: 0.005). "
            "It is retained for calibration and does not gate this preflight."
        ),
    )
    parser.add_argument(
        "--parameter-parity-delta-max-abs",
        type=_finite_positive,
        default=None,
        help=(
            "Diagnostic-only target for max adapter-update coordinate error. Defaults to one quarter of --learning-rate."
        ),
    )
    parser.add_argument(
        "--parameter-parity-envelope-relative-l2-max",
        type=_unit_interval,
        default=0.035,
        help=(
            "Monitoring-only post-Adam adapter-update relative-L2 envelope pending cross-allocation "
            "calibration (default: 0.035)."
        ),
    )
    parser.add_argument(
        "--parameter-parity-b-sign-mismatch-max",
        type=_unit_interval,
        default=4e-4,
        help=(
            "Monitoring-only LoRA-B post-Adam update sign-mismatch envelope pending cross-allocation "
            "calibration (default: 0.0004)."
        ),
    )
    parser.add_argument(
        "--gradient-parity-cosine-min",
        type=_unit_interval,
        default=0.999,
        help="Minimum FP64 cosine similarity for the pre-optimizer global gradient (default: 0.999).",
    )
    parser.add_argument(
        "--gradient-parity-norm-ratio-min",
        type=_finite_positive,
        default=0.95,
        help="Minimum replicated/reference pre-optimizer gradient L2 norm ratio (default: 0.95).",
    )
    parser.add_argument(
        "--gradient-parity-norm-ratio-max",
        type=_finite_positive,
        default=1.05,
        help="Maximum replicated/reference pre-optimizer gradient L2 norm ratio (default: 1.05).",
    )
    parser.add_argument(
        "--gradient-parity-relative-l2-max",
        type=_unit_interval,
        default=0.05,
        help="Maximum replicated-minus-reference gradient L2 residual divided by reference norm (default: 0.05).",
    )
    parser.add_argument(
        "--gradient-parity-magnitude-weighted-sign-mismatch-max",
        type=_unit_interval,
        default=0.01,
        help="Maximum magnitude-weighted pre-optimizer gradient sign disagreement (default: 0.01).",
    )
    parser.add_argument("--start-timeout-seconds", type=_finite_positive, default=1_800.0)
    parser.add_argument("--request-timeout-seconds", type=_finite_positive, default=7_200.0)
    parser.add_argument("--shutdown-timeout-seconds", type=_finite_positive, default=30.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the contract without probing CUDA, creating files, or loading a model",
    )
    args = parser.parse_args(argv)
    if args.model != MODEL:
        parser.error(f"this is a Qwen3.5-9B-specific preflight; --model must be {MODEL!r}")
    if args.worker_seed_base < 0:
        parser.error("--worker-seed-base must be non-negative")
    if args.lora_seed < 0:
        parser.error("--lora-seed must be non-negative")
    if args.effect_probe_completion_tokens >= args.parity_sequence_tokens:
        parser.error("--effect-probe-completion-tokens must be shorter than --parity-sequence-tokens")
    if args.effect_norm_ratio_min > args.effect_norm_ratio_max:
        parser.error("--effect-norm-ratio-min must not exceed --effect-norm-ratio-max")
    if args.gradient_parity_norm_ratio_min > args.gradient_parity_norm_ratio_max:
        parser.error("--gradient-parity-norm-ratio-min must not exceed --gradient-parity-norm-ratio-max")
    if len(set(args.topology_contract_sizes)) != len(args.topology_contract_sizes):
        parser.error("--topology-contract-sizes must not contain duplicates")
    if len(set(args.packing_budgets)) != len(args.packing_budgets):
        parser.error("--packing-budgets must not contain duplicates")
    missing_budgets = sorted(set(REQUIRED_PACKING_BUDGETS) - set(args.packing_budgets))
    if missing_budgets:
        parser.error(
            "--packing-budgets must include the required production-candidate budgets "
            + ", ".join(str(value) for value in missing_budgets)
        )
    if args.parameter_parity_atol is None:
        # This preflight deliberately exercises one *first* AdamW update using
        # AdamConfig's zero weight decay.  With zero initialized moments, a
        # coordinate's ideal AdamW movement is bounded by the learning rate;
        # two learning-rate units leave one full unit for BF16/reduction
        # rounding without the previous, unscaled 1e-3 allowance.  The small
        # extra 0.05x is explicit in the receipt rather than being hidden in
        # an arbitrary absolute adapter tolerance.
        args.parameter_parity_atol = 2.05 * args.learning_rate
        args.parameter_parity_atol_source = "derived_2.05x_learning_rate_first_adamw_step"
    else:
        args.parameter_parity_atol_source = "explicit"
    if args.parameter_parity_delta_max_abs is None:
        args.parameter_parity_delta_max_abs = 0.25 * args.learning_rate
        args.parameter_parity_delta_max_abs_source = "derived_0.25x_learning_rate_first_adamw_step"
    else:
        args.parameter_parity_delta_max_abs_source = "explicit"
    return args


@dataclass(frozen=True)
class ResolvedPreflight:
    """Pure, GPU-count-independent plan for one concrete preflight invocation."""

    topology: Any
    checkpoint_layers: int | None
    topology_contract_sweep: tuple[dict[str, Any], ...]

    @property
    def world_size(self) -> int:
        return int(self.topology.world_size)

    @property
    def coordinator_device(self) -> str:
        return f"cuda:{self.topology.coordinator.logical_index}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "visible_devices": list(self.topology.visible_devices),
            "training_gpus": [
                {"logical_index": gpu.logical_index, "device_token": gpu.device_token}
                for gpu in self.topology.train_gpus
            ],
            "rollout_gpus": [
                {"logical_index": gpu.logical_index, "device_token": gpu.device_token}
                for gpu in self.topology.rollout_gpus
            ],
            "world_size": self.world_size,
            "coordinator_device": self.coordinator_device,
            "gradient_checkpoint_layers": "all" if self.checkpoint_layers is None else self.checkpoint_layers,
            "topology_contract_sweep": list(self.topology_contract_sweep),
        }


def _visible_devices(cuda_visible_devices: str | None) -> tuple[str, ...]:
    raw = (cuda_visible_devices or "").strip()
    if not raw or raw in {"-1", "NoDevFiles"}:
        raise ValueError("phase-shared preflight requires an explicit non-empty CUDA_VISIBLE_DEVICES allocation")
    values = tuple(value.strip() for value in raw.split(","))
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique, non-empty device tokens")
    return values


def _all_prefix_spec(size: int) -> str:
    return ",".join(str(index) for index in range(size))


def resolve_preflight_contract(
    args: argparse.Namespace,
    *,
    cuda_visible_devices: str | None = None,
) -> ResolvedPreflight:
    """Resolve a no-CUDA contract and reject unsafe topology assumptions.

    The preflight intentionally exercises a full phase-sharing layout: each
    selected GPU hosts a persistent HF trainer and one vLLM worker at different
    times.  Permitting a partially overlapping set here would make a successful
    probe weaker than the planned production resource topology.
    """

    from ctm.backends.local.phase_shared import resolve_phase_shared_topology

    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES") if cuda_visible_devices is None else cuda_visible_devices
    visible = _visible_devices(visible_raw)
    rollout_spec = args.training_gpus if args.rollout_gpus is None else args.rollout_gpus
    topology = resolve_phase_shared_topology(
        train_gpus_spec=args.training_gpus,
        rollout_gpus_spec=rollout_spec,
        cuda_visible_devices=",".join(visible),
        allow_overlap=True,
    )
    train_indices = tuple(gpu.logical_index for gpu in topology.train_gpus)
    rollout_indices = tuple(gpu.logical_index for gpu in topology.rollout_gpus)
    if train_indices != rollout_indices:
        raise ValueError(
            f"this full phase-sharing preflight requires identical ordered --training-gpus and --rollout-gpus; got training={train_indices}, rollout={rollout_indices}"
        )
    if topology.world_size < 2:
        raise ValueError("fixed-update replicated parity requires at least two real trainer GPUs")
    if args.parity_datums_per_rank < 1:
        raise ValueError("--parity-datums-per-rank must be positive")

    sweep: list[dict[str, Any]] = []
    for size in args.topology_contract_sizes:
        if size > len(visible):
            sweep.append(
                {
                    "world_size": size,
                    "status": "not_available_in_this_allocation",
                    "required_visible_gpu_count": size,
                    "available_visible_gpu_count": len(visible),
                }
            )
            continue
        prefix = _all_prefix_spec(size)
        probe_topology = resolve_phase_shared_topology(
            train_gpus_spec=prefix,
            rollout_gpus_spec=prefix,
            cuda_visible_devices=",".join(visible),
            allow_overlap=True,
        )
        sweep.append(
            {
                "world_size": size,
                "status": "resolved_against_explicit_cuda_visible_devices",
                "training_logical_gpus": [gpu.logical_index for gpu in probe_topology.train_gpus],
                "rollout_logical_gpus": [gpu.logical_index for gpu in probe_topology.rollout_gpus],
                "coordinator_device": f"cuda:{probe_topology.coordinator.logical_index}",
            }
        )
    unavailable = [
        entry["world_size"] for entry in sweep if entry["status"] != "resolved_against_explicit_cuda_visible_devices"
    ]
    if args.require_full_topology_contract_sweep and unavailable:
        raise ValueError(
            f"--require-full-topology-contract-sweep needs an allocation large enough for every requested size; unavailable={unavailable}, visible_gpu_count={len(visible)}"
        )
    checkpoint_layers = None if args.gradient_checkpoint_layers == "all" else int(args.gradient_checkpoint_layers)
    return ResolvedPreflight(
        topology=topology,
        checkpoint_layers=checkpoint_layers,
        topology_contract_sweep=tuple(sweep),
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read one JSON object with a precise cache-readiness failure message."""

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not a readable JSON object: {path}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return parsed


def _snapshot_file_identity(snapshot: Path, path: Path, *, label: str) -> dict[str, Any]:
    """Hash one snapshot file while retaining its local cache provenance.

    Hugging Face cache snapshots normally use direct symlinks to immutable
    blobs, so the resolved file legitimately sits outside ``snapshot``.  The
    lexical path is nevertheless required to be inside the exact snapshot;
    this rules out an index-driven ``..`` escape while preserving the normal
    cache layout in the receipt.
    """

    try:
        relative_path = path.relative_to(snapshot)
    except ValueError as exc:
        raise RuntimeError(f"{label} escapes the resolved Hugging Face snapshot: {path}") from exc
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"{label} is absent or has a broken cache link: {path}") from exc
    if not resolved.is_file():
        raise RuntimeError(f"{label} is not a regular file: {path}")
    size_bytes = int(resolved.stat().st_size)
    if size_bytes < 1:
        raise RuntimeError(f"{label} is empty: {path}")
    identity: dict[str, Any] = {
        "relative_path": str(relative_path),
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "sha256": _sha256_file(resolved),
        "is_symlink": path.is_symlink(),
    }
    if path.is_symlink():
        try:
            identity["link_target"] = os.readlink(path)
        except OSError as exc:
            raise RuntimeError(f"cannot inspect {label} cache link: {path}") from exc
    return identity


def _safe_snapshot_child(snapshot: Path, filename: str, *, label: str) -> Path:
    """Resolve a repository-relative snapshot filename without traversal."""

    candidate = Path(filename)
    if (
        not filename
        or candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.name != filename
        or len(candidate.parts) != 1
    ):
        raise RuntimeError(f"{label} has an unsafe snapshot-relative filename: {filename!r}")
    return snapshot / candidate


def _safetensors_header_keys(path: Path, *, label: str) -> set[str]:
    """Validate a safetensors header without materializing model tensors."""

    try:
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise RuntimeError("missing 8-byte header length")
            header_length = int.from_bytes(raw_length, byteorder="little", signed=False)
            file_size = int(path.stat().st_size)
            # A real Qwen shard header is tiny.  This bound makes malformed
            # cache entries fail before attempting an unbounded allocation.
            if header_length < 2 or header_length > 64 * 1024 * 1024 or header_length > file_size - 8:
                raise RuntimeError(f"invalid safetensors header length {header_length}")
            raw_header = handle.read(header_length)
            if len(raw_header) != header_length:
                raise RuntimeError("truncated safetensors header")
    except OSError as exc:
        raise RuntimeError(f"cannot read {label}: {path}") from exc
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} has an invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise RuntimeError(f"{label} safetensors header is not an object: {path}")
    keys: set[str] = set()
    for key, metadata in header.items():
        if key == "__metadata__":
            if not isinstance(metadata, dict):
                raise RuntimeError(f"{label} safetensors metadata is not an object: {path}")
            continue
        if not isinstance(key, str) or not key or not isinstance(metadata, dict):
            raise RuntimeError(f"{label} has an invalid safetensors tensor header: {path}")
        keys.add(key)
    if not keys:
        raise RuntimeError(f"{label} safetensors header contains no tensor keys: {path}")
    return keys


def _prepare_pinned_hf_snapshot(
    *,
    repo_id: str,
    revision: str,
    AutoConfig: Any,
    AutoTokenizer: Any,
    snapshot_download: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Prefetch and validate one exact Qwen snapshot before process fan-out.

    A previous four-rank run demonstrated that independent ``from_pretrained``
    calls can race while Hugging Face's shard cache is being populated.  This
    single serial download is deliberately complete before any replica or
    vLLM process starts.  Later consumers receive the returned *local path*,
    not the mutable repository name or ``main`` ref.

    Each indexed safetensors shard is header-checked and SHA-256 hashed once.
    That is intentionally one model-sized sequential cache read paid only by
    this non-production readiness gate: it is cheaper than discovering a
    partial shard after allocating all trainer/worker GPUs, and it makes the
    receipt identify the exact bytes later loaded from the local path.
    """

    if repo_id != MODEL:
        raise ValueError(f"snapshot readiness is fixed to {MODEL!r}, got {repo_id!r}")
    if revision != MODEL_REVISION:
        raise ValueError(f"snapshot readiness is fixed to {MODEL_REVISION!r}, got {revision!r}")
    if snapshot_download is None:
        try:
            from huggingface_hub import snapshot_download as imported_snapshot_download
        except ImportError as exc:  # pragma: no cover - deployment dependency failure
            raise RuntimeError("huggingface_hub is required for Qwen snapshot readiness") from exc
        snapshot_download = imported_snapshot_download
    try:
        snapshot = (
            Path(
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_files_only=False,
                )
            )
            .expanduser()
            .resolve(strict=True)
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not prefetch pinned Qwen snapshot {repo_id}@{revision}: {exc}") from exc
    if not snapshot.is_dir():
        raise RuntimeError(f"pinned Qwen snapshot is not a directory: {snapshot}")
    if snapshot.name != revision:
        raise RuntimeError(
            "Hugging Face returned a snapshot that does not match the requested immutable Qwen commit: "
            f"requested={revision}, returned={snapshot.name}, path={snapshot}"
        )
    expected_cache_root_name = f"models--{repo_id.replace('/', '--')}"
    if snapshot.parent.name != "snapshots" or snapshot.parent.parent.name != expected_cache_root_name:
        raise RuntimeError(
            "Hugging Face did not return the canonical cache snapshot location for the pinned Qwen commit: "
            f"path={snapshot}, expected=.../{expected_cache_root_name}/snapshots/{revision}"
        )

    config_path = snapshot / "config.json"
    config_identity = _snapshot_file_identity(snapshot, config_path, label="pinned Qwen config")
    tokenizer_paths = [
        snapshot / name
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "chat_template.json")
        if (snapshot / name).exists() or (snapshot / name).is_symlink()
    ]
    if not tokenizer_paths:
        raise RuntimeError(f"pinned Qwen snapshot has no tokenizer artifact: {snapshot}")
    tokenizer_identities = [
        _snapshot_file_identity(snapshot, path, label="pinned Qwen tokenizer artifact") for path in tokenizer_paths
    ]

    index_paths = sorted(snapshot.glob("*.safetensors.index.json"))
    if len(index_paths) != 1:
        raise RuntimeError(
            "pinned Qwen snapshot must contain exactly one safetensors shard index; "
            f"found={[path.name for path in index_paths]}"
        )
    index_path = index_paths[0]
    index = _read_json_object(index_path, label="pinned Qwen safetensors shard index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"pinned Qwen safetensors shard index has no non-empty weight_map: {index_path}")
    if not all(
        isinstance(name, str) and name and isinstance(shard, str) and shard for name, shard in weight_map.items()
    ):
        raise RuntimeError(f"pinned Qwen safetensors shard index has an invalid weight_map: {index_path}")

    expected_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard_name in weight_map.items():
        expected_by_shard.setdefault(shard_name, set()).add(tensor_name)
    shard_receipts: list[dict[str, Any]] = []
    for shard_name in sorted(expected_by_shard):
        shard_path = _safe_snapshot_child(snapshot, shard_name, label="pinned Qwen index shard")
        if Path(shard_name).suffix != ".safetensors":
            raise RuntimeError(f"pinned Qwen index names a non-safetensors shard: {shard_name!r}")
        actual_keys = _safetensors_header_keys(shard_path, label=f"pinned Qwen indexed shard {shard_name}")
        missing_keys = sorted(expected_by_shard[shard_name] - actual_keys)
        if missing_keys:
            raise RuntimeError(
                f"pinned Qwen indexed shard {shard_name!r} is missing {len(missing_keys)} indexed tensor keys; "
                f"first={missing_keys[:3]}"
            )
        identity = _snapshot_file_identity(snapshot, shard_path, label=f"pinned Qwen indexed shard {shard_name}")
        identity.update(
            {
                "indexed_tensor_count": len(expected_by_shard[shard_name]),
                "header_tensor_count": len(actual_keys),
                "unindexed_header_tensor_count": len(actual_keys - expected_by_shard[shard_name]),
            }
        )
        shard_receipts.append(identity)

    # These local-only resolutions prove that the files which will be handed
    # to the trainers/workers are a Qwen3.5 config and usable tokenizer.  They
    # cannot repopulate or change the hub cache because the input is a local
    # directory and local_files_only is explicit.
    try:
        config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"pinned Qwen snapshot cannot be resolved locally by Transformers: {snapshot}: {exc}"
        ) from exc
    if getattr(config, "model_type", None) != "qwen3_5":
        raise RuntimeError(
            "pinned snapshot did not resolve as Qwen3.5; "
            f"model_type={getattr(config, 'model_type', None)!r}, snapshot={snapshot}"
        )

    return {
        "schema": HF_SNAPSHOT_READINESS_SCHEMA,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_commit": snapshot.name,
        "resolved_snapshot_path": str(snapshot),
        "runtime_model_argument": str(snapshot),
        "prefetch": {
            "method": "huggingface_hub.snapshot_download",
            "local_files_only": False,
            "completed_before_replica_or_vllm_startup": True,
            "indexed_shard_validation": "safetensors_header_and_full_sha256_once_per_indexed_shard",
        },
        "config": config_identity,
        "tokenizer_files": tokenizer_identities,
        "safetensors_index": {
            "identity": _snapshot_file_identity(snapshot, index_path, label="pinned Qwen safetensors shard index"),
            "indexed_tensor_count": len(weight_map),
            "indexed_shard_count": len(shard_receipts),
            "indexed_shards": shard_receipts,
            "indexed_shard_total_bytes": sum(int(entry["size_bytes"]) for entry in shard_receipts),
        },
        "local_consumer_validation": {
            "config_class": config.__class__.__name__,
            "tokenizer_class": tokenizer.__class__.__name__,
            "config_model_type": getattr(config, "model_type", None),
            "local_files_only": True,
        },
    }


def _runtime_model_identity(*, canonical_repo_id: str, snapshot_readiness: Mapping[str, Any]) -> dict[str, str]:
    """Make the canonical and actual model identities explicit in evidence."""

    revision = snapshot_readiness.get("requested_revision")
    commit = snapshot_readiness.get("resolved_commit")
    runtime_model = snapshot_readiness.get("runtime_model_argument")
    if not all(isinstance(value, str) and value for value in (revision, commit, runtime_model)):
        raise RuntimeError("validated Qwen snapshot receipt has no complete runtime model identity")
    if revision != MODEL_REVISION or commit != MODEL_REVISION:
        raise RuntimeError("validated Qwen snapshot receipt does not bind the pinned Qwen commit")
    return {
        "canonical_repo_id": canonical_repo_id,
        "pinned_revision": revision,
        "resolved_commit": commit,
        "runtime_model_argument": runtime_model,
        # The existing worker-attestation validator compares this exact field
        # with the model string used to create the worker pool.  Because the
        # pool receives a local snapshot path, keeping the canonical repo here
        # would be a false attestation rather than a convenient alias.
        "worker_attestation_expected_model": runtime_model,
        "worker_attestation_model_semantics": "exact_local_runtime_model_argument",
        "worker_attestation_portability": "runtime_path_bound_nonportable_topology_generic",
    }


def _atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_event(path: Path, event: str, **fields: Any) -> None:
    record = {"at": _utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_success_marker(path: Path) -> None:
    """Durably create the terminal success marker exactly once.

    ``SUCCESS`` is deliberately an exclusive marker rather than an ordinary
    text file.  The caller writes the result receipt and its final event
    first, then invokes this function as its last durable action.  Refusing
    to replace an existing marker makes an accidental reuse or a late
    exception visible rather than silently making stale success look current.
    """

    prepared = path.with_name(f".{path.name}.prepared-{os.getpid()}")
    descriptor = os.open(prepared, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write("phase-shared preflight passed\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Link is exclusive: if a marker already exists, no previous success
        # receipt can be replaced.  Keeping the immutable prepared file after
        # a successful link means the link itself is the final durable action;
        # on a write failure it is merely a diagnostic partial, never SUCCESS.
        os.link(prepared, path)
    except BaseException:
        # ``os.fdopen`` owns the descriptor only after it succeeds.  If it
        # does not, close the descriptor so an evidence-write failure cannot
        # leak a file handle.  The prepared file is intentionally preserved.
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _command_output(argv: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _provenance() -> dict[str, Any]:
    return {
        "at": _utc_now(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "repository_root": str(PROJECT_ROOT),
        "git_revision": _command_output(["git", "rev-parse", "HEAD"]),
        "python": sys.version,
        "platform": platform.platform(),
    }


def _gpu_snapshot(torch: Any, *, coordinator_device: str) -> dict[str, Any]:
    """Record memory without exposing process arguments or environment secrets."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if completed.returncode != 0:
        message = " ".join(completed.stderr.split())[:500]
        raise RuntimeError(f"nvidia-smi memory snapshot failed: {message or 'unknown failure'}")
    coordinator_index = int(str(coordinator_device).split(":", maxsplit=1)[1])
    torch.cuda.synchronize(coordinator_index)
    return {
        "nvidia_smi_rows": [line.strip() for line in completed.stdout.splitlines() if line.strip()],
        "coordinator_device": coordinator_device,
        "coordinator_torch_allocated_bytes": int(torch.cuda.memory_allocated(coordinator_index)),
        "coordinator_torch_reserved_bytes": int(torch.cuda.memory_reserved(coordinator_index)),
        "host_meminfo": _host_memory_summary(),
    }


def _host_memory_summary() -> dict[str, int] | None:
    source = Path("/proc/meminfo")
    if not source.is_file():
        return None
    values: dict[str, int] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if not separator or key not in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            continue
        pieces = raw.split()
        if pieces and pieces[0].isdigit():
            values[key] = int(pieces[0]) * 1024
    return values


def _optional_gdn_prefill_backend(backend: str | None) -> dict[str, str]:
    """Return the vLLM override only when this preflight explicitly requested it."""

    return {"gdn_prefill_backend": backend} if backend is not None else {}


def _rank_zero_vllm_options(args: argparse.Namespace) -> dict[str, Any]:
    """Build the cold rank-zero sampler options without coupling them to topology."""

    return {
        "gpu_memory_utilization": args.worker_gpu_mem_util,
        "dtype": args.worker_dtype,
        "enable_sleep_mode": True,
        "max_model_len": args.worker_max_model_len,
        "max_num_seqs": args.worker_max_num_seqs,
        "max_num_batched_tokens": args.worker_max_num_batched_tokens,
        "language_model_only": True,
        "logprobs_mode": "processed_logprobs",
        **_optional_gdn_prefill_backend(args.worker_gdn_prefill_backend),
    }


def _worker_vllm_options(args: argparse.Namespace) -> dict[str, Any]:
    """Build per-worker vLLM options without assuming a particular GPU layout."""

    return {
        "gpu_memory_utilization": args.worker_gpu_mem_util,
        # Do not let a preflight attest vLLM's mutable ``auto`` default when
        # the production target pins BF16 explicitly.
        "dtype": args.worker_dtype,
        "enable_sleep_mode": True,
        "max_model_len": args.worker_max_model_len,
        "max_num_seqs": args.worker_max_num_seqs,
        "max_num_batched_tokens": args.worker_max_num_batched_tokens,
        "language_model_only": True,
        "logprobs_mode": "processed_logprobs",
        "seed": args.worker_seed_base,
        **_optional_gdn_prefill_backend(args.worker_gdn_prefill_backend),
    }


def _contract_config(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize the launch knobs that must be reproduced with this evidence."""

    return {
        "model": args.model,
        "worker_gpu_memory_utilization": args.worker_gpu_mem_util,
        "worker_dtype": args.worker_dtype,
        "worker_max_model_len": args.worker_max_model_len,
        "worker_max_num_seqs": args.worker_max_num_seqs,
        "worker_max_num_batched_tokens": args.worker_max_num_batched_tokens,
        "worker_gdn_prefill_backend": args.worker_gdn_prefill_backend,
        "worker_seed_base": args.worker_seed_base,
        "target_logprob_chunk_size": args.target_logprob_chunk_size,
        "forward_microbatch_max_datums": args.forward_microbatch_max_datums,
        "packing_budgets": args.packing_budgets,
        "packing_probe_datums": args.packing_probe_datums,
        "parity_datums_per_rank": args.parity_datums_per_rank,
        "parity_sequence_tokens": args.parity_sequence_tokens,
        "effect_candidate_count": args.effect_candidate_count,
        "effect_probe_completion_tokens": args.effect_probe_completion_tokens,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": 0.0,
            "seed": args.lora_seed,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": False,
        },
        "learning_rate": args.learning_rate,
        "thresholds": {
            "adapter_effect_min": args.adapter_effect_min,
            "effect_cosine_min": args.effect_cosine_min,
            "effect_norm_ratio_min": args.effect_norm_ratio_min,
            "effect_norm_ratio_max": args.effect_norm_ratio_max,
            "effect_relative_l2_max": args.effect_relative_l2_max,
            "wake_score_max_abs": args.wake_score_max_abs,
            "loss_parity_atol": args.loss_parity_atol,
            "loss_parity_rtol": args.loss_parity_rtol,
            "parameter_parity_atol": args.parameter_parity_atol,
            "parameter_parity_atol_source": args.parameter_parity_atol_source,
            "parameter_parity_delta_cosine_min": args.parameter_parity_delta_cosine_min,
            "parameter_parity_delta_relative_l2_max": args.parameter_parity_delta_relative_l2_max,
            "parameter_parity_delta_max_abs": args.parameter_parity_delta_max_abs,
            "parameter_parity_delta_max_abs_source": args.parameter_parity_delta_max_abs_source,
            "parameter_parity_envelope_relative_l2_max": args.parameter_parity_envelope_relative_l2_max,
            "parameter_parity_b_sign_mismatch_max": args.parameter_parity_b_sign_mismatch_max,
            "parameter_parity_hard_safety_semantics": (
                "exact_common_initial_adapter_finite_nonzero_updates_and_final_coordinate_bound"
            ),
            "parameter_parity_monitoring_semantics": "non_gating_pending_cross_allocation_calibration",
            "gradient_parity_cosine_min": args.gradient_parity_cosine_min,
            "gradient_parity_norm_ratio_min": args.gradient_parity_norm_ratio_min,
            "gradient_parity_norm_ratio_max": args.gradient_parity_norm_ratio_max,
            "gradient_parity_relative_l2_max": args.gradient_parity_relative_l2_max,
            "gradient_parity_magnitude_weighted_sign_mismatch_max": (
                args.gradient_parity_magnitude_weighted_sign_mismatch_max
            ),
        },
    }


def _normalize_chat_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, list) and encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise RuntimeError(f"unexpected chat-template batch size {len(encoded)}")
        encoded = encoded[0]
    if not isinstance(encoded, list) or not encoded or not all(isinstance(token, int) for token in encoded):
        raise RuntimeError("tokenizer returned an empty or non-integer chat prompt")
    return [int(token) for token in encoded]


def _repeated_tokens(seed_tokens: Sequence[int], length: int, *, offset: int = 0) -> list[int]:
    if not seed_tokens:
        raise ValueError("cannot repeat an empty token sequence")
    cycle = list(seed_tokens)
    return [cycle[(position + offset) % len(cycle)] for position in range(length)]


def _cross_entropy_seed_tokens(tokenizer: Any) -> list[int]:
    """Return the exact cyclic token seed shared by update and effect probe.

    Keeping this as a single helper makes the post-update transport probe
    genuinely update-aligned: it scores suffixes from the very synthetic token
    distribution on which the fixed AdamW update was taken, rather than an
    unrelated natural-language arithmetic prompt whose tiny effect can be
    poorly conditioned.
    """

    seed = tokenizer.encode(
        "Solve the question carefully, state a concise answer, and justify it with relevant reasoning.",
        add_special_tokens=True,
    )
    if not seed:
        raise RuntimeError("tokenizer produced no seed tokens for fixed cross-entropy datums")
    if not all(isinstance(token, int) for token in seed):
        raise RuntimeError("tokenizer produced a non-integer fixed cross-entropy seed")
    return [int(token) for token in seed]


def _make_cross_entropy_datums(
    *,
    tokenizer: Any,
    count: int,
    sequence_length: int,
    torch: Any,
    types: Any,
    datum_from_model_input_weights: Any,
) -> list[Any]:
    seed = _cross_entropy_seed_tokens(tokenizer)
    datums: list[Any] = []
    for index in range(count):
        tokens = _repeated_tokens(seed, sequence_length, offset=index)
        weights = torch.ones(len(tokens), dtype=torch.float32)
        datums.append(datum_from_model_input_weights(types.ModelInput.from_ints(tokens=tokens), weights))
    return datums


def _parity_reference_datums_in_lpt_shard_order(
    datums: Sequence[Any],
    *,
    contract: ResolvedPreflight,
) -> tuple[tuple[Any, ...], tuple[tuple[int, ...], ...]]:
    """Order the single-rank reference exactly as replicated ranks see work.

    The replicated backend uses stable LPT token-cost sharding.  The reference
    needs the same rank-local groups, serialized in rank order, so its
    shard-sized BF16 forwards have the same datum grouping as the distributed
    forwards.  Importing the planner lazily keeps ``--dry-run`` CUDA-free.
    """

    from ctm.backends.local.phase_shared import plan_token_cost_balanced_shards
    from ctm.backends.local.replicated import datum_token_cost

    values = tuple(datums)
    shards = plan_token_cost_balanced_shards(
        values,
        tuple(datum_token_cost(datum) for datum in values),
        ranks=contract.topology.training_ranks,
    )
    original_indices = tuple(shard.original_indices for shard in shards)
    ordered = tuple(item.value for shard in shards for item in shard.items)
    flattened = tuple(index for indices in original_indices for index in indices)
    if tuple(sorted(flattened)) != tuple(range(len(values))):
        raise RuntimeError("LPT reference order does not cover the fixed parity batch exactly once")
    return ordered, original_indices


def _make_update_aligned_probe_candidates(
    tokenizer: Any,
    types: Any,
    *,
    count: int,
    sequence_length: int,
    completion_tokens: int,
) -> tuple[list[Any], list[list[int]], dict[str, Any]]:
    """Create fixed score suffixes from the repeated-token CE update distribution.

    Every candidate is one deterministic rotation of the exact cyclic token
    source used by :func:`_make_cross_entropy_datums`.  We split a fixed
    sequence into a non-empty prefix and a long scored suffix.  That makes the
    probe independent of the number of trainer/rollout GPUs while amplifying
    the one-update adapter signal over many target positions.  The returned
    receipt records the recipe and hashes the seed, never the token IDs.
    """

    if count < 1:
        raise ValueError("update-aligned probe requires at least one candidate")
    if sequence_length < 2:
        raise ValueError("update-aligned probe sequence length must be at least two tokens")
    if not 0 < completion_tokens < sequence_length:
        raise ValueError("update-aligned probe completion length must be in [1, sequence_length)")

    seed = _cross_entropy_seed_tokens(tokenizer)
    prompts: list[Any] = []
    completions: list[list[int]] = []
    for index in range(count):
        sequence = _repeated_tokens(seed, sequence_length, offset=index)
        prefix = sequence[:-completion_tokens]
        completion = sequence[-completion_tokens:]
        if not prefix or len(completion) != completion_tokens:
            raise RuntimeError("update-aligned probe split did not produce a non-empty prefix and fixed suffix")
        prompts.append(types.ModelInput.from_ints(tokens=prefix))
        completions.append(completion)
    seed_payload = json.dumps(seed, separators=(",", ":")).encode("ascii")
    return (
        prompts,
        completions,
        {
            "schema": "phase-shared-update-aligned-fixed-token-probe-v1",
            "kind": "repeated-cross-entropy-sequence-suffix-score-completions-v1",
            "candidate_count": count,
            "sequence_token_count": sequence_length,
            "prompt_token_count": sequence_length - completion_tokens,
            "completion_token_count_per_candidate": completion_tokens,
            "candidate_rotation_offsets": list(range(count)),
            "cross_entropy_seed_token_sha256": hashlib.sha256(seed_payload).hexdigest(),
        },
    )


def _flatten_rows(rows: Sequence[Sequence[float]], *, label: str) -> list[float]:
    flattened: list[float] = []
    for row_index, row in enumerate(rows):
        if not row:
            raise RuntimeError(f"{label} row {row_index} is empty")
        for value in row:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"{label} contains a non-finite score")
            flattened.append(numeric)
    if not flattened:
        raise RuntimeError(f"{label} has no scores")
    return flattened


def _max_abs_row_difference(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    label: str,
) -> float:
    if len(left) != len(right):
        raise RuntimeError(f"{label} row count differs: {len(left)} != {len(right)}")
    differences: list[float] = []
    for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if len(left_row) != len(right_row):
            raise RuntimeError(f"{label} token count differs at row {row_index}")
        differences.extend(abs(float(a) - float(b)) for a, b in zip(left_row, right_row, strict=True))
    if not differences:
        raise RuntimeError(f"{label} has no aligned scores")
    return max(differences)


def _effect_summary(
    *,
    worker_policy: Sequence[Sequence[float]],
    worker_base: Sequence[Sequence[float]],
    hf_policy: Sequence[Sequence[float]],
    hf_base: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Compare adapter-minus-base effect, not raw worker and HF scores."""

    for label, first, second in (
        ("worker policy/base", worker_policy, worker_base),
        ("HF policy/base", hf_policy, hf_base),
        ("worker/HF policy", worker_policy, hf_policy),
    ):
        if len(first) != len(second):
            raise RuntimeError(f"{label} row count differs")
        for row_index, (left, right) in enumerate(zip(first, second, strict=True)):
            if len(left) != len(right):
                raise RuntimeError(f"{label} token count differs at row {row_index}")
    worker_effect = [
        float(policy) - float(base)
        for policy_row, base_row in zip(worker_policy, worker_base, strict=True)
        for policy, base in zip(policy_row, base_row, strict=True)
    ]
    hf_effect = [
        float(policy) - float(base)
        for policy_row, base_row in zip(hf_policy, hf_base, strict=True)
        for policy, base in zip(policy_row, base_row, strict=True)
    ]
    if len(worker_effect) != len(hf_effect) or not worker_effect:
        raise RuntimeError("worker and HF effect vectors are missing or misaligned")
    if not all(math.isfinite(value) for value in [*worker_effect, *hf_effect]):
        raise RuntimeError("worker/HF effect vectors contain a non-finite value")
    vector_diagnostics = _effect_vector_diagnostics(worker_effect, hf_effect)
    absolute_differences = [abs(left - right) for left, right in zip(worker_effect, hf_effect, strict=True)]
    return {
        "token_count": len(worker_effect),
        "worker_effect_max_abs": max(abs(value) for value in worker_effect),
        "hf_effect_max_abs": max(abs(value) for value in hf_effect),
        "effect_difference_max_abs": max(absolute_differences),
        "effect_difference_mean_abs": fmean(absolute_differences),
        **vector_diagnostics,
    }


def _aligned_score_differences(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    label: str,
) -> list[float]:
    """Return aligned signed score differences, rejecting truncated probes."""

    if len(left) != len(right):
        raise RuntimeError(f"{label} row count differs: {len(left)} != {len(right)}")
    differences: list[float] = []
    for row_index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        if len(left_row) != len(right_row):
            raise RuntimeError(f"{label} token count differs at row {row_index}")
        for first, second in zip(left_row, right_row, strict=True):
            value = float(first) - float(second)
            if not math.isfinite(value):
                raise RuntimeError(f"{label} contains a non-finite score difference")
            differences.append(value)
    if not differences:
        raise RuntimeError(f"{label} has no aligned score positions")
    return differences


def _absolute_difference_summary(differences: Sequence[float]) -> dict[str, Any]:
    """Use the established worker-attestation metric shape without raw scores."""

    absolute = [abs(float(value)) for value in differences]
    if not absolute or not all(math.isfinite(value) for value in absolute):
        raise RuntimeError("effect parity has no finite score differences")
    ordered = sorted(absolute)
    p99_index = max(0, math.ceil(0.99 * len(ordered)) - 1)
    return {
        "token_count": len(absolute),
        "max_abs_difference": max(absolute),
        "mean_abs_difference": fmean(absolute),
        "p99_abs_difference": ordered[p99_index],
        # Keep the attestation compact: it is a cryptographically bound
        # provenance receipt, while the full synthetic inputs remain in this
        # preflight's result artifact only as a hash.
    }


def _effect_vector_diagnostics(worker_effect: Sequence[float], hf_effect: Sequence[float]) -> dict[str, float | None]:
    """Return compact directional diagnostics without retaining raw logprobs."""

    if len(worker_effect) != len(hf_effect) or not worker_effect:
        raise RuntimeError("worker and HF effect vectors are missing or misaligned")
    if not all(math.isfinite(float(value)) for value in [*worker_effect, *hf_effect]):
        raise RuntimeError("worker/HF effect vectors contain a non-finite value")
    worker_norm = math.sqrt(sum(float(value) * float(value) for value in worker_effect))
    hf_norm = math.sqrt(sum(float(value) * float(value) for value in hf_effect))
    difference_norm = math.sqrt(
        sum((float(worker) - float(hf)) ** 2 for worker, hf in zip(worker_effect, hf_effect, strict=True))
    )
    scale = max(worker_norm, hf_norm)
    relative_error = difference_norm / scale if scale > 0 else None
    norm_ratio = worker_norm / hf_norm if hf_norm > 0 else None
    relative_to_hf_error = difference_norm / hf_norm if hf_norm > 0 else None
    cosine = None
    if worker_norm > 0 and hf_norm > 0:
        cosine = sum(float(left) * float(right) for left, right in zip(worker_effect, hf_effect, strict=True)) / (
            worker_norm * hf_norm
        )
        # A finite-precision reduction can be outside the mathematical range
        # by a few ulps.  Clip only the reported/comparison value.
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "worker_effect_l2_norm": worker_norm,
        "hf_effect_l2_norm": hf_norm,
        "effect_difference_l2_norm": difference_norm,
        "effect_relative_l2_scale": scale,
        "effect_difference_relative_l2_error": relative_error,
        "worker_to_hf_effect_l2_norm_ratio": norm_ratio,
        "effect_difference_relative_to_hf_l2_error": relative_to_hf_error,
        "cosine_similarity": cosine,
    }


def _formal_worker_effect_parity(
    *,
    worker_policy: Sequence[Sequence[float]],
    worker_base: Sequence[Sequence[float]],
    hf_policy: Sequence[Sequence[float]],
    hf_base: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Build the repository's hash-bound Qwen worker-attestation evidence.

    This deliberately uses policy-minus-base effects.  Comparing raw policy
    logits alone could pass if both transports happened to score similarly
    while vLLM silently ignored the translated adapter.
    """

    worker_effect = _aligned_score_differences(
        worker_policy,
        worker_base,
        label="worker policy/base parity",
    )
    hf_effect = _aligned_score_differences(
        hf_policy,
        hf_base,
        label="HF policy/base parity",
    )
    effect_difference = _aligned_score_differences(
        [worker_effect],
        [hf_effect],
        label="worker/HF LoRA-effect parity",
    )
    vector_diagnostics = _effect_vector_diagnostics(worker_effect, hf_effect)
    return {
        "worker_v2_minus_base": _absolute_difference_summary(worker_effect),
        "coordinator_updated_minus_base": _absolute_difference_summary(hf_effect),
        "worker_minus_coordinator_effect": _absolute_difference_summary(effect_difference),
        **vector_diagnostics,
    }


def _effect_gate_summary(
    summary: Mapping[str, Any],
    *,
    min_effect: float,
    cosine_min: float,
    norm_ratio_min: float,
    norm_ratio_max: float,
    relative_l2_max: float,
) -> dict[str, Any]:
    """Describe strict magnitude, direction, scale, and residual gates for failure receipts."""

    worker_effect = float(summary["worker_effect_max_abs"])
    hf_effect = float(summary["hf_effect_max_abs"])
    cosine = summary["cosine_similarity"]
    worker_effect_passed = worker_effect >= min_effect
    hf_effect_passed = hf_effect >= min_effect
    cosine_passed = cosine is not None and float(cosine) >= cosine_min
    norm_ratio = summary["worker_to_hf_effect_l2_norm_ratio"]
    relative_l2_error = summary["effect_difference_relative_to_hf_l2_error"]
    norm_ratio_passed = norm_ratio is not None and norm_ratio_min <= float(norm_ratio) <= norm_ratio_max
    relative_l2_passed = relative_l2_error is not None and float(relative_l2_error) <= relative_l2_max
    return {
        "adapter_effect_min": min_effect,
        "effect_cosine_min": cosine_min,
        "effect_norm_ratio_min": norm_ratio_min,
        "effect_norm_ratio_max": norm_ratio_max,
        "effect_relative_l2_max": relative_l2_max,
        "worker_effect_passed": worker_effect_passed,
        "hf_effect_passed": hf_effect_passed,
        "cosine_passed": cosine_passed,
        "norm_ratio_passed": norm_ratio_passed,
        "relative_l2_passed": relative_l2_passed,
        "passed": worker_effect_passed
        and hf_effect_passed
        and cosine_passed
        and norm_ratio_passed
        and relative_l2_passed,
    }


def _require_effect(
    summary: dict[str, Any],
    *,
    min_effect: float,
    cosine_min: float,
    norm_ratio_min: float,
    norm_ratio_max: float,
    relative_l2_max: float,
    label: str,
) -> None:
    gate = _effect_gate_summary(
        summary,
        min_effect=min_effect,
        cosine_min=cosine_min,
        norm_ratio_min=norm_ratio_min,
        norm_ratio_max=norm_ratio_max,
        relative_l2_max=relative_l2_max,
    )
    worker_effect = float(summary["worker_effect_max_abs"])
    hf_effect = float(summary["hf_effect_max_abs"])
    cosine = summary["cosine_similarity"]
    if not gate["worker_effect_passed"]:
        raise RuntimeError(
            f"{label}: worker adapter has no measurable non-base effect (max_abs={worker_effect:.3e}, required>={min_effect:.3e})"
        )
    if not gate["hf_effect_passed"]:
        raise RuntimeError(
            f"{label}: HF adapter has no measurable non-base effect (max_abs={hf_effect:.3e}, required>={min_effect:.3e})"
        )
    if not gate["cosine_passed"]:
        raise RuntimeError(
            f"{label}: worker and HF adapter effects disagree (cosine_similarity={cosine!r}, required>={cosine_min:.3f})"
        )
    if not gate["norm_ratio_passed"]:
        raise RuntimeError(
            f"{label}: worker/HF adapter effect L2 norm ratio is outside the strict fidelity interval "
            f"(ratio={summary['worker_to_hf_effect_l2_norm_ratio']!r}, "
            f"required=[{norm_ratio_min:.3f}, {norm_ratio_max:.3f}])"
        )
    if not gate["relative_l2_passed"]:
        raise RuntimeError(
            f"{label}: worker/HF adapter effect residual is too large "
            f"(relative_to_hf_l2={summary['effect_difference_relative_to_hf_l2_error']!r}, "
            f"required<={relative_l2_max:.3f})"
        )


async def _resolve_pending(pending: Any) -> Any:
    result = getattr(pending, "result", None)
    if not callable(result):
        return pending
    return await result()


def _parity_reference_microbatch_max_datums(args: argparse.Namespace) -> int:
    """Match the reference's local forward shape to one replica's shard.

    The synthetic parity batch contains exactly ``parity_datums_per_rank``
    equal-length datums for every replica.  A single-rank reference that puts
    the whole global batch in one BF16 forward tests a different GEMM shape
    from the replicated run, making its scalar loss needlessly sensitive to
    kernel-level BF16 rounding.  It still accumulates the same global
    objective and takes one optimizer step; it simply performs the forwards in
    local-shard-sized chunks, as every replica does.
    """

    return min(args.forward_microbatch_max_datums, args.parity_datums_per_rank)


def _fixed_update_loss_parity(
    *,
    single_rank_loss: float,
    replicated_loss: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Describe a loss-equivalence candidate without bypassing state parity.

    Absolute equality remains the preferred gate.  For a BF16 forward split
    across replicas, a small shape/reduction-dependent scalar difference may
    exceed a fixed absolute threshold while the objective and resulting adapter
    remain equivalent.  Such a relative-only candidate is *not* accepted here:
    the caller must subsequently obtain the raw adapter-state parity proof.
    Large differences fail before any adapter publication work is attempted.
    """

    single = float(single_rank_loss)
    replicated = float(replicated_loss)
    if not all(math.isfinite(value) for value in (single, replicated, atol, rtol)):
        raise ValueError("fixed-update loss parity requires finite losses and tolerances")
    if atol <= 0 or rtol <= 0:
        raise ValueError("fixed-update loss parity tolerances must be positive")
    absolute_difference = abs(single - replicated)
    scale = max(abs(single), abs(replicated))
    relative_difference = 0.0 if scale == 0.0 else absolute_difference / scale
    relative_tolerance_absolute = rtol * scale
    absolute_passed = absolute_difference <= atol
    relative_passed = scale > 0.0 and absolute_difference <= relative_tolerance_absolute
    candidate_passed = absolute_passed or relative_passed
    return {
        "single_rank_loss": single,
        "replicated_loss": replicated,
        "abs_difference": absolute_difference,
        "relative_difference": relative_difference,
        "scale": scale,
        "atol": atol,
        "rtol": rtol,
        "relative_tolerance_absolute": relative_tolerance_absolute,
        "effective_candidate_tolerance": max(atol, relative_tolerance_absolute),
        "absolute_passed": absolute_passed,
        "relative_passed": relative_passed,
        "candidate_passed": candidate_passed,
        # This becomes true only after the existing raw-adapter comparison
        # verifies the update state if the absolute loss condition failed.
        "passed": absolute_passed,
        "acceptance_basis": "absolute_loss" if absolute_passed else None,
        "requires_adapter_state_parity": not absolute_passed,
    }


def _ordered_trainable_parameters(
    backend: Any,
    *,
    expected_parameters: Sequence[Any] | None = None,
) -> list[tuple[str, Any]]:
    """Return the exact optimizer-order trainable parameter schema.

    ``LocalBackend.submit_optim_step`` builds its optimizer list from
    ``model.parameters()``.  The pre-optimizer receipt needs named tensors, so
    prove the named traversal has exactly that same object order instead of
    silently assuming it.
    """

    require_model = getattr(backend, "_require_model", None)
    model = require_model() if callable(require_model) else getattr(backend, "model", None)
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError("pre-optimizer gradient capture requires an initialized named-parameter model")
    ordered = [(str(name), parameter) for name, parameter in named_parameters() if parameter.requires_grad]
    if not ordered:
        raise RuntimeError("pre-optimizer gradient capture found no trainable parameters")
    names = [name for name, _ in ordered]
    if len(names) != len(set(names)):
        raise RuntimeError("pre-optimizer gradient capture found duplicate trainable parameter names")
    if expected_parameters is not None:
        expected_ids = [id(parameter) for parameter in expected_parameters]
        named_ids = [id(parameter) for _, parameter in ordered]
        if named_ids != expected_ids:
            raise RuntimeError(
                "pre-optimizer gradient capture's named trainable order differs from LocalBackend's optimizer order"
            )
    return ordered


def _gradient_tensor_name_receipt(names: Sequence[str]) -> dict[str, Any]:
    encoded = json.dumps(list(names), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "tensor_count": len(names),
        "ordered_tensor_names": list(names),
        "ordered_tensor_names_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _atomic_save_safetensors(
    path: Path,
    tensors: Mapping[str, Any],
    *,
    metadata: Mapping[str, str],
) -> None:
    """Write one immutable safetensors snapshot without replacing evidence."""

    from safetensors.torch import save_file

    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable gradient evidence: {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite partial gradient evidence: {temporary}")
    # Retain any partial write after a serializer failure: it is diagnostic
    # evidence inside a deliberately fresh run directory, not disposable cache.
    save_file(dict(tensors), str(temporary), metadata=dict(metadata))
    os.replace(temporary, path)


def _capture_pre_optimizer_gradients(
    backend: Any,
    *,
    output_path: Path,
    capture_point: str,
    expected_parameters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Snapshot raw un-clipped gradients after F/B and before AdamW.

    The snapshots retain their native dtype.  Comparisons load the immutable
    files and promote to FP64, so diagnostic evidence never changes the actual
    optimizer input.
    """

    ordered = _ordered_trainable_parameters(backend, expected_parameters=expected_parameters)
    tensors: dict[str, Any] = {}
    dtype_counts: dict[str, int] = {}
    value_count = 0
    for name, parameter in ordered:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"pre-optimizer gradient capture found no gradient for trainable tensor {name}")
        if bool(getattr(gradient, "is_sparse", False)):
            raise RuntimeError(f"pre-optimizer gradient capture does not admit sparse gradient {name}")
        snapshot = gradient.detach().cpu().clone().contiguous()
        if not bool(snapshot.is_floating_point()):
            raise RuntimeError(f"pre-optimizer gradient capture found non-floating gradient {name}")
        if not bool(snapshot.isfinite().all().item()):
            raise RuntimeError(f"pre-optimizer gradient capture found non-finite gradient {name}")
        tensors[name] = snapshot
        dtype_name = str(snapshot.dtype)
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + int(snapshot.numel())
        value_count += int(snapshot.numel())
    if value_count < 1:
        raise RuntimeError("pre-optimizer gradient capture found zero gradient values")
    names = list(tensors)
    name_receipt = _gradient_tensor_name_receipt(names)
    _atomic_save_safetensors(
        output_path,
        tensors,
        metadata={
            "schema": "qwen35-phase-shared-pre-optimizer-gradients-v1",
            "capture_point": capture_point,
            "ordered_tensor_names_sha256": str(name_receipt["ordered_tensor_names_sha256"]),
        },
    )
    return {
        "schema": "qwen35-phase-shared-pre-optimizer-gradients-v1",
        "capture_point": capture_point,
        "path": str(output_path),
        "sha256": _sha256_file(output_path),
        "value_count": value_count,
        "dtype_value_counts": dtype_counts,
        **name_receipt,
    }


def _current_gradient_summary(
    backend: Any,
    *,
    expected_parameters: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Summarize currently resident raw gradients without retaining values."""

    import torch

    ordered = _ordered_trainable_parameters(backend, expected_parameters=expected_parameters)
    value_count = 0
    active_value_count = 0
    active_tensor_count = 0
    l1_norm = 0.0
    squared_norm = 0.0
    max_abs = 0.0
    for name, parameter in ordered:
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(f"pre-optimizer gradient summary found no gradient for trainable tensor {name}")
        values = gradient.detach()
        if not bool(torch.isfinite(values).all().item()):
            raise RuntimeError(f"pre-optimizer gradient summary found non-finite gradient {name}")
        absolute = values.abs()
        tensor_value_count = int(values.numel())
        tensor_active_count = int((values != 0).sum().item())
        value_count += tensor_value_count
        active_value_count += tensor_active_count
        active_tensor_count += int(tensor_active_count > 0)
        l1_norm += float(absolute.double().sum().item())
        squared_norm += float(values.double().square().sum().item())
        max_abs = max(max_abs, float(absolute.max().item()) if tensor_value_count else 0.0)
    if value_count < 1:
        raise RuntimeError("pre-optimizer gradient summary found zero values")
    return {
        "tensor_count": len(ordered),
        "value_count": value_count,
        "active_value_count": active_value_count,
        "active_tensor_count": active_tensor_count,
        "l1_norm": l1_norm,
        "l2_norm": math.sqrt(squared_norm),
        "max_abs": max_abs,
    }


def _gradient_vector_parity(
    *,
    reference: Mapping[str, Any],
    replicated: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare ordered raw-gradient snapshots in FP64.

    This is independent from the first AdamW state update.  It therefore
    catches a sharding/reducer failure before moment and sign normalisation can
    make a bad raw gradient appear superficially plausible downstream.
    """

    import torch

    reference_names = list(reference)
    replicated_names = list(replicated)
    if reference_names != replicated_names:
        reference_set = set(reference_names)
        replicated_set = set(replicated_names)
        missing = sorted(reference_set - replicated_set)
        extra = sorted(replicated_set - reference_set)
        if not missing and not extra:
            raise RuntimeError("pre-optimizer gradient tensor order differs despite an identical schema")
        raise RuntimeError("pre-optimizer gradient tensor schemas differ: " f"missing={missing[:3]}, extra={extra[:3]}")
    if not reference_names:
        raise RuntimeError("pre-optimizer gradient comparison found no tensors")

    value_count = 0
    reference_squared_norm = 0.0
    replicated_squared_norm = 0.0
    error_squared_norm = 0.0
    dot_product = 0.0
    max_abs_error = -1.0
    max_abs_tensor: str | None = None
    total_abs_error = 0.0
    sign_mismatch_count = 0
    magnitude_weighted_sign_disagreement = 0.0
    magnitude_weight = 0.0
    per_tensor: list[dict[str, Any]] = []

    for name in reference_names:
        reference_tensor = reference[name]
        replicated_tensor = replicated[name]
        if tuple(reference_tensor.shape) != tuple(replicated_tensor.shape):
            raise RuntimeError(
                f"pre-optimizer gradient tensor shape differs for {name}: "
                f"{tuple(reference_tensor.shape)} != {tuple(replicated_tensor.shape)}"
            )
        if reference_tensor.dtype != replicated_tensor.dtype:
            raise RuntimeError(
                f"pre-optimizer gradient tensor dtype differs for {name}: "
                f"{reference_tensor.dtype} != {replicated_tensor.dtype}"
            )
        reference64 = reference_tensor.detach().to(device="cpu", dtype=torch.float64)
        replicated64 = replicated_tensor.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(reference64).all().item()) or not bool(torch.isfinite(replicated64).all().item()):
            raise RuntimeError(f"pre-optimizer gradient tensor {name} contains a non-finite value")
        error64 = replicated64 - reference64
        abs_error = error64.abs()
        reference_abs = reference64.abs()
        replicated_abs = replicated64.abs()
        # ``torch.sign`` makes a dropped nonzero gradient disagree with zero,
        # while a shared exact zero remains neutral.  Weight each disagreement
        # by the larger magnitude, so tiny near-zero sign noise cannot defeat
        # an otherwise identical global gradient.
        sign_mismatch = torch.sign(reference64) != torch.sign(replicated64)
        magnitude = torch.maximum(reference_abs, replicated_abs)

        tensor_value_count = int(reference64.numel())
        reference_sq = float(reference64.square().sum().item())
        replicated_sq = float(replicated64.square().sum().item())
        error_sq = float(error64.square().sum().item())
        tensor_dot = float((reference64 * replicated64).sum().item())
        reference_l2 = math.sqrt(reference_sq)
        replicated_l2 = math.sqrt(replicated_sq)
        error_l2 = math.sqrt(error_sq)
        tensor_norm_ratio = replicated_l2 / reference_l2 if reference_l2 > 0 else None
        tensor_relative_l2 = error_l2 / reference_l2 if reference_l2 > 0 else None
        tensor_cosine = None
        if reference_l2 > 0 and replicated_l2 > 0:
            tensor_cosine = max(-1.0, min(1.0, tensor_dot / (reference_l2 * replicated_l2)))
        tensor_max_abs = float(abs_error.max().item()) if tensor_value_count else 0.0
        tensor_total_abs = float(abs_error.sum().item())
        tensor_sign_mismatches = int(sign_mismatch.sum().item())
        tensor_weight = float(magnitude.sum().item())
        tensor_weighted_disagreement = float((magnitude * sign_mismatch.to(torch.float64)).sum().item())
        if tensor_max_abs > max_abs_error:
            max_abs_error = tensor_max_abs
            max_abs_tensor = name
        value_count += tensor_value_count
        reference_squared_norm += reference_sq
        replicated_squared_norm += replicated_sq
        error_squared_norm += error_sq
        dot_product += tensor_dot
        total_abs_error += tensor_total_abs
        sign_mismatch_count += tensor_sign_mismatches
        magnitude_weight += tensor_weight
        magnitude_weighted_sign_disagreement += tensor_weighted_disagreement
        per_tensor.append(
            {
                "name": name,
                "value_count": tensor_value_count,
                "reference_l2_norm": reference_l2,
                "replicated_l2_norm": replicated_l2,
                "difference_l2_norm": error_l2,
                "norm_ratio": tensor_norm_ratio,
                "relative_l2_error": tensor_relative_l2,
                "cosine_similarity": tensor_cosine,
                "max_abs_error": tensor_max_abs,
                "mean_abs_error": tensor_total_abs / tensor_value_count if tensor_value_count else 0.0,
                "sign_mismatch_fraction": tensor_sign_mismatches / tensor_value_count if tensor_value_count else 0.0,
                "magnitude_weighted_sign_disagreement": (
                    tensor_weighted_disagreement / tensor_weight if tensor_weight > 0 else 0.0
                ),
            }
        )

    if value_count < 1:
        raise RuntimeError("pre-optimizer gradient comparison found zero values")
    reference_l2 = math.sqrt(reference_squared_norm)
    replicated_l2 = math.sqrt(replicated_squared_norm)
    error_l2 = math.sqrt(error_squared_norm)
    cosine = None
    if reference_l2 > 0 and replicated_l2 > 0:
        cosine = max(-1.0, min(1.0, dot_product / (reference_l2 * replicated_l2)))
    norm_ratio = replicated_l2 / reference_l2 if reference_l2 > 0 else None
    relative_l2 = error_l2 / reference_l2 if reference_l2 > 0 else None
    top_offenders = sorted(
        per_tensor,
        key=lambda entry: (
            # A shared exact-zero LoRA-A gradient has no relative error; it
            # is evidence of agreement, not a top offender.
            -1.0 if entry["relative_l2_error"] is None else float(entry["relative_l2_error"]),
            float(entry["magnitude_weighted_sign_disagreement"]),
            float(entry["max_abs_error"]),
        ),
        reverse=True,
    )[:20]
    return {
        "schema": "qwen35-phase-shared-pre-optimizer-gradient-parity-v1",
        "tensor_count": len(reference_names),
        "value_count": value_count,
        "reference_l2_norm": reference_l2,
        "replicated_l2_norm": replicated_l2,
        "difference_l2_norm": error_l2,
        "cosine_similarity": cosine,
        "norm_ratio": norm_ratio,
        "relative_l2_error": relative_l2,
        "max_abs_error": max_abs_error,
        "max_abs_tensor": max_abs_tensor,
        "mean_abs_error": total_abs_error / value_count,
        "sign_mismatch_fraction": sign_mismatch_count / value_count,
        "magnitude_weighted_sign_disagreement": (
            magnitude_weighted_sign_disagreement / magnitude_weight if magnitude_weight > 0 else 0.0
        ),
        "reference_gradient_nonzero": reference_l2 > 0,
        "replicated_gradient_nonzero": replicated_l2 > 0,
        "per_tensor": per_tensor,
        "top_offenders": top_offenders,
    }


def _gradient_parity_gate_summary(
    report: Mapping[str, Any],
    *,
    cosine_min: float,
    norm_ratio_min: float,
    norm_ratio_max: float,
    relative_l2_max: float,
    magnitude_weighted_sign_mismatch_max: float,
) -> dict[str, Any]:
    """Apply the scientific pre-optimizer gradient envelope to a receipt."""

    thresholds = (
        cosine_min,
        norm_ratio_min,
        norm_ratio_max,
        relative_l2_max,
        magnitude_weighted_sign_mismatch_max,
    )
    if not all(math.isfinite(float(value)) for value in thresholds):
        raise ValueError("pre-optimizer gradient parity thresholds must be finite")
    if not (
        0 < cosine_min <= 1
        and 0 < norm_ratio_min <= norm_ratio_max
        and 0 < relative_l2_max <= 1
        and 0 < magnitude_weighted_sign_mismatch_max <= 1
    ):
        raise ValueError("pre-optimizer gradient parity thresholds are outside their valid ranges")
    cosine = report["cosine_similarity"]
    norm_ratio = report["norm_ratio"]
    relative_l2 = report["relative_l2_error"]
    weighted_sign = float(report["magnitude_weighted_sign_disagreement"])
    reference_nonzero = bool(report["reference_gradient_nonzero"])
    replicated_nonzero = bool(report["replicated_gradient_nonzero"])
    ordered_tensor_schema_match = bool(report.get("ordered_tensor_schema_match", True))
    return {
        "gradient_parity_cosine_min": cosine_min,
        "gradient_parity_norm_ratio_min": norm_ratio_min,
        "gradient_parity_norm_ratio_max": norm_ratio_max,
        "gradient_parity_relative_l2_max": relative_l2_max,
        "gradient_parity_magnitude_weighted_sign_mismatch_max": magnitude_weighted_sign_mismatch_max,
        "reference_gradient_nonzero": reference_nonzero,
        "replicated_gradient_nonzero": replicated_nonzero,
        "ordered_tensor_schema_match": ordered_tensor_schema_match,
        "cosine_passed": cosine is not None and float(cosine) >= cosine_min,
        "norm_ratio_passed": norm_ratio is not None and norm_ratio_min <= float(norm_ratio) <= norm_ratio_max,
        "relative_l2_passed": relative_l2 is not None and float(relative_l2) <= relative_l2_max,
        "magnitude_weighted_sign_mismatch_passed": weighted_sign <= magnitude_weighted_sign_mismatch_max,
        "passed": (
            reference_nonzero
            and replicated_nonzero
            and ordered_tensor_schema_match
            and cosine is not None
            and float(cosine) >= cosine_min
            and norm_ratio is not None
            and norm_ratio_min <= float(norm_ratio) <= norm_ratio_max
            and relative_l2 is not None
            and float(relative_l2) <= relative_l2_max
            and weighted_sign <= magnitude_weighted_sign_mismatch_max
        ),
    }


def _compare_pre_optimizer_gradients(
    *,
    reference_path: Path,
    replicated_path: Path,
) -> dict[str, Any]:
    """Load immutable raw gradients and return their full comparison receipt."""

    from safetensors.torch import load_file

    if not reference_path.is_file() or not replicated_path.is_file():
        raise RuntimeError(
            "pre-optimizer gradient evidence is incomplete: "
            f"reference={reference_path.is_file()}, replicated={replicated_path.is_file()}"
        )
    report = _gradient_vector_parity(
        reference=load_file(str(reference_path), device="cpu"),
        replicated=load_file(str(replicated_path), device="cpu"),
    )
    report.update(
        {
            "reference_gradient_path": str(reference_path),
            "reference_gradient_sha256": _sha256_file(reference_path),
            "replicated_gradient_path": str(replicated_path),
            "replicated_gradient_sha256": _sha256_file(replicated_path),
        }
    )
    return report


def _require_gradient_parity(report: Mapping[str, Any]) -> None:
    """Raise only after the full pre-optimizer receipt is attached to result."""

    gate = report.get("gate")
    if not isinstance(gate, Mapping) or bool(gate.get("passed")):
        return
    raise RuntimeError(
        "single-rank and replicated pre-optimizer gradients differ beyond the strict envelope: "
        f"ordered_schema_match={report.get('ordered_tensor_schema_match')!r}, "
        f"cosine={report['cosine_similarity']!r}, norm_ratio={report['norm_ratio']!r}, "
        f"relative_l2={report['relative_l2_error']!r}, "
        f"magnitude_weighted_sign_disagreement={report['magnitude_weighted_sign_disagreement']:.6g}"
    )


def _install_rank_zero_gradient_capture(
    rank_zero: Any,
    *,
    output_path: Path,
    diagnostic_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Wrap rank zero's reducer after setup, without changing peer collectives.

    The wrapper invokes the already-configured global-SUM reducer first, then
    captures its post-SUM raw gradient before clipping or AdamW.  It is only a
    rank-zero diagnostic wrapper; the distributed engine and every peer
    reducer retain their original implementation and collective count.
    """

    original_reducer = getattr(rank_zero, "_gradient_reducer", None)
    setter = getattr(rank_zero, "set_gradient_reducer", None)
    if not callable(original_reducer) or not callable(setter):
        raise RuntimeError("phase-shared rank zero has no configured optimizer-boundary gradient reducer to wrap")
    state: dict[str, Any] = {
        "status": "installed",
        "capture_count": 0,
        "receipt": None,
        "output_path": str(output_path),
        "gradient_accumulations_before_reduce": None,
        "pre_sum_rank_zero_gradient_summary": None,
        "wrapped_existing_reducer": True,
        "capture_after_original_global_sum": True,
    }

    def publish_diagnostic() -> None:
        if diagnostic_sink is not None:
            # Pass the same mutable receipt that the caller attaches before
            # F/B.  A peer optimizer/hash failure after rank zero captures the
            # artifact therefore still serializes its path and checksum.
            diagnostic_sink(state)

    def capture_after_reduce(parameters: Sequence[Any]) -> Any:
        # A second call would add an unexpected NCCL collective.  Reject it
        # before touching the original reducer so a programming error cannot
        # enter a peer-mismatched all-reduce.
        if int(state["capture_count"]) != 0:
            raise RuntimeError("pre-optimizer global gradient capture was invoked more than once")
        accumulations = int(getattr(rank_zero, "_gradient_accumulations", -1))
        state["gradient_accumulations_before_reduce"] = accumulations
        state["pre_sum_rank_zero_gradient_summary"] = _current_gradient_summary(
            rank_zero,
            expected_parameters=parameters,
        )
        # Mark the invocation before NCCL.  If the original reducer itself
        # raises, a retry must still be rejected locally rather than issuing a
        # second collective while peers may be in an unknown state.
        state["capture_count"] = 1
        state["status"] = "original_reducer_in_flight"
        publish_diagnostic()
        # Complete the original all-reduce before enforcing the local
        # accumulation invariant, so a diagnostic failure cannot strand peers
        # inside NCCL.  The snapshot below then remains available in failure
        # evidence even for an unexpected accumulation count.
        try:
            result = original_reducer(parameters)
        except BaseException as exc:
            state["status"] = "original_reducer_failed"
            state["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            publish_diagnostic()
            raise
        state["status"] = "capturing"
        publish_diagnostic()
        try:
            state["receipt"] = _capture_pre_optimizer_gradients(
                rank_zero,
                output_path=output_path,
                capture_point="after_replicated_global_gradient_sum_before_adamw",
                expected_parameters=parameters,
            )
        except BaseException as exc:
            state["status"] = "capture_failed"
            state["failure"] = {"type": type(exc).__name__, "message": str(exc)}
            publish_diagnostic()
            raise
        state["status"] = "captured"
        publish_diagnostic()
        if accumulations != 1:
            raise RuntimeError(
                "replicated rank zero must have exactly one accumulated raw gradient before AdamW; "
                f"got {accumulations}"
            )
        return result

    setter(capture_after_reduce)
    publish_diagnostic()
    return state


async def _single_rank_reference_update(
    *,
    args: argparse.Namespace,
    contract: ResolvedPreflight,
    datums: Sequence[Any],
    planned_original_indices: Sequence[Sequence[int]],
    output_dir: Path,
    torch: Any,
    LocalBackend: Any,
    LoRAConfig: Any,
    AdamConfig: Any,
    trainable_state_hash: Any,
    runtime_model: str | None = None,
    diagnostic_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Make one reference update from base before starting any vLLM worker."""

    single_dir = output_dir / "single_rank_reference"
    runtime_model = args.model if runtime_model is None else runtime_model
    if not isinstance(runtime_model, str) or not runtime_model:
        raise ValueError("single-rank reference requires a non-empty runtime model argument")
    parity_reference_microbatch_max_datums = _parity_reference_microbatch_max_datums(args)
    reference: dict[str, Any] = {
        "status": "starting",
        "scope": (
            "one-rank HF/PEFT reference with replica-shape-matched forward microbatches; no vLLM worker was started"
        ),
        "canonical_model_repo": args.model,
        "runtime_model_argument": runtime_model,
        "run_dir": str(single_dir),
        "forward_microbatch_max_datums": parity_reference_microbatch_max_datums,
        "planned_original_indices": [list(indices) for indices in planned_original_indices],
    }

    def publish_diagnostic() -> None:
        if diagnostic_sink is not None:
            # The outer result stores this receipt before model construction,
            # so an F/B, capture, or AdamW failure keeps every immutable path
            # and checksum already earned by the reference run.
            diagnostic_sink(reference)

    publish_diagnostic()
    backend: Any | None = None
    try:
        single_dir.mkdir(parents=True, exist_ok=False)
        reference["status"] = "constructing_backend"
        publish_diagnostic()
        torch.manual_seed(args.lora_seed)
        backend = LocalBackend(
            device=contract.coordinator_device,
            dtype=torch.bfloat16,
            use_lora=True,
            sampler="hf",
            gradient_checkpointing=True,
            gradient_checkpointing_layers=contract.checkpoint_layers,
            forward_microbatch_max_datums=parity_reference_microbatch_max_datums,
            forward_microbatch_max_tokens=min(args.packing_budgets),
            target_logprob_chunk_size=args.target_logprob_chunk_size,
        )
        reference["status"] = "setting_up"
        publish_diagnostic()
        setup_started = time.monotonic()
        backend.setup(
            model=runtime_model,
            lora=LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=0.0,
                seed=args.lora_seed,
                train_attn=True,
                train_mlp=True,
                train_unembed=False,
            ),
        )
        torch.cuda.synchronize(int(contract.coordinator_device.split(":", maxsplit=1)[1]))
        setup_seconds = time.monotonic() - setup_started
        state_before = trainable_state_hash(backend)
        reference["setup_seconds"] = setup_seconds
        reference["state_hash_before"] = state_before
        initial_adapter_dir = single_dir / "initial_adapter"
        reference["initial_adapter_dir"] = str(initial_adapter_dir)
        reference["status"] = "saving_initial_adapter"
        publish_diagnostic()
        backend._require_model().save_pretrained(str(initial_adapter_dir))
        initial_adapter_weights = initial_adapter_dir / "adapter_model.safetensors"
        if not initial_adapter_weights.is_file():
            raise RuntimeError(f"single-rank reference did not write {initial_adapter_weights.name} before its update")
        reference["initial_adapter_model_sha256"] = _sha256_file(initial_adapter_weights)
        reference["status"] = "forward_backward"
        publish_diagnostic()
        forward_started = time.monotonic()
        forward_output = await _resolve_pending(await backend.submit_forward_backward(datums, "cross_entropy"))
        torch.cuda.synchronize(int(contract.coordinator_device.split(":", maxsplit=1)[1]))
        forward_seconds = time.monotonic() - forward_started
        gradient_accumulations_before_optimizer = int(getattr(backend, "_gradient_accumulations", -1))
        reference["forward_backward_seconds"] = forward_seconds
        reference["loss"] = float(forward_output.metrics["loss"])
        reference["gradient_accumulations_before_optimizer"] = gradient_accumulations_before_optimizer
        if gradient_accumulations_before_optimizer != 1:
            raise RuntimeError(
                "single-rank reference must have exactly one accumulated raw gradient before AdamW; "
                f"got {gradient_accumulations_before_optimizer}"
            )
        pre_optimizer_gradient_summary = _current_gradient_summary(backend)
        reference["pre_optimizer_gradient_summary"] = pre_optimizer_gradient_summary
        reference["status"] = "capturing_pre_optimizer_gradient"
        publish_diagnostic()
        pre_optimizer_gradient = _capture_pre_optimizer_gradients(
            backend,
            output_path=single_dir / "pre_optimizer_gradients.safetensors",
            capture_point="after_single_rank_forward_backward_before_adamw",
        )
        reference["pre_optimizer_gradient"] = pre_optimizer_gradient
        reference["status"] = "optimizer_step"
        publish_diagnostic()
        optimizer_started = time.monotonic()
        await _resolve_pending(
            await backend.submit_optim_step(
                learning_rate=args.learning_rate,
                adam=AdamConfig(learning_rate=args.learning_rate),
            )
        )
        torch.cuda.synchronize(int(contract.coordinator_device.split(":", maxsplit=1)[1]))
        optimizer_seconds = time.monotonic() - optimizer_started
        state_after = trainable_state_hash(backend)
        adapter_dir = single_dir / "adapter"
        reference["optimizer_seconds"] = optimizer_seconds
        reference["state_hash_after"] = state_after
        reference["adapter_dir"] = str(adapter_dir)
        reference["status"] = "saving_final_adapter"
        publish_diagnostic()
        backend._require_model().save_pretrained(str(adapter_dir))
        adapter_weights = adapter_dir / "adapter_model.safetensors"
        if not adapter_weights.is_file():
            raise RuntimeError(f"single-rank reference did not write {adapter_weights.name}")
        reference["adapter_model_sha256"] = _sha256_file(adapter_weights)
        reference["status"] = "completed"
        publish_diagnostic()
        return reference
    except BaseException as exc:
        reference["status"] = "failed"
        reference["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        publish_diagnostic()
        raise
    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except BaseException as exc:
                # Do not obscure an already-recorded F/B or optimizer error.
                # A shutdown failure after a nominal update is itself evidence
                # of an incomplete reference run and must be visible upstream.
                if reference.get("status") != "failed":
                    reference["status"] = "failed"
                    reference["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                    publish_diagnostic()
                    raise
                reference["teardown_failure"] = {"type": type(exc).__name__, "message": str(exc)}
                publish_diagnostic()
        gc.collect()
        torch.cuda.empty_cache()


def _adapter_update_delta_parity(
    *,
    reference_initial: Mapping[str, Any],
    reference_final: Mapping[str, Any],
    replicated_initial: Mapping[str, Any],
    replicated_final: Mapping[str, Any],
    atol: float,
    delta_cosine_min: float,
    delta_relative_l2_max: float,
    delta_max_abs_atol: float,
    envelope_relative_l2_max: float = 0.035,
    b_sign_mismatch_max: float = 4e-4,
) -> dict[str, Any]:
    """Compare final adapters *and* the update vectors from a common start.

    The raw-gradient receipt is the hard semantic comparison.  This post-Adam
    receipt still requires an exact common initial state, finite non-zero
    update norms, and a bounded final coordinate discrepancy.  Direction,
    relative-L2, diagnostic max-error, and LoRA-B sign metrics are preserved
    for calibration across allocations, but do not reject a run on their own:
    first-step AdamW can amplify a tiny BF16 gradient sign difference into a
    full learning-rate coordinate jump.  The implementation streams tensors
    and promotes norm accumulation to FP64 so the evidence itself does not
    add a float32 reduction artefact.
    """

    import torch

    if not all(
        math.isfinite(value)
        for value in (
            atol,
            delta_cosine_min,
            delta_relative_l2_max,
            delta_max_abs_atol,
            envelope_relative_l2_max,
            b_sign_mismatch_max,
        )
    ):
        raise ValueError("adapter parity tolerances must be finite")
    if (
        atol <= 0
        or delta_max_abs_atol <= 0
        or not 0 < delta_cosine_min <= 1
        or not 0 < delta_relative_l2_max <= 1
        or not 0 < envelope_relative_l2_max <= 1
        or not 0 < b_sign_mismatch_max <= 1
    ):
        raise ValueError("adapter parity tolerances are outside their valid ranges")
    states = {
        "reference_initial": reference_initial,
        "reference_final": reference_final,
        "replicated_initial": replicated_initial,
        "replicated_final": replicated_final,
    }
    reference_keys = set(reference_initial)
    for label, state in states.items():
        if set(state) != reference_keys:
            missing = sorted(reference_keys - set(state))
            extra = sorted(set(state) - reference_keys)
            raise RuntimeError(f"adapter tensor schemas differ for {label}: missing={missing[:3]}, extra={extra[:3]}")

    total_values = 0
    final_total_abs = 0.0
    final_max_abs = -1.0
    final_max_name = None
    initial_total_abs = 0.0
    initial_max_abs = -1.0
    initial_max_name = None
    delta_total_abs = 0.0
    delta_max_abs = -1.0
    delta_max_name = None
    reference_delta_squared_norm = 0.0
    replicated_delta_squared_norm = 0.0
    delta_error_squared_norm = 0.0
    delta_dot_product = 0.0
    lora_b_value_count = 0
    lora_b_sign_mismatch_count = 0

    for name in sorted(reference_keys):
        tensors = {label: state[name] for label, state in states.items()}
        metadata = {(tuple(tensor.shape), tensor.dtype) for tensor in tensors.values()}
        if len(metadata) != 1:
            details = ", ".join(f"{label}={tuple(tensor.shape)}/{tensor.dtype}" for label, tensor in tensors.items())
            raise RuntimeError(f"adapter tensor metadata differs for {name}: {details}")

        reference_initial_tensor = tensors["reference_initial"].float()
        reference_final_tensor = tensors["reference_final"].float()
        replicated_initial_tensor = tensors["replicated_initial"].float()
        replicated_final_tensor = tensors["replicated_final"].float()
        initial_difference = (reference_initial_tensor - replicated_initial_tensor).abs()
        final_difference = (reference_final_tensor - replicated_final_tensor).abs()
        reference_delta = reference_final_tensor - reference_initial_tensor
        replicated_delta = replicated_final_tensor - replicated_initial_tensor
        delta_error = reference_delta - replicated_delta
        delta_difference = delta_error.abs()

        local_initial_max = float(initial_difference.max().item()) if initial_difference.numel() else 0.0
        local_final_max = float(final_difference.max().item()) if final_difference.numel() else 0.0
        local_delta_max = float(delta_difference.max().item()) if delta_difference.numel() else 0.0
        if local_initial_max > initial_max_abs:
            initial_max_abs = local_initial_max
            initial_max_name = name
        if local_final_max > final_max_abs:
            final_max_abs = local_final_max
            final_max_name = name
        if local_delta_max > delta_max_abs:
            delta_max_abs = local_delta_max
            delta_max_name = name
        total_values += int(final_difference.numel())
        initial_total_abs += float(initial_difference.sum().item())
        final_total_abs += float(final_difference.sum().item())
        delta_total_abs += float(delta_difference.sum().item())
        reference_delta64 = reference_delta.double()
        replicated_delta64 = replicated_delta.double()
        delta_error64 = delta_error.double()
        reference_delta_squared_norm += float(reference_delta64.square().sum().item())
        replicated_delta_squared_norm += float(replicated_delta64.square().sum().item())
        delta_error_squared_norm += float(delta_error64.square().sum().item())
        delta_dot_product += float((reference_delta64 * replicated_delta64).sum().item())
        if "lora_B" in name:
            lora_b_value_count += int(reference_delta64.numel())
            lora_b_sign_mismatch_count += int(
                (torch.sign(reference_delta64) != torch.sign(replicated_delta64)).sum().item()
            )

    if total_values < 1:
        raise RuntimeError("adapter comparison found no tensors")
    reference_delta_l2 = math.sqrt(reference_delta_squared_norm)
    replicated_delta_l2 = math.sqrt(replicated_delta_squared_norm)
    delta_error_l2 = math.sqrt(delta_error_squared_norm)
    update_scale_l2 = max(reference_delta_l2, replicated_delta_l2)
    cosine = None
    relative_l2_error = None
    if update_scale_l2 > 0:
        relative_l2_error = delta_error_l2 / update_scale_l2
    if reference_delta_l2 > 0 and replicated_delta_l2 > 0:
        cosine = delta_dot_product / (reference_delta_l2 * replicated_delta_l2)
        # A finite-precision dot product can exceed the mathematical interval
        # by a few ulps; clip only for reporting/comparison, never to turn an
        # invalid non-finite result into a pass.
        cosine = max(-1.0, min(1.0, cosine))
    initial_state_exact = initial_max_abs == 0.0
    final_absolute_passed = final_max_abs <= atol
    update_nonzero = reference_delta_l2 > 0 and replicated_delta_l2 > 0
    delta_cosine_diagnostic_passed = cosine is not None and cosine >= delta_cosine_min
    diagnostic_delta_relative_l2_passed = relative_l2_error is not None and relative_l2_error <= delta_relative_l2_max
    diagnostic_delta_max_abs_passed = delta_max_abs <= delta_max_abs_atol
    envelope_relative_l2_diagnostic_passed = (
        relative_l2_error is not None and relative_l2_error <= envelope_relative_l2_max
    )
    lora_b_sign_mismatch_fraction = lora_b_sign_mismatch_count / lora_b_value_count if lora_b_value_count > 0 else None
    lora_b_sign_mismatch_diagnostic_passed = (
        lora_b_sign_mismatch_fraction is not None and lora_b_sign_mismatch_fraction <= b_sign_mismatch_max
    )
    update_norms_finite = math.isfinite(reference_delta_l2) and math.isfinite(replicated_delta_l2)
    update_norm_sanity_passed = update_norms_finite and update_nonzero
    return {
        "tensor_count": len(reference_keys),
        "value_count": total_values,
        "initial_max_abs_difference": initial_max_abs,
        "initial_max_abs_tensor": initial_max_name,
        "initial_mean_abs_difference": initial_total_abs / total_values,
        "initial_state_exact": initial_state_exact,
        "max_abs_difference": final_max_abs,
        "max_abs_tensor": final_max_name,
        "mean_abs_difference": final_total_abs / total_values,
        "atol": atol,
        "final_absolute_passed": final_absolute_passed,
        "reference_delta_l2_norm": reference_delta_l2,
        "replicated_delta_l2_norm": replicated_delta_l2,
        "delta_error_l2_norm": delta_error_l2,
        "delta_relative_l2_error": relative_l2_error,
        "delta_relative_l2_diagnostic_max": delta_relative_l2_max,
        "delta_relative_l2_diagnostic_passed": diagnostic_delta_relative_l2_passed,
        # Compatibility aliases are retained in receipts, but their
        # diagnostic-only status is explicit and they are not in ``passed``.
        "delta_relative_l2_max": delta_relative_l2_max,
        "delta_relative_l2_passed": diagnostic_delta_relative_l2_passed,
        "delta_cosine_similarity": cosine,
        "delta_cosine_min": delta_cosine_min,
        "delta_cosine_diagnostic_passed": delta_cosine_diagnostic_passed,
        "delta_cosine_passed": delta_cosine_diagnostic_passed,
        "delta_max_abs_error": delta_max_abs,
        "delta_max_abs_tensor": delta_max_name,
        "delta_max_abs_diagnostic_atol": delta_max_abs_atol,
        "delta_max_abs_diagnostic_passed": diagnostic_delta_max_abs_passed,
        "delta_max_abs_atol": delta_max_abs_atol,
        "delta_max_abs_passed": diagnostic_delta_max_abs_passed,
        "delta_mean_abs_error": delta_total_abs / total_values,
        "update_nonzero": update_nonzero,
        "update_norms_finite": update_norms_finite,
        "update_norm_sanity_passed": update_norm_sanity_passed,
        "envelope_relative_l2_max": envelope_relative_l2_max,
        "envelope_relative_l2_diagnostic_passed": envelope_relative_l2_diagnostic_passed,
        "envelope_relative_l2_passed": envelope_relative_l2_diagnostic_passed,
        "lora_b_value_count": lora_b_value_count,
        "lora_b_sign_mismatch_count": lora_b_sign_mismatch_count,
        "lora_b_sign_mismatch_fraction": lora_b_sign_mismatch_fraction,
        "lora_b_sign_mismatch_max": b_sign_mismatch_max,
        "lora_b_sign_mismatch_diagnostic_passed": lora_b_sign_mismatch_diagnostic_passed,
        "lora_b_sign_mismatch_passed": lora_b_sign_mismatch_diagnostic_passed,
        "post_adam_diagnostics_are_non_gating": {
            "delta_cosine": True,
            "delta_relative_l2": True,
            "delta_max_abs": True,
            "envelope_relative_l2": True,
            "lora_b_sign_mismatch": True,
        },
        "passed": (initial_state_exact and update_norm_sanity_passed and final_absolute_passed),
    }


def _compare_adapter_weights(
    *,
    reference_initial: Path,
    reference_final: Path,
    replicated_initial: Path,
    replicated_final: Path,
    atol: float,
    delta_cosine_min: float,
    delta_relative_l2_max: float,
    delta_max_abs_atol: float,
    envelope_relative_l2_max: float,
    b_sign_mismatch_max: float,
) -> dict[str, Any]:
    """Load four immutable adapter snapshots and return the full drift receipt.

    The caller applies the gate only after adding this complete report to the
    run result.  That guarantees a failed envelope still leaves all raw
    numerical evidence available in ``failure.json``.
    """

    from safetensors.torch import load_file

    report = _adapter_update_delta_parity(
        reference_initial=load_file(str(reference_initial), device="cpu"),
        reference_final=load_file(str(reference_final), device="cpu"),
        replicated_initial=load_file(str(replicated_initial), device="cpu"),
        replicated_final=load_file(str(replicated_final), device="cpu"),
        atol=atol,
        delta_cosine_min=delta_cosine_min,
        delta_relative_l2_max=delta_relative_l2_max,
        delta_max_abs_atol=delta_max_abs_atol,
        envelope_relative_l2_max=envelope_relative_l2_max,
        b_sign_mismatch_max=b_sign_mismatch_max,
    )
    report.update(
        {
            "reference_initial_adapter": str(reference_initial),
            "reference_final_adapter": str(reference_final),
            "replicated_initial_adapter": str(replicated_initial),
            "replicated_final_adapter": str(replicated_final),
        }
    )
    return report


def _require_adapter_update_delta_parity(report: Mapping[str, Any]) -> None:
    """Raise only after the complete post-Adam adapter receipt is persisted."""

    if bool(report.get("passed")):
        return
    raise RuntimeError(
        "single-rank and replicated fixed-update adapter states/updates failed hard post-Adam safety checks: "
        f"final_max_abs={report['max_abs_difference']:.6g} (max={report['atol']:.6g}), "
        f"initial_max_abs={report['initial_max_abs_difference']:.6g}, "
        f"reference_update_l2={report['reference_delta_l2_norm']:.6g}, "
        f"replicated_update_l2={report['replicated_delta_l2_norm']:.6g}"
    )


async def _worker_scores(
    *,
    backend: Any,
    rank_zero: Any,
    prompts: Sequence[Any],
    completions: Sequence[Sequence[int]],
) -> dict[str, list[list[float]]]:
    policy_handle = backend.policy_sampler("phase_shared_preflight")
    base_handle = backend.base_sampler()
    # A LocalBackend owns one live HF model, and a RolloutWorkerPool owns one
    # serialized command stream per worker.  Concurrent score calls here would
    # test accidental concurrent access rather than policy parity, so preserve
    # a deterministic request order.
    worker_policy = await policy_handle.score_completions(prompts, completions)
    worker_base = await base_handle.score_completions(prompts, completions)
    hf_policy = await asyncio.to_thread(rank_zero._score_completions, prompts, completions, use_base=False)
    hf_base = await asyncio.to_thread(rank_zero._score_completions, prompts, completions, use_base=True)
    for label, rows in {
        "worker_policy": worker_policy,
        "worker_base": worker_base,
        "hf_policy": hf_policy,
        "hf_base": hf_base,
    }.items():
        _flatten_rows(rows, label=label)
    return {
        "worker_policy": worker_policy,
        "worker_base": worker_base,
        "hf_policy": hf_policy,
        "hf_base": hf_base,
    }


async def _worker_coexistence_probe(
    *,
    backend: Any,
    prompt: Any,
    max_tokens: int,
) -> dict[str, Any]:
    """Use an awake worker while all HF replicas remain resident."""

    started = time.monotonic()
    samples = await backend.policy_sampler("phase_shared_coexistence").sample(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        stop=[],
        num_samples=1,
    )
    elapsed = time.monotonic() - started
    if len(samples) != 1:
        raise RuntimeError(f"awake-worker coexistence probe returned {len(samples)} samples, expected one")
    sequence = samples[0]
    token_digest = hashlib.sha256(
        json.dumps([int(token) for token in sequence.tokens], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    return {
        "seconds": elapsed,
        "returned_completion_tokens": len(sequence.tokens),
        "completion_token_sha256": token_digest,
    }


def _update_aligned_fixed_token_probe_receipt(
    probe_contract: Mapping[str, Any],
    *,
    anchor_candidate_index: int | None,
    prompts: Sequence[Any],
    completions: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Bind scored input IDs by hash while retaining no raw score values."""

    if len(prompts) != len(completions):
        raise RuntimeError("fixed-token probe prompt/completion row count differs")
    prompt_token_ids = [prompt.to_ints() for prompt in prompts]
    completion_token_ids = [[int(token) for token in row] for row in completions]
    score_inputs = {
        "schema": "phase-shared-post-update-worker-score-inputs-v2",
        "probe_contract": dict(probe_contract),
        "anchor_candidate_index": anchor_candidate_index,
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
    }
    return {
        **dict(probe_contract),
        "anchor_candidate_index": anchor_candidate_index,
        "score_rows": len(prompts),
        "prompt_token_count": sum(len(row) for row in prompt_token_ids),
        "completion_token_count": sum(len(row) for row in completion_token_ids),
        "score_inputs_sha256": hashlib.sha256(
            json.dumps(score_inputs, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _attestation_fixed_token_probe(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Project the detailed receipt onto the existing strict attestation ABI."""

    return {
        "kind": "post-update-worker-score-completions-v1",
        "score_rows": int(receipt["score_rows"]),
        "completion_token_count": int(receipt["completion_token_count"]),
        "score_inputs_sha256": str(receipt["score_inputs_sha256"]),
    }


def _all_worker_update_probe_score_grid(
    candidates: Sequence[Any],
    candidate_completions: Sequence[Sequence[int]],
    *,
    worker_count: int,
) -> tuple[list[Any], list[list[int]], list[dict[str, int]]]:
    """Repeat every fixed candidate once on every worker in round-robin order.

    ``RolloutWorkerPool.score_completions`` assigns score row ``i`` to worker
    ``i % worker_count``.  Candidate-major rows therefore give every worker
    each candidate exactly once: ``candidate 0 × all workers``, then
    ``candidate 1 × all workers``, and so on.  This is deliberately stronger
    than duplicating a single maximum-effect anchor and remains independent of
    a particular 2/4/8 GPU layout.
    """

    if len(candidates) != len(candidate_completions) or not candidates:
        raise RuntimeError("post-update worker effect requires aligned non-empty probe candidates")
    if worker_count < 1:
        raise RuntimeError("post-update worker effect requires at least one rollout worker")
    prompts: list[Any] = []
    completions: list[list[int]] = []
    assignments: list[dict[str, int]] = []
    for candidate_index, (candidate, completion) in enumerate(zip(candidates, candidate_completions, strict=True)):
        for worker_index in range(worker_count):
            score_row = len(prompts)
            prompts.append(candidate)
            completions.append([int(token) for token in completion])
            if score_row % worker_count != worker_index:
                raise RuntimeError("candidate-major worker probe grid no longer agrees with round-robin routing")
            assignments.append(
                {
                    "score_row": score_row,
                    "candidate_index": candidate_index,
                    "worker_index": worker_index,
                }
            )
    return prompts, completions, assignments


async def _post_update_worker_effect(
    *,
    backend: Any,
    rank_zero: Any,
    candidates: Sequence[Any],
    candidate_completions: Sequence[Sequence[int]],
    fixed_token_probe_contract: Mapping[str, Any],
    worker_count: int,
    worker_gpus: Sequence[Any],
    min_effect: float,
    cosine_min: float,
    norm_ratio_min: float,
    norm_ratio_max: float,
    relative_l2_max: float,
    diagnostic_sink: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Choose an aligned HF anchor, retain diagnostics, then gate every worker.

    The diagnostic sink is called before every effect gate.  This lets the
    outer preflight persist aggregate and per-worker summaries into
    ``failure.json`` even when one strict parity condition rejects the run.
    Raw worker/HF logprobs are deliberately never included in the report.
    """

    if len(candidates) != len(candidate_completions) or not candidates:
        raise RuntimeError("post-update worker effect requires aligned non-empty probe candidates")
    if worker_count != len(worker_gpus) or worker_count < 1:
        raise RuntimeError("post-update worker effect requires one rollout GPU per score row")

    def publish_diagnostic(report: dict[str, Any]) -> None:
        if diagnostic_sink is not None:
            diagnostic_sink(report)

    hf_policy = await asyncio.to_thread(rank_zero._score_completions, candidates, candidate_completions, use_base=False)
    hf_base = await asyncio.to_thread(rank_zero._score_completions, candidates, candidate_completions, use_base=True)
    candidate_effects: list[dict[str, Any]] = []
    for index in range(len(candidates)):
        summary = _effect_summary(
            worker_policy=[hf_policy[index]],
            worker_base=[hf_base[index]],
            hf_policy=[hf_policy[index]],
            hf_base=[hf_base[index]],
        )
        candidate_effects.append({"candidate_index": index, **summary})
    anchor_index = max(range(len(candidate_effects)), key=lambda index: candidate_effects[index]["hf_effect_max_abs"])
    anchor = candidate_effects[anchor_index]
    anchor_measurable = float(anchor["hf_effect_max_abs"]) >= min_effect
    report: dict[str, Any] = {
        "status": "anchor_selected",
        "passed": False,
        "anchor_candidate_index": anchor_index,
        "candidate_effects": candidate_effects,
        "anchor_measurable_gate": {
            "adapter_effect_min": min_effect,
            "hf_effect_max_abs": float(anchor["hf_effect_max_abs"]),
            "passed": anchor_measurable,
        },
        "aggregate": None,
        "aggregate_gate": None,
        "per_worker": [],
        "formal_aggregate_effect_parity": None,
        "formal_per_worker_effect_parity": [],
        "fixed_token_probe": None,
        "update_aligned_fixed_token_probe": _update_aligned_fixed_token_probe_receipt(
            fixed_token_probe_contract,
            anchor_candidate_index=anchor_index,
            prompts=(),
            completions=(),
        ),
    }
    publish_diagnostic(report)
    if not anchor_measurable:
        raise RuntimeError(
            f"real fixed update did not create a usable LoRA-effect probe across candidate completions: max_abs={float(anchor['hf_effect_max_abs']):.3e}, required>={min_effect:.3e}"
        )
    prompts, completions, score_grid = _all_worker_update_probe_score_grid(
        candidates,
        candidate_completions,
        worker_count=worker_count,
    )
    scores = await _worker_scores(
        backend=backend,
        rank_zero=rank_zero,
        prompts=prompts,
        completions=completions,
    )
    aggregate = _effect_summary(**scores)
    formal_aggregate = _formal_worker_effect_parity(**scores)
    per_worker: list[dict[str, Any]] = []
    formal_per_worker: list[dict[str, Any]] = []
    for worker_index in range(worker_count):
        worker_rows = [row for row in score_grid if row["worker_index"] == worker_index]
        score_row_indices = [row["score_row"] for row in worker_rows]
        candidate_indices = [row["candidate_index"] for row in worker_rows]
        if candidate_indices != list(range(len(candidates))):
            raise RuntimeError(
                f"worker {worker_index} did not receive every update-aligned probe candidate: {candidate_indices}"
            )
        summary = _effect_summary(
            worker_policy=[scores["worker_policy"][index] for index in score_row_indices],
            worker_base=[scores["worker_base"][index] for index in score_row_indices],
            hf_policy=[scores["hf_policy"][index] for index in score_row_indices],
            hf_base=[scores["hf_base"][index] for index in score_row_indices],
        )
        gpu = worker_gpus[worker_index]
        per_worker.append(
            {
                "worker_index": worker_index,
                "logical_gpu": int(gpu.logical_index),
                "device_token": str(gpu.device_token),
                "probe_rows": score_row_indices,
                "candidate_indices": candidate_indices,
                **summary,
                "gate": _effect_gate_summary(
                    summary,
                    min_effect=min_effect,
                    cosine_min=cosine_min,
                    norm_ratio_min=norm_ratio_min,
                    norm_ratio_max=norm_ratio_max,
                    relative_l2_max=relative_l2_max,
                ),
            }
        )
        formal_per_worker.append(
            {
                "worker_index": worker_index,
                "worker_gpu": gpu.as_dict(),
                "effect_parity": _formal_worker_effect_parity(
                    worker_policy=[scores["worker_policy"][index] for index in score_row_indices],
                    worker_base=[scores["worker_base"][index] for index in score_row_indices],
                    hf_policy=[scores["hf_policy"][index] for index in score_row_indices],
                    hf_base=[scores["hf_base"][index] for index in score_row_indices],
                ),
            }
        )
    update_aligned_probe_receipt = _update_aligned_fixed_token_probe_receipt(
        fixed_token_probe_contract,
        anchor_candidate_index=anchor_index,
        prompts=prompts,
        completions=completions,
    )
    report.update(
        {
            "status": "scored",
            "aggregate": aggregate,
            "aggregate_gate": _effect_gate_summary(
                aggregate,
                min_effect=min_effect,
                cosine_min=cosine_min,
                norm_ratio_min=norm_ratio_min,
                norm_ratio_max=norm_ratio_max,
                relative_l2_max=relative_l2_max,
            ),
            "per_worker": per_worker,
            "formal_aggregate_effect_parity": formal_aggregate,
            "formal_per_worker_effect_parity": formal_per_worker,
            # This narrow object is intentionally frozen to the existing
            # worker-parity-attestation ABI.  Keep rich recipe metadata in the
            # sibling update_aligned_fixed_token_probe receipt below.
            "fixed_token_probe": _attestation_fixed_token_probe(update_aligned_probe_receipt),
            "update_aligned_fixed_token_probe": update_aligned_probe_receipt,
            "score_grid": {
                "assignment_policy": "candidate_major_round_robin_all_workers-v1",
                "score_row_count": len(score_grid),
                "candidate_count": len(candidates),
                "rows_per_worker": len(candidates),
                "all_workers_receive_all_candidates": True,
            },
        }
    )
    # Persist every summary before any strict aggregate or per-worker gate can
    # raise.  This is intentionally after all worker rows are calculated so a
    # failed aggregate cannot hide disagreement on a particular GPU.
    publish_diagnostic(report)
    _require_effect(
        aggregate,
        min_effect=min_effect,
        cosine_min=cosine_min,
        norm_ratio_min=norm_ratio_min,
        norm_ratio_max=norm_ratio_max,
        relative_l2_max=relative_l2_max,
        label="aggregate post-update worker effect",
    )
    for worker in per_worker:
        _require_effect(
            worker,
            min_effect=min_effect,
            cosine_min=cosine_min,
            norm_ratio_min=norm_ratio_min,
            norm_ratio_max=norm_ratio_max,
            relative_l2_max=relative_l2_max,
            label=f"post-update worker {worker['worker_index']}",
        )
    report["status"] = "passed"
    report["passed"] = True
    publish_diagnostic(report)
    return report


async def _packing_sweep(
    *,
    args: argparse.Namespace,
    rank_zero: Any,
    tokenizer: Any,
    torch: Any,
    types: Any,
    datum_from_model_input_weights: Any,
    trainable_state_hash: Any,
    coordinator_device: str,
    diagnostic_sink: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Measure per-trainer packing while workers are asleep, without a parameter step.

    Changing the physical packing budget dynamically on peer processes would
    require a new replicated control command.  Capacity is a per-trainer
    memory property, so this probe intentionally runs rank zero only, clears
    gradients after each backward, and verifies that no trainable tensor
    changed.  The report labels this limitation explicitly instead of implying
    a distributed timing result it did not measure.
    """

    coordinator_index = int(coordinator_device.split(":", maxsplit=1)[1])
    model = rank_zero._require_model()
    original_budget = rank_zero.forward_microbatch_max_tokens
    original_datums = rank_zero.forward_microbatch_max_datums
    state_before = trainable_state_hash(rank_zero)
    results: list[dict[str, Any]] = []

    def publish_diagnostic() -> None:
        if diagnostic_sink is not None:
            diagnostic_sink(results)

    # Attach the mutable list before the first capacity attempt.  Every append
    # below then becomes visible to the outer failure handler even if a later
    # packing budget raises and this coroutine never returns normally.
    publish_diagnostic()
    try:
        for budget in args.packing_budgets:
            sequence_length = max(2, budget // args.packing_probe_datums)
            datums = _make_cross_entropy_datums(
                tokenizer=tokenizer,
                count=args.packing_probe_datums,
                sequence_length=sequence_length,
                torch=torch,
                types=types,
                datum_from_model_input_weights=datum_from_model_input_weights,
            )
            token_counts = [len(datum.model_input.to_ints()) for datum in datums]
            rank_zero.forward_microbatch_max_tokens = budget
            rank_zero.forward_microbatch_max_datums = args.packing_probe_datums
            model.zero_grad(set_to_none=True)
            rank_zero._gradient_accumulations = 0
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(coordinator_index)
            torch.cuda.synchronize(coordinator_index)
            started = time.monotonic()
            try:
                output = await _resolve_pending(await rank_zero.submit_forward_backward(datums, "cross_entropy"))
                torch.cuda.synchronize(coordinator_index)
                elapsed = time.monotonic() - started
                state_after = trainable_state_hash(rank_zero)
                if state_after != state_before:
                    raise RuntimeError(
                        "rank-zero packing probe changed trainable adapter state without an optimizer step"
                    )
                results.append(
                    {
                        "scope": "rank_zero_capacity_only_with_all_vllm_workers_asleep",
                        "packing_budget": budget,
                        "probe_datum_count": len(datums),
                        "probe_sequence_length": sequence_length,
                        "padded_token_slots": sequence_length * len(datums),
                        "forward_microbatch_count": len(rank_zero._forward_microbatches(token_counts)),
                        "forward_backward_seconds": elapsed,
                        "loss": float(output.metrics["loss"]),
                        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(coordinator_index)),
                        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(coordinator_index)),
                        "passed": True,
                    }
                )
                publish_diagnostic()
            except BaseException as exc:
                synchronization_failure = None
                try:
                    torch.cuda.synchronize(coordinator_index)
                except BaseException as sync_exc:  # noqa: BLE001 - retain the original packing failure
                    synchronization_failure = f"{type(sync_exc).__name__}: {sync_exc}"
                elapsed = time.monotonic() - started
                failure_record: dict[str, Any] = {
                    "scope": "rank_zero_capacity_only_with_all_vllm_workers_asleep",
                    "packing_budget": budget,
                    "probe_datum_count": len(datums),
                    "probe_sequence_length": sequence_length,
                    "padded_token_slots": sequence_length * len(datums),
                    "forward_microbatch_count": len(rank_zero._forward_microbatches(token_counts)),
                    "forward_backward_seconds": elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(coordinator_index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(coordinator_index)),
                    "passed": False,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
                if synchronization_failure is not None:
                    failure_record["synchronization_failure"] = synchronization_failure
                results.append(failure_record)
                publish_diagnostic()
                raise RuntimeError(f"packing probe failed at budget {budget}: {type(exc).__name__}: {exc}") from exc
            finally:
                model.zero_grad(set_to_none=True)
                rank_zero._gradient_accumulations = 0
                torch.cuda.empty_cache()
    finally:
        rank_zero.forward_microbatch_max_tokens = original_budget
        rank_zero.forward_microbatch_max_datums = original_datums
    return results


async def _run_preflight(args: argparse.Namespace, contract: ResolvedPreflight, output_dir: Path) -> dict[str, Any]:
    """Run the actual CUDA-only checks and return a serializable evidence record."""

    import torch
    from tinker import types
    from tinker_cookbook.supervised.common import datum_from_model_input_weights
    from transformers import AutoConfig, AutoTokenizer

    from ctm.backends.local.engine import LocalBackend
    from ctm.backends.local.qwen35_vllm_compat import (
        WORKER_PARITY_ATTESTATION_NAME,
        validate_qwen35_rollout_worker_parity_attestation,
        write_qwen35_rollout_worker_parity_attestation,
    )
    from ctm.backends.local.replicated import (
        LocalBackendConstructorSpec,
        ReplicatedTrainingBackend,
        trainable_state_hash,
    )
    from ctm.backends.local.rollout_workers import RolloutGPU, RolloutParallelBackend
    from ctm.core.config import AdamConfig, LoRAConfig

    if not torch.cuda.is_available():
        raise RuntimeError("phase-shared preflight requires CUDA")
    visible_count = len(contract.topology.visible_devices)
    if torch.cuda.device_count() < visible_count:
        raise RuntimeError(
            f"torch reports fewer CUDA devices than explicit CUDA_VISIBLE_DEVICES entries: torch={torch.cuda.device_count()}, visible={visible_count}"
        )
    coordinator_index = int(contract.coordinator_device.split(":", maxsplit=1)[1])
    torch.cuda.set_device(coordinator_index)

    output_dir.mkdir(parents=True, exist_ok=False)
    events = output_dir / "events.jsonl"
    _append_event(events, "hf_snapshot_prefetch_started", repo_id=args.model, revision=MODEL_REVISION)
    try:
        snapshot_readiness = _prepare_pinned_hf_snapshot(
            repo_id=args.model,
            revision=MODEL_REVISION,
            AutoConfig=AutoConfig,
            AutoTokenizer=AutoTokenizer,
        )
    except BaseException as exc:
        # Preserve a dedicated cache-readiness receipt even though the
        # detailed preflight result has not yet been created.  The outer main
        # handler also writes its normal minimal failure.json without
        # replacing this diagnostic artifact.
        _atomic_write_json(
            output_dir / "hf_snapshot_readiness_failure.json",
            {
                "schema": HF_SNAPSHOT_READINESS_SCHEMA,
                "status": "failed",
                "repo_id": args.model,
                "requested_revision": MODEL_REVISION,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        _append_event(events, "hf_snapshot_prefetch_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    snapshot_readiness_path = output_dir / "hf_snapshot_readiness.json"
    _atomic_write_json(snapshot_readiness_path, snapshot_readiness)
    runtime_model_identity = _runtime_model_identity(
        canonical_repo_id=args.model,
        snapshot_readiness=snapshot_readiness,
    )
    runtime_model = runtime_model_identity["runtime_model_argument"]
    _append_event(
        events,
        "hf_snapshot_prefetch_validated",
        repo_id=args.model,
        revision=MODEL_REVISION,
        resolved_commit=snapshot_readiness["resolved_commit"],
        indexed_shard_count=snapshot_readiness["safetensors_index"]["indexed_shard_count"],
    )
    contract_document = {
        "schema": SCHEMA,
        "kind": "non_production_phase_shared_preflight_contract",
        "provenance": _provenance(),
        "config": _contract_config(args),
        "model_snapshot_readiness": {
            "path": str(snapshot_readiness_path.resolve()),
            "sha256": _sha256_file(snapshot_readiness_path),
            "repo_id": snapshot_readiness["repo_id"],
            "pinned_revision": snapshot_readiness["requested_revision"],
            "resolved_commit": snapshot_readiness["resolved_commit"],
            "runtime_model_argument": runtime_model,
        },
        "runtime_model_identity": runtime_model_identity,
        "resolved": contract.as_dict(),
        "non_production": True,
        "production_output_touched": False,
    }
    _atomic_write_json(output_dir / "contract.json", contract_document)
    _append_event(events, "contract_written", world_size=contract.world_size)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "running",
        "provenance": contract_document["provenance"],
        "contract_path": str((output_dir / "contract.json").resolve()),
        "model_snapshot_readiness": {
            "path": str(snapshot_readiness_path.resolve()),
            "sha256": _sha256_file(snapshot_readiness_path),
            **runtime_model_identity,
        },
        "resolved": contract.as_dict(),
        "single_rank_reference": {
            "status": "not_started",
            "run_dir": str(output_dir / "single_rank_reference"),
        },
        "phase_shared": {},
        "passed": False,
    }

    reference_evidence = result["single_rank_reference"]

    def update_reference_evidence(receipt: dict[str, Any]) -> None:
        # Preserve this exact mutable location in ``result`` while the
        # reference creates adapters, raw gradients, and optimizer evidence.
        # If any operation fails, the pre-backend failure handler below has a
        # complete partial receipt instead of falling back to ``early_failure``.
        reference_evidence.clear()
        reference_evidence.update(receipt)

    backend: Any | None = None
    try:
        # The readiness gate already exercised this exact local directory.
        # Repeat only the cheap local tokenizer construction for the actual
        # synthetic datum recipe; neither this call nor any later backend can
        # mutate/populate the Hugging Face cache.
        tokenizer = AutoTokenizer.from_pretrained(runtime_model, local_files_only=True)
        parity_datums = _make_cross_entropy_datums(
            tokenizer=tokenizer,
            count=args.parity_datums_per_rank * contract.world_size,
            sequence_length=args.parity_sequence_tokens,
            torch=torch,
            types=types,
            datum_from_model_input_weights=datum_from_model_input_weights,
        )
        _append_event(events, "fixed_datums_materialized", count=len(parity_datums))
        reference_parity_datums, reference_shard_original_indices = _parity_reference_datums_in_lpt_shard_order(
            parity_datums,
            contract=contract,
        )
        _append_event(
            events,
            "fixed_datums_lpt_ordered_for_reference",
            shard_original_indices=[list(indices) for indices in reference_shard_original_indices],
        )

        # The reference must finish and leave CUDA before vLLM starts.  Otherwise a
        # later parity pass could accidentally compare against a contended model.
        _append_event(events, "single_rank_reference_started")
        single_reference = await _single_rank_reference_update(
            args=args,
            contract=contract,
            datums=reference_parity_datums,
            planned_original_indices=reference_shard_original_indices,
            output_dir=output_dir,
            torch=torch,
            LocalBackend=LocalBackend,
            LoRAConfig=LoRAConfig,
            AdamConfig=AdamConfig,
            trainable_state_hash=trainable_state_hash,
            runtime_model=runtime_model,
            diagnostic_sink=update_reference_evidence,
        )
        result["single_rank_reference"] = single_reference
        _append_event(events, "single_rank_reference_completed", loss=single_reference["loss"])

        training_kwargs = {
            "dtype": torch.bfloat16,
            "use_lora": True,
            "sampler": "vllm",
            # The outer worker pool owns actual vLLM engines.  Keep the rank-zero
            # in-process sampler cold, while giving each trainer the same portable
            # options it would receive in a production phase-shared run.
            "vllm_options": _rank_zero_vllm_options(args),
            "gradient_checkpointing": True,
            "gradient_checkpointing_layers": contract.checkpoint_layers,
            "forward_microbatch_max_datums": args.forward_microbatch_max_datums,
            "forward_microbatch_max_tokens": min(args.packing_budgets),
            "target_logprob_chunk_size": args.target_logprob_chunk_size,
        }
        torch.manual_seed(args.lora_seed)
        rank_zero = LocalBackend(device=contract.coordinator_device, **training_kwargs)
        replicated = ReplicatedTrainingBackend(
            rank_zero,
            topology=contract.topology,
            child_backend_spec=LocalBackendConstructorSpec(LocalBackend, kwargs=training_kwargs),
            start_timeout_seconds=args.start_timeout_seconds,
            command_timeout_seconds=args.request_timeout_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
        )
        worker_gpus = tuple(RolloutGPU(gpu.logical_index, gpu.device_token) for gpu in contract.topology.rollout_gpus)
        worker_options = _worker_vllm_options(args)
        backend = RolloutParallelBackend(
            replicated,
            gpus=worker_gpus,
            status_dir=output_dir / "rollout_workers",
            worker_vllm_options=worker_options,
            start_timeout_seconds=args.start_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
            shutdown_timeout_seconds=args.shutdown_timeout_seconds,
            # This is the deliberately non-production path that constructs an
            # isolated compatibility snapshot before the full existing attestation
            # pipeline is applied to any production run.
            qwen35_rollout_parity_bootstrap=True,
        )
    except BaseException as exc:
        result["status"] = "failed"
        result["passed"] = False
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_write_json(output_dir / "failure.json", result)
        _append_event(events, "preflight_failed", error_type=type(exc).__name__, error=str(exc))
        raise

    if backend is None:
        exc = RuntimeError("phase-shared backend construction returned no backend")
        result["status"] = "failed"
        result["passed"] = False
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_write_json(output_dir / "failure.json", result)
        _append_event(events, "preflight_failed", error_type=type(exc).__name__, error=str(exc))
        raise exc
    try:
        _append_event(events, "phase_shared_setup_started", worker_count=len(worker_gpus))
        setup_started = time.monotonic()
        backend.setup(
            model=runtime_model,
            lora=LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=0.0,
                seed=args.lora_seed,
                train_attn=True,
                train_mlp=True,
                train_unembed=False,
            ),
        )
        # ReplicatedTrainingBackend has now installed its short-timeout
        # optimizer-boundary SUM reducer on rank zero.  Wrap that exact
        # reducer before any forward/backward work; peer reducers and the
        # distributed orchestration are intentionally untouched.
        replicated_gradient_capture = _install_rank_zero_gradient_capture(
            rank_zero,
            output_path=output_dir / "replicated_rank_zero_pre_optimizer_gradients.safetensors",
            diagnostic_sink=lambda receipt: result["phase_shared"].__setitem__(
                "replicated_rank_zero_pre_optimizer_gradient_capture", receipt
            ),
        )
        # Attach the mutable capture state before any F/B.  Its wrapper
        # publishes the immutable safetensors path/checksum at creation time,
        # so a peer optimizer or state-hash failure cannot lose rank zero's
        # successful post-SUM evidence from ``failure.json``.
        result["phase_shared"]["replicated_rank_zero_pre_optimizer_gradient_capture"] = replicated_gradient_capture
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["setup_seconds"] = time.monotonic() - setup_started
        result["phase_shared"]["awake_with_hf_trainers_memory"] = _gpu_snapshot(
            torch,
            coordinator_device=contract.coordinator_device,
        )
        if backend.pool is None:
            raise RuntimeError("phase-shared backend setup completed without a rollout worker pool")
        initial_adapter_version = backend.pool.adapter_version
        if initial_adapter_version < 1:
            raise RuntimeError(
                f"phase-shared worker pool started without a positive initial adapter version; got {initial_adapter_version}"
            )
        initial_adapter_root = backend.pool.status_dir / "adapters" / f"v{initial_adapter_version:08d}"
        initial_raw_adapter = initial_adapter_root / "raw" / "adapter_model.safetensors"
        initial_compat_adapter = initial_adapter_root / "vllm_compat" / "adapter_model.safetensors"
        if not initial_raw_adapter.is_file() or not initial_compat_adapter.is_file():
            raise RuntimeError("initial Qwen3.5 raw or translated vLLM adapter snapshot is missing")
        result["phase_shared"]["initial_adapter"] = {
            "version": initial_adapter_version,
            "raw_adapter": str(initial_raw_adapter),
            "raw_adapter_sha256": _sha256_file(initial_raw_adapter),
            "vllm_compat_adapter": str(initial_compat_adapter),
            "vllm_compat_adapter_sha256": _sha256_file(initial_compat_adapter),
        }
        initial_health = await backend.pool.health()
        if any(response.get("sleeping") is not False for response in initial_health):
            raise RuntimeError("vLLM workers did not start awake alongside HF trainer replicas")
        result["phase_shared"]["initial_pool_health"] = initial_health
        result["phase_shared"]["replica_state_hash_after_setup"] = trainable_state_hash(rank_zero)
        _append_event(events, "phase_shared_setup_completed", setup_seconds=result["phase_shared"]["setup_seconds"])

        candidates, candidate_completions, fixed_token_probe_contract = _make_update_aligned_probe_candidates(
            tokenizer,
            types,
            count=args.effect_candidate_count,
            sequence_length=args.parity_sequence_tokens,
            completion_tokens=args.effect_probe_completion_tokens,
        )
        result["phase_shared"]["update_aligned_fixed_token_probe_contract"] = fixed_token_probe_contract
        _append_event(
            events,
            "update_aligned_fixed_token_probe_materialized",
            candidate_count=args.effect_candidate_count,
            sequence_token_count=args.parity_sequence_tokens,
            completion_token_count=args.effect_probe_completion_tokens,
            cross_entropy_seed_token_sha256=fixed_token_probe_contract["cross_entropy_seed_token_sha256"],
        )
        worker_count = len(worker_gpus)
        repeated_prompt = [candidates[0] for _ in range(worker_count)]
        repeated_completion = [candidate_completions[0] for _ in range(worker_count)]
        coexistence = await _worker_coexistence_probe(
            backend=backend,
            prompt=candidates[0],
            max_tokens=args.rollout_probe_max_tokens,
        )
        awake_scores = await _worker_scores(
            backend=backend,
            rank_zero=rank_zero,
            prompts=repeated_prompt,
            completions=repeated_completion,
        )
        result["phase_shared"]["awake_worker_hf_trainer_coexistence"] = {
            "sample": coexistence,
            "worker_score_rows": len(awake_scores["worker_policy"]),
            "worker_count": worker_count,
        }
        _append_event(events, "awake_coexistence_probe_completed", seconds=coexistence["seconds"])

        sleep_started = time.monotonic()
        await backend.enter_training_phase()
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["initial_sleep_seconds"] = time.monotonic() - sleep_started
        result["phase_shared"]["asleep_memory"] = _gpu_snapshot(torch, coordinator_device=contract.coordinator_device)
        asleep_health = await backend.pool.health()
        if any(response.get("sleeping") is not True for response in asleep_health):
            raise RuntimeError("not every vLLM worker acknowledged the initial sleep barrier")
        result["phase_shared"]["asleep_pool_health"] = asleep_health
        _append_event(events, "initial_sleep_completed", seconds=result["phase_shared"]["initial_sleep_seconds"])

        wake_started = time.monotonic()
        await backend.enter_rollout_phase()
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["initial_wake_seconds"] = time.monotonic() - wake_started
        result["phase_shared"]["rewoken_memory"] = _gpu_snapshot(torch, coordinator_device=contract.coordinator_device)
        post_wake_scores = await _worker_scores(
            backend=backend,
            rank_zero=rank_zero,
            prompts=repeated_prompt,
            completions=repeated_completion,
        )
        worker_policy_drift = _max_abs_row_difference(
            awake_scores["worker_policy"],
            post_wake_scores["worker_policy"],
            label="worker policy scores before/after sleep-wake",
        )
        worker_base_drift = _max_abs_row_difference(
            awake_scores["worker_base"],
            post_wake_scores["worker_base"],
            label="worker base scores before/after sleep-wake",
        )
        if max(worker_policy_drift, worker_base_drift) > args.wake_score_max_abs:
            raise RuntimeError(
                f"vLLM sleep/wake changed fixed worker scores beyond tolerance: policy={worker_policy_drift:.3e}, base={worker_base_drift:.3e}, atol={args.wake_score_max_abs:.3e}"
            )
        result["phase_shared"]["sleep_wake_score_stability"] = {
            "policy_max_abs_difference": worker_policy_drift,
            "base_max_abs_difference": worker_base_drift,
            "atol": args.wake_score_max_abs,
            "passed": True,
        }
        _append_event(events, "initial_wake_completed", seconds=result["phase_shared"]["initial_wake_seconds"])

        train_sleep_started = time.monotonic()
        await backend.enter_training_phase()
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["update_sleep_seconds"] = time.monotonic() - train_sleep_started
        forward_started = time.monotonic()
        replicated_output = await _resolve_pending(
            await backend.submit_forward_backward(parity_datums, "cross_entropy")
        )
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["replicated_forward_backward_seconds"] = time.monotonic() - forward_started
        optimizer_started = time.monotonic()
        await _resolve_pending(
            await backend.submit_optim_step(
                learning_rate=args.learning_rate,
                adam=AdamConfig(learning_rate=args.learning_rate),
            )
        )
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["replicated_optimizer_seconds"] = time.monotonic() - optimizer_started
        result["phase_shared"]["replicated_loss"] = float(replicated_output.metrics["loss"])
        result["phase_shared"]["replica_state_hash_after_update"] = trainable_state_hash(rank_zero)
        if int(replicated_gradient_capture["capture_count"]) != 1:
            raise RuntimeError(
                "replicated fixed update did not invoke the rank-zero global-gradient capture exactly once; "
                f"got {replicated_gradient_capture['capture_count']}"
            )
        replicated_gradient_receipt = replicated_gradient_capture["receipt"]
        if not isinstance(replicated_gradient_receipt, Mapping):
            raise RuntimeError("replicated fixed update completed without a rank-zero pre-optimizer gradient receipt")
        gradient_parity = _compare_pre_optimizer_gradients(
            reference_path=Path(single_reference["pre_optimizer_gradient"]["path"]),
            replicated_path=Path(replicated_gradient_receipt["path"]),
        )
        gradient_parity["reference_capture"] = {
            "gradient_accumulations_before_optimizer": single_reference["gradient_accumulations_before_optimizer"],
            "pre_optimizer_gradient_summary": single_reference["pre_optimizer_gradient_summary"],
            **dict(single_reference["pre_optimizer_gradient"]),
        }
        gradient_parity["replicated_rank_zero_capture"] = {
            "capture_count": int(replicated_gradient_capture["capture_count"]),
            "gradient_accumulations_before_reduce": replicated_gradient_capture["gradient_accumulations_before_reduce"],
            "pre_sum_rank_zero_gradient_summary": replicated_gradient_capture["pre_sum_rank_zero_gradient_summary"],
            "wrapped_existing_reducer": bool(replicated_gradient_capture["wrapped_existing_reducer"]),
            "capture_after_original_global_sum": bool(replicated_gradient_capture["capture_after_original_global_sum"]),
            **dict(replicated_gradient_receipt),
        }
        gradient_parity["ordered_tensor_schema_match"] = (
            single_reference["pre_optimizer_gradient"]["ordered_tensor_names_sha256"]
            == replicated_gradient_receipt["ordered_tensor_names_sha256"]
            and single_reference["pre_optimizer_gradient"]["ordered_tensor_names"]
            == replicated_gradient_receipt["ordered_tensor_names"]
        )
        gradient_parity["gate"] = _gradient_parity_gate_summary(
            gradient_parity,
            cosine_min=args.gradient_parity_cosine_min,
            norm_ratio_min=args.gradient_parity_norm_ratio_min,
            norm_ratio_max=args.gradient_parity_norm_ratio_max,
            relative_l2_max=args.gradient_parity_relative_l2_max,
            magnitude_weighted_sign_mismatch_max=args.gradient_parity_magnitude_weighted_sign_mismatch_max,
        )
        gradient_parity["receipt_path"] = str(output_dir / "pre_optimizer_gradient_parity.json")
        # Keep a complete standalone diagnostic receipt even if the strict
        # gate below rejects the run before any adapter publication.
        _atomic_write_json(Path(gradient_parity["receipt_path"]), gradient_parity)
        result["phase_shared"]["fixed_update_pre_optimizer_gradient_parity"] = gradient_parity
        _append_event(
            events,
            "fixed_update_pre_optimizer_gradient_parity_completed",
            cosine=gradient_parity["cosine_similarity"],
            relative_l2=gradient_parity["relative_l2_error"],
            gate_passed=gradient_parity["gate"]["passed"],
        )
        _require_gradient_parity(gradient_parity)
        _append_event(
            events,
            "replicated_fixed_update_completed",
            loss=result["phase_shared"]["replicated_loss"],
            forward_backward_seconds=result["phase_shared"]["replicated_forward_backward_seconds"],
        )

        loss_parity = _fixed_update_loss_parity(
            single_rank_loss=float(single_reference["loss"]),
            replicated_loss=float(replicated_output.metrics["loss"]),
            atol=args.loss_parity_atol,
            rtol=args.loss_parity_rtol,
        )
        result["phase_shared"]["fixed_update_loss_parity"] = loss_parity
        if not loss_parity["candidate_passed"]:
            raise RuntimeError(
                f"single-rank and replicated fixed-update loss differs beyond absolute/relative tolerance: abs={loss_parity['abs_difference']:.6g}, relative={loss_parity['relative_difference']:.6g}, atol={args.loss_parity_atol:.6g}, rtol={args.loss_parity_rtol:.6g}"
            )

        publish_started = time.monotonic()
        await backend.refresh_policy_sampler("phase_shared_preflight_after_update")
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["publish_wake_seconds"] = time.monotonic() - publish_started
        result["phase_shared"]["post_publish_awake_memory"] = _gpu_snapshot(
            torch,
            coordinator_device=contract.coordinator_device,
        )
        post_publish_health = await backend.pool.health()
        if any(response.get("sleeping") is not False for response in post_publish_health):
            raise RuntimeError("adapter publication did not leave every vLLM worker awake")
        result["phase_shared"]["post_publish_pool_health"] = post_publish_health
        expected_updated_adapter_version = initial_adapter_version + 1
        if backend.pool.adapter_version != expected_updated_adapter_version:
            raise RuntimeError(
                f"expected exactly one updated adapter version after the fixed update; initial={initial_adapter_version}, current={backend.pool.adapter_version}"
            )
        adapter_root = backend.pool.status_dir / "adapters" / f"v{backend.pool.adapter_version:08d}"
        raw_adapter = adapter_root / "raw" / "adapter_model.safetensors"
        compat_adapter = adapter_root / "vllm_compat" / "adapter_model.safetensors"
        if not raw_adapter.is_file() or not compat_adapter.is_file():
            raise RuntimeError("post-update Qwen3.5 raw or translated vLLM adapter snapshot is missing")
        result["phase_shared"]["post_update_adapter"] = {
            "version": backend.pool.adapter_version,
            "raw_adapter": str(raw_adapter),
            "raw_adapter_sha256": _sha256_file(raw_adapter),
            "vllm_compat_adapter": str(compat_adapter),
            "vllm_compat_adapter_sha256": _sha256_file(compat_adapter),
        }
        _append_event(
            events, "adapter_published_and_workers_woken", seconds=result["phase_shared"]["publish_wake_seconds"]
        )

        parameter_parity = _compare_adapter_weights(
            reference_initial=Path(single_reference["initial_adapter_dir"]) / "adapter_model.safetensors",
            reference_final=Path(single_reference["adapter_dir"]) / "adapter_model.safetensors",
            replicated_initial=initial_raw_adapter,
            replicated_final=raw_adapter,
            atol=args.parameter_parity_atol,
            delta_cosine_min=args.parameter_parity_delta_cosine_min,
            delta_relative_l2_max=args.parameter_parity_delta_relative_l2_max,
            delta_max_abs_atol=args.parameter_parity_delta_max_abs,
            envelope_relative_l2_max=args.parameter_parity_envelope_relative_l2_max,
            b_sign_mismatch_max=args.parameter_parity_b_sign_mismatch_max,
        )
        parameter_parity["receipt_path"] = str(output_dir / "fixed_update_parameter_parity.json")
        _atomic_write_json(Path(parameter_parity["receipt_path"]), parameter_parity)
        result["phase_shared"]["fixed_update_parameter_parity"] = parameter_parity
        _require_adapter_update_delta_parity(parameter_parity)
        loss_parity["pre_optimizer_gradient_parity_passed"] = bool(gradient_parity["gate"]["passed"])
        loss_parity["adapter_safety_passed"] = bool(parameter_parity["passed"])
        loss_parity["adapter_state_parity_semantics"] = "hard_post_adam_safety_not_diagnostic_delta_equality"
        loss_parity["requires_pre_optimizer_gradient_parity"] = not loss_parity["absolute_passed"]
        loss_parity["requires_adapter_safety"] = not loss_parity["absolute_passed"]
        if not loss_parity["absolute_passed"]:
            # The relative branch is deliberately conditional on the earlier
            # raw pre-optimizer-gradient proof plus hard post-Adam adapter
            # safety.  This prevents a scalar BF16 loss tolerance from
            # masking an objective, sharding, or reducer error that changes
            # the update while allowing calibrated first-step AdamW diagnostics
            # to remain non-gating.
            if (
                not loss_parity["relative_passed"]
                or not loss_parity["pre_optimizer_gradient_parity_passed"]
                or not loss_parity["adapter_safety_passed"]
            ):
                raise RuntimeError(
                    "fixed-update loss passed neither the absolute loss gate nor the "
                    "relative-loss-plus-pre-optimizer-gradient-plus-adapter-safety gate"
                )
            loss_parity["passed"] = True
            loss_parity["acceptance_basis"] = "relative_loss_and_pre_optimizer_gradient_and_adapter_safety"
            loss_parity["adapter_state_parity_passed"] = bool(parameter_parity["passed"])
        else:
            loss_parity["adapter_state_parity_passed"] = bool(parameter_parity["passed"])
        _append_event(
            events,
            "fixed_update_loss_parity_completed",
            acceptance_basis=loss_parity["acceptance_basis"],
            abs_difference=loss_parity["abs_difference"],
            relative_difference=loss_parity["relative_difference"],
        )
        post_update_effect = await _post_update_worker_effect(
            backend=backend,
            rank_zero=rank_zero,
            candidates=candidates,
            candidate_completions=candidate_completions,
            fixed_token_probe_contract=fixed_token_probe_contract,
            worker_count=worker_count,
            worker_gpus=worker_gpus,
            min_effect=args.adapter_effect_min,
            cosine_min=args.effect_cosine_min,
            norm_ratio_min=args.effect_norm_ratio_min,
            norm_ratio_max=args.effect_norm_ratio_max,
            relative_l2_max=args.effect_relative_l2_max,
            diagnostic_sink=lambda report: result["phase_shared"].__setitem__("post_update_worker_lora_effect", report),
        )
        result["phase_shared"]["post_update_worker_lora_effect"] = post_update_effect
        worker_parity_attestation_path = output_dir / WORKER_PARITY_ATTESTATION_NAME
        worker_parity_attestation = write_qwen35_rollout_worker_parity_attestation(
            worker_parity_attestation_path,
            # The compatibility manifest and worker pool were built with the
            # immutable local snapshot path.  The attestation validator uses
            # exact string identity, so record that actual runtime argument
            # rather than falsely claiming the mutable repository alias.
            model=runtime_model,
            raw_adapter=adapter_root / "raw",
            vllm_adapter=adapter_root / "vllm_compat",
            adapter_version=backend.pool.adapter_version,
            worker_gpus=worker_gpus,
            worker_engine_kwargs=backend.pool.engine_kwargs,
            fixed_token_probe=post_update_effect["fixed_token_probe"],
            aggregate_effect_parity=post_update_effect["formal_aggregate_effect_parity"],
            per_worker_effect_parity=post_update_effect["formal_per_worker_effect_parity"],
        )
        validated_worker_parity_attestation = validate_qwen35_rollout_worker_parity_attestation(
            worker_parity_attestation_path,
            expected_model=runtime_model,
            expected_worker_gpus=worker_gpus,
            expected_worker_engine_kwargs=backend.pool.engine_kwargs,
        )
        if validated_worker_parity_attestation != worker_parity_attestation:
            raise RuntimeError("written Qwen3.5 worker parity attestation did not revalidate identically")
        result["phase_shared"]["post_update_worker_parity_attestation"] = {
            "path": str(worker_parity_attestation_path),
            "sha256": _sha256_file(worker_parity_attestation_path),
            "schema": worker_parity_attestation["schema"],
            "canonical_model_repo": args.model,
            "attestation_model": runtime_model,
            "attestation_model_semantics": "exact_local_runtime_model_argument",
            "attestation_portability": "runtime_path_bound_nonportable_topology_generic",
        }
        _append_event(events, "post_update_worker_effect_completed")

        pack_sleep_started = time.monotonic()
        await backend.enter_training_phase()
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["packing_sleep_seconds"] = time.monotonic() - pack_sleep_started
        result["phase_shared"]["packing_workers_asleep_memory"] = _gpu_snapshot(
            torch,
            coordinator_device=contract.coordinator_device,
        )
        # Install the mutable receipt before the first budget.  A later
        # out-of-memory or state-integrity failure must not discard completed
        # lower-budget measurements from ``failure.json``.
        packing_sweep_receipt: list[dict[str, Any]] = []
        result["phase_shared"]["rank_zero_packing_sweep"] = packing_sweep_receipt
        result["phase_shared"]["rank_zero_packing_sweep_status"] = "running"
        await _packing_sweep(
            args=args,
            rank_zero=rank_zero,
            tokenizer=tokenizer,
            torch=torch,
            types=types,
            datum_from_model_input_weights=datum_from_model_input_weights,
            trainable_state_hash=trainable_state_hash,
            coordinator_device=contract.coordinator_device,
            diagnostic_sink=lambda partial: result["phase_shared"].__setitem__("rank_zero_packing_sweep", partial),
        )
        result["phase_shared"]["rank_zero_packing_sweep_status"] = "completed"
        _append_event(events, "rank_zero_packing_sweep_completed")
        final_wake_started = time.monotonic()
        await backend.enter_rollout_phase()
        torch.cuda.synchronize(coordinator_index)
        result["phase_shared"]["final_wake_seconds"] = time.monotonic() - final_wake_started
        result["phase_shared"]["final_awake_memory"] = _gpu_snapshot(
            torch, coordinator_device=contract.coordinator_device
        )
        final_health = await backend.pool.health()
        if any(response.get("sleeping") is not False for response in final_health):
            raise RuntimeError("packing-sweep wake did not leave every worker awake")
        result["phase_shared"]["final_pool_health"] = final_health
        result["phase_shared"]["post_final_wake_worker_lora_effect"] = await _post_update_worker_effect(
            backend=backend,
            rank_zero=rank_zero,
            candidates=candidates,
            candidate_completions=candidate_completions,
            fixed_token_probe_contract=fixed_token_probe_contract,
            worker_count=worker_count,
            worker_gpus=worker_gpus,
            min_effect=args.adapter_effect_min,
            cosine_min=args.effect_cosine_min,
            norm_ratio_min=args.effect_norm_ratio_min,
            norm_ratio_max=args.effect_norm_ratio_max,
            relative_l2_max=args.effect_relative_l2_max,
            diagnostic_sink=lambda report: result["phase_shared"].__setitem__(
                "post_final_wake_worker_lora_effect", report
            ),
        )
        _append_event(events, "final_wake_completed", seconds=result["phase_shared"]["final_wake_seconds"])

        # Teardown is deliberately before terminal success publication.  A
        # shutdown/cleanup fault therefore reaches the failure handler; once
        # SUCCESS exists, no later operation in this function can turn the
        # run into an exception while leaving a misleading marker behind.
        backend.shutdown()
        backend = None
        gc.collect()
        torch.cuda.empty_cache()

        result["status"] = "passed"
        result["passed"] = True
        _atomic_write_json(output_dir / "result.json", result)
        # The event is fsync'd before the terminal marker.  SUCCESS is the
        # final durable action and is created exclusively, not overwritten.
        _append_event(events, "preflight_passed")
        _write_success_marker(output_dir / "SUCCESS")
        return result
    except BaseException as exc:
        result["status"] = "failed"
        result["passed"] = False
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        _atomic_write_json(output_dir / "failure.json", result)
        _append_event(events, "preflight_failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if backend is not None:
            backend.shutdown()
            gc.collect()
            torch.cuda.empty_cache()


def _dry_run_document(args: argparse.Namespace, contract: ResolvedPreflight) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "dry_run": True,
        "non_production": True,
        "model": args.model,
        "output_dir": str(args.output_dir.expanduser()),
        "resolved": contract.as_dict(),
        "checks": [
            "explicit CUDA_VISIBLE_DEVICES topology contract",
            "awake vLLM plus resident HF/PEFT trainer coexistence",
            "level-1 worker sleep/wake memory and score stability",
            "single-rank versus post-SUM replicated raw-gradient parity before AdamW",
            "single-rank versus replicated fixed-update loss and adapter parity",
            "per-worker Qwen3.5 translated-LoRA post-update effect parity",
            "rank-zero capacity sweep at 20480, 40960, and 49152 padded tokens with workers asleep",
        ],
    }


def _record_early_failure(output_dir: Path, exc: BaseException) -> None:
    """Leave a minimal failure receipt when setup failed before a result exists."""

    failure_path = output_dir / "failure.json"
    if not output_dir.is_dir() or failure_path.exists():
        return
    try:
        _atomic_write_json(
            failure_path,
            {
                "schema": SCHEMA,
                "status": "failed",
                "passed": False,
                "non_production": True,
                "failure": {"type": type(exc).__name__, "message": str(exc)},
                "note": "failure occurred before the detailed phase-shared result object was initialized",
            },
        )
    except OSError:
        # The original failure (for example a filesystem permission error) is
        # more useful than a secondary attempt to record it.
        return


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path | None = None
    preflight_started = False
    try:
        contract = resolve_preflight_contract(args)
        if args.dry_run:
            print(json.dumps(_dry_run_document(args, contract), indent=2, sort_keys=True))
            return 0
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing preflight evidence directory: {output_dir}")
        preflight_started = True
        result = asyncio.run(_run_preflight(args, contract, output_dir))
        print(
            "CTM_QWEN35_PHASE_SHARED_PREFLIGHT="
            + json.dumps(
                {
                    "output_dir": str(output_dir),
                    "passed": result["passed"],
                    "world_size": contract.world_size,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - preserve an actionable non-production failure code
        if preflight_started and output_dir is not None:
            _record_early_failure(output_dir, exc)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
