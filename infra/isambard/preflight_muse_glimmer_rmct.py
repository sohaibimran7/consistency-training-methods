#!/usr/bin/env python3
"""Four-GH200 Muse text-only, memory, LoRA parity, and no-cap preflight.

GPU 0 owns the exact HF/PEFT policy used by training. GPUs 1--3 own the
independent vLLM engines used for rollouts.  A deterministic non-zero rank-8
adapter is scored through both stacks on identical token IDs.  vLLM obtains
teacher-forced prompt logprobs while its otherwise-unused continuation is
``max_tokens=None`` and must finish through EOS; any length termination aborts
the preflight before an attestation can be written.

This is a preflight, not a scientific run.  Its disposable adapter never
becomes a training parent.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import transformers
from safetensors.torch import load_file
from tinker import types
from transformers import AutoTokenizer

from ctm.backends.local.engine import LocalBackend
from ctm.backends.local.muse_glimmer import (
    MODEL_ID,
    MODEL_REVISION,
    PARITY_ATTESTATION_NAME,
    PARITY_ATTESTATION_SCHEMA,
    TRANSFORMERS_VERSION,
    VLLM_COMMIT,
    VLLM_VERSION,
    VLLM_WHEEL_SHA256,
    VLLM_WHEEL_URL,
    file_sha256,
    is_muse_glimmer_model_name,
    validate_muse_rollout_worker_parity_attestation,
)
from ctm.backends.local.rollout_workers import RolloutWorkerPool, resolve_rollout_gpus
from ctm.core.config import LoRAConfig
from experiments.muse_glimmer_rmct_replication import plan as replication_plan


PROMPTS = (
    "Choose the correct label for a harmless format check: the requested label is A.",
    "Answer this format check with the stated label B.",
    "For this tokenizer check, give the literal label C.",
    "Return the requested label D for this format check.",
    "The answer token requested by this test is A.",
    "Use B as the response label in this format check.",
)
COMPLETIONS = (
    "The answer is A.",
    "The answer is B.",
    "The answer is C.",
    "The answer is D.",
    "Answer: A.",
    "Answer: B.",
)
PERTURBED_LORA_B_TENSORS = 8
PERTURBATION_SCALE = 0.20
RAW_CORRELATION_GATE = 0.999
RAW_MAX_ABS_ERROR_GATE = 0.50
EFFECT_L2_MIN = 0.50
EFFECT_COSINE_GATE = 0.90
EFFECT_NORM_RATIO_MIN = 0.80
EFFECT_NORM_RATIO_MAX = 1.25
EFFECT_RELATIVE_L2_ERROR_GATE = 0.25


class PreflightError(RuntimeError):
    """The Muse production topology or numerical contract did not pass."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-receipt-dir", type=Path, required=True)
    parser.add_argument("--worker-gpus", default="1,2,3")
    parser.add_argument("--worker-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--worker-max-model-len", type=int, default=131072)
    parser.add_argument("--worker-max-num-seqs", type=int, default=128)
    parser.add_argument("--worker-max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error(f"refusing to overwrite preflight output: {args.output_dir}")
    if not 0 < args.worker_gpu_memory_utilization <= 1:
        parser.error("--worker-gpu-memory-utilization must be in (0, 1]")
    for field in ("worker_max_model_len", "worker_max_num_seqs", "worker_max_num_batched_tokens"):
        if isinstance(getattr(args, field), bool) or getattr(args, field) < 1:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    return args


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise PreflightError(f"required regular file is absent: {resolved}")
    return {"path": str(resolved), "sha256": file_sha256(resolved), "size_bytes": resolved.stat().st_size}


def _source_manifest() -> dict[str, Any]:
    """Bind the exact local scientific/runtime source consumed after preflight."""

    paths: set[Path] = set()
    for directory in (PROJECT_ROOT / "ctm", PROJECT_ROOT / "ctm_data"):
        paths.update(directory.rglob("*.py"))
    paths.update(
        {
            PROJECT_ROOT / "scripts" / "run_experiment.py",
            PROJECT_ROOT / "scripts" / "train_rlct.py",
            PROJECT_ROOT / "experiments" / "muse_glimmer_rmct_replication" / "plan.py",
            PROJECT_ROOT
            / "experiments"
            / "muse_glimmer_rmct_replication"
            / "muse_glimmer_rmct_replication.yaml",
            PROJECT_ROOT / "infra" / "isambard" / "muse-runtime-requirements.txt",
            PROJECT_ROOT / "infra" / "isambard" / "muse_glimmer_runtime_env.sh",
            PROJECT_ROOT / "infra" / "isambard" / "setup_muse_glimmer_env.sh",
            PROJECT_ROOT / "infra" / "isambard" / "setup_muse_glimmer_runtime.sbatch",
            PROJECT_ROOT / "infra" / "isambard" / "build_muse_glimmer_vllm_cuda129_source.sbatch",
            PROJECT_ROOT / "infra" / "isambard" / "retry_muse_glimmer_vllm_cuda129_source.sbatch",
            PROJECT_ROOT / "infra" / "isambard" / "muse_glimmer_cuda129_runtime_env.sh",
            PROJECT_ROOT / "infra" / "isambard" / "attest_muse_glimmer_cuda129_source_runtime.sbatch",
            PROJECT_ROOT / "infra" / "isambard" / "preflight_muse_glimmer_rmct.py",
            PROJECT_ROOT / "infra" / "isambard" / "preflight_muse_glimmer_rmct.sbatch",
            PROJECT_ROOT / "infra" / "isambard" / "muse_glimmer_rmct_segment_contract.py",
            PROJECT_ROOT / "infra" / "isambard" / "run_muse_glimmer_rmct_segment.py",
            PROJECT_ROOT / "infra" / "isambard" / "run_muse_glimmer_rmct_segment.sbatch",
            PROJECT_ROOT / "infra" / "isambard" / "submit_muse_glimmer_rmct_next.sh",
        }
    )
    files = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise PreflightError(f"Muse source manifest requires a regular file: {path}")
        identity = _file_identity(path)
        identity["relative_path"] = str(path.relative_to(PROJECT_ROOT))
        files.append(identity)
    digest = _sha256_json(
        [
            {
                "relative_path": item["relative_path"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
            for item in files
        ]
    )
    return {"schema": "muse-glimmer-source-manifest-v1", "sha256": digest, "files": files}


def _runtime_receipt_dir(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to((PROJECT_ROOT / "artifacts").resolve())
    except ValueError as exc:
        raise PreflightError("runtime receipt directory must remain under repository artifacts") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise PreflightError(f"runtime receipt directory is absent or not regular: {resolved}")
    return resolved


def _validate_source_runtime(
    *,
    direct: Mapping[str, Any],
    receipt_dir: Path,
) -> str:
    """Revalidate the clean source checkout and every source-runtime identity."""

    runtime_path = receipt_dir / "runtime.json"
    freeze_path = receipt_dir / "pip-freeze.txt"
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        freeze = freeze_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("source-built vLLM lacks valid runtime receipts") from exc
    if runtime.get("schema") != "muse-glimmer-isambard-source-runtime-v1":
        raise PreflightError("source-built vLLM runtime receipt has the wrong schema")
    record = runtime.get("vllm")
    freeze_record = runtime.get("pip_freeze")
    if not isinstance(record, Mapping) or not isinstance(freeze_record, Mapping):
        raise PreflightError("source-built vLLM runtime receipt is incomplete")
    source = Path(str(record.get("source_path", ""))).resolve()
    url = direct.get("url")
    dir_info = direct.get("dir_info")
    editable_sources = []
    for line in freeze.splitlines():
        if not line.startswith("-e file://"):
            continue
        parsed = urlparse(line.removeprefix("-e "))
        editable_sources.append(Path(unquote(parsed.path)).resolve())
    if (
        not isinstance(url, str)
        or urlparse(url).scheme != "file"
        or Path(unquote(urlparse(url).path)).resolve() != source
        or not isinstance(dir_info, Mapping)
        or dir_info.get("editable") is not True
        or record.get("version") != VLLM_VERSION
        or record.get("commit") != VLLM_COMMIT
        or record.get("source_clean") is not True
        or freeze_record.get("path") != str(freeze_path.resolve())
        or freeze_record.get("sha256") != file_sha256(freeze_path)
        or editable_sources.count(source) != 1
    ):
        raise PreflightError("source-built vLLM PEP-610/freeze identity changed")
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tree = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError("cannot revalidate pinned vLLM source checkout") from exc
    if commit != VLLM_COMMIT or tree != record.get("tree") or status:
        raise PreflightError("pinned vLLM checkout commit/tree/cleanliness changed")
    extensions = record.get("extensions")
    if not isinstance(extensions, list) or not extensions:
        raise PreflightError("source runtime receipt has no extension identities")
    for index, extension in enumerate(extensions):
        if not isinstance(extension, Mapping):
            raise PreflightError("source runtime extension receipt is malformed")
        relative = extension.get("relative_path")
        if not isinstance(relative, str) or not relative:
            raise PreflightError("source runtime extension path is malformed")
        path = (source / relative).resolve()
        try:
            path.relative_to(source)
        except ValueError as exc:
            raise PreflightError("source runtime extension escapes the checkout") from exc
        actual = _file_identity(path)
        for key in ("path", "sha256", "size_bytes"):
            if extension.get(key) != actual[key]:
                raise PreflightError(f"source runtime extension {index} identity changed")
    dependencies = record.get("native_dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise PreflightError("source runtime receipt has no native dependency identities")
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, Mapping):
            raise PreflightError("source runtime dependency receipt is malformed")
        path = Path(str(dependency.get("path", ""))).resolve()
        try:
            path.relative_to(source / ".deps")
        except ValueError as exc:
            raise PreflightError("source runtime dependency escapes the pinned dependency root") from exc
        if path.is_symlink() or not path.is_dir():
            raise PreflightError("source runtime dependency is absent or not a regular directory")
        try:
            dependency_commit = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dependency_tree = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD^{tree}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dependency_status = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            dependency_submodules = subprocess.run(
                ["git", "-C", str(path), "submodule", "status", "--recursive"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            raise PreflightError(f"cannot revalidate native dependency {index}") from exc
        if (
            dependency_commit != dependency.get("commit")
            or dependency_tree != dependency.get("tree")
            or dependency_status
            or dependency.get("source_clean") is not True
            or dependency_submodules != dependency.get("submodules")
            or any(line.startswith(("-", "+", "U")) for line in dependency_submodules)
        ):
            raise PreflightError(f"source runtime native dependency {index} changed")
    build_inputs = runtime.get("build_inputs")
    if not isinstance(build_inputs, Mapping) or not build_inputs:
        raise PreflightError("source runtime receipt has no build-input identities")
    for label, expected in build_inputs.items():
        if not isinstance(expected, Mapping):
            raise PreflightError(f"source runtime build input {label!r} is malformed")
        actual = _file_identity(Path(str(expected.get("path", ""))))
        if dict(expected) != actual:
            raise PreflightError(f"source runtime build input {label!r} changed")
    return VLLM_COMMIT


def _vllm_source_commit(receipt_dir_value: Path) -> str:
    """Read and verify the exact source or commit-wheel PEP-610 identity."""

    distribution = importlib.metadata.distribution("vllm")
    if distribution.version != VLLM_VERSION:
        raise PreflightError(f"vLLM version is {distribution.version!r}, expected {VLLM_VERSION!r}")
    raw = distribution.read_text("direct_url.json")
    if not raw:
        raise PreflightError("vLLM installation lacks direct_url.json; source commit cannot be attested")
    try:
        direct = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PreflightError("vLLM direct_url.json is invalid") from exc
    receipt_dir = _runtime_receipt_dir(receipt_dir_value)
    if isinstance(direct, Mapping) and isinstance(direct.get("dir_info"), Mapping):
        return _validate_source_runtime(direct=direct, receipt_dir=receipt_dir)
    vcs = direct.get("vcs_info") if isinstance(direct, Mapping) else None
    commit = vcs.get("commit_id") if isinstance(vcs, Mapping) else None
    if commit == VLLM_COMMIT:
        return str(commit)
    url = direct.get("url") if isinstance(direct, Mapping) else None
    archive = direct.get("archive_info") if isinstance(direct, Mapping) else None
    hashes = archive.get("hashes") if isinstance(archive, Mapping) else None
    digest = hashes.get("sha256") if isinstance(hashes, Mapping) else None
    if digest is None and isinstance(archive, Mapping):
        legacy_hash = archive.get("hash")
        if isinstance(legacy_hash, str) and legacy_hash.startswith("sha256="):
            digest = legacy_hash.removeprefix("sha256=")
    if not isinstance(url, str) or unquote(url) != unquote(VLLM_WHEEL_URL):
        raise PreflightError(
            "vLLM PEP-610 metadata does not bind the exact commit-specific AArch64 wheel"
        )
    if digest is None:
        # uv currently records the exact wheel URL but leaves archive_info
        # empty for a direct wheel whose URL requirement carried a sha256
        # fragment. Rebind that enforced install to the immutable runtime
        # receipt, its exact freeze, and the source which issued the hashed
        # requirement; all three are included in the final attestation.
        installer = (distribution.read_text("INSTALLER") or "").strip()
        runtime_path = receipt_dir / "runtime.json"
        freeze_path = receipt_dir / "pip-freeze.txt"
        try:
            runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            frozen = freeze_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreflightError("vLLM uv hash fallback lacks valid runtime receipts") from exc
        vllm_record = runtime.get("vllm") if isinstance(runtime, Mapping) else None
        freeze_record = runtime.get("pip_freeze") if isinstance(runtime, Mapping) else None
        expected_freeze_line = f"vllm @ {VLLM_WHEEL_URL}"
        if (
            installer != "uv"
            or runtime.get("schema") != "muse-glimmer-isambard-runtime-v1"
            or not isinstance(vllm_record, Mapping)
            or vllm_record.get("version") != VLLM_VERSION
            or vllm_record.get("commit") != VLLM_COMMIT
            or unquote(str(vllm_record.get("wheel_url", ""))) != unquote(VLLM_WHEEL_URL)
            or vllm_record.get("wheel_sha256") != VLLM_WHEEL_SHA256
            or not isinstance(freeze_record, Mapping)
            or freeze_record.get("path") != str(freeze_path.resolve())
            or freeze_record.get("sha256") != file_sha256(freeze_path)
            or frozen.count(expected_freeze_line) != 1
        ):
            raise PreflightError("vLLM uv hash fallback does not bind the exact hashed wheel install")
    elif digest != VLLM_WHEEL_SHA256:
        raise PreflightError("vLLM PEP-610 wheel hash differs from the pinned AArch64 wheel")
    return VLLM_COMMIT


def _validate_snapshot(path: Path) -> Path:
    snapshot = path.resolve()
    config = snapshot / "config.json"
    if (
        path.is_symlink()
        or snapshot.name != MODEL_REVISION
        or snapshot.is_symlink()
        or not snapshot.is_dir()
        or not config.is_file()
        or not is_muse_glimmer_model_name(snapshot)
    ):
        raise PreflightError(f"not the pinned regular Muse snapshot {MODEL_REVISION}: {snapshot}")
    return snapshot


def _render_inputs(tokenizer: Any) -> tuple[list[list[int]], list[list[int]]]:
    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    eos = tokenizer.eos_token_id
    if eos != 200001:
        raise PreflightError(f"Muse tokenizer EOS is {eos}, expected 200001")
    for prompt_text, completion_text in zip(PROMPTS, COMPLETIONS, strict=True):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if isinstance(rendered, Mapping):
            rendered = rendered.get("input_ids")
        prompt = [int(token) for token in rendered]
        completion = [int(token) for token in tokenizer.encode(completion_text, add_special_tokens=False)]
        completion.append(int(eos))
        if not prompt or len(completion) < 3:
            raise PreflightError("Muse parity prompt/completion tokenization is unexpectedly empty")
        prompt_ids.append(prompt)
        completion_ids.append(completion)
    return prompt_ids, completion_ids


def _perturb_lora(model: torch.nn.Module) -> int:
    changed = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" not in name or not parameter.requires_grad:
                continue
            flat = torch.linspace(
                -PERTURBATION_SCALE,
                PERTURBATION_SCALE,
                steps=parameter.numel(),
                device=parameter.device,
                dtype=torch.float32,
            ).reshape(parameter.shape)
            parameter.copy_(flat.to(dtype=parameter.dtype))
            changed += 1
            if changed == PERTURBED_LORA_B_TENSORS:
                break
    if changed != PERTURBED_LORA_B_TENSORS:
        raise PreflightError(
            f"Muse adapter exposed only {changed} trainable LoRA-B tensors; expected at least {PERTURBED_LORA_B_TENSORS}"
        )
    return changed


def _nonzero_adapter_tensors(adapter_model: Path) -> int:
    tensors = load_file(str(adapter_model), device="cpu")
    return sum(int(torch.count_nonzero(tensor).item() > 0) for tensor in tensors.values())


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise PreflightError("parity vectors must have equal length >= 2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    denominator = math.sqrt(sum(value * value for value in centered_left) * sum(value * value for value in centered_right))
    if denominator <= 0:
        raise PreflightError("parity vectors have zero variance")
    return sum(a * b for a, b in zip(centered_left, centered_right, strict=True)) / denominator


def _parity(
    left: Sequence[float],
    right: Sequence[float],
    *,
    gate_kind: str = "raw_logprob",
) -> dict[str, Any]:
    if len(left) != len(right) or len(left) < 2:
        raise PreflightError("parity vectors must have equal length >= 2")
    if gate_kind not in {"raw_logprob", "policy_minus_base"}:
        raise PreflightError(f"unknown Muse parity gate kind: {gate_kind}")
    left_l2 = math.sqrt(sum(value * value for value in left))
    right_l2 = math.sqrt(sum(value * value for value in right))
    if left_l2 <= 1.0e-8 or right_l2 <= 1.0e-8:
        cosine = None
    else:
        cosine = sum(a * b for a, b in zip(left, right, strict=True)) / (left_l2 * right_l2)
    errors = [abs(a - b) for a, b in zip(left, right, strict=True)]
    difference_l2 = math.sqrt(sum(error * error for error in errors))
    relative_l2_error = difference_l2 / left_l2 if left_l2 > 1.0e-8 else None
    norm_ratio = right_l2 / left_l2 if left_l2 > 1.0e-8 else None
    try:
        pearson = _pearson(left, right)
    except PreflightError:
        pearson = None
    maximum = max(errors)
    if gate_kind == "raw_logprob":
        passed = (
            cosine is not None
            and pearson is not None
            and cosine >= RAW_CORRELATION_GATE
            and pearson >= RAW_CORRELATION_GATE
            and maximum <= RAW_MAX_ABS_ERROR_GATE
        )
    else:
        passed = (
            cosine is not None
            and norm_ratio is not None
            and relative_l2_error is not None
            and left_l2 >= EFFECT_L2_MIN
            and right_l2 >= EFFECT_L2_MIN
            and cosine >= EFFECT_COSINE_GATE
            and EFFECT_NORM_RATIO_MIN <= norm_ratio <= EFFECT_NORM_RATIO_MAX
            and relative_l2_error <= EFFECT_RELATIVE_L2_ERROR_GATE
        )
    return {
        "gate_kind": gate_kind,
        "passed": passed,
        "count": len(left),
        "hf_l2": left_l2,
        "vllm_l2": right_l2,
        "difference_l2": difference_l2,
        "relative_to_hf_l2_error": relative_l2_error,
        "vllm_to_hf_l2_ratio": norm_ratio,
        "cosine_similarity": cosine,
        "pearson_r": pearson,
        "max_abs_error": maximum,
    }


def _flatten(rows: Sequence[Sequence[float]], indices: Sequence[int] | None = None) -> list[float]:
    selected = range(len(rows)) if indices is None else indices
    return [float(value) for index in selected for value in rows[index]]


def _subtract(policy: Sequence[Sequence[float]], base: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(policy) != len(base):
        raise PreflightError("base and policy parity rows are misaligned")
    output: list[list[float]] = []
    for policy_row, base_row in zip(policy, base, strict=True):
        if len(policy_row) != len(base_row):
            raise PreflightError("base and policy parity token rows are misaligned")
        output.append([float(policy_value) - float(base_value) for policy_value, base_value in zip(policy_row, base_row, strict=True)])
    return output


def main() -> int:
    args = _parse_args()
    snapshot = _validate_snapshot(args.model_snapshot)
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise PreflightError(
            f"Transformers is {transformers.__version__}, expected exactly {TRANSFORMERS_VERSION}"
        )
    runtime_receipt_dir = _runtime_receipt_dir(args.runtime_receipt_dir)
    vllm_commit = _vllm_source_commit(runtime_receipt_dir)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [] if visible is None else [item.strip() for item in visible.split(",")]
    if len(tokens) != 4 or any(not token or token in {"-1", "NoDevFiles"} for token in tokens) or len(set(tokens)) != 4:
        raise PreflightError("Muse preflight requires exactly four distinct CUDA_VISIBLE_DEVICES tokens")
    workers = resolve_rollout_gpus(
        args.worker_gpus,
        cuda_visible_devices=visible,
        coordinator_device="cuda:0",
    )
    if [gpu.logical_index for gpu in workers] != [1, 2, 3]:
        raise PreflightError("Muse preflight requires rollout workers on logical GPUs 1,2,3")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    adapter_dir = output / "nonzero_parity_adapter"
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    prompt_ids, completion_ids = _render_inputs(tokenizer)
    model_inputs = [types.ModelInput.from_ints(tokens=value) for value in prompt_ids]

    backend = LocalBackend(
        device="cuda:0",
        dtype=torch.bfloat16,
        use_lora=True,
        sampler="vllm",
        hf_language_model_only=True,
        forward_microbatch_max_datums=1,
        forward_microbatch_max_tokens=16384,
        target_logprob_chunk_size=64,
    )
    pool: RolloutWorkerPool | None = None
    try:
        backend.setup(
            model=str(snapshot),
            lora=LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.0,
                train_mlp=True,
                train_attn=True,
                train_unembed=False,
                seed=args.seed,
            ),
        )
        changed_tensors = _perturb_lora(backend.model)
        source_hf_base = backend._score_completions(model_inputs, completion_ids, use_base=True)
        source_hf_policy = backend._score_completions(model_inputs, completion_ids, use_base=False)
        source_hf_effect = _subtract(source_hf_policy, source_hf_base)
        anchor_source_index = max(
            range(len(source_hf_effect)),
            key=lambda index: sum(value * value for value in source_hf_effect[index]),
        )
        probe_source_indices = [anchor_source_index] * len(workers) + list(range(len(prompt_ids)))
        probe_prompt_ids = [prompt_ids[index] for index in probe_source_indices]
        probe_completion_ids = [completion_ids[index] for index in probe_source_indices]
        hf_base = [source_hf_base[index] for index in probe_source_indices]
        hf_policy = [source_hf_policy[index] for index in probe_source_indices]
        backend.model.save_pretrained(str(adapter_dir))
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_model = adapter_dir / "adapter_model.safetensors"
        nonzero_tensor_count = _nonzero_adapter_tensors(adapter_model)
        if nonzero_tensor_count < changed_tensors:
            raise PreflightError("saved Muse adapter lost one or more deliberately non-zero LoRA tensors")

        engine_kwargs = {
            "gpu_memory_utilization": args.worker_gpu_memory_utilization,
            "dtype": "bfloat16",
            "language_model_only": True,
            "max_model_len": args.worker_max_model_len,
            "max_num_seqs": args.worker_max_num_seqs,
            "max_num_batched_tokens": args.worker_max_num_batched_tokens,
            "seed": args.seed,
        }
        pool = RolloutWorkerPool(
            model=str(snapshot),
            gpus=workers,
            enable_lora=True,
            engine_kwargs=engine_kwargs,
            status_dir=output / "workers",
        )
        pool.start()
        pool.publish_adapter_sync(adapter_dir, version=1)
        vllm_base = asyncio.run(
            pool.score_completions_uncapped_eos_tail(
                probe_prompt_ids,
                probe_completion_ids,
                use_base=True,
            )
        )
        vllm_policy = asyncio.run(
            pool.score_completions_uncapped_eos_tail(
                probe_prompt_ids,
                probe_completion_ids,
                use_base=False,
            )
        )
    finally:
        if pool is not None:
            pool.shutdown()
        backend.shutdown()

    hf_effect = _subtract(hf_policy, hf_base)
    vllm_effect = _subtract(vllm_policy, vllm_base)
    aggregate_base = _parity(_flatten(hf_base), _flatten(vllm_base))
    aggregate_policy = _parity(_flatten(hf_policy), _flatten(vllm_policy))
    aggregate_effect = _parity(
        _flatten(hf_effect),
        _flatten(vllm_effect),
        gate_kind="policy_minus_base",
    )
    per_worker = []
    for worker_index, gpu in enumerate(workers):
        indices = [
            index
            for index in range(len(probe_prompt_ids))
            if index % len(workers) == worker_index
        ]
        per_worker.append(
            {
                "worker_index": worker_index,
                "worker_gpu": gpu.as_dict(),
                "effect_parity": _parity(
                    _flatten(hf_effect, indices),
                    _flatten(vllm_effect, indices),
                    gate_kind="policy_minus_base",
                ),
            }
        )
    required_results = [aggregate_base, aggregate_policy, aggregate_effect, *(entry["effect_parity"] for entry in per_worker)]
    if any(result["passed"] is not True for result in required_results):
        failure = {
            "schema": "muse-glimmer-rollout-worker-parity-failure-v1",
            "aggregate_base_parity": aggregate_base,
            "aggregate_policy_parity": aggregate_policy,
            "aggregate_effect_parity": aggregate_effect,
            "per_worker_effect_parity": per_worker,
            "anchor_source_index": anchor_source_index,
            "probe_source_indices": probe_source_indices,
            "diagnostic_score_vectors": {
                "hf_base": hf_base,
                "hf_policy": hf_policy,
                "vllm_base": vllm_base,
                "vllm_policy": vllm_policy,
            },
        }
        (output / "parity-failure.json").write_bytes(_canonical(failure))
        raise PreflightError("Muse HF/PEFT↔vLLM parity fell below the frozen numerical gates")

    assert pool is not None
    document = {
        "schema": PARITY_ATTESTATION_SCHEMA,
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_path": str(snapshot),
            "config_sha256": file_sha256(snapshot / "config.json"),
        },
        "runtime": {
            "transformers_version": transformers.__version__,
            "vllm_commit": vllm_commit,
            "torch_version": torch.__version__,
            "platform": platform.platform(),
            "receipts": {
                "runtime": _file_identity(
                    runtime_receipt_dir / "runtime.json"
                ),
                "pip_freeze": _file_identity(
                    runtime_receipt_dir / "pip-freeze.txt"
                ),
                "model_snapshot": _file_identity(
                    runtime_receipt_dir / "model-snapshot.json"
                ),
            },
        },
        "scientific_contract": {
            "frozen_spec_sha256": replication_plan.FROZEN_SPEC_SHA256,
            "data": _file_identity(PROJECT_ROOT / replication_plan.DATA_PATH),
            "manifest": _file_identity(PROJECT_ROOT / replication_plan.MANIFEST_PATH),
            "generation": {
                "max_tokens": None,
                "termination": "eos_only",
                "non_eos_termination_policy": "fail_run",
            },
        },
        "source_manifest": _source_manifest(),
        "worker_gpus": [gpu.as_dict() for gpu in workers],
        "worker_engine_kwargs": pool.engine_kwargs,
        "adapter": {
            "adapter_config": _file_identity(adapter_config),
            "adapter_model": _file_identity(adapter_model),
            "nonzero_tensor_count": nonzero_tensor_count,
        },
        "probe": {
            "kind": "vllm-prompt-logprobs-uncapped-eos-tail-v1",
            "max_tokens": None,
            "termination": "eos_only",
            "source_prompt_count": len(prompt_ids),
            "score_row_count": len(probe_prompt_ids),
            "anchor_source_index": anchor_source_index,
            "source_indices": probe_source_indices,
            "request_count": len(probe_prompt_ids) * 2,
            "non_eos_termination_count": 0,
            "score_inputs_sha256": _sha256_json(
                {
                    "prompt_token_ids": probe_prompt_ids,
                    "completion_token_ids": probe_completion_ids,
                }
            ),
        },
        "aggregate_base_parity": aggregate_base,
        "aggregate_policy_parity": aggregate_policy,
        "aggregate_effect_parity": aggregate_effect,
        "per_worker_effect_parity": per_worker,
    }
    receipt = output / PARITY_ATTESTATION_NAME
    with receipt.open("xb") as handle:
        handle.write(_canonical(document))
    validate_muse_rollout_worker_parity_attestation(
        receipt,
        expected_model=snapshot,
        expected_worker_gpus=workers,
        expected_worker_engine_kwargs=pool.engine_kwargs,
    )
    print(f"CTM_MUSE_GLIMMER_PREFLIGHT={receipt}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
